#!/usr/bin/env python3
# coding=utf-8

"""
进程管理器模块，负责多进程架构下的进程创建、通信和数据同步
"""

import os
import time
import logging
import multiprocessing as mp
from multiprocessing import Process, Queue, Event, Manager
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from queue import Empty
import csv
import cv2

class DataPacket:
    """用于进程间传递的数据包"""
    def __init__(self, data_type: str, timestamp: float, data: Any):
        self.data_type = data_type  # 'arm' 或 摄像头名称
        self.timestamp = timestamp  # 数据时间戳
        self.data = data            # 数据内容

class ProcessManager:
    """进程管理器，负责创建和管理各个子进程"""

    def __init__(self, config: Dict[str, Any], mode: str = "eval"):
        """
        初始化进程管理器
        
        参数:
            config: 配置信息
        """
        self.logger = logging.getLogger("ProcessManager")
        self.config = config
        self.processes = {}
        self.mode= mode
        
        # 进程通信队列 - 为提高实时性，减小队列大小
        self.arm_queue = Queue(maxsize=10)  # 机械臂数据队列，减小队列以减少延迟
        self.camera_queues = {}              # 摄像头数据队列，每个摄像头一个
        self.sync_queue = Queue(maxsize=5)

        self.display_queue = Queue(maxsize=5) # 显示数据队列，减小队列以减少延迟
        
        self.save_queue = Queue() # 同步后的数据队列，减小队列以减少延迟
        
        # 进程控制事件
        self.stop_event = Event()
        

        # 共享管理器
        self.manager = Manager()
        self.shared_state = self.manager.dict()
        self.shared_state['running'] = False
        self.shared_state['sync_data'] = None  # 添加这行存储同步数据
        self.shared_state['save_mode'] = False  # 添加这行存储同步数据
        self.shared_state['save_path'] = None  # 

        
    def start_processes(self) -> bool:
        """
        启动所有子进程
        
        返回:
            启动是否成功
        """
        try:
            self.logger.info("正在启动所有进程...")
            self.shared_state['running'] = True
            
            # 1. 启动机械臂数据采集进程
            arm_process = Process(
                target=self._arm_process_worker,
                args=(self.config["arm"], self.arm_queue, self.stop_event)
            )
            arm_process.daemon = True
            arm_process.start()
            self.processes['arm'] = arm_process
            
            # 2. 启动摄像头数据采集进程
            for idx, (camera_id, camera_name) in enumerate(
                zip(self.config["camera"]["ids"], self.config["camera"]["names"])
            ):
                # 为每个摄像头创建一个队列
                camera_queue = Queue(maxsize=30)  # 每个摄像头保持约1秒的数据
                self.camera_queues[camera_name] = camera_queue
                
                # 启动摄像头进程
                camera_config = {
                    "id": camera_id,
                    "name": camera_name,
                    "width": self.config["camera"]["width"],
                    "height": self.config["camera"]["height"],
                    "fps": self.config["camera"]["fps"]
                }
                
                camera_process = Process(
                    target=self._camera_process_worker,
                    args=(camera_config, camera_queue, self.stop_event)
                )
                camera_process.daemon = True
                camera_process.start()
                self.processes[f'camera_{camera_name}'] = camera_process
            
            # 3. 启动数据同步进程
            sync_process = Process(
                target=self._sync_process_worker,
                args=(self.arm_queue, self.camera_queues, self.sync_queue, self.stop_event)
            )
            sync_process.daemon = True
            sync_process.start()
            self.processes['sync'] = sync_process

            #4. 数据显示进程
            display_process = Process(
                target=self._display_process_worker,
                args=( self.mode,self.sync_queue, self.display_queue, self.save_queue,self.stop_event)
            )
            display_process.daemon = True
            display_process.start()
            self.processes['display'] = display_process

            #5.保存数据进程
            save_process = Process(
                target=self._data_record_process_worker,
                args=(self.save_queue,self.stop_event)
            )
            save_process.daemon = True
            save_process.start()
            #self.processes['save'] = save_process

            self.logger.info(f"所有进程已启动: {len(self.processes)}个进程")
            return True
            
        except Exception as e:
            self.logger.exception(f"启动进程时发生异常: {e}")
            self.stop_processes()
            return False
    
    def stop_processes(self) -> None:
        """停止所有子进程"""
        self.logger.info("正在停止所有进程...")
        self.shared_state['running'] = False
        self.stop_event.set()
        
        # 等待所有进程结束
        for name, process in self.processes.items():
            if process.is_alive():
                self.logger.info(f"等待进程结束: {name}")
                process.join(timeout=2.0)  # 等待最多2秒
                
                if process.is_alive():
                    self.logger.warning(f"进程未响应，强制终止: {name}")
                    process.terminate()
        
        # 清空所有队列
        self._clear_queue(self.arm_queue)
        self._clear_queue(self.display_queue)
        #self._clear_queue(self.save_queue)

        for queue in self.camera_queues.values():
            self._clear_queue(queue)
        self._clear_queue(self.sync_queue)
        
        self.processes.clear()
        self.logger.info("所有进程已停止")
    
    def get_synchronized_data(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        从同步队列获取一帧同步后的数据
        
        参数:
            timeout: 超时时间(秒)
            
        返回:
            同步后的数据，如果超时则返回None
        """
        if not self.shared_state['running']:
            self.logger.warning("进程管理器未运行，无法获取数据")
            return None
            
        try:
            # 从同步队列获取数据
            if self.mode=="eval":
                data = self.display_queue.get(timeout=timeout)
            else:
                data=self.shared_state.get('sync_data')
            # self.shared_state['sync_data'] = None

            return data

        except Exception as e:
            # 队列为空或其他错误sync_queue
            return None

    def start_save(self, save_path: str) -> None:

        self.shared_state['save_path'] = save_path
        self.shared_state['save_mode'] = True


    def stop_save(self) -> None:
        
        self.shared_state['save_mode'] = False

    def _clear_queue(self, queue: Queue) -> None:
        """清空队列"""
        while not queue.empty():
            try:
                queue.get_nowait()
            except:
                pass
    
    def _arm_process_worker(self, arm_config: Dict[str, Any], output_queue: Queue, stop_event: Event) -> None:
        """机械臂数据采集进程"""
        from alicia_recorder.core.arm_control import ArmController
        
        logger = logging.getLogger("ArmProcess")
        logger.info("机械臂数据采集进程已启动")
        
        try:
            # 创建机械臂控制器
            arm_controller = ArmController(
                port=arm_config["port"],
                baudrate=arm_config["baudrate"],
                debug_mode=arm_config["debug_mode"]
            )
            
            # 连接机械臂
            if not arm_controller.connect():
                logger.error("无法连接机械臂，进程终止")
                return
                
            logger.info("机械臂已连接，开始数据采集")
            
            # 主循环
            while not stop_event.is_set():
                # 读取机械臂状态
                arm_data = arm_controller.read_joint_state()
                
                # 获取当前时间戳
                timestamp = time.time()
                
                # 创建数据包
                packet = DataPacket(
                    data_type='arm',
                    timestamp=timestamp,
                    data=arm_data
                )
                
                # 将数据包放入队列
                try:
                    output_queue.put(packet, timeout=0.1)
                except:
                    # 队列已满，丢弃旧数据
                    if not output_queue.empty():
                        try:
                            output_queue.get_nowait()
                            output_queue.put(packet, timeout=0.1)
                        except:
                            pass
                
                # 取消time.sleep，以最快速度采集数据
                
        except Exception as e:
            logger.exception(f"机械臂数据采集进程异常: {e}")
        finally:
            # 断开连接
            if 'arm_controller' in locals() and arm_controller.is_connected:
                arm_controller.disconnect()
                
            logger.info("机械臂数据采集进程已结束")
    
    def _camera_process_worker(self, camera_config: Dict[str, Any], output_queue: Queue, stop_event: Event) -> None:
        """摄像头数据采集进程"""
        import cv2
        
        camera_name = camera_config["name"]
        logger = logging.getLogger(f"CameraProcess[{camera_name}]")
        logger.info(f"摄像头[{camera_name}]数据采集进程已启动")
        
        try:
            # 连接摄像头并设置高性能参数
            cap = cv2.VideoCapture(camera_config["id"])
            
            # 设置OpenCV摄像头性能参数
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config["width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config["height"])
            cap.set(cv2.CAP_PROP_FPS, camera_config["fps"])
            
            # 优化性能关键参数
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 设置缓冲区为1，减少延迟
            
            # 尝试设置更多高性能参数（取决于相机驱动是否支持）
            try:
                # 设置为非阻塞模式 (仅在支持的设备上有效)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))  # 使用MJPG编码
            except:
                pass
            
            if not cap.isOpened():
                logger.error(f"无法连接摄像头[{camera_name}]，进程终止")
                return
                
            # 测试读取一帧
            ret, _ = cap.read()
            if not ret:
                logger.error(f"无法从摄像头[{camera_name}]读取图像，进程终止")
                cap.release()
                return
                
            logger.info(f"摄像头[{camera_name}]已连接，开始数据采集")
            
            # 获取实际参数
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"摄像头[{camera_name}]实际参数: {actual_width}x{actual_height}, {actual_fps}fps")
            
            # 跟踪帧率性能
            frame_count = 0
            last_fps_check = time.time()
            
            # 主循环
            while not stop_event.is_set():
                # 读取图像
                ret, frame = cap.read()
                
                # 立即记录时间戳
                timestamp = time.time()
                
                if not ret or frame is None:
                    logger.warning(f"摄像头[{camera_name}]读取图像失败")
                    continue
                
                # 创建数据包
                packet = DataPacket(
                    data_type=camera_name,
                    timestamp=timestamp,
                    data=frame
                )
                
                # 将数据包放入队列 - 使用非阻塞方式，保证实时性
                try:
                    # 如果队列满了，直接丢弃当前帧，保证只有最新的帧会被处理
                    if output_queue.full():
                        try:
                            # 清空队列，只保留最新帧
                            while not output_queue.empty():
                                output_queue.get_nowait()
                        except:
                            pass
                    
                    # 放入新帧
                    output_queue.put(packet, block=False)
                    
                    # 跟踪帧率
                    frame_count += 1
                    now = time.time()
                    if now - last_fps_check >= 5.0:  # 每5秒记录一次实际帧率
                        fps = frame_count / (now - last_fps_check)
                        logger.debug(f"摄像头[{camera_name}]实时帧率: {fps:.1f}fps")
                        frame_count = 0
                        last_fps_check = now
                        
                except:
                    # 如果无法放入队列，直接跳过
                    pass
                
        except Exception as e:
            logger.exception(f"摄像头[{camera_name}]数据采集进程异常: {e}")
        finally:
            # 释放资源
            if 'cap' in locals() and cap.isOpened():
                cap.release()
                
            logger.info(f"摄像头[{camera_name}]数据采集进程已结束")
    
    # def _sync_process_worker(self, arm_queue: Queue, camera_queues: Dict[str, Queue], 
    #                         output_queue: Queue, stop_event: Event) -> None:
    #     """数据同步进程"""
    #     logger = logging.getLogger("SyncProcess")
    #     logger.info("数据同步进程已启动")
        
    #     # 数据缓冲区，存储一段时间内的数据
    #     arm_buffer = []
    #     camera_buffers = {name: [] for name in camera_queues.keys()}
        
    #     # 缓冲区最大大小 - 为保证实时性，大幅减小缓冲区
    #     ARM_BUFFER_SIZE = 3   # 机械臂数据缓冲(约0.5秒)，减小缓冲以减少延迟
    #     CAMERA_BUFFER_SIZE = 3 # 摄像头数据缓冲(约0.1秒，30fps)，减小缓冲以减少延迟
        
    #     # 最大允许的时间差（秒）- 放宽时间差要求，保证实时性
    #     CAM_MAX_TIME_DIFF = 0.100  # 摄像头之间的最大时间差：100毫秒，增大容忍度以提高实时性
    #     ARM_MAX_TIME_DIFF = 0.010  # 机械臂与摄像头平均时间的最大时间差：200毫秒，增大容忍度以提高实时性
        
    #     try:
    #         while not stop_event.is_set():
    #             # 1. 从所有队列读取数据并添加到缓冲区
                
    #             # 读取机械臂数据
    #             while not arm_queue.empty():
    #                 try:
    #                     packet = arm_queue.get_nowait()
    #                     arm_buffer.append(packet)
    #                 except:
    #                     break
                        
    #             # 读取所有摄像头数据
    #             for name, queue in camera_queues.items():
    #                 while not queue.empty():
    #                     try:
    #                         packet = queue.get_nowait()
    #                         camera_buffers[name].append(packet)
    #                     except:
    #                         break
                
    #             # 2. 限制缓冲区大小，保留最新的数据
    #             if len(arm_buffer) > ARM_BUFFER_SIZE:
    #                 arm_buffer = arm_buffer[-ARM_BUFFER_SIZE:]
                    
    #             for name in camera_buffers:
    #                 if len(camera_buffers[name]) > CAMERA_BUFFER_SIZE:
    #                     camera_buffers[name] = camera_buffers[name][-CAMERA_BUFFER_SIZE:]
                
    #             # 3. 快速检查是否有足够的数据进行同步
    #             # 为保证实时性，我们检查每个摄像头是否至少有一帧数据和至少一个机械臂数据
    #             if len(arm_buffer) < 1 or any(len(buf) < 1 for buf in camera_buffers.values()):
    #                 continue
                
    #             # 4. 获取摄像头名称列表
    #             camera_names = list(camera_buffers.keys())
    #             if len(camera_names) < 2:
    #                 logger.warning("摄像头数量不足，无法进行同步")
    #                 continue
                
    #             # 确保每个摄像头至少有一帧最新数据
    #             # 简化逻辑，直接使用最新的数据进行同步，保证实时性
                
    #             # 使用最新的一帧进行快速同步匹配
    #             # 这样可以保证实时性，避免过度计算和延迟
    #             packet1 = camera_buffers[camera_names[0]][-1]  # 取最新的一帧
    #             t1 = packet1.timestamp
                
    #             # 在第二个摄像头中找最接近的帧
    #             min_cam_diff = float('inf')
    #             best_j = 0
                
    #             # 从新到旧查找最接近的帧
    #             for j in range(len(camera_buffers[camera_names[1]])-1, -1, -1):
    #                 packet2 = camera_buffers[camera_names[1]][j]
    #                 t2 = packet2.timestamp
    #                 time_diff = abs(t1 - t2)
                    
    #                 if time_diff < min_cam_diff:
    #                     min_cam_diff = time_diff
    #                     best_j = j
                    
    #                 # 一旦找到足够好的匹配立即退出
    #                 if time_diff < 0.015:  # 15ms内的差异就足够好了
    #                     break
                
    #             best_match = {
    #                 camera_names[0]: (len(camera_buffers[camera_names[0]])-1, packet1),
    #                 camera_names[1]: (best_j, camera_buffers[camera_names[1]][best_j])
    #             }
                
    #             # 如果没有找到匹配的帧对或时间差太大，继续等待
    #             if best_match is None or min_cam_diff > CAM_MAX_TIME_DIFF:
    #                 continue
                
    #             # 5. 计算摄像头的平均时间戳
    #             camera_packets = {name: packet for name, (_, packet) in best_match.items()}
    #             camera_timestamps = [packet.timestamp for packet in camera_packets.values()]
    #             avg_camera_timestamp = sum(camera_timestamps) / len(camera_timestamps)
                
    #             # 6. 找到与平均时间戳最接近的机械臂数据
    #             closest_arm_packet = None
    #             min_arm_diff = float('inf')
                
    #             for arm_packet in arm_buffer:
    #                 time_diff = abs(arm_packet.timestamp - avg_camera_timestamp)
    #                 if time_diff < min_arm_diff:
    #                     min_arm_diff = time_diff
    #                     closest_arm_packet = arm_packet
                
    #             # 7. 构建同步数据包
    #             sync_data = {
    #                 'timestamp': avg_camera_timestamp,
    #                 'arm_data': closest_arm_packet.data,
    #                 'arm_timestamp': closest_arm_packet.timestamp,
    #                 'images': {},
    #                 'images_data': {}
    #             }
                
    #             # 添加图像数据
    #             for name, packet in camera_packets.items():
    #                 sync_data['images'][name] = packet.data
    #                 sync_data['images_data'][name] = {
    #                     'image': packet.data,
    #                     'timestamp': packet.timestamp
    #                 }
                
    #             # 添加时间差信息
    #             sync_data['time_diff'] = {
    #                 'arm_to_avg': closest_arm_packet.timestamp - avg_camera_timestamp,
    #             }
                
    #             for name, packet in camera_packets.items():
    #                 sync_data['time_diff'][f'{name}_to_avg'] = packet.timestamp - avg_camera_timestamp
                
    #             # 8. 输出调试信息
    #             logger.debug(f"摄像头时间差: {min_cam_diff*1000:.2f}ms, 机械臂时间差: {min_arm_diff*1000:.2f}ms")
                
    #             # 9. 将同步数据放入输出队列
    #             try:
    #                 output_queue.put(sync_data, block=False)  # 非阻塞，避免同步进程卡住
                    
    #                 # 清除已成功同步并发送的帧
    #                 # 从各自的摄像头缓冲区中移除这些帧，以及所有在它们之前的帧，
    #                 # 以确保下次迭代时处理的是更新的数据。
                    
    #                 cam0_name = camera_names[0]
    #                 # best_match[cam0_name][0] 是 camera_buffers[cam0_name] 中被使用的帧的索引
    #                 if cam0_name in best_match and best_match[cam0_name] is not None:
    #                     idx0_used = best_match[cam0_name][0]
    #                     # 保留此索引之后的所有帧。如果使用的是最后一个帧，则列表变为空。
    #                     camera_buffers[cam0_name] = camera_buffers[cam0_name][idx0_used + 1:]
                    
    #                 cam1_name = camera_names[1]
    #                 # best_match[cam1_name][0] 是 camera_buffers[cam1_name] 中被使用的帧的索引 (best_j)
    #                 if cam1_name in best_match and best_match[cam1_name] is not None:
    #                     idx1_used = best_match[cam1_name][0]
    #                     camera_buffers[cam1_name] = camera_buffers[cam1_name][idx1_used + 1:]
                    
    #                 # 清理机械臂缓冲区，只保留比当前匹配摄像头时间戳新的数据 (此逻辑保持不变)
    #                 if arm_buffer and closest_arm_packet: # 确保 closest_arm_packet 不是 None
    #                     # 保留时间戳大于或等于平均摄像头时间戳的臂数据
    #                     # 或者，更精确地，可以考虑移除 closest_arm_packet 及更早的数据
    #                     # 当前的逻辑是基于 avg_camera_timestamp，这通常是合理的
    #                     arm_buffer = [p for p in arm_buffer if p.timestamp >= avg_camera_timestamp]
                        
    #             except Exception as e: # 更具体的异常捕获可能更好，例如 queue.Full
    #                 # 输出队列已满，说明数据处理延迟，直接丢弃该帧
    #                 logger.debug(f"同步队列已满，丢弃当前帧以保持实时性: {e}")
                
    #             # 取消time.sleep，以最快速度处理同步
                
    #     except Exception as e:
    #         logger.exception(f"数据同步进程异常: {e}")
    #     finally:
    #         logger.info("数据同步进程已结束")


    def _sync_process_worker(self, arm_queue: Queue, camera_queues: Dict[str, Queue], 
                            output_queue: Queue, stop_event: Event) -> None:
        """数据同步进程"""
        logger = logging.getLogger("SyncProcess")
        logger.info("数据同步进程已启动")
        
        arm_buffer = []
        camera_buffers = {name: [] for name in camera_queues.keys()}
        
        ARM_BUFFER_SIZE = 3
        CAMERA_BUFFER_SIZE = 3
        CAM_MAX_TIME_DIFF = 0.100 
        ARM_MAX_TIME_DIFF = 0.010
        
        try:
            while not stop_event.is_set():
                # 1. 从所有队列读取数据并添加到缓冲区
                while not arm_queue.empty():
                    try:
                        packet = arm_queue.get_nowait()
                        arm_buffer.append(packet)
                    except:
                        break
                        
                for name, queue in camera_queues.items():
                    while not queue.empty():
                        try:
                            packet = queue.get_nowait()
                            camera_buffers[name].append(packet)
                        except:
                            break
                
                # 2. 限制缓冲区大小，保留最新的数据
                if len(arm_buffer) > ARM_BUFFER_SIZE:
                    arm_buffer = arm_buffer[-ARM_BUFFER_SIZE:]
                    
                for name in camera_buffers:
                    if len(camera_buffers[name]) > CAMERA_BUFFER_SIZE:
                        camera_buffers[name] = camera_buffers[name][-CAMERA_BUFFER_SIZE:]
                
                # 3. 快速检查是否有足够的数据进行同步
                if len(arm_buffer) < 1 or any(len(buf) < 1 for buf in camera_buffers.values()):
                    # time.sleep(0.001) # 避免CPU空转，但可能引入延迟，当前已注释
                    continue
                
                # 4. 摄像头同步逻辑
                camera_names = list(camera_buffers.keys())
                num_cameras = len(camera_names)

                if num_cameras == 0:
                    logger.warning("没有配置摄像头，无法进行同步")
                    continue

                synchronized_camera_packets: Dict[str, Tuple[int, DataPacket]] = {}
                avg_camera_timestamp: Optional[float] = None
                max_timestamp_diff_among_cameras = float('inf')

                if num_cameras == 1:
                    cam_name = camera_names[0]
                    # camera_buffers[cam_name] guaranteed to be non-empty by check at step 3
                    latest_frame_idx = len(camera_buffers[cam_name]) - 1
                    latest_frame_packet = camera_buffers[cam_name][latest_frame_idx]
                    
                    synchronized_camera_packets[cam_name] = (latest_frame_idx, latest_frame_packet)
                    avg_camera_timestamp = latest_frame_packet.timestamp
                    max_timestamp_diff_among_cameras = 0.0
                else: # num_cameras >= 2
                    found_sync_set = False
                    ref_cam_name = camera_names[0]
                    ref_cam_buffer = camera_buffers[ref_cam_name]

                    # Iterate through recent frames of the reference camera as potential anchors
                    for ref_idx in range(len(ref_cam_buffer) - 1, -1, -1):
                        ref_packet = ref_cam_buffer[ref_idx]
                        t_ref = ref_packet.timestamp
                        
                        current_selection_candidate: Dict[str, Tuple[int, DataPacket]] = {ref_cam_name: (ref_idx, ref_packet)}
                        current_timestamps_in_candidate = [t_ref]
                        all_others_matched = True

                        for i in range(1, num_cameras): # For other cameras
                            other_cam_name = camera_names[i]
                            other_cam_buffer = camera_buffers[other_cam_name]
                            # other_cam_buffer guaranteed non-empty by check at step 3

                            best_match_other_idx = -1
                            min_diff_for_this_other_cam = float('inf')

                            for k_idx in range(len(other_cam_buffer) - 1, -1, -1):
                                other_packet_candidate = other_cam_buffer[k_idx]
                                diff = abs(other_packet_candidate.timestamp - t_ref)
                                if diff < min_diff_for_this_other_cam:
                                    min_diff_for_this_other_cam = diff
                                    best_match_other_idx = k_idx
                                if diff < 0.015: # Heuristic: 15ms is a very good match
                                    break 
                            
                            if best_match_other_idx != -1:
                                current_selection_candidate[other_cam_name] = (best_match_other_idx, other_cam_buffer[best_match_other_idx])
                                current_timestamps_in_candidate.append(other_cam_buffer[best_match_other_idx].timestamp)
                            else:
                                all_others_matched = False
                                break 
                        
                        if not all_others_matched:
                            continue # Try next reference frame

                        # Check spread of timestamps in the current candidate set
                        if len(current_timestamps_in_candidate) == num_cameras:
                            max_ts = max(current_timestamps_in_candidate)
                            min_ts = min(current_timestamps_in_candidate)
                            current_spread = max_ts - min_ts
                            
                            if current_spread <= CAM_MAX_TIME_DIFF:
                                synchronized_camera_packets = current_selection_candidate
                                avg_camera_timestamp = sum(current_timestamps_in_candidate) / num_cameras
                                max_timestamp_diff_among_cameras = current_spread
                                found_sync_set = True
                                break # Found a good sync set
                    
                    if not found_sync_set:
                        # logger.debug("多摄像头同步失败：未找到满足 CAM_MAX_TIME_DIFF 的帧组合")
                        continue
                
                # 5. 检查摄像头同步是否成功
                if avg_camera_timestamp is None:
                    continue

                # 6. 找到与平均摄像头时间戳最接近的机械臂数据
                closest_arm_packet = None
                min_arm_diff = float('inf')
                
                if not arm_buffer: # Should be caught by step 3, but as a safeguard
                    logger.debug("机械臂缓冲区为空，无法同步机械臂数据")
                    continue

                for arm_packet_candidate in arm_buffer:
                    time_diff = abs(arm_packet_candidate.timestamp - avg_camera_timestamp)
                    if time_diff < min_arm_diff:
                        min_arm_diff = time_diff
                        closest_arm_packet = arm_packet_candidate
                
                if closest_arm_packet is None or min_arm_diff > ARM_MAX_TIME_DIFF:
                    # logger.debug(f"未找到合适的机械臂数据，最小差值: {min_arm_diff*1000:.2f}ms")
                    continue
                
                # 7. 构建同步数据包
                sync_data = {
                    'timestamp': avg_camera_timestamp,
                    'arm_data': closest_arm_packet.data,
                    'arm_timestamp': closest_arm_packet.timestamp,
                    'images': {},
                    'images_data': {} # 保留此结构以兼容旧格式或未来使用
                }
                
                camera_packets_for_data = {name: packet for name, (_, packet) in synchronized_camera_packets.items()}

                for name, packet_data in camera_packets_for_data.items():
                    sync_data['images'][name] = packet_data.data
                    sync_data['images_data'][name] = { # 保留此详细结构
                        'image': packet_data.data,
                        'timestamp': packet_data.timestamp
                    }
                
                sync_data['time_diff'] = {
                    'arm_to_avg': closest_arm_packet.timestamp - avg_camera_timestamp,
                }
                for name, packet_data in camera_packets_for_data.items():
                    sync_data['time_diff'][f'{name}_to_avg'] = packet_data.timestamp - avg_camera_timestamp
                
                # 8. 输出调试信息
                log_cam_diff_ms = max_timestamp_diff_among_cameras * 1000
                logger.debug(f"摄像头时间戳最大差值: {log_cam_diff_ms:.2f}ms, 机械臂时间差: {min_arm_diff*1000:.2f}ms")
                
                # 9. 将同步数据放入输出队列
                try:
                    output_queue.put(sync_data, block=False)
                    
                    # 清除已成功同步并发送的帧
                    for cam_name_to_clear, (idx_used, _) in synchronized_camera_packets.items():
                        # Ensure buffer still exists and has enough elements, though it should.
                        if cam_name_to_clear in camera_buffers and len(camera_buffers[cam_name_to_clear]) > idx_used:
                             camera_buffers[cam_name_to_clear] = camera_buffers[cam_name_to_clear][idx_used + 1:]
                        elif cam_name_to_clear in camera_buffers and len(camera_buffers[cam_name_to_clear]) == idx_used +1: # used last element
                             camera_buffers[cam_name_to_clear] = []


                    if arm_buffer and closest_arm_packet:
                        # 清理机械臂缓冲区，保留比当前同步的摄像头平均时间戳更新的数据
                        # 或者移除已使用的 arm_packet 及更早的数据
                        # 当前策略：移除时间戳小于等于平均摄像头时间戳的数据，或严格大于已使用的arm_packet的时间戳
                        # arm_buffer = [p for p in arm_buffer if p.timestamp > closest_arm_packet.timestamp]
                        # 更安全的做法是移除已使用的那一个以及它之前的（如果按时间排序）
                        # 考虑到arm_buffer可能不完全有序且有重复时间戳的可能，
                        # 移除所有时间戳小于等于 avg_camera_timestamp 的数据是一个相对简单且有效的策略
                        arm_buffer = [p for p in arm_buffer if p.timestamp >= avg_camera_timestamp]

                except Exception as e: 
                    logger.debug(f"同步队列已满或输出时出错，丢弃当前帧: {e}")
                
        except Exception as e:
            logger.exception(f"数据同步进程异常: {e}")
        finally:
            logger.info("数据同步进程已结束")

    def _display_process_worker(self, mode: str , sync_queue: Queue, display_queue: Queue, save_queue: Queue, stop_event: Event) -> None:
        """数据显示进程，用于实时展示同步数据
        
        参数:
            sync_queue: 同步数据队列，包含已同步的机械臂和摄像头数据
            sync_state: 显示状态共享对象，用于跨进程通信
        """
        logger = logging.getLogger("DisplayProcess")
        logger.info("数据显示进程已启动")
        
        try:
            while not stop_event.is_set():
                try:
                    # 从同步队列获取数据
                    #print(mode)
                    sync_data = sync_queue.get(timeout=0.1)  # 短超时确保响应停止信号
                    if mode == "eval":
                        display_queue.put(sync_data)


                    elif mode == "record":
                        self.shared_state['sync_data'] = sync_data

                        if self.shared_state.get('save_mode'):
                            self.save_queue.put(sync_data)
                    else:
                        print("未知模式，无法处理数据")
                        #停止全部程序
                        stop_event.set()

                except Empty:
                    # 队列为空，继续等待
                    continue
                except Exception as e:
                    logger.error(f"处理显示数据时出错: {e}")
                    
        except Exception as e:
            logger.exception(f"数据显示进程异常: {e}")
        finally:
            logger.info("数据显示进程已结束")

    def _data_record_process_worker(self, save_queue: Queue, stop_event: Event) -> None:

        logger = logging.getLogger("recordProcess")
        logger.info("数据保存进程已启动")
        # 图像计数器字典，每个摄像头单独计数
        image_counters = {}
        # 机械臂数据计数
        arm_counter = 0

        arm_name = self.config.get("arm", {}).get("name")

        try:
            while not stop_event.is_set():
                try:
                    if not self.shared_state.get('save_mode'):
                        image_counters = {}
                        # 机械臂数据计数
                        arm_counter = 0
                    data=save_queue.get()
                    save_path=self.shared_state.get('save_path')
                    #print(f"保存数据到: {save_path}")
                    #print(data)
                    timestamp = data.get('timestamp', 0)
                    # 1. 保存摄像头图像
                    for camera_name, image in data.get('images', {}).items():
                        # 初始化摄像头计数器
                        if camera_name not in image_counters:
                            image_counters[camera_name] = 1
                        
                        # 构建图像文件路径
                        img_filename = f"{image_counters[camera_name]:05d}.jpg"  # 00001.jpg 格式
                        camera_dir = os.path.join(save_path, "camera", camera_name)
                        img_path = os.path.join(camera_dir, img_filename)
                        
                        # 确保目录存在
                        os.makedirs(camera_dir, exist_ok=True)
                        
                        # 保存图像
                        if isinstance(image, np.ndarray):
                            cv2.imwrite(img_path, image)
                            logger.debug(f"已保存图像: {img_path}")
                        
                        # 计数器增加
                        image_counters[camera_name] += 1
                    
                    # 2. 保存机械臂关节数据
                    if data.get('arm_data') and 'joint_state' in data.get('arm_data', {}):
                        # 只保存关节状态数据
                        joint_state = data['arm_data']['joint_state']
                        if isinstance(joint_state, np.ndarray):
                            joint_state = joint_state.tolist()
                        
                        # 创建机械臂数据目录
                        arm_dir = os.path.join(save_path, arm_name)
                        os.makedirs(arm_dir, exist_ok=True)
                        
                        # 保存关节数据到CSV文件
                        arm_file = os.path.join(arm_dir, "joint_data.csv")
                        is_new_file = not os.path.exists(arm_file)
                        
                        with open(arm_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            
                            # 如果是新文件，写入表头
                            if is_new_file:
                                header = ["frame_id", "timestamp"]
                                for i in range(len(joint_state)):
                                    header.append(f"joint{i+1}")
                                writer.writerow(header)
                            
                            # 写入关节数据行
                            row = [arm_counter, timestamp] + list(joint_state)
                            writer.writerow(row)
                            arm_counter += 1
                        
                    # 定期记录保存进度
                    if arm_counter % 20 == 0:
                        print(f"已保存 {arm_counter} 帧数据")

                except Empty:
                    # 队列为空，继续等待
                    continue
                except Exception as e:
                    logger.error(f"处理显示数据时出错: {e}")


            while not save_queue.empty():
                try:
                    data=save_queue.get()
                    save_path=self.shared_state.get('save_path')
                    #print(f"保存数据到: {save_path}")
                    #print(data)
                    timestamp = data.get('timestamp', 0)
                    # 1. 保存摄像头图像
                    for camera_name, image in data.get('images', {}).items():
                        # 初始化摄像头计数器
                        if camera_name not in image_counters:
                            image_counters[camera_name] = 1
                        
                        # 构建图像文件路径
                        img_filename = f"{image_counters[camera_name]:05d}.jpg"  # 00001.jpg 格式
                        camera_dir = os.path.join(save_path, "camera", camera_name)
                        img_path = os.path.join(camera_dir, img_filename)
                        
                        # 确保目录存在
                        os.makedirs(camera_dir, exist_ok=True)
                        
                        # 保存图像
                        if isinstance(image, np.ndarray):
                            cv2.imwrite(img_path, image)
                            logger.debug(f"已保存图像: {img_path}")
                        
                        # 计数器增加
                        image_counters[camera_name] += 1
                    
                    # 2. 保存机械臂关节数据
                    if data.get('arm_data') and 'joint_state' in data.get('arm_data', {}):
                        # 只保存关节状态数据
                        joint_state = data['arm_data']['joint_state']
                        if isinstance(joint_state, np.ndarray):
                            joint_state = joint_state.tolist()
                        
                        # 创建机械臂数据目录
                        arm_dir = os.path.join(save_path, arm_name)
                        os.makedirs(arm_dir, exist_ok=True)
                        
                        # 保存关节数据到CSV文件
                        arm_file = os.path.join(arm_dir, "joint_data.csv")
                        is_new_file = not os.path.exists(arm_file)
                        
                        with open(arm_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            
                            # 如果是新文件，写入表头
                            if is_new_file:
                                header = ["frame_id", "timestamp"]
                                for i in range(len(joint_state)):
                                    header.append(f"joint{i+1}")
                                writer.writerow(header)
                            
                            # 写入关节数据行
                            row = [arm_counter, timestamp] + list(joint_state)
                            writer.writerow(row)
                            arm_counter += 1
                        
                    # 定期记录保存进度
                    
                    print(f"已保存剩余 {arm_counter} 帧数据")

                except Empty:
                    # 队列为空，继续等待
                    continue
                except Exception as e:
                    logger.error(f"处理显示数据时出错: {e}")
                    
        except Exception as e:
            logger.exception(f"数据显示进程异常: {e}")
        finally:
            logger.info("数据显示进程已结束")

