import cv2
import time
import logging
import threading
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

class Camera:
    """单个摄像头类，负责与摄像头设备通信和捕获图像"""
    
    def __init__(self, camera_id: Union[int, str], name: str, width: int = 640, height: int = 480, fps: int = 30):
        """
        初始化摄像头
        
        参数:
            camera_id: 摄像头ID，可以是整数索引或视频设备路径
            name: 摄像头名称，用于标识不同摄像头
            width: 捕获图像宽度
            height: 捕获图像高度
            fps: 摄像头帧率
        """
        self.logger = logging.getLogger(f"Camera[{name}]")
        self.camera_id = camera_id
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps
        
        # 摄像头连接状态
        self.cap = None
        self.is_connected = False
        
        # 最近捕获的图像
        self.current_frame = None
        self.frame_count = 0
        self.last_frame_time = 0
        
        self.logger.info(f"初始化摄像头: ID={camera_id}, 名称={name}, 分辨率={width}x{height}, FPS={fps}")
        
    def connect(self) -> bool:
        """
        连接到摄像头
        
        返回:
            连接是否成功
        """
        if self.is_connected:
            self.logger.warning("摄像头已连接，忽略请求")
            return True
            
        try:
            self.logger.info(f"正在连接摄像头: {self.camera_id}")
            self.cap = cv2.VideoCapture(self.camera_id)
            
            # 设置摄像头性能优先级
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小缓冲区，减少延迟
            
            # 设置分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # 设置帧率
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # 检查连接是否成功
            if not self.cap.isOpened():
                self.logger.error(f"无法连接到摄像头: {self.camera_id}")
                return False
                
            # 读取一帧测试是否工作正常
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.logger.error("无法从摄像头读取图像")
                self.cap.release()
                return False
                
            self.is_connected = True
            self.current_frame = frame
            self.frame_count = 1
            self.last_frame_time = time.time()
            
            # 获取实际分辨率和帧率
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"摄像头连接成功，实际分辨率: {actual_width}x{actual_height}, 实际帧率: {actual_fps}")
            
            # 如果实际值与请求值不同，更新参数
            if actual_width != self.width or actual_height != self.height:
                self.logger.warning(f"实际分辨率与请求不符: 请求={self.width}x{self.height}, 实际={actual_width}x{actual_height}")
                self.width = actual_width
                self.height = actual_height
                
            if abs(actual_fps - self.fps) > 0.1:  # 允许浮点数差异
                self.logger.warning(f"实际帧率与请求不符: 请求={self.fps}, 实际={actual_fps}")
                self.fps = actual_fps
                
            return True
        except Exception as e:
            self.logger.error(f"连接摄像头时出错: {e}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False
            
    def disconnect(self) -> None:
        """断开与摄像头的连接"""
        if not self.is_connected:
            return
            
        self.logger.info("正在断开摄像头连接")
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_connected = False
        
    def capture_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """
        捕获一帧图像
        
        返回:
            (捕获的图像, 时间戳)元组, 如果失败则返回(None, 0)
        """
        if not self.is_connected or self.cap is None:
            self.logger.warning("摄像头未连接，无法捕获图像")
            return None, 0.0
            
        try:
            # 读取图像
            ret, frame = self.cap.read()
            
            # 立即记录时间戳
            timestamp = time.time()
            
            if not ret or frame is None:
                self.logger.warning("读取图像失败")
                return None, 0.0
                
            self.current_frame = frame
            self.frame_count += 1
            
            # 更新帧时间
            self.last_frame_time = timestamp
            
            return frame, timestamp
        except Exception as e:
            self.logger.error(f"捕获图像时出错: {e}")
            return None, 0.0
            
    def get_resolution(self) -> Tuple[int, int]:
        """
        获取摄像头当前分辨率
        
        返回:
            (宽度, 高度) 元组
        """
        return (self.width, self.height)
        
    def set_resolution(self, width: int, height: int) -> bool:
        """
        设置摄像头分辨率
        
        参数:
            width: 宽度
            height: 高度
            
        返回:
            设置是否成功
        """
        if not self.is_connected or self.cap is None:
            self.logger.warning("摄像头未连接，无法设置分辨率")
            return False
            
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # 验证设置是否成功
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width != width or actual_height != height:
                self.logger.warning(f"设置分辨率失败: 请求={width}x{height}, 实际={actual_width}x{actual_height}")
                return False
                
            self.width = width
            self.height = height
            self.logger.info(f"分辨率已设置为: {width}x{height}")
            return True
        except Exception as e:
            self.logger.error(f"设置分辨率时出错: {e}")
            return False


class CameraManager:
    """摄像头管理类，负责管理多个摄像头"""
    
    def __init__(self):
        """初始化摄像头管理器"""
        self.logger = logging.getLogger("CameraManager")
        self.cameras = {}  # 摄像头字典，键为摄像头名称
        self.logger.info("初始化摄像头管理器")
        
    def add_camera(self, camera_id: Union[int, str], name: str, width: int = 640, height: int = 480, fps: int = 30) -> bool:
        """
        添加摄像头
        
        参数:
            camera_id: 摄像头ID
            name: 摄像头名称
            width: 图像宽度
            height: 图像高度
            fps: 帧率
            
        返回:
            添加是否成功
        """
        if name in self.cameras:
            self.logger.warning(f"名称为 '{name}' 的摄像头已存在")
            return False
            
        try:
            camera = Camera(camera_id, name, width, height, fps)
            self.cameras[name] = camera
            self.logger.info(f"添加摄像头: {name}")
            return True
        except Exception as e:
            self.logger.error(f"添加摄像头时出错: {e}")
            return False
            
    def remove_camera(self, name: str) -> bool:
        """
        移除摄像头
        
        参数:
            name: 摄像头名称
            
        返回:
            移除是否成功
        """
        if name not in self.cameras:
            self.logger.warning(f"名称为 '{name}' 的摄像头不存在")
            return False
            
        try:
            camera = self.cameras[name]
            if camera.is_connected:
                camera.disconnect()
            del self.cameras[name]
            self.logger.info(f"移除摄像头: {name}")
            return True
        except Exception as e:
            self.logger.error(f"移除摄像头时出错: {e}")
            return False
            
    def connect_all(self) -> Dict[str, bool]:
        """
        连接所有摄像头
        
        返回:
            包含每个摄像头连接结果的字典
        """
        self.logger.info("正在连接所有摄像头...")
        results = {}
        
        for name, camera in self.cameras.items():
            results[name] = camera.connect()
            
        return results
        
    def disconnect_all(self) -> None:
        """断开所有摄像头的连接"""
        self.logger.info("正在断开所有摄像头连接...")
        for camera in self.cameras.values():
            if camera.is_connected:
                camera.disconnect()
                
    def capture_all(self) -> Dict[str, Dict[str, Any]]:
        """
        从所有摄像头捕获图像
        
        返回:
            包含每个摄像头捕获图像和时间戳的字典
            格式: {camera_name: {'image': image, 'timestamp': timestamp}}
        """
        images_data = {}
        for name, camera in self.cameras.items():
            if camera.is_connected:
                frame, timestamp = camera.capture_frame()
                images_data[name] = {
                    'image': frame,
                    'timestamp': timestamp
                }
            else:
                images_data[name] = {
                    'image': None,
                    'timestamp': 0.0
                }
                
        return images_data
        
    def get_camera(self, name: str) -> Optional[Camera]:
        """
        获取指定摄像头
        
        参数:
            name: 摄像头名称
            
        返回:
            摄像头对象，如果不存在则返回None
        """
        return self.cameras.get(name)
        
    def get_all_cameras(self) -> Dict[str, Camera]:
        """
        获取所有摄像头
        
        返回:
            所有摄像头的字典
        """
        return self.cameras.copy()
        
    def get_connected_cameras(self) -> Dict[str, Camera]:
        """
        获取所有已连接的摄像头
        
        返回:
            已连接摄像头的字典
        """
        return {name: camera for name, camera in self.cameras.items() if camera.is_connected}
        
    def save_images(self, images_data: Dict[str, Dict[str, Any]], save_dir: Union[str, Path], prefix: str = "") -> Dict[str, str]:
        """
        保存图像到指定目录
        
        参数:
            images_data: 要保存的图像数据字典
            save_dir: 保存目录
            prefix: 文件名前缀
            
        返回:
            保存路径的字典
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time() * 1000)
        paths = {}
        
        for name, data in images_data.items():
            image = data.get('image')
            if image is None:
                continue
                
            filename = f"{prefix}_{name}_{timestamp}.png"
            file_path = save_path / filename
            
            try:
                cv2.imwrite(str(file_path), image)
                paths[name] = str(file_path)
            except Exception as e:
                self.logger.error(f"保存图像时出错: {e}")
                
        return paths 