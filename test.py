#!/usr/bin/env python3
# coding=utf-8

"""
测试Alicia机械臂和摄像头API (多进程版本)
"""

import os
import sys
import time
import cv2
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alicia_recorder.api import initialize, capture_data, cleanup, create_dataset_directories ,stop_save_data,save_data

def test_api():
    """
    测试API功能
    """
    print("开始测试Alicia API (多进程版本)...")
    
    # 配置文件路径
    config_path = os.path.join(os.path.dirname(__file__), "config", "api_test_config.json")
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        # 创建示例配置
        example_config = {
            "arm": {
                "port": "/dev/ttyUSB0",  # 请根据实际情况修改
                "baudrate": 921600,
                "debug_mode": False
            },
            "camera": {
                "ids": [0, 2],  # 请根据实际情况修改
                "names": ["main_camera", "secondary_camera"],
                "width": 640,
                "height": 480,
                "fps": 30
            }
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # 写入示例配置
        with open(config_path, 'w') as f:
            json.dump(example_config, f, indent=4)
        
        print(f"已创建示例配置文件: {config_path}")
        print("请根据实际情况修改配置文件后重新运行测试")
        return
    
    try:
        # 1. 测试初始化API
        print(f"\n=== 测试初始化API ===")
        success, config = initialize(config_path,mode="record")
        
        if not success:
            print("初始化API失败，测试终止")
            return
        
        print("初始化API成功")
        print(f"加载的配置: {config}")
        
        # # 测试创建数据集目录
        # print(f"\n=== 测试创建数据集目录 ===")
        # for i in range(10):
        #     dataset_root = create_dataset_directories(config)
        #     if dataset_root:
        #         print(f"数据集目录已成功创建/验证于: {dataset_root}")

        #dataset_root = create_dataset_directories(config)
        # 2. 测试采集数据
        print(f"\n=== 测试采集数据 ===")
        print("开始采集数据，按'q'键退出...")
        
        # 创建窗口用于显示图像
        # for name in config["camera"]["names"]:
        #     cv2.namedWindow(f"cap: {name}", cv2.WINDOW_NORMAL)
        # Initialize variables for FPS calculation
        prev_time = 0
        fps_display = 0
        total_frames_for_avg_fps = 0
        accumulated_time_for_avg_fps = 0.0
        while True:
            # 采集一帧数据
            data = capture_data()
            if data is None:
                #print("未获取到数据，可能是超时或其他错误")
                continue
            current_time = time.time()
            if prev_time > 0: # Avoid division by zero on the first frame
                time_diff_fps = current_time - prev_time
                if time_diff_fps > 0: # Ensure positive time difference
                    total_frames_for_avg_fps += 1
                    accumulated_time_for_avg_fps += time_diff_fps
                    fps_display = total_frames_for_avg_fps / accumulated_time_for_avg_fps
            prev_time = current_time

            #print(1)
            # 显示时间戳
            #timestamp = data['timestamp']
            #print(f"\n主时间戳: {timestamp:.6f}, FPS: {fps_display:.2f}")
            
            # 显示机械臂数据
            # arm_data = data['arm_data']
            # arm_timestamp = data.get('arm_timestamp', 0)
            
            # if arm_data:
            #     print(f"机械臂时间戳: {arm_timestamp:.6f}")
            #     if 'joint_state' in arm_data:
            #         joint_state = arm_data['joint_state']
            #         if len(joint_state) >= 6:
            #             print(f"关节角度: {joint_state[:6]}")
            #         if len(joint_state) > 6:
            #             print(f"夹爪状态: {joint_state[6]}")
            
            # 显示图像数据
            images_data = data['images_data']
            for name, image_data in images_data.items():
                image = image_data['image']
                img_timestamp = image_data['timestamp']
                #print(f"图像: {name}, 时间戳: {img_timestamp:.6f}")

                if image is not None:
                    # 在图像上添加时间戳和摄像头名称
                    display_img = image.copy()
                    
                    # 显示基本信息
                    cv2.putText(display_img, f"{name}: {img_timestamp:.6f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 显示时间差
                    time_diff = data.get('time_diff', {})
                    if name + '_to_avg' in time_diff:
                        diff_ms = time_diff[name + '_to_avg'] * 1000
                        cv2.putText(display_img, f"time_diff: {diff_ms:.2f}ms", (10, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # 显示与机械臂的时间差
                    if 'arm_to_avg' in time_diff:
                        arm_diff_ms = time_diff['arm_to_avg'] * 1000
                        cv2.putText(display_img, f"arm_time_diff: {arm_diff_ms:.2f}ms", (10, 90), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    # 显示图像
                    cv2.imshow(f"cap: {name}", display_img)
            
            # 检查按键

            # 输出fps
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                print("用户请求退出")
                break
            elif key == ord('s'):
                dataset_root = create_dataset_directories(config)
                print(f"数据集目录已成功创建/验证于: {dataset_root}")
                save_data(dataset_root)
            elif key == ord('z'):
                print("停止保存数据")
                stop_save_data()

            
    except Exception as e:
        print(f"测试过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        print("\n=== 清理资源 ===")
        cleanup()
        
        # 关闭所有窗口
        cv2.destroyAllWindows()
        
        print("测试完成")

if __name__ == "__main__":
    test_api()