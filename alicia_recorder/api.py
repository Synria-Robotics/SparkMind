#!/usr/bin/env python3
# coding=utf-8

"""
Alicia机械臂和摄像头API
提供简单的接口用于数据采集 - 多进程版本
"""

import os
import sys
import time
import json
import logging
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .core.process_manager import ProcessManager


logger = logging.getLogger(__name__)

# 全局变量来管理 ProcessManager 实例和配置
_process_manager: Optional[ProcessManager] = None
_current_config: Optional[Dict] = None


DEFAULT_ARM_CSV_HEADERS = [
    'sync_timestamp', 'arm_timestamp',
    'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper'
]

def initialize(config_path: str, mode: str = "eval") -> Tuple[bool, Optional[Dict]]:
    """
    初始化数据采集系统，包括 ProcessManager。
    """
    global _process_manager, _current_config, _data_saver_instance
    if _process_manager is not None:
        logger.warning("系统已初始化。如需重新初始化，请先调用 cleanup()")
        return True, _current_config

    # 清理任何可能残留的旧状态，以确保干净的初始化
    _process_manager = None
    _current_config = None
    _data_saver_instance = None


    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        _current_config = config_data
        
        _process_manager = ProcessManager(config_data,mode)
        if not _process_manager.start_processes():
            logger.error("ProcessManager 启动失败。")
            _process_manager = None
            _current_config = None
            return False, None
        
        logger.info("系统初始化成功，ProcessManager 已启动。")
        return True, _current_config
    except FileNotFoundError:
        logger.error(f"配置文件未找到: {config_path}")
        return False, None
    except Exception as e:
        logger.error(f"初始化过程中发生错误: {e}", exc_info=True)
        return False, None

# #创建数据集
# def create_dataset_directories(
#     config: Dict, 
#     base_dataset_dir: str = "data/alicia_datasets", 
#     dataset_name_prefix: str = "dataset"
# ) -> Optional[str]:
#     """
#     创建版本化的数据集目录结构，包括 epochN 子文件夹。
#     返回创建的 epochN 目录的绝对路径。
#     """
#     if not config:
#         logger.error("配置信息未提供，无法创建数据集目录。")
#         return None

#     try:
#         base_dir = Path(base_dataset_dir).resolve()
#         dataset_name = config.get("dataset_name", dataset_name_prefix) 
        
#         main_dataset_path = base_dir / dataset_name
#         main_dataset_path.mkdir(parents=True, exist_ok=True)
        
#         epoch_idx = 0
#         while True:

#             epoch_dir_name = f"epoch{epoch_idx}"
#             current_epoch_path = main_dataset_path / epoch_dir_name
#             if not current_epoch_path.exists():
#                 current_epoch_path.mkdir(parents=True)
#                 logger.info(f"创建 epoch 目录: {current_epoch_path}")
#                 break
#             epoch_idx += 1
#             if epoch_idx > 10000: 
#                 logger.error("无法找到可用的 epoch 目录，已尝试超过10000次。")
#                 return None
                    
#         return str(current_epoch_path)

#     except Exception as e:
#         logger.error(f"创建数据集目录时出错: {e}", exc_info=True)
#         return None

def create_dataset_directories(
    config: Dict, 
    base_dataset_dir: str = "data/", 
    dataset_name_prefix: str = "dataset"
) -> Optional[str]:
    """
    创建版本化的数据集目录结构，包括 epochN 及其机械臂和摄像头子目录。
    
    参数:
        config: 配置字典，包含机械臂和摄像头名称
        base_dataset_dir: 数据集基础目录路径
        dataset_name_prefix: 数据集名称前缀
        
    返回:
        创建的 epochN 目录的绝对路径，创建失败则返回 None
    """
    if not config:
        logger.error("配置信息未提供，无法创建数据集目录。")
        return None

    try:
        # 获取机械臂和摄像头名称
        arm_name = config.get("arm", {}).get("name", "robot_arm")
        camera_names = config.get("camera", {}).get("names", [])
        
        # 创建基础目录结构
        base_dir = Path(base_dataset_dir).resolve()
        dataset_name = config.get("dataset", {}).get("name", dataset_name_prefix)
        
        main_dataset_path = base_dir / dataset_name
        main_dataset_path.mkdir(parents=True, exist_ok=True)
        
        # 查找可用的 epoch 目录
        epoch_idx = 0
        while True:
            epoch_dir_name = f"epoch{epoch_idx}"
            current_epoch_path = main_dataset_path / epoch_dir_name
            
            if not current_epoch_path.exists():
                # 创建主 epoch 目录
                current_epoch_path.mkdir(parents=True)
                logger.info(f"创建 epoch 目录: {current_epoch_path}")
                
                # 创建机械臂数据目录
                arm_dir = current_epoch_path / arm_name
                arm_dir.mkdir(parents=True)
                logger.info(f"创建机械臂数据目录: {arm_dir}")
                
                # 创建摄像头总目录
                camera_base_dir = current_epoch_path / "camera"
                camera_base_dir.mkdir(parents=True)
                
                # 为每个摄像头创建子目录
                for cam_name in camera_names:
                    camera_dir = camera_base_dir / cam_name
                    camera_dir.mkdir(parents=True)
                    logger.info(f"创建摄像头数据目录: {camera_dir}")
                
                break
                
            epoch_idx += 1
            if epoch_idx > 10000:
                logger.error("无法找到可用的 epoch 目录，已尝试超过10000次。")
                return None
                    
        return str(current_epoch_path)
        
    except Exception as e:
        logger.exception(f"创建数据集目录时出错: {e}")
        return None


def capture_data(timeout: float = 1.0) -> Optional[Dict[str, Any]]:
    """
    获取同步数据，并在记录激活时自动保存。
    """
    global _process_manager, _data_saver_instance

    if _process_manager is None:
        logger.error("ProcessManager 未初始化。请先调用 initialize()")
        return None

    try:
        sync_data_packet = _process_manager.get_synchronized_data(timeout=timeout)
        #print("获取到同步数据:", sync_data_packet)
        if sync_data_packet is not None:
            if _data_saver_instance is not None:
                _data_saver_instance.add_data_to_queue(sync_data_packet)
            return sync_data_packet
        else:
            return None
    except Exception as e:
        logger.error(f"capture_data 时发生错误: {e}", exc_info=True)
        return None

def cleanup():
    """
    清理所有资源。
    """
    global _process_manager, _current_config, _data_saver_instance
    
    logger.info("开始清理系统资源...")


    if _process_manager is not None:
        try:
            _process_manager.stop_processes()
            logger.info("ProcessManager 已停止。")
        except Exception as e:
            logger.error(f"停止 ProcessManager 时发生错误: {e}", exc_info=True)
        finally:
            _process_manager = None
    else:
        logger.info("ProcessManager 未运行，无需停止。")

    _current_config = None
    _data_saver_instance = None
    
    logger.info("系统资源清理完成。")

def save_data(save_path) :
    """
    获取同步数据，并在记录激活时自动保存。
    """
    global _process_manager, _data_saver_instance

    if _process_manager is None:
        logger.error("ProcessManager 未初始化。请先调用 initialize()")
        return None
    
    _process_manager.start_save(save_path)

def stop_save_data() :
    """
    获取同步数据，并在记录激活时自动保存。
    """
    global _process_manager, _data_saver_instance

    if _process_manager is None:
        logger.error("ProcessManager 未初始化。请先调用 initialize()")
        return None
    
    _process_manager.stop_save()
    
    
    



