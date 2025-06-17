#!/usr/bin/env python3
# coding=utf-8

"""
机械臂控制类
封装Alicia Duo SDK，提供简单的接口用于数据采集
"""

import os
import sys
import time
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

# 添加SDK路径
sdk_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Alicia_duo_sdk')
if sdk_path not in sys.path:
    sys.path.insert(0, sdk_path)

try:
    from alicia_duo_sdk.controller import ArmController as AliciaController
except ImportError:
    raise ImportError("无法导入Alicia SDK，请确保Alicia_duo_sdk目录在正确的位置")

class ArmController:
    """
    机械臂控制类，用于简化与Alicia机械臂的交互
    主要功能：连接/断开连接、读取关节角度和按钮状态
    """
    
    def __init__(self, port: str = "", baudrate: int = 921600, debug_mode: bool = False):
        """
        初始化机械臂控制器
        
        参数:
            port: 串口名称，留空则自动搜索
            baudrate: 波特率，默认921600
            debug_mode: 是否启用调试模式
        """
        self.logger = logging.getLogger("ArmController")
        self.logger.info("初始化机械臂控制器")
        
        # 创建SDK控制器实例
        self.controller = AliciaController(port=port, baudrate=baudrate, debug_mode=debug_mode)
        
        # 状态变量
        self.is_connected = False
        
    def connect(self) -> bool:
        """
        连接到机械臂
        
        返回:
            连接是否成功
        """
        self.logger.info("正在连接机械臂...")
        self.is_connected = self.controller.connect()
        
        if self.is_connected:
            self.logger.info("机械臂连接成功")
        else:
            self.logger.error("机械臂连接失败")
            
        return self.is_connected
    
    def disconnect(self) -> None:
        """
        断开与机械臂的连接
        """
        if self.is_connected:
            self.logger.info("断开机械臂连接")
            self.controller.disconnect()
            self.is_connected = False
    
    def read_joint_state(self) -> Dict[str, Any]:
        """
        读取机械臂关节角度和按钮状态
        
        返回:
            包含状态数据的字典:
            - 'joint_state': 7维数组，包含6个关节角度和夹爪角度（弧度）
            - 'button_states': 2维数组，表示2个按钮的状态
            - 'timestamp': 数据时间戳（秒）
        """
        if not self.is_connected:
            self.logger.warning("机械臂未连接，无法读取状态")
            return {
                'joint_state': np.zeros(7),
                'button_states': np.array([False, False]),
                'timestamp': time.time()
            }
        
        try:
            # 读取状态
            state = self.controller.read_joint_state()
            
            # 优先使用机械臂自带的时间戳，如果有的话
            timestamp = getattr(state, 'timestamp', time.time())
            # 构建联合状态数组
            joint_state = np.zeros(7)
            joint_state[:6] = state.angles  # 6个关节角度
            joint_state[6] = state.gripper  # 夹爪角度
            
            # 构建按钮状态数组
            button_states = np.array([state.button1, state.button2], dtype=bool)
            
            return {
                'joint_state': joint_state,
                'button_states': button_states,
                'timestamp': timestamp
            }
        except Exception as e:
            self.logger.error(f"读取状态时出错: {e}")
            return {
                'joint_state': np.zeros(7),
                'button_states': np.array([False, False]),
                'timestamp': time.time()
            } 