#!/usr/bin/env python3
# coding=utf-8

"""
数据集可视化工具 - 显示图像和关节数据的折线图
"""

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
from matplotlib.widgets import Slider, Button


class DatasetVisualizer:
    def __init__(self, config_path, dataset_path):
        """初始化数据集可视化器"""
        self.dataset_path = dataset_path
        self.current_frame = 0
        self.total_frames = 0
        
        # 读取配置文件
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # 获取摄像头名称
        self.camera_names = self.config['camera']['names']
        self.arm_name = self.config['arm']['name']
        
        # 加载数据集信息
        self.load_dataset_info()
        
        # 创建可视化界面
        self.create_visualization()
    
    def load_dataset_info(self):
        """加载数据集信息，确定图像和关节数据文件"""
        # 检查数据集路径是否存在
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"数据集路径不存在: {self.dataset_path}")
        
        # 加载关节数据
        arm_csv_path = os.path.join(self.dataset_path, self.arm_name, "joint_data.csv")
        if not os.path.exists(arm_csv_path):
            raise FileNotFoundError(f"找不到关节数据文件: {arm_csv_path}")
        
        self.joint_data = pd.read_csv(arm_csv_path)
        self.total_frames = len(self.joint_data)
        
        if self.total_frames == 0:
            raise ValueError("关节数据文件中没有有效数据")
        
        # 确定关节数量
        joint_columns = [col for col in self.joint_data.columns if col.startswith('joint')]
        self.joint_count = len(joint_columns)
        
        print(f"加载了关节数据: {self.total_frames} 帧, {self.joint_count} 个关节")
        
        # 检查每个摄像头的图像文件
        self.camera_image_paths = {}
        for camera_name in self.camera_names:
            camera_dir = os.path.join(self.dataset_path, "camera", camera_name)
            if not os.path.exists(camera_dir):
                print(f"警告: 找不到摄像头目录: {camera_dir}")
                continue
            
            # 获取该摄像头的所有图像路径
            image_paths = sorted(glob.glob(os.path.join(camera_dir, "*.jpg")))
            if not image_paths:
                print(f"警告: 在 {camera_dir} 中没有发现图像文件")
                continue
            
            self.camera_image_paths[camera_name] = image_paths
            print(f"加载了摄像头 {camera_name} 的图像: {len(image_paths)} 帧")
    
    def create_visualization(self):
        """创建可视化界面"""
        # 设置图表
        plt.ion()  # 交互模式
        self.fig = plt.figure(figsize=(15, 8))
        self.fig.canvas.manager.set_window_title("Dataset Visualization Tool")

        # 根据摄像头数量和关节数据创建图表布局
        n_cameras = len([name for name in self.camera_names if name in self.camera_image_paths])
        
        # 创建网格布局
        gs = GridSpec(2, max(n_cameras, 1))
        
        # 创建摄像头图像子图
        self.camera_axes = {}
        for i, camera_name in enumerate([name for name in self.camera_names if name in self.camera_image_paths]):
            ax = self.fig.add_subplot(gs[0, i])
            ax.set_title(camera_name)
            ax.axis('off')
            self.camera_axes[camera_name] = ax
        
        # 创建关节数据图表
        self.joint_ax = self.fig.add_subplot(gs[1, :])
        self.joint_ax.set_title("Joint Angle Data")
        self.joint_ax.set_xlabel("Frame")
        self.joint_ax.set_ylabel("Joint Angle Values")
        
        # 在控制区域添加滑块和按钮
        plt.subplots_adjust(bottom=0.2)  # 为控制区域留出空间
        
        # 添加帧滑块
        ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
        self.frame_slider = Slider(
            ax=ax_slider,
            label='Frame',
            valmin=0,
            valmax=max(self.total_frames - 1, 1),
            valinit=0,
            valstep=1
        )
        self.frame_slider.on_changed(self.update_frame)
        
        # 添加导航按钮
        ax_prev = plt.axes([0.1, 0.1, 0.05, 0.03])
        ax_next = plt.axes([0.9, 0.1, 0.05, 0.03])
        self.btn_prev = Button(ax_prev, 'next')
        self.btn_next = Button(ax_next, 'prev')
        self.btn_prev.on_clicked(self.prev_frame)
        self.btn_next.on_clicked(self.next_frame)
        
        # 显示初始帧
        self.update_display()
        
        # 保持窗口打开
        plt.show(block=True)
    
    def update_frame(self, val):
        """通过滑块更新当前帧"""
        self.current_frame = int(val)
        self.update_display()
    
    def prev_frame(self, event):
        """切换到上一帧"""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.frame_slider.set_val(self.current_frame)
    
    def next_frame(self, event):
        """切换到下一帧"""
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.frame_slider.set_val(self.current_frame)
    
    def update_display(self):
        """更新显示的图像和关节数据"""
        # 获取当前帧的关节数据
        current_joint_data = self.joint_data.iloc[self.current_frame]
        frame_id = int(current_joint_data['frame_id'])  # 添加类型转换
        timestamp = current_joint_data['timestamp']
        
        # 更新标题
        self.fig.suptitle(f"Frame {frame_id} (Timestamp: {timestamp:.3f})")
        
        # 显示每个摄像头的图像
        for camera_name, ax in self.camera_axes.items():
            if camera_name in self.camera_image_paths:
                # 如果当前帧号小于该摄像头的图像数量
                if frame_id < len(self.camera_image_paths[camera_name]):
                    image_path = self.camera_image_paths[camera_name][frame_id]
                    if os.path.exists(image_path):
                        img = cv2.imread(image_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB
                        ax.clear()
                        ax.imshow(img)
                        ax.set_title(f"{camera_name} - Frame {frame_id}")
                        ax.axis('off')
                    else:
                        ax.clear()
                        ax.text(0.5, 0.5, f"图像不存在\n{image_path}", 
                                ha='center', va='center', transform=ax.transAxes)
                        ax.axis('off')
                else:
                    ax.clear()
                    ax.text(0.5, 0.5, f"帧 {frame_id} 超出图像范围\n最大帧: {len(self.camera_image_paths[camera_name])-1}", 
                            ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
        
        # 更新关节数据图表
        self.joint_ax.clear()
        
        # 获取所有关节数据列
        joint_columns = [col for col in self.joint_data.columns if col.startswith('joint')]
        
        # 绘制过去和当前的关节数据
        window_size = min(100, self.total_frames)  # 显示最多100帧的历史
        start_idx = max(0, self.current_frame - window_size)
        end_idx = self.current_frame + 1  # 包含当前帧
        
        x_values = self.joint_data['frame_id'].iloc[start_idx:end_idx].values
        
        for i, joint_col in enumerate(joint_columns):
            y_values = self.joint_data[joint_col].iloc[start_idx:end_idx].values
            self.joint_ax.plot(x_values, y_values, 'o-', markersize=2, label=f"Joint {i+1}")
            
            # 标记当前值
            if self.current_frame >= start_idx:
                current_y = self.joint_data[joint_col].iloc[self.current_frame]
                self.joint_ax.plot(frame_id, current_y, 'ro', markersize=6)
        
        # 设置x轴范围，确保当前帧在图表的右侧
        self.joint_ax.set_xlim(x_values[0], x_values[-1] + (x_values[-1] - x_values[0]) * 0.1)
        
        # 添加图例
        self.joint_ax.legend(loc='upper left', ncol=min(5, len(joint_columns)))
        self.joint_ax.set_title("Joint Angle History")
        self.joint_ax.set_xlabel("Frame Number")
        self.joint_ax.set_ylabel("Joint Angle Values")
        self.joint_ax.grid(True)
        
        # 刷新图表
        self.fig.canvas.draw_idle()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='显示数据集图像和关节数据')
    parser.add_argument('-c', '--config', default='config/api_test_config.json',
                        help='配置文件路径 (默认: config/api_test_config.json)')
    parser.add_argument('-d', '--dataset', required=True,default='data/my_dataset/epoch1',
                        help='数据集路径，包含camera和robot_arm目录的根目录')
    
    args = parser.parse_args()
    
    try:
        visualizer = DatasetVisualizer(args.config, args.dataset)
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()