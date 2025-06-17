#!/usr/bin/env python3
# coding=utf-8

"""
Dataset Video Export Tool - Directly convert dataset to video without GUI
"""

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
import time


class DatasetExporter:
    def __init__(self, config_path, dataset_path, output_video_path, fps=30):
        """Initialize dataset exporter"""
        self.dataset_path = dataset_path
        self.output_video_path = output_video_path
        self.fps = fps
        
        # Read config file
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Get camera names
        self.camera_names = self.config['camera']['names']
        self.arm_name = self.config['arm']['name']
        
        # Load dataset info
        self.load_dataset_info()
        
    def load_dataset_info(self):
        """Load dataset information, identify image and joint data files"""
        # Check if dataset path exists
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
        
        # Load joint data
        arm_csv_path = os.path.join(self.dataset_path, self.arm_name, "joint_data.csv")
        if not os.path.exists(arm_csv_path):
            raise FileNotFoundError(f"Joint data file not found: {arm_csv_path}")
        
        self.joint_data = pd.read_csv(arm_csv_path)
        self.total_frames = len(self.joint_data)
        
        if self.total_frames == 0:
            raise ValueError("No valid data in joint data file")
        
        # Determine number of joints
        joint_columns = [col for col in self.joint_data.columns if col.startswith('joint')]
        self.joint_count = len(joint_columns)
        
        print(f"Loaded joint data: {self.total_frames} frames, {self.joint_count} joints")
        
        # Check image files for each camera
        self.camera_image_paths = {}
        for camera_name in self.camera_names:
            camera_dir = os.path.join(self.dataset_path, "camera", camera_name)
            if not os.path.exists(camera_dir):
                print(f"Warning: Camera directory not found: {camera_dir}")
                continue
            
            # Get all image paths for this camera
            image_paths = sorted(glob.glob(os.path.join(camera_dir, "*.jpg")))
            if not image_paths:
                print(f"Warning: No image files found in {camera_dir}")
                continue
            
            self.camera_image_paths[camera_name] = image_paths
            print(f"Loaded camera {camera_name} images: {len(image_paths)} frames")

    def export_video(self):
        """Export dataset to video file"""
        print(f"Starting video export to {self.output_video_path}...")
        start_time = time.time()
        
        # Create figure for rendering
        fig = plt.figure(figsize=(15, 8))
        
        # Create layout based on camera count and joint data
        n_cameras = len([name for name in self.camera_names if name in self.camera_image_paths])
        
        # Create grid layout
        gs = GridSpec(2, max(n_cameras, 1))
        
        # Create camera image subplots
        camera_axes = {}
        for i, camera_name in enumerate([name for name in self.camera_names if name in self.camera_image_paths]):
            ax = fig.add_subplot(gs[0, i])
            ax.set_title(camera_name)
            ax.axis('off')
            camera_axes[camera_name] = ax
        
        # Create joint data plot
        joint_ax = fig.add_subplot(gs[1, :])
        joint_ax.set_title("Joint Angle Data")
        joint_ax.set_xlabel("Frame")
        joint_ax.set_ylabel("Joint Angle Values")
        
        # Get figure dimensions
        fig_width, fig_height = fig.get_size_inches() * fig.dpi
        fig_width, fig_height = int(fig_width), int(fig_height)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (fig_width, fig_height))
        
        try:
            # Process each frame
            for current_frame in range(self.total_frames):
                # Get current frame joint data
                current_joint_data = self.joint_data.iloc[current_frame]
                frame_id = int(current_joint_data['frame_id'])
                timestamp = current_joint_data['timestamp']
                
                # Update title
                fig.suptitle(f"Frame {frame_id} (Timestamp: {timestamp:.3f})")
                
                # Display images for each camera
                for camera_name, ax in camera_axes.items():
                    if camera_name in self.camera_image_paths:
                        # If current frame exists in camera images
                        if frame_id < len(self.camera_image_paths[camera_name]):
                            image_path = self.camera_image_paths[camera_name][frame_id]
                            if os.path.exists(image_path):
                                img = cv2.imread(image_path)
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
                                ax.clear()
                                ax.imshow(img)
                                ax.set_title(f"{camera_name} - Frame {frame_id}")
                                ax.axis('off')
                            else:
                                ax.clear()
                                ax.text(0.5, 0.5, f"Image not found\n{image_path}", 
                                        ha='center', va='center', transform=ax.transAxes)
                                ax.axis('off')
                        else:
                            ax.clear()
                            ax.text(0.5, 0.5, f"Frame {frame_id} out of range\nMax frame: {len(self.camera_image_paths[camera_name])-1}", 
                                    ha='center', va='center', transform=ax.transAxes)
                            ax.axis('off')
                
                # Update joint data plot
                joint_ax.clear()
                
                # Get all joint data columns
                joint_columns = [col for col in self.joint_data.columns if col.startswith('joint')]
                
                # Plot past and current joint data
                window_size = min(100, self.total_frames)
                start_idx = max(0, current_frame - window_size)
                end_idx = current_frame + 1
                
                x_values = self.joint_data['frame_id'].iloc[start_idx:end_idx].values
                
                for i, joint_col in enumerate(joint_columns):
                    y_values = self.joint_data[joint_col].iloc[start_idx:end_idx].values
                    joint_ax.plot(x_values, y_values, 'o-', markersize=2, label=f"Joint {i+1}")
                    
                    # Mark current value
                    if current_frame >= start_idx:
                        current_y = self.joint_data[joint_col].iloc[current_frame]
                        joint_ax.plot(frame_id, current_y, 'ro', markersize=6)
                
                # Set x-axis range
                joint_ax.set_xlim(x_values[0], x_values[-1] + (x_values[-1] - x_values[0]) * 0.1)
                
                # Add legend
                joint_ax.legend(loc='upper left', ncol=min(5, len(joint_columns)))
                joint_ax.set_title("Joint Angle History")
                joint_ax.set_xlabel("Frame Number")
                joint_ax.set_ylabel("Joint Angle Values")
                joint_ax.grid(True)
                
                # Convert matplotlib figure to image
                fig.canvas.draw()
                img_buf = fig.canvas.buffer_rgba()
                img = np.frombuffer(img_buf, dtype=np.uint8)
                img = img.reshape(fig_height, fig_width, 4) # RGBA
                img = img[..., :3] # Convert to RGB by slicing off the alpha channel
                
                # Convert RGB to BGR (OpenCV uses BGR)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # Write to video
                writer.write(img)
                
                # Update progress
                if current_frame % 10 == 0 or current_frame == self.total_frames - 1:
                    elapsed = time.time() - start_time
                    progress = (current_frame + 1) / self.total_frames * 100
                    est_total = elapsed / (current_frame + 1) * self.total_frames
                    remaining = est_total - elapsed
                    print(f"Progress: {current_frame+1}/{self.total_frames} ({progress:.1f}%) - "
                          f"Time: {elapsed:.1f}s / Est. remaining: {remaining:.1f}s")
        
        finally:
            # Close video writer
            writer.release()
            plt.close(fig)
            
            # Print completion information
            total_time = time.time() - start_time
            print(f"Video export completed in {total_time:.1f} seconds")
            print(f"Saved to: {self.output_video_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Export dataset to video without GUI')
    parser.add_argument('-c', '--config', default='config/api_test_config.json',
                        help='Config file path (default: config/api_test_config.json)')
    parser.add_argument('-d', '--dataset', required=True,
                        help='Dataset path, root directory containing camera and robot_arm folders')
    parser.add_argument('-o', '--output', required=True,
                        help='Output video file path (.mp4)')
    parser.add_argument('-f', '--fps', type=int, default=30,
                        help='Frames per second for output video (default: 30)')
    
    args = parser.parse_args()
    
    try:
        exporter = DatasetExporter(args.config, args.dataset, args.output, args.fps)
        exporter.export_video()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()