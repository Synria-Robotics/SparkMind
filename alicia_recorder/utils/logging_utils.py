"""
日志工具模块
"""

import os
import logging
import datetime
from pathlib import Path
from typing import Optional

def setup_logging(log_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    设置日志记录器
    
    参数:
        log_dir: 日志文件目录，如果为None则不输出到文件
        level: 日志级别
        
    返回:
        根日志记录器
    """
    # 创建根日志记录器
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # 清理现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # 添加处理器到记录器
    logger.addHandler(console_handler)
    
    # 如果指定了日志目录，创建文件处理器
    if log_dir:
        # 确保日志目录存在
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # 生成日志文件名，包含时间戳
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_path / f"recorder_{timestamp}.log"
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        
        # 添加到记录器
        logger.addHandler(file_handler)
        
        logger.info(f"日志文件位置: {log_file}")
    
    return logger 