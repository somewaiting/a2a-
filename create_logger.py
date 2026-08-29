#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: create_logger.py
作者: ZZS
项目: My_agent（基于A2A协议的企业签订进度智能查询/修改系统）
创建日期: 2026/8/26
描述: 日志初始化
"""
import logging
import os

from My_agent.config import Config


def setup_logger(name, log_file='logs/my_agent.log'):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter('%(name)s - %(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename=log_file, encoding="utf-8", mode="a")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


logger = setup_logger('My_agent', Config().log_file)
