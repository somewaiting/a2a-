#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: health.py
作者: ZZS
项目: My_agent（基于A2A协议的企业签订进度智能查询/修改系统）
创建日期: 2026/8/26
描述: 健康检查与端口就绪等待工具（供前端降级、启动脚本等待/守护使用）
"""
import socket
import time


def check_tcp(host: str, port: int, timeout: float = 2.0) -> bool:
    """TCP 探测端口是否可连接。"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def check_agent_http(url: str, timeout: float = 3.0) -> bool:
    """探测 A2A Agent 的 HTTP 端点（/agent.json）是否存活。"""
    import urllib.request
    try:
        with urllib.request.urlopen(url.rstrip('/') + "/agent.json", timeout=timeout) as resp:
            return 200 <= resp.status < 500
    except Exception:
        return False


def wait_ports(host: str, ports, timeout: float = 60.0, interval: float = 1.0) -> set:
    """
    轮询等待所有端口就绪。
    :return: 超时后仍未就绪的端口集合（为空表示全部就绪）
    """
    pending = set(ports)
    start = time.time()
    while pending and time.time() - start < timeout:
        for p in list(pending):
            if check_tcp(host, p, timeout=1.0):
                pending.discard(p)
        if pending:
            time.sleep(interval)
    return pending
