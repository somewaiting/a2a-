#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: mcp_status_server.py
作者: ZZS
项目: My_agent（基于A2A协议的企业签订进度智能查询/修改系统）
创建日期: 2026/8/26
描述: 签订状态修改 MCP 服务器（StatusTools，端口 8022）
      工具：update_sign_status —— 修改签订状态（驳回需操作人姓名 + 原因）
"""
import sys
from pathlib import Path

# 将 My_agent 的父目录（New）加入 sys.path
_root = Path(__file__).resolve().parent
while _root.name != "My_agent" and _root.parent != _root:
    _root = _root.parent
if str(_root.parent) not in sys.path:
    sys.path.insert(0, str(_root.parent))

from mcp.server.fastmcp import FastMCP

from My_agent.config import Config
from My_agent.create_logger import logger
from My_agent.query_data.my_agent_service import MyAgentService

conf = Config()


def create_status_mcp_server():
    status_mcp = FastMCP(name="StatusTools",
                         instructions="签订状态修改工具，通过专用接口修改 sign_progress 状态，不直接执行 SQL。",
                         log_level="ERROR",
                         host="127.0.0.1", port=conf.mcp_ports["status"])

    service = MyAgentService()

    @status_mcp.tool(
        name="update_sign_status",
        description="修改某公司/项目的签订状态：传入目标（公司名称，支持模糊，或进度ID）、新状态(洽谈中/待签约/已签约/已驳回)、"
                    "操作人姓名；若状态为已驳回必须提供原因(reason)；trace_id 为日志追踪ID（可选，从对话上下文中提取）"
    )
    def update_sign_status(target: str, new_status: str, operator_name: str, reason: str = '', trace_id: str = '') -> str:
        logger.info(f"修改签订状态: target={target}, 新状态={new_status}, 操作人={operator_name}, trace={trace_id}")
        return service.update_sign_status(target, new_status, operator_name, reason, trace_id=trace_id)

    logger.info("=== 签订状态修改MCP服务器信息 ===")
    logger.info(f"名称: {status_mcp.name} / 端口: {conf.mcp_ports['status']}")

    try:
        print("服务器已启动，请访问 http://127.0.0.1:8022/mcp")
        status_mcp.run(transport="streamable-http")
    except Exception as e:
        print(f"服务器启动失败: {e}")


if __name__ == '__main__':
    create_status_mcp_server()
