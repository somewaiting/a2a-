#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: mcp_progress_server.py
作者: ZZS
项目: My_agent（基于A2A协议的企业签订进度智能查询/修改系统）
创建日期: 2026/8/26
描述: 签订进度查询 MCP 服务器（ProgressTools，端口 8020）
      工具：query_progress —— 仅允许 SELECT，强行禁止 DELETE 等写操作
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


def create_progress_mcp_server():
    progress_mcp = FastMCP(name="ProgressTools",
                           instructions="签订进度查询工具，基于 sign_progress 表，仅支持 SELECT 查询。",
                           log_level="ERROR",
                           host="127.0.0.1", port=conf.mcp_ports["progress"])

    service = MyAgentService()

    @progress_mcp.tool(
        name="query_progress",
        description="查询签订进度数据，输入 SELECT SQL（仅查询，支持 LIKE 模糊查询，如 "
                    "SELECT * FROM sign_progress WHERE company_name LIKE '%西湖%'）"
    )
    def query_progress(sql: str) -> str:
        logger.info(f"执行签订进度查询: {sql}")
        return service.execute_query(sql)

    logger.info("=== 签订进度MCP服务器信息 ===")
    logger.info(f"名称: {progress_mcp.name} / 端口: {conf.mcp_ports['progress']}")

    try:
        print("服务器已启动，请访问 http://127.0.0.1:8020/mcp")
        progress_mcp.run(transport="streamable-http")
    except Exception as e:
        print(f"服务器启动失败: {e}")


if __name__ == '__main__':
    create_progress_mcp_server()
