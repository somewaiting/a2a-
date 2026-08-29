#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: mcp_staff_server.py
作者: ZZS
项目: My_agent（基于A2A协议的企业签订进度智能查询/修改系统）
创建日期: 2026/8/26
描述: 职员查询 MCP 服务器（StaffTools，端口 8021）
      工具：query_staff —— 查询职员名单，返回职员所有相关信息（含其负责的签订项目）
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


def create_staff_mcp_server():
    staff_mcp = FastMCP(name="StaffTools",
                        instructions="职员名单查询工具，基于 employees 表，仅支持 SELECT 查询。",
                        log_level="ERROR",
                        host="127.0.0.1", port=conf.mcp_ports["staff"])

    service = MyAgentService()

    @staff_mcp.tool(
        name="query_staff",
        description="查询职员名单数据，输入 SELECT SQL（仅查询，支持 LIKE 模糊匹配姓名；"
                    "可通过 sign_progress.person_in_charge 关联查询该职员负责的签订项目）"
    )
    def query_staff(sql: str) -> str:
        logger.info(f"执行职员查询: {sql}")
        return service.execute_query(sql)

    logger.info("=== 职员查询MCP服务器信息 ===")
    logger.info(f"名称: {staff_mcp.name} / 端口: {conf.mcp_ports['staff']}")

    try:
        print("服务器已启动，请访问 http://127.0.0.1:8021/mcp")
        staff_mcp.run(transport="streamable-http")
    except Exception as e:
        print(f"服务器启动失败: {e}")


if __name__ == '__main__':
    create_staff_mcp_server()
