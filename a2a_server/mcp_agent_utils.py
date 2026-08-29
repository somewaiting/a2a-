#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: mcp_agent_utils.py
作者: ZZS
项目: My_agent（基于A2A协议的企业签订进度智能查询/修改系统）
创建日期: 2026/8/26
描述: 共享的「A2A Agent 调 MCP 工具」工具函数：
      通过 streamable-http 连接 MCP 服务器 -> 自动发现工具 -> LangChain tool-calling Agent 抽取参数并调用。
      带超时与指数退避重试，MCP 短暂不可用/抖动时可自愈。
"""

import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from My_agent.create_logger import logger
from My_agent.utils.retry import with_retry


async def _do_run(llm, mcp_url: str, system_prompt: str, query: str) -> dict:
    """单次执行：建立 MCP 会话 → tool-calling Agent 完成任务（带 60s 超时）。"""
    async with streamablehttp_client(mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ])
            agent = create_tool_calling_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            response = await asyncio.wait_for(agent_executor.ainvoke({"input": query}), timeout=60)
            return {"status": "success", "message": f"{response['output']}"}


async def run_tool_agent(llm, mcp_url: str, system_prompt: str, query: str) -> dict:
    """
    连接 MCP 服务器并让 tool-calling Agent 完成任务（带超时与指数退避重试）。
    :param llm: LangChain ChatOpenAI 实例
    :param mcp_url: MCP 服务器地址，如 http://127.0.0.1:8022/mcp
    :param system_prompt: Agent 系统提示词（职责 + 工具使用约束 + 参数不足追问）
    :param query: 用户查询（含对话上下文）
    :return: {"status": "success"/"error", "message": ...}
    """
    try:
        return await with_retry(
            lambda: _do_run(llm, mcp_url, system_prompt, query),
            retries=2, base_delay=0.5,
        )
    except asyncio.TimeoutError:
        logger.error(f"MCP 工具调用超时: {mcp_url}")
        return {"status": "error", "message": "服务响应超时，请稍后重试。"}
    except Exception as e:
        logger.error(f"MCP 工具调用失败: {mcp_url} {str(e)}")
        return {"status": "error", "message": f"MCP 工具调用失败：{str(e)}"}
