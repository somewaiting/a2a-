#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: document_query_server.py
作者: ZZS
项目: My_agent（基于A2A协议的企业签订进度智能查询/修改系统）
创建日期: 2026/8/29
描述: 文档问答 Agent 服务器（DocumentQueryAgent，端口 5023）
      流程：从对话中提取用户问题 → 调 MCP 8023（RAG 检索 + 生成答案）→ 返回答案。
"""
import sys
import re
import json
import asyncio
from pathlib import Path

import httpx

# 将 My_agent 的父目录（New）加入 sys.path
_root = Path(__file__).resolve().parent
while _root.name != "My_agent" and _root.parent != _root:
    _root = _root.parent
if str(_root.parent) not in sys.path:
    sys.path.insert(0, str(_root.parent))

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from python_a2a import A2AServer, run_server, AgentCard, AgentSkill, TaskStatus, TaskState

from My_agent.config import Config
from My_agent.create_logger import logger
from My_agent.utils.retry import with_retry

conf = Config()

# 目标 MCP：RAG 文档问答 8023
RAG_MCP_URL = "http://127.0.0.1:8023/mcp"


def _no_proxy_http_client_factory(headers=None, timeout=None, auth=None):
    """
    构造 httpx.AsyncClient 时禁用系统代理（trust_env=False）。
    本机开启了 Windows 系统代理（如 127.0.0.1:12000），若不禁用，
    本地 MCP(8023) 请求会被代理转发并返回 502 Bad Gateway。
    """
    return httpx.AsyncClient(
        headers=headers,
        timeout=timeout or httpx.Timeout(120.0, read=300.0),
        auth=auth,
        follow_redirects=True,
        trust_env=False,
    )

agent_card = AgentCard(
    name="DocumentQueryAgent",
    description="基于已入库的企业文档（方案/合同/招标文件等）进行问答的助手",
    url="http://localhost:5023",
    version="1.0.0",
    capabilities={"streaming": True, "memory": True},
    skills=[
        AgentSkill(
            name="query uploaded documents",
            description="对已入库的企业文档内容进行问答，如招标文件、技术标、合同等",
            examples=["招标文件中B2区消防工程的主要技术要求是什么", "这份技术标包含哪些内容", "合同金额是多少"]
        )
    ]
)


async def _call_rag_mcp(query: str):
    async with streamablehttp_client(RAG_MCP_URL, httpx_client_factory=_no_proxy_http_client_factory) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await asyncio.wait_for(
                session.call_tool("search_documents", {"query": query}), timeout=120
            )
            return result.content[0].text


async def query_documents(query: str) -> dict:
    """调用 RAG MCP，返回 {"answer": str, "images": [{alt, description, mime, data(base64)}]}"""
    try:
        raw = await with_retry(lambda: _call_rag_mcp(query), retries=2, base_delay=0.5)
    except asyncio.TimeoutError:
        logger.error(f"RAG MCP 查询超时: {query}")
        return {"answer": "文档检索响应超时，请稍后重试。", "images": []}
    except Exception as e:
        logger.error(f"RAG MCP 查询出错：{str(e)}")
        return {"answer": f"文档检索出错：{str(e)}", "images": []}
    try:
        data = json.loads(raw)
        return {
            "answer": data.get("answer", raw),
            "images": data.get("images", []) or [],
        }
    except (json.JSONDecodeError, TypeError):
        # 兼容旧版：MCP 直接返回纯文本
        return {"answer": raw, "images": []}


def _build_artifact_parts(result: dict) -> list:
    """把答案与图片构造成 A2A artifacts 的 parts 列表（text + file）"""
    parts = [{"type": "text", "text": result.get("answer", "")}]
    for i, img in enumerate(result.get("images", []), start=1):
        mime = img.get("mime", "image/png")
        ext = mime.split("/")[-1] if "/" in mime else "png"
        if ext == "jpeg":
            ext = "jpg"
        parts.append({
            "type": "file",
            "file": {
                "mimeType": mime,
                "bytes": img.get("data", ""),
                "name": f"image_{i}.{ext}",
            },
        })
    return parts


class DocumentQueryServer(A2AServer):
    def __init__(self):
        super().__init__(agent_card=agent_card)

    def handle_task(self, task):
        content = (task.message or {}).get("content", {})
        conversation = content.get("text", "") if isinstance(content, dict) else ""
        trace_id = ""
        m = re.search(r'\[trace:([0-9a-f]+)\]', conversation)
        if m:
            trace_id = m.group(1)
        logger.info(f"[trace:{trace_id}] 对话历史及用户问题: {conversation}")

        # 提取最后一条 User 消息作为实际查询（避免把历史/追踪信息当作问题）
        user_matches = re.findall(r'User:\s*(.*)', conversation)
        query = next((q.strip() for q in reversed(user_matches) if q.strip()), conversation.strip())
        logger.info(f"[trace:{trace_id}] 提取到的文档问题: {query}")

        try:
            result = asyncio.run(query_documents(query))
            parts = _build_artifact_parts(result)
            task.artifacts = [{"parts": parts}]
            task.status = TaskStatus(state=TaskState.COMPLETED)
        except Exception as e:
            logger.error(f"[trace:{trace_id}] 文档问答失败: {str(e)}")
            task.status = TaskStatus(state=TaskState.FAILED,
                                     message={"role": "agent", "content": {"text": f"文档问答失败: {str(e)} 请重试。"}})
        return task


if __name__ == "__main__":
    server = DocumentQueryServer()
    print("\n=== 服务器信息 ===")
    print(f"名称: {server.agent_card.name}")
    print(f"描述: {server.agent_card.description}")
    for skill in server.agent_card.skills:
        print(f"- {skill.name}: {skill.description}")
    run_server(server, host="127.0.0.1", port=5023)
