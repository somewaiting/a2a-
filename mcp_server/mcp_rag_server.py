#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: mcp_rag_server.py
作者: ZZS
项目: My_agent（基于A2A协议的企业签订进度智能查询/修改系统）
创建日期: 2026/8/26
描述: RAG 知识检索 MCP 服务器（RagTools，端口 8023）
      工具：search_documents —— 基于已入库文档（Milvus my_rag）进行问答。
      内部复用 My_agent/RAG_Part 的 RAGSystem（混合检索 + 重排 + Qwen 生成答案）。
"""
import sys
import json
from pathlib import Path

# 将 My_agent 的父目录（New）加入 sys.path
_root = Path(__file__).resolve().parent
while _root.name != "My_agent" and _root.parent != _root:
    _root = _root.parent
if str(_root.parent) not in sys.path:
    sys.path.insert(0, str(_root.parent))

# 将 RAG_Part 目录加入 sys.path，复用其 RAGSystem / VectorStore
RAG_PART_DIR = Path(__file__).resolve().parent.parent / "RAG_Part"
if str(RAG_PART_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_PART_DIR))

from mcp.server.fastmcp import FastMCP

from My_agent.config import Config
from My_agent.create_logger import logger
from rag_system import RAGSystem, build_default_llm
from vector_store import VectorStore

conf = Config()

# 懒加载单例：首次调用时构建 RAG 系统（加载 BGE-M3 与重排模型耗时较长）。
# 注意：必须在服务器启动（rag_mcp.run）前预加载，
# 否则首次工具调用时长时间阻塞会断开 A2A/MCP 流式连接，导致任务重发/失败。
_rag_system = None


def _get_rag_system() -> RAGSystem:
    global _rag_system
    if _rag_system is None:
        logger.info("初始化 RAG 系统（加载向量模型与重排模型）...")
        _rag_system = RAGSystem(VectorStore(), build_default_llm())
        logger.info("RAG 系统初始化完成")
    return _rag_system


def create_rag_mcp_server():
    rag_mcp = FastMCP(name="RagTools",
                      instructions="RAG 知识检索工具：基于已入库的企业文档（方案/合同/招标文件等）进行检索问答。",
                      log_level="ERROR",
                      host="127.0.0.1", port=conf.mcp_ports["rag"])

    @rag_mcp.tool(
        name="search_documents",
        description=(
            "对已入库的企业文档进行问答，返回基于检索内容的回答，输入为用户的文档相关问题。"
            "返回 JSON：{\"answer\": 答案文本, \"images\": [{alt, description, src, abs_path, mime, data(base64)}]}，"
            "当问题涉及图片/图表时，images 会携带对应的完整图片（base64），供前端展示。"
        )
    )
    def search_documents(query: str) -> str:
        logger.info(f"RAG 文档检索请求: {query}")
        try:
            rag_system = _get_rag_system()
            result = rag_system.generate_answer_with_images(query)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"RAG 文档检索失败: {e}")
            return json.dumps({"answer": f"RAG 文档检索失败：{e}", "images": []}, ensure_ascii=False)

    logger.info("=== RAG MCP服务器信息 ===")
    logger.info(f"名称: {rag_mcp.name} / 端口: {conf.mcp_ports['rag']}")

    # 启动前预加载 RAG 系统（首次加载模型约需 10~30 秒），保证工具调用时响应快速、连接稳定
    try:
        _get_rag_system()
        print("RAG 系统已预加载完成（模型就绪）")
    except Exception as e:
        logger.error(f"RAG 系统预加载失败: {e}")
        print(f"RAG 系统预加载失败: {e}")

    try:
        print("服务器已启动，请访问 http://127.0.0.1:8023/mcp （RAG 文档问答）")
        rag_mcp.run(transport="streamable-http")
    except Exception as e:
        print(f"服务器启动失败: {e}")


if __name__ == '__main__':
    create_rag_mcp_server()
