# -*- coding: utf-8 -*-
"""
RAG 交互式问答入口（仿 integrated_qa_system/rag_qa/rag_main.py 的查询模式）。
用法：cd My_agent/RAG_Part && python rag_main.py
输入文档相关问题即可获得基于已入库文档的答案，输入 exit 退出。
"""

import sys
import os

rag_dir = os.path.dirname(os.path.abspath(__file__))
if rag_dir not in sys.path:
    sys.path.insert(0, rag_dir)

from vector_store import VectorStore
from rag_system import RAGSystem, build_default_llm


def main():
    print("正在初始化向量存储与语言模型（首次加载模型较慢）...")
    try:
        vector_store = VectorStore()
        llm = build_default_llm()
        rag_system = RAGSystem(vector_store, llm)
    except Exception as e:
        print(f"初始化失败，请检查 Milvus 连接与模型配置: {e}")
        return

    print("\n欢迎使用 RAG 文档问答系统！")
    print("输入您的问题，或输入 'exit' 退出。")
    while True:
        query = input("\n请输入您的问题: ").strip()
        if query.lower() in ("exit", "quit"):
            print("再见！")
            break
        if not query:
            continue
        try:
            print("正在检索并生成答案，请稍候...")
            answer = rag_system.generate_answer(query)
            print("-" * 40)
            print(f"问题: {query}")
            print(f"回答:\n{answer}")
            print("-" * 40)
        except Exception as e:
            print(f"处理查询失败: {e}")


if __name__ == "__main__":
    main()
