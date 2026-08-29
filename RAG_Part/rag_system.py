# -*- coding: utf-8 -*-
"""
RAG 问答系统（简化版，仿 integrated_qa_system/rag_qa/core/new_rag_system.py）。

只保留「直接检索 → 组织上下文 → LLM 生成」这一条路径，
不引入问题分类、策略选择（HyDE/子查询/回溯）等老方案。

流程：
  1. 通过 VectorStore.hybrid_search_with_rerank 混合检索 + 重排，取 Top-N 父块；
  2. 将父块拼接为上下文，套用 rag_answer_prompt 模板；
  3. 调用 LLM（默认 Qwen，OpenAI 兼容接口）生成最终答案。
"""

import sys
import os
import time

# 兼容项目结构：确保能从本目录导入 config / prompt / vector_store
rag_dir = os.path.dirname(os.path.abspath(__file__))
if rag_dir not in sys.path:
    sys.path.insert(0, rag_dir)

from openai import OpenAI

from config import Config
from prompt import prompts
from vector_store import VectorStore

conf = Config()


def build_default_llm():
    """
    使用 RAG_Part 自带的 Qwen（OpenAI 兼容接口）构造 LLM 调用函数。
    返回 call_llm(prompt) -> str（同步，非流式）。
    """
    client = OpenAI(api_key=conf.qwen_api, base_url=conf.qwen_url)

    def call_llm(prompt):
        completion = client.chat.completions.create(
            model=conf.qwen_model_name,
            messages=[
                {"role": "system", "content": "你是一个有用的助手。"},
                {"role": "user", "content": prompt},
            ],
            timeout=60,
        )
        if completion.choices and completion.choices[0].message:
            return completion.choices[0].message.content
        return "错误: LLM返回无效响应"

    return call_llm


class RAGSystem:
    """封装 RAG 问答核心逻辑：检索相关文档，并基于检索上下文生成答案。"""

    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm
        self.rag_prompt = prompts.rag_answer_prompt()

    def retrieve(self, query, source_filter=None, k=None):
        """
        混合检索 + 重排，返回重排后的父块 Document 列表。
        k 默认取 conf.RETRIEVAL_K，最终只保留 conf.CANDIDATE_M 个作为上下文。
        """
        k = k or conf.RETRIEVAL_K
        docs = self.vector_store.hybrid_search_with_rerank(
            query, k=k, source_filter=source_filter
        )
        return docs[:conf.CANDIDATE_M]

    def generate_answer(self, query, source_filter=None, history=None):
        """
        生成答案：检索 → 拼接上下文 → 构造 Prompt → 调用 LLM。
        history 可选（对话历史字符串或 {"question","answer"} 列表），注入 Prompt 增强连贯性。
        """
        start_time = time.time()

        # 1. 检索相关文档
        docs = self.retrieve(query, source_filter=source_filter)
        if docs:
            context = "\n\n".join([doc.page_content for doc in docs])
            print(f"   已检索到 {len(docs)} 个相关文档块作为上下文")
        else:
            context = ""
            print("   未检索到相关文档，上下文为空")

        # 2. 组织对话历史（如有）
        history_text = ""
        if history:
            if isinstance(history, list):
                history_text = "\n".join(
                    [f"Q:{h.get('question', '')}\nA:{h.get('answer', '')}" for h in history[-5:]]
                )
            else:
                history_text = str(history)

        # 3. 构造 Prompt 并调用 LLM
        prompt_input = self.rag_prompt.format(context=context, question=query)
        if history_text:
            prompt_input = f"以下是与用户的部分历史对话，可作为参考：\n{history_text}\n\n" + prompt_input

        try:
            result = self.llm(prompt_input)
            # 兼容流式（生成器）与一次性字符串两种返回
            if hasattr(result, "__iter__") and not isinstance(result, str):
                answer = "".join(result)
            else:
                answer = result
        except Exception as e:
            print(f"   调用 LLM 生成答案失败: {e}")
            answer = "抱歉，生成答案时出错，请稍后重试。"

        processing_time = time.time() - start_time
        print(f"   问答处理耗时 {processing_time:.2f}秒")
        return answer


if __name__ == "__main__":
    print("初始化向量存储与 LLM（首次加载模型较慢）...")
    vs = VectorStore()
    llm = build_default_llm()
    rag_system = RAGSystem(vs, llm)
    answer = rag_system.generate_answer(query="漯河天纵城项目B2区消防工程的主要技术要求是什么？")
    print("-" * 40)
    print(f"问题: 漯河天纵城项目B2区消防工程的主要技术要求是什么？")
    print(f"回答:\n{answer}")
