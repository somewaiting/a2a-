# -*- coding: utf-8 -*-
"""
文件名: config.example.py
说明: RAG_Part 配置示例。真实密钥不入库，克隆后请复制本文件为 config.py 并填入密钥。
用法:  cp config.example.py config.py   然后编辑 config.py 填入 API Key。
"""
import os


class Config():
    def __init__(self):
        rag_dir = os.path.dirname(os.path.abspath(__file__))

        # ============ Qwen 大模型配置（DashScope 兼容接口）============
        # TODO: 填入你的 DashScope / 通义千问 API Key（https://bailian.console.aliyun.com/ 获取）
        self.qwen_api = "PASTE_YOUR_DASHSCOPE_API_KEY_HERE"
        self.qwen_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.qwen_model_name = "qwen3.8-max"

        # ============ MILVUS 配置 ============
        self.milvus_host = 'localhost'
        self.milvus_port = 19530
        self.milvus_database_name = 'My_agent'
        self.milvus_collection_name = 'my_rag'

        # 检索返回数量
        self.RETRIEVAL_K = 5
        # 最终候选数量
        self.CANDIDATE_M = 2

        # 文档切分参数（parse_json.py 使用）
        self.PARENT_CHUNK_SIZE = 4096   # 父块大小上限（字符），用于 bge-reranker 重排序与最终返回
        self.CHILD_CHUNK_SIZE = 2048    # 子块大小上限（字符），用于 BGE-M3 嵌入检索
        self.CHUNK_OVERLAP = 200        # 子块之间重叠大小（字符）

        # 入库数据源标识（metadata["source"]）
        self.SOURCE_TAG = "html_rag"

        # Qwen 图片描述参数
        self.QWEN_IMAGE_TEMPERATURE = 0.1
        self.QWEN_IMAGE_MAX_TOKENS = 2048

        # 多模态问答/前端展示的图片数量与大小上限（避免 token 与响应体积过大）
        self.ANSWER_IMAGE_LIMIT = 1       # 单次问答最多附带/返回的图片数（只返回最相关的一张）
        self.IMAGE_MAX_BYTES = 6 * 1024 * 1024  # 单张图片超过该大小（字节）则跳过

        # 本地模型路径（HuggingFace 缓存快照目录）
        self.bge_m3_model_path = os.path.join(rag_dir, "models", "BAAI--bge-m3", "snapshots", "master")
        self.bge_reranker_model_path = os.path.join(rag_dir, "models", "BAAI--bge-reranker-v2-m3", "snapshots", "master")

        # 构建结果缓存路径：保存切分后的 Document，
        # 入库失败或后续调整参数时可复用，避免重复调用 Qwen 等大模型重建
        self.DOCUMENTS_CACHE_PATH = os.path.join(rag_dir, "RagAble_data", "RagAble_data/rag_documents_cache.json")
