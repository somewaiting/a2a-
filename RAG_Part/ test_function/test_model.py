# -*- coding: utf-8 -*-
"""
测试 models 文件夹下的两个模型：
  1) BAAI/bge-m3            —— 嵌入模型
  2) BAAI/bge-reranker-v2-m3 —— 重排序模型

输出每个模型的：
  - 最大输入长度（max input length / max_position_embeddings）
  - 词嵌入维度（embedding dimension / hidden_size）
并做一次真实编码/推理验证维度正确。
"""
import os
import json
import sys
import warnings

warnings.filterwarnings("ignore")

# Windows 下统一输出为 UTF-8，避免重定向时乱码
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BGE_M3_DIR = os.path.join(BASE_DIR, "../models", "BAAI--bge-m3", "snapshots", "master")
RERANKER_DIR = os.path.join(BASE_DIR, "../models", "BAAI--bge-reranker-v2-m3", "snapshots", "master")


def load_config(model_dir):
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("1) 嵌入模型  BAAI/bge-m3")
    print("=" * 70)

    from pymilvus.model.hybrid import BGEM3EmbeddingFunction

    emb_fn = BGEM3EmbeddingFunction(
        model_name_or_path=BGE_M3_DIR,
        device="cpu",
        use_fp16=False,
    )

    cfg = load_config(BGE_M3_DIR)
    print(f"  模型架构        : {cfg.get('architectures')}")
    print(f"  hidden_size     : {cfg.get('hidden_size')}  (词嵌入维度, 配置声明)")
    print(f"  max_position_embeddings : {cfg.get('max_position_embeddings')}  (配置声明的最大位置编码)")
    print(f"  pymilvus dim    : {emb_fn.dim}")

    # 用真实文本编码验证
    texts = ["今天天气很好", "RAG 系统中的混合检索"]
    emb = emb_fn(texts)
    print(f"  实测 dense 向量维度 : {len(emb['dense'][0])}  (输入 {len(texts)} 条文本)")

    # 词元化最大输入长度（tokenizer 允许的最大长度）
    tokenizer = emb_fn.model.tokenizer
    print(f"  分词器 model_max_length : {tokenizer.model_max_length}")

    print()
    print("=" * 70)
    print("2) 重排序模型  BAAI/bge-reranker-v2-m3")
    print("=" * 70)

    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(RERANKER_DIR, device="cpu")

    r_cfg = load_config(RERANKER_DIR)
    print(f"  模型架构        : {r_cfg.get('architectures')}")
    print(f"  hidden_size     : {r_cfg.get('hidden_size')}  (词嵌入维度, 配置声明)")
    print(f"  max_position_embeddings : {r_cfg.get('max_position_embeddings')}  (配置声明的最大位置编码)")
    print(f"  实际最大输入长度      : {reranker.max_length}")

    # 实测一次推理
    pairs = [["什么是RAG", "RAG 是检索增强生成技术"], ["什么是RAG", "今天中午吃了什么"]]
    scores = reranker.predict(pairs)
    print(f"  实测推理得分    : {scores}")

    print()
    print("=" * 70)
    print("汇总：模型的最大输入长度与词嵌入维度")
    print("=" * 70)
    emb_len = tokenizer.model_max_length or cfg.get("max_position_embeddings", 0)
    rerank_len = reranker.max_length or r_cfg.get("max_position_embeddings", 0)
    print(f"  BGE-M3              -> 最大输入长度: {emb_len} tokens, 词嵌入维度: {emb_fn.dim.get('dense')}")
    print(f"  BGE-Reranker-V2-M3  -> 最大输入长度: {rerank_len} tokens, 词嵌入维度: {r_cfg.get('hidden_size')}")


if __name__ == "__main__":
    main()
