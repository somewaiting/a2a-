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
import base64
from pathlib import Path

# 兼容项目结构：确保能从本目录导入 config / prompt / vector_store
rag_dir = os.path.dirname(os.path.abspath(__file__))
if rag_dir not in sys.path:
    sys.path.insert(0, rag_dir)

from openai import OpenAI

from config import Config
from prompt import prompts
from vector_store import VectorStore

try:
    from parse_json import resolve_image_path
except Exception:
    resolve_image_path = None

conf = Config()


def _mime_for_path(path) -> str:
    """根据文件扩展名推断 MIME 类型（用于前端展示与多模态请求）"""
    ext = Path(str(path)).suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(ext, "image/png")


def _resolve_image_abs_path(src: str, source_file: str) -> str | None:
    """
    在回答阶段重新解析图片的绝对路径。

    入库时 metadata["abs_path"] 记录的是当时机器的绝对路径（可能已失效），
    而 src 是相对路径（如 images/xxx.jpeg），结合 source_file（json 文件名）
    可在当前项目 parsed/<stem>_output/images/ 下重新定位。
    """
    if not src or not source_file:
        return None
    stem = source_file[:-5] if source_file.endswith(".json") else source_file
    if resolve_image_path is not None:
        try:
            path = resolve_image_path(stem, src)
            if path and path.exists():
                return str(path)
        except Exception:
            pass
    # 兜底：手动拼接
    base = Path(rag_dir) / "parsed" / f"{stem}_output" / "images"
    if not base.exists():
        return None
    clean = src.replace("\\", "/")
    for prefix in ("./images/", "images/", "./", "/"):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
            break
    target = base / clean
    if target.exists():
        return str(target.resolve())
    target2 = base / Path(clean).name
    if target2.exists():
        return str(target2.resolve())
    return None


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

    def _build_history_text(self, history=None) -> str:
        """把对话历史转成字符串（history 可为字符串或 {"question","answer"} 列表）"""
        if not history:
            return ""
        if isinstance(history, list):
            return "\n".join(
                [f"Q:{h.get('question', '')}\nA:{h.get('answer', '')}" for h in history[-5:]]
            )
        return str(history)

    def _rank_images(self, query: str, images: list) -> list:
        """
        按图片 alt/描述与查询的相关性（BGE-M3 稠密向量余弦相似度）降序排序，
        让多模态读取与返回给前端的图片优先是真正相关的那几张。
        """
        if len(images) <= 1 or not query:
            return images
        try:
            desc_texts = [
                (f"{img['alt']}\n{img['description']}").strip() or img["abs_path"]
                for img in images
            ]
            embs = self.vector_store.embedding_function(desc_texts)["dense"]
            q_emb = self.vector_store.embedding_function([query])["dense"][0]
            import numpy as np
            q = np.asarray(q_emb, dtype=np.float32)
            q_norm = float(np.linalg.norm(q)) or 1.0
            scores = []
            for e in embs:
                v = np.asarray(e, dtype=np.float32)
                v_norm = float(np.linalg.norm(v)) or 1.0
                scores.append(float(np.dot(q, v) / (q_norm * v_norm)))
            ranked = [img for _, img in sorted(zip(scores, images), key=lambda x: x[0], reverse=True)]
            return ranked
        except Exception as e:
            print(f"   ⚠️ 图片相关性排序失败（保持原顺序）: {e}")
            return images

    def _collect_images(self, docs, query: str = "") -> list:
        """
        从检索到的父文档中收集图片（含同父块内的兄弟图片），按绝对路径去重，
        按与查询的相关性排序后只保留前 N 张，再读取 base64。
        返回 [{alt, description, src, abs_path, mime, data(base64)}]
        """
        # 先收集元数据（不读文件），避免为几十张无关兄弟图都读 base64
        candidates = []
        seen = set()

        # 同父块的兄弟 image 子块：命中文本子块时也能找回父块里的图片
        parent_ids = [d.metadata.get("parent_id") for d in docs if d.metadata.get("parent_id")]
        sibling = []
        if parent_ids:
            try:
                sibling = self.vector_store.get_images_by_parents(parent_ids)
            except Exception as e:
                print(f"   ⚠️ 回查兄弟图片失败: {e}")

        # 直接命中的 image 文档 + 兄弟图片
        for doc in [d for d in docs if d.metadata.get("type") == "image"] + sibling:
            src = doc.metadata.get("src", "")
            source_file = doc.metadata.get("source_file", "")
            abs_path = _resolve_image_abs_path(src, source_file)
            if not abs_path or not Path(abs_path).exists():
                continue
            key = abs_path
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "alt": doc.metadata.get("alt") or "",
                "description": doc.metadata.get("description") or "",
                "src": src,
                "abs_path": abs_path,
                "mime": _mime_for_path(abs_path),
                "data": "",  # 占位，排序后只对保留的图片读取
            })

        # 按与查询的相关性排序，只保留前 N 张，再读取 base64
        limit = getattr(conf, "ANSWER_IMAGE_LIMIT", 1)
        ranked = self._rank_images(query, candidates)[:limit]
        for img in ranked:
            try:
                raw = Path(img["abs_path"]).read_bytes()
            except Exception:
                continue
            if not raw or len(raw) > getattr(conf, "IMAGE_MAX_BYTES", 6 * 1024 * 1024):
                continue
            img["data"] = base64.b64encode(raw).decode("ascii")
        return [img for img in ranked if img["data"]]

    def _answer_with_images(self, query: str, context: str, images: list, fallback_prompt: str) -> str:
        """
        多模态回答：把真实图片（base64）连同上下文与问题一起交给 Qwen-VL，
        让模型真正读取图片并转录文字。失败时回退到纯文本生成。
        """
        client = OpenAI(api_key=conf.qwen_api, base_url=conf.qwen_url)
        system_prompt = prompts.rag_answer_multimodal_system_prompt()
        user_text = (
            f"**文档上下文（含图片文字描述）**：\n{context}\n\n"
            f"**用户问题**：\n{query}\n\n"
            f"请结合上下文与所附图片回答。若问题涉及图片中的文字、数字、表格、盖章、"
            f"签名或日期等内容，请直接转录完整内容。\n\n**回答**："
        )
        content = [{"type": "text", "text": user_text}]
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{img['mime']};base64,{img['data']}"},
            })
        try:
            completion = client.chat.completions.create(
                model=conf.qwen_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=conf.QWEN_IMAGE_TEMPERATURE,
                max_tokens=conf.QWEN_IMAGE_MAX_TOKENS,
                timeout=90,
            )
            if completion.choices and completion.choices[0].message:
                return completion.choices[0].message.content
            return "错误: LLM返回无效响应"
        except Exception as e:
            print(f"   多模态生成失败，回退纯文本: {e}")
            return self.llm(fallback_prompt)

    def generate_answer(self, query, source_filter=None, history=None):
        """生成答案（纯文本返回，兼容旧入口）。图片相关查询建议使用 generate_answer_with_images。"""
        return self.generate_answer_with_images(query, source_filter, history)["answer"]

    def generate_answer_with_images(self, query, source_filter=None, history=None):
        """
        生成答案并返回检索到的图片，供前端展示/多模态读取。
        返回 {"answer": str, "images": [{alt, description, src, abs_path, mime, data(base64)}]}
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

        # 2. 收集图片（按相关性排序，用于多模态读取 + 前端展示）
        images = self._collect_images(docs, query)
        if images:
            print(f"   关联到 {len(images)} 张相关图片，将交给多模态模型读取并随答案返回")
            # 父块中图片占位（[图片: ]）可能为空描述，把图片的文字描述补进上下文，
            # 保证纯文本路径也能拿到图中内容（如营业执照的经营范围等）。
            # vector_store 已把直接命中的图片描述补进父块，这里只补充尚未包含的，避免重复。
            desc_section = "\n\n".join(
                f"[图片: {img['alt'] or '无标题'}]\n{img['description']}"
                for img in images
                if img.get("description") and img["description"] not in context
            )
            if desc_section:
                context = context + "\n\n===== 检索到的相关图片内容（请一并参考）=====\n" + desc_section

        # 3. 组织对话历史（如有）
        history_text = self._build_history_text(history)

        # 4. 构造 Prompt 并调用 LLM
        prompt_input = self.rag_prompt.format(context=context, question=query)
        if history_text:
            prompt_input = f"以下是与用户的部分历史对话，可作为参考：\n{history_text}\n\n" + prompt_input

        if images:
            answer = self._answer_with_images(query, context, images, prompt_input)
        else:
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
        return {"answer": answer, "images": images}


if __name__ == "__main__":
    print("初始化向量存储与 LLM（首次加载模型较慢）...")
    vs = VectorStore()
    llm = build_default_llm()
    rag_system = RAGSystem(vs, llm)
    answer = rag_system.generate_answer(query="漯河天纵城项目B2区消防工程的主要技术要求是什么？")
    print("-" * 40)
    print(f"问题: 漯河天纵城项目B2区消防工程的主要技术要求是什么？")
    print(f"回答:\n{answer}")
