# -*- coding: utf-8 -*-
"""
RAG 文档构建器
功能：
  1. 读取 html_to_json/ 下的 JSON，严格保序
  2. 按 heading_path + 大小上限构建父块，内部按类型切分子块
  3. 图片调用 Qwen3.8-Max（带系统提示词）生成描述
  4. 子块用于 BGE-M3 向量检索，父块用于 bge-reranker-v2-m3 重排序与最终返回
  5. 输出兼容现有 VectorStore.add_documents() 的 Document 列表
"""

import os
import sys
import json
import base64
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from openai import OpenAI

# 兼容你的项目结构：优先从当前目录或上级 core 导入
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
        from langchain_core.documents import Document
    except ImportError:
        # 兜底：自定义最小 Document 结构
        class Document:
            def __init__(self, page_content: str, metadata: dict):
                self.page_content = page_content
                self.metadata = metadata

# ================= 路径常量 =================
SCRIPT_DIR = Path(__file__).parent.resolve()
PREPROCESS_DIR = SCRIPT_DIR / "html_to_json"          # 输入：解析后的 JSON
PARSED_DIR = SCRIPT_DIR / "parsed"                       # 图片根目录

# 可调参数统一收敛到 config.py（Config 类）：
#   PARENT_CHUNK_SIZE  父块大小上限（用于 bge-reranker 重排序与最终返回）
#   CHILD_CHUNK_SIZE   子块大小上限（用于 BGE-M3 嵌入检索）
#   CHUNK_OVERLAP      子块重叠大小
#   SOURCE_TAG         metadata["source"] 统一标识
from config import Config
conf = Config()

# ================= 提示词 =================
from prompt import prompts

# 初始化 Qwen 客户端
client = OpenAI(api_key=conf.qwen_api, base_url=conf.qwen_url)


# ================= 工具函数 =================

def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def md5_id(text: str, length: int = 16) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]


def resolve_image_path(json_stem: str, src: str) -> Optional[Path]:
    """
    解析图片绝对路径。
    规则：parsed / {json_stem}_output / images / {clean_src}
    """
    if not src:
        return None

    base = PARSED_DIR / f"{json_stem}_output" / "images"
    if not base.exists():
        return None

    # 统一正斜杠，去掉冗余的 images/ 前缀（因为 base 已经是 images 目录）
    clean = src.replace("\\", "/")
    for prefix in ("./images/", "images/", "./", "/"):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
            break

    # 尝试直接拼接
    target = base / clean
    if target.exists():
        return target.resolve()

    # 兜底：仅用文件名查找
    target2 = base / Path(clean).name
    if target2.exists():
        return target2.resolve()

    return None


def save_data_uri_image(src: str, json_stem: str) -> Optional[Tuple[Path, str]]:
    """
    将 data:image/xxx;base64,.... 内嵌图片解码落盘到
    parsed / {json_stem}_output / images / {sha256}.{ext}，
    返回 (绝对路径, 相对路径 src)。非 data URI 或解码失败返回 None。
    """
    if not src.startswith("data:"):
        return None
    try:
        header, _, b64 = src.partition(",")
        mime = header.split(";")[0]
        ext = mime.split("/")[-1].lower() if "/" in mime else "png"
        if ext not in ("png", "jpg", "jpeg", "gif", "webp", "bmp"):
            ext = "png"
        raw = base64.b64decode(b64)
    except Exception:
        return None

    img_dir = PARSED_DIR / f"{json_stem}_output" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{hashlib.sha256(raw).hexdigest()}.{ext}"
    target = img_dir / filename
    if not target.exists():
        target.write_bytes(raw)
    return target.resolve(), f"images/{filename}"


def describe_image_with_qwen(image_path: Path) -> str:
    """
    调用 Qwen3.8-Max 生成图片描述。
    系统提示词 + 用户指令，失败时返回空字符串。
    """
    if not image_path or not image_path.exists():
        return ""

    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        response = client.chat.completions.create(
            model=conf.qwen_model_name,
            messages=[
                {"role": "system", "content": prompts.image_system_prompt()},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompts.image_user_prompt(),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                },
            ],
            temperature=conf.QWEN_IMAGE_TEMPERATURE,
            max_tokens=conf.QWEN_IMAGE_MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"      ❌ Qwen 调用失败 [{image_path.name}]: {e}")
        return ""


def chunk_text_paragraphs(paragraphs: List[str], max_len: int, overlap: int = 0) -> List[str]:
    """
    按段落累积切分，不在段落中间切断。
    单个段落超长时，强制按 max_len 截断。
    overlap：相邻子块间的重叠字符数，切分时按段落回退，把上一块末尾内容带到下一块开头。
    """
    if not paragraphs:
        return []

    chunks = []
    current = []
    current_len = 0

    def flush():
        if current:
            chunks.append("\n".join(current))

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        p_len = len(para)

        # 段落本身超长：先 flush 当前累积，再强制截断该段落
        if p_len > max_len:
            flush()
            current, current_len = [], 0
            for i in range(0, p_len, max_len):
                chunks.append(para[i : i + max_len])
            continue

        # 正常累积：接近上限时先保存，保证段落完整
        if current_len + p_len > max_len and current:
            flush()
            if overlap > 0:
                # 重叠：从已 flush 的块末尾取不超过 overlap 字符的段落，作为新块开头
                carry, carry_len = [], 0
                for p in reversed(current):
                    if carry_len + len(p) > overlap and carry:
                        break
                    carry.insert(0, p)
                    carry_len += len(p)
                current, current_len = carry, carry_len
            else:
                current, current_len = [], 0

        current.append(para)
        current_len += p_len

    flush()
    return chunks


# ================= 核心：父子块构建 =================

class ParentBlock:
    """父块：有大小上限的语义聚合单元，适配 bge-reranker-v2-m3 的 8192 tokens"""
    def __init__(self, heading_path: str, source_file: str, parent_index: int = 0):
        self.heading_path = heading_path
        self.source_file = source_file
        self.parent_index = parent_index
        # parent_id 包含序号，确保切分后的多个父块不冲突
        self.parent_id = md5_id(f"{heading_path}::{source_file}::{parent_index}")
        self.units: List[Dict] = []
        self.char_count = 0

    def _estimate_unit_len(self, unit: Dict) -> int:
        """估算单元贡献的字符数"""
        u_type = unit.get("type")
        if u_type == "text":
            return len(unit.get("content", ""))
        elif u_type == "table":
            return len(unit.get("caption", "")) + len(unit.get("content", ""))
        elif u_type == "image":
            return len(unit.get("alt", "")) + len(unit.get("content", ""))
        return 0

    def try_add_unit(self, unit: Dict) -> bool:
        """
        尝试将单元加入当前父块。
        返回 True（成功）或 False（超出上限，需要新建父块）
        """
        add_len = self._estimate_unit_len(unit)

        # 空父块强制放入，避免死循环
        if not self.units:
            self.units.append(unit)
            self.char_count += add_len
            return True

        # 超出父块上限则拒绝
        if self.char_count + add_len > conf.PARENT_CHUNK_SIZE:
            return False

        self.units.append(unit)
        self.char_count += add_len
        return True

    @property
    def parent_content(self) -> str:
        """父块完整文本，用于 bge-reranker-v2-m3 重排序和返回给 LLM"""
        parts = []
        for u in self.units:
            t = u.get("type")
            if t == "text":
                parts.append(u.get("content", ""))
            elif t == "table":
                cap = u.get("caption", "")
                parts.append(f"[表格: {cap}]\n{u.get('content', '')}")
            elif t == "image":
                alt = u.get("alt", "")
                desc = u.get("description", "") or u.get("content", "")
                parts.append(f"[图片: {alt}]\n{desc}")
        return "\n\n".join(parts)

    @property
    def text_paragraphs(self) -> List[str]:
        """提取纯文本段落，用于子块切分"""
        return [
            u.get("content", "")
            for u in self.units
            if u.get("type") == "text" and u.get("subtype") == "paragraph"
        ]


def build_parent_blocks(units: List[Dict], heading_path: str, source_file: str) -> List[ParentBlock]:
    """
    将单元列表按 conf.PARENT_CHUNK_SIZE 切分为多个父块，保持自然边界（不在单元中间切断）
    """
    parents: List[ParentBlock] = []
    current = ParentBlock(heading_path, source_file, parent_index=0)

    for unit in units:
        if not current.try_add_unit(unit):
            # 当前父块已满，封存并新建
            parents.append(current)
            new_index = len(parents)
            current = ParentBlock(heading_path, source_file, parent_index=new_index)
            # 强制放入新父块（前面已保证空父块必成功）
            current.try_add_unit(unit)

    if current.units:
        parents.append(current)

    return parents


def build_documents_from_json(json_path: Path) -> List[Document]:
    """
    处理单个 JSON 文件，返回 Document 列表（子块），
    metadata 中已嵌入 parent_id / parent_content / category。
    """
    json_stem = json_path.stem
    with open(json_path, "r", encoding="utf-8") as f:
        units = json.load(f)

    # ---- 第一步：按 heading_path 分组 ----
    grouped: Dict[str, List[Dict]] = {}
    for u in units:
        hp = u.get("heading_path", "") or json_stem
        grouped.setdefault(hp, []).append(u)

    # ---- 第二步：每个 heading_path 内按 conf.PARENT_CHUNK_SIZE 切分父块 ----
    all_parents: List[ParentBlock] = []
    for hp, hp_units in grouped.items():
        parents = build_parent_blocks(hp_units, hp, json_path.name)
        all_parents.extend(parents)

    # ---- 第三步：按原文顺序遍历父块，生成子块 Document ----
    documents: List[Document] = []
    timestamp = get_timestamp()

    for pb in all_parents:
        parent_content = pb.parent_content
        text_paras = pb.text_paragraphs
        text_chunks = chunk_text_paragraphs(text_paras, conf.CHILD_CHUNK_SIZE, conf.CHUNK_OVERLAP)

        # --- 文本子块：每个切分块一个 Document，用于 BGE-M3 检索 ---
        for chunk_text in text_chunks:
            if chunk_text.strip():
                documents.append(Document(
                    page_content=chunk_text,
                    metadata={
                        "type": "text",
                        "category": "text",           # 类别标识
                        "heading_path": pb.heading_path,
                        "source": conf.SOURCE_TAG,
                        "parent_id": pb.parent_id,
                        "parent_content": parent_content,
                        "timestamp": timestamp,
                        "source_file": json_path.name,
                    }
                ))

        for u in pb.units:
            u_type = u.get("type")

            # --- 表格子块：完整保留 ---
            if u_type == "table":
                caption = u.get("caption", "")
                embed_text = caption
                if u.get("content"):
                    embed_text += "\n" + u["content"]

                documents.append(Document(
                    page_content=embed_text,
                    metadata={
                        "type": "table",
                        "category": "table",          # 类别标识
                        "heading_path": pb.heading_path,
                        "source": conf.SOURCE_TAG,
                        "parent_id": pb.parent_id,
                        "parent_content": parent_content,
                        "timestamp": timestamp,
                        "source_file": json_path.name,
                        "full_html": u.get("full_html", ""),      # 检索后前端直接渲染
                        "caption": caption,
                    }
                ))

            # --- 图片子块：调用 Qwen 生成描述 ---
            elif u_type == "image":
                src_raw = u.get("src", "")            # 原始值：可能是 data URI 或相对路径
                alt = u.get("alt", "")

                # 归一化：data URI 先解码落盘为哈希命名文件，得到相对路径 + 绝对路径
                src = src_raw
                img_path = None
                if src.startswith("data:"):
                    saved = save_data_uri_image(src, json_stem)
                    if saved:
                        img_path, src = saved         # (绝对路径, images/xxx.ext)
                    else:
                        src = ""                      # 解码失败，标记无有效来源
                else:
                    img_path = resolve_image_path(json_stem, src)

                # 调用多模态模型（仅在首次处理时调用，可缓存）
                description = u.get("description", "")
                if not description and img_path and img_path.exists():
                    print(f"      🔍 识别图片: {src}")
                    description = describe_image_with_qwen(img_path)
                    u["description"] = description  # 缓存回写
                    if description:
                        print(f"      ✅ 描述已生成 ({len(description)} 字)")
                    else:
                        description = alt or "图片暂无描述"
                elif not description:
                    print(f"      ⚠️ 图片无有效来源: {src_raw[:60]}...")
                    description = alt or "图片暂无描述"

                # 用于 Embedding 的文本
                searchable = f"{alt}\n图片描述: {description}" if alt else f"图片描述: {description}"

                documents.append(Document(
                    page_content=searchable,
                    metadata={
                        "type": "image",
                        "category": "image",          # 类别标识
                        "heading_path": pb.heading_path,
                        "source": conf.SOURCE_TAG,
                        "parent_id": pb.parent_id,
                        "parent_content": parent_content,
                        "timestamp": timestamp,
                        "source_file": json_path.name,
                        "src": src,                               # 相对路径，前端 <img src>
                        "abs_path": str(img_path) if img_path else "",
                        "alt": alt,
                        "description": description,               # 展示给 LLM/用户
                    }
                ))

    return documents


# ================= Document 持久化 =================

def save_documents(documents: List[Document], path: Path) -> None:
    """将构建好的 Document 列表序列化保存到本地 JSON，
    入库失败或后续调整切分/检索参数时可复用，避免重复调用 Qwen 等大模型重建。"""
    payload = [
        {"page_content": d.page_content, "metadata": d.metadata}
        for d in documents
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_documents(path: Path) -> List[Document]:
    """从持久化 JSON 恢复 Document 列表（与 save_documents 格式对应）"""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return [
        Document(page_content=item["page_content"], metadata=item.get("metadata", {}))
        for item in payload
    ]


# ================= 主程序 =================

def main(from_cache: bool = False):
    cache_path = Path(conf.DOCUMENTS_CACHE_PATH)

    # ---- 构建或复用 ----
    if from_cache and cache_path.exists():
        print(f"📂 从缓存加载 Document: {cache_path}")
        all_documents = load_documents(cache_path)
        print(f"   ✅ 已加载 {len(all_documents)} 个子块（跳过重建，不调用大模型）")
    else:
        if not PREPROCESS_DIR.exists():
            print(f"❌ 未找到输入目录: {PREPROCESS_DIR}")
            return

        PREPROCESS_DIR.mkdir(parents=True, exist_ok=True)

        json_files = sorted(PREPROCESS_DIR.glob("*.json"))
        if not json_files:
            print(f"⚠️ {PREPROCESS_DIR} 下没有 .json 文件")
            return

        all_documents: List[Document] = []

        for jf in json_files:
            print(f"\n📄 处理: {jf.name}")
            docs = build_documents_from_json(jf)
            all_documents.extend(docs)

            # 统计
            type_counts = {}
            for d in docs:
                t = d.metadata.get("category", "unknown")
                type_counts[t] = type_counts.get(t, 0) + 1
            print(f"   ✅ 生成 {len(docs)} 个子块: {type_counts}")

        print(f"\n🎉 全部完成！共 {len(json_files)} 个文件，{len(all_documents)} 个子块")

        # 始终保存构建结果，入库失败后可 --from-cache 复用，避免重复调用 Qwen
        save_documents(all_documents, cache_path)
        print(f"💾 已保存 Document 缓存: {cache_path}")

    if not all_documents:
        print("⚠️ 没有 Document 可入库")
        return

    # ---- 入库：通过 VectorStore 存入 Milvus ----
    try:
        sys.path.insert(0, str(SCRIPT_DIR))
        from vector_store import VectorStore

        print("\n🔧 开始写入 Milvus...")
        vs = VectorStore()
        vs.add_documents(all_documents)
        print("✅ 入库完成")
    except Exception as e:
        print(f"\n⚠️ 入库失败: {e}")
        print(f"   💡 修复问题后可执行: python parse_json.py --from-cache  直接复用缓存重试入库，无需重新生成")


# def debug_single_file():
#     """
#     单文件调试入口：检查图片路径解析与 Document 生成。
#     用法：from parse_json import debug_single_file; debug_single_file()
#     """
#     global PREPROCESS_DIR, PARSED_DIR
#     # ================= 单文件调试调用 =================
#     test_json = Path(
#         r"C:\Users\28449\Desktop\new_project\RAG_Part\html_to_json\(2023.09.20）漯河天纵资信投标文件docx.json")
#
#     PREPROCESS_DIR = test_json.parent
#     PARSED_DIR = Path(r"C:\Users\28449\Desktop\new_project\RAG_Part\parsed")
#
#     print(f"🔧 PARSED_DIR 设置为: {PARSED_DIR}")
#     print(f"🔧 PARSED_DIR 是否存在: {PARSED_DIR.exists()}")
#     print("=" * 60)
#
#     # 先手动测试图片路径解析
#     json_stem = test_json.stem
#     print(f"🔧 json_stem (文件名前缀): {json_stem}")
#
#     expected_base = PARSED_DIR / f"{json_stem}_output" / "images"
#     print(f"🔧 期望的图片文件夹: {expected_base}")
#     print(f"🔧 期望文件夹是否存在: {expected_base.exists()}")
#
#     # 列出 parsed 目录下实际有什么
#     if PARSED_DIR.exists():
#         print(f"\n📂 {PARSED_DIR} 下的实际内容:")
#         for item in PARSED_DIR.iterdir():
#             print(f"   {item.name} {'(dir)' if item.is_dir() else '(file)'}")
#     else:
#         print(f"❌ {PARSED_DIR} 不存在！")
#
#     print("\n" + "=" * 60)
#
#     if not test_json.exists():
#         print(f"❌ JSON 文件不存在: {test_json}")
#         return
#
#     # 读取 JSON 看看图片 src 是什么
#     with open(test_json, "r", encoding="utf-8") as f:
#         units = json.load(f)
#
#     img_units = [u for u in units if u.get("type") == "image"]
#     print(f"📊 JSON 中共有 {len(img_units)} 个图片单元")
#     for u in img_units[:3]:
#         src = u.get("src", "")
#         if src.startswith("data:"):
#             print(f"\n   JSON 中的 src: <data URI，长度 {len(src)}>")
#             saved = save_data_uri_image(src, json_stem)
#             print(f"   落盘结果: {saved[0] if saved else 'None'}")
#             print(f"   相对路径: {saved[1] if saved else 'None'}")
#             if saved:
#                 print(f"   是否存在: {saved[0].exists()}")
#         else:
#             print(f"\n   JSON 中的 src: {src}")
#             resolved = resolve_image_path(json_stem, src)
#             print(f"   解析结果: {resolved}")
#             if resolved:
#                 print(f"   是否存在: {resolved.exists()}")
#
#     print("\n" + "=" * 60)
#     print("开始生成 Documents...")
#
#     documents = build_documents_from_json(test_json)
#
#     print(f"\n✅ 共生成 {len(documents)} 个 Document")
#
#     # 检查图片类型的结果
#     img_docs = [d for d in documents if d.metadata.get("type") == "image"]
#     print(f"📊 其中图片类型: {len(img_docs)} 个")
#     for i, doc in enumerate(img_docs[:3]):
#         print(f"\n--- Image Document [{i}] ---")
#         print(f"src      : {doc.metadata.get('src')}")
#         print(f"abs_path : {doc.metadata.get('abs_path')}")
#         print(f"desc     : {doc.metadata.get('description', 'N/A')[:100]}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG 文档构建与 Milvus 入库")
    parser.add_argument("--from-cache", action="store_true",
                        help="直接从缓存加载已构建的 Document 入库，跳过重建（不调用 Qwen 等大模型）")
    args = parser.parse_args()
    main(from_cache=args.from_cache)