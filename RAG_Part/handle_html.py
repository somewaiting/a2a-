import os
import json
from pathlib import Path
from bs4 import BeautifulSoup

# ================= 配置 =================
SCRIPT_DIR = Path(__file__).parent.resolve()
HTML_DIR = SCRIPT_DIR / "HTML"
OUTPUT_DIR = SCRIPT_DIR / "html_to_json"

# 被认为是"块级"的标签，用于判断嵌套和重复
BLOCK_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "div", "table", "img", "figure", "section", "article"}


# =======================================


def parse_html_file(html_path: Path):
    """
    按原文顺序解析 HTML，提取文本/表格/图片三种语义单元
    """
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    root = soup.body if soup.body else soup
    units = []
    heading_stack = []  # 当前标题层级，如 ["第一章", "1.1 背景"]

    # BeautifulSoup 的 find_all 按文档流前序遍历，天然保序
    for elem in root.find_all(BLOCK_TAGS, recursive=True):

        # --- 跳过被 table/figure 包裹的元素（如 td、tr、figcaption），避免拆碎 ---
        if elem.find_parent(["table", "figure"]):
            continue

        tag = elem.name

        # ==================== 标题：更新层级栈 ====================
        if tag.startswith("h"):
            level = int(tag[1])
            while len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(elem.get_text(strip=True))

            units.append({
                "type": "text",
                "subtype": "heading",
                "level": level,
                "content": elem.get_text(strip=True),
                "html": str(elem),
                "heading_path": " > ".join(heading_stack),
                "source_file": html_path.name
            })
            continue

        # ==================== 表格：完整保留 HTML ====================
        if tag == "table":
            caption = elem.find("caption")
            caption_text = caption.get_text(strip=True) if caption else ""

            # 生成文本摘要（用于后续 Embedding），保留前10行避免过长
            rows = elem.find_all("tr")[:10]
            text_summary = caption_text
            if rows:
                text_summary += "\n" + "\n".join(
                    " | ".join(td.get_text(strip=True) for td in row.find_all(["td", "th"]))
                    for row in rows
                )

            units.append({
                "type": "table",
                "caption": caption_text,
                "content": text_summary,  # 用于向量化的文本
                "full_html": str(elem),  # 完整原始 HTML，检索后直接渲染
                "heading_path": " > ".join(heading_stack),
                "source_file": html_path.name
            })
            continue

        # ==================== 图片 / Figure ====================
        if tag in ("img", "figure"):
            img = elem.find("img") if tag == "figure" else elem
            if not img:
                continue

            src = img.get("src", "")
            alt = img.get("alt", "")

            units.append({
                "type": "image",
                "src": src,  # 相对路径，前端可直接用
                "alt": alt,
                "content": alt,  # 初始为 alt，后续替换为多模态描述
                "html": str(elem),
                "heading_path": " > ".join(heading_stack),
                "source_file": html_path.name
            })
            continue

        # ==================== 文本块（p / div）====================
        if tag in ("p", "div"):
            # 关键：如果当前 div/p 内部还包含其他块级元素（如嵌套 div、table、img），
            # 则跳过，让子元素自己独立被提取。这样避免了 div 和内部 p 的文本重复。
            has_block_child = any(
                child.name in BLOCK_TAGS
                for child in elem.find_all(recursive=False)
            )
            if has_block_child:
                continue

            text = elem.get_text(separator="\n", strip=True)
            if not text:
                continue

            units.append({
                "type": "text",
                "subtype": "paragraph",
                "content": text,
                "html": str(elem),
                "heading_path": " > ".join(heading_stack),
                "source_file": html_path.name
            })
            continue

    return units


def main():
    # 检查输入文件夹
    if not HTML_DIR.exists():
        print(f"❌ 未找到输入文件夹: {HTML_DIR}")
        print("   请在脚本同级目录下创建 'HTML' 文件夹并放入 .html 文件")
        return

    # 创建输出文件夹
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 遍历 HTML 文件夹下的所有 .html 文件
    html_files = sorted(HTML_DIR.glob("*.html"))
    if not html_files:
        print(f"⚠️ {HTML_DIR} 下没有找到 .html 文件")
        return

    for html_file in html_files:
        print(f"📄 处理: {html_file.name}")

        # 解析
        units = parse_html_file(html_file)

        # 生成输出文件名：去掉 .html 后缀，加上 .json
        out_name = html_file.stem + ".json"
        out_path = OUTPUT_DIR / out_name

        # 保存为单独 JSON
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(units, f, ensure_ascii=False, indent=2)

        print(f"   ✅ 提取 {len(units)} 个语义单元 → {out_path}")

    print(f"\n🎉 全部处理完成，结果保存在: {OUTPUT_DIR}")
    print(f"   共处理 {len(html_files)} 个文件")


if __name__ == "__main__":
    main()