import os
import shutil
from pathlib import Path
from Parse import UploadAndGet

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# data 文件夹路径
data_dir = os.path.join(SCRIPT_DIR, "data")

# 读取data文件夹，将文件夹下的路径传入UploadAndGet中

def UploadData():
    # 检查 data 文件夹是否存在
    if not os.path.exists(data_dir):
        print(f"❌ 未找到 data 文件夹: {data_dir}")
        print("   请在脚本同级目录下创建 'data' 文件夹并放入待处理的文件。")
        exit(1)

    # 遍历 data 文件夹下的所有文件
    files = sorted(os.listdir(data_dir))
    if not files:
        print("⚠️ data 文件夹为空，没有需要处理的文件。")
        exit(0)

    print(f"📁 发现 {len(files)} 个文件/项目，开始批量处理...\n")

    for item in files:
        item_path = os.path.join(data_dir, item)

        # 只处理文件（跳过子文件夹）
        if os.path.isfile(item_path):
            # 计算相对于脚本所在文件夹的相对路径
            rel_path = os.path.relpath(item_path, SCRIPT_DIR)
            # 统一使用正斜杠
            rel_path = rel_path.replace(os.sep, "/")

            # 依次传入 UploadAndGet
            UploadAndGet(rel_path)

    print(f"\n{'='*60}")
    print("🎉 所有文件处理完毕！")
    print(f"{'='*60}")

def collect_full_html():
    # 获取脚本所在文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = current_dir + r'\parsed'

    # 目标保存目录：脚本所在文件夹下的 HTML 文件夹
    html_dir = os.path.join(current_dir, "HTML")
    os.makedirs(html_dir, exist_ok=True)

    print(f"📁 脚本所在目录: {script_dir}")
    print(f"📂 目标保存目录: {html_dir}")
    print("-" * 50)

    collected_count = 0

    # 遍历脚本所在文件夹下的所有子文件夹
    for item in os.listdir(script_dir):
        item_path = os.path.join(script_dir, item)

        # 只处理后缀为 "_output" 的文件夹
        if os.path.isdir(item_path) and item.endswith("_output"):
            print(f"🔍 正在搜索: {item}/")

            # 在该文件夹下递归查找 full.html
            for root, dirs, files in os.walk(item_path):
                if "full.html" in files:
                    source_path = os.path.join(root, "full.html")

                    # 生成新文件名，避免冲突：使用 _output 文件夹名作为前缀
                    # 例如：443-548.pdf_output → 443-548.pdf_full.html
                    new_name = item.replace("_output", "") + ".html"
                    target_path = os.path.join(html_dir, new_name)

                    # 如果文件名已存在，添加序号
                    counter = 1
                    base_name = new_name.replace(".html", "")
                    while os.path.exists(target_path):
                        new_name = f"{base_name}_{counter}.html"
                        target_path = os.path.join(html_dir, new_name)
                        counter += 1

                    shutil.copy2(source_path, target_path)
                    print(f"   ✅ 已复制: {source_path}")
                    print(f"      → {target_path}")
                    collected_count += 1
                    break  # 每个 _output 文件夹只取第一个 full.html

    print("-" * 50)
    if collected_count > 0:
        print(f"🎉 共收集 {collected_count} 个 full.html 文件到 HTML 文件夹")
    else:
        print("⚠️ 未找到任何 full.html 文件")


if __name__ == "__main__":
    UploadData()
    collect_full_html()