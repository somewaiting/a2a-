# -*- coding: utf-8 -*-
"""
文件名: Parse.example.py
说明: MinerU 文档解析上传脚本示例。真实 Token 不入库，克隆后请复制本文件为 Parse.py 并填入 Token。
用法:  cp Parse.example.py Parse.py   然后编辑 Parse.py 填入 MinerU Token。
"""
import requests
import zipfile
import io
import time
import os
from pathlib import Path
from fastapi import FastAPI, Form, Request
from fastapi.responses import PlainTextResponse
import uvicorn
import threading
import hashlib
import json
# 启动 ngrok ： ngrok http 9000

# 通过MinerU的callback，拿到解析后的压缩包，进行压缩，得到完整的解析数据，保存在./parsed/（对应文件名）_output

# ================= 配置区 =================
# TODO: 填入你的 MinerU Token（https://mineru.net/ 获取）
token = "PASTE_YOUR_MINERU_TOKEN_HERE"
SEED = "abc123"
MY_UID = "824000145"  # 你的 MinerU UID

BASE_URL = "https://mineru.net/api/v4"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {token}"
}

# 用于在回调和主流程间传递数据的共享变量
callback_result = None
callback_event = threading.Event()
# =========================================

# --- 回调服务器相关代码 ---
app = FastAPI()


@app.post("/callback")
async def receive_callback(request: Request):
    global callback_result
    content_type = request.headers.get("content-type", "")
    if "json" in content_type:
        body = await request.json()
    else:
        body = await request.form()

    checksum = body.get("checksum", "")
    content = body.get("content", "")

    # 签名校验
    raw_string = MY_UID + SEED + content
    expected_checksum = hashlib.sha256(raw_string.encode("utf-8")).hexdigest()
    if checksum != expected_checksum:
        print("⚠️ 签名校验失败！")
        return PlainTextResponse("checksum mismatch", status_code=403)

    result = json.loads(content)
    print("\n" + "=" * 50)
    print("✅ 收到 MinerU 回调通知！")
    print(f"   Batch ID: {result.get('batch_id')}")

    # 提取第一个文件的结果
    if result.get("extract_result"):
        item = result["extract_result"][0]
        if item.get("state") == "done":
            callback_result = item.get("full_zip_url")
            print(f"   📥 解析成功，获取到下载链接")
            callback_event.set()  # 通知主流程
        elif item.get("state") == "failed":
            print(f"   ❌ 解析失败: {item.get('err_msg')}")
            callback_event.set()
    print("=" * 50 + "\n")
    return PlainTextResponse("success", status_code=200)


def run_server():
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="error")


# ----------------------------

def apply_upload_url(file_path, enable_ocr=True):
    """步骤1: 申请上传链接 - 增加 OCR 支持"""
    print(f"[1/5] 正在为文件 {os.path.basename(file_path)} 申请上传链接...")
    url = f"{BASE_URL}/file-urls/batch"

    # 判断文件类型
    file_ext = Path(file_path).suffix.lower()
    is_html = file_ext == '.html'

    # 对于图片文件，启用 OCR
    image_exts = ['.png', '.jpg', '.jpeg', '.jp2', '.webp', '.gif', '.bmp']
    is_image = file_ext in image_exts

    # 如果文件是图片或者需要提取图片内容，启用 OCR
    should_ocr = enable_ocr or is_image

    data = {
        "files": [{
            "name": os.path.basename(file_path),
            "data_id": "rag_doc_001",
            "is_ocr": should_ocr,  # 启用 OCR 识别图片和扫描件
        }],
        "model_version": "vlm" if not is_html else "MinerU-HTML",  # HTML文件使用专用模型
        "enable_formula": True,  # 开启公式识别
        "enable_table": True,  # 开启表格识别
        "language": "ch",  # 中文
        "extra_formats": ["html", "docx"]  # 导出 HTML 和 docx 格式
    }

    # 如果是 HTML 文件，移除不支持 HTML 导出的 extra_formats
    if is_html:
        data["extra_formats"] = ["docx"]  # HTML 文件不能导出为 HTML

    response = requests.post(url, headers=HEADERS, json=data)
    result = response.json()

    if result.get("code") != 0:
        raise Exception(f"申请上传链接失败: {result.get('msg')}")

    batch_id = result["data"]["batch_id"]
    upload_url = result["data"]["file_urls"][0]
    print(f"   ✅ 申请成功! Batch ID: {batch_id}")
    print(f"   📝 OCR 功能: {'启用' if should_ocr else '禁用'}")
    return batch_id, upload_url


def upload_file(file_path, upload_url):
    """步骤2: 上传文件"""
    print(f"[2/5] 正在上传文件到云端...")
    with open(file_path, 'rb') as f:
        res_upload = requests.put(upload_url, data=f)
    if res_upload.status_code != 200:
        raise Exception(f"文件上传失败，状态码: {res_upload.status_code}")
    print("   ✅ 文件上传成功，解析任务已自动提交！")


def download_and_extract(file_path, zip_url):
    """步骤3: 下载并解压ZIP文件，递归查找 full.html"""
    print(f"[3/5] 正在下载解析结果...")
    response = requests.get(zip_url)
    # 强制不走代理，直连下载
    response = requests.get(zip_url, proxies={"http": None, "https": None}, timeout=120)
    if response.status_code != 200:
        raise Exception("下载ZIP文件失败")

    print("   ✅ 下载完成，正在解压...")
    script_dir = Path(__file__).parent.absolute()
    parsed_root = script_dir / "parsed"
    parsed_root.mkdir(exist_ok=True)
    file_basename = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = parsed_root / f"{file_basename}_output"
    output_dir.mkdir(exist_ok=True)
    print(f"   📁 输出目录: {output_dir}")

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(output_dir)

    print(f"[4/5] 正在查找 full.html 文件...")
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file == "full.html":
                html_path = os.path.join(root, file)
                print(f"   ✅ 找到目标文件: {html_path}")
                return html_path
    print(f"   ⚠️ 未找到 full.html 文件")
    return None


def UploadAndGet(path):
    global callback_result, callback_event
    callback_result = None
    callback_event.clear()

    # 1. 启动回调服务器 (在后台线程)
    print("[0/5] 正在启动本地回调服务器...")
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(1)  # 等待服务器启动

    # 2. 申请链接并上传
    # 注意：这里使用 ngrok 的公网地址。你需要将下面的地址替换为你自己的。
    ngrok_url = "https://unbiased-dodgy-outbid.ngrok-free.dev"
    batch_id, upload_url = apply_upload_url(path, f"{ngrok_url}/callback")
    upload_file(path, upload_url)

    # 3. 等待回调
    print(f"[3/5] 已提交任务 (Batch ID: {batch_id})，正在等待 MinerU 回调...")
    callback_event.wait()  # 阻塞，直到回调函数调用 callback_event.set()

    if callback_result:
        print("   ✅ 成功从回调中获取结果链接")
        # 4. 下载并解压
        html_path = download_and_extract(path, callback_result)
        if html_path:
            print("\n✅ 全部流程执行完毕！")
        else:
            print("\n❌ 流程结束，但未能找到目标HTML文件。")
    else:
        print("\n❌ 流程结束，回调通知解析失败。")


# ================= 主程序 =================
if __name__ == "__main__":
    # 确保你已运行 ngrok http 9000，并替换下面的 URL
    UploadAndGet(r"example.pdf")
