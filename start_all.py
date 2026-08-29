#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: start_all.py
作者: ZZS
项目: My_agent（基于A2A协议的企业签订进度智能查询/修改系统）
创建日期: 2026/8/26
描述: 一键启动脚本：先启动 4 个 MCP 服务器（含 RAG 文档问答 8023），再启动 4 个 A2A Agent 服务器（含文档问答 Agent 5023）。
      特性：
      - 就绪等待：启动后轮询探测全部端口，全部就绪再提示启动完成。
      - 进程守护自愈：监测子进程异常退出并自动重启（单进程连续重启超过阈值则跳过并告警）。
用法：在 New 目录下执行  python My_agent/start_all.py
"""
import os
import sys
import time
import subprocess
from pathlib import Path

# 将 My_agent 的父目录（New）加入 sys.path，保证直接运行脚本时也能导入 My_agent 包
_root = Path(__file__).resolve().parent
while _root.name != "My_agent" and _root.parent != _root:
    _root = _root.parent
if str(_root.parent) not in sys.path:
    sys.path.insert(0, str(_root.parent))

from My_agent.utils import health

# 修复 Windows GBK 控制台下打印 emoji/特殊字符导致的 UnicodeEncodeError 崩溃
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs" / "my_agent"
LOG_DIR.mkdir(parents=True, exist_ok=True)

HOST = "127.0.0.1"

# 服务启动清单：MCP 先，A2A 后（含端口）
MCP_SERVERS = [
    ("mcp_progress_server", "My_agent/mcp_server/mcp_progress_server.py", 8020),
    ("mcp_staff_server", "My_agent/mcp_server/mcp_staff_server.py", 8021),
    ("mcp_status_server", "My_agent/mcp_server/mcp_status_server.py", 8022),
    ("mcp_rag_server", "My_agent/mcp_server/mcp_rag_server.py", 8023),
]

A2A_AGENTS = [
    ("progress_query_server", "My_agent/a2a_server/progress_query_server.py", 5020),
    ("staff_query_server", "My_agent/a2a_server/staff_query_server.py", 5021),
    ("status_modify_server", "My_agent/a2a_server/status_modify_server.py", 5022),
    ("document_query_server", "My_agent/a2a_server/document_query_server.py", 5023),
]

ALL_SERVICES = MCP_SERVERS + A2A_AGENTS
ALL_PORTS = [p for _, _, p in ALL_SERVICES]

MAX_RESTART = 5   # 单进程连续重启上限
CHECK_INTERVAL = 3


def spawn(entry):
    """启动一个服务进程，返回 (name, script, port, proc, log_file, restart_count)。"""
    name, script, port = entry
    log_file = open(LOG_DIR / f"{name}.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, script],
        cwd=str(ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
    )
    print(f"    启动 {name} (端口 {port}) -> PID {proc.pid}")
    return {"name": name, "script": script, "port": port, "proc": proc, "log": log_file, "restart": 0}


def free_ports():
    """清理占用项目目标端口的残留 python 进程（上次异常退出留下的孤儿），保证重复启动干净。"""
    import subprocess as sp

    occupied = []
    out = sp.run(["netstat", "-ano", "-p", "tcp"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "listening" not in line.lower():
            continue
        for port in ALL_PORTS:
            if f":{port} " in line:
                pid = line.strip().split()[-1]
                if pid.isdigit() and int(pid) not in occupied:
                    occupied.append(int(pid))
    if not occupied:
        return
    print(f"发现 {len(occupied)} 个占用项目端口的残留进程，正在清理（仅限 python 进程）...")
    for pid in occupied:
        info = sp.run(["tasklist", "/fi", f"PID eq {pid}", "/fo", "csv", "/nh"],
                      capture_output=True, text=True).stdout.lower()
        if "python" in info:
            sp.run(["taskkill", "/pid", str(pid), "/f"],
                   capture_output=True, text=True)
            print(f"    已清理孤儿进程 PID {pid}")


def prewarm_langchain():
    """单进程预热 langchain：预编译字节码并触发一次 ChatOpenAI 构造，
    避免多个子进程并发首次导入触发 .pyc 竞争导致偶发的 ModuleNotFoundError / AttributeError。
    若 langchain 本身有问题，这里会立刻给出明确报错，便于定位。"""
    print("正在预热 langchain（单进程预编译字节码）...")
    try:
        from langchain_openai import ChatOpenAI
        from langchain.agents import create_tool_calling_agent, AgentExecutor
        from My_agent.config import Config as _C
        _c = _C()
        ChatOpenAI(model=_c.model_name, base_url=_c.base_url, api_key=_c.api_key,
                   temperature=0.1, timeout=30, max_retries=2)
        print("[OK] langchain 预热完成")
    except Exception as e:
        print(f"[WARN] langchain 预热失败（三个 A2A Agent 可能无法启动，请检查依赖）: {e}")


def main():
    print("=" * 60)
    print("My_agent 一键启动（就绪等待 + 进程守护）")
    print(f"MCP 服务器 {len(MCP_SERVERS)} 个 + A2A Agent {len(A2A_AGENTS)} 个（含 RAG 文档问答 8023）")
    print("=" * 60)

    # 0. 预热 langchain，避免子进程并发首次导入触发竞争
    prewarm_langchain()

    # 1. 清理残留进程并错峰启动全部服务
    #    （错峰避免多进程并发首次导入触发 .pyc 竞争，导致偶发的 ModuleNotFoundError / AttributeError）
    free_ports()
    running = []
    for e in ALL_SERVICES:
        running.append(spawn(e))
        time.sleep(1.5)

    # 2. 就绪等待
    print(f"\n正在等待全部服务就绪（{HOST}:{ALL_PORTS}）...")
    not_ready = health.wait_ports(HOST, ALL_PORTS, timeout=60, interval=1.5)
    if not_ready:
        print(f"[WARN] 以下服务未在 60s 内就绪：{[p for _, _, p in ALL_SERVICES if p in not_ready]}")
    else:
        print("[OK] 全部服务已就绪。")

    print("启动前端：streamlit run My_agent/app.py（或命令行：python My_agent/main.py），前端端口 9200")
    print("查看日志：logs/my_agent/*.log（Ctrl+C 结束全部服务）")

    # 3. 进程守护：监测退出并自动重启
    try:
        while True:
            for svc in running:
                proc = svc["proc"]
                if proc.poll() is None:
                    continue
                name, port = svc["name"], svc["port"]
                if svc["restart"] >= MAX_RESTART:
                    print(f"[WARN] [{name}] 连续重启已达上限({MAX_RESTART})，已停止自动重启，请人工排查日志。")
                    continue
                print(f"[WARN] [{name}] (端口 {port}) 异常退出，正在自动重启...")
                svc["log"].close()
                new = spawn((svc["name"], svc["script"], svc["port"]))
                new["restart"] = svc["restart"] + 1
                running[running.index(svc)] = new
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        print("\n收到中断，正在关闭全部服务...")
        for svc in running:
            try:
                svc["proc"].terminate()
                svc["log"].close()
            except Exception:
                pass
        print("已关闭。")


if __name__ == '__main__':
    main()
