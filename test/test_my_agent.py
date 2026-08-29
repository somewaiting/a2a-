#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: test_my_agent.py
作者: ZZS
项目: My_agent（基于A2A协议的企业签订进度智能查询/修改系统）
创建日期: 2026/8/26
描述: 端到端测试：模糊查询、无结果提示、禁 DELETE、职员查询、状态修改与驳回。
      前置：已启动 4 个 MCP（8020-8023）与 4 个 Agent（5020-5023）。
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import asyncio
import uuid
import json
from python_a2a import A2AClient, Message, TextContent, MessageRole, Task

from My_agent.query_data.my_agent_service import MyAgentService

PASS = 0
FAIL = 0


def call_agent(url: str, query: str) -> str:
    # timeout=120：RAG 文档问答生成较慢（约 40s），默认 30s 客户端超时会导致任务失败
    client = A2AClient(url, timeout=120)
    message = Message(content=TextContent(text=query), role=MessageRole.USER)
    task = Task(id="task-" + str(uuid.uuid4()), message=message.to_dict())
    result = asyncio.run(client.send_task_async(task))
    if result.status.state == 'completed':
        return result.artifacts[0]['parts'][0]['text']
    return f"[{result.status.state}] " + result.status.message['content']['text']


def short(s: str, n: int = 180) -> str:
    return s if len(s) <= n else s[:n] + "..."


def check(name, cond, detail):
    global PASS, FAIL
    mark = "PASS" if cond else "FAIL"
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f"[{mark}] {name}: {detail}")


def main():
    print("=" * 70)
    print("My_agent 端到端测试")
    print("=" * 70)

    svc = MyAgentService()

    # ===== 服务层测试（确定性） =====
    print("\n[0] 服务层：SQL 安全校验")
    r = json.loads(svc.execute_query("DELETE FROM sign_progress WHERE company_name='西湖'"))
    check("非 SELECT 语句被拒绝", r.get("status") == "rejected" and "请联系相关负责人处理" in r.get("message", ""), r.get("message"))
    r2 = json.loads(svc.execute_query("SELECT * FROM sign_progress WHERE 1=1; DROP TABLE sign_progress"))
    check("含危险关键字被拒绝", r2.get("status") == "rejected", r2.get("message"))
    r3 = json.loads(svc.execute_query("SELECT * FROM sign_progress"))
    check("合法 SELECT 可执行", r3.get("status") == "success" and len(r3.get("data", [])) == 6, f"返回 {len(r3.get('data', []))} 条")

    # SQL 强白名单：多语句 / UNION / 注释 / 非白名单表 / 非法列
    r_multi = json.loads(svc.execute_query("SELECT * FROM sign_progress; SELECT * FROM employees"))
    check("多语句被拒绝", r_multi.get("status") == "rejected", r_multi.get("message"))
    r_union = json.loads(svc.execute_query("SELECT * FROM sign_progress UNION SELECT * FROM employees"))
    check("UNION 被拒绝", r_union.get("status") == "rejected", r_union.get("message"))
    r_cmt = json.loads(svc.execute_query("SELECT * FROM sign_progress WHERE 1=1 -- 注释"))
    check("注释被拒绝", r_cmt.get("status") == "rejected", r_cmt.get("message"))
    r_tab = json.loads(svc.execute_query("SELECT * FROM users"))
    check("非白名单表被拒绝", r_tab.get("status") == "rejected", r_tab.get("message"))
    r_col = json.loads(svc.execute_query("SELECT password FROM sign_progress"))
    check("非法列被拒绝", r_col.get("status") == "rejected", r_col.get("message"))

    print("\n[0.1] 服务层：模糊查询 + 状态修改参数校验")
    r4 = json.loads(svc.update_sign_status("西湖", "已驳回", "孙志远", ""))
    check("驳回缺原因被拦截", r4.get("status") == "error" and "原因" in r4.get("message", ""), r4.get("message"))
    r5 = json.loads(svc.update_sign_status("不存在的公司", "已签约", "张伟", ""))
    check("目标不存在返回无相关信息", r5.get("status") == "no_data", r5.get("message"))

    print("\n[0.2] 服务层：乐观锁（并发冲突保护）")
    # 西湖酒店当前为【待签约】，用期望旧状态【洽谈中】修改应被拒绝
    r_opt = json.loads(svc.update_sign_status("西湖酒店", "已签约", "张伟", "", expected_old_status="洽谈中"))
    check("乐观锁拦截状态不一致", r_opt.get("status") == "error" and "已被他人修改" in r_opt.get("message", ""), r_opt.get("message"))
    r_opt2 = json.loads(svc.update_sign_status("西湖酒店", "已签约", "张伟", "", expected_old_status="待签约"))
    check("乐观锁：期望状态一致时可修改", r_opt2.get("status") == "success", r_opt2.get("message"))

    # ===== Agent 测试（A2A） =====
    print("\n[1] ProgressQueryAgent：模糊查询「西湖」")
    resp = call_agent("http://127.0.0.1:5020", "查询西湖的签订进度")
    print("   ", short(resp))
    check("模糊查询命中多家「西湖」公司", "西湖文化旅游" in resp or "西湖酒店" in resp, "命中")

    print("\n[2] ProgressQueryAgent：无结果提示")
    resp = call_agent("http://127.0.0.1:5020", "查询火星开发公司的签订进度")
    print("   ", short(resp))
    check("无结果输出统一提示", "无相关信息" in resp, short(resp))

    print("\n[3] ProgressQueryAgent：危险操作被拒绝")
    resp = call_agent("http://127.0.0.1:5020", "把西湖的记录删掉")
    print("   ", short(resp))
    check("删除意图被拒绝并提示联系负责人", "请联系相关负责人处理" in resp, short(resp))

    print("\n[4] StaffQueryAgent：查询职员信息")
    resp = call_agent("http://127.0.0.1:5021", "查一下张伟的资料")
    print("   ", short(resp))
    check("返回职员所有相关信息", "张伟" in resp and ("销售" in resp or "商务部" in resp), short(resp))

    print("\n[5] SignStatusModifyAgent：修改签订状态（先查进度再修改）")
    resp = call_agent("http://127.0.0.1:5022", "把西湖酒店的签订状态改成待签约，我是王强")
    print("   ", short(resp, 220))
    wf = json.loads(svc.execute_query("SELECT company_name, status, operator_name FROM sign_progress WHERE company_name LIKE '%西湖酒店%'"))
    status = wf["data"][0]["status"] if wf.get("data") else ""
    check("状态已修改为待签约", status == "待签约", f"实际={status}")

    print("\n[6] SignStatusModifyAgent：驳回（需姓名+原因）")
    resp = call_agent("http://127.0.0.1:5022", "驳回西湖酒店的项目，我是孙志远，原因是客户付款条件不符合要求")
    print("   ", short(resp, 220))
    wf = json.loads(svc.execute_query("SELECT company_name, status, operator_name, reason FROM sign_progress WHERE company_name LIKE '%西湖酒店%'"))
    d = wf["data"][0] if wf.get("data") else {}
    print(f"    DB 状态: {d.get('status')}, 操作人: {d.get('operator_name')}, 原因: {d.get('reason')}")
    check("状态已修改为已驳回", d.get("status") == "已驳回", f"实际={d.get('status')}")
    check("操作人已记录", d.get("operator_name") == "孙志远", f"实际={d.get('operator_name')}")
    check("驳回原因已记录", d.get("reason") == "客户付款条件不符合要求", f"实际={d.get('reason')}")

    print("\n[6.1] 审计落库校验")
    audit = json.loads(svc.execute_query("SELECT * FROM audit_log WHERE company_name LIKE '%西湖酒店%' ORDER BY audit_id DESC LIMIT 1"))
    if audit.get("data"):
        a = audit["data"][0]
        print(f"    审计记录: {a.get('old_status')} -> {a.get('new_status')} | 操作人={a.get('operator_name')} | 原因={a.get('reason')} | trace={a.get('trace_id')}")
        check("审计记录字段完整", a.get("new_status") == "已驳回" and a.get("operator_name") == "孙志远"
              and a.get("reason") == "客户付款条件不符合要求", f"old={a.get('old_status')} new={a.get('new_status')}")
    else:
        check("审计记录字段完整", False, "audit_log 无记录")

    print("\n[7] SignStatusModifyAgent：信息不足追问（未指定新状态）")
    resp = call_agent("http://127.0.0.1:5022", "修改西湖文旅的进度")
    print("   ", short(resp, 220))
    # 模糊兜底后「西湖文旅」应命中「西湖文化旅游发展集团」，展示当前进度并追问要进行的操作
    check("命中西湖文旅并展示进度/追问", "西湖" in resp and "无相关信息" not in resp, short(resp))

    print("\n" + "=" * 70)
    print(f"测试结果: PASS={PASS}, FAIL={FAIL}")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
