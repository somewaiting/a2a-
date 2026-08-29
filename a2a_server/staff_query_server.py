#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: staff_query_server.py
作者: ZZS
项目: My_agent（基于A2A协议的企业签订进度智能查询/修改系统）
创建日期: 2026/8/26
描述: 职员查询 Agent 服务器（StaffQueryAgent，端口 5021）
      流程：LLM 生成 SELECT SQL（禁止写操作）→ 调 MCP 8021 查库 → 格式化 → LLM 润色返回。
      查询职员时返回该职员所有相关信息（含其负责的签订项目）。
"""
import sys
import re
import json
import asyncio
from pathlib import Path

# 将 My_agent 的父目录（New）加入 sys.path
_root = Path(__file__).resolve().parent
while _root.name != "My_agent" and _root.parent != _root:
    _root = _root.parent
if str(_root.parent) not in sys.path:
    sys.path.insert(0, str(_root.parent))

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from python_a2a import A2AServer, run_server, AgentCard, AgentSkill, TaskStatus, TaskState

from My_agent.config import Config
from My_agent.create_logger import logger
from My_agent.main_prompts import MyAgentPrompts
from My_agent.utils.retry import with_retry

conf = Config()

SCHEMA = """CREATE TABLE employees (
    emp_id INT PRIMARY KEY COMMENT '职员ID',
    emp_no VARCHAR(30) COMMENT '工号',
    name VARCHAR(50) COMMENT '姓名',
    role VARCHAR(20) COMMENT '岗位(销售/技术/财务/法务/项目经理等)',
    dept VARCHAR(50) COMMENT '部门',
    phone VARCHAR(20) COMMENT '电话',
    email VARCHAR(100) COMMENT '邮箱',
    status VARCHAR(10) COMMENT '在职状态'
);
CREATE TABLE sign_progress (
    progress_id INT PRIMARY KEY COMMENT '进度ID',
    company_name VARCHAR(100) COMMENT '公司名称',
    project_name VARCHAR(200) COMMENT '项目名称',
    status VARCHAR(20) COMMENT '签订状态',
    person_in_charge VARCHAR(50) COMMENT '负责人(职员姓名)'
);"""

# 生成 SQL 的提示词模板字符串（运行时再按需构建 ChatPromptTemplate，避免启动期依赖 langchain）
SQL_PROMPT_TEMPLATE = """
系统提示：你是一个专业的企业职员名单查询 SQL 生成器，基于 employees 表生成 SELECT 语句。
- 仅允许 SELECT 查询，禁止 DELETE / UPDATE / INSERT / DROP 等任何写操作！
- 仅允许查询 employees、sign_progress 两张表，禁止 UNION、注释（--、#、/*）、多语句（分号;）、子查询写操作。
- 如果用户意图删除/修改/清空职员数据，输出：{{"status": "rejected", "message": "请联系相关负责人处理"}}
- 查询某职员时，应返回其所有字段；若需同时返回其负责的签订项目，可 JOIN sign_progress（employees.name = sign_progress.person_in_charge）。
- 支持姓名模糊查询，如查询 "张伟" 用 WHERE e.name LIKE '%张伟%'。
- 查询整个职员名单时可只查 employees 表核心字段。
- 信息不足时输出：{{"status": "input_required", "message": "请问您想查询哪位职员的信息？例如：查询张伟的资料"}}

表结构：{schema}

示例：
- 对话: user: 查一下张伟的资料
输出: SELECT e.emp_id, e.emp_no, e.name, e.role, e.dept, e.phone, e.email, e.status, p.company_name AS charge_company, p.project_name AS charge_project, p.status AS project_status FROM employees e LEFT JOIN sign_progress p ON e.name = p.person_in_charge WHERE e.name LIKE '%张伟%'
- 对话: user: 职员名单里有哪些人
输出: SELECT emp_id, emp_no, name, role, dept, phone, email, status FROM employees
- 对话: user: 技术部有哪些人
输出: SELECT emp_id, emp_no, name, role, dept, phone FROM employees WHERE dept = '技术部'
- 对话: user: 把张伟删了
输出: {{"status": "rejected", "message": "请联系相关负责人处理"}}
- 对话: user: 职员
输出: {{"status": "input_required", "message": "请问您想查询哪位职员的信息？例如：查询张伟的资料"}}

对话历史: {conversation}
当前日期: {current_date} (Asia/Shanghai)
"""

async def _call_staff_mcp(sql):
    async with streamablehttp_client("http://127.0.0.1:8021/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await asyncio.wait_for(session.call_tool("query_staff", {"sql": sql}), timeout=30)
            return result.content[0].text


async def query_staff(sql):
    try:
        return await with_retry(lambda: _call_staff_mcp(sql), retries=2, base_delay=0.5)
    except asyncio.TimeoutError:
        logger.error(f"职员 MCP 查询超时: {sql}")
        return json.dumps({"status": "error", "message": "服务响应超时，请稍后重试。"}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"职员 MCP 查询出错：{str(e)}")
        return json.dumps({"status": "error", "message": f"职员 MCP 查询出错：{str(e)}"}, ensure_ascii=False)


agent_card = AgentCard(
    name="StaffQueryAgent",
    description="基于 LangChain 提供职员名单查询服务的助手（支持模糊查询，返回职员所有相关信息）",
    url="http://localhost:5021",
    version="1.0.0",
    capabilities={"streaming": True, "memory": True},
    skills=[
        AgentSkill(
            name="query staff",
            description="查询职员名单或某职员的所有信息（含其负责的签订项目）",
            examples=["查一下张伟的资料", "职员名单里有哪些人", "技术部有哪些人"]
        )
    ]
)


class StaffQueryServer(A2AServer):
    def __init__(self):
        super().__init__(agent_card=agent_card)
        self._llm = None
        self._sql_prompt = None
        self.schema = SCHEMA

    @property
    def llm(self):
        """延迟构造 ChatOpenAI：避免启动时因 langchain 导入问题导致服务起不来"""
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=conf.model_name,
                base_url=conf.base_url,
                api_key=conf.api_key,
                temperature=0.1,
                timeout=30,
                max_retries=2,
            )
        return self._llm

    @property
    def sql_prompt(self):
        """延迟构建 SQL 提示词模板（langchain 相关依赖按需加载）"""
        if self._sql_prompt is None:
            from langchain_core.prompts import ChatPromptTemplate
            self._sql_prompt = ChatPromptTemplate.from_template(SQL_PROMPT_TEMPLATE)
        return self._sql_prompt

    def generate_sql_query(self, conversation: str) -> dict:
        try:
            from datetime import datetime
            import pytz
            chain = self.sql_prompt | self.llm
            current_date = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d')
            output = chain.invoke({"conversation": conversation, "current_date": current_date, "schema": self.schema}).content.strip()
            logger.info(f"原始 LLM 输出: {output}")
            if output.startswith('{'):
                return json.loads(output)
            return {"status": "sql", "sql": output}
        except Exception as e:
            logger.error(f"SQL 生成失败: {str(e)}")
            return {"status": "input_required", "message": "查询无效，请提供职员姓名。"}

    def _format_rows(self, rows) -> str:
        lines = []
        for d in rows:
            lines.append(
                f"职员 {d.get('name', '')}（工号 {d.get('emp_no', '')}，{d.get('role', '')}，{d.get('dept', '')}）："
                f"电话 {d.get('phone', '')}，邮箱 {d.get('email', '')}，状态 {d.get('status', '')}"
                + (f"，负责项目：{d.get('charge_company', '')} - {d.get('charge_project', '')}（{d.get('project_status', '')}）"
                   if d.get('charge_project') else ""))
        return "\n".join(lines)

    def handle_task(self, task):
        content = (task.message or {}).get("content", {})
        conversation = content.get("text", "") if isinstance(content, dict) else ""
        trace_id = ""
        m = re.search(r'\[trace:([0-9a-f]+)\]', conversation)
        if m:
            trace_id = m.group(1)
        logger.info(f"[trace:{trace_id}] 对话历史及用户问题: {conversation}")

        try:
            gen_result = self.generate_sql_query(conversation)
            logger.info(f"[trace:{trace_id}] SQL 生成结果: {str(gen_result)[:200]}")
            if gen_result.get("status") == "rejected":
                task.artifacts = [{"parts": [{"type": "text", "text": gen_result.get("message", "请联系相关负责人处理")}]}]
                task.status = TaskStatus(state=TaskState.COMPLETED)
                return task
            if gen_result.get("status") == "input_required":
                task.status = TaskStatus(state=TaskState.INPUT_REQUIRED,
                                         message={"role": "agent", "content": {"text": gen_result["message"]}})
                return task

            sql = gen_result.get("sql", "")
            logger.info(f"生成的 SQL: {sql}")
            result = asyncio.run(query_staff(sql))
            response = json.loads(result) if isinstance(result, str) else result

            if response.get("status") == "rejected":
                task.artifacts = [{"parts": [{"type": "text", "text": response.get("message", "请联系相关负责人处理")}]}]
                task.status = TaskStatus(state=TaskState.COMPLETED)
            elif response.get("status") == "success":
                rows = response.get("data", [])
                formatted = self._format_rows(rows)
                chain = MyAgentPrompts.polish_staff_prompt() | self.llm
                polished = chain.invoke({"query": conversation, "raw_response": formatted}).content.strip()
                task.artifacts = [{"parts": [{"type": "text", "text": polished}]}]
                task.status = TaskStatus(state=TaskState.COMPLETED)
            elif response.get("status") == "no_data":
                task.artifacts = [{"parts": [{"type": "text", "text": "无相关信息，请联系相关负责人处理或查找其他内容"}]}]
                task.status = TaskStatus(state=TaskState.COMPLETED)
            else:
                task.status = TaskStatus(state=TaskState.FAILED,
                                         message={"role": "agent", "content": {"text": response.get("message", "查询失败，请重试。")}})
            return task
        except Exception as e:
            logger.error(f"查询失败: {str(e)}")
            task.status = TaskStatus(state=TaskState.FAILED,
                                     message={"role": "agent", "content": {"text": f"查询失败: {str(e)} 请重试或提供更多细节。"}})
            return task


if __name__ == "__main__":
    server = StaffQueryServer()
    print("\n=== 服务器信息 ===")
    print(f"名称: {server.agent_card.name}")
    print(f"描述: {server.agent_card.description}")
    for skill in server.agent_card.skills:
        print(f"- {skill.name}: {skill.description}")
    run_server(server, host="127.0.0.1", port=5021)
