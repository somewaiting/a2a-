#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: progress_query_server.py
作者: ZZS
项目: My_agent（基于A2A协议的企业签订进度智能查询/修改系统）
创建日期: 2026/8/26
描述: 签订进度查询 Agent 服务器（ProgressQueryAgent，端口 5020）
      流程：LLM 生成 SELECT SQL（强行禁止 DELETE 等写操作）→ 调 MCP 8020 查库 → 格式化 → LLM 润色返回。
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

# 签订进度表结构（供 Text-to-SQL）
SCHEMA = """CREATE TABLE sign_progress (
    progress_id INT PRIMARY KEY COMMENT '进度ID',
    company_name VARCHAR(100) COMMENT '客户公司名称',
    project_name VARCHAR(200) COMMENT '项目名称',
    status VARCHAR(20) COMMENT '签订状态(洽谈中/待签约/已签约/已驳回)',
    amount DECIMAL(14,2) COMMENT '合同金额(元)',
    person_in_charge VARCHAR(50) COMMENT '负责人(职员姓名)',
    start_date DATE COMMENT '启动日期',
    sign_date DATE COMMENT '签约日期',
    operator_name VARCHAR(50) COMMENT '最近操作人姓名',
    reason VARCHAR(200) COMMENT '最近操作原因',
    remark TEXT COMMENT '备注'
);"""

# 生成 SQL 的提示词模板字符串（运行时再按需构建 ChatPromptTemplate，避免启动期依赖 langchain）
SQL_PROMPT_TEMPLATE = """
系统提示：你是一个专业的企业签订进度查询 SQL 生成器，基于 sign_progress 表生成 SELECT 语句。
- 仅允许 SELECT 查询，可 JOIN employees（sign_progress.person_in_charge = employees.name）附带负责人工号等。
- 强行限定：绝对禁止生成 DELETE / UPDATE / INSERT / DROP / TRUNCATE / ALTER 等任何写操作语句！
- 仅允许查询 sign_progress、employees 两张表，禁止 UNION、注释（--、#、/*）、多语句（分号;）、子查询写操作。
- 如果用户意图删除、清空、修改数据，或执行非查询操作，输出：{{"status": "rejected", "message": "请联系相关负责人处理"}}
- 支持模糊查询：公司名用 LIKE 模糊匹配，如查询 "西湖" 应生成 WHERE company_name LIKE '%西湖%'。
- 信息不足（缺公司名/项目名/状态等条件）时输出：{{"status": "input_required", "message": "请问您想查询哪个公司或项目的签订进度？例如：查询西湖文旅的进度"}}
- 如果用户问与签订进度无关的问题，模仿最后一个示例回复。

表结构：{schema}

示例：
- 对话: user: 查询西湖的签订进度
输出: SELECT progress_id, company_name, project_name, status, amount, person_in_charge, start_date, sign_date, operator_name, reason FROM sign_progress WHERE company_name LIKE '%西湖%'
- 对话: user: 西湖文旅的项目有哪些
输出: SELECT progress_id, company_name, project_name, status, amount FROM sign_progress WHERE company_name LIKE '%西湖文旅%'
- 对话: user: 已签约的项目有哪些
输出: SELECT progress_id, company_name, project_name, status, amount, sign_date FROM sign_progress WHERE status = '已签约'
- 对话: user: 把西湖的记录删掉
输出: {{"status": "rejected", "message": "请联系相关负责人处理"}}
- 对话: user: 进度
输出: {{"status": "input_required", "message": "请问您想查询哪个公司或项目的签订进度？例如：查询西湖文旅的进度"}}

对话历史: {conversation}
当前日期: {current_date} (Asia/Shanghai)
"""

# 查询函数：调用 MCP 8020（带超时与指数退避重试）
async def _call_progress_mcp(sql):
    async with streamablehttp_client("http://127.0.0.1:8020/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await asyncio.wait_for(session.call_tool("query_progress", {"sql": sql}), timeout=30)
            return result.content[0].text


async def query_progress(sql):
    try:
        return await with_retry(lambda: _call_progress_mcp(sql), retries=2, base_delay=0.5)
    except asyncio.TimeoutError:
        logger.error(f"进度 MCP 查询超时: {sql}")
        return json.dumps({"status": "error", "message": "服务响应超时，请稍后重试。"}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"进度 MCP 查询出错：{str(e)}")
        return json.dumps({"status": "error", "message": f"进度 MCP 查询出错：{str(e)}"}, ensure_ascii=False)


agent_card = AgentCard(
    name="ProgressQueryAgent",
    description="基于 LangChain 提供签订进度查询服务的助手（支持模糊查询，禁止写操作）",
    url="http://localhost:5020",
    version="1.0.0",
    capabilities={"streaming": True, "memory": True},
    skills=[
        AgentSkill(
            name="query sign progress",
            description="查询公司/项目的签订进度，支持模糊匹配，如输入“西湖”可匹配“西湖文化旅游发展集团”",
            examples=["查询西湖的签订进度", "西湖文旅的项目有哪些", "已签约的项目"]
        )
    ]
)


class ProgressQueryServer(A2AServer):
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
                timeout=30,        # LLM 调用超时
                max_retries=2,     # LLM 调用失败重试
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
            return {"status": "input_required", "message": "查询无效，请提供公司名或项目名。"}

    def _format_rows(self, rows) -> str:
        lines = []
        for d in rows:
            lines.append(
                f"[{d.get('company_name', '')} | {d.get('project_name', '')}] "
                f"状态：{d.get('status', '')}，金额：{d.get('amount', '')}元，负责人：{d.get('person_in_charge', '')}，"
                f"启动日期：{d.get('start_date', '')}，签约日期：{d.get('sign_date', '未签约')}，"
                f"最近操作人：{d.get('operator_name', '')}，原因：{d.get('reason', '')}")
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
            result = asyncio.run(query_progress(sql))
            response = json.loads(result) if isinstance(result, str) else result

            if response.get("status") == "rejected":
                task.artifacts = [{"parts": [{"type": "text", "text": response.get("message", "请联系相关负责人处理")}]}]
                task.status = TaskStatus(state=TaskState.COMPLETED)
            elif response.get("status") == "success":
                rows = response.get("data", [])
                formatted = self._format_rows(rows)
                chain = MyAgentPrompts.polish_progress_prompt() | self.llm
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
    server = ProgressQueryServer()
    print("\n=== 服务器信息 ===")
    print(f"名称: {server.agent_card.name}")
    print(f"描述: {server.agent_card.description}")
    for skill in server.agent_card.skills:
        print(f"- {skill.name}: {skill.description}")
    run_server(server, host="127.0.0.1", port=5020)
