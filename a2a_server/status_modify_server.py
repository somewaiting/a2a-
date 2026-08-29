#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: status_modify_server.py
作者: ZZS
项目: My_agent（基于A2A协议的企业签订进度智能查询/修改系统）
创建日期: 2026/8/26
描述: 签订状态修改 Agent 服务器（SignStatusModifyAgent，端口 5022）
      流程：提取目标 → 调用查询进度 Agent（A2A→A2A）展示当前进度 →
            tool-calling 调 MCP 8022 修改状态（驳回需姓名+原因；信息不足则追问）。
"""
import sys
import re
import json
import uuid
import asyncio
from pathlib import Path

# 将 My_agent 的父目录（New）加入 sys.path
_root = Path(__file__).resolve().parent
while _root.name != "My_agent" and _root.parent != _root:
    _root = _root.parent
if str(_root.parent) not in sys.path:
    sys.path.insert(0, str(_root.parent))

from langchain_core.prompts import ChatPromptTemplate
from python_a2a import A2AServer, run_server, AgentCard, AgentSkill, TaskStatus, TaskState, A2AClient, Message, TextContent, MessageRole, Task

from My_agent.config import Config
from My_agent.create_logger import logger

conf = Config()

# 目标 MCP：状态修改 8022
STATUS_MCP_URL = "http://127.0.0.1:8022/mcp"

# 目标提取提示词
extract_prompt = ChatPromptTemplate.from_template(
"""
从用户输入中提取"要修改签订状态"的目标公司名或项目名。
- 若存在，输出 JSON：{{"target": "公司名或项目名"}}
- 若无法确定目标，输出 JSON：{{"target": ""}}
只输出 JSON，不要添加任何其他文本。
用户输入：{query}
""")

# 状态修改 tool-calling 系统提示词
STATUS_SYSTEM_PROMPT = """你是企业签订状态修改助手，通过调用工具修改签订状态。
可用工具：
- update_sign_status：修改签订状态，参数包括目标(target, 公司名支持模糊或进度ID)、新状态(new_status, 洽谈中/待签约/已签约/已驳回)、操作人姓名(operator_name)、原因(reason, 可选)、trace_id(日志追踪ID, 可选，若上下文中出现[trace:xxxx]请原样传入)。
使用规则：
- 从用户输入中提取目标公司、新状态、操作人姓名、原因。
- 驳回（新状态为已驳回）必须提供操作人姓名与原因，缺失则向用户追问，不要编造、不要调用工具。
- 若用户未明确要改成的状态，则向用户追问，不要调用工具。
- 上下文已包含该公司当前签订进度，先向用户说明当前进度，再执行修改。"""


agent_card = AgentCard(
    name="SignStatusModifyAgent",
    description="基于 MCP 提供签订状态修改服务的助手（先查询进度再修改，驳回需姓名与原因）",
    url="http://localhost:5022",
    version="1.0.0",
    capabilities={"streaming": True, "memory": True},
    skills=[
        AgentSkill(
            name="update sign status",
            description="修改公司/项目的签订状态（洽谈中/待签约/已签约/已驳回），先展示当前进度，驳回需提供姓名与原因",
            examples=["把西湖酒店的签订状态改成待签约", "驳回西湖文旅的项目，我是孙志远，原因是付款条件不符"]
        )
    ]
)


class SignStatusModifyServer(A2AServer):
    def __init__(self):
        super().__init__(agent_card=agent_card)
        self._llm = None
        self.extract_prompt = extract_prompt
        # 反向调用查询进度 Agent（A2A 调 A2A）；timeout 放宽，避免慢响应被 30s 默认超时截断
        self.progress_client = A2AClient("http://localhost:5020", timeout=120)

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

    def _extract_target(self, conversation: str) -> str:
        try:
            chain = self.extract_prompt | self.llm
            output = chain.invoke({"query": conversation}).content.strip()
            output = re.sub(r'^```json\s*|\s*```$', '', output).strip()
            return json.loads(output).get("target", "")
        except Exception as e:
            logger.error(f"目标提取失败: {str(e)}")
            return ""

    def _call_progress_agent(self, target: str) -> str:
        """调用查询进度 Agent（5020）获取该公司当前签订进度（带超时）。"""
        query = f"查询 {target} 的签订进度"
        message = Message(content=TextContent(text=query), role=MessageRole.USER)
        task = Task(id="task-" + str(uuid.uuid4()), message=message.to_dict())
        try:
            result = asyncio.run(asyncio.wait_for(self.progress_client.send_task_async(task), timeout=60))
        except asyncio.TimeoutError:
            return "无相关信息，请联系相关负责人处理或查找其他内容"
        if result.status.state == 'completed':
            return result.artifacts[0]['parts'][0]['text']
        return result.status.message['content']['text']

    def handle_task(self, task):
        content = (task.message or {}).get("content", {})
        conversation = content.get("text", "") if isinstance(content, dict) else ""
        trace_id = ""
        m = re.search(r'\[trace:([0-9a-f]+)\]', conversation)
        if m:
            trace_id = m.group(1)
        logger.info(f"[trace:{trace_id}] 对话历史及用户问题: {conversation}")

        try:
            # 1 提取修改目标
            target = self._extract_target(conversation)
            logger.info(f"[trace:{trace_id}] 提取目标: {target}")
            if not target:
                task.status = TaskStatus(state=TaskState.INPUT_REQUIRED,
                                         message={"role": "agent", "content": {"text": "请问您想修改哪个公司/项目的签订状态？例如：把西湖酒店的签订状态改成待签约"}})
                return task

            # 2 调用查询进度 Agent，展示当前进度
            progress_text = self._call_progress_agent(target)
            logger.info(f"[trace:{trace_id}] 当前进度: {progress_text[:120]}")
            if "无相关信息" in progress_text:
                task.artifacts = [{"parts": [{"type": "text", "text": "无相关信息，请联系相关负责人处理或查找其他内容"}]}]
                task.status = TaskStatus(state=TaskState.COMPLETED)
                return task

            # 3 tool-calling 执行状态修改（带上当前进度上下文与 trace_id）
            # 延迟导入：langchain.agents 依赖较重，避免启动期引入
            from My_agent.a2a_server.mcp_agent_utils import run_tool_agent
            context = (f"[trace:{trace_id}] {conversation}\n\n该公司的当前签订进度：{progress_text}\n"
                       f"请先向用户说明当前进度，再执行状态修改操作。")
            result = asyncio.run(run_tool_agent(self.llm, STATUS_MCP_URL, STATUS_SYSTEM_PROMPT, context))
            logger.info(f"[trace:{trace_id}] 修改结果: {str(result)[:150]}")

            if result.get("status") == "success":
                task.artifacts = [{"parts": [{"type": "text", "text": result["message"]}]}]
                task.status = TaskStatus(state=TaskState.COMPLETED)
            else:
                task.status = TaskStatus(state=TaskState.FAILED,
                                         message={"role": "agent", "content": {"text": result.get("message", "修改失败")}})
            return task
        except Exception as e:
            logger.error(f"状态修改失败: {str(e)}")
            task.status = TaskStatus(state=TaskState.FAILED,
                                     message={"role": "agent", "content": {"text": f"状态修改失败: {str(e)} 请重试或提供更多细节。"}})
            return task


if __name__ == "__main__":
    server = SignStatusModifyServer()
    print("\n=== 服务器信息 ===")
    print(f"名称: {server.agent_card.name}")
    print(f"描述: {server.agent_card.description}")
    for skill in server.agent_card.skills:
        print(f"- {skill.name}: {skill.description}")
    run_server(server, host="127.0.0.1", port=5022)
