#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: app.py
作者: ZZS
项目: My_agent（基于A2A协议的企业签订进度智能查询/修改系统）
创建日期: 2026/8/26
描述: Streamlit 前端界面（用户自由提问 → 意图识别 → 路由到查询进度/查询职员/修改状态 Agent）
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import asyncio
import uuid
import json
import re
from datetime import datetime
import pytz
import streamlit as st
from python_a2a import AgentNetwork, A2AClient, Message, TextContent, MessageRole, Task
from langchain_openai import ChatOpenAI

from My_agent.config import Config
from My_agent.create_logger import logger
from My_agent.main_prompts import MyAgentPrompts
from My_agent.utils import health

conf = Config()

st.set_page_config(page_title="My_agent 企业签订进度智能查询系统", layout="wide", page_icon="🤖")

st.markdown("""
<style>
.stChatMessage { background-color: #1e3a5f !important; border-radius: 12px !important; padding: 15px !important;
                margin-bottom: 15px !important; box-shadow: 0 3px 6px rgba(0,0,0,0.2) !important; }
.stChatMessage.user { background-color: #2a4d73 !important; }
.stChatMessage .stMarkdown, .stChatMessage .stMarkdown p, .stChatMessage .stMarkdown span,
.stChatMessage .stMarkdown div, .stChatMessage .stMarkdown strong, .stChatMessage .stMarkdown em {
    color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# ========== 会话状态初始化 ==========
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_network" not in st.session_state:
    st.session_state.agent_urls = {
        "ProgressQueryAgent": "http://localhost:5020",
        "StaffQueryAgent": "http://localhost:5021",
        "SignStatusModifyAgent": "http://localhost:5022",
        "DocumentQueryAgent": "http://localhost:5023",
    }
    network = AgentNetwork(name="My_agent 签订进度智能助手网络")
    for name, url in st.session_state.agent_urls.items():
        # timeout=120：RAG 文档问答生成较慢（约 40s），默认 30s 客户端超时会导致任务失败
        network.add(name, A2AClient(url, timeout=120))
    st.session_state.agent_network = network
    st.session_state.llm = ChatOpenAI(
        model=conf.model_name,
        api_key=conf.api_key,
        base_url=conf.base_url,
        temperature=0.1,
        timeout=30,
        max_retries=2,
    )
    st.session_state.conversation_history = ""


# ========== 意图识别 agent ==========
def intent_agent(user_input):
    chain = MyAgentPrompts.intent_prompt() | st.session_state.llm
    current_date = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d')
    intent_response = chain.invoke({
        "conversation_history": '\n'.join(st.session_state.conversation_history.split("\n")[-6:]),
        "query": user_input,
        "current_date": current_date,
        "help_text": MyAgentPrompts.HELP_TEXT,
    }).content.strip()
    logger.info(f"意图识别原始响应: {intent_response}")
    intent_response = re.sub(r'^```json\s*|\s*```$', '', intent_response).strip()
    intent_output = json.loads(intent_response)
    intent = intent_output.get("intent", "")
    rewritten_query = intent_output.get("rewritten_query", "")
    follow_up = intent_output.get("follow_up", "")
    help_text = intent_output.get("help", "")
    logger.info(f"intent: {intent}||rewritten_query: {rewritten_query}||follow_up: {follow_up}||help: {help_text[:60]}")
    return intent, rewritten_query, follow_up, help_text


# ========== 侧边栏：功能说明 ==========
with st.sidebar:
    st.header("🤖 My_agent 功能说明")
    st.markdown(MyAgentPrompts.HELP_TEXT)
    st.markdown("---")
    st.markdown("**技术架构：** A2A 多智能体 + MCP 工具 + MySQL + Milvus(RAG) + DeepSeek")
    st.markdown("**文档问答：** RAG 知识检索（8023）已接入，可对已入库文档进行问答")

# ========== 主界面 ==========
st.title("🤖 My_agent 企业签订进度智能查询系统")
st.markdown("面向公司内部：查询签订进度（支持模糊查询）、查询职员名单、修改签订状态、已入库文档问答。输入问题即可。")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 对话")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("请输入您的问题，例如：查询西湖文旅的签订进度..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.conversation_history += f"\nUser: {prompt}"
        trace_id = uuid.uuid4().hex[:8]
        logger.info(f"[trace:{trace_id}] 用户输入: {prompt}")

        with st.status("正在处理您的请求...", expanded=False) as status:
            try:
                status.update(label="正在识别意图...")
                intent, rewritten_query, follow_up, help_text = intent_agent(prompt)

                if intent == "help":
                    status.update(label="完成")
                    response = help_text if help_text else MyAgentPrompts.HELP_TEXT
                    st.session_state.conversation_history += f"\nAssistant: {response}"
                elif intent == "follow_up":
                    status.update(label="完成")
                    response = follow_up if follow_up else "请提供更明确的信息，例如：查询西湖文旅的签订进度。"
                    st.session_state.conversation_history += f"\nAssistant: {response}"
                else:
                    agent_name = conf.intent.get(intent)
                    if not agent_name:
                        status.update(label="完成")
                        response = "暂不支持该操作。您可以输入「查询西湖文旅的进度」或「查一下张伟的资料」。"
                    else:
                        query_str = rewritten_query if rewritten_query else prompt
                        logger.info(f"[trace:{trace_id}] 路由到 {agent_name}：{query_str}")

                        # 健康检查：Agent 服务不可用时友好降级
                        from urllib.parse import urlparse
                        agent_url = st.session_state.agent_urls[agent_name]
                        _p = urlparse(agent_url)
                        if not health.check_tcp(_p.hostname, _p.port, timeout=2):
                            response = f"【{agent_name}】服务暂时不可用，请稍后重试或联系相关负责人处理。"
                        else:
                            status.update(label=f"正在调用 {agent_name} 处理...")
                            agent = st.session_state.agent_network.get_agent(agent_name)
                            chat_history = (f"[trace:{trace_id}] " +
                                            '\n'.join(st.session_state.conversation_history.split("\n")[-7:-1]) +
                                            f'\nUser: {query_str}')
                            message = Message(content=TextContent(text=chat_history), role=MessageRole.USER)
                            task = Task(id="task-" + str(uuid.uuid4()), message=message.to_dict())
                            try:
                                raw_response = asyncio.run(asyncio.wait_for(agent.send_task_async(task), timeout=120))
                            except asyncio.TimeoutError:
                                response = f"【{agent_name}】响应超时，请稍后重试或联系相关负责人处理。"
                            else:
                                logger.info(f"[trace:{trace_id}] {agent_name} 原始响应: {raw_response}")
                                if raw_response.status.state == 'completed':
                                    response = raw_response.artifacts[0]['parts'][0]['text']
                                else:
                                    response = raw_response.status.message['content']['text']
                        status.update(label="完成")
                    st.session_state.conversation_history += f"\nAssistant: {response}"

                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except json.JSONDecodeError as json_err:
                logger.error(f"[trace:{trace_id}] 意图识别JSON解析失败")
                error_message = f"意图识别解析失败，请重试。"
                with st.chat_message("assistant"):
                    st.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                status.update(label="解析失败")
            except Exception as e:
                logger.error(f"[trace:{trace_id}] 处理异常: {str(e)}")
                error_message = "处理失败，请稍后重试或联系相关负责人处理。"
                with st.chat_message("assistant"):
                    st.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                status.update(label="处理失败")

with col2:
    st.subheader("🛠️ Agent 卡片")
    for agent_name in st.session_state.agent_network.agents.keys():
        agent_card = st.session_state.agent_network.get_agent_card(agent_name)
        agent_url = st.session_state.agent_urls.get(agent_name, "未知地址")
        with st.expander(f"Agent: {agent_name}", expanded=False):
            st.markdown(f"**技能**\n\n{agent_card.skills}")
            st.markdown(f"**描述**\n\n{agent_card.description}")
            st.markdown(f"**地址**\n\n{agent_url}")
            st.markdown("**状态**\n\n在线")
    st.markdown("---")
    st.markdown("**🔌 RagTools（8023）**  \nRAG 知识检索已接入：基于已入库文档（招标文件、技术标、合同等）进行问答。")

st.markdown("---")
st.markdown('<div class="footer">My_agent | 基于 A2A + MCP 的企业签订进度智能查询/修改系统 v1.0</div>', unsafe_allow_html=True)
