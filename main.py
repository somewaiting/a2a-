#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: main.py
作者: ZZS
项目: My_agent（基于A2A协议的企业签订进度智能查询/修改系统）
创建日期: 2026/8/26
描述: 命令行入口，初始化代理网络并进入交互循环
"""
import sys
from pathlib import Path

# 将 My_agent 的父目录（New）加入 sys.path
_root = Path(__file__).resolve().parent
while _root.name != "My_agent" and _root.parent != _root:
    _root = _root.parent
if str(_root.parent) not in sys.path:
    sys.path.insert(0, str(_root.parent))

import asyncio
import json
import uuid
import re
from datetime import datetime
import pytz
from python_a2a import AgentNetwork, A2AClient, TextContent, Message, MessageRole, Task
from langchain_openai import ChatOpenAI

from My_agent.config import Config
from My_agent.create_logger import logger
from My_agent.main_prompts import MyAgentPrompts
from My_agent.utils import health

conf = Config()

messages = []
agent_network = None
llm = None
agent_urls = {}
conversation_history = ""


def initialize_system():
    global agent_network, llm, agent_urls, conversation_history
    agent_urls = {
        "ProgressQueryAgent": "http://localhost:5020",
        "StaffQueryAgent": "http://localhost:5021",
        "SignStatusModifyAgent": "http://localhost:5022",
        "DocumentQueryAgent": "http://localhost:5023",
    }
    network = AgentNetwork(name="My_agent 签订进度智能助手网络")
    for name, url in agent_urls.items():
        # timeout=120：RAG 文档问答生成较慢（约 40s），默认 30s 客户端超时会导致任务失败
        network.add(name, A2AClient(url, timeout=120))
    agent_network = network

    llm = ChatOpenAI(
        model=conf.model_name,
        api_key=conf.api_key,
        base_url=conf.base_url,
        temperature=0.1,
        timeout=30,
        max_retries=2,
    )
    conversation_history = ""


def intent_agent(user_input):
    global conversation_history, llm
    chain = MyAgentPrompts.intent_prompt() | llm
    current_date = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d')
    intent_response = chain.invoke({
        "conversation_history": '\n'.join(conversation_history.split("\n")[-6:]),
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
    logger.info(f"intent: {intent}||rewritten_query: {rewritten_query}||follow_up: {follow_up}")
    return intent, rewritten_query, follow_up, help_text


def process_user_input(prompt):
    global messages, conversation_history
    messages.append({"role": "user", "content": prompt})
    conversation_history += f"\nUser: {prompt}"

    print("正在分析您的意图...")
    trace_id = uuid.uuid4().hex[:8]
    try:
        intent, rewritten_query, follow_up, help_text = intent_agent(prompt)

        if intent == "help":
            response = help_text if help_text else MyAgentPrompts.HELP_TEXT
            conversation_history += f"\nAssistant: {response}"
        elif intent == "follow_up":
            response = follow_up if follow_up else "请提供更明确的信息，例如：查询西湖文旅的签订进度。"
            conversation_history += f"\nAssistant: {response}"
        else:
            agent_name = conf.intent.get(intent)
            if not agent_name:
                response = "暂不支持该操作。您可以输入「查询西湖文旅的进度」或「查一下张伟的资料」。"
            else:
                query_str = rewritten_query if rewritten_query else prompt
                logger.info(f"[trace:{trace_id}] 路由到 {agent_name}：{query_str}")
                from urllib.parse import urlparse
                agent_url = agent_urls[agent_name]
                _p = urlparse(agent_url)
                if not health.check_tcp(_p.hostname, _p.port, timeout=2):
                    response = f"【{agent_name}】服务暂时不可用，请稍后重试或联系相关负责人处理。"
                else:
                    print(f"正在调用 {agent_name} 处理...")
                    agent = agent_network.get_agent(agent_name)
                    chat_history = (f"[trace:{trace_id}] " +
                                    '\n'.join(conversation_history.split("\n")[-7:-1]) +
                                    f'\nUser: {query_str}')
                    message = Message(content=TextContent(text=chat_history), role=MessageRole.USER)
                    task = Task(id="task-" + str(uuid.uuid4()), message=message.to_dict())
                    try:
                        raw_response = asyncio.run(asyncio.wait_for(agent.send_task_async(task), timeout=120))
                    except asyncio.TimeoutError:
                        response = f"【{agent_name}】响应超时，请稍后重试或联系相关负责人处理。"
                        conversation_history += f"\nAssistant: {response}"
                        print(f"\n助手回复：\n{response}\n")
                        messages.append({"role": "assistant", "content": response})
                        return
                    logger.info(f"[trace:{trace_id}] {agent_name} 原始响应: {raw_response}")
                    if raw_response.status.state == 'completed':
                        response = raw_response.artifacts[0]['parts'][0]['text']
                    else:
                        response = raw_response.status.message['content']['text']
            conversation_history += f"\nAssistant: {response}"

        print(f"\n助手回复：\n{response}\n")
        messages.append({"role": "assistant", "content": response})
    except json.JSONDecodeError as json_err:
        logger.error(f"[trace:{trace_id}] 意图识别JSON解析失败")
        error_message = "意图识别解析失败，请重试。"
        print(f"\n助手回复：\n{error_message}\n")
        messages.append({"role": "assistant", "content": error_message})
    except Exception as e:
        logger.error(f"[trace:{trace_id}] 处理异常: {str(e)}")
        error_message = "处理失败，请稍后重试或联系相关负责人处理。"
        print(f"\n助手回复：\n{error_message}\n")
        messages.append({"role": "assistant", "content": error_message})


def display_agent_cards():
    print("\n🛠️ Agent Cards:")
    for agent_name in agent_network.agents.keys():
        agent_card = agent_network.get_agent_card(agent_name)
        agent_url = agent_urls.get(agent_name, "未知地址")
        print(f"\n--- Agent: {agent_name} ---")
        print(f"技能: {agent_card.skills}")
        print(f"描述: {agent_card.description}")
        print(f"地址: {agent_url}")
        print(f"状态: 在线")


if __name__ == "__main__":
    initialize_system()
    print("🤖 My_agent 企业签订进度智能查询系统")
    print(MyAgentPrompts.HELP_TEXT)
    print("\n输入问题按回车提交；输入 'quit' 退出；输入 'cards' 查看代理卡片。")

    display_agent_cards()

    while True:
        prompt = input("\n请输入您的问题: ").strip()
        if prompt.lower() == 'quit':
            print("感谢使用 My_agent！再见！")
            break
        elif prompt.lower() == 'cards':
            display_agent_cards()
            continue
        elif not prompt:
            continue
        else:
            process_user_input(prompt)

    print("\n---")
    print("My_agent | 基于 A2A + MCP 的企业签订进度智能查询/修改系统 v1.0")
