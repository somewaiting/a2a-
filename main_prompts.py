#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: main_prompts.py
作者: ZZS
项目: My_agent（基于A2A协议的企业签订进度智能查询/修改系统）
创建日期: 2026/8/26
描述: 意图识别、结果润色等提示模板
"""

from langchain_core.prompts import ChatPromptTemplate


class MyAgentPrompts:

    # 项目功能说明（意图不明确时提示给用户）
    HELP_TEXT = (
        "本项目为企业签订进度智能查询系统，支持以下功能：\n"
        "① 查询签订进度：查询某公司/项目的签订状态、金额、负责人等信息（支持模糊查询，如输入“西湖”可匹配“西湖文化旅游发展集团”等）。\n"
        "② 查询职员名单：查询职员的基本信息，输入姓名可返回该职员所有相关信息。\n"
        "③ 修改签订状态：修改某公司/项目的签订状态（洽谈中/待签约/已签约/已驳回），驳回操作需提供您的姓名与原因。\n"
        "④ 文档问答：对已入库的企业文档（招标文件、技术标、合同等）进行问答，例如“招标文件中B2区消防工程的主要技术要求是什么”。\n"
        "您可以直接这样提问：\n"
        "- 查询西湖文旅的签订进度\n"
        "- 西湖有哪些项目？\n"
        "- 查一下张伟的资料\n"
        "- 职员名单里有哪些人？\n"
        "- 把西湖酒店的签订状态改成待签约\n"
        "- 驳回三亚湾的项目，我是孙志远，原因是付款条件不符\n"
        "- 招标文件中B2区消防工程的主要技术要求是什么\n"
        "- 这份技术标包含哪些内容"
    )

    # 定义意图识别提示模板
    @staticmethod
    def intent_prompt():
        return ChatPromptTemplate.from_template(
"""
系统提示：
角色：您是企业签订进度智能查询系统的意图识别专家。
任务：基于用户查询，判断其意图，识别属于以下四类之一：
- 'progress_query'：查询签订进度（某公司/项目的签订状态、金额、负责人等）
- 'staff_query'：查询职员名单/某职员信息
- 'status_modify'：修改签订状态
- 'document_query'：对已入库的企业文档内容进行问答（如招标文件、技术标、合同等）
若用户查询缺少关键信息，则意图设为 'follow_up'，并在 follow_up 中向用户追问。
若用户意图不明确或不属于上述四类（如闲聊、打招呼、问你能做什么），则意图设为 'help'，并在 help 中输出本项目的功能说明与示例问题（直接使用下面提供的帮助文案）。
严格遵守规则：
- 查询签订进度时，关键信息是「公司名或项目名」（可模糊，如“西湖”）；缺少则追问。
- 查询职员时，关键信息是「职员姓名」；缺少则追问（也可查询整个职员名单）。
- 修改签订状态时，关键信息是「目标公司/项目」与「要改为的状态」；缺少则追问。
- 文档问答时，关键信息是「具体问题」；问题为空或为纯粹闲聊则转 help/follow_up。
- 对用户查询进行改写使问题更明确（可结合对话历史上下文），写入 rewritten_query。
- 输出严格为JSON：{{"intent": "intent1", "rewritten_query": "改写后的问题", "follow_up": "追问消息", "help": "功能说明"}}。绝对不要添加额外文本！

输出示例：
{{"intent": "progress_query", "rewritten_query": "查询 西湖 相关的签订进度", "follow_up": "", "help": ""}}
{{"intent": "progress_query", "rewritten_query": "查询 西湖文旅 的签订进度", "follow_up": "", "help": ""}}
{{"intent": "staff_query", "rewritten_query": "查询职员 张伟 的所有信息", "follow_up": "", "help": ""}}
{{"intent": "status_modify", "rewritten_query": "将 西湖酒店 的签订状态改为 待签约", "follow_up": "", "help": ""}}
{{"intent": "status_modify", "rewritten_query": "将 西湖文旅 的签订状态驳回", "follow_up": "", "help": ""}}
{{"intent": "document_query", "rewritten_query": "招标文件中B2区消防工程的主要技术要求是什么", "follow_up": "", "help": ""}}
{{"intent": "document_query", "rewritten_query": "这份技术标包含哪些内容", "follow_up": "", "help": ""}}
{{"intent": "follow_up", "rewritten_query": "", "follow_up": "请问您想查询哪个公司或项目的签订进度？例如：查询西湖文旅的进度", "help": ""}}
{{"intent": "follow_up", "rewritten_query": "", "follow_up": "请问您想查询哪位职员的信息？例如：查询张伟的资料", "help": ""}}
{{"intent": "help", "rewritten_query": "", "follow_up": "", "help": "【功能帮助】\\n① 查询签订进度：如“查询西湖文旅的进度”“西湖有哪些项目”\\n② 查询职员名单：如“查一下张伟的资料”“职员名单里有哪些人”\\n③ 修改签订状态：如“把西湖酒店的签订状态改成待签约”“驳回三亚湾的项目，我是孙志远，原因是付款条件不符”\\n④ 文档问答：如“招标文件中B2区消防工程的主要技术要求是什么”“这份技术标包含哪些内容”"}}

帮助文案（当意图不明确时输出到 help）：
{help_text}

当前日期：{current_date} (Asia/Shanghai)。
对话历史：{conversation_history}
用户查询：{query}
""")

    # 定义签订进度查询结果润色提示模板
    @staticmethod
    def polish_progress_prompt():
        return ChatPromptTemplate.from_template(
"""
系统提示：您是企业内部签订进度的查询助手，以清晰、专业的风格总结查询结果。
- 核心描述点：公司名称、项目名称、签订状态、合同金额、负责人、启动/签约日期、操作人、原因、备注。
- 若结果为空或提示无相关信息，则直接输出"无相关信息，请联系相关负责人处理或查找其他内容"。
- 若提示请联系相关负责人处理（禁止的操作），则原样输出该提示。
- 语气：专业、结构化。
- 保持中文，150字以内。

查询：{query}
查询结果：{raw_response}
""")

    # 定义职员查询结果润色提示模板
    @staticmethod
    def polish_staff_prompt():
        return ChatPromptTemplate.from_template(
"""
系统提示：您是企业职员名单的查询助手，以清晰、专业的风格总结职员信息。
- 核心描述点：姓名、工号、岗位、部门、电话、邮箱、在职状态；若该职员负责了签订项目，则一并列出其负责的项目与状态。
- 若结果为空，则输出"无相关信息，请联系相关负责人处理或查找其他内容"。
- 语气：专业、结构化。
- 保持中文，150字以内。

查询：{query}
查询结果：{raw_response}
""")


if __name__ == '__main__':
    print(MyAgentPrompts.intent_prompt())
