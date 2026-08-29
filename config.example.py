#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名: config.example.py
说明: 配置文件示例。真实密钥不入库，克隆后请复制本文件为 config.py 并填入密钥。
用法:  cp config.example.py config.py   然后编辑 config.py 填入 API Key。
"""
import os

# 项目根目录
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


class Config:

    def __init__(self):
        # ============ 大模型配置（换模型只需改这三处）============
        self.base_url = 'https://api.deepseek.com'
        # TODO: 填入你的 DeepSeek API Key（https://platform.deepseek.com/ 获取）
        self.api_key = 'PASTE_YOUR_DEEPSEEK_API_KEY_HERE'
        self.model_name = 'deepseek-v4-flash'

        # ============ 数据库配置（my_agent_db 库）============
        self.host = 'localhost'
        self.port = 33060
        self.user = 'root'
        self.password = '123456'
        self.database = 'my_agent_db'

        # ============ MILVUS 配置 ============
        self.milvus_host = 'localhost'
        self.milvus_port = 19530
        self.milvus_database_name = 'My_agent'
        self.milvus_collection_name = 'my_rag'

        # ============ 日志配置 ============
        self.log_file = os.path.join(project_root, 'logs', 'my_agent.log')

        # 意图路由映射：意图 -> A2A Agent 名称
        self.intent = {
            "progress_query": "ProgressQueryAgent",     # 查询签订进度
            "staff_query":    "StaffQueryAgent",        # 查询职员名单/职员信息
            "status_modify":  "SignStatusModifyAgent",  # 修改签订状态
            "document_query": "DocumentQueryAgent",     # 已入库文档问答
        }

        # MCP 服务器端口
        self.mcp_ports = {
            "progress": 8020,   # 签订进度查询
            "staff":    8021,   # 职员查询
            "status":   8022,   # 签订状态修改
            "rag":      8023,   # RAG 文档问答模块接口
        }

        # A2A Agent 服务器端口
        self.agent_ports = {
            "ProgressQueryAgent": 5020,
            "StaffQueryAgent": 5021,
            "SignStatusModifyAgent": 5022,
            "DocumentQueryAgent": 5023,
        }

        self.temperature = 0.1


if __name__ == '__main__':
    conf = Config()
    print('日志文件:', conf.log_file)
    print('数据库:', conf.database)
    print('意图路由:', conf.intent)
    print('MCP端口:', conf.mcp_ports)
    print('Agent端口:', conf.agent_ports)
