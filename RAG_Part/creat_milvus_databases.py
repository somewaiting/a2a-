from pymilvus import MilvusClient
import os, sys
from dotenv import load_dotenv

rag_qa_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, rag_qa_path)
root_path = os.path.dirname(rag_qa_path)
sys.path.insert(0, root_path)
from My_agent.config import Config

conf = Config()

# 连接到默认数据库
client = MilvusClient(uri=f"http://{conf.milvus_host}:{conf.milvus_port}")

database = client.list_databases()
if conf.milvus_database_name not in database:
    client.create_database(db_name=conf.milvus_database_name)
    print(f"数据库 {conf.milvus_database_name} 创建成功")
else:
    client.using_database(db_name=conf.milvus_database_name)
    print(f"数据库 {conf.milvus_database_name} 已存在")
