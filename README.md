# My_agent —— 企业签订进度智能查询/修改系统（多智能体）

基于 **A2A（Agent2Agent）+ MCP（Model Context Protocol）** 双协议构建的企业内部智能助手：用户（不限角色）从前端自然语言提问，系统自动识别意图并路由到对应 Agent，完成**签订进度查询（支持模糊匹配）、职员名单查询、签订状态修改、已入库文档问答**，全程 A2A + MCP + MySQL + Milvus + 大模型润色。

核心链路：**意图识别（LLM 输出 JSON）→ 路由 → A2A Agent（Text-to-SQL 生成 SQL → 调 MCP 工具查库 → 格式化 → 润色）→ 返回前端**。

---

## 1. 功能与业务流程

```
用户输入（不限人物）
   │ 意图识别 Agent
   ├── progress_query  查询签订进度  ──► ProgressQueryAgent    (A2A: 5020, MCP: 8020)
   ├── staff_query     查询职员名单  ──► StaffQueryAgent       (A2A: 5021, MCP: 8021)
   ├── status_modify   修改签订状态  ──► SignStatusModifyAgent (A2A: 5022, MCP: 8022)
   ├── document_query  文档问答      ──► DocumentQueryAgent    (A2A: 5023, MCP: 8023)
   ├── follow_up       信息不足 → 追问
   └── help            意图不明确 → 提示功能 + 示例问题
```

- **查询签订进度**：Text-to-SQL 查询 `sign_progress` 表；**支持模糊查询**（输入「西湖」可匹配「西湖文化旅游发展集团」「西湖酒店管理有限公司」等），返回公司/项目/状态/金额/负责人等信息，经 LLM 润色后返回。
- **查询职员名单**：查询职员所有相关信息（含其负责的签订项目），支持姓名模糊匹配。
- **修改签订状态**：**先调用查询进度 Agent**（A2A 调 A2A）展示当前签订进度，再执行修改（洽谈中/待签约/已签约/已驳回）；**驳回操作必须提供操作人姓名与原因**，信息不足则追问。
- **文档问答**：对已入库的企业文档（招标文件、技术标、合同等）进行问答，基于 `My_agent/RAG_Part` 的混合检索（BGE-M3 + bge-reranker）+ Qwen 生成答案。

### 强制安全约束
- 查询类 Agent 的 SQL 生成**强行限定仅允许 SELECT，禁止 DELETE/UPDATE/INSERT 等任何写操作**；若用户意图删除/修改数据，返回 **「请联系相关负责人处理」**。
- 数据服务层二次校验：非 SELECT 语句或含危险关键字一律拒绝。
- 无查询结果时统一返回 **「无相关信息，请联系相关负责人处理或查找其他内容」**。

### 文档问答（RAG）
- **RAG 知识检索模块（MCP 8023，RagTools）**：基于已入库文档（Milvus `my_rag` 集合）进行检索问答，工具为 `search_documents(query)`。
- 文档入库由 `My_agent/RAG_Part/parse_json.py` 完成（MinerU 解析 → 父子块切分 → 入库 Milvus）。
- 单独测试 RAG 问答：`cd My_agent/RAG_Part && python rag_main.py`。

---

## 2. 端口分配

| 类型 | 端口 | 服务 | 文件 |
|---|---|---|---|
| MCP | 8020 | 签订进度查询 ProgressTools（`query_progress`） | `My_agent/mcp_server/mcp_progress_server.py` |
| MCP | 8021 | 职员查询 StaffTools（`query_staff`） | `My_agent/mcp_server/mcp_staff_server.py` |
| MCP | 8022 | 状态修改 StatusTools（`update_sign_status`） | `My_agent/mcp_server/mcp_status_server.py` |
| MCP | 8023 | RAG 文档问答 RagTools（`search_documents`） | `My_agent/mcp_server/mcp_rag_server.py` |
| A2A | 5020 | ProgressQueryAgent（进度查询） | `My_agent/a2a_server/progress_query_server.py` |
| A2A | 5021 | StaffQueryAgent（职员查询） | `My_agent/a2a_server/staff_query_server.py` |
| A2A | 5022 | SignStatusModifyAgent（状态修改） | `My_agent/a2a_server/status_modify_server.py` |
| A2A | 5023 | DocumentQueryAgent（文档问答） | `My_agent/a2a_server/document_query_server.py` |
| 前端 | 8501 | Streamlit 界面 / 命令行 | `My_agent/app.py` / `My_agent/main.py` |

---

## 3. 目录结构

```
New/
├── My_agent/
│   ├── a2a_server/        # 4 个 A2A Agent + 共享 MCP 工具调用函数
│   ├── mcp_server/        # 4 个 MCP 工具服务器（含 RAG 文档问答）
│   ├── query_data/my_agent_service.py  # MySQL 服务（SELECT 强制白名单 + 状态修改）
│   ├── sql/               # 建库 + 初始化数据脚本
│   ├── app.py             # Streamlit 前端
│   ├── main.py            # 命令行入口
│   ├── main_prompts.py    # 提示词模板（意图识别/润色/功能帮助）
│   ├── config.py          # 全局配置（LLM/数据库/端口/意图路由）
│   ├── create_logger.py   # 日志
│   ├── start_all.py       # 一键批量启动
│   ├── requirements.txt   # 依赖
│   └── test/              # 端到端测试
└── logs/my_agent/         # 日志（运行时自动创建）
```

---

## 4. 环境准备与启动

1. **Python 3.12**：`pip install -r My_agent/requirements.txt`
2. **MySQL**：`My_agent/config.py` 默认 `localhost:33060, root/123456`
3. **大模型**：`My_agent/config.py` 顶部 `base_url/api_key/model_name`（DeepSeek）

### 密钥配置（首次，真实 Key 不入库）

仓库不包含真实 API Key，克隆后需先从示例复制并填入：

```bash
cp config.example.py config.py                   # 填入 DeepSeek API Key
cp RAG_Part/config.example.py RAG_Part/config.py # 填入 Qwen / DashScope API Key
cp RAG_Part/Parse.example.py RAG_Part/Parse.py   # 填入 MinerU Token（如需重新入库）
```

各 `config.py` 中以 `# TODO: 填入...` 标注的位置即需要 API Key 的地方。

### 建库（首次）

```bash
mysql -h127.0.0.1 -P33060 -uroot -p123456 < My_agent/sql/sql_data.sql     # 建库 my_agent_db
mysql -h127.0.0.1 -P33060 -uroot -p123456 < My_agent/sql/insert_data.sql # 职员 + 签订进度数据
```

### 启动（在 New 目录下）

```bash
python My_agent/start_all.py          # 一键拉起 4 MCP + 4 Agent
streamlit run My_agent/app.py         # 前端（或 python My_agent/main.py 命令行）
```

> 前端端口固定为 **9200**（见 `My_agent/.streamlit/config.toml`）。原因：Windows 的 Hyper-V/Winnat 会动态保留一段 TCP 端口，默认的 8501 落入保留段（8486-8585）时绑定会报 `PermissionError: [WinError 10013]`。如 9200 也被保留，可用 `net stop winnat && net start winnat` 释放后重试，或修改 `config.toml` 换一个空闲端口。

---

## 5. 示例对话

| 类型 | 示例 |
|---|---|
| 进度查询（模糊） | 「查询西湖的签订进度」「西湖文旅的项目有哪些」「已签约的项目有哪些」 |
| 职员查询 | 「查一下张伟的资料」「职员名单里有哪些人」「技术部有哪些人」 |
| 状态修改 | 「把西湖酒店的签订状态改成待签约」 |
| 状态修改（驳回） | 「驳回西湖文旅的项目，我是孙志远，原因是付款条件不符合公司要求」 |
| 文档问答 | 「招标文件中B2区消防工程的主要技术要求是什么」「这份技术标包含哪些内容」 |
| 追问（信息不足） | 「进度」→ 追问「请问您想查询哪个公司或项目的签订进度？」 |
| 功能提示（意图不明确） | 「你好」「你能做什么」→ 展示功能说明与示例问题 |
| 危险操作 | 「把西湖的记录删掉」→ 「请联系相关负责人处理」 |
| 无结果 | 「查询不存在公司的进度」→ 「无相关信息，请联系相关负责人处理或查找其他内容」 |

---

## 6. 数据库表（my_agent_db）

- **employees** 职员名单：工号、姓名、岗位、部门、电话、邮箱、状态。
- **sign_progress** 签订进度表：公司名称（模糊查询目标）、项目名称、签订状态（洽谈中/待签约/已签约/已驳回）、合同金额、负责人、启动/签约日期、最近操作人、原因、备注。

---

## 7. 测试

```bash
python My_agent/test/test_my_agent.py    # 端到端：模糊查询/无结果/禁 DELETE/职员查询/状态修改与驳回
```

> 启动全部服务后运行；覆盖模糊查询「西湖」命中、无结果提示、非 SELECT 拒绝、职员信息返回、修改状态与驳回（需姓名+原因）。
