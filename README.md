```markdown
# Agentic RAG 智能问答系统

基于 **FastAPI + LangGraph + Chroma + Streamlit** 构建的 Agentic RAG 智能问答系统。

项目实现了从 PDF 文档上传、解析、向量化、检索到 LLM 生成回答的完整 RAG 流程，并通过 LangGraph 构建 Agent 工作流，支持根据用户问题调用不同检索工具。

## 1. 项目简介

系统主要面向 PDF 文档知识库问答场景。

用户上传 PDF 后，系统会自动完成文档解析、文本切分、Embedding 和向量数据库构建。用户提出问题后，系统首先从 PDF 知识库中召回相关内容，再通过 Rerank 模型对候选结果进行重新排序，最终将相关上下文交给 LLM 生成回答。

同时使用 LangGraph 管理 Agent 工作流程，并提供 PDF 检索和 arXiv 学术论文检索工具。

## 2. 核心功能

- PDF 文档上传与解析
- 文本切分与向量化
- Chroma 向量数据库
- PDF 知识库问答
- LangGraph Agent 工作流
- Tool Calling
- PDF / arXiv 检索工具
- DashScope GTE Rerank
- PDF 来源页码标注
- RAGAS 评估
- LangSmith 调试与追踪
- FastAPI 后端
- Streamlit 前端

## 3. 核心 RAG 流程

```text
PDF 文档
   ↓
文档解析
   ↓
文本切分
   ↓
Embedding
   ↓
Chroma Vector Store
   ↓
向量检索 Top 10
   ↓
DashScope GTE Rerank
   ↓
筛选 Top 3
   ↓
LLM
   ↓
生成最终回答
```

其中通过 **Rerank** 对向量检索得到的候选内容进行二次排序，在保证一定召回范围的同时，提高最终上下文与用户问题之间的相关性。

## 4. Agent 工作流

项目使用 **LangGraph** 构建 Agent 工作流。

```text
用户问题
   ↓
PDF 检索
   ↓
Rerank
   ↓
LLM
   ↓
回答 / Tool Calling
          ↓
    ┌─────┴─────┐
    ↓           ↓
PDF Search   arXiv Search
```

PDF 相关问题优先使用上传的 PDF 作为知识来源。

当检索内容不足时，系统不会主动编造 PDF 中不存在的信息。

PDF 回答会尽量保留来源页码，例如：

```text
[Source: page 3]
```

## 5. 技术栈

| 类型 | 技术 |
|---|---|
| 编程语言 | Python 3.11 |
| 后端 | FastAPI |
| Agent | LangGraph |
| LLM | Qwen / DashScope |
| RAG | LangChain |
| Embedding | HuggingFace Sentence Transformers |
| 向量数据库 | Chroma |
| Rerank | DashScope GTE Rerank |
| 前端 | Streamlit |
| Evaluation | RAGAS |
| Observability | LangSmith |
| 包管理 | uv |

## 6. 项目结构

```text
Agentic-RAG
├── client/
│   └── app.py                  # Streamlit 前端
│
├── server/
│   ├── agent/
│   │   ├── graph.py            # LangGraph Agent
│   │   ├── state.py            # Agent 状态
│   │   └── tools.py            # Agent Tools
│   │
│   ├── rag/
│   │   ├── embeddings.py       # Embedding
│   │   ├── loaders.py          # PDF 加载
│   │   └── vectorstore.py      # Chroma
│   │
│   ├── observability/
│   │   └── langsmith.py        # LangSmith
│   │
│   ├── config.py
│   └── main.py                 # FastAPI 服务
│
├── evaluation/
│   ├── datasets/
│   └── run_ragas.py            # RAGAS 评估
│
├── shared/
├── pyproject.toml
└── uv.lock
```

## 7. 本地运行

### 安装依赖

项目使用 uv 管理依赖：

```bash
uv sync
```

### 配置环境变量

在项目根目录创建 `.env`：

```env
DASHSCOPE_API_KEY=你的_API_Key
```

不要将真实 API Key 提交到 GitHub。

### 启动后端

```bash
uv run uvicorn server.main:app --reload --port 8000
```

后端地址：

```text
http://127.0.0.1:8000
```

API 文档：

```text
http://127.0.0.1:8000/docs
```

### 启动前端

```bash
uv run streamlit run client/app.py --server.port 8501
```

访问：

```text
http://localhost:8501
```

## 8. 主要接口

### 健康检查

```http
GET /health
```

### 上传 PDF

```http
POST /upload_pdf
```

上传 PDF 后创建独立会话，并完成对应知识库构建。

### 对话

```http
POST /chat
```

请求示例：

```json
{
  "session_id": "your-session-id",
  "message": "请介绍一下这个 PDF 的主要内容"
}
```

## 9. RAG 评估

项目提供 RAGAS 评估脚本：

```text
evaluation/run_ragas.py
```

用于从以下方向对 RAG 效果进行评估：

- Context Relevance
- Faithfulness
- Answer Relevancy

可以通过评估结果进一步分析检索内容和生成答案的质量。

## 10. 项目重点

本项目重点实践了：

**RAG Pipeline**

```text
文档 → Chunk → Embedding → Retrieval → Rerank → LLM
```

**Agent Workflow**

```text
User Query → LangGraph → Tool Calling → Retrieval → LLM
```

**RAG 检索优化**

```text
Vector Search Top 10
        ↓
   GTE Rerank
        ↓
      Top 3
        ↓
       LLM
```

通过以上流程完成了一个从文档处理、知识检索、Agent 编排到最终回答生成的完整 LLM 应用。

## License

本项目使用了相关开源组件，请遵循对应项目的开源许可证及使用要求。
```

