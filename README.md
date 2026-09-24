# Agentic RAG with FastAPI and Streamlit

一个基于 FastAPI、LangGraph、Chroma 和 DashScope 构建的 Agentic RAG 智能问答系统。

项目支持上传 PDF 文档，并基于文档内容进行问答。系统采用“向量检索 + Rerank + LLM”的检索增强生成流程，同时结合 LangGraph 构建 Agent 工作流，并通过来源页码引用增强回答的可追溯性。

> 本项目基于开源项目进行二次开发，在原有 Agentic RAG 架构基础上进行了 RAG 检索流程、Rerank、来源引用及 Agent 工作流相关功能的扩展。

---

## 项目功能

- PDF 文档上传
- PDF 文档解析
- 文本切分与向量化
- Chroma 向量数据库
- 向量相似度检索
- DashScope `gte-rerank-v2` 重排序
- LangGraph Agent 工作流
- Tool Calling
- PDF 文档检索
- arXiv 学术论文检索
- PDF 来源页码引用
- 检索信息不足时避免编造答案
- FastAPI 后端 API
- Streamlit 前端交互
- LangSmith 可观测性
- RAGAS 评估脚本

---

## 核心 RAG 流程

项目采用两阶段检索流程：

```text
PDF
 │
 ▼
PDF 解析
 │
 ▼
文本切分
 │
 ▼
Embedding
 │
 ▼
Chroma 向量检索
 │
 │ Top 10
 ▼
DashScope GTE Rerank
 │
 │ Top 3
 ▼
LLM
 │
 ▼
最终回答
```

### 第一阶段：向量召回

首先通过 Embedding 将 PDF 文档转换为向量，并存储到 Chroma。

用户提出问题后，通过向量相似度检索召回候选文档。

当前配置召回 Top 10：

```python
retriever = vectordb.as_retriever(
    search_kwargs={"k": 10}
)
```

### 第二阶段：Rerank

对向量检索得到的 Top 10 文档进一步进行相关性排序。

项目使用 DashScope `gte-rerank-v2`：

```python
response = dashscope.TextReRank.call(
    model="gte-rerank-v2",
    query=query,
    documents=texts,
    top_n=3,
)
```

最终选取相关性更高的 Top 3 文档作为 LLM 的上下文。

这种方式将向量检索的候选召回与最终上下文筛选分开，使检索流程更加清晰。

---

## Agent 工作流

项目使用 LangGraph 构建 Agent 工作流。

整体流程：

```text
用户问题
   │
   ▼
PDF Retrieval
   │
   ▼
LLM
   │
   ├───────────────┐
   │               │
无需其他工具       需要其他工具
   │               │
   ▼               ▼
最终回答        Tool Calling
                   │
                   ▼
                  Tools
                   │
                   ▼
                  LLM
                   │
                   ▼
                最终回答
```

### 当前主要工具

#### `search_pdf`

搜索用户上传的 PDF，并经过 Rerank 后返回相关内容。

处理流程：

```text
用户问题
   ↓
Chroma Top 10
   ↓
GTE Rerank
   ↓
Top 3
   ↓
返回相关 PDF 内容
```

#### `search_arxiv`

通过 arXiv 检索学术论文，用于补充 PDF 文档之外的信息。

---

## 来源引用与回答约束

为了增强 RAG 回答的可追溯性，PDF 检索结果会保留来源页码。

回答基于 PDF 内容时，系统要求引用对应来源，例如：

```text
[Source: page 1]
```

同时对回答增加以下约束：

- 基于检索到的 PDF 内容回答问题
- 不编造 PDF 中不存在的信息
- 不虚构来源页码
- 检索内容不足时明确说明信息不足
- 仅引用实际检索结果中存在的页码

当当前 PDF 内容无法支持回答时，系统会提示：

```text
根据当前 PDF 内容，无法找到足够的信息回答这个问题。
```

---

## 技术架构

```text
                    ┌─────────────────┐
                    │    Streamlit    │
                    │    Frontend     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     FastAPI     │
                    │     Backend     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    LangGraph    │
                    │      Agent      │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
         PDF Retrieval    arXiv Tool      LLM
              │
              ▼
          Chroma
              │
              ▼
         Vector Search
              │
              ▼
      DashScope Rerank
```

---

## 技术栈

### Backend

- Python
- FastAPI
- LangGraph
- LangChain

### RAG

- Chroma
- HuggingFace Embedding
- PDF Loader
- Text Splitting
- Vector Retrieval
- DashScope `gte-rerank-v2`

### LLM

- DashScope
- Qwen

### Agent / Tools

- LangGraph
- Tool Calling
- arXiv Retriever

### Frontend

- Streamlit

### Evaluation / Observability

- RAGAS
- LangSmith

### Development

- uv
- Git

---

## 项目结构

```text
Agentic-RAG-with-FastAPI-and-Streamlit
│
├── client/
│   └── app.py
│
├── evaluation/
│   ├── datasets/
│   └── run_ragas.py
│
├── server/
│   │
│   ├── agent/
│   │   ├── graph.py
│   │   ├── state.py
│   │   └── tools.py
│   │
│   ├── rag/
│   │   ├── embeddings.py
│   │   ├── loaders.py
│   │   └── vectorstore.py
│   │
│   ├── observability/
│   │   └── langsmith.py
│   │
│   ├── config.py
│   └── main.py
│
├── shared/
│
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

---

## 环境要求

- Python 3.11
- uv

---

## 环境配置

在项目根目录创建 `.env` 文件：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
```

`.env` 仅用于本地环境配置，不应提交到 Git 仓库。

项目已经通过 `.gitignore` 忽略 `.env` 文件。

---

## 安装依赖

进入项目目录：

```bash
uv sync
```

---

## 启动后端

```bash
uv run uvicorn server.main:app --reload --port 8000
```

后端启动后，可以访问 FastAPI Swagger 文档：

```text
http://127.0.0.1:8000/docs
```

---

## 启动前端

重新打开一个终端：

```bash
uv run streamlit run client/app.py --server.port 8501
```

然后访问：

```text
http://localhost:8501
```

---

## API

### 健康检查

```http
GET /health
```

返回：

```json
{
  "status": "ok"
}
```

### 上传 PDF

```http
POST /upload_pdf
```

上传 PDF 后，系统会：

1. 保存 PDF 文件
2. 解析 PDF
3. 对文本进行切分
4. 创建 Embedding
5. 构建 Chroma 向量库
6. 创建 Retriever
7. 创建 Agent Session

接口返回对应的 `session_id`。

### 对话

```http
POST /chat
```

请求示例：

```json
{
  "session_id": "your-session-id",
  "message": "请总结这份文档的主要内容"
}
```

系统会根据当前 Session 对用户问题进行处理，并返回回答。

---

## Rerank 实现

项目对传统的“向量检索后直接交给 LLM”的流程进行了扩展。

### 原始流程

```text
Query
 ↓
Embedding
 ↓
Vector Search
 ↓
LLM
```

### 当前流程

```text
Query
 ↓
Embedding
 ↓
Chroma Vector Search
 ↓
Top 10
 ↓
DashScope GTE Rerank
 ↓
Top 3
 ↓
LLM
```

其中：

- Chroma 负责初步召回候选文档
- GTE Rerank 负责对候选文档进行相关性排序
- 最终 Top 3 文档作为 LLM 上下文

这样可以将“候选召回”和“最终相关性排序”分成两个阶段。

---

## Agent 核心流程

项目使用 LangGraph 管理 Agent 工作流。

核心节点包括：

```text
pdf_retrieve
      ↓
     llm
      ↓
   tools
      ↓
     llm
```

其中：

### `pdf_retrieve`

在用户提出问题后首先执行 PDF 检索，并将检索结果提供给 LLM。

### `llm`

负责根据用户问题和检索结果生成回答，同时判断是否需要调用其他工具。

### `tools`

负责执行 Agent 请求的工具，例如 arXiv 检索。

---

## RAG 评估

项目提供 RAGAS 评估脚本和数据集目录：

```text
evaluation/
├── datasets/
└── run_ragas.py
```

可用于后续对 RAG 系统的回答质量进行量化评估。

评估方向包括：

- Context Relevance
- Faithfulness
- Answer Relevancy

---

## LangSmith

项目包含 LangSmith 可观测性相关代码：

```text
server/observability/langsmith.py
```

用于后续对 Agent 执行过程和调用链进行追踪与分析。

---

## 二次开发说明

本项目基于开源项目进行二次开发。

在原有 Agentic RAG 项目的基础上，主要进行了以下开发和调整：

1. 增加 PDF 检索结果的 Rerank 流程
2. 使用 DashScope `gte-rerank-v2` 对向量召回结果进行重排序
3. 将向量召回数量设置为 Top 10，并筛选 Top 3 结果提供给 LLM
4. 增加 PDF 来源页码引用
5. 增加检索信息不足时的回答约束
6. 调整 Agent 工作流，使 PDF 检索在回答前执行

项目使用的原始开源代码、作者信息及许可证请以原项目仓库中的说明为准。

---

## License

本项目遵循原开源项目所采用的许可证。

进行二次开发、修改或再次分发时，请遵守原项目的许可证要求，并保留必要的原作者及版权信息。
