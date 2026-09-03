# RAG多Agent企业智能问答系统

基于 **LangChain + LangGraph + FastAPI** 的企业级检索增强生成（RAG）问答系统，采用 **路由 / 检索 / 生成 / 反思** 多 Agent 协作架构，面向企业内部知识库提供精准、带引用溯源的问答服务。

## ✨ 核心特性

- **RAG 检索增强生成**：企业内部文档（Markdown / PDF / Word / TXT）解析 → 切分 → 向量化入库，实现基于语义的精准检索
- **多 Agent 协作**：路由 Agent 拆解问题 → 检索 Agent 多路召回 → 生成 Agent 融合作答 → 反思 Agent 质量校验，通过状态机编排完成复杂问题的端到端处理
- **多路召回 + 重排序**：稠密向量检索 + BM25 关键词检索双路召回，RRF 融合后经 Rerank 精排
- **引用溯源**：回答附带来源文档与片段引用，支持人工核验
- **流式输出 + 多轮对话**：SSE 流式返回，Redis 缓存热点问答与对话历史
- **可观测评估**：内置召回率（Recall@K）评测脚本

## 🏗️ 系统架构

```
用户提问
   │
   ▼
┌─────────────┐  简单问题  ┌──────────────┐
│ 路由 Agent   │──────────▶│ 直接回答      │
└─────┬───────┘            └──────────────┘
      │ 知识/复杂问题
      ▼
┌─────────────┐  拆解子问题
│ 子问题队列   │
└─────┬───────┘
      ▼
┌────────────────────────────────────┐
│ 检索 Agent：向量检索 + BM25 关键词    │
│ 检索 → RRF 融合 → Rerank 精排        │
└─────┬──────────────────────────────┘
      ▼
┌─────────────┐  引用上下文生成答案
│ 生成 Agent   │
└─────┬───────┘
      ▼
┌─────────────┐  不通过（最多2轮修订）
│ 反思 Agent   │──────────────────────┐
└─────┬───────┘                       │
      │ 通过                          │
      ▼                               │
  答案 + 引用 ─────────────────────────┘
```

## 🚀 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 配置

```bash
cp .env.example .env
# 编辑 .env，至少配置 LLM 相关项（兼容 OpenAI 接口，可使用 DeepSeek / 通义千问 / OpenAI）
```

默认使用 DeepSeek 作为 LLM（`OPENAI_API_BASE=https://api.deepseek.com`），
本地嵌入模型使用 `BAAI/bge-small-zh-v1.5`（首次运行自动下载，无需额外 API Key）。

### 3. 启动可选依赖（Milvus / Redis）

```bash
docker compose up -d          # 启动 Milvus + Redis（可选，默认使用本地 FAISS + 内存缓存）
```

### 4. 导入示例知识库

```bash
python scripts/ingest_docs.py --dir data/sample_docs
```

### 5. 启动服务

```bash
uvicorn app.main:app --reload
# API 文档: http://localhost:8000/docs
```

### 6. 调用示例

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "年假如何申请？", "session_id": "demo"}'
```

## 🔌 主要接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/chat` | 问答（非流式） |
| POST | `/api/chat/stream` | 问答（SSE 流式） |
| POST | `/api/documents/ingest` | 文档入库（目录/文件） |
| GET | `/api/documents` | 已入库文档列表 |
| GET | `/api/health` | 健康检查 |

## ⚙️ 配置项（.env）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `OPENAI_API_KEY` | - | LLM API Key（必填） |
| `OPENAI_API_BASE` | `https://api.deepseek.com` | LLM 服务地址 |
| `LLM_MODEL` | `deepseek-chat` | 模型名称 |
| `EMBEDDING_PROVIDER` | `local` | `local`（sentence-transformers）或 `openai` |
| `VECTOR_STORE` | `faiss` | `faiss` 或 `milvus` |
| `MILVUS_URI` | `http://localhost:19530` | Milvus 地址 |
| `REDIS_URL` | （空） | Redis 地址，留空使用内存缓存 |
| `TOP_K` | `6` | 召回数量 |
| `RERANK_TOP_N` | `3` | 重排序后保留数量 |
| `CACHE_TTL` | `3600` | 热点问答缓存时长（秒） |

## 📁 目录结构

```
├── app/
│   ├── agents/          # 路由 / 检索 / 生成 / 反思 四个 Agent
│   ├── rag/             # 文档解析、切分、向量库、重排序
│   ├── llm/             # OpenAI 兼容 LLM 客户端
│   ├── api/             # FastAPI 路由
│   ├── graph.py         # 多 Agent 状态编排（LangGraph）
│   ├── cache.py         # Redis 热点缓存
│   └── schemas.py       # Pydantic 数据模型
├── scripts/
│   ├── ingest_docs.py   # 文档入库
│   └── eval_retrieval.py# 召回评测
├── data/sample_docs/    # 示例企业知识库文档
├── tests/
├── docker-compose.yml   # Milvus + Redis
└── Dockerfile
```

## 📄 License

[MIT](LICENSE)
