# -*- coding: utf-8 -*-
"""FastAPI 路由"""
import time
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app import cache
from app.agents.generation import GenerationAgent
from app.graph import run_pipeline
from app.rag import loader, splitter, vectorstore
from app.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    DocumentInfo,
    HealthResponse,
    IngestRequest,
    IngestResponse,
)
from app.config import settings

router = APIRouter(prefix="/api")
generation_agent = GenerationAgent()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    start = time.time()

    # 1. 热点缓存命中直接返回
    if req.use_cache:
        cached = cache.get(req.question)
        if cached:
            cached["from_cache"] = True
            cached["latency_ms"] = int((time.time() - start) * 1000)
            return ChatResponse(**cached)

    # 2. 多 Agent 流水线
    final = run_pipeline(req.question, req.session_id, history=[])

    citations = []
    docs = final.get("sub_answers") or []
    for sa in docs:
        for d in sa.get("docs", []):
            citations.append(
                Citation(
                    source=d.metadata.get("source", "未知来源"),
                    snippet=d.page_content[:120],
                    score=d.metadata.get("score", 0.0),
                )
            )
    # 去重
    seen = set()
    unique = []
    for c in citations:
        if (c.source, c.snippet) not in seen:
            seen.add((c.source, c.snippet))
            unique.append(c)

    resp = ChatResponse(
        question=req.question,
        answer=final.get("answer") or final.get("draft_answer") or "",
        citations=unique[: settings.rerank_top_n * 2],
        agents_trace=final.get("trace", []),
        latency_ms=int((time.time() - start) * 1000),
    )
    cache.set_cache(req.question, resp.model_dump())
    return resp


@router.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """SSE 流式输出。"""

    def gen():
        final = run_pipeline(req.question, req.session_id, history=[])
        docs = final.get("docs") or []
        yield "data: " + _sse({"type": "meta", "trace": final.get("trace", [])}) + "\n\n"
        for token in generation_agent.stream({**final, "docs": docs}):
            yield "data: " + _sse({"type": "token", "content": token}) + "\n\n"
        yield "data: " + _sse({"type": "done"}) + "\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/documents/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    docs = loader.load_documents(req.path, req.recursive)
    chunks = splitter.split_documents(docs)
    vectorstore.build_vector_store(chunks)
    vectorstore.build_bm25(chunks)
    return IngestResponse(ingested_docs=len(docs), chunks=len(chunks), message="入库完成")


@router.get("/documents", response_model=List[DocumentInfo])
def list_documents() -> List[DocumentInfo]:
    from collections import Counter

    store = vectorstore.get_vector_store()
    if store is None:
        return []
    counter = Counter()
    for d in _iter_store_docs(store):
        counter[d.metadata.get("source", "未知")] += 1
    return [DocumentInfo(source=k, chunks=v) for k, v in counter.items()]


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        vector_store=settings.vector_store,
        embedding_provider=settings.embedding_provider,
    )


def _sse(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


def _iter_store_docs(store):
    """遍历向量库中的文档（FAISS / Milvus）。"""
    try:
        from langchain_community.vectorstores import FAISS

        if isinstance(store, FAISS):
            return store.docstore._dict.values()
    except Exception:
        pass
    return []
