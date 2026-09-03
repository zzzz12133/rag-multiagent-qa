# -*- coding: utf-8 -*-
"""FastAPI 入口"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.rag.vectorstore import load_bm25

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(title="RAG多Agent企业智能问答系统", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    # 预加载 BM25 关键词索引（若已构建）
    try:
        load_bm25()
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).warning("BM25 索引未加载: %s", e)


app.include_router(router)
