# -*- coding: utf-8 -*-
"""向量库与关键词索引：
- 稠密向量检索：FAISS（默认，本地）或 Milvus（生产）
- 关键词检索：BM25（多路召回的第二路）
"""
import logging
import os
import pickle
from typing import List, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.config import settings
from app.llm.client import get_embeddings

logger = logging.getLogger(__name__)

_store = None
_bm25 = None
_bm25_docs: List[Document] = []


# ---------------- 向量库 ----------------

def _build_faiss(docs: List[Document]):
    from langchain_community.vectorstores import FAISS

    return FAISS.from_documents(docs, get_embeddings())


def _build_milvus(docs: List[Document]):
    from langchain_community.vectorstores import Milvus

    return Milvus.from_documents(
        docs,
        get_embeddings(),
        collection_name=settings.milvus_collection,
        connection_args={"uri": settings.milvus_uri},
        drop_old=False,
    )


def get_vector_store():
    """获取向量库单例。"""
    global _store
    if _store is not None:
        return _store
    if settings.vector_store == "milvus":
        from langchain_community.vectorstores import Milvus

        _store = Milvus(
            embedding_function=get_embeddings(),
            collection_name=settings.milvus_collection,
            connection_args={"uri": settings.milvus_uri},
        )
    else:
        if not os.path.exists(settings.faiss_index_dir):
            logger.warning("本地 FAISS 索引为空，请先运行 scripts/ingest_docs.py 导入文档")
            _store = None
            return None
        from langchain_community.vectorstores import FAISS

        _store = FAISS.load_local(
            settings.faiss_index_dir, get_embeddings(), allow_dangerous_deserialization=True
        )
    return _store


def build_vector_store(docs: List[Document]):
    """（重新）构建向量库。"""
    global _store
    if settings.vector_store == "milvus":
        _store = _build_milvus(docs)
    else:
        os.makedirs(settings.faiss_index_dir, exist_ok=True)
        _store = _build_faiss(docs)
        _store.save_local(settings.faiss_index_dir)
    logger.info("向量库构建完成: %s", settings.vector_store)
    return _store


def dense_search(query: str, k: int) -> List[Tuple[Document, float]]:
    """稠密向量检索，返回 (文档, 相似度得分)。"""
    store = get_vector_store()
    if store is None:
        return []
    return store.similarity_search_with_score(query, k=k)


# ---------------- BM25 关键词索引 ----------------

def build_bm25(docs: List[Document]):
    """构建 BM25 关键词索引。"""
    global _bm25, _bm25_docs
    tokenized = [_tokenize(d.page_content) for d in docs]
    _bm25 = BM25Okapi(tokenized)
    _bm25_docs = docs
    # 持久化到磁盘，供服务重启后加载
    os.makedirs(os.path.dirname(_bm25_path()), exist_ok=True)
    with open(_bm25_path(), "wb") as f:
        pickle.dump((tokenized, docs), f)
    logger.info("BM25 索引构建完成: %d 个文档", len(docs))


def _tokenize(text: str) -> List[str]:
    # 中文按单字 + 英文按词切分
    import re

    tokens = re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text.lower())
    return tokens


def _bm25_path() -> str:
    return os.path.join(settings.faiss_index_dir, "bm25.pkl")


def load_bm25():
    """从磁盘加载 BM25 索引。"""
    global _bm25, _bm25_docs
    path = _bm25_path()
    if os.path.exists(path):
        with open(path, "rb") as f:
            tokenized, docs = pickle.load(f)
        _bm25 = BM25Okapi(tokenized)
        _bm25_docs = docs


def bm25_search(query: str, k: int) -> List[Tuple[Document, float]]:
    """BM25 关键词检索，返回 (文档, 得分)。"""
    global _bm25, _bm25_docs
    if _bm25 is None:
        load_bm25()
    if _bm25 is None:
        return []
    scores = _bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [(_bm25_docs[i], float(scores[i])) for i in ranked if scores[i] > 0]
