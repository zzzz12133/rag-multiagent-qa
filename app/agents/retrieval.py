# -*- coding: utf-8 -*-
"""检索 Agent：多路召回（稠密向量 + BM25 关键词）→ RRF 融合 → Rerank 精排"""
import logging
from typing import Any, Dict, List

from langchain_core.documents import Document

from app.agents.base import BaseAgent
from app.config import settings
from app.rag.rerank import reciprocal_rank_fusion, rerank
from app.rag.vectorstore import bm25_search, dense_search

logger = logging.getLogger(__name__)


class RetrievalAgent(BaseAgent):
    name = "retrieval"

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("current_question") or state["question"]
        trace = state.get("trace", [])

        # 第一路：稠密向量检索
        dense = dense_search(query, k=settings.top_k)
        # 第二路：BM25 关键词检索
        bm25 = bm25_search(query, k=settings.bm25_top_k)

        if not dense and not bm25:
            return {**state, "docs": [], "trace": trace + ["retrieval: 未召回任何文档"]}

        fused = reciprocal_rank_fusion(dense, bm25)
        final = rerank(query, fused, settings.rerank_top_n)

        # 相关度得分取自稠密检索（用于引用展示）
        score_map = {d.metadata.get("chunk_id", i): s for i, (d, s) in enumerate(dense)}
        for i, d in enumerate(final):
            d.metadata["score"] = round(score_map.get(d.metadata.get("chunk_id", i), 0.0), 4)

        trace.append(
            f"retrieval: 向量召回{len(dense)}条 + BM25召回{len(bm25)}条，融合重排后取{len(final)}条"
        )
        return {**state, "docs": final, "trace": trace}
