# -*- coding: utf-8 -*-
"""召回评测：计算 Recall@K

用法:
    python scripts/eval_retrieval.py
"""
import logging
from typing import Dict, List

from app.agents.retrieval import RetrievalAgent
from app.rag.rerank import reciprocal_rank_fusion, rerank
from app.rag.vectorstore import bm25_search, dense_search
from app.config import settings

logging.basicConfig(level=logging.WARNING)

# 示例标注集：query -> 相关文档名（以 sample_docs 为例）
QRELS: Dict[str, List[str]] = {
    "年假如何申请": ["公司制度手册.md"],
    "病假需要什么材料": ["公司制度手册.md"],
    "如何重置密码": ["产品FAQ.md"],
    "会员有哪些权益": ["产品FAQ.md"],
}


def evaluate(k: int = 3) -> Dict[str, float]:
    recall_sum = 0.0
    total = 0
    for query, relevant_sources in QRELS.items():
        dense = dense_search(query, k=settings.top_k)
        bm25 = bm25_search(query, k=settings.bm25_top_k)
        fused = reciprocal_rank_fusion(dense, bm25)
        ranked = rerank(query, fused, k)
        hit = 0
        for doc in ranked:
            src = doc.metadata.get("source", "")
            if any(r in src for r in relevant_sources):
                hit += 1
        recall = hit / len(relevant_sources) if relevant_sources else 0.0
        recall_sum += recall
        total += 1
        print(f"[{query}] Recall@{k} = {recall:.2f}")
    avg = recall_sum / total if total else 0.0
    print(f"\n平均 Recall@{k} = {avg:.4f}")
    return {"avg_recall": avg}


if __name__ == "__main__":
    evaluate()
