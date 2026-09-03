# -*- coding: utf-8 -*-
"""多路召回融合（RRF）与重排序"""
from typing import List, Tuple

from langchain_core.documents import Document


def reciprocal_rank_fusion(
    dense: List[Tuple[Document, float]],
    bm25: List[Tuple[Document, float]],
    k: int = 60,
) -> List[Document]:
    """RRF 融合双路召回结果：score(d) = sum(1 / (k + rank))，按融合分排序。"""
    fused: dict[str, tuple[Document, float]] = {}
    for rank, (doc, _) in enumerate(dense):
        key = _doc_key(doc)
        prev = fused.get(key, (doc, 0.0))
        fused[key] = (doc, prev[1] + 1.0 / (k + rank + 1))
    for rank, (doc, _) in enumerate(bm25):
        key = _doc_key(doc)
        prev = fused.get(key, (doc, 0.0))
        fused[key] = (doc, prev[1] + 1.0 / (k + rank + 1))

    ranked = sorted(fused.values(), key=lambda x: -x[1])
    return [doc for doc, _ in ranked]


def _doc_key(doc: Document) -> str:
    return f"{doc.metadata.get('source', '')}:{doc.metadata.get('chunk_id', id(doc))}"


def rerank(query: str, docs: List[Document], top_n: int) -> List[Document]:
    """重排序：使用交叉编码器打分，取 top_n。

    优先使用 bge-reranker（本地可复现）；未安装时退化为按原始顺序截断。
    """
    if not docs:
        return []
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
        pairs = [(query, d.page_content) for d in docs]
        scores = model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: -x[1])
        return [d for d, _ in ranked[:top_n]]
    except Exception:  # 模型不可用（离线/首次下载失败）时回退
        return docs[:top_n]
