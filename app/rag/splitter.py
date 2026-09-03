# -*- coding: utf-8 -*-
"""文本切分"""
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )


def split_documents(docs: List[Document]) -> List[Document]:
    """切分文档并保留来源元数据。"""
    chunks = get_splitter().split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks
