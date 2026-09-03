# -*- coding: utf-8 -*-
"""文档解析：支持 Markdown / TXT / PDF / Word，统一输出 Document 列表"""
import logging
import os
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".md", ".txt", ".pdf", ".docx"}


def _load_single(path: str) -> List[Document]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        from langchain_community.document_loaders import PyMuPDFLoader

        return PyMuPDFLoader(path).load()
    if ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        return Docx2txtLoader(path).load()
    if ext in (".md", ".txt"):
        from langchain_community.document_loaders import TextLoader

        return TextLoader(path, encoding="utf-8").load()
    raise ValueError(f"不支持的文件类型: {ext}")


def load_documents(path: str, recursive: bool = True) -> List[Document]:
    """加载文件或目录，返回带 source 元数据的 Document 列表。"""
    docs: List[Document] = []
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            if not recursive:
                dirs[:] = []
            for name in sorted(files):
                if os.path.splitext(name)[1].lower() in SUPPORTED_EXTS:
                    full = os.path.join(root, name)
                    docs.extend(_load_single(full))
                    logger.info("已解析: %s", full)
    elif os.path.isfile(path):
        docs.extend(_load_single(path))
    else:
        raise FileNotFoundError(f"路径不存在: {path}")

    if not docs:
        raise ValueError(f"未找到可解析的文档: {path}")
    return docs
