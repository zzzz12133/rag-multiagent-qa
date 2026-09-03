# -*- coding: utf-8 -*-
"""OpenAI 兼容 LLM 客户端封装（DeepSeek / 通义千问 / OpenAI 均可）"""
import logging
from typing import Any, Dict, Iterator, List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import settings

logger = logging.getLogger(__name__)

_llm: ChatOpenAI | None = None
_embeddings: Any = None


def get_llm() -> ChatOpenAI:
    """获取 Chat LLM 单例（支持流式）。"""
    global _llm
    if _llm is None:
        if not settings.openai_api_key:
            raise RuntimeError("未配置 OPENAI_API_KEY，请先编辑 .env 文件")
        _llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            streaming=True,
        )
    return _llm


def get_embeddings() -> Any:
    """获取嵌入模型单例：local = sentence-transformers，openai = 兼容接口。"""
    global _embeddings
    if _embeddings is None:
        if settings.embedding_provider == "openai":
            if not settings.openai_api_key:
                raise RuntimeError("EMBEDDING_PROVIDER=openai 需要配置 OPENAI_API_KEY")
            from langchain_openai import OpenAIEmbeddings as OE

            _embeddings = OE(
                model=settings.embedding_model,
                api_key=settings.openai_api_key,
                base_url=settings.openai_api_base,
            )
        else:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            logger.info("加载本地嵌入模型: %s（首次运行会自动下载）", settings.embedding_model)
            _embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return _embeddings


def invoke_llm(messages: List[Dict[str, str]]) -> str:
    """同步调用 LLM。"""
    return get_llm().invoke(messages).content


def stream_llm(messages: List[Dict[str, str]]) -> Iterator[str]:
    """流式调用 LLM，逐 token 产出。"""
    for chunk in get_llm().stream(messages):
        if chunk.content:
            yield chunk.content
