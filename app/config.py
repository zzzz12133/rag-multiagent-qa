# -*- coding: utf-8 -*-
"""应用配置：从环境变量 / .env 读取"""
import os

from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _get_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


class Settings:
    # LLM（OpenAI 兼容接口）
    openai_api_key: str = _get("OPENAI_API_KEY")
    openai_api_base: str = _get("OPENAI_API_BASE", "https://api.deepseek.com")
    llm_model: str = _get("LLM_MODEL", "deepseek-chat")
    llm_temperature: float = float(_get("LLM_TEMPERATURE", "0.1"))
    llm_max_tokens: int = _get_int("LLM_MAX_TOKENS", 1024)

    # 嵌入模型
    embedding_provider: str = _get("EMBEDDING_PROVIDER", "local")
    embedding_model: str = _get("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

    # 向量库
    vector_store: str = _get("VECTOR_STORE", "faiss")
    milvus_uri: str = _get("MILVUS_URI", "http://localhost:19530")
    milvus_collection: str = _get("MILVUS_COLLECTION", "enterprise_kb")

    # 检索参数
    top_k: int = _get_int("TOP_K", 6)
    rerank_top_n: int = _get_int("RERANK_TOP_N", 3)
    bm25_top_k: int = _get_int("BM25_TOP_K", 6)

    # 缓存
    redis_url: str = _get("REDIS_URL")
    cache_ttl: int = _get_int("CACHE_TTL", 3600)

    # 文档切分
    chunk_size: int = _get_int("CHUNK_SIZE", 500)
    chunk_overlap: int = _get_int("CHUNK_OVERLAP", 80)

    # 本地索引路径
    faiss_index_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        "data", "faiss_index")


settings = Settings()
