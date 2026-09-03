# -*- coding: utf-8 -*-
"""热点问答缓存：优先 Redis，未配置时回退进程内 dict"""
import hashlib
import json
import time
from typing import Any, Dict, Optional

from app.config import settings

_redis = None
_memory: Dict[str, tuple] = {}  # key -> (value, expire_at)


def _get_redis():
    global _redis
    if _redis is not None:
        return _redis
    if settings.redis_url:
        import redis

        _redis = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis


def _normalize(question: str) -> str:
    return hashlib.md5(question.strip().lower().encode("utf-8")).hexdigest()


def get(question: str) -> Optional[Dict[str, Any]]:
    key = "qa:" + _normalize(question)
    r = _get_redis()
    if r:
        raw = r.get(key)
        if raw:
            return json.loads(raw)
        return None
    item = _memory.get(key)
    if item and item[1] > time.time():
        return item[0]
    return None


def set_cache(question: str, payload: Dict[str, Any]) -> None:
    key = "qa:" + _normalize(question)
    r = _get_redis()
    if r:
        r.set(key, json.dumps(payload, ensure_ascii=False), ex=settings.cache_ttl)
    else:
        _memory[key] = (payload, time.time() + settings.cache_ttl)
