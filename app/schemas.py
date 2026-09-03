# -*- coding: utf-8 -*-
"""Pydantic 数据模型"""
from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    session_id: str = Field("default", description="会话ID，用于多轮对话")
    use_cache: bool = Field(True, description="是否查询热点缓存")


class Citation(BaseModel):
    source: str = Field(..., description="来源文档名")
    snippet: str = Field(..., description="引用的原文片段")
    score: float = Field(..., description="相关度得分")


class ChatResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation] = []
    agents_trace: List[str] = Field([], description="Agent 执行轨迹，便于调试")
    latency_ms: int = Field(0, description="端到端耗时（毫秒）")
    from_cache: bool = False


class IngestRequest(BaseModel):
    path: str = Field(..., description="文件或目录路径")
    recursive: bool = Field(True, description="目录是否递归")


class IngestResponse(BaseModel):
    ingested_docs: int
    chunks: int
    message: str


class DocumentInfo(BaseModel):
    source: str
    chunks: int


class HealthResponse(BaseModel):
    status: str
    vector_store: str
    embedding_provider: str
