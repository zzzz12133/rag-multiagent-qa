# -*- coding: utf-8 -*-
"""接口基础测试（需先启动服务并导入样例文档）"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["vector_store"] in ("faiss", "milvus")


def test_chat_greeting():
    resp = client.post("/api/chat", json={"question": "你好", "session_id": "t1"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data


def test_documents_list():
    resp = client.get("/api/documents")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
