# -*- coding: utf-8 -*-
"""生成 Agent：基于召回上下文生成带引用的答案，支持流式输出"""
import logging
from typing import Any, Dict, Iterator, List

from langchain_core.documents import Document

from app.agents.base import BaseAgent
from app.llm.client import invoke_llm, stream_llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是企业智能问答助手。请严格依据【参考资料】回答用户问题：
1. 答案必须基于参考资料，不得编造；资料无法回答时明确说明"知识库中未找到相关信息"
2. 回答简洁、结构化，必要时使用分点
3. 在答案末尾以 [1][2] 形式标注引用来源编号（对应资料片段）
4. 若参考资料相互矛盾，说明矛盾点"""


def _build_context(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "未知来源")
        parts.append(f"[{i}] 来源《{source}》\n{d.page_content}")
    return "\n\n".join(parts)


class GenerationAgent(BaseAgent):
    name = "generation"

    def _messages(self, question: str, docs: List[Document], history: List[Dict[str, str]]) -> list:
        context = _build_context(docs)
        user = (
            f"【最近对话】\n{_format_history(history)}\n\n"
            f"【参考资料】\n{context}\n\n"
            f"【用户问题】{question}"
        )
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state.get("current_question") or state["question"]
        docs: List[Document] = state.get("docs") or []
        history = state.get("history") or []
        if not docs:
            answer = invoke_llm(
                [
                    {"role": "system", "content": "你是企业智能问答助手，知识库未命中时如实说明。"},
                    {"role": "user", "content": question},
                ]
            )
        else:
            answer = invoke_llm(self._messages(question, docs, history))
        return {
            **state,
            "draft_answer": answer,
            "trace": state.get("trace", []) + ["generation: 已生成草稿答案"],
        }

    def stream(self, state: Dict[str, Any]) -> Iterator[str]:
        """流式生成，逐 token 产出。"""
        question = state.get("current_question") or state["question"]
        docs = state.get("docs") or []
        history = state.get("history") or []
        if not docs:
            msgs = [
                {"role": "system", "content": "你是企业智能问答助手，知识库未命中时如实说明。"},
                {"role": "user", "content": question},
            ]
        else:
            msgs = self._messages(question, docs, history)
        yield from stream_llm(msgs)


def _format_history(history: List[Dict[str, str]]) -> str:
    if not history:
        return "（无）"
    lines = []
    for h in history[-4:]:
        lines.append(f"问：{h.get('question', '')}\n答：{h.get('answer', '')}")
    return "\n".join(lines)
