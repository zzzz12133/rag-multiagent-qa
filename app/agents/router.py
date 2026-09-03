# -*- coding: utf-8 -*-
"""路由 Agent：对用户问题分类，决定处理路径，复杂问题拆解为子问题"""
import json
import logging
import re
from typing import Any, Dict, List

from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)

# 简单寒暄/闲聊直接回答，不消耗检索
GREETING_PATTERNS = [
    r"^(你好|您好|hi|hello|在吗|谢谢|再见|bye)",
    r"^(你是谁|你能做什么|介绍一下你自己)",
]


class RouterAgent(BaseAgent):
    name = "router"

    system_prompt = """你是企业智能问答系统的路由 Agent，负责对用户问题分类与拆解。
请严格按照以下 JSON 格式输出（不要输出其他内容）：
{"intent": "greeting|knowledge|complex", "sub_questions": ["子问题1", "子问题2"], "reason": "简要说明"}

分类规则：
- greeting：寒暄、闲聊、与业务无关的问题（无需检索）
- knowledge：单步即可回答的企业知识问题（需要检索）
- complex：需要多步推理或多知识点融合的问题，sub_questions 列出 2-4 个拆解后的子问题

示例：
用户问题：我们公司的年假政策是什么？
输出：{"intent": "knowledge", "sub_questions": ["我们公司的年假政策是什么"], "reason": "单一知识检索"}

用户问题：实习生加班了能调休吗，调休和年假可以连休吗？
输出：{"intent": "complex", "sub_questions": ["实习生加班是否可以调休", "调休是否可以与年假连休"], "reason": "涉及两个政策点"}"""

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state["question"]
        for pattern in GREETING_PATTERNS:
            if re.search(pattern, question.strip().lower()):
                return {
                    "intent": "greeting",
                    "sub_questions": [],
                    "trace": state.get("trace", []) + ["router: 寒暄问题，直接回答"],
                }
        raw = self._chat(f"用户问题：{question}")
        try:
            parsed = json.loads(_extract_json(raw))
            intent = parsed.get("intent", "knowledge")
            subs = parsed.get("sub_questions") or [question]
            if intent == "complex" and len(subs) < 2:
                intent = "knowledge"
                subs = [question]
        except Exception:
            logger.warning("路由 Agent 输出解析失败，回退为 knowledge: %s", raw)
            intent, subs = "knowledge", [question]
        return {
            "intent": intent,
            "sub_questions": subs,
            "trace": state.get("trace", []) + [f"router: {intent}，子问题 {subs}"],
        }


def _extract_json(text: str) -> str:
    """从 LLM 输出中提取 JSON 片段。"""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        return m.group(0)
    raise ValueError("no json")
