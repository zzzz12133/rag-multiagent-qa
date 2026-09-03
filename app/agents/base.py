# -*- coding: utf-8 -*-
"""Agent 基类"""
from typing import Any, Dict

from app.llm.client import invoke_llm


class BaseAgent:
    """基于 LLM 的 Agent：通过 JSON 输出完成结构化决策。"""

    name: str = "base"
    system_prompt: str = ""

    def _chat(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return invoke_llm(messages)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
