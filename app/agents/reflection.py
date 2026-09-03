# -*- coding: utf-8 -*-
"""反思 Agent：校验答案质量（相关性、依据充分性、是否幻觉），不通过时给出修订意见"""
import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.documents import Document

from app.agents.base import BaseAgent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是问答质量评审专家。请评审【答案】是否合格，严格按以下 JSON 输出（不要输出其他内容）：
{"pass": true|false, "reason": "评审意见"}

评审标准：
1. 相关性：答案是否正面回答用户问题
2. 依据充分性：答案是否有【参考资料】支撑，不得凭空编造
3. 准确性：数字、日期、流程等关键信息与资料一致
4. 完整性：涉及多个知识点时是否全部覆盖

任一标准不满足则 pass=false，reason 中给出具体修订意见。"""


class ReflectionAgent(BaseAgent):
    name = "reflection"

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        question = state.get("current_question") or state["question"]
        answer = state.get("draft_answer") or ""
        docs: List[Document] = state.get("docs") or []
        context = "\n\n".join(d.page_content for d in docs[:5]) or "（无参考资料）"

        raw = self._chat(
            f"【用户问题】{question}\n\n【参考资料】\n{context}\n\n【答案】\n{answer}"
        )
        try:
            parsed = json.loads(_extract_json(raw))
            passed = bool(parsed.get("pass", True))
            reason = parsed.get("reason", "")
        except Exception:
            logger.warning("反思 Agent 输出解析失败，默认通过: %s", raw)
            passed, reason = True, ""

        rounds = state.get("reflection_rounds", 0)
        return {
            **state,
            "reflection_pass": passed,
            "reflection_reason": reason,
            "reflection_rounds": rounds + 1,
            "trace": state.get("trace", []) + [f"reflection: {'通过' if passed else '不通过 - ' + reason}"],
        }


def _extract_json(text: str) -> str:
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        return m.group(0)
    raise ValueError("no json")
