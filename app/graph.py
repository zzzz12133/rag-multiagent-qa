# -*- coding: utf-8 -*-
"""多 Agent 状态编排（LangGraph StateGraph）

流程：
    START -> 路由 Agent
      ├─ greeting  -> 直接回答 -> END
      └─ knowledge/complex -> 子问题循环
            检索 Agent -> 生成 Agent -> 反思 Agent
              ├─ 不通过且未超2轮 -> 携带修订意见回到生成
              └─ 通过/超限 -> 合并子问题答案 -> END
"""
import logging
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.generation import GenerationAgent
from app.agents.reflection import ReflectionAgent
from app.agents.retrieval import RetrievalAgent
from app.agents.router import RouterAgent
from app.llm.client import invoke_llm

logger = logging.getLogger(__name__)

MAX_REFLECTION_ROUNDS = 2

router_agent = RouterAgent()
retrieval_agent = RetrievalAgent()
generation_agent = GenerationAgent()
reflection_agent = ReflectionAgent()


class AgentState(TypedDict, total=False):
    question: str
    session_id: str
    history: List[Dict[str, str]]
    intent: str
    sub_questions: List[str]
    current_question: str
    sub_answers: List[Dict[str, str]]
    docs: list
    draft_answer: str
    reflection_pass: bool
    reflection_reason: str
    reflection_rounds: int
    answer: str
    trace: List[str]


def route_node(state: AgentState) -> AgentState:
    out = router_agent.run(state)
    if out["intent"] == "greeting":
        out["answer"] = invoke_llm(
            [
                {"role": "system", "content": "你是企业智能问答助手，友好、简洁地回答寒暄。"},
                {"role": "user", "content": state["question"]},
            ]
        )
    return out


def should_retrieve(state: AgentState) -> str:
    if state.get("intent") == "greeting":
        return "end"
    subs = state.get("sub_questions") or []
    if not state.get("current_question") and subs:
        state["current_question"] = subs[0]
    return "retrieval"


def retrieval_node(state: AgentState) -> AgentState:
    return retrieval_agent.run(state)


def generation_node(state: AgentState) -> AgentState:
    return generation_agent.run(state)


def reflection_node(state: AgentState) -> AgentState:
    return reflection_agent.run(state)


def after_reflection(state: AgentState) -> str:
    """反思不通过且未超轮数 → 回生成 Agent 修订；否则进入下一个子问题或结束。"""
    if not state.get("reflection_pass") and state.get("reflection_rounds", 0) < MAX_REFLECTION_ROUNDS:
        return "generation_retry"
    return "next_sub"


def next_sub_node(state: AgentState) -> AgentState:
    """记录当前子问题答案，切换到下一个子问题。"""
    subs = state.get("sub_questions") or []
    answers = list(state.get("sub_answers") or [])
    answers.append(
        {
            "question": state.get("current_question", state["question"]),
            "answer": state.get("draft_answer") or "",
            "docs": state.get("docs") or [],
        }
    )
    idx = len(answers)
    state["sub_answers"] = answers
    state["reflection_rounds"] = 0
    state["reflection_pass"] = False
    if idx < len(subs):
        state["current_question"] = subs[idx]
        state["trace"] = state.get("trace", []) + [f"graph: 处理子问题 {idx + 1}/{len(subs)}"]
    return state


def more_sub(state: AgentState) -> str:
    subs = state.get("sub_questions") or []
    if len(state.get("sub_answers") or []) < len(subs):
        return "retrieval"
    return "merge"


def merge_node(state: AgentState) -> AgentState:
    """融合多个子问题答案，生成最终回答（多路召回与答案融合）。"""
    subs = state.get("sub_questions") or [state["question"]]
    answers = state.get("sub_answers") or []
    if not answers:
        answers = [{"question": state["question"], "answer": state.get("draft_answer") or ""}]
    merged_parts = []
    for a in answers:
        merged_parts.append(f"问：{a['question']}\n答：{a['answer']}")
    prompt = (
        "以下是多个子问题的回答，请整合为一段完整、连贯、无重复的最终答案，"
        "保留 [n] 引用标注：\n\n" + "\n\n".join(merged_parts)
    )
    state["answer"] = invoke_llm(
        [{"role": "system", "content": "你是企业智能问答助手，负责整合答案。"},
         {"role": "user", "content": prompt}]
    )
    state["trace"] = state.get("trace", []) + [f"graph: 融合 {len(answers)} 个子问题答案"]
    return state


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("router", route_node)
    g.add_node("retrieval", retrieval_node)
    g.add_node("generation", generation_node)
    g.add_node("reflection", reflection_node)
    g.add_node("next_sub", next_sub_node)
    g.add_node("merge", merge_node)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", should_retrieve, {"end": END, "retrieval": "retrieval"})
    g.add_edge("retrieval", "generation")
    g.add_edge("generation", "reflection")
    g.add_conditional_edges(
        "reflection",
        after_reflection,
        {"generation_retry": "generation", "next_sub": "next_sub"},
    )
    g.add_conditional_edges("next_sub", more_sub, {"retrieval": "retrieval", "merge": "merge"})
    g.add_edge("merge", END)
    return g.compile()


graph = build_graph()


def run_pipeline(question: str, session_id: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
    """执行完整多 Agent 流水线。"""
    state: AgentState = {
        "question": question,
        "session_id": session_id,
        "history": history,
        "sub_answers": [],
        "trace": [],
    }
    final = graph.invoke(state)
    return final
