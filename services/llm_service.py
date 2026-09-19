from __future__ import annotations

import json
import traceback
from typing import Any, Dict, List

from config.settings import ENABLE_QWEN, DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, QWEN_TEXT_MODEL
from config.prompts import (
    AGENT_PLANNER_SYSTEM_PROMPT,
    AGENT_RESPONSE_SYSTEM_PROMPT,
    KNOWLEDGE_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    WEB_RESEARCH_SUMMARY_PROMPT,
)

_llm = None
_last_llm_error = ""


def _log(msg: str) -> None:
    print(f"[LLM] {msg}")


def get_llm_health() -> Dict[str, Any]:
    return {
        "ENABLE_QWEN": ENABLE_QWEN,
        "api_key_exists": bool(DASHSCOPE_API_KEY),
        "base_url": DASHSCOPE_BASE_URL,
        "model": QWEN_TEXT_MODEL,
        "llm_initialized": _llm is not None,
        "last_error": _last_llm_error,
    }


def _get_llm():
    global _llm, _last_llm_error
    if _llm is not None:
        return _llm
    if not ENABLE_QWEN:
        _last_llm_error = "ENABLE_QWEN=0，LLM 已禁用"
        _log(_last_llm_error)
        return None
    if not DASHSCOPE_API_KEY:
        _last_llm_error = "未读取到 DASHSCOPE_API_KEY / QWEN_API_KEY / OPENAI_API_KEY"
        _log(_last_llm_error)
        return None
    try:
        from langchain_openai import ChatOpenAI
        _llm = ChatOpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL,
            model=QWEN_TEXT_MODEL,
            temperature=0.2,
            timeout=45,
        )
        _last_llm_error = ""
        _log(f"初始化成功 | model={QWEN_TEXT_MODEL} | base_url={DASHSCOPE_BASE_URL}")
        return _llm
    except Exception as e:
        _last_llm_error = f"LLM 初始化失败: {repr(e)}"
        _log(_last_llm_error)
        _log(traceback.format_exc())
        return None


def _history_messages(chat_history: List[Dict[str, str]] | None, limit: int = 20):
    messages = []
    for item in (chat_history or [])[-limit:]:
        role = item.get("role")
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        messages.append(("assistant" if role == "assistant" else "user", content))
    return messages


def _strip_code_fence(content: str) -> str:
    content = str(content or "").strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) >= 2:
            content = "\n".join(lines[1:-1])
    return content.strip()


def _invoke_messages(
    system_prompt: str,
    user_prompt: str,
    chat_history: List[Dict[str, str]] | None = None,
    history_limit: int = 8,
):
    global _last_llm_error
    llm = _get_llm()
    if llm is None:
        _log("invoke skipped：llm is None")
        return None
    try:
        messages = [("system", system_prompt)]
        messages.extend(_history_messages(chat_history, limit=history_limit))
        messages.append(("user", user_prompt))
        _log(f"开始调用 | history={len(messages)-2}条 | user_prompt前80字={user_prompt[:80]!r}")
        msg = llm.invoke(messages)
        content = str(getattr(msg, "content", "")).strip()
        _last_llm_error = ""
        _log(f"调用成功 | reply前120字={content[:120]!r}")
        return content
    except Exception as e:
        _last_llm_error = f"LLM 调用失败: {repr(e)}"
        _log(_last_llm_error)
        _log(traceback.format_exc())
        return None


def route_with_llm_json(
    user_text: str,
    context: Dict[str, Any] | None = None,
    chat_history: List[Dict[str, str]] | None = None,
):
    context = context or {}
    prompt = f"上下文：\n{json.dumps(context, ensure_ascii=False)}\n\n用户当前输入：{user_text}"
    raw = _invoke_messages(ROUTER_SYSTEM_PROMPT, prompt, chat_history=chat_history, history_limit=6)
    if not raw:
        return None
    try:
        return json.loads(_strip_code_fence(raw))
    except Exception as e:
        _log(f"route_with_llm_json JSON 解析失败: {repr(e)} | raw={raw!r}")
        return None


def plan_agent_action(
    user_text: str,
    session_context: Dict[str, Any] | None = None,
    result_context: str = "",
    chat_history: List[Dict[str, str]] | None = None,
):
    payload = {
        "session_context": session_context or {},
        "has_result_context": bool(result_context.strip()),
        "result_context_preview": (result_context or "")[:1200],
        "user_text": user_text,
    }
    raw = _invoke_messages(
        AGENT_PLANNER_SYSTEM_PROMPT,
        f"请根据下面信息规划当前动作，只输出 JSON：\n{json.dumps(payload, ensure_ascii=False)}",
        chat_history=chat_history,
        history_limit=8,
    )
    if not raw:
        return None
    try:
        return json.loads(_strip_code_fence(raw))
    except Exception as e:
        _log(f"plan_agent_action JSON 解析失败: {repr(e)} | raw={raw!r}")
        return None


def answer_knowledge_question(
    user_text: str,
    extra_context: str = "",
    chat_history: List[Dict[str, str]] | None = None,
):
    prompt = user_text
    if extra_context:
        prompt = f"补充上下文：\n{extra_context}\n\n用户当前问题：{user_text}"
    return _invoke_messages(KNOWLEDGE_SYSTEM_PROMPT, prompt, chat_history=chat_history, history_limit=16)


def summarize_web_research(
    user_text: str,
    snippets: list[str],
    extra_context: str = "",
    chat_history: List[Dict[str, str]] | None = None,
):
    joined = "\n\n".join(f"- {s}" for s in snippets[:8])
    prompt = f"用户问题：{user_text}\n\n检索摘要：\n{joined}"
    if extra_context:
        prompt += f"\n\n当前结果背景：\n{extra_context}"
    return _invoke_messages(WEB_RESEARCH_SUMMARY_PROMPT, prompt, chat_history=chat_history, history_limit=6)


def compose_agent_reply(
    user_text: str,
    extra_context: str = "",
    web_snippets: list[str] | None = None,
    result_context: str = "",
    chat_history: List[Dict[str, str]] | None = None,
):
    payload = {
        "user_text": user_text,
        "extra_context": extra_context or "",
        "result_context": result_context or "",
        "web_snippets": (web_snippets or [])[:8],
    }
    prompt = f"请基于以下材料，直接回答用户：\n{json.dumps(payload, ensure_ascii=False)}"
    return _invoke_messages(AGENT_RESPONSE_SYSTEM_PROMPT, prompt, chat_history=chat_history, history_limit=20)
