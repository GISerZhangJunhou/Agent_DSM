from __future__ import annotations

import json
from typing import Any, Dict, List


def _safe(v: Any, limit: int = 800) -> Any:
    if isinstance(v, str):
        return v if len(v) <= limit else v[:limit] + "…"
    if isinstance(v, list):
        return [_safe(x, limit=limit) for x in v[:20]]
    if isinstance(v, dict):
        out = {}
        for k, val in list(v.items())[:80]:
            out[str(k)] = _safe(val, limit=limit)
        return out
    return v


def build_agent_memory_payload(session, history_limit: int = 20) -> Dict[str, Any]:
    """Build a compact but explicit memory payload for the AI-first agent.

    This payload is the primary context passed to the LLM. It intentionally includes
    task state, uploaded data state, last outputs, layout state and recent dialogue so
    that follow-up commands such as “继续”, “用刚才的数据”, “把图例移到右下角” and
    “做不确定性分析” can be resolved from context instead of keyword matching.
    """
    try:
        recent_history = session.recent_history(history_limit)
    except Exception:
        recent_history = []
    payload = {
        "session_id": getattr(session, "session_id", None),
        "current_project": getattr(session, "current_project", {}) or {},
        "task_memory": getattr(session, "task_memory", {}) or {},
        "conversation_summary": getattr(session, "conversation_summary", "") or "",
        "active_data_source": getattr(session, "active_data_source", "default"),
        "last_task_type": getattr(session, "last_task_type", None),
        "latest_result_kind": getattr(session, "latest_result_kind", None),
        "pending_mapping_request_text": getattr(session, "pending_mapping_request_text", None),
        "pending_mapping_reason": getattr(session, "pending_mapping_reason", None),
        "uploaded_files": getattr(session, "latest_uploaded_files", []) or [],
        "upload_summary": (getattr(session, "data_role_report", {}) or {}).get("summary") or {},
        "preprocess_status": getattr(session, "preprocess_report", {}) or {},
        "mapping_result_paths": getattr(session, "rfk_result_paths", {}) or {},
        "uncertainty_result_paths": getattr(session, "gcp_result_paths", {}) or {},
        "last_result_paths": getattr(session, "last_result_paths", {}) or {},
        "som_cartography": getattr(session, "som_cartography", {}) or {},
        "gcp_cartography": getattr(session, "gcp_cartography", {}) or {},
        "map_style_prefs": getattr(session, "map_style_prefs", {}) or {},
        "recent_history": recent_history,
    }
    return _safe(payload, limit=1200)


def build_agent_memory_text(session, history_limit: int = 20) -> str:
    payload = build_agent_memory_payload(session, history_limit=history_limit)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def remember_user_turn(session, user_text: str, decision: Dict[str, Any] | None = None) -> None:
    """Update lightweight task memory after every agent decision."""
    memory = dict(getattr(session, "task_memory", {}) or {})
    text = str(user_text or "")
    if text:
        memory["last_user_instruction"] = text
    if decision:
        memory["last_agent_decision"] = _safe(decision, limit=800)
        mode = decision.get("mode") or (decision.get("planner") or {}).get("mode")
        if mode:
            memory["last_decision_mode"] = mode
    try:
        session.task_memory = memory
    except Exception:
        pass
