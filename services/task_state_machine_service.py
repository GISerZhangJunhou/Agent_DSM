from __future__ import annotations

import time
from typing import Any

RUNNING_STATUSES = {"queued", "running"}
FINISHED_STATUSES = {"done", "error", "cancelled", "interrupted"}
MODEL_RUNNING_KINDS = {"rfk", "gcp", "mapping", "model", "uncertainty"}


def _get_task(registry: Any, task_id: str | None):
    try:
        return registry.get(task_id) if task_id else None
    except Exception:
        return None


def has_running_compute_task(session: Any, registry: Any, kinds: set[str] | None = None) -> bool:
    kinds = kinds or {"rfk", "gcp"}
    for attr in ["latest_rfk_task_id", "latest_gcp_task_id"]:
        task = _get_task(registry, getattr(session, attr, None))
        if task and getattr(task, "kind", None) in kinds and getattr(task, "status", None) in RUNNING_STATUSES:
            return True
    return False


def running_task_labels(session: Any, registry: Any) -> list[str]:
    labels = []
    mapping = {"rfk": "制图/建模", "gcp": "不确定性分析"}
    for attr in ["latest_rfk_task_id", "latest_gcp_task_id"]:
        task = _get_task(registry, getattr(session, attr, None))
        if task and getattr(task, "status", None) in RUNNING_STATUSES:
            labels.append(mapping.get(getattr(task, "kind", ""), getattr(task, "kind", "任务")))
    return labels


def repair_session_task_state(session: Any, registry: Any) -> dict[str, Any]:
    """Repair stale UI/session flags from the authoritative backend task state.

    This function must be safe to call before every user turn and every poll.
    It never blocks the conversation; it only releases stale running states and
    restores done result paths when possible.
    """
    changed = False
    notes: list[str] = []
    rfk = _get_task(registry, getattr(session, "latest_rfk_task_id", None))
    gcp = _get_task(registry, getattr(session, "latest_gcp_task_id", None))

    if getattr(session, "latest_result_kind", None) == "rfk_running":
        if rfk and getattr(rfk, "status", None) == "done" and isinstance(getattr(rfk, "result_paths", None), dict):
            session.som_map_shown = True
            session.latest_result_kind = "som_map"
            session.rfk_result_paths = dict(rfk.result_paths or {})
            session.last_result_paths = dict(rfk.result_paths or {})
            changed = True
            notes.append("制图任务已完成，已释放运行状态。")
        elif (not rfk) or getattr(rfk, "status", None) not in RUNNING_STATUSES:
            session.latest_result_kind = "som_map" if getattr(session, "rfk_result_paths", {}) else None
            changed = True
            notes.append("已清理残留的制图运行状态。")

    if getattr(session, "latest_result_kind", None) == "gcp_running":
        if gcp and getattr(gcp, "status", None) == "done" and isinstance(getattr(gcp, "result_paths", None), dict):
            session.gcp_shown = True
            session.latest_result_kind = "gcp_map"
            session.gcp_result_paths = dict(gcp.result_paths or {})
            session.last_result_paths = dict(gcp.result_paths or {})
            changed = True
            notes.append("不确定性分析任务已完成，已释放运行状态。")
        elif (not gcp) or getattr(gcp, "status", None) not in RUNNING_STATUSES:
            session.latest_result_kind = "gcp_map" if getattr(session, "gcp_result_paths", {}) else ("som_map" if getattr(session, "rfk_result_paths", {}) else None)
            changed = True
            notes.append("已清理残留的不确定性分析运行状态。")

    if changed:
        try:
            session.last_task_type = "general_chat"
            decision = dict(getattr(session, "last_route_decision", {}) or {})
            decision["state_machine_repair"] = {"time": time.time(), "notes": notes}
            session.last_route_decision = decision
        except Exception:
            pass
    return {"changed": changed, "notes": notes, "running_labels": running_task_labels(session, registry)}


def can_start_compute_task(session: Any, registry: Any) -> tuple[bool, str]:
    repair_session_task_state(session, registry)
    labels = running_task_labels(session, registry)
    if labels:
        return False, "当前仍有任务正在运行：" + "、".join(labels) + "。可以继续普通对话，或先发送“停止当前任务”。"
    return True, ""
