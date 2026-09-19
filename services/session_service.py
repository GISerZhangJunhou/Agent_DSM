import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from config.settings import BROWSER_HEARTBEAT_TIMEOUT_SEC, SESSION_DIR


ACTIVE_SESSION_DIR = Path(SESSION_DIR) / "active"
ARCHIVE_SESSION_DIR = Path(SESSION_DIR) / "archive"
ACTIVE_SESSION_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_SESSION_DIR.mkdir(parents=True, exist_ok=True)


class SessionStore:
    def __init__(self):
        self._lock = threading.Lock()

    def _active_path(self, session_id: str) -> Path:
        return ACTIVE_SESSION_DIR / f"{session_id}.json"

    def save_active(self, session_data: Dict[str, Any]) -> None:
        session_id = (session_data or {}).get("session_id")
        if not session_id:
            return
        payload = {
            "session_id": session_id,
            "updated_at": time.time(),
            "session_data": session_data,
        }
        with self._lock:
            self._active_path(session_id).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def read_active(self, session_id: str) -> Dict[str, Any] | None:
        if not session_id:
            return None
        path = self._active_path(session_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def delete_active(self, session_id: str) -> None:
        if not session_id:
            return
        path = self._active_path(session_id)
        with self._lock:
            if path.exists():
                try:
                    path.unlink()
                except Exception:
                    pass

    def archive_active(self, session_id: str, reason: str = "browser_closed", client_id: str | None = None) -> Dict[str, Any] | None:
        active = self.read_active(session_id)
        if not active:
            return None
        session_data = active.get("session_data") or {}
        history = session_data.get("chat_history") or []
        if not history and not session_data.get("latest_result_kind"):
            self.delete_active(session_id)
            return None
        now = time.time()
        archive_id = f"archive-{int(now * 1000)}-{uuid.uuid4().hex[:8]}"
        record = {
            "archive_id": archive_id,
            "session_id": session_id,
            "title": self._build_title(session_data),
            "summary": self._build_summary(session_data),
            "turn_count": len(history),
            "archived_at": now,
            "updated_at": active.get("updated_at", now),
            "reason": reason,
            "client_id": client_id,
            "session_data": session_data,
        }
        out = ARCHIVE_SESSION_DIR / f"{archive_id}.json"
        with self._lock:
            out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        self.delete_active(session_id)
        return record

    def list_archives(self, limit: int = 200) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for path in ARCHIVE_SESSION_DIR.glob("*.json"):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                items.append({
                    "archive_id": raw.get("archive_id"),
                    "title": raw.get("title") or "未命名记录",
                    "summary": raw.get("summary") or "",
                    "turn_count": raw.get("turn_count") or 0,
                    "archived_at": raw.get("archived_at") or 0,
                    "reason": raw.get("reason") or "",
                })
            except Exception:
                continue
        items.sort(key=lambda x: x.get("archived_at") or 0, reverse=True)
        return items[:limit]

    def restore_archive(self, archive_id: str) -> Dict[str, Any] | None:
        if not archive_id:
            return None
        path = ARCHIVE_SESSION_DIR / f"{archive_id}.json"
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            session_data = raw.get("session_data") or {}
        except Exception:
            return None
        restored = dict(session_data)
        restored["session_id"] = str(uuid.uuid4())
        restored["latest_rfk_task_id"] = None
        restored["latest_gcp_task_id"] = None
        if restored.get("latest_result_kind") in {"rfk_running", "gcp_running"}:
            restored["latest_result_kind"] = None
        restored["created_at"] = time.time()
        history = list(restored.get("chat_history") or [])
        history.append({
            "role": "assistant",
            "content": f"已从历史记录“{raw.get('title') or '未命名会话'}”恢复。你可以继续追问，或直接发起新的制图任务。",
        })
        # 保留完整对话历史，不能只恢复最近几条；AI 记忆和用户回看都依赖完整上下文。
        restored["chat_history"] = history
        return restored

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        text = (text or "").strip().replace("\n", " ")
        if len(text) <= limit:
            return text
        return text[: limit - 1] + "…"

    def _build_title(self, session_data: Dict[str, Any]) -> str:
        explicit = (session_data or {}).get("session_title")
        if explicit:
            return self._truncate(str(explicit), 22)
        history = session_data.get("chat_history") or []
        # Prefer the first meaningful user instruction; this behaves like an AI chat title
        # instead of forcing the user to name the conversation manually.
        for item in history:
            if item.get("role") == "user" and item.get("content"):
                text = str(item.get("content") or "").strip()
                if text:
                    return self._truncate(text, 22)
        role_report = session_data.get("data_role_report") or {}
        summary = role_report.get("summary") or {}
        try:
            sample_count = int(summary.get("sample_count") or 0)
            cov_count = int(summary.get("model_covariate_count") or 0)
            file_count = int(summary.get("file_count") or 0)
            if sample_count and cov_count:
                return "上传数据审查与制图准备"
            if cov_count:
                return "环境协变量上传检查"
            if sample_count:
                return "采样点数据上传检查"
            if file_count:
                return "上传数据检查"
        except Exception:
            pass
        kind = session_data.get("latest_result_kind")
        if kind == "som_map":
            return "土壤有机质制图结果"
        if kind == "gcp_map":
            return "不确定性分析结果"
        return "新对话"

    def _build_summary(self, session_data: Dict[str, Any]) -> str:
        history = session_data.get("chat_history") or []
        user_tail = ""
        assistant_tail = ""
        for item in reversed(history):
            if not user_tail and item.get("role") == "user":
                user_tail = item.get("content") or ""
            if not assistant_tail and item.get("role") == "assistant":
                assistant_tail = item.get("content") or ""
            if user_tail and assistant_tail:
                break
        if user_tail and assistant_tail:
            return f"用户：{self._truncate(user_tail, 28)} | 助手：{self._truncate(assistant_tail, 40)}"
        if assistant_tail:
            return self._truncate(assistant_tail, 72)
        if user_tail:
            return self._truncate(user_tail, 72)
        return "无摘要"


class BrowserSessionLifecycle:
    def __init__(self, store: SessionStore):
        self.store = store
        self._lock = threading.Lock()
        self._clients: Dict[str, Dict[str, Any]] = {}
        self._monitor_started = False

    def heartbeat(self, client_id: str | None, session_id: str | None) -> None:
        self.cleanup_stale()
        if not client_id or not session_id:
            return
        with self._lock:
            self._clients[client_id] = {
                "session_id": session_id,
                "last_seen": time.time(),
                "shutdown": False,
            }

    def shutdown(self, client_id: str | None, session_id: str | None) -> None:
        """
        不再收到前端 shutdown_intent 就立即终止任务。
        浏览器真正关闭时，heartbeat 会自然停止；
        后续统一由 heartbeat_timeout 回收。
        """
        if not client_id:
            return
        with self._lock:
            entry = self._clients.get(client_id, {})
            self._clients[client_id] = {
                "session_id": session_id or entry.get("session_id"),
                "last_seen": time.time(),
                "shutdown": False,
            }

    def cleanup_stale(self) -> None:
        now = time.time()
        stale: List[tuple[str, str]] = []
        with self._lock:
            for client_id, meta in list(self._clients.items()):
                if meta.get("shutdown"):
                    stale.append((client_id, meta.get("session_id")))
                    continue
                if now - float(meta.get("last_seen") or 0) > BROWSER_HEARTBEAT_TIMEOUT_SEC:
                    stale.append((client_id, meta.get("session_id")))
            for client_id, _sid in stale:
                self._clients.pop(client_id, None)
        for client_id, sid in stale:
            if sid:
                self._close_session(sid, client_id=client_id, reason="heartbeat_timeout")

    def _close_session(self, session_id: str, client_id: str | None, reason: str) -> None:
        # V194: a missed heartbeat must not be treated as an explicit user close
        # while RFK/GCP tasks are still queued or running.  Long LLM calls or heavy
        # raster operations can briefly delay frontend callbacks; the agent should
        # remain available for continuous conversation unless the user actually
        # leaves the page for a sustained period and no backend task is active.
        try:
            from services.task_service import TASKS
            with TASKS._lock:
                active = [
                    t for t in TASKS._tasks.values()
                    if t.session_id == session_id and t.status in {"queued", "running"}
                ]
            if active:
                with self._lock:
                    self._clients[client_id or f"session-{session_id}"] = {
                        "session_id": session_id,
                        "last_seen": time.time(),
                        "shutdown": False,
                    }
                return
        except Exception:
            pass
        # Do not terminate in-process Python tasks on heartbeat_timeout.  Archiving
        # preserves the conversation for restoration; explicit interpreter/process
        # shutdown still ends tasks naturally.
        if reason != "heartbeat_timeout":
            try:
                from services.task_service import terminate_session_tasks
                terminate_session_tasks(session_id, reason="网页已关闭，本次会话对应后台任务已终止")
            except Exception:
                pass
        self.store.archive_active(session_id, reason=reason, client_id=client_id)

    def start_monitor(self) -> None:
        with self._lock:
            if self._monitor_started:
                return
            self._monitor_started = True

        def _loop():
            while True:
                time.sleep(5)
                try:
                    self.cleanup_stale()
                except Exception:
                    pass

        t = threading.Thread(target=_loop, daemon=True)
        t.start()


SESSION_STORE = SessionStore()
BROWSER_SESSIONS = BrowserSessionLifecycle(SESSION_STORE)
