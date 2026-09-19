from __future__ import annotations

"""统一的任务执行控制台日志工具。

目标：
1. 所有关键节点都能在 PyCharm 控制台实时看到。
2. 不打印账号、密码、token、cookie 等敏感信息。
3. 日志格式固定，方便搜索 [TASK-LOG]、[UPLOAD]、[NETWORK]、[MODEL]。
"""

import datetime as _dt
import json
import threading
from typing import Any

_SENSITIVE_KEYS = {"password", "passwd", "pwd", "token", "cookie", "secret", "authorization", "api_key", "private_key"}


def _safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            key_lower = str(k).lower()
            if (key_lower in _SENSITIVE_KEYS) or any(s in key_lower for s in ["password", "passwd", "pwd", "token", "cookie", "secret", "authorization"]):
                out[k] = "***"
            else:
                out[k] = _safe_value(v)
        return out
    if isinstance(value, (list, tuple)):
        return [_safe_value(v) for v in value]
    return value


def pro_console_log(stage: str, message: str, payload: Any | None = None, task_id: str | None = None) -> None:
    """打印一行结构化控制台日志，供 PyCharm 控制台实时检查。"""
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    thread_name = threading.current_thread().name
    prefix = f"[{ts}][TASK-LOG][{stage}]"
    if task_id:
        prefix += f"[task={task_id[:8]}]"
    prefix += f"[thread={thread_name}]"
    line = f"{prefix} {message}"
    if payload is not None:
        try:
            safe_payload = _safe_value(payload)
            line += " | " + json.dumps(safe_payload, ensure_ascii=False, default=str)[:3000]
        except Exception:
            line += " | " + str(payload)[:3000]
    print(line, flush=True)



def emit(stage: str, message: str, payload: Any | None = None, task_id: str | None = None) -> None:
    """Emit a structured log to both the backend console and Dash task logs.

    This is used by long in-process RFK calls.  If a task_id is available, the
    UI progress panel receives the same message as the console.  The lazy import
    avoids a module import cycle with services.task_service.
    """
    if task_id:
        try:
            from services.task_service import TASKS  # lazy import
            line = f"[{stage}] {message}"
            if payload is not None:
                try:
                    brief = json.dumps(_safe_value(payload), ensure_ascii=False, default=str)
                    if len(brief) > 1200:
                        brief = brief[:1200] + "..."
                    line += " | " + brief
                except Exception:
                    line += " | " + str(payload)[:1200]
            TASKS.append_log(task_id, line)
            return
        except Exception:
            pass
    pro_console_log(stage, message, payload=payload, task_id=task_id)
