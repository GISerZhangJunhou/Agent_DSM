from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict

try:
    from config.settings import TASK_STATE_DIR
except Exception:
    TASK_STATE_DIR = r"E:\\Agent_DSM\\runtime_state"


def _audit_root() -> Path:
    root = Path(TASK_STATE_DIR) / "task_step_audit"
    root.mkdir(parents=True, exist_ok=True)
    return root


def audit_paths(task_id: str) -> Dict[str, str]:
    root = _audit_root()
    safe_id = str(task_id or "unknown").replace("/", "_").replace("\\\\", "_")
    return {
        "step_audit_jsonl": str(root / f"{safe_id}_steps.jsonl"),
        "step_audit_csv": str(root / f"{safe_id}_steps.csv"),
    }


def append_step_record(task_id: str, kind: str, status: str, progress: int | float | None,
                       stage: str | None, message: str | None = None,
                       event: str = "stage", extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    paths = audit_paths(task_id)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        progress_val = int(progress) if progress is not None else None
    except Exception:
        progress_val = None
    rec = {
        "time": ts,
        "task_id": str(task_id or ""),
        "kind": str(kind or ""),
        "event": str(event or "stage"),
        "status": str(status or ""),
        "progress": progress_val,
        "stage": str(stage or ""),
        "message": str(message or ""),
    }
    if extra:
        rec["extra"] = extra
    jsonl = Path(paths["step_audit_jsonl"])
    csvp = Path(paths["step_audit_csv"])
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    write_header = not csvp.exists()
    with csvp.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "task_id", "kind", "event", "status", "progress", "stage", "message", "extra"])
        if write_header:
            writer.writeheader()
        row = dict(rec)
        row["extra"] = json.dumps(extra or {}, ensure_ascii=False)
        writer.writerow(row)
    return rec


def attach_audit_paths(result_paths: Dict[str, Any] | None, task_id: str) -> Dict[str, Any]:
    out = dict(result_paths or {})
    out.update(audit_paths(task_id))
    return out
