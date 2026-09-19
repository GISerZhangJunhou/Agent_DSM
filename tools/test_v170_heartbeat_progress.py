"""Quick smoke test for V170 backend progress heartbeat and RFK progress parsing.
Run from work_progress:
    python tools/test_v170_heartbeat_progress.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("PRO_TASK_HEARTBEAT_SEC", "10")

from services.task_service import TASKS, _formal_rfk_progress_from_log_line, _task_heartbeat
from utils.pro_console import emit


def main():
    task = TASKS.create("rfk", session_id="v170_test")
    TASKS.update(task.task_id, status="running", progress=48, stage="生成建模样本表")
    emit("MODEL", "V170 Clean RFK 正在评分候选参数 6/24", {"candidate_id": 6}, task_id=task.task_id)
    t = TASKS.get(task.task_id)
    assert t and t.progress >= 59, (t.progress, t.stage)
    parsed = _formal_rfk_progress_from_log_line("[MODEL] V170 可出图特征RFK候选参数 12/24 完成", t.progress)
    assert parsed and parsed[0] >= 76, parsed
    with _task_heartbeat(task.task_id, label="测试任务", interval_sec=10):
        time.sleep(0.2)
    TASKS.update(task.task_id, status="done", progress=100, stage="测试完成")
    print("V170 heartbeat/progress smoke test OK")


if __name__ == "__main__":
    main()
