from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from config.settings import DATA_DIR
from services.vip_service import request_target_summary
from services.pro_public_data_engine import run_public_data_test
from utils.pro_console import pro_console_log

PRO_DATA_DIR = DATA_DIR / "pro_online_data"
PRO_USAGE_PATH = DATA_DIR / "pro_usage" / "usage.json"

PRO_DATA_STEPS = [
    "解析目标年份与行政区范围",
    "检查用户上传 SOM 样点字段（不联网爬取样点）",
    "检索/下载公开行政区划、统计年鉴与农业管理代理变量",
    "检索 NODA 等国内公开遥感/地球观测数据入口",
    "检索农业农村部等农业统计公开页面",
    "完成临时缓存、数据质量检查与 provenance 记录",
]


def _developer_credentials_status() -> dict[str, bool]:
    """只暴露是否配置，不返回任何密钥值。"""
    return {
        "gee_service_account": bool(os.getenv("GEE_SERVICE_ACCOUNT") or os.getenv("EE_SERVICE_ACCOUNT")),
        "gee_private_key_file": bool(os.getenv("GEE_PRIVATE_KEY_FILE") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")),
        "nasa_token": bool(os.getenv("NASA_EARTHDATA_TOKEN") or os.getenv("EARTHDATA_TOKEN")),
        "noda_account": bool(os.getenv("PRO_NODA_EMAIL") and os.getenv("PRO_NODA_PASSWORD")),
        "applicant_profile": bool(os.getenv("PRO_APPLICANT_NAME") and os.getenv("PRO_APPLICANT_PURPOSE")),
        "qq_mail_imap": bool(os.getenv("QQ_MAIL_ADDRESS") and os.getenv("QQ_MAIL_AUTH_CODE")),
        "sms_email_forward": (os.getenv("VERIFICATION_SMS_MODE", "").lower() == "email_forward" and bool(os.getenv("QQ_MAIL_ADDRESS") and os.getenv("QQ_MAIL_AUTH_CODE"))),
        "openai_or_qwen_key": bool(os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")),
    }


def _append_usage_record(record: dict[str, Any]) -> None:
    PRO_USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(PRO_USAGE_PATH.read_text(encoding="utf-8")) if PRO_USAGE_PATH.exists() else {"records": []}
        if not isinstance(data, dict):
            data = {"records": []}
        data.setdefault("records", []).append(record)
        # 防止任务记录文件无限变大，只保留最近 300 条。
        data["records"] = data["records"][-300:]
        PRO_USAGE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def _choose_uploaded_sample_path(uploaded_files: list[dict] | None) -> str | None:
    """从当前会话上传文件中选择最像 SOM 样点/标签数据的 CSV/Excel/TIF。"""
    if not uploaded_files:
        return None
    priority_suffixes = [".csv", ".xlsx", ".xls", ".tif", ".tiff"]
    for suffix in priority_suffixes:
        for item in uploaded_files:
            name = str(item.get("name") or "").lower()
            path = str(item.get("path") or "")
            if name.endswith(suffix) and path:
                return path
    return None


def prepare_online_data_manifest(
    session_id: str | None,
    user_text: str,
    root: Path | None = None,
    user_id: str | None = None,
    uploaded_files: list[dict] | None = None,
    sample_path: str | None = None,
) -> dict[str, Any]:
    """
    Pro 自动数据准备。

    默认 real：只生成数据需求清单。
    public_test：真实访问公开页面、下载可公开获取的统计/协变量数据，并生成可用性报告。

    注意：本函数不联网爬取 SOM 样点；样点由用户上传。验证码/人机验证/审核型下载均保留人工介入。
    """
    PRO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = request_target_summary(user_text)
    safe_session = (session_id or "default").replace("/", "_")
    ts = int(time.time())
    out_dir = (root or PRO_DATA_DIR) / safe_session / str(ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    online_mode = (os.getenv("PRO_ONLINE_DATA_MODE") or "real").strip().lower()
    credential_status = _developer_credentials_status()
    resolved_sample_path = sample_path or _choose_uploaded_sample_path(uploaded_files) or os.getenv("PRO_TEST_SAMPLE_PATH") or None
    pro_console_log("PRO_SERVICE", "开始 PRO 在线数据准备", {
        "session_id": session_id,
        "user_id": user_id,
        "online_mode": online_mode,
        "target": target,
        "uploaded_file_count": len(uploaded_files or []),
        "resolved_sample_path": resolved_sample_path,
        "credential_status": credential_status,
    })

    if online_mode in {"public_test", "test", "real"}:
        test_result = run_public_data_test(out_dir, user_text, sample_path=resolved_sample_path)
        payload = {
            "created_at": ts,
            "session_id": session_id,
            "user_id": user_id,
            "target": target,
            "status": test_result.status,
            "mode": "pro_public_data_test",
            "online_mode": online_mode,
            "credential_status": credential_status,
            "policy_note": "当前版本不爬取SOM样点；只下载/解析公开协变量、统计年鉴和公开页面。邮箱/短信转发验证码可通过QQ邮箱IMAP辅助读取；滑块/图形验证码和审核仍需人工介入。",
            "data_sources": test_result.records,
            "steps": PRO_DATA_STEPS,
            "warnings": test_result.warnings,
            "sample_check": test_result.sample_check,
            "uploaded_files": uploaded_files or [],
            "resolved_sample_path": resolved_sample_path,
            "usable_outputs": test_result.usable_outputs,
            "outputs": {
                "manifest": test_result.manifest,
                "report_md": test_result.report_md,
                "work_dir": test_result.work_dir,
            },
        }
        # 同步写一份顶层 manifest，方便旧 UI 仍能找到。
        (out_dir / "pro_online_data_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["outputs"]["legacy_manifest"] = str(out_dir / "pro_online_data_manifest.json")
        pro_console_log("PRO_SERVICE", "PRO 公开数据处理完成", {"status": test_result.status, "manifest": test_result.manifest, "report_md": test_result.report_md, "records": len(test_result.records), "warnings": test_result.warnings})
        _append_usage_record({
            "created_at": ts,
            "session_id": session_id,
            "user_id": user_id,
            "target": target,
            "manifest": payload["outputs"].get("manifest"),
            "report_md": payload["outputs"].get("report_md"),
            "online_mode": online_mode,
            "status": test_result.status,
        })
        return payload

    payload = {
        "created_at": ts,
        "session_id": session_id,
        "user_id": user_id,
        "target": target,
        "status": "prepared",
        "mode": "pro_online_data_real",
        "online_mode": online_mode,
        "credential_status": credential_status,
        "price_note": "Pro 版自动数据准备；设置 PRO_ONLINE_DATA_MODE=public_test 可启用公开数据下载流程。",
        "policy_note": "不爬取SOM样点；正式制图前需要用户上传样点。",
        "uploaded_files": uploaded_files or [],
        "resolved_sample_path": resolved_sample_path,
        "data_sources": [
            {"name": "用户上传SOM样点", "type": "sample", "status": "required_user_upload"},
            {"name": "四川统计年鉴/地方统计局", "type": "management_proxy", "status": "queued_or_mocked"},
            {"name": "NODA公开遥感/地球观测数据", "type": "remote_sensing", "status": "queued_or_mocked"},
            {"name": "农业农村部公开统计资料", "type": "agriculture_statistics", "status": "queued_or_mocked"},
            {"name": "内置分省DEM与地形因子", "type": "terrain", "status": "internal"},
        ],
        "steps": PRO_DATA_STEPS,
        "outputs": {
            "manifest": str(out_dir / "pro_online_data_manifest.json"),
            "work_dir": str(out_dir),
        },
    }
    (out_dir / "pro_online_data_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pro_console_log("PRO_SERVICE", "PRO real manifest 已生成", {"manifest": payload["outputs"].get("manifest"), "online_mode": online_mode, "target": target})
    _append_usage_record({
        "created_at": ts,
        "session_id": session_id,
        "user_id": user_id,
        "target": target,
        "manifest": payload["outputs"]["manifest"],
        "online_mode": online_mode,
    })
    return payload


def list_recent_pro_usage(user_id: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
    try:
        data = json.loads(PRO_USAGE_PATH.read_text(encoding="utf-8")) if PRO_USAGE_PATH.exists() else {"records": []}
        rows = data.get("records", []) if isinstance(data, dict) else []
        if user_id:
            rows = [r for r in rows if r.get("user_id") == user_id]
        return list(reversed(rows[-limit:]))
    except Exception:
        return []
