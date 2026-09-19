from __future__ import annotations

import os
import re
from typing import Any

DOMESTIC_KEYWORDS = (
    "国内", "国内平台", "国产", "国家青藏高原", "青藏高原", "TPDC", "tpdc",
    "地理空间数据云", "GSCloud", "gscloud", "地球系统科学数据中心", "资源环境科学与数据中心",
    "中国气象数据网", "国家气象信息中心", "风云", "FTP", "ftp"
)

GEE_KEYWORDS = ()


def env_truth(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def detect_mapping_source(text: str | None) -> str | None:
    """Return 'domestic', 'gee', or None from the user's mapping instruction."""
    s = str(text or "")
    if not s.strip():
        return None
    domestic_score = sum(1 for k in DOMESTIC_KEYWORDS if k in s)
    gee_score = 0
    if re.search(r"(只用|仅用|使用|选择|采用).{0,8}(用户上传|上传数据|已有数据|我的数据|本地数据)", s):
        return "user"
    if re.search(r"(用|使用|选择|采用|走|从).{0,10}(国内|国产|TPDC|青藏高原|地理空间数据云|FTP)", s, re.I):
        return "domestic"
    if domestic_score and not gee_score:
        return "domestic"
    return None


def should_ask_mapping_source(text: str | None, looks_like_mapping: bool) -> bool:
    if not looks_like_mapping:
        return False
    if os.getenv("PRO_MAPPING_SOURCE_CHOICE_MODE", "ask").strip().lower() != "ask":
        return False
    if detect_mapping_source(text):
        return False
    return env_truth("PRO_MAPPING_REQUIRE_SOURCE_CHOICE", "1")


def data_source_for_choice(choice: str | None) -> str:
    c = str(choice or "").strip().lower()
    if c in {"user", "uploaded", "user_only", "只用上传", "用户数据"}:
        return "pro_user"
    if c in {"domestic", "国内", "domestic_platform"}:
        return "pro_domestic"
    if c in {"gee", "google", "earth_engine", "google_earth_engine"}:
        return "pro_domestic"
    return "pro_web"


def choice_from_data_source(data_source: str | None, request_text: str | None = None) -> str:
    ds = str(data_source or "").strip().lower()
    if ds in {"pro_user", "user", "uploaded", "user_only"}:
        return "user"
    if ds in {"pro_domestic", "domestic", "domestic_platform"}:
        return "domestic"
    if ds in {"pro_gee", "gee", "google_earth_engine"}:
        return "domestic"
    return detect_mapping_source(request_text) or os.getenv("PRO_MAPPING_DEFAULT_SOURCE", "user").strip().lower()


def build_source_choice_payload(request_text: str, session_id: str | None = None, task_type: str = "mapping") -> dict[str, Any]:
    is_download = str(task_type or "mapping").lower() == "download"
    return {
        "kind": "mapping_source_choice",
        "task_type": "download_only" if is_download else "mapping",
        "title": "选择下载数据源" if is_download else "选择制图数据源",
        "request_text": request_text,
        "session_id": session_id,
        "message": ("请自主选择本次单数据下载来源。国内数据平台为国家青藏高原科学数据中心（TPDC），免费数据多、来源审计更强，但下载较慢，常需要验证码、人机验证、协议确认或 FTP 账号/密码。若下载到 NetCDF/HDF，多时相数据会自动提取用户指定年份并转换为 GeoTIFF 在地图上预览。" if is_download else "制图前请自主选择本次环境协变量来源：只使用用户上传/本地数据，或使用国家青藏高原科学数据中心（TPDC）补充数据。"),
        "choices": ([{
                "id": "user",
                "title": "只使用用户上传数据",
                "subtitle": "不下载缺失协变量；适合已有充分样点和TIF/NC/HDF协变量数据的基线制图。",
                "platform": "用户上传数据",
            }] if not is_download else []) + [
            {
                "id": "domestic",
                "title": "国内数据平台：国家青藏高原科学数据中心（TPDC）",
                "subtitle": "免费数据较多，适合正式来源审计；可能需要验证码、网页登录、人机验证或 FTP 账号/密码。",
                "platform": "国家青藏高原科学数据中心 TPDC",
            },
        ],
    }
