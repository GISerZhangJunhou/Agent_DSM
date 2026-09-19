from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

APP_TITLE = "数字土壤制图智能体"
APP_SUBTITLE = "正式制图、模型优化与不确定性分析"

BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=False)

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
SESSION_DIR = DATA_DIR / "sessions"
RESULTS_DIR = DATA_DIR / "results"
TASK_STATE_DIR = DATA_DIR / "task_registry"
EXPORT_DIR = DATA_DIR / "exports"

for _p in [UPLOAD_DIR, SESSION_DIR, RESULTS_DIR, TASK_STATE_DIR, EXPORT_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

PYTHON_EXECUTABLE = os.getenv("PYTHON_EXECUTABLE", sys.executable)

ENABLE_WEB_RESEARCH = os.getenv("ENABLE_WEB_RESEARCH", "1") == "1"
ENABLE_QWEN = os.getenv("ENABLE_QWEN", "1") == "1"
def _clean_api_key(value: str | None) -> str:
    v = str(value or "").strip()
    bad_tokens = {"", "your-key-here", "your_api_key_here", "填写你的key", "填入你的key", "请填写", "YOUR_DASHSCOPE_API_KEY"}
    if v in bad_tokens or "请" in v or "填写" in v or v.lower().startswith("your_"):
        return ""
    return v

DASHSCOPE_API_KEY = _clean_api_key(
    os.getenv("DASHSCOPE_API_KEY")
    or os.getenv("QWEN_API_KEY")
    or os.getenv("OPENAI_API_KEY", "")
)
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
QWEN_TEXT_MODEL = os.getenv("QWEN_TEXT_MODEL", "qwen-plus")

CHAT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", "1000"))
SEARCH_RESULT_LIMIT = 5
TASK_LOG_LIMIT = 400
PERSIST_TASKS_TO_JSON = False
BROWSER_HEARTBEAT_TIMEOUT_SEC = int(os.getenv("BROWSER_HEARTBEAT_TIMEOUT_SEC", "1800"))

DEFAULT_USER_MODE = "ordinary"
SHOW_MODEL_ONLY_WHEN_ASKED = True
REQUIRE_EXPLICIT_REQUEST_FOR_MAP = True
REQUIRE_EXPLICIT_REQUEST_FOR_GCP = True

ENABLE_CUSTOM_USER_RUN = False
USER_RFK_COMMAND_TEMPLATE = None
USER_GCP_COMMAND_TEMPLATE = None


ENABLE_QGIS_STYLE_BRIDGE = os.getenv("ENABLE_QGIS_STYLE_BRIDGE", "1") == "1"
QGIS_BRIDGE_MODE = os.getenv("QGIS_BRIDGE_MODE", "external").strip().lower()
QGIS_PREFIX_PATH = os.getenv("QGIS_PREFIX_PATH", "")
QGIS_BIN_PATH = os.getenv("QGIS_BIN_PATH", "")
QGIS_PYTHON_BAT = os.getenv("QGIS_PYTHON_BAT", "")
QGIS_EXECUTABLE = os.getenv("QGIS_EXECUTABLE", "")
QGIS_SVG_PATHS = [p for p in os.getenv("QGIS_SVG_PATHS", "").split(os.pathsep) if p]
QGIS_STYLE_CACHE_DIR = Path(os.getenv("QGIS_STYLE_CACHE_DIR", str(DATA_DIR / "qgis_style_cache")))
QGIS_STYLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
