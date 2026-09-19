from __future__ import annotations

import re
from typing import Any

from core.query_normalizer import normalize_query
from services.auth_service import PLAN_PRO, PLAN_STANDARD, PRO_PRICE_RMB_MONTH, is_pro_user, normalize_plan
try:
    from services.cartography_instruction_service import resolve_palette_from_text
except Exception:
    resolve_palette_from_text = None

AUTO_DATA_HINTS = [
    "联网", "网上", "网络", "自动下载", "下载数据", "自己下载", "不用上传", "不上传", "无需上传", "帮我找数据",
    "公开数据", "在线数据", "gee", "google earth engine", "遥感数据", "气候数据", "地形数据",
]
USER_DATA_WORDS = ["用我的数据", "用我上传的", "按我这个", "使用我的数据", "用这个csv", "用这个tif", "按我上传的数据"]
CHENGDU_WORDS = ["成都", "成都市"]
DEFAULT_YEAR = "2020"

# 粗粒度识别行政区名。只用于权限拦截/提示，不作为严肃地名解析器。
LOCATION_PATTERN = re.compile(r"([\u4e00-\u9fa5]{2,16}(?:省|市|县|区|州|盟|地区|自治州|旗))")
YEAR_PATTERN = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")

STANDARD_FEATURES = [
    "正常聊天、DSM 方法解释、结果解读",
    "上传用户自己的 tif/csv 数据并预览/质检",
    "使用内置 2020 年成都市数据执行 SOM 制图",
    "基于已完成制图结果执行 GCP 不确定性分析",
]
PRO_FEATURES = STANDARD_FEATURES + [
    "按用户指定年份与地区自动准备在线数据",
    "自动生成在线数据准备 manifest，便于替换真实 GEE/NASA/SoilGrids 下载流程",
    "未来可接入开发者统一管理的外部数据平台凭据",
]





PALETTE_KEYWORDS = {
    "红": "red", "红色": "red", "红色系": "red", "热力": "heat", "暖色": "heat",
    "绿": "green", "绿色": "green", "绿色系": "green",
    "蓝": "blue", "蓝色": "blue", "蓝色系": "blue", "冷色": "blue",
    "紫": "purple", "紫色": "purple", "紫色系": "purple", "青紫": "cyan_purple",
    "橙": "orange", "橙色": "orange", "橙色系": "orange",
    "黄": "yellow", "黄色": "yellow", "黄色系": "yellow", "棕黄": "yellow",
    "黄橙红": "ArcGIS Pro｜Yellow Orange Red", "黄绿蓝": "ArcGIS Pro｜Yellow Green Blue", "红黄绿": "ArcGIS Pro｜RdYlGn",
    "地形": "terrain", "Terrain": "terrain", "光谱": "spectral", "Spectral": "spectral", "Viridis": "viridis", "viridis": "viridis",
    "灰": "gray", "灰色": "gray", "黑白": "gray", "灰度": "gray",
}


def _extract_first_group(patterns: list[str], text: str) -> str | None:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            val = str(m.group(1) or '').strip(' ：:;；，,。"“”')
            if val:
                return val
    return None


def _extract_palette(text: str) -> str | None:
    # First use the central palette resolver so every workflow supports the same
    # set of color ramps: blue, green, yellow/brown, red/heat, purple, gray,
    # terrain, Viridis, Spectral, YlGnBu, YlOrRd, RdYlGn, etc.
    try:
        if resolve_palette_from_text is not None:
            ramp, _alias = resolve_palette_from_text(text)
            if ramp:
                return ramp
    except Exception:
        pass
    m = re.search(r"(?:颜色|配色|色带|调色板|palette|colormap|cmap)\s*(?:设置为|设为|选择|选用|指定为|改成|改为|换成|变成|为|用)?\s*([一-龥a-zA-Z]+)", text, flags=re.IGNORECASE)
    if m:
        raw = str(m.group(1) or '').strip(' ：:;；，,。"“”')
        for k in sorted(PALETTE_KEYWORDS, key=len, reverse=True):
            if k in raw:
                return PALETTE_KEYWORDS[k]
        return raw.lower() if raw else None
    for k in sorted(PALETTE_KEYWORDS, key=len, reverse=True):
        if f"用{k}" in text or f"选择{k}" in text or f"选用{k}" in text or f"改成{k}" in text or f"改为{k}" in text or f"换成{k}" in text:
            return PALETTE_KEYWORDS[k]
    return None


def _component_requested(text: str, component_words: list[str]) -> bool | None:
    # 否定优先。支持“不显示图例比例尺”“不要图例和比例尺”“不带图例、比例尺、指北针”。
    for comp in component_words:
        if re.search(rf"(?:不显示|不要|不带|隐藏|取消)[^，,。；;]{{0,12}}{re.escape(comp)}", text):
            return False
    for comp in component_words:
        if re.search(rf"(?:显示|添加|带|要|加上)[^，,。；;]{{0,12}}{re.escape(comp)}", text):
            return True
    return None


def parse_map_style_preferences(user_text: str) -> dict[str, Any]:
    text = normalize_query(user_text)
    prefs: dict[str, Any] = {}

    palette = _extract_palette(text)
    if palette:
        prefs["palette"] = palette

    title_text = _extract_first_group([r"(?<!图例)标题\s*(?:设置为|设为|为|叫|命名为)\s*(.+?)(?:[,，。；;]|$)"], text)
    if title_text:
        prefs["title_text"] = title_text
    legend_title = _extract_first_group([r"图例标题\s*(?:设置为|设为|为|叫|命名为)\s*(.+?)(?:[,，。；;]|$)"], text)
    if legend_title:
        prefs["legend_title"] = legend_title

    show_legend = _component_requested(text, ["图例"])
    show_scale = _component_requested(text, ["比例尺"])
    show_north = _component_requested(text, ["指北针", "北箭", "指南针"])
    show_title = _component_requested(text, ["标题"])

    if title_text and show_title is None:
        show_title = True
    if legend_title and show_legend is None:
        show_legend = True

    if show_title is not None:
        prefs["show_title"] = show_title
    if show_legend is not None:
        prefs["show_legend"] = show_legend
    if show_scale is not None:
        prefs["show_scale"] = show_scale
    if show_north is not None:
        prefs["show_north"] = show_north
    return prefs

def normalize_agent_version(value: str | None) -> str:
    value = str(value or "standard").strip().lower()
    return PLAN_PRO if value in {"pro", "vip"} else PLAN_STANDARD


def extract_years(text: str) -> list[str]:
    return YEAR_PATTERN.findall(normalize_query(text))


def _clean_location(value: str) -> str:
    value = value or ""
    while value.startswith(("年", "月", "日", "的")):
        value = value[1:]
    for prefix in ("帮我制作", "帮我绘制", "帮我生成", "为我制作", "为我绘制", "为我生成", "制作", "绘制", "生成", "画", "做"):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    while value.startswith(("年", "月", "日", "的")):
        value = value[1:]
    return value


def extract_locations(text: str) -> list[str]:
    text = normalize_query(text)
    ignored = {"有机质", "土壤", "不确定性", "制图", "分布图"}
    locations = []
    for raw in LOCATION_PATTERN.findall(text):
        m = _clean_location(raw)
        if not m or any(bad in m for bad in ignored):
            continue
        # 避免过长的句段被误识别成行政区。
        if len(m) > 18:
            continue
        if m not in locations:
            locations.append(m)
    return locations


def request_target_summary(text: str) -> dict[str, Any]:
    years = extract_years(text)
    locations = extract_locations(text)
    year = years[0] if years else DEFAULT_YEAR
    location = locations[0] if locations else "成都市"
    return {
        "year": year,
        "location": location,
        "years": years,
        "locations": locations,
        "is_default_target": (year == DEFAULT_YEAR and any(w in location for w in CHENGDU_WORDS)),
    }


def needs_pro_auto_data(user_text: str, has_uploaded_files: bool = False, selected_version: str | None = None) -> bool:
    """判断一次制图请求是否需要进入 Pro 自动数据准备流程。

    上传样点并不等于所有建模数据都已准备好。当前设计里，用户上传 SOM 样点，
    系统联网准备行政边界、遥感、气候、土壤背景、统计年鉴等协变量，仍然属于 Pro 能力。
    """
    text = normalize_query(user_text)
    selected_version = normalize_agent_version(selected_version)

    if selected_version == PLAN_PRO:
        # 用户主动选择 Pro 时，默认启用自动数据准备能力。
        # 即使用户已经上传了样点，也仍然要联网准备协变量。
        return True

    years = extract_years(text)
    locations = extract_locations(text)

    # 标准版只覆盖内置 2020 成都内置数据。非 2020 或非成都仍应进入 Pro 权限判断。
    if any(y != DEFAULT_YEAR for y in years):
        return True

    if locations and not any(any(c in loc for c in CHENGDU_WORDS) for loc in locations):
        return True

    # 对标准版而言，上传文件只表示用户提供了部分数据；这里保留原来的本地/用户数据路径。
    if any(k in text for k in USER_DATA_WORDS) or has_uploaded_files:
        return False

    if any(k.lower() in text.lower() for k in AUTO_DATA_HINTS):
        return True

    return False


def build_standard_limit_message(user_text: str, selected_version: str | None = None) -> str:
    target = request_target_summary(user_text)
    prefix = "当前选择的是 Pro 版，但该账号尚未升级。" if normalize_agent_version(selected_version) == PLAN_PRO else "当前账号为标准版。"
    return (
        f"{prefix}\n\n"
        "标准版支持：\n"
        "1. 正常聊天、结果解释与方法问答；\n"
        "2. 使用你自己上传的 tif/csv 数据；\n"
        "3. 使用系统内置的 2020 年成都市土壤有机质制图与不确定性分析。\n\n"
        "你这次的需求涉及自动联网准备数据"
        f"（识别目标：{target.get('year')} 年，{target.get('location')}），属于 Pro 版能力。\n"
        f"Pro 版收费 {PRO_PRICE_RMB_MONTH}￥/月。请在左侧版本卡片中点击“开通 Pro”，支付成功后可点击“继续上次 Pro 请求”或重新发送原问题。"
    )


def build_version_brief(auth_user: dict[str, Any] | None = None) -> dict[str, Any]:
    user = auth_user or {}
    pro_active = is_pro_user(user)
    return {
        "standard_features": STANDARD_FEATURES,
        "pro_features": PRO_FEATURES,
        "price_rmb_month": PRO_PRICE_RMB_MONTH,
        "current_plan": normalize_plan(user.get("plan")),
        "pro_active": pro_active,
        "pro_remaining_days": user.get("pro_remaining_days"),
        "pro_expires_at": user.get("pro_expires_at"),
    }


def can_use_selected_version(selected_version: str | None, auth_user: dict[str, Any] | None) -> bool:
    selected = normalize_agent_version(selected_version)
    if selected == PLAN_STANDARD:
        return True
    return is_pro_user(auth_user)
