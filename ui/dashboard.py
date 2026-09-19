from __future__ import annotations

import os
import json
import time
import re
import shutil
from urllib.parse import quote
from datetime import datetime
from pathlib import Path

from flask import jsonify, request, send_file

from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from config.settings import APP_TITLE, APP_SUBTITLE, BASE_DIR, DATA_DIR
from core.data_source_resolver import pick_latest_uploaded_tif
from core.response_builder import (
    build_accuracy_text,
    build_gcp_completion_text,
    build_guidance_text,
    build_map_completion_text,
    build_model_text,
    build_user_data_unavailable_text,
)
from core.state_manager import SessionState
from services.agent_service import execute_agent_turn
from services.auth_service import PRO_PRICE_RMB_MONTH, PRO_PERIOD_DAYS, authenticate, get_user_plan_status, is_pro_user, register_user, reset_password, simulate_send_code, upgrade_user_to_pro
from services.json_report_service import load_json_if_exists
from services.layout_map_service import build_stage_payload, normalize_palette_name, get_overlay_bundle, palette_options, COLOR_RAMPS, raster_valid_mask
from services.qgis_style_bridge_service import get_style_catalog
from services.result_context_service import build_result_context_text
from services.task_state_machine_service import repair_session_task_state, has_running_compute_task, can_start_compute_task
from services.result_reader_service import read_mapping_result, read_gcp_result
from services.session_service import BROWSER_SESSIONS, SESSION_STORE
from services.task_service import TASKS, start_gcp_task, start_rfk_task, _resolve_gcp_input_paths, cancel_session_tasks
from services.upload_service import save_multiple_uploads
from services.sample_reader import validate_uploaded_file
from services.data_health_service import inspect_csv, build_health_text
from services.upload_role_service import build_role_report, build_covariate_plan, generate_ai_covariate_review, write_upload_analysis, DEFAULT_COVARIATES, covariate_display_name
from services.preprocess_service import run_upload_preprocessing, summarize_preprocess_for_user
from services.domestic_download_service import (
    is_download_only_request,
    prepare_download_only_task,
    mark_download_started,
    scan_downloaded_files,
    launch_capture_browser,
    launch_ftp_capture_now,
    start_manual_ftp_download,
    cancel_download_task,
)
from services.download_postprocess_service import process_download_manifest_for_map, finalize_tpdc_download_after_ftp
from services.gee_download_service import run_gee_download_task
from services.spatial_standardization_service import (
    standardize_session_rasters,
    build_standardization_chat_message,
)
from services.mapping_source_choice_service import (
    build_source_choice_payload,
    data_source_for_choice,
    detect_mapping_source,
    should_ask_mapping_source,
)
from services.vip_service import build_standard_limit_message, build_version_brief, normalize_agent_version, request_target_summary, parse_map_style_preferences
from services.pro_data_service import list_recent_pro_usage
from services.payment_qr_service import get_payment_qr_payload
from services.cartography_instruction_service import parse_cartography_instruction, merge_layout_instruction, infer_map_scope
from services.pipeline_step_logger import append_step_record, audit_paths
from services.aoi_preflight_service import preflight_mapping_region
try:
    from services.aoi_preflight_service import detect_requested_aoi_from_local_admin
except Exception:
    detect_requested_aoi_from_local_admin = None  # type: ignore
from ui.components import chat_bubble, progress_bar, title_block
from utils.pro_console import pro_console_log


# Progress should not resurrect stale status from a previous app run.
APP_BOOT_TS = int(time.time())


def _session_from_store(store_data):
    if store_data:
        return SessionState.from_dict(store_data)
    return SessionState()


THINKING_MESSAGE = "已收到，正在结合当前任务上下文分析……"


def _default_covariates_cn_text() -> str:
    return "、".join(covariate_display_name(c, with_code=False) for c in DEFAULT_COVARIATES)


def _welcome_card():
    default_covs = _default_covariates_cn_text()
    line_style = {"display": "flex", "gap": "8px", "alignItems": "flex-start", "marginTop": "6px"}
    tag_style = {"fontWeight": 700, "minWidth": "24px"}
    return html.Div([
        html.Div("你好，我是数字土壤制图智能体。", style={"fontWeight": 800, "fontSize": "16px", "marginBottom": "8px", "color": "#0f172a"}),
        html.Div([
            html.Div([html.Span("🧭", style=tag_style), html.Span("当前主线：自动模型搜索与正式制图，按本轮实际选择模型显示模型名称")], style=line_style),
            html.Div([html.Span("📌", style=tag_style), html.Span("你需要上传土壤有机质采样点数据，通常为 CSV/Excel，并包含经度、纬度和有机质字段。")], style=line_style),
            html.Div([html.Span("🧱", style=tag_style), html.Span("环境协变量可以不上传；未上传时使用默认环境协变量：" + default_covs + "。")], style=line_style),
            html.Div([html.Span("🌐", style=tag_style), html.Span("支持数据下载：国家青藏高原科学数据中心。")], style=line_style),
        ], style={"background": "#eef6ff", "border": "1px solid #bfdbfe", "borderRadius": "12px", "padding": "10px 12px", "marginBottom": "10px"}),
        html.Div("需求提示", style={"fontWeight": 800, "margin": "8px 0 6px", "color": "#065f46"}),
        html.Div([
            html.Div("⬇️ 数据下载", style=line_style),
            html.Div("🗺️ 有机质制图", style=line_style),
            html.Div("🎯 不确定性分析（须先完成有机质制图）", style=line_style),
            html.Div("🧪 数据检查：检查样点、协变量、投影、分辨率、对齐和缺失值", style=line_style),
            html.Div("📊 结果解释：解释制图结果、精度、误差和空间分布", style=line_style),
            html.Div("💬 DSM 知识问答", style=line_style),
        ], style={"background": "#f0fdf4", "border": "1px solid #bbf7d0", "borderRadius": "12px", "padding": "10px 12px"}),
    ])


def _render_chat(history):
    if not history:
        return [chat_bubble("assistant", _welcome_card())]
    return [chat_bubble(item["role"], item["content"]) for item in history]


def _result_text_from_stage_payload(stage_payload: dict) -> str:
    result = (stage_payload or {}).get("result") or {}
    kind = result.get("kind") or ""
    if not kind and (stage_payload or {}).get("preview_errors"):
        try:
            return "地图预览生成异常：" + str((stage_payload or {}).get("preview_errors", [])[0].get("name") or "请查看后台记录")
        except Exception:
            return "地图预览生成异常，请查看后台记录"
    if kind == "uploaded_raster":
        return "上传栅格：" + str(result.get("source_name") or "未命名图层")
    if kind == "uploaded_raster_stack":
        return "上传协变量叠加预览"
    if kind == "som_map":
        return str((stage_payload or {}).get("title") or result.get("title") or "有机质制图结果")
    if kind == "gcp_map":
        return str((stage_payload or {}).get("title") or result.get("title") or "不确定性分析结果")
    return "暂无"


def _drop_pending_thinking_message(session: SessionState) -> None:
    try:
        if session.chat_history and session.chat_history[-1].get("role") == "assistant":
            content = str(session.chat_history[-1].get("content") or "")
            if content == THINKING_MESSAGE or content.startswith(THINKING_MESSAGE) or "智能体对话主控已接收请求" in content:
                session.chat_history.pop()
    except Exception:
        pass


def _compose_analysis_context(session: SessionState) -> str:
    return build_result_context_text(
        rfk_paths=session.rfk_result_paths,
        gcp_paths=session.gcp_result_paths,
    )


def _append_extra_context(session: SessionState, user_text: str) -> str:
    parts = []
    if session.has_rfk_result or session.has_gcp_result:
        parts.append(_compose_analysis_context(session))

    lower = user_text.lower()
    wants_metrics = any(k in user_text for k in ["精度", "评价", "误差"]) or any(k in lower for k in ["r2", "rmse", "mae"])
    wants_model = ("模型" in user_text) or ("rfk" in lower) or ("gcp" in lower)

    if wants_metrics:
        metrics = None
        if session.rfk_result_paths.get("report_json"):
            metrics = load_json_if_exists(session.rfk_result_paths["report_json"])
        parts.append(build_accuracy_text(metrics))
    if wants_model:
        parts.append(build_model_text())
    return "\n\n".join([p for p in parts if p])


def _history_toolbar():
    archives = SESSION_STORE.list_archives(limit=200)
    options = [{"label": _archive_label(item), "value": item.get("archive_id")} for item in archives]
    summary = ""
    return html.Div([
        html.Div("会话管理", style={"fontSize": "13px", "fontWeight": 700, "color": "#334155", "minWidth": "70px"}),
        dcc.Dropdown(
            id="archive-session-select",
            options=options,
            placeholder="选择一条历史记录继续",
            clearable=True,
            style={"minWidth": "320px", "maxWidth": "460px"},
            maxHeight=520,
        ),
        html.Button("继续该记录", id="load-archive-btn", n_clicks=0, style={
            "padding": "8px 12px", "borderRadius": "10px", "border": "1px solid #cbd5e1", "background": "#ffffff", "cursor": "pointer"
        }),
        html.Button("新建对话", id="new-session-btn", n_clicks=0, style={
            "padding": "8px 12px", "borderRadius": "10px", "border": "none", "background": "#0f172a", "color": "white", "cursor": "pointer"
        }),
    ], style={
        "display": "flex",
        "alignItems": "center",
        "gap": "10px",
        "flexWrap": "wrap",
        "marginBottom": "14px",
        "padding": "10px 12px",
        "border": "1px solid #e5e7eb",
        "borderRadius": "12px",
        "background": "#f8fafc",
    })



def _fmt_ts(ts: int | float | None) -> str:
    try:
        if not ts:
            return "--"
        return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d")
    except Exception:
        return "--"


def _render_agent_version_panel(agent_version: str | None, auth_user: dict | None, session: SessionState | None = None):
    """顶部紧凑版版本选择器。

    旧版把版本说明作为大卡片放在聊天区顶部，严重挤占对话空间。
    现在改为用户栏中的折叠拉条：默认只占一行，点击后以浮层展示标准版/Pro版差异与续费入口。
    """
    selected = normalize_agent_version(agent_version)
    auth_user = auth_user or {}
    pro_active = is_pro_user(auth_user)
    brief = build_version_brief(auth_user)
    remaining = auth_user.get("pro_remaining_days")
    expires_at = auth_user.get("pro_expires_at")
    plan_label = "已开通" if pro_active else "标准版"
    selected_label = "Pro" if selected == "pro" else "标准版"
    pro_note = (
        f"已开通：Pro 有效期至 {_fmt_ts(expires_at)}，剩余 {remaining if remaining is not None else '--'} 天。"
        if pro_active
        else f"未开通：Pro 版 {PRO_PRICE_RMB_MONTH}￥/月，开通后可使用自动数据准备与高级制图功能，有效期 {PRO_PERIOD_DAYS} 天。"
    )
    pending_text = getattr(session, "pending_pro_request_text", None) if session else None
    pending_target = getattr(session, "pending_pro_target", {}) if session else {}
    recent_usage = list_recent_pro_usage(auth_user.get("id"), limit=2) if pro_active else []

    pending_block = []
    if pending_text:
        target_line = ""
        if isinstance(pending_target, dict) and pending_target:
            target_line = f"识别目标：{pending_target.get('year', '--')} 年，{pending_target.get('location', '--')}"
        pending_block = [
            html.Div(className="pro-pending-card compact", children=[
                html.Div("待继续的 Pro 请求", className="pro-pending-title"),
                html.Div(target_line, className="pro-pending-target") if target_line else html.Div(),
                html.Div(pending_text[:80] + ("……" if len(pending_text) > 80 else ""), className="pro-pending-text"),
                html.Button("继续上次 Pro 请求", id="retry-pro-request-btn", type="button", n_clicks=0, className="pro-retry-btn", style={"display": "inline-flex" if pro_active else "none"}),
            ])
        ]

    usage_block = []
    if recent_usage:
        usage_block = [
            html.Div("最近 Pro 数据准备", className="pro-usage-title"),
            html.Div(className="pro-usage-list", children=[
                html.Div(f"{(r.get('target') or {}).get('year', '--')} 年 · {(r.get('target') or {}).get('location', '--')}", className="pro-usage-item")
                for r in recent_usage
            ]),
        ]

    return html.Details(className="top-version-drawer", children=[
        html.Summary(className="top-version-summary", children=[
            html.Span("版本", className="top-version-summary-title"),
            html.Span(selected_label, className=("top-version-selected pro" if selected == "pro" else "top-version-selected")),
            html.Span(plan_label, className=("top-version-plan pro" if pro_active else "top-version-plan")),
            html.Span("设置", className="top-version-hint"),
        ]),
        html.Div(className="top-version-body", children=[
            html.Div(className="top-version-body-head", children=[
                html.Div("智能体版本", className="agent-version-title"),
                html.Div(plan_label, className=("agent-version-badge pro" if pro_active else "agent-version-badge")),
            ]),
            dcc.RadioItems(
                id="agent-version-radio",
                options=[
                    {"label": "标准版", "value": "standard"},
                    {"label": "Pro 版", "value": "pro"},
                ],
                value=selected,
                className="agent-version-radio compact",
                inputStyle={"marginRight": "6px"},
                labelStyle={"display": "inline-flex", "alignItems": "center", "marginRight": "14px"},
            ),
            html.Div(className="agent-version-compare compact", children=[
                html.Div(className="version-mini-card", children=[
                    html.Div("标准版", className="version-mini-title"),
                    html.Div("上传数据 / 2020 成都内置数据 / GCP", className="version-mini-text"),
                ]),
                html.Div(className="version-mini-card pro", children=[
                    html.Div("Pro 版", className="version-mini-title"),
                    html.Div("用户上传样点，系统按年份与地区联网准备协变量", className="version-mini-text"),
                ]),
            ]),
            html.Details(className="version-detail-box compact", children=[
                html.Summary("查看版本能力说明"),
                html.Div(className="version-feature-cols", children=[
                    html.Div([html.Div("标准版能力", className="version-col-title")] + [html.Div("• " + x, className="version-feature-line") for x in brief.get("standard_features", [])]),
                    html.Div([html.Div("Pro 版增加", className="version-col-title pro")] + [html.Div("• " + x, className="version-feature-line") for x in brief.get("pro_features", [])[len(brief.get("standard_features", [])):]]),
                ]),
            ]),
            html.Div(pro_note, className=("agent-version-note pro compact" if pro_active else "agent-version-note warn compact")),
            html.Button(
                (f"续费 Pro（{PRO_PRICE_RMB_MONTH}￥/月）" if pro_active else f"开通 Pro（{PRO_PRICE_RMB_MONTH}￥/月）"),
                id="pro-pay-btn",
                type="button",
                n_clicks=0,
                className="pro-pay-btn compact",
                title="开通 Pro 功能。",
            ),
            *pending_block,
            *usage_block,
        ]),
    ])

def _archive_label(item: dict) -> str:
    ts = item.get("archived_at") or 0
    dt = datetime.fromtimestamp(ts).strftime("%m-%d %H:%M") if ts else "--"
    return f"{dt} | {item.get('title') or '未命名记录'}"


def _commit_session(session: SessionState) -> dict:
    data = session.to_dict()
    SESSION_STORE.save_active(data)
    return data


def _push_ui_notification(session: SessionState, title: str, message: str, typ: str = "warning") -> None:
    """Set a non-chat UI toast. CSS handles 15s fade-out.

    Avoid re-pushing the exact same notification while a polling callback is
    retrying or the browser store is catching up.
    """
    if not message:
        return
    fp = f"{typ or 'warning'}|{title or '系统提醒'}|{message}"
    last = (getattr(session, "task_memory", {}) or {}).get("last_ui_notification_fp")
    if last == fp:
        return
    mem = dict(getattr(session, "task_memory", {}) or {})
    mem["last_ui_notification_fp"] = fp
    try:
        session.task_memory = mem
    except Exception:
        pass
    session.ui_notification_seq = int(getattr(session, "ui_notification_seq", 0) or 0) + 1
    session.ui_notification = {
        "seq": session.ui_notification_seq,
        "type": typ or "warning",
        "title": title or "系统提醒",
        "message": message,
        "created_at": int(time.time()),
    }


def _render_ui_notification(session: SessionState):
    note = getattr(session, "ui_notification", None) or {}
    if not isinstance(note, dict) or not note.get("message"):
        return html.Div(id="dsm-toast-host", className="dsm-toast-host empty")
    typ = str(note.get("type") or "warning")
    return html.Div(id="dsm-toast-host", className="dsm-toast-host", children=[
        html.Div(className=f"dsm-toast dsm-toast-{typ}", key=str(note.get("seq") or note.get("created_at") or "0"), children=[
            html.Div(str(note.get("title") or "系统提醒"), className="dsm-toast-title"),
            html.Div(str(note.get("message") or ""), className="dsm-toast-message"),
        ])
    ])





def _toast_duration_ms(typ: str | None, duration_ms: int | None = None) -> int:
    if duration_ms is not None:
        try:
            return max(1500, int(duration_ms))
        except Exception:
            pass
    typ = str(typ or "warning").lower()
    if typ == "success":
        return 5000
    if typ == "info":
        return 5000
    if typ == "warning":
        return 8000
    if typ == "error":
        return 12000
    return 8000


def _make_toast(title: str, message: str, typ: str = "warning", duration_ms: int | None = None) -> dict:
    """Frontend toast payload. This does not depend on chat history rendering."""
    typ = typ or "warning"
    return {
        "seq": int(time.time() * 1000),
        "type": typ,
        "title": title or "系统提醒",
        "message": message or "",
        "duration_ms": _toast_duration_ms(typ, duration_ms),
        "created_at": int(time.time()),
    }

def _extract_opacity_pref(text: str) -> float | None:
    """Parse natural-language opacity commands such as 透明度60% / 半透明."""
    t = str(text or "")
    m = re.search(r"(?:透明度|不透明度)[^0-9]{0,8}(\d{1,3})(?:\s*%|％)?", t)
    if m:
        try:
            val = float(m.group(1))
            if val > 1:
                val = val / 100.0
            return max(0.15, min(1.0, val))
        except Exception:
            pass
    if any(w in t for w in ["半透明", "透明一点", "透明一些"]):
        return 0.55
    if any(w in t for w in ["不透明", "完全显示", "更实"]):
        return 1.0
    return None



def _resolve_layer_switch_instruction(user_text: str, session: SessionState) -> tuple[str | None, str | None]:
    """Resolve explicit natural-language layer switching commands.

    This is not a keyword task trigger; it only runs after the user explicitly asks
    to display/switch/view a map layer. It respects the layouts that actually
    exist in the current session payload.
    """
    t = str(user_text or "")
    if not re.search(r"(显示|切换|打开|查看|换到|转到|回到|调到|展示)", t):
        return None, None
    if re.search(r"(样点|采样点|底图|图例|比例尺|指北针|标题|图名)", t) and not re.search(r"(有机质图|不确定性|GCP|AOA|区间宽度|风险|图层|\.tif|tif)", t, flags=re.I):
        return None, None
    try:
        payload = build_stage_payload(session)
        layouts = payload.get("layouts") or []
    except Exception:
        layouts = []
    if not layouts:
        return None, None

    def has_layout(k: str) -> bool:
        return any(str(x.get("key")) == k for x in layouts)

    # Formal result layers first.
    if re.search(r"(不确定性|地理共形|GCP|AOA|区间宽度|风险)", t, flags=re.I) and has_layout("gcp_layout"):
        return "gcp_layout", "不确定性分析图"
    if re.search(r"(有机质图|土壤有机质图|预测图|原始预测图|根底土壤有机质图|基础有机质图|SOM)", t, flags=re.I) and has_layout("som_layout"):
        return "som_layout", "土壤有机质图"

    # Uploaded raster layers by visible name, e.g. “显示 aspect_250m_sin.tif”.
    low = t.lower()
    best = None
    best_len = 0
    for item in layouts:
        key = str(item.get("key") or "")
        label = str(item.get("tab_title") or item.get("title") or "")
        src = str(((item.get("result") or {}).get("source_name")) or "")
        for cand in [label, src]:
            c = cand.strip()
            if not c:
                continue
            if c.lower() in low and len(c) > best_len:
                best = (key, c)
                best_len = len(c)
    if best:
        return best

    # Generic uploaded covariate command: keep current uploaded layer if any,
    # otherwise switch to the first uploaded preview layer.
    if re.search(r"(协变量|上传图层|上传栅格|环境变量|环境协变量|tif|\.tif)", t, flags=re.I):
        active = str(getattr(session, "active_layout_key", "") or "")
        if active.startswith("uploaded_layout_") and has_layout(active):
            return active, "当前上传栅格图层"
        first = next((x for x in layouts if str(x.get("key") or "").startswith("uploaded_layout_")), None)
        if first:
            return str(first.get("key")), str(first.get("tab_title") or "上传栅格图层")
    return None, None

def _apply_ai_style_instruction(user_text: str, session: SessionState) -> tuple[bool, str]:
    """Apply map style changes from natural language.

    This is the AI-control bridge for styling: the visible manual style panel is
    hidden, but users can still say “把颜色改成蓝色、透明度 60%”。
    """
    prefs = parse_map_style_preferences(user_text) or {}
    opacity = _extract_opacity_pref(user_text)
    changed = []
    layer_key, layer_label = _resolve_layer_switch_instruction(user_text, session)
    if layer_key:
        session.active_layout_key = layer_key
        if layer_key == "som_layout":
            session.latest_result_kind = "som_map"
        elif layer_key == "gcp_layout":
            session.latest_result_kind = "gcp_map"
        elif str(layer_key).startswith("uploaded_layout_"):
            session.latest_result_kind = "uploaded_raster"
        changed.append("切换显示图层：" + str(layer_label or layer_key))
    if prefs.get("palette"):
        session.som_palette = prefs.get("palette")
        session.gcp_palette = prefs.get("palette")
        session.map_style_prefs = {**(getattr(session, "map_style_prefs", {}) or {}), **prefs}
        changed.append("色带")
    if opacity is not None:
        session.map_style_prefs = {**(getattr(session, "map_style_prefs", {}) or {}), "opacity": float(opacity)}
        changed.append(f"透明度 {int(float(opacity)*100)}%")
    t = str(user_text or "")
    if re.search(r"(隐藏|关闭|不要|不显示).{0,8}(样点|采样点)", t):
        session.map_style_prefs = {**(getattr(session, "map_style_prefs", {}) or {}), "show_samples": False}
        changed.append("隐藏样点")
    elif re.search(r"(显示|打开|保留).{0,8}(样点|采样点)", t):
        session.map_style_prefs = {**(getattr(session, "map_style_prefs", {}) or {}), "show_samples": True}
        changed.append("显示样点")
    if re.search(r"(隐藏|关闭|不要|不显示).{0,8}(底图|参考底图|天地图)", t):
        session.map_style_prefs = {**(getattr(session, "map_style_prefs", {}) or {}), "show_basemap": False}
        changed.append("隐藏参考底图")
    elif re.search(r"(显示|打开|保留).{0,8}(底图|参考底图|天地图)", t):
        session.map_style_prefs = {**(getattr(session, "map_style_prefs", {}) or {}), "show_basemap": True}
        changed.append("显示参考底图")
    # Formal cartographic layout instructions: map frame size, position, palette, map scope.
    layout_instr = parse_cartography_instruction(t)
    if layout_instr:
        session.map_layout_prefs = merge_layout_instruction(getattr(session, "map_layout_prefs", {}) or {}, layout_instr)
        if layout_instr.get("palette"):
            session.som_palette = layout_instr.get("palette")
            session.gcp_palette = layout_instr.get("palette")
            session.map_style_prefs = {**(getattr(session, "map_style_prefs", {}) or {}), "palette": layout_instr.get("palette")}
            if "色带" not in changed:
                changed.append("色带")
        if layout_instr.get("map_frame_percent"):
            changed.append(f"主图占页面 {layout_instr.get('map_frame_percent')}%")
        if layout_instr.get("map_position"):
            changed.append("主图位置" + str(layout_instr.get("map_position")))
        if layout_instr.get("map_scope") == "cropland":
            changed.append("耕地范围制图，非耕地区域透明掩膜")
        elif layout_instr.get("map_scope") == "full_domain":
            changed.append("全域制图范围")

    if not changed:
        return False, ""
    session.active_layout_key = getattr(session, "active_layout_key", None)
    return True, "已接收地图显示与制图版式更新指令：" + "、".join(changed) + "。地图图层刷新完成后会在对话历史中确认。"



def _looks_like_stop_task_request(text: str | None) -> bool:
    t = str(text or "").strip()
    if not t:
        return False
    # Explicit stop/cancel/terminate commands.  Do not treat analytical phrases
    # like “停止条件/停止规则” as task cancellation.
    if re.search(r"(停止|取消|终止|中止|暂停|不要继续|别继续|停下|停掉).{0,18}(当前任务|任务|下载|制图|建模|模型训练|不确定性|GCP|AOA|预处理|标准化|处理)", t, re.I):
        return True
    if re.fullmatch(r"\s*(停止|取消|终止|中止|暂停|停下|停掉)(吧|一下|当前任务)?\s*", t):
        return True
    return False



def _is_data_review_only_request(text: str | None) -> bool:
    """True when the user asks to inspect whether a dataset can be used, not to start mapping.

    Phrases such as “分析这组数据能不能用于制图 / 看看能不能用来制图 / 数据能不能支撑制图”
    must stay in data-review mode.  The word “制图” inside this type of question is
    the evaluation target, not an execution command.
    """
    t = str(text or "")
    if not t:
        return False
    review_signal = re.search(r"(分析|审查|检查|评估|看看|判断|诊断|预审|能不能|是否可以|可不可以|适不适合|能否|能不能用|可否|是否能).{0,40}(数据|样点|协变量|这组|文件|路径)", t)
    used_for_mapping = re.search(r"(用于|用来|支撑|进行|做).{0,12}(制图|建模|预测|有机质图)|制图.{0,12}(条件|可行|能不能|是否|适合)", t)
    explicit_start = re.search(r"(开始|立即|现在|执行|启动|生成|绘制|制作|输出|跑|运行).{0,10}(制图|建模|预测|有机质图)", t)
    if review_signal and used_for_mapping and not explicit_start:
        return True
    if re.search(r"(分析|审查|检查|评估|看看|判断).{0,20}(这组数据|这些数据|当前数据).{0,30}(能不能|是否可以|可不可以|适不适合).{0,20}(制图|建模|预测)", t):
        return True
    return False


def _start_new_mapping_context(session: SessionState, request_text: str) -> str:
    """Mark that a fresh mapping attempt has started and invalidate old GCP inputs.

    This prevents a subsequent “进行不确定性分析” from accidentally reusing an
    older completed map when the current mapping run has failed, been cancelled,
    or has not finished yet.
    """
    token = f"mapctx_{int(time.time() * 1000)}"
    try:
        mem = dict(getattr(session, "task_memory", {}) or {})
        mem["current_mapping_context_id"] = token
        mem["current_mapping_request_text"] = str(request_text or "")
        mem["current_mapping_started_at"] = time.time()
        mem["current_completed_mapping_task_id"] = ""
        mem["current_completed_mapping_context_id"] = ""
        mem["current_mapping_status"] = "running"
        session.task_memory = mem
    except Exception:
        pass
    # Do not let old results satisfy new GCP/AOA preconditions.
    session.som_map_shown = False
    session.gcp_shown = False
    session.rfk_result_paths = {}
    session.gcp_result_paths = {}
    session.last_result_paths = {}
    return token


def _mark_mapping_completed_for_current_context(session: SessionState, task) -> None:
    try:
        mem = dict(getattr(session, "task_memory", {}) or {})
        ctx = str(mem.get("current_mapping_context_id") or "")
        mem["current_completed_mapping_context_id"] = ctx
        mem["current_completed_mapping_task_id"] = str(getattr(task, "task_id", "") or "")
        mem["current_mapping_status"] = "done"
        mem["current_mapping_finished_at"] = time.time()
        session.task_memory = mem
        paths = dict(getattr(task, "result_paths", {}) or {})
        if paths is not None:
            paths["mapping_context_id"] = ctx
            paths["mapping_task_id"] = str(getattr(task, "task_id", "") or "")
            try:
                task.result_paths = dict(paths)
            except Exception:
                pass
    except Exception:
        pass


def _mark_mapping_not_completed(session: SessionState, status: str = "error") -> None:
    try:
        mem = dict(getattr(session, "task_memory", {}) or {})
        mem["current_mapping_status"] = status
        mem["current_completed_mapping_context_id"] = ""
        mem["current_completed_mapping_task_id"] = ""
        session.task_memory = mem
    except Exception:
        pass


def _has_current_completed_mapping_result(session: SessionState) -> tuple[bool, list[str]]:
    """Require the currently requested mapping context to have finished successfully."""
    missing: list[str] = []
    mem = dict(getattr(session, "task_memory", {}) or {})
    current_ctx = str(mem.get("current_mapping_context_id") or "")
    done_ctx = str(mem.get("current_completed_mapping_context_id") or "")
    done_task = str(mem.get("current_completed_mapping_task_id") or "")
    if not current_ctx or current_ctx != done_ctx:
        missing.append("当前制图任务尚未完成")
    task = TASKS.get(getattr(session, "latest_rfk_task_id", None)) if getattr(session, "latest_rfk_task_id", None) else None
    if not task or getattr(task, "status", None) != "done":
        missing.append("后台未确认当前制图任务完成")
    elif done_task and str(getattr(task, "task_id", "")) != done_task:
        missing.append("当前完成任务与会话记录不一致")
    if getattr(session, "latest_result_kind", None) != "som_map" or not getattr(session, "som_map_shown", False):
        missing.append("当前地图状态不是已完成的有机质图")
    if not getattr(session, "rfk_result_paths", None):
        missing.append("当前制图结果路径为空")
    return (not missing), missing

def _stop_current_user_task(session: SessionState, user_text: str) -> str:
    t = str(user_text or "")
    stopped: list[str] = []
    messages: list[str] = []
    wants_download = bool(re.search(r"下载|FTP|TPDC|数据获取|获取数据", t, re.I))
    wants_gcp = bool(re.search(r"不确定性|GCP|AOA|适用域", t, re.I))
    wants_mapping = bool(re.search(r"制图|建模|模型训练|预测|有机质图|模型", t, re.I))
    wants_preprocess = bool(re.search(r"预处理|标准化|重采样|对齐|统一", t, re.I))

    if wants_download or not (wants_mapping or wants_gcp or wants_preprocess):
        manifest_path = _download_manifest_path_from_session(session)
        if manifest_path:
            res = cancel_download_task(manifest_path, reason="用户已停止当前下载任务。")
            if res.get("ok"):
                stopped.append("download")
                messages.append("当前下载任务已停止。")

    kinds: set[str] = set()
    if wants_gcp:
        kinds.add("gcp")
    if wants_mapping:
        kinds.add("rfk")
    # Generic “停止/取消当前任务” stops every queued/running model task in this session.
    if not kinds and not wants_download:
        kinds = {"rfk", "gcp"}
    elif not kinds and not stopped:
        kinds = {"rfk", "gcp"}
    if kinds:
        labels = []
        if "rfk" in kinds:
            labels.append("制图/建模")
        if "gcp" in kinds:
            labels.append("不确定性分析")
        ids = cancel_session_tasks(session.session_id, kinds=kinds, reason="用户已停止当前" + "、".join(labels or ["任务"]) + "。")
        if ids:
            stopped.extend(list(kinds))
            if "rfk" in kinds:
                messages.append("当前制图/建模任务已停止。")
            if "gcp" in kinds:
                messages.append("当前不确定性分析任务已停止。")

    # Release stale UI state regardless of whether a backend id was found.
    if getattr(session, "latest_result_kind", None) in {"rfk_running", "gcp_running"}:
        if "rfk" in stopped:
            session.latest_result_kind = "som_map" if getattr(session, "rfk_result_paths", {}) else None
        elif "gcp" in stopped:
            session.latest_result_kind = "gcp_map" if getattr(session, "gcp_result_paths", {}) else ("som_map" if getattr(session, "rfk_result_paths", {}) else None)
        else:
            session.latest_result_kind = None
    session.last_task_type = "general_chat"
    try:
        decision = dict(getattr(session, "last_route_decision", {}) or {})
        decision["last_user_stop_request"] = {"text": t, "stopped": sorted(set(stopped)), "time": time.time()}
        session.last_route_decision = decision
    except Exception:
        pass
    if messages:
        return "✅ " + "\n".join(dict.fromkeys(messages)) + "\n你可以继续发送新的下载、制图、优化、预处理或结果解释指令。"
    return "当前没有检测到正在运行的下载、制图或不确定性分析任务；智能体仍处于可继续对话状态。"

def _extract_local_paths_from_text(text: str) -> list[str]:
    t = str(text or "")
    # Windows absolute paths and quoted paths. Keep simple and conservative.
    candidates = re.findall(r"[A-Za-z]:\\[^\n，,。；;]+", t)
    quoted = re.findall(r"[\"'“”‘’]([^\"'“”‘’]*[A-Za-z]:\\[^\"'“”‘’]+)[\"'“”‘’]", t)
    out = []
    for c in quoted + candidates:
        c = c.strip().strip("\'\"“”‘’")
        if c and c not in out:
            out.append(c)
    return out


def _import_local_paths_from_user_text(user_text: str, session: SessionState) -> tuple[list[dict], list[dict]]:
    """Register local files/directories named by the user into this session.

    用户经常直接给一个本机文件夹路径，里面可能包含子文件夹、瓦片数据、NC 文件、
    GeoTIFF 和矢量旁侧文件。正式流程不再把整套原始数据复制到上传目录，而是登记
    原始路径并递归识别主数据文件；这样速度更快，也不会破坏瓦片/旁侧文件结构。
    """
    paths = _extract_local_paths_from_text(user_text)
    if not paths:
        return [], []

    primary_exts = {".csv", ".txt", ".xls", ".xlsx", ".tif", ".tiff", ".nc", ".hdf", ".h5", ".hdf5", ".geojson", ".gpkg", ".zip", ".shp"}
    sidecar_exts = {".dbf", ".shx", ".prj", ".cpg", ".qpj", ".sbn", ".sbx", ".tfw", ".tif.aux.xml", ".tiff.aux.xml", ".ovr", ".xml"}

    def _suffix_key(path: Path) -> str:
        name = path.name.lower()
        if name.endswith(".tif.aux.xml"):
            return ".tif.aux.xml"
        if name.endswith(".tiff.aux.xml"):
            return ".tiff.aux.xml"
        return path.suffix.lower()

    imported, failed = [], []
    for raw in paths:
        p = Path(raw)
        try:
            if not p.exists():
                failed.append({"name": raw, "error": "路径不存在或程序无权限访问。"})
                continue
            if p.is_dir():
                all_files = [x for x in p.rglob("*") if x.is_file()]
                primary_files = [x for x in all_files if _suffix_key(x) in primary_exts]
                sidecar_count = sum(1 for x in all_files if _suffix_key(x) in sidecar_exts)
            elif p.is_file():
                primary_files = [p] if _suffix_key(p) in primary_exts else []
                sidecar_count = 1 if _suffix_key(p) in sidecar_exts else 0
            else:
                failed.append({"name": raw, "error": "路径不是可读取文件或文件夹。"})
                continue

            if not primary_files:
                failed.append({"name": raw, "error": f"未发现可直接识别的主数据文件；旁侧文件数量 {sidecar_count}。"})
                continue

            # Prefer fast path registration for large folders. Validate CSV/Excel/shp
            # and the first several rasters for immediate feedback; all files still enter
            # the inventory and later preprocessing/model tools can use their original paths.
            raster_seen = 0
            for src in primary_files:
                ext = _suffix_key(src)
                do_validate = ext in {".csv", ".txt", ".xls", ".xlsx", ".shp", ".geojson", ".gpkg"}
                if ext in {".tif", ".tiff", ".nc", ".hdf", ".h5", ".hdf5"}:
                    raster_seen += 1
                    do_validate = raster_seen <= 30
                validation = {"ok": True, "path": str(src), "fast_local_registration": True}
                if do_validate:
                    try:
                        validation = validate_uploaded_file(src)
                    except Exception as vex:
                        validation = {"ok": False, "stage": "validation_error", "path": str(src), "problems": [str(vex)], "message": str(vex)}
                if not validation.get("ok"):
                    failed.append({"name": str(src), "error": validation.get("message") or "文件读取或字段校验失败。", "validation": validation})
                    continue
                imported.append({
                    "name": src.name,
                    "original_name": src.name,
                    "path": str(src),
                    "source_path": str(src),
                    "source_root": str(raw),
                    "size": src.stat().st_size if src.exists() else 0,
                    "role_guess": "local_path_import",
                    "validation": validation,
                })
            if p.is_dir():
                # Store a compact folder inventory hint for later report text.
                imported.append({
                    "name": f"目录清单：{p.name}",
                    "original_name": p.name,
                    "path": str(p),
                    "source_root": str(raw),
                    "size": 0,
                    "role_guess": "folder_inventory",
                    "validation": {"ok": True, "path": str(p), "folder_primary_count": len(primary_files), "folder_sidecar_count": sidecar_count},
                    "folder_inventory_only": True,
                })
        except Exception as exc:
            failed.append({"name": raw, "error": str(exc)})
    # Do not keep folder_inventory_only pseudo-items as model inputs. They are only
    # useful for a user-facing summary.
    real_imported = [x for x in imported if not x.get("folder_inventory_only")]
    if not real_imported and imported:
        # If there were only folder inventory records, surface them as failures.
        for item in imported:
            failed.append({"name": item.get("name"), "error": "目录中未发现可建模主数据文件。"})
    return real_imported, failed

def _render_global_toast(note: dict | list | None):
    notes = note if isinstance(note, list) else ([note] if isinstance(note, dict) else [])
    notes = [n for n in notes if isinstance(n, dict) and n.get("message")]
    if not notes:
        return html.Div(className="dsm-toast-host empty")
    children = []
    for idx, n in enumerate(notes[-3:]):
        typ = str(n.get("type") or "warning")
        duration = _toast_duration_ms(typ, n.get("duration_ms"))
        children.append(html.Div(
            className=f"dsm-toast dsm-toast-{typ}",
            key=str(n.get("seq") or n.get("created_at") or idx),
            style={"--toast-duration": f"{duration}ms"},
            children=[
                html.Div(str(n.get("title") or "系统提醒"), className="dsm-toast-title"),
                html.Div(str(n.get("message") or ""), className="dsm-toast-message"),
            ],
        ))
    return html.Div(className="dsm-toast-host", children=children)


def _text_has_local_admin_region(text: str) -> bool:
    text = str(text or "").strip()
    if not text:
        return False
    if detect_requested_aoi_from_local_admin is not None:
        try:
            hit = detect_requested_aoi_from_local_admin(text)
            if isinstance(hit, dict) and hit.get("ok"):
                return True
        except Exception:
            pass
    import re
    return bool(re.search(r"[\u4e00-\u9fff]{2,}(省|市|区|县|旗|州|盟)", text))


def _session_has_mapping_context(session) -> bool:
    if bool(getattr(session, "latest_uploaded_files", [])):
        return True
    if getattr(session, "last_task_type", None) in {"rfk_mapping", "preflight_blocked"}:
        return True
    hist = "\n".join(str(m.get("content", "")) for m in getattr(session, "recent_history", lambda n=8: [])(8))
    low = hist.lower()
    return any(w in low for w in ["土壤有机质", "有机质", "som", "soc", "土壤有机碳", "制图", "绘制", "rfk"])


def _looks_like_mapping_request(text: str, session=None) -> bool:
    """Legacy semantic hint only; must not be used to start tasks directly.

    V209: task execution comes from the AI planner's structured decision.  This
    helper can only support UI hints or preflight after AI has already chosen a
    mapping mode.
    """
    raw = str(text or "")
    t = raw.lower()
    if not t.strip():
        return False
    map_words = ["绘制", "制图", "制作", "生成", "出图", "画", "做", "预测", "mapping", "map"]
    target_words = ["土壤有机质", "有机质", "som", "soc", "土壤有机碳"]
    if not any(w in t for w in map_words):
        return False
    if any(w in t for w in target_words):
        return True
    return _text_has_local_admin_region(raw) and _session_has_mapping_context(session)


def _notification_from_mapping_reports(result_paths: dict | None) -> dict | None:
    paths = result_paths or {}
    reports = []
    for key in ["platform_report_json", "report_json"]:
        p = paths.get(key)
        if p:
            data = load_json_if_exists(p)
            if isinstance(data, dict):
                reports.append(data)
    # Prefer explicit region mismatch notes from data_source_audit_report.json.
    for data in reports:
        notes = data.get("region_inference_notes") or []
        if isinstance(notes, list):
            for n in notes:
                if isinstance(n, dict) and n.get("popup") and n.get("message"):
                    method = str(n.get("method") or "")
                    if method in {"sample_target_region_mismatch", "sample_aoi_overlap_zero"}:
                        return {"type": "warning", "title": "样点与制图区域不匹配", "message": str(n.get("message"))}
                    if method == "sample_aoi_overlap_low":
                        return {"type": "warning", "title": "目标区域样点数偏少", "message": str(n.get("message"))}
        target = data.get("target") or {}
        if isinstance(target, dict):
            mismatch = target.get("sample_region_mismatch") or {}
            if isinstance(mismatch, dict) and mismatch.get("message"):
                return {"type": "warning", "title": "样点与制图区域不匹配", "message": str(mismatch.get("message"))}
            overlap = target.get("sample_aoi_overlap") or data.get("sample_aoi_overlap") or {}
            if isinstance(overlap, dict) and overlap.get("ok"):
                try:
                    inside = int(overlap.get("inside_count") or 0)
                    total = int(overlap.get("total_count") or 0)
                    region = target.get("region") or overlap.get("region") or "目标区域"
                    if inside == 0 and total > 0:
                        return {"type": "warning", "title": "样点与制图区域不匹配", "message": f"{region} 内没有检测到上传样点，当前结果属于跨区外推，不能直接交付。"}
                    if 0 < inside < int(os.getenv("PRO_AOI_MIN_SAMPLE_WARNING", "30")) <= total:
                        return {"type": "warning", "title": "目标区域样点数偏少", "message": f"{region} 内样点数较少：{inside}/{total}。模型已按区域样点规模调整复杂度，但结果仍需谨慎解释。"}
                except Exception:
                    pass
    return None


def _metrics_for_session(session: SessionState) -> list[html.Div]:
    report = None
    if session.latest_result_kind == "som_map":
        report = load_json_if_exists(session.rfk_result_paths.get("report_json", ""))
    elif session.latest_result_kind == "gcp_map":
        report = load_json_if_exists(session.gcp_result_paths.get("report_json", ""))
    if not isinstance(report, dict):
        return [html.Div("暂无", className="muted-line")]

    metric_source = report.get("metrics") if isinstance(report.get("metrics"), dict) else report
    fields = []
    for key in ["final_r2", "R2", "r2", "RMSE", "rmse", "MAE", "mae", "PICP", "MPIW", "NMPIW", "CCB", "IntervalScore"]:
        if key in metric_source and metric_source[key] is not None:
            v = metric_source[key]
            try:
                if isinstance(v, (int, float)):
                    v = f"{float(v):.4f}"
            except Exception:
                pass
            fields.append(html.Div([
                html.Span(f"{key}: ", style={"fontWeight": 700}),
                html.Span(str(v)),
            ], className="metric-line"))
    return fields or [html.Div("暂无", className="muted-line")]




def _load_gee_formal_report(session: SessionState) -> dict:
    paths = session.rfk_result_paths or {}
    report_path = paths.get("gee_formal_report_json")
    if not report_path and paths.get("pred_tif"):
        try:
            report_path = str(Path(paths.get("pred_tif")).parent / "gee_formal_prediction_grid_report.json")
        except Exception:
            report_path = None
    report = load_json_if_exists(report_path or "")
    return report if isinstance(report, dict) else {}


def _gee_flow_audit_lines(session: SessionState) -> list[html.Div]:
    report = _load_gee_formal_report(session)
    if not report:
        return [html.Div("暂无GEE完整流程审计。", className="muted-line")]
    stats = report.get("mask_stats") or {}
    rows = [
        html.Div([html.Span("行政边界：", className="status-label"), html.Span("已裁剪" if report.get("admin_boundary_applied") else "未裁剪")], className=("status-row status-done" if report.get("admin_boundary_applied") else "status-row status-error")),
        html.Div([html.Span("耕地掩膜：", className="status-label"), html.Span("已启用" if report.get("cropland_mask_applied") else "未启用")], className=("status-row status-done" if report.get("cropland_mask_applied") else "status-row status-error")),
        html.Div([html.Span("边界来源：", className="status-label"), html.Span(str(report.get("bounds_source", "--")))], className="status-row"),
    ]
    if stats:
        rows.append(html.Div(
            f"格网审计：行政区内 {stats.get('admin_valid_cells', 'NA')}；耕地 {stats.get('cropland_valid_cells', 'NA')}；最终有效 {stats.get('final_valid_cells', 'NA')}；NoData {stats.get('final_nodata_cells', 'NA')}。",
            className="muted-line"
        ))
    return rows


def _result_file_lines(session: SessionState) -> list[html.Div]:
    paths = session.rfk_result_paths if session.latest_result_kind in {"som_map", "rfk_running"} else session.last_result_paths
    paths = paths or {}
    labels = [
        ("预测图", "pred_tif"),
        ("GEE审计报告", "gee_formal_report_json"),
        ("建模CSV", "model_ready_csv"),
        ("预测网格CSV", "gee_grid_covariates_csv"),
        ("行政边界mask", "gee_admin_mask_tif"),
        ("耕地mask", "gee_cropland_mask_tif"),
        ("最终有效区mask", "gee_final_valid_mask_tif"),
        ("自动GCP报告", "gee_gcp_report_json"),
        ("自动GCP区间宽度图", "gee_gcp_width_tif"),
        ("自动GCP下限图", "gee_gcp_lower_tif"),
        ("自动GCP上限图", "gee_gcp_upper_tif"),
        ("GCP报告", "report_json"),
        ("GCP区间宽度图", "width_tif"),
        ("GCP下限图", "lower_tif"),
        ("GCP上限图", "upper_tif"),
    ]
    rows = []
    for label, key in labels:
        val = paths.get(key)
        if val:
            rows.append(html.Div([html.Span(label + "：", className="status-label"), html.Span(str(val))], className="result-file-line"))
    return rows or [html.Div("暂无", className="muted-line")]

def _safe_existing_path(value: str | None) -> bool:
    try:
        return bool(value) and Path(value).exists()
    except Exception:
        return False


def _has_active_model_task(session: SessionState) -> bool:
    """Return True only when backend registry confirms a queued/running compute task."""
    return bool(has_running_compute_task(session, TASKS, kinds={"rfk", "gcp"}))


def _release_finished_model_task_state(session: SessionState) -> None:
    """Repair stale running flags before routing a new user instruction."""
    try:
        repair_session_task_state(session, TASKS)
    except Exception:
        pass


def _gcp_input_status(session: SessionState) -> tuple[bool, list[str]]:
    paths = session.rfk_result_paths or {}
    ok_current, current_missing = _has_current_completed_mapping_result(session)
    if not ok_current:
        return False, current_missing
    # V83: PRO/GEE tasks may already have generated uncertainty maps automatically.
    if _safe_existing_path(paths.get("gee_gcp_width_tif")) and _safe_existing_path(paths.get("gee_gcp_report_json")):
        return True, []
    resolved = _resolve_gcp_input_paths(paths)
    checks = [
        ("manifest/report_json", resolved.get("manifest")),
        ("pred_tif", resolved.get("pred_tif")),
        ("strict_oof_csv", resolved.get("strict_oof_csv")),
    ]
    missing = [name for name, path in checks if not _safe_existing_path(path)]
    if not missing:
        # Persist resolved paths back into session memory so later AI turns can
        # understand “继续/进一步做不确定性分析” without re-parsing files.
        session.rfk_result_paths = {**paths, **resolved}
    return (len(missing) == 0), missing



def _assign_display_map_title(session: SessionState, result_paths: dict | None) -> dict:
    """Assign stable user-facing map names: Title, Title（2）, Title（3）..."""
    paths = dict(result_paths or {})
    base = str(paths.get("map_title_base") or paths.get("map_title") or paths.get("display_map_title") or "土壤有机质图").strip()
    base = re.sub(r"\s+", "", base) or "土壤有机质图"
    existing = str(paths.get("display_map_title") or "").strip()
    if existing:
        paths["map_title_base"] = base
        return paths
    mem = getattr(session, "task_memory", {}) or {}
    counters = dict(mem.get("map_title_counters") or {})
    count = int(counters.get(base) or 0) + 1
    counters[base] = count
    mem["map_title_counters"] = counters
    try:
        session.task_memory = mem
    except Exception:
        pass
    display = base if count <= 1 else f"{base}（{count}）"
    paths["map_title_base"] = base
    paths["map_title"] = base
    paths["display_map_title"] = display
    return paths


def _assign_uncertainty_display_title(session: SessionState, result_paths: dict | None) -> dict:
    paths = dict(result_paths or {})
    if paths.get("display_map_title"):
        return paths
    rfk_paths = getattr(session, "rfk_result_paths", {}) or {}
    base = str(rfk_paths.get("display_map_title") or rfk_paths.get("map_title") or rfk_paths.get("map_title_base") or "土壤有机质图").strip()
    title = base.replace("图", "不确定性分析图") if base.endswith("图") else (base + "不确定性分析图")
    paths["map_title"] = title
    paths["display_map_title"] = title
    return paths

def _task_status_lines(session: SessionState) -> list[html.Div]:
    rows = []
    if session.latest_rfk_task_id:
        task = TASKS.get(session.latest_rfk_task_id)
        if task:
            rows.append(html.Div([html.Span("制图任务：", className="status-label"), html.Span(f"{task.status} / {task.progress}% / {task.stage}")], className=f"status-row status-{task.status}"))
            if task.status == "error" and task.error:
                rows.append(html.Div(_clean_task_error(task.error), className="task-error-box"))
    if session.latest_gcp_task_id:
        task = TASKS.get(session.latest_gcp_task_id)
        if task:
            rows.append(html.Div([html.Span("GCP：", className="status-label"), html.Span(f"{task.status} / {task.progress}% / {task.stage}")], className=f"status-row status-{task.status}"))
            if task.status == "error" and task.error:
                rows.append(html.Div(_clean_task_error(task.error), className="task-error-box"))
    if not rows:
        rows.append(html.Div("暂无运行任务", className="muted-line"))
    return rows




def _render_task_step_panel(task, title: str = "任务步骤记录"):
    if not task:
        return None
    try:
        logs = list(getattr(task, "logs", []) or [])[-6:]
    except Exception:
        logs = []
    try:
        audit = getattr(task, "step_audit_paths", {}) or {}
    except Exception:
        audit = {}
    rows = []
    for line in logs:
        text = str(line or "")
        if len(text) > 180:
            text = text[:180] + "……"
        rows.append(html.Div(text, className="task-step-line"))
    audit_lines = []
    if audit.get("csv"):
        audit_lines.append(html.Div("步骤审计CSV：" + str(audit.get("csv")), className="task-audit-path"))
    if audit.get("jsonl"):
        audit_lines.append(html.Div("步骤审计JSONL：" + str(audit.get("jsonl")), className="task-audit-path"))
    if not rows and not audit_lines:
        rows.append(html.Div("任务已创建，正在等待第一条后台记录。", className="muted-line"))
    return html.Div(className="task-step-card", children=[
        html.Div(title, className="panel-title"),
        html.Div(rows, className="task-step-list"),
        html.Div(audit_lines, className="task-audit-list"),
    ])


def _active_model_task(session: SessionState):
    """Return the task whose step records should be shown on the right panel.

    Once a GCP/AOA task has been explicitly started, the right-side step card
    must track the uncertainty-analysis workflow instead of continuing to show
    the already-finished SOM mapping logs.  If a new RFK task is currently
    running, it still takes precedence.
    """
    rfk_task = TASKS.get(session.latest_rfk_task_id) if getattr(session, "latest_rfk_task_id", None) else None
    gcp_task = TASKS.get(session.latest_gcp_task_id) if getattr(session, "latest_gcp_task_id", None) else None
    if rfk_task and getattr(rfk_task, "status", None) in {"queued", "running"}:
        return rfk_task, "制图任务步骤"
    if gcp_task and (
        getattr(session, "latest_result_kind", "") in {"gcp_running", "gcp_map"}
        or bool(getattr(session, "gcp_shown", False))
        or getattr(gcp_task, "status", None) in {"queued", "running"}
    ):
        return gcp_task, "不确定性分析步骤"
    if rfk_task:
        return rfk_task, "制图任务步骤"
    return None, "任务步骤"

def _clean_task_error(error: str | None) -> str:
    if not error:
        return "任务运行失败，但没有返回详细错误。"
    text = str(error)
    # 给普通用户优先显示最后几行，避免满屏 traceback。
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) > 10:
        return "\n".join(lines[-10:])
    return text


def _render_upload_feedback_box(feedback: str | None):
    if not feedback:
        return ""
    text = str(feedback)
    ok = ("上传处理完成" in text) or ("上传成功" in text) or ("读取/校验成功" in text) or ("读取与校验成功" in text) or ("校验通过" in text)
    cls = "upload-feedback-box ok" if ok else "upload-feedback-box warn"
    title = "上传读取结果：成功" if ok else "上传读取结果：需要处理"
    return html.Div(className=cls, children=[
        html.Div(title, className="upload-feedback-title"),
        html.Pre(text, className="upload-feedback-text"),
    ])


def _role_item_map(role_report: dict) -> dict[str, dict]:
    out = {}
    for item in (role_report or {}).get("items") or []:
        pth = str(item.get("path") or "")
        if pth:
            out[pth] = item
    return out


def _preprocess_record_map(preprocess_report: dict) -> dict[str, dict]:
    out = {}
    for key in ["sample_records", "covariate_records", "pending_conversion_records", "excluded_records"]:
        for rec in (preprocess_report or {}).get(key) or []:
            pth = str(rec.get("input_path") or "")
            if pth:
                out[pth] = rec
    return out


def _format_upload_success_message(saved_item: dict, role_item: dict | None, preprocess_record: dict | None) -> str:
    role_item = role_item or {}
    preprocess_record = preprocess_record or {}
    name = str(saved_item.get("original_name") or saved_item.get("name") or role_item.get("display_name") or "未命名文件")
    role = str(role_item.get("role") or "unknown")
    cov = role_item.get("matched_covariate")
    rec_status = str(preprocess_record.get("status") or "")
    rec_msg = str(preprocess_record.get("message") or role_item.get("reason") or "")

    if role == "sample_points":
        n = preprocess_record.get("row_count_cleaned")
        n_text = f"，有效样点 {n} 个" if n is not None else ""
        return (
            f"已成功上传采样点数据：{name}{n_text}。\n"
            "已识别为土壤有机质训练样点，将在地图上持续显示，直到完成有机质制图；你也可以在右侧“显示样点”中手动关闭。"
        )

    if role in {"covariate_raster", "covariate_categorical"}:
        cov_name = covariate_display_name(cov, with_code=True) if cov else "未匹配到默认名称的环境协变量"
        status_note = ""
        if rec_status in {"warning", "failed", "pending"} and rec_msg:
            status_note = f"\n注意：{rec_msg}"
        return (
            f"已成功上传环境协变量数据：{cov_name}（{name}）。\n"
            "已加入环境协变量图层队列，可在中间地图按上传顺序叠加/切换检查范围、分辨率、投影和对齐情况。"
            + status_note
        )

    if role == "multidimensional_covariate":
        cov_name = covariate_display_name(cov, with_code=True) if cov else "多维环境协变量"
        return (
            f"已成功接收环境协变量数据：{cov_name}（{name}）。\n"
            "该文件为 NC/HDF/H5 等多维数据，后续需要按目标年份/变量提取并转换为 GeoTIFF 后才能入模。"
        )

    if role == "covariate_table_points":
        cov_name = covariate_display_name(cov, with_code=True) if cov else "点状环境观测数据"
        return (
            f"已成功上传点状环境数据：{cov_name}（{name}）。\n"
            "已识别经纬度字段，但它还不是全域栅格；需要插值或栅格化后才能作为环境协变量。"
        )

    if role in {"mask_raster", "admin_boundary"}:
        return f"已成功上传参考/边界数据：{name}。该数据可用于显示、裁剪或审计，默认不作为当前模型协变量。"

    if role == "target_prediction_raster":
        return f"已成功上传已有预测结果图：{name}。该图可用于预览或对比，默认不作为环境协变量，以避免目标泄漏。"

    if role == "archive_package":
        return f"已成功上传压缩包：{name}。系统会继续识别其中可用的采样点、栅格或矢量数据。"

    detail = rec_msg or str(role_item.get("reason") or "系统已保存文件，但暂未确定其建模角色。")
    return f"已成功上传文件：{name}。\n当前识别结果：{role_item.get('label') or '未知类型'}。{detail}"


def _format_upload_failure_message(item: dict) -> str:
    name = str(item.get("original_name") or item.get("name") or "未知文件")
    reason = str(item.get("upload_error") or item.get("user_feedback") or "未知原因")
    return f"上传失败：{name}。\n原因：{reason}"


def _format_spatial_alignment_chat_messages(preprocess_report: dict, current_paths: set[str] | None = None) -> list[str]:
    messages = []
    current_paths = current_paths or set()
    for rec in (preprocess_report or {}).get("covariate_records") or []:
        inp = str(rec.get("input_path") or "")
        if current_paths and inp not in current_paths:
            continue
        align = rec.get("alignment") or {}
        action = str(align.get("action") or "")
        status = str(align.get("status") or "")
        name = str(rec.get("name") or "环境协变量")
        cov = rec.get("matched_covariate")
        cov_name = covariate_display_name(cov, with_code=True) if cov else name
        if status == "reference":
            messages.append(f"空间基准检查：{cov_name}（{name}）已作为本批环境协变量的对齐基准。")
        elif status == "success" and action in {"aligned", "reprojected", "reprojected_and_aligned"}:
            messages.append("空间处理完成：" + str(align.get("message") or f"{cov_name} 已完成自动统一/对齐。"))
        elif status == "failed":
            messages.append("空间处理失败：" + str(align.get("message") or f"{cov_name} 自动对齐失败。"))
    return messages


def _build_spatial_alignment_toast(preprocess_report: dict, current_paths: set[str] | None = None):
    current_paths = current_paths or set()
    actions = []
    failed = []
    for rec in (preprocess_report or {}).get("covariate_records") or []:
        inp = str(rec.get("input_path") or "")
        if current_paths and inp not in current_paths:
            continue
        align = rec.get("alignment") or {}
        status = str(align.get("status") or "")
        action = str(align.get("action") or "")
        if status == "success" and action in {"aligned", "reprojected", "reprojected_and_aligned"}:
            actions.append(rec)
        elif status == "failed":
            failed.append(rec)
    if actions or failed:
        msg_parts = []
        if actions:
            msg_parts.append(f"发现 {len(actions)} 个协变量坐标系或像元中心网格不一致，已自动统一到基准栅格。")
        if failed:
            msg_parts.append(f"{len(failed)} 个协变量自动处理失败，原因已写入对话历史。")
        typ = "warning" if failed else "info"
        return _make_toast("空间基准检查", " ".join(msg_parts), typ, 5000)
    return None



def _activate_first_uploaded_raster_preview(session: SessionState, prefer_latest: bool = False) -> bool:
    """Make uploaded rasters visible in the central map immediately after upload/import."""
    try:
        files = list(getattr(session, "latest_uploaded_files", []) or [])
        raster_items = [f for f in files if str(f.get("path") or "").lower().endswith((".tif", ".tiff")) and Path(str(f.get("path") or "")).exists()]
        if not raster_items:
            # NetCDF/NC cannot be previewed directly until preprocessing converts it,
            # but the session should still remember that uploaded data exists.
            return False
        chosen = raster_items[-1] if prefer_latest else raster_items[0]
        session.latest_result_kind = "uploaded_raster"
        session.uploaded_result_paths = {"uploaded_tif": str(chosen.get("path") or "")}
        session.last_result_paths = dict(session.uploaded_result_paths)
        # Keep sample points visible over the raster preview.
        session.map_style_prefs = {**(getattr(session, "map_style_prefs", {}) or {}), "show_samples": True}
        if not getattr(session, "active_layout_key", None) or str(getattr(session, "active_layout_key", "")).startswith("uploaded_layout_"):
            # build_stage_payload will resolve uploaded_layout_1 according to the current inventory.
            session.active_layout_key = "uploaded_layout_1"
        return True
    except Exception:
        return False


def _uploaded_inventory_brief(session: SessionState) -> str:
    try:
        role_report = getattr(session, "data_role_report", {}) or {}
        summary = role_report.get("summary") or {}
        total = int(summary.get("file_count") or len(getattr(session, "latest_uploaded_files", []) or []))
        sample = int(summary.get("sample_count") or 0)
        cov = int(summary.get("model_covariate_count") or 0)
        pending = int(summary.get("pending_conversion_count") or 0)
        return f"当前会话共识别文件 {total} 个，采样点数据 {sample} 个，可直接入模环境协变量 {cov} 个，待转换/插值数据 {pending} 个。"
    except Exception:
        return "当前会话数据已保存，系统将继续进行角色识别和预处理检查。"



def _perform_requested_data_usability_review(session: SessionState, user_text: str | None = None) -> bool:
    """Answer the user's data-usability question after paths/uploads are ingested.

    This function is intentionally separate from task execution.  When the user
    gives file paths and asks whether the data can be used for mapping, the
    correct second action is a data review, not starting a model task and not
    stopping after import feedback.
    """
    if not (_analysis_intent_requested(str(user_text or "")) or _is_data_review_only_request(user_text)):
        return False
    role_report = getattr(session, "data_role_report", {}) or {}
    plan = getattr(session, "covariate_readiness_plan", {}) or {}
    preprocess_report = getattr(session, "preprocess_report", {}) or {}
    summary = role_report.get("summary") or {}
    if not role_report:
        session.add_message("assistant", "我还没有读取到当前会话的数据识别报告，因此暂时不能判断这组数据是否适合制图。请先确认样点文件和环境协变量路径已成功导入。")
        return True

    sample_count = int(summary.get("sample_count") or 0)
    cov_count = int(summary.get("model_covariate_count") or 0)
    pending_count = int(summary.get("pending_conversion_count") or 0)
    mask_count = int(summary.get("mask_or_boundary_count") or 0)
    blocked_count = int(summary.get("blocked_count") or 0)
    total = int(summary.get("file_count") or len(getattr(session, "latest_uploaded_files", []) or []))
    gate = (preprocess_report.get("gate") or {}) if isinstance(preprocess_report, dict) else {}
    ready = list((plan or {}).get("ready") or [])
    missing = list((plan or {}).get("missing") or [])
    provided = dict((plan or {}).get("provided") or {})
    sample_items = [x for x in (role_report.get("items") or []) if x.get("role") == "sample_points"]
    cov_items = [x for x in (role_report.get("items") or []) if x.get("can_model") is True]
    pending_items = [x for x in (role_report.get("items") or []) if str(x.get("can_model")) in {"after_conversion", "after_interpolation", "after_rasterization"}]

    try:
        ai_review = generate_ai_covariate_review(role_report, {**(plan or {}), "preprocess": preprocess_report}, chat_history=session.chat_history)
    except Exception as exc:
        ai_review = f"AI 数据审查服务未返回有效内容：{exc}"

    if sample_count <= 0:
        conclusion = "当前不能进入正式制图：未识别到包含经纬度和土壤有机质/SOM目标字段的训练样点。"
    elif cov_count > 0:
        conclusion = "当前具备进入基线制图的基本条件：已识别训练样点和至少一个可直接入模的环境协变量栅格。若目标是成都市耕地制图，还需要确认 AOI、耕地掩膜/CLCD 与所有协变量在空间范围、坐标系和分辨率上匹配。"
    elif pending_count > 0:
        conclusion = "当前暂不建议直接进入正式制图：已识别训练样点和待转换/插值的环境数据，但这些数据需要先完成 NC/HDF 转栅格、插值或栅格化后才能作为全域协变量使用。"
    else:
        conclusion = "当前不能完成正式空间制图：虽然可能已有训练样点，但未识别到可用于全域预测的环境协变量栅格。可以继续上传 TIF/NC/HDF 协变量，或让系统从 TPDC 补充。"

    ready_cn = "、".join(covariate_display_name(x, with_code=True) for x in ready[:16]) if ready else "暂无"
    missing_cn = "、".join(covariate_display_name(x, with_code=True) for x in missing[:12]) if missing else "无"
    sample_names = "、".join(str(x.get("display_name") or x.get("name") or "样点文件") for x in sample_items[:5]) or "未识别"
    cov_names = "、".join(str(x.get("display_name") or x.get("name") or "协变量") for x in cov_items[:8]) or "暂无"
    pending_names = "、".join(str(x.get("display_name") or x.get("name") or "待处理数据") for x in pending_items[:8]) or "暂无"

    lines = [
        "✅ 已按你的第二个要求完成当前数据可制图性审查。",
        "",
        "1. 数据识别概况",
        f"- 共识别文件：{total} 个；训练样点：{sample_count} 个；可直接入模协变量：{cov_count} 个；待转换/插值数据：{pending_count} 个；掩膜/边界类数据：{mask_count} 个；阻断类文件：{blocked_count} 个。",
        f"- 样点文件：{sample_names}。",
        f"- 可直接入模协变量文件：{cov_names}。",
        f"- 待处理数据：{pending_names}。",
        "",
        "2. 可用性结论",
        conclusion,
        "",
        "3. 当前可用/建议补充协变量",
        f"- 当前可用或可派生：{ready_cn}。",
        f"- 建议补充：{missing_cn}。这些是提升模型稳定性和解释力的建议，不等同于硬性缺失。",
    ]
    if isinstance(gate, dict) and gate.get("message"):
        lines.extend(["", "4. 预处理门控信息", "- " + str(gate.get("message"))])
    lines.extend(["", "5. AI 对本组数据的判断", str(ai_review or "AI 数据审查服务未返回有效内容；上方结论已基于本地文件识别和预处理结果给出。")])
    lines.extend(["", "我不会自动启动制图。只有你明确说“开始制图/执行制图/根据这组数据生成图”时，才会进入模型任务。"])
    final_text = "\n".join(lines)
    session.add_message("assistant", final_text)
    try:
        paths = write_upload_analysis(session.session_id, role_report, {**(plan or {}), "preprocess": preprocess_report}, final_text)
        session.upload_analysis_paths = paths
        if paths:
            session.add_message("assistant", "数据可制图性审查报告已保存：" + str(paths.get("数据审查") or paths.get("report_dir") or ""))
    except Exception:
        pass
    return True

def _render_preprocess_box(session: SessionState):
    report = getattr(session, "preprocess_report", {}) or {}
    if not report:
        return html.Div("数据预处理将在上传后自动执行；制图前必须先通过预处理检查。", className="muted-line")
    gate = report.get("gate") or {}
    ready = report.get("ready_covariates") or []
    missing = report.get("missing_default_covariates") or []
    status = str(report.get("status") or "unknown")
    status_cls = "ok" if status == "ready_for_baseline" else "warn"
    rows = [
        html.Div(className="preprocess-summary-line", children=[
            html.Span("预处理状态：", className="preprocess-label"),
            html.Span(status, className=f"data-role-tag {status_cls}"),
        ]),
        html.Div(f"训练样点：{'已就绪' if gate.get('has_training_sample') else '未识别/未通过'}；可用协变量：{gate.get('ready_covariate_count', 0)} 个；待转换：{gate.get('pending_conversion_count', 0)} 个。", className="preprocess-summary-line"),
        html.Div("结论：" + str(gate.get("message") or ""), className="preprocess-summary-line"),
    ]
    rows.append(html.Details(open=False, className="preprocess-details", children=[
        html.Summary("查看预处理详情"),
        html.Div("已就绪协变量：" + ("、".join(ready) if ready else "暂无"), className="muted-line"),
        html.Div("建议补充默认协变量：" + ("、".join(missing) if missing else "无"), className="muted-line"),
        html.Pre(str(summarize_preprocess_for_user(report)), className="preprocess-report-text"),
    ]))
    return html.Div(className="preprocess-box", children=rows)




def _session_uploaded_raster_count(session: SessionState) -> int:
    n = 0
    for f in getattr(session, "latest_uploaded_files", []) or []:
        p = str((f or {}).get("path") or (f or {}).get("name") or "").lower()
        if p.endswith((".tif", ".tiff", ".nc", ".hdf", ".h5", ".hdf5", ".img")):
            n += 1
    return n

def _has_training_ready_fused_csv(session: SessionState) -> bool:
    report = getattr(session, "preprocess_report", {}) or {}
    gate = report.get("gate") or {}
    if bool(gate.get("has_training_sample")):
        # A sample table can be a plain point sample or a fused training CSV.
        # The RFK core will dynamically detect covariate columns from the final model CSV.
        return True
    summary = ((getattr(session, "data_role_report", {}) or {}).get("summary") or {})
    return int(summary.get("sample_count") or 0) > 0

def _set_pending_mapping(session: SessionState, request_text: str, reason: str) -> None:
    session.pending_mapping_request_text = str(request_text or "").strip() or "继续执行土壤有机质制图"
    session.pending_mapping_reason = str(reason or "waiting_for_prediction_raster")
    session.has_requested_map = True

def _pop_pending_mapping_text(session: SessionState, fallback: str = "继续执行土壤有机质制图") -> str:
    txt = str(getattr(session, "pending_mapping_request_text", "") or "").strip() or fallback
    session.pending_mapping_request_text = None
    session.pending_mapping_reason = None
    return txt


def _mapping_data_gate(session: SessionState, choice: str | None = None) -> dict:
    """Post-agent safety gate for mapping data readiness.

    This function is deliberately NOT an intent router. It is only called after
    the AI-first agent has already returned mode=start_rfk.

    Data-source rules:
    - default: built-in local default data; upload is not required.
    - user/pro_user: uploaded training sample + at least one ready covariate required.
    - pro_gee/pro_domestic/pro_web: online platform may fill covariates; if files
      were uploaded, a training sample must be identifiable.
    """
    report = getattr(session, "preprocess_report", {}) or {}
    gate = report.get("gate") or {}
    has_sample = bool(gate.get("has_training_sample"))
    ready_cov_count = int(gate.get("ready_covariate_count") or 0)
    normalized_choice = str(choice or getattr(session, "active_data_source", "default") or "default").strip().lower()

    if not getattr(session, "latest_uploaded_files", []):
        if normalized_choice in {"user", "uploaded", "pro_user"}:
            return {
                "ok": False,
                "title": "请先上传数据",
                "message": "你选择只使用上传数据，因此需要先上传土壤有机质训练样点和至少一个可入模环境协变量。",
            }
        return {
            "ok": True,
            "title": "使用默认数据",
            "message": "当前未上传数据，将使用电脑内置默认样点和默认环境协变量进行基线制图。",
            "ready_covariate_count": 0,
        }

    if not has_sample:
        return {"ok": False, "title": "缺少训练样点", "message": "尚未识别到包含经纬度和有机质/SOM字段的训练样点，不能进入制图。请先上传采样点CSV/Excel。"}
    if normalized_choice in {"user", "uploaded", "pro_user"} and ready_cov_count <= 0:
        raster_count = _session_uploaded_raster_count(session)
        if raster_count > 0:
            return {"ok": True, "title": "允许降级制图", "message": "已识别训练CSV/样点和至少一个预测栅格。正式制图流程将按最终融合CSV动态识别训练特征，并仅用有栅格支撑的特征输出GeoTIFF；缺少栅格的CSV特征只用于训练诊断。", "ready_covariate_count": ready_cov_count, "uploaded_raster_count": raster_count, "degraded_mapping": True}
        if _has_training_ready_fused_csv(session):
            return {"ok": True, "title": "仅训练验证", "message": "已识别可训练CSV，但当前没有预测栅格。正式制图流程将先执行训练/验证与模型诊断；若需正式GeoTIFF，请继续上传与CSV特征对应的TIF/NC/HDF栅格。", "ready_covariate_count": 0, "diagnostic_only": True}
        return {"ok": False, "title": "用户数据不足", "message": "你选择只使用上传数据制图，但当前没有通过预处理的全域环境协变量栅格。请上传至少一个TIF/NC/HDF协变量，或选择TPDC/GEE补齐缺失项。"}
    return {"ok": True, "title": "数据预处理已通过", "message": gate.get("message") or "可以进入下一步。", "ready_covariate_count": ready_cov_count}



def _analysis_intent_requested(text: str) -> bool:
    """Whether the user explicitly asked to examine/assess uploaded data."""
    return bool(re.search(r"(分析|审查|检查|判断|评估|合理|能否|是否|可以.*制图|能不能.*制图|数据.*制图)", str(text or "")))


def _defer_messages_until_map_loaded(session: SessionState, messages, reason: str = "layer") -> None:
    """Queue chat messages until the browser confirms the active map layer is visible.

    The server can finish file ingestion or start a modeling task before Leaflet has
    actually drawn the preview layer.  For user-facing confirmations that depend on
    the map being visible, store them in task_memory and let assets/layout_editor.js
    append/persist them after imageOverlay load.
    """
    if not messages:
        return
    mem = dict(getattr(session, "task_memory", {}) or {})
    key = "pending_style_loaded_messages" if reason == "style" else "pending_layer_loaded_messages"
    pending = list(mem.get(key) or [])
    seen = {str(x.get("text") if isinstance(x, dict) else x) for x in pending}
    now = int(time.time() * 1000)
    for msg in messages if isinstance(messages, (list, tuple)) else [messages]:
        text = str(msg or "").strip()
        if not text or text in seen:
            continue
        pending.append({"text": text, "created_at": now, "reason": reason})
        seen.add(text)
    mem[key] = pending[-30:]
    session.task_memory = mem


def _run_upload_review_if_ready(session: SessionState, trigger_text: str | None = None, defer_until_map_loaded: bool = False) -> bool:
    """Run the AI upload review once sample + covariates are both available.

    This is used by both Dash upload and AI/local-path ingestion. It keeps the
    detailed review in chat history and report files, not in the right panel.
    """
    role_report = getattr(session, "data_role_report", {}) or {}
    plan = getattr(session, "covariate_readiness_plan", {}) or {}
    preprocess_report = getattr(session, "preprocess_report", {}) or {}
    summary = role_report.get("summary") or {}
    sample_count = int(summary.get("sample_count") or 0)
    cov_count = int(summary.get("model_covariate_count") or 0)
    if sample_count <= 0 or cov_count <= 0:
        session.ai_covariate_review = None
        session.upload_analysis_paths = {}
        return False
    # Avoid repeating the same long review on every refresh/import unless the
    # user explicitly asks to analyze/check the data again.
    inventory_sig = json.dumps({
        "files": sorted([str(x.get("path") or x.get("name") or "") for x in (role_report.get("items") or [])]),
        "sample": sample_count,
        "cov": cov_count,
    }, ensure_ascii=False)
    prev_sig = getattr(session, "upload_review_inventory_sig", None)
    if prev_sig == inventory_sig and not _analysis_intent_requested(trigger_text or ""):
        return False
    ai_review = generate_ai_covariate_review(role_report, {**plan, "preprocess": preprocess_report}, chat_history=session.chat_history)
    paths = write_upload_analysis(session.session_id, role_report, plan, ai_review)
    session.ai_covariate_review = ai_review
    session.upload_analysis_paths = paths
    session.upload_review_inventory_sig = inventory_sig
    review_messages = []
    if ai_review:
        review_messages.append("我已完成对上传数据的综合审查：\n" + str(ai_review).strip())
    if paths:
        report_dir = str(paths.get("report_dir") or "")
        file_names = []
        for k, v in (paths or {}).items():
            if k == "report_dir" or not v:
                continue
            try:
                file_names.append(Path(str(v)).name)
            except Exception:
                file_names.append(str(v))
        if report_dir:
            msg_lines = ["上传数据分析报告已保存。", f"文件夹：{report_dir}"]
            if file_names:
                msg_lines.append("报告：" + "、".join(file_names))
            review_messages.append("\n".join(msg_lines))
    if review_messages:
        if defer_until_map_loaded:
            _defer_messages_until_map_loaded(session, review_messages, reason="layer")
        else:
            for m in review_messages:
                session.add_message("assistant", m)
    return True

def _render_upload_analysis_box(session: SessionState):
    """Very compact upload recognition box for the right panel.

    Right panel only keeps status; detailed file review and AI conclusions are
    written into the chat history so the interaction remains AI-first.
    """
    role_report = getattr(session, "data_role_report", {}) or {}
    if not role_report:
        return html.Div(id="uploaded-file-role-detail", style={"display": "none"})

    items = role_report.get("items") or []
    summary = role_report.get("summary") or {}
    if not items:
        return html.Div(id="uploaded-file-role-detail", style={"display": "none"})
    sample_count = int(summary.get("sample_count") or 0)
    cov_count = int(summary.get("model_covariate_count") or 0)
    pending_count = int(summary.get("pending_conversion_count") or 0)
    status_text = f"已识别 {len(items)} 个文件：样点 {sample_count} 个，协变量 {cov_count} 个，待转换 {pending_count} 个。"
    if sample_count > 0 and cov_count > 0:
        hint = "数据审查结果已写入左侧对话历史。"
    elif cov_count > 0:
        hint = "已接收协变量；等待样点，或直接告诉我使用默认样点/内置数据。"
    elif sample_count > 0:
        hint = "已接收样点；等待协变量，或直接告诉我使用默认环境协变量。"
    else:
        hint = "我会继续等待可用于制图的样点或协变量。"
    return html.Div(className="upload-analysis-box compact-only", children=[
        html.Div(status_text, className="upload-role-summary-line"),
        html.Div(hint, className="muted-line tiny"),
        html.Div(id="uploaded-file-role-detail", style={"display": "none"}),
    ])


def _render_upload_role_detail(it: dict | None):
    if not it:
        return html.Div("暂无文件详情。", className="muted-line")
    can = it.get("can_model")
    if can is True:
        tag = "可作为环境协变量入模"
        tag_cls = "ok"
    elif can == "training_target":
        tag = "训练样点"
        tag_cls = "ok"
    elif str(can) in {"after_conversion", "after_interpolation", "after_rasterization"}:
        tag = str(can)
        tag_cls = "warn"
    else:
        tag = "不入模"
        tag_cls = "warn"
    meta = it.get("spatial_meta") or {}
    meta_lines = []
    if meta:
        if meta.get("crs"):
            meta_lines.append(f"CRS：{meta.get('crs')}")
        if meta.get("resolution"):
            meta_lines.append(f"分辨率：{meta.get('resolution')}")
        if meta.get("valid_ratio") is not None:
            try:
                meta_lines.append(f"有效像元比例：{float(meta.get('valid_ratio')):.3f}")
            except Exception:
                pass
    return html.Div([
        html.Div(str(it.get("display_name") or it.get("name") or "--"), className="uploaded-file-detail-name"),
        html.Div(className="uploaded-file-detail-line", children=[
            html.Span(str(it.get("label") or it.get("role") or "--"), className="uploaded-file-detail-role"),
            html.Span(tag, className=f"data-role-tag {tag_cls}"),
            html.Span("置信度：" + str(it.get("confidence") if it.get("confidence") is not None else "--"), className="uploaded-file-detail-confidence"),
        ]),
        html.Div("匹配协变量：" + (covariate_display_name(it.get("matched_covariate"), with_code=True) if it.get("matched_covariate") else "无"), className="uploaded-file-detail-meta"),
        html.Div(str(it.get("reason") or ""), className="uploaded-file-detail-reason"),
        html.Div("；".join(meta_lines), className="uploaded-file-detail-meta") if meta_lines else html.Div(),
    ])


def _render_upload_panel(session: SessionState) -> html.Div:
    """Right-side upload panel. Keeps the chat column available for conversation."""
    return html.Div(className="right-upload-panel", children=[
        html.Div(className="panel-title upload-panel-title", children="上传数据（样点/协变量均可先传）"),
        dcc.Upload(
            id="upload-data",
            children=html.Div([
                html.Div("拖拽文件到这里或点击上传", className="upload-drop-main"),
            ]),
            multiple=True,
            className="right-upload-drop",
        ),
        html.Div(
            id="upload-status",
            className="upload-status-wrap right-upload-status",
            children=_render_upload_feedback_box(getattr(session, "latest_upload_feedback", None)),
        ),
        html.Div(
            id="upload-analysis-status",
            className="upload-analysis-wrap",
            children=_render_upload_analysis_box(session),
        ),
    ])

def _result_info_panel(session: SessionState, ramp_default: str, reverse_default: list[str]) -> html.Div:
    """Right side panel kept intentionally compact.

    The map layer is selected only in the center workspace. Upload analysis,
    AI covariate review and report paths are written into chat history instead
    of taking over the right panel.
    """
    stage_payload = build_stage_payload(session)
    result = stage_payload.get("result") or {}
    style_prefs = getattr(session, "map_style_prefs", {}) or {}
    show_basemap_default = style_prefs.get("show_basemap")
    if show_basemap_default is None:
        show_basemap_default = not bool(getattr(session, "latest_uploaded_files", []))
    if result.get("kind") in {"som_map", "gcp_map"} or session.latest_result_kind in {"som_map", "gcp_map"}:
        show_samples_default = False
    else:
        show_samples_default = style_prefs.get("show_samples")
        if show_samples_default is None:
            sample_payload = stage_payload.get("uploaded_samples") or {}
            # During upload/covariate preview, sample points remain visible above
            # rasters.  Once a formal map is produced, final layouts hide samples.
            show_samples_default = bool(sample_payload.get("point_count") or sample_payload.get("points"))

    running_notice = []
    rfk_task = TASKS.get(session.latest_rfk_task_id) if getattr(session, "latest_rfk_task_id", None) else None
    gcp_task = TASKS.get(session.latest_gcp_task_id) if getattr(session, "latest_gcp_task_id", None) else None

    def _task_notice(panel_title: str, running_title: str, done_title: str, error_title: str, task_obj):
        progress_value = int(getattr(task_obj, "progress", 0) or 0)
        stage_value = getattr(task_obj, "stage", None) or "等待开始"
        title_value = running_title
        if getattr(task_obj, "status", None) == "done":
            title_value = done_title
            progress_value = 100
            stage_value = "已完成"
        elif getattr(task_obj, "status", None) == "error":
            title_value = error_title
            progress_value = 100
            stage_value = "运行失败"
        return [
            html.Div(className="panel-title", children=panel_title),
            html.Div(title_value, className="task-running-title"),
            progress_bar(progress_value, stage_value),
        ]

    show_gcp_progress = bool(
        gcp_task
        and (
            session.latest_result_kind in {"gcp_running", "gcp_map"}
            or bool(getattr(session, "gcp_shown", False))
            or getattr(gcp_task, "status", None) in {"queued", "running"}
        )
    )
    show_rfk_progress = bool(rfk_task and getattr(rfk_task, "status", None) in {"queued", "running", "done", "error"})

    # A new RFK run should own the progress panel while it is running. After SOM
    # mapping has completed and the user explicitly starts GCP/AOA, the same
    # right-side panel switches to the uncertainty-analysis workflow and remains
    # there after completion.
    if rfk_task and getattr(rfk_task, "status", None) in {"queued", "running"}:
        running_notice = _task_notice("制图实施进度", "正在执行土壤有机质制图", "土壤有机质制图已完成", "土壤有机质制图失败", rfk_task)
    elif show_gcp_progress:
        running_notice = _task_notice("不确定性分析实施进度", "正在执行 GCP + AOA 不确定性分析", "不确定性分析已完成", "不确定性分析失败", gcp_task)
    elif show_rfk_progress:
        running_notice = _task_notice("制图实施进度", "正在执行土壤有机质制图", "土壤有机质制图已完成", "土壤有机质制图失败", rfk_task)

    def _current_result_text() -> str:
        kind = result.get("kind") or session.latest_result_kind or ""
        name = result.get("source_name") or result.get("title") or ""
        if kind == "uploaded_raster":
            return "上传栅格：" + (str(name) if name else "未命名图层")
        if kind == "uploaded_raster_stack":
            return "上传协变量叠加预览"
        if kind == "som_map":
            return str(stage_payload.get("title") or result.get("title") or "有机质制图结果")
        if kind == "gcp_map":
            return str(stage_payload.get("title") or result.get("title") or "不确定性分析结果")
        if kind == "rfk_running":
            return "有机质制图运行中"
        if kind == "gcp_running":
            return "不确定性分析运行中"
        return "暂无"

    style_catalog = get_style_catalog()
    palette_cards = []
    summary_colors = None
    for opt in style_catalog.get("palettes") or []:
        colors = opt.get("colors") or COLOR_RAMPS.get(opt.get("value"), ["#ffffff", "#d9e2ef", "#1f77b4"])
        label = str(opt.get("label") or opt.get("value") or "").split("｜", 1)[-1]
        if opt.get("value") == ramp_default:
            summary_colors = colors
        palette_cards.append(
            html.Button(
                [
                    html.Span(className="palette-swatch-bar", style={"background": f"linear-gradient(90deg, {', '.join(colors)})"}),
                    html.Span(label, className="palette-swatch-label"),
                ],
                className=("palette-swatch-btn active" if opt.get("value") == ramp_default else "palette-swatch-btn"),
                title=str(opt.get("label") or opt.get("value") or ""),
                type="button",
                **{
                    "data-palette": str(opt.get("value") or ""),
                    "data-colors": json.dumps(colors, ensure_ascii=False),
                },
            )
        )
    if not summary_colors:
        summary_colors = COLOR_RAMPS.get(ramp_default, ["#fff", "#ddd"])

    return html.Div([
        html.Div("数据与结果", className="right-title"),
        _render_upload_panel(session),
        _render_download_progress_panel(session),
        *running_notice,
        _render_task_step_panel(*_active_model_task(session)),
        html.Div(className="panel-title", children="当前结果"),
        html.Div(_current_result_text(), id="result-kind-box", className="value-box"),

        # 样式、底图、样点显隐保留为隐藏状态锚点，由 AI 指令和前端地图逻辑控制，
        # 不再让用户在右侧手动点选一堆制图样式控件。
        html.Div(style={"display": "none"}, children=[
            dcc.Checklist(
                id="basemap-toggle-input",
                options=[{"label": "显示参考底图", "value": "show"}],
                value=(["show"] if show_basemap_default else []),
                className="native-checklist",
            ),
            dcc.Checklist(
                id="sample-points-toggle-input",
                options=[{"label": "显示样点", "value": "show"}],
                value=(["show"] if show_samples_default else []),
                className="native-checklist",
            ),
            html.Select(
                id="result-ramp-select",
                className="native-select",
                children=[
                    html.Option(opt["label"], value=opt["value"], selected=(opt["value"] == ramp_default))
                    for opt in palette_options()
                ],
            ),
            dcc.Input(
                id="result-opacity-range",
                type="range",
                min=0.15,
                max=1.0,
                step=0.05,
                value=(style_prefs.get("opacity") if style_prefs.get("opacity") is not None else (result.get("opacity") if result.get("opacity") is not None else 1.0)),
                className="native-input",
            ),
            dcc.Checklist(
                id="result-reverse-input",
                options=[{"label": "反转色带", "value": "reverse"}],
                value=(reverse_default or []),
                className="native-checklist",
            ),
            html.Button("应用结果层样式", id="apply-result-style-btn"),
            html.Div(id="palette-summary-label"),
            html.Div(id="palette-grid"),
        ]),

        # Hidden compatibility anchors for the existing layout_editor.js handlers.
        html.Div(id="layout-layer-list", style={"display": "none"}),
        html.Div(id="selected-id", style={"display": "none"}),
        html.Button("", id="btn-layer-up", style={"display": "none"}),
        html.Button("", id="btn-layer-down", style={"display": "none"}),
        html.Button("", id="btn-layer-top", style={"display": "none"}),
        html.Button("", id="btn-layer-bottom", style={"display": "none"}),
        html.Button("", id="btn-align-h", style={"display": "none"}),
        html.Button("", id="btn-align-v", style={"display": "none"}),
        html.Button("", id="btn-duplicate-item", style={"display": "none"}),
        html.Button("", id="btn-save-template", style={"display": "none"}),
        html.Button("", id="btn-load-template", style={"display": "none"}),
        html.Button("", id="btn-delete-item", style={"display": "none"}),
        html.Button("", id="btn-export-png", style={"display": "none"}),
        html.Button("", id="btn-export-pdf", style={"display": "none"}),
        html.Button("", id="btn-export-tif", style={"display": "none"}),
    ], className="side-panel")



# =========================
# 登录 / 注册 / 找回密码页面
# =========================

def _auth_flash_box(flash: dict | None):
    if not flash or not flash.get("text"):
        return html.Div(id="auth-flash-box", className="auth-flash empty")
    typ = flash.get("type") or "info"
    return html.Div(flash.get("text"), id="auth-flash-box", className=f"auth-flash {typ}")


def _auth_input(input_id: str, placeholder: str = "", value: str = "", input_type: str = "text"):
    return dcc.Input(
        id=input_id,
        type=input_type,
        value=value or "",
        placeholder=placeholder,
        autoComplete="off",
        className="auth-input",
    )


def _auth_shell(title: str, subtitle: str, children, side_children=None, flash: dict | None = None):
    return html.Div(className="auth-page", children=[
        html.Div(className="auth-bg-orb auth-bg-orb-a"),
        html.Div(className="auth-bg-orb auth-bg-orb-b"),
        html.Div(className="auth-top-brand", children=[
            html.Div("数字土壤制图智能体", className="auth-brand-title"),
            html.Div("Digital Soil Mapping Agent · 正式交付版", className="auth-brand-subtitle"),
        ]),
        html.Div(className="auth-card-wrap", children=[
            html.Div(className="auth-main-card", children=[
                html.Div(title, className="auth-card-title"),
                html.Div(subtitle, className="auth-card-subtitle"),
                _auth_flash_box(flash),
                html.Div(children, className="auth-form"),
            ]),
            html.Div(className="auth-side-card", children=side_children or [
                html.Div("标准版 / Pro 版", className="auth-side-title"),
                html.Div("标准版可上传自有数据或使用本机内置数据；Pro 版可按年份与地区执行自动数据准备，价格 20￥/月。", className="auth-side-text"),
            ]),
        ]),
        html.Div("提示：账号注册与找回密码仅支持邮箱验证码。请使用可正常接收邮件的邮箱完成验证。", className="auth-footnote"),
    ])


def _render_login_page(memory: dict | None = None, flash: dict | None = None):
    memory = memory or {}
    remember_values = ["remember"] if memory.get("remember") else []
    return _auth_shell(
        title="登录账号",
        subtitle="登录后进入土壤有机质制图、GCP 不确定性分析与专题图布局工作区。",
        flash=flash,
        children=[
            html.Div(className="auth-field", children=[html.Label("用户名 / 邮箱"), _auth_input("login-name", "请输入用户名或邮箱", memory.get("username", ""))]),
            html.Div(className="auth-field", children=[html.Label("密码"), _auth_input("login-password", "请输入密码", memory.get("password", ""), "password")]),
            html.Div(className="auth-row between", children=[
                dcc.Checklist(id="login-remember", options=[{"label": "记住账号密码", "value": "remember"}], value=remember_values, className="auth-check", inputStyle={"marginRight": "6px"}),
                html.Button("找回密码", id="login-reset-link", type="button", className="auth-link-btn"),
            ]),
            html.Button("登录", id="login-submit", type="button", className="auth-primary-btn"),
            html.Div(className="auth-switch-line", children=["还没有账号？", html.Button("立即注册", id="login-register-link", type="button", className="auth-link-btn")]),
        ],
        side_children=[
            html.Div("系统能力", className="auth-side-title"),
            html.Ul(className="auth-feature-list", children=[
                html.Li("土壤有机质 制图"),
                html.Li("GCP 不确定性分析"),
                html.Li("ArcGIS Pro / QGIS 风格布局编辑"),
                html.Li("标准版 / Pro 版权限"),
            ]),
            html.Div("忘记密码时，请通过注册邮箱验证码重置。", className="auth-side-tip"),
        ],
    )


def _render_register_page(agreed: bool = False, flash: dict | None = None):
    contact_exists = bool(flash and flash.get("reason") == "contact_exists")
    return _auth_shell(
        title="注册账号",
        subtitle="注册后默认进入标准版。需要自动数据准备时，可升级为 Pro 版。",
        flash=flash,
        children=[
            html.Div(className="auth-field code-field", children=[html.Label("邮箱"), _auth_input("register-contact", "请输入邮箱地址"), html.Button("获取验证码", id="register-code-btn", type="button", className="auth-code-btn")]),
            html.Div(id="register-code-msg", className="auth-inline-msg"),
            html.Div(className="auth-field", children=[html.Label("用户名"), _auth_input("register-username", "请设置用户名")]),
            html.Div(className="auth-field", children=[html.Label("密码"), _auth_input("register-password", "至少 6 位", input_type="password")]),
            html.Div(className="auth-field", children=[html.Label("确认密码"), _auth_input("register-password2", "请再次输入密码", input_type="password")]),
            html.Div(className="auth-field", children=[html.Label("验证码"), _auth_input("register-code", "请输入验证码")]),
            html.Div(className="auth-row agreement-row", children=[
                dcc.Checklist(id="register-agreed", options=[{"label": "您已阅读并同意", "value": "yes"}], value=(["yes"] if agreed else []), className="auth-check", inputStyle={"marginRight": "6px"}),
                html.Button("用户协议", id="register-agreement-link", type="button", className="auth-link-btn strong"),
            ]),
            html.Button("注册", id="register-submit", type="button", className="auth-primary-btn"),
            html.Div(id="register-duplicate-action", className="auth-duplicate-action", children=(
                html.Button("该账号可能已存在，点击找回密码", id="register-duplicate-reset-btn", type="button", className="auth-secondary-btn") if contact_exists else ""
            )),
        ],
        side_children=[
            html.Div("已有账号？", className="auth-side-title"),
            html.Button("直接登录", id="register-login-link", type="button", className="auth-side-login-btn"),
            html.Div("注册邮箱用于接收验证码、登录和找回密码。当前系统仅开放邮箱注册。", className="auth-side-text"),
        ],
    )


def _render_reset_page(flash: dict | None = None):
    return _auth_shell(
        title="找回密码",
        subtitle="请输入注册时使用的邮箱，并填写系统发送的验证码。",
        flash=flash,
        children=[
            html.Div(className="auth-field code-field", children=[html.Label("邮箱"), _auth_input("reset-contact", "请输入注册邮箱"), html.Button("获取验证码", id="reset-code-btn", type="button", className="auth-code-btn")]),
            html.Div(id="reset-code-msg", className="auth-inline-msg"),
            html.Div(className="auth-field", children=[html.Label("验证码"), _auth_input("reset-code", "请输入验证码")]),
            html.Div(className="auth-field", children=[html.Label("新密码"), _auth_input("reset-password", "请输入新密码", input_type="password")]),
            html.Div(className="auth-field", children=[html.Label("确认密码"), _auth_input("reset-password2", "请再次输入新密码", input_type="password")]),
            html.Button("确认修改", id="reset-submit", type="button", className="auth-primary-btn"),
        ],
        side_children=[
            html.Div("已有账号？", className="auth-side-title"),
            html.Button("返回登录", id="reset-login-link", type="button", className="auth-side-login-btn"),
            html.Div("如果没有注册记录，请先进入注册页面创建账号。", className="auth-side-text"),
        ],
    )


def _render_agreement_page(flash: dict | None = None):
    agreement_text = [
        "一、系统用途：本系统为数字土壤制图智能体正式应用平台，用于土壤有机质制图、不确定性分析和专题图布局输出。",
        "二、账号与权限：系统包含标准版和 Pro 版。Pro 版用于自动数据准备能力，价格为 20￥/月；支付与权限开通按本地部署配置执行。",
        "三、数据责任：用户上传的数据应确保来源合法、内容准确。系统输出结果仅用于学习、科研和技术验证，不应直接作为正式生产决策的唯一依据。",
        "四、外部数据源：若后续接入 GEE、NASA、Copernicus 等外部平台，应遵守对应平台的教育、科研和非商业使用规定。",
        "五、隐私与安全：系统会保存注册账号、密码哈希和任务记录。密码不会以明文形式保存。验证码能力按本地部署配置执行。",
        "六、使用限制：不得利用系统进行违法用途、批量恶意注册、攻击外部数据平台或违反数据平台服务条款的行为。",
        "七、免责声明：本系统处于原型开发阶段，模型精度、数据完整性和制图结果仍需结合专业判断进行解释。",
    ]
    return html.Div(className="agreement-page", children=[
        html.Div(className="agreement-card", children=[
            html.Div("用户协议内容", className="agreement-title"),
            _auth_flash_box(flash),
            html.Div(className="agreement-scroll", children=[html.P(t) for t in agreement_text] + [
                html.P("请下拉阅读后选择是否同意。选择同意后将自动返回注册页面并勾选协议。")
            ]),
            html.Div(className="agreement-actions", children=[
                html.Button("同意协议", id="agreement-accept-btn", type="button", className="auth-primary-btn"),
                html.Button("不同意协议", id="agreement-reject-btn", type="button", className="auth-danger-outline-btn"),
            ]),
        ])
    ])


def _render_auth_page(page: str | None, memory: dict | None, agreed: bool, flash: dict | None):
    if page == "register":
        return _render_register_page(agreed=agreed, flash=flash)
    if page == "reset":
        return _render_reset_page(flash=flash)
    if page == "agreement":
        return _render_agreement_page(flash=flash)
    return _render_login_page(memory=memory, flash=flash)

DOMESTIC_MANUAL_REQUIRED_MARKER = "DOMESTIC_MANUAL_REQUIRED_JSON::"

def _parse_domestic_handoff_error(error: str | None) -> dict | None:
    if not error or DOMESTIC_MANUAL_REQUIRED_MARKER not in str(error):
        return None
    raw = str(error)
    try:
        part = raw.split(DOMESTIC_MANUAL_REQUIRED_MARKER, 1)[1]
        json_text = part.split("\n", 1)[0].strip()
        payload = json.loads(json_text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {
            "title": "需要人工接管国内平台下载",
            "message": "国内平台真实数据下载未完成，请打开平台完成登录/注册/下载后重试。",
            "raw_error": raw[:3000],
        }
    return None



def _domestic_manifest_load_for_ui(manifest_path: str | None) -> dict:
    if not manifest_path:
        return {}
    try:
        p = Path(str(manifest_path))
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_domestic_live_state(manifest_path: str | None) -> dict:
    manifest = _domestic_manifest_load_for_ui(manifest_path)
    state_path = manifest.get("browser_live_state") or ""
    if not state_path:
        return {"status": manifest.get("status") or "pending", "message": "正在等待 浏览器浏览器进程写入状态。"}
    try:
        p = Path(str(state_path))
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception as exc:
        return {"status": "state_read_failed", "message": f"读取浏览器状态失败：{exc}"}
    return {"status": manifest.get("status") or "pending", "message": "浏览器已启动，正在等待平台页面状态。"}

def _projection_placeholder_src(message: str = "平台状态暂不可用") -> str:
    """Return an inline SVG placeholder instead of showing a black/broken image."""
    safe = str(message or "平台状态暂不可用")[:80]
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">'
        '<rect width="1280" height="720" rx="24" fill="#f8fafc"/>'
        '<rect x="40" y="40" width="1200" height="640" rx="24" fill="#eef2ff" stroke="#cbd5e1" stroke-width="3"/>'
        f'<text x="640" y="320" font-family="Microsoft YaHei, Arial" font-size="42" fill="#1e3a8a" text-anchor="middle" font-weight="700">{safe}</text>'
        '<text x="640" y="382" font-family="Microsoft YaHei, Arial" font-size="26" fill="#64748b" text-anchor="middle">请以真实 Chrome 窗口为准；页面状态显示不会影响 FTP 下载。</text>'
        '</svg>'
    )
    return "data:image/svg+xml;charset=utf-8," + quote(svg)


def _read_ftp_download_status(manifest_path: str | None) -> dict:
    """Read ftp_download_status.json from the current manifest save directory."""
    manifest = _domestic_manifest_load_for_ui(manifest_path)
    candidates = []
    local_dir = manifest.get("local_save_dir") or manifest.get("capture_dir") or ""
    if local_dir:
        candidates.append(Path(str(local_dir)) / "ftp_download_status.json")
    try:
        state = _read_domestic_live_state(manifest_path)
        status_path = state.get("ftp_download_status_path")
        if not status_path and isinstance(state.get("ftp_download_process"), dict):
            status_path = state.get("ftp_download_process", {}).get("ftp_download_status_path")
        if status_path:
            candidates.append(Path(str(status_path)))
    except Exception:
        pass
    for p in candidates:
        try:
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data["_path"] = str(p)
                    try:
                        updated = int(float(data.get("updated_at") or 0))
                    except Exception:
                        updated = 0
                    # Ignore previous-run progress files. The user expects a full app exit
                    # to reset the visible progress area, so old ftp_download_status.json must
                    # not appear as an active download in a fresh Dash session.
                    if updated and updated < APP_BOOT_TS:
                        return {
                            "status": "not_started",
                            "message": "已忽略上一次运行遗留的 FTP 进度文件；本次尚未开始下载。",
                            "stale_status_path": str(p),
                            "stale_updated_at": updated,
                        }
                    return data
        except Exception as exc:
            return {"status": "read_failed", "error": str(exc), "_path": str(p)}
    return {"status": "not_started", "message": "尚未发现 FTP 下载状态文件。"}


def _format_bytes(n) -> str:
    try:
        n = float(n or 0)
    except Exception:
        return "未知"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    return f"{n:.2f} {units[i]}" if i else f"{int(n)} {units[i]}"


def _download_manifest_path_from_session(session: SessionState) -> str:
    """Return current TPDC download manifest path from the session, if any."""
    try:
        decision = getattr(session, "last_route_decision", {}) or {}
        for key in ("download_manifest", "manifest_path"):
            val = decision.get(key)
            if val:
                return str(val)
        result = decision.get("result") if isinstance(decision.get("result"), dict) else {}
        if result.get("manifest_path"):
            return str(result.get("manifest_path"))
    except Exception:
        pass
    try:
        paths = getattr(session, "uploaded_result_paths", {}) or {}
        if paths.get("download_manifest"):
            return str(paths.get("download_manifest"))
    except Exception:
        pass
    return ""




def _download_ui_marker_path(manifest_path: str | None, key: str) -> Path | None:
    try:
        if not manifest_path:
            return None
        p = Path(str(manifest_path))
        base = p.parent if p.suffix else p
        safe_key = re.sub(r"[^A-Za-z0-9_\-]", "_", str(key or "event"))
        return base / ".ui_once" / f"{safe_key}.done"
    except Exception:
        return None


def _download_ui_marker_seen(manifest_path: str | None, key: str) -> bool:
    p = _download_ui_marker_path(manifest_path, key)
    try:
        return bool(p and p.exists())
    except Exception:
        return False


def _download_ui_mark_seen(manifest_path: str | None, key: str) -> None:
    p = _download_ui_marker_path(manifest_path, key)
    try:
        if p:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(str(int(time.time())), encoding="utf-8")
    except Exception:
        pass

def _normalize_ftp_status_for_ui(manifest_path: str | None) -> dict:
    """Merge browser live state and ftp_download_status.json into one UI snapshot."""
    if not manifest_path:
        return {"active": False, "status": "not_started", "message": "尚未启动下载任务。"}
    live = _read_domestic_live_state(manifest_path)
    ftp = _read_ftp_download_status(manifest_path)
    status = str(ftp.get("status") or ftp.get("ftp_download_status") or live.get("status") or "waiting")
    watcher = str(live.get("ftp_watcher_status") or "")
    # If FTP status file does not exist yet, live state is still important after capture.
    if status == "not_started" and live.get("status") not in {None, "pending", "not_started"}:
        status = str(live.get("status") or status)
    total = ftp.get("remote_total_known_bytes") or ftp.get("total_known_bytes") or 0
    done = ftp.get("downloaded_bytes_total") or ftp.get("local_bytes_total") or 0
    percent = ftp.get("progress_percent")
    try:
        if percent is None and total and done:
            percent = max(0.0, min(100.0, float(done) / float(total) * 100.0))
        percent_f = max(0.0, min(100.0, float(percent or 0.0)))
    except Exception:
        percent_f = 0.0
    if status in {"ftp_download_completed", "download_verified"}:
        percent_f = 100.0
    host = ftp.get("host") or ftp.get("ftp_host") or (live.get("ftp_primary_account") or {}).get("host") or ""
    download_dir = ftp.get("download_root") or live.get("download_dir") or (_domestic_manifest_load_for_ui(manifest_path).get("local_save_dir") or "")
    current_remote = ftp.get("current_remote") or ""
    message = live.get("message") or ftp.get("message") or ftp.get("note") or status
    return {
        "active": bool(manifest_path),
        "status": status,
        "watcher": watcher,
        "message": str(message or ""),
        "ftp_accounts_count": live.get("ftp_accounts_count") or 0,
        "pid": (live.get("ftp_download_process") or {}).get("pid") if isinstance(live.get("ftp_download_process"), dict) else None,
        "host": host,
        "download_dir": download_dir,
        "total_bytes": total,
        "downloaded_bytes": done,
        "progress_percent": round(percent_f, 2),
        "current_index": ftp.get("current_index"),
        "file_count": ftp.get("file_count"),
        "current_remote": current_remote,
        "current_local": ftp.get("current_local") or "",
        "current_local_bytes": ftp.get("current_local_bytes") or 0,
        "current_remote_size": ftp.get("current_remote_size") or 0,
        "space_warning": ftp.get("space_warning") or "",
        "error": ftp.get("error") or live.get("error") or "",
        "status_path": (ftp.get("_path") or live.get("ftp_download_status_path") or ((live.get("ftp_download_process") or {}).get("ftp_download_status_path") if isinstance(live.get("ftp_download_process"), dict) else "")),
        "manifest_path": str(manifest_path or ""),
        "updated_at": ftp.get("updated_at") or live.get("ftp_watcher_updated_at") or live.get("updated_at") or "",
    }




def _infer_download_mapping_intent_from_session(session: SessionState, manifest_path: str | None) -> bool:
    """Infer whether a completed TPDC download should be integrated for mapping.

    Rule of thumb requested by the user:
    - If the user came only to download data, keep it as a download product.
    - If the user had already uploaded samples/covariates or discussed SOM
      mapping, treat the downloaded raster as a covariate for the current mapping
      workflow.
    """
    manifest = _domestic_manifest_load_for_ui(manifest_path)
    recent_text = " ".join(str(m.get("content") or "") for m in (getattr(session, "chat_history", []) or [])[-12:])
    request_text = str(manifest.get("request_text") or "") + " " + recent_text
    has_uploaded = bool(getattr(session, "latest_uploaded_files", []) or [])
    role_summary = ((getattr(session, "data_role_report", {}) or {}).get("summary") or {})
    has_samples = int(role_summary.get("sample_count") or 0) > 0
    has_covariates = int(role_summary.get("model_covariate_count") or 0) > 0
    explicit_mapping = bool(re.search(r"用于制图|用来制图|继续制图|补充协变量|入模|结合当前样点|结合当前数据|制图工作流|开始制图|重新制图|建模", request_text, re.I))
    explicit_only_download = bool(re.search(r"只(?:要|需|需要)?下载|仅下载|下载一下|先下载|不要制图|不制图", request_text, re.I))
    if _is_data_review_only_request(request_text):
        return False
    # A plain “下载某数据” after an earlier map/GCP must stay a download task
    # unless the current message explicitly says the download is for mapping.
    manifest_request = str(manifest.get("request_text") or "")
    if re.search(r"下载|获取", manifest_request) and not re.search(r"用于制图|用来制图|补充协变量|入模|继续制图|开始制图|重新制图|建模", manifest_request):
        return False
    if explicit_only_download:
        return False
    if explicit_mapping:
        return True
    return False


def _register_downloaded_tifs_as_covariates(session: SessionState, tif_paths: list[str]) -> None:
    """Add postprocessed TPDC GeoTIFFs to the current upload inventory."""
    if not tif_paths:
        return
    existing = list(getattr(session, "latest_uploaded_files", []) or [])
    seen = {str(x.get("path") or "") for x in existing if isinstance(x, dict)}
    added = []
    for tif in tif_paths:
        p = Path(str(tif))
        if not p.exists() or str(p) in seen:
            continue
        try:
            validation = validate_uploaded_file(str(p))
        except Exception as exc:
            validation = {"ok": False, "stage": "post_download_validation_failed", "message": str(exc)}
        rec = {
            "name": p.name,
            "original_name": p.name,
            "path": str(p),
            "size": p.stat().st_size if p.exists() else 0,
            "role_guess": "raster_sample_or_covariate",
            "validation": validation,
            "usable_for_modeling": bool((validation or {}).get("ok", True)),
            "source": "tpdc_download_postprocess",
        }
        existing.append(rec)
        added.append(rec)
        seen.add(str(p))
    if added:
        session.latest_uploaded_files = existing
        session.active_data_source = "user"
        try:
            role_report = build_role_report(existing)
            plan = build_covariate_plan(role_report)
            preprocess_report = run_upload_preprocessing(session.session_id, existing, role_report=role_report, covariate_plan=plan)
            session.data_role_report = role_report
            session.covariate_readiness_plan = plan
            session.preprocess_report = preprocess_report
            session.preprocess_paths = preprocess_report.get("paths") or {}
        except Exception as exc:
            session.add_message("assistant", "下载数据已加入会话，但重新生成协变量预处理计划失败：" + str(exc))


def _explain_ftp_download_error(error_text: str) -> str:
    """Return a user-facing explanation for common TPDC FTP errors."""
    e = str(error_text or "")
    if "421" in e and "Home directory" in e:
        return "FTP服务器返回 421 Home directory not available，表示当前捕获到的FTP账号登录后没有可用的默认目录，常见原因是该账号/主机组合已失效、数据集FTP入口不匹配、临时账号过期，或需要重新在TPDC详情页点击当前数据集的下载按钮重新生成FTP参数。"
    if "530" in e or "Login incorrect" in e:
        return "FTP登录失败，通常是用户名/密码已过期或验证码登录后生成的临时FTP账号无效。请重新进入TPDC数据详情页点击下载，重新捕获FTP参数。"
    if "timed out" in e.lower() or "timeout" in e.lower():
        return "FTP连接超时，通常与网络、服务器繁忙或被动模式端口不可达有关。可以稍后重试，或使用生成的WinSCP脚本接管下载。"
    if "No such file" in e or "550" in e:
        return "FTP路径不存在或没有权限访问，可能是自动选择的远程目录不属于当前数据集，需要重新选择正确数据集或重新捕获FTP下载入口。"
    return "请查看任务目录中的 ftp_python_downloader.log 和 open_winscp_download.bat；必要时重新从TPDC详情页点击下载以生成新的FTP参数。"

def _render_download_progress_panel(session: SessionState):
    manifest_path = _download_manifest_path_from_session(session)
    if not manifest_path:
        return None
    decision = getattr(session, "last_route_decision", {}) or {}
    snap = decision.get("download_progress_snapshot") if isinstance(decision.get("download_progress_snapshot"), dict) else None
    status = snap or _normalize_ftp_status_for_ui(manifest_path)
    st = str(status.get("status") or "waiting")
    watcher = str(status.get("watcher") or "")
    percent = float(status.get("progress_percent") or 0.0)
    status_label_map = {
        "ftp_account_waiting": "等待 FTP 弹窗",
        "ftp_credentials_detected": "已捕捉 FTP 信息",
        "ftp_download_process_started": "已启动 FTP 下载",
        "ftp_connecting": "正在连接 FTP",
        "ftp_downloading": "正在下载",
        "ftp_download_completed": "下载完成",
        "download_verified": "下载完成",
        "ftp_download_partial": "部分下载完成",
        "ftp_download_failed": "下载失败",
        "ftp_host_failed": "主机连接失败",
        "not_started": "等待下载",
    }
    title = status_label_map.get(st, st or "下载任务")
    tone = "ok" if st in {"ftp_credentials_detected", "ftp_download_process_started", "ftp_downloading", "ftp_download_completed", "download_verified"} else ("error" if "failed" in st else "wait")
    meta_lines = []
    if status.get("ftp_ticket_count") or status.get("ftp_accounts_count"):
        ticket_count = status.get("ftp_ticket_count") or status.get("ftp_accounts_count")
        host_count = status.get("ftp_host_count")
        if host_count:
            meta_lines.append(f"已捕捉 FTP 用户名/密码：{ticket_count} 组；主备主机：{host_count} 个")
        else:
            meta_lines.append(f"已捕捉 FTP 用户名/密码：{ticket_count} 组")
    if status.get("host"):
        meta_lines.append(f"主机：{status.get('host')}")
    if status.get("pid"):
        meta_lines.append(f"下载进程：{status.get('pid')}")
    if status.get("total_bytes"):
        meta_lines.append(f"总量：{_format_bytes(status.get('total_bytes'))}")
    if status.get("downloaded_bytes"):
        meta_lines.append(f"已下载：{_format_bytes(status.get('downloaded_bytes'))}")
    if status.get("file_count"):
        idx = status.get("current_index") or 0
        meta_lines.append(f"文件：{idx}/{status.get('file_count')}")
    current_remote = str(status.get("current_remote") or "")
    if current_remote:
        meta_lines.append("当前文件：" + current_remote[-80:])
    if status.get("download_dir"):
        meta_lines.append("保存到：" + str(status.get("download_dir")))
    if status.get("space_warning"):
        meta_lines.append("磁盘提示：" + str(status.get("space_warning")))
    if status.get("error"):
        meta_lines.append("错误：" + str(status.get("error")))
    return html.Div(className=f"download-progress-card {tone}", children=[
        html.Div(className="panel-title", children="数据下载进度"),
        html.Div(className="download-progress-head", children=[
            html.Span(title, className="download-progress-status"),
            html.Span(f"{percent:.1f}%", className="download-progress-percent"),
        ]),
        html.Div(className="domestic-ftp-progress-track", children=html.Div(className="domestic-ftp-progress-bar-inner", style={"width": f"{percent:.1f}%"})),
        html.Div(className="download-progress-message", children=status.get("message") or ("FTP捕捉器运行中。" if watcher else "等待下载状态。")),
        html.Div(className="download-progress-meta", children=[html.Div(x) for x in meta_lines[:9]]),
    ])

def _render_mapping_source_choice_modal(payload: dict):
    request_text = payload.get("request_text") or ""
    is_download = str(payload.get("task_type") or "").lower() == "download_only"
    instruction_label = "当前下载指令" if is_download else "当前制图指令"
    hint_text = ("选择后将立即进入单数据下载流程；若选择国内数据平台，当前固定使用国家青藏高原科学数据中心（TPDC）。若下载得到 NetCDF/HDF，多时相数据会自动仅导出用户指定年份并转换为 GeoTIFF 加载到地图。" if is_download else "选择后将立即进入完整 制图流程。若选择国内数据平台，当前固定使用国家青藏高原科学数据中心（TPDC）。如 TPDC 要求验证码、人机验证、协议确认或 FTP 参数，系统会弹出接管窗口。")
    return html.Div(className="domestic-handoff-host", children=[
        html.Div(className="domestic-modal-backdrop"),
        html.Div(className="domestic-modal-card domestic-modal-card-compact mapping-source-modal", children=[
            html.Div(className="domestic-modal-head", children=[
                html.Div(payload.get("title") or "选择制图数据源", className="domestic-modal-title"),
                html.Button("×", id="mapping-source-cancel-btn", n_clicks=0, className="domestic-modal-close"),
            ]),
            html.Div(payload.get("message") or "请选择本次制图使用的数据源。", className="domestic-modal-message"),
            html.Div(className="domestic-modal-section", children=[
                html.Div(instruction_label, className="domestic-modal-subtitle"),
                html.Code(request_text, className="domestic-capture-path domestic-capture-path-inline"),
            ]),
            html.Div(className="domestic-modal-section mapping-source-choice-grid", children=([
                html.Button([
                    html.Strong("只使用用户上传数据"),
                    html.Span("不额外下载协变量；适合用户已准备足够 TIF/NC/HDF 协变量数据时做基线制图。")
                ], id="mapping-source-user-btn", n_clicks=0, type="button", className="mapping-source-choice-btn mapping-source-user-btn")
            ] if not is_download else []) + [
                html.Button([
                    html.Strong("国内数据平台：国家青藏高原科学数据中心（TPDC）"),
                    html.Span("已有用户数据优先；缺失项从 TPDC 补充。下载较慢，可能需要验证码、人机验证或 FTP 账号/密码。")
                ], id="mapping-source-domestic-btn", n_clicks=0, type="button", className="mapping-source-choice-btn mapping-source-domestic-btn"),
            ]),
            html.Div(hint_text, className="domestic-modal-hint"),
            html.Div("点击任一选项后会立即关闭弹窗并弹出任务反馈；若无反应，请刷新页面后重试。", className="mapping-source-click-hint"),
        ]),
    ])


def _render_domestic_handoff_modal(payload: dict | None):
    if not payload:
        return html.Div(className="domestic-handoff-host empty")
    if isinstance(payload, dict) and payload.get("kind") == "mapping_source_choice":
        return _render_mapping_source_choice_modal(payload)
    # AI-first TPDC downloads no longer show a middle handoff panel.
    # The assistant directly opens the TPDC browser workflow and reports status in chat/toast.
    if isinstance(payload, dict) and payload.get("suppress_modal"):
        return html.Div(className="domestic-handoff-host empty")
    product = payload.get("auto_selected_product") or {}
    recommended = payload.get("recommended_platform") or payload.get("selected_platform") or {}
    platforms = payload.get("platforms") or []
    capture_dir = payload.get("capture_dir") or "E:\\Agent_DSM\\runs\\manual_downloads\\all"
    path_warning = payload.get("path_warning")
    account = payload.get("account_hint") or {}
    account_cfg = account.get("platform_account_config") or {}
    creds = account_cfg.get("credentials") or []
    active_pid = str(recommended.get("id") or product.get("preferred_platform_id") or "")
    active_cred = next((c for c in creds if str(c.get("platform_id")) == active_pid), creds[0] if creds else {})
    credential_ok = bool(active_cred.get("has_username") and active_cred.get("has_password"))
    mode = product.get("platform_selection_mode") or ("user_specified" if (payload.get("request") or {}).get("requested_platform_id") else "auto_selected")
    mode_text = "固定 TPDC 单平台"
    # V115: the ranked platform queue is backend-only. The modal must not show
    # a chain such as “平台A → 平台B → 平台C”, because the user should only see
    # the single platform selected for takeover.
    platform_names = str(recommended.get("name") or (platforms[0].get("name") if platforms else "智能体后台筛选中"))
    spatial = payload.get("spatial_query_scales") or product.get("spatial_query_scales") or []
    spatial_text = " → ".join([str(x) for x in spatial]) if spatial else "目标区 → 省级/全国数据"
    res_m = product.get("requested_resolution_meters")
    res_text = (f"≤ {float(res_m):g} m；允许更精细数据后重采样，禁止更粗分辨率" if res_m else "未指定；按数据产品默认分辨率审查")
    return html.Div(className="domestic-handoff-host", children=[
        html.Div(className="domestic-modal-backdrop"),
        html.Div(className="domestic-modal-card domestic-modal-card-compact", children=[
            html.Div(className="domestic-modal-head", children=[
                html.Div(payload.get("title") or "公开数据获取", className="domestic-modal-title"),
                html.Button("×", id="domestic-handoff-close", n_clicks=0, className="domestic-modal-close"),
            ]),
            html.Div("本任务使用国家青藏高原科学数据中心（TPDC）作为数据来源。系统会优先使用 Microsoft Edge 打开 TPDC 并自动填充平台账号；验证码、人机验证、协议确认以及平台下载按钮由用户在浏览器中完成。", className="domestic-modal-message"),

            html.Div(className="domestic-modal-section domestic-selected-product domestic-compact-grid", children=[
                html.Div([html.Span("数据", className="domestic-kv-label"), html.Strong(product.get("product_name") or "国内免费数据产品")], className="domestic-kv-line"),
                html.Div([html.Span("平台", className="domestic-kv-label"), html.Strong(platform_names), html.Span(f"（{mode_text}）", className="domestic-modal-hint-inline")], className="domestic-kv-line"),
                html.Div([html.Span("空间策略", className="domestic-kv-label"), html.Code(spatial_text)], className="domestic-kv-line"),
                html.Div([html.Span("分辨率约束", className="domestic-kv-label"), html.Code(res_text)], className="domestic-kv-line"),
                html.Div([html.Span("检索词", className="domestic-kv-label"), html.Code(product.get("keyword_string") or " ".join(product.get("search_keywords") or []))], className="domestic-kv-line"),
                html.Div([html.Span("默认保存到", className="domestic-kv-label"), html.Code(capture_dir, className="domestic-capture-path domestic-capture-path-inline")], className="domestic-kv-line"),
                html.Div([
                    html.Label("FTP数据保存位置（可自定义）", className="domestic-ftp-label"),
                    dcc.Input(id="domestic-ftp-save-dir-input", type="text", value=str(capture_dir), placeholder="如 E:\\Agent_DSM\\runs\\downloads\\solar_2010", className="domestic-ftp-input domestic-ftp-save-dir-input"),
                    html.Div("点击“开始 FTP 下载”时会使用这里的目录；为空则使用默认保存目录。", className="domestic-modal-hint"),
                ], className="domestic-kv-line domestic-save-dir-line"),
                html.Div(path_warning, className="domestic-modal-warning") if path_warning else None,
            ]),

            html.Div(className="domestic-modal-section", children=[
                html.Div("账号与接管", className="domestic-modal-subtitle"),
                html.Div([html.Span("平台账号", className="domestic-kv-label"), html.Code(str(active_cred.get("username_masked") or "未配置")), html.Span("已配置账号+密码" if credential_ok else "未完整配置，请在 .env 填写 TPDC_USERNAME/TPDC_PASSWORD", className="domestic-modal-hint-inline")], className="domestic-kv-line"),
                html.Div("系统会自动填充 .env 中配置的 TPDC 账号密码。FTP 下载参数由用户从 TPDC 弹窗复制到下方输入区，然后点击“开始 FTP 下载”。", className="domestic-modal-hint"),
            ]),

            # Remove the platform preview from the visible UI. It was unreliable
            # with TPDC opening data details/FTP dialogs in new Chrome tabs and misled the user.
            # Hidden placeholders keep legacy callbacks harmless without showing a black/old image.
            html.Div(style={"display": "none"}, children=[
                html.Div(id="domestic-browser-status"),
                html.Img(id="domestic-browser-projection-img", src=""),
                dcc.Interval(id="domestic-browser-projection-poll", interval=10000, n_intervals=0, disabled=True),
            ]),
            html.Div("平台页面将直接在浏览器中打开。请以浏览器中的 TPDC 页面为准；FTP 参数以 TPDC 弹窗内容为准。", className="domestic-modal-hint"),
            html.Div("若出现验证码、人机验证或协议确认，请在弹出的浏览器窗口中完成。若用户只要某一年，覆盖该年份的多年数据集也可下载，后续只保留目标年份。", className="domestic-modal-hint"),

            html.Div(className="domestic-modal-section domestic-manual-ftp-section", children=[
                html.Div("FTP 下载参数", className="domestic-modal-subtitle"),
                html.Div("请在 TPDC 弹出 FTP 账号窗口后，填写用户名和密码。当前版本固定使用 ftp2.tpdc.ac.cn:6201，不启用备用主机。", className="domestic-modal-hint"),
                html.Div(style={"display": "none"}, children=[
                    html.Button("FTP捕捉器", id="domestic-ftp-capture-restart-btn", n_clicks=0),
                ]),
                html.Div([
                    html.Label("整段FTP账号粘贴（可选，自动拆分到下方）", className="domestic-ftp-label"),
                    dcc.Textarea(id="domestic-ftp-bulk-text", placeholder="可直接粘贴：用户名 download_xxx；密码 xxxxx。主机固定 ftp2.tpdc.ac.cn，端口固定 6201。", className="domestic-ftp-input domestic-ftp-bulk-text", style={"width": "100%", "minHeight": "74px"}),
                    html.Div("推荐直接粘贴整段 FTP 账号，系统会自动拆分到下方字段。点击开始下载后，系统会立即写入状态文件并启动后台下载进程。", className="domestic-modal-hint"),
                ], className="domestic-ftp-bulk-wrap"),
                html.Div(className="domestic-ftp-form-grid", children=[
                    html.Div([html.Label("主机1", className="domestic-ftp-label"), dcc.Input(id="domestic-ftp-host-input", type="text", placeholder="如 ftp2.tpdc.ac.cn", className="domestic-ftp-input")]),
                    html.Div([html.Label("备用主机（已停用）", className="domestic-ftp-label"), dcc.Input(id="domestic-ftp-backup-host-input", type="text", placeholder="当前版本不使用备用主机", disabled=True, className="domestic-ftp-input")]),
                    html.Div([html.Label("端口", className="domestic-ftp-label"), dcc.Input(id="domestic-ftp-port-input", type="text", placeholder="如 6201", value="6201", className="domestic-ftp-input")]),
                    html.Div([html.Label("用户名", className="domestic-ftp-label"), dcc.Input(id="domestic-ftp-username-input", type="text", placeholder="如 download_81747713", className="domestic-ftp-input")]),
                    html.Div([html.Label("密码", className="domestic-ftp-label"), dcc.Input(id="domestic-ftp-password-input", type="password", placeholder="填写 FTP 密码", className="domestic-ftp-input")]),
                ]),
                html.Div(className="domestic-modal-actions domestic-ftp-actions", children=[
                    html.Button("开始 FTP 下载", id="domestic-manual-ftp-download-btn", n_clicks=0, className="domestic-modal-primary-btn"),
                ]),
                html.Div(id="domestic-manual-ftp-inline-status", className="domestic-manual-ftp-inline-status", children="当前尚未启动 FTP 下载。填写或粘贴 FTP 信息后点击“开始 FTP 下载”，系统会立即创建下载任务并显示状态。"),
                html.Div(className="domestic-ftp-progress-wrap", children=[
                    html.Div(id="domestic-ftp-progress-text", className="domestic-ftp-progress-text", children="下载状态：未开始。"),
                    html.Div(className="domestic-ftp-progress-track", children=html.Div(id="domestic-ftp-progress-bar-inner", className="domestic-ftp-progress-bar-inner", style={"width": "0%"})),
                    dcc.Interval(id="domestic-ftp-progress-poll", interval=5000, n_intervals=0),
                ]),
                html.Div("100G 级数据会在后台下载；进度文件：ftp_download_status.json，日志：ftp_python_downloader.log。若 Python 下载中断，可用自动生成的 open_winscp_download.bat 接管续传。", className="domestic-modal-hint"),
            ]),

            html.Div(className="domestic-modal-actions", children=[
                html.Button("打开/继续访问 TPDC", id="domestic-launch-capture-btn", n_clicks=0, className="domestic-modal-primary-btn"),
                html.Button("标记为已开始下载", id="domestic-download-started-btn", n_clicks=0, className="domestic-modal-secondary-btn"),
                html.Button("标记为已完成下载", id="domestic-download-completed-btn", n_clicks=0, className="domestic-modal-secondary-btn"),
            ]),
            html.Div(id="domestic-download-action-state", style={"display": "none"}, **{
                "data-manifest-path": str(payload.get("manifest_path") or ""),
            }),
        ]),
    ])


def _render_pro_payment_modal(payload: dict | None):
    if not payload:
        return html.Div(style={"display": "none"})
    qr = payload.get("qr") or {}
    price = payload.get("price") or PRO_PRICE_RMB_MONTH
    wechat_src = qr.get("wechat_src") or ""
    alipay_src = qr.get("alipay_src") or ""
    def _qr_card(title, src, path):
        if src:
            body = html.Img(src=src, className="payment-qr-img")
        else:
            body = html.Div("未识别到收款码图片，请检查 E:\\Agent_DSM\\收款码", className="payment-qr-missing")
        return html.Div(className="payment-qr-card", children=[
            html.Div(title, className="payment-qr-title"),
            body,
            html.Div(),
        ])
    return html.Div(className="payment-modal-backdrop", children=[
        html.Div(className="payment-modal", children=[
            html.Div(className="payment-modal-title", children=f"Pro 版开通/续费（{price}￥/月）"),
            html.Div(className="payment-modal-subtitle", children="请选择微信或支付宝完成支付。完成后点击“已完成支付”开通当前账号。"),
            html.Div(className="payment-qr-grid", children=[
                _qr_card("微信支付", wechat_src, qr.get("wechat_path")),
                _qr_card("支付宝支付", alipay_src, qr.get("alipay_path")),
            ]),
            html.Div(className="payment-modal-actions", children=[
                html.Button("未完成支付", id="pro-payment-cancel-btn", n_clicks=0, className="payment-cancel-btn"),
                html.Button("已完成支付", id="pro-payment-finish-btn", n_clicks=0, className="payment-finish-btn"),
            ]),
        ])
    ])


def create_app():
    app = Dash(
        __name__,
        title=APP_TITLE,
        suppress_callback_exceptions=True,
        assets_folder=str(BASE_DIR / 'assets'),
        assets_url_path='/assets',
        external_stylesheets=["https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"],
        external_scripts=["https://unpkg.com/leaflet@1.9.4/dist/leaflet.js", "https://unpkg.com/html2canvas@1.4.1/dist/html2canvas.min.js"],
    )

    app.layout = html.Div(
        style={"fontFamily": '"Microsoft YaHei", Arial, sans-serif', "height": "100vh", "padding": "12px", "background": "#eef2f7", "boxSizing": "border-box", "overflow": "hidden"},
        children=[
            dcc.Store(id="session-store", storage_type="session"),
            dcc.Store(id="auth-store", storage_type="session"),
            # Auth sub-page is deliberately memory-only. If it is session/local storage,
            # a stale value such as "reset" can be restored after refresh and make
            # the login screen flicker into the password-reset screen.
            dcc.Store(id="auth-page", storage_type="memory", data="login"),
            dcc.Store(id="auth-flash", storage_type="memory"),
            dcc.Store(id="auth-agreed", storage_type="memory", data=False),
            dcc.Store(id="auth-memory", storage_type="local"),
            dcc.Store(id="agent-version-store", storage_type="session", data="standard"),
            dcc.Store(id="ui-toast-store", storage_type="memory"),
            dcc.Store(id="domestic-handoff-store", storage_type="memory"),
            dcc.Store(id="pro-payment-store", storage_type="memory"),
            # Two-stage chat dispatch: first callback appends the user bubble and clears
            # the input immediately; second callback runs the slow AI/tool workflow.
            dcc.Store(id="pending-agent-store", storage_type="memory"),
            dcc.Interval(id="task-poll", interval=2200, n_intervals=0),
            dcc.Interval(id="map-stage-sync-poll", interval=1000, n_intervals=0),
            html.Div(id="session-sync-status", style={"display": "none"}),
            html.Div(id="global-toast-host"),
            html.Div(id="domestic-handoff-host"),
            html.Div(id="pro-payment-host"),
            html.Div(id="app-body"),
        ]
    )




    @app.server.route("/__domestic_capture_screenshot")
    def domestic_capture_screenshot_route():
        manifest_path = request.args.get("manifest", "")
        manifest = _domestic_manifest_load_for_ui(manifest_path)
        shot = manifest.get("browser_live_screenshot") or ""
        if not shot:
            return jsonify({"ok": False, "error": "no screenshot path"}), 404
        try:
            p = Path(str(shot))
            if not p.exists() or not p.is_file():
                return jsonify({"ok": False, "error": "screenshot not ready"}), 404
            return send_file(str(p), mimetype="image/png", max_age=0)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.server.route("/__manual_ftp_download", methods=["POST"])
    def manual_ftp_download_route():
        """Direct backend path for the FTP button.

        Dash callbacks can be hard to diagnose inside dynamically rendered modals.
        This explicit Flask endpoint and front-end script ensure immediate feedback after clicking “开始 FTP 下载”.
        """
        try:
            payload = request.get_json(silent=True) or {}
            result = start_manual_ftp_download(
                payload.get("manifest_path"),
                payload.get("host"),
                payload.get("port"),
                payload.get("username"),
                payload.get("password"),
                payload.get("backup_host"),
                save_dir_override=payload.get("save_dir"),
            )
            return jsonify(result), (200 if result.get("ok") else 400)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.callback(
        Output("global-toast-host", "children"),
        Input("ui-toast-store", "data"),
    )
    def render_global_toast(toast_data):
        return _render_global_toast(toast_data)

    @app.callback(
        Output("domestic-handoff-host", "children"),
        Input("domestic-handoff-store", "data"),
    )
    def render_domestic_handoff(handoff_data):
        return _render_domestic_handoff_modal(handoff_data)

    @app.callback(
        Output("domestic-browser-projection-img", "src"),
        Output("domestic-browser-status", "children"),
        Input("domestic-browser-projection-poll", "n_intervals", allow_optional=True),
        State("domestic-download-action-state", "data-manifest-path", allow_optional=True),
        State("domestic-browser-projection-img", "src", allow_optional=True),
        prevent_initial_call=False,
    )
    def update_domestic_browser_projection(n, manifest_path, current_src):
        if not manifest_path:
            return no_update, "等待下载任务。"
        state = _read_domestic_live_state(manifest_path)
        msg = state.get("message") or state.get("status") or "浏览器浏览器进程运行中。"
        platform_name = state.get("platform_name") or state.get("platform")
        if platform_name:
            msg = f"{msg} 当前平台：{platform_name}。"
        captured = state.get("captured_count")
        if captured is not None:
            msg = f"{msg} 已捕获下载数：{captured}。"
        ftp_count = state.get("ftp_accounts_count")
        ftp_status = state.get("ftp_watcher_status")
        if ftp_status:
            msg = f"{msg} FTP捕捉：{ftp_status}。"
        if ftp_count:
            host_count = state.get("ftp_host_count")
            if host_count:
                msg = f"{msg} 已识别FTP用户名/密码：{ftp_count}组；主备主机：{host_count}个，正在准备/启动下载。"
            else:
                msg = f"{msg} 已识别FTP用户名/密码：{ftp_count}组，正在准备/启动下载。"
        ftp_proc = (state.get("ftp_download_process") or {}).get("pid") if isinstance(state.get("ftp_download_process"), dict) else None
        if ftp_proc:
            msg = f"{msg} Python FTP下载进程PID：{ftp_proc}。"
        projection = state.get("browser_projection") if isinstance(state.get("browser_projection"), dict) else {}
        projection_mode = ""
        if projection:
            projection_mode = str(projection.get("mode") or "")
            page_count = projection.get("page_count")
            if projection_mode:
                msg = f"{msg} 页面状态模式：{projection_mode}" + (f"，已扫描TPDC标签{page_count}个。" if page_count is not None else "。")
        if projection_mode in {"screenshot_failed", "none"}:
            return _projection_placeholder_src("请查看真实 Chrome 窗口"), msg + " 当前页面状态截图不可用，已显示占位提示。"
        manifest = _domestic_manifest_load_for_ui(manifest_path)
        shot = manifest.get("browser_live_screenshot") or state.get("live_screenshot") or ""
        if not shot:
            return no_update, msg
        try:
            p = Path(str(shot))
            if not p.exists() or not p.is_file():
                return no_update, msg
            version = str(p.stat().st_mtime_ns)
        except Exception:
            return no_update, msg
        src = f"/__domestic_capture_screenshot?manifest={manifest_path}&v={version}"
        if current_src and f"v={version}" in str(current_src):
            return no_update, msg
        # 页面状态显示宁可低频稳定，也不要频繁刷新。
        # 10 秒轮询下仅每 3 次允许替换一次图片源，约 30 秒刷新一帧。
        if current_src and (int(n or 0) % 3 != 0):
            return no_update, msg
        return src, msg

    @app.callback(
        Output("domestic-handoff-store", "data", allow_duplicate=True),
        Input("domestic-handoff-close", "n_clicks"),
        prevent_initial_call=True,
    )
    def close_domestic_handoff(n):
        if not n:
            raise PreventUpdate
        return None

    @app.callback(
        Output("ui-toast-store", "data", allow_duplicate=True),
        Input("domestic-launch-capture-btn", "n_clicks"),
        State("domestic-download-action-state", "data-manifest-path"),
        prevent_initial_call=True,
    )
    def launch_domestic_capture(n, manifest_path):
        if not n:
            raise PreventUpdate
        try:
            result = launch_capture_browser(manifest_path)
            if result.get("ok"):
                msg = (
                    f"已打开 TPDC 浏览器。平台：{result.get('platform')}；"
                    f"保存目录：{result.get('save_dir')}；PID：{result.get('pid')}。"
                    "验证码、人机验证、协议确认或平台下载按钮请在浏览器中完成。"
                )
                return _make_toast("TPDC 浏览器已打开", msg, "success", 10000)
            return _make_toast("TPDC 浏览器启动失败", str(result.get("error") or result), "error", 12000)
        except Exception as exc:
            return _make_toast("TPDC 浏览器启动失败", str(exc), "error", 12000)


    @app.callback(
        Output("ui-toast-store", "data", allow_duplicate=True),
        Input("domestic-ftp-capture-restart-btn", "n_clicks"),
        State("domestic-download-action-state", "data-manifest-path"),
        prevent_initial_call=True,
    )
    def restart_ftp_capture_callback(n, manifest_path):
        if not n:
            raise PreventUpdate
        try:
            result = launch_ftp_capture_now(manifest_path)
            if result.get("ok"):
                msg = (
                    f"FTP 参数处理器已启动。PID：{result.get('pid')}；"
                    f"CDP：{result.get('endpoint') or '未连接'}；"
                    "它不会点击、不跳转、不关闭浏览器。"
                )
                return _make_toast("FTP 参数处理器已启动", msg, "success", 12000)
            return _make_toast("FTP 参数处理器启动失败", str(result.get("error") or result), "error", 15000)
        except Exception as exc:
            return _make_toast("FTP 参数处理器启动失败", str(exc), "error", 15000)


    @app.callback(
        Output("ui-toast-store", "data", allow_duplicate=True),
        Output("domestic-manual-ftp-inline-status", "children"),
        Input("domestic-manual-ftp-download-btn", "n_clicks"),
        State("domestic-download-action-state", "data-manifest-path"),
        State("domestic-ftp-host-input", "value"),
        State("domestic-ftp-port-input", "value"),
        State("domestic-ftp-username-input", "value"),
        State("domestic-ftp-password-input", "value"),
        State("domestic-ftp-backup-host-input", "value"),
        State("domestic-ftp-save-dir-input", "value"),
        prevent_initial_call=True,
    )
    def start_manual_ftp_download_callback(n, manifest_path, host, port, username, password, backup_host, save_dir):
        if not n:
            raise PreventUpdate
        try:
            print(f"[MANUAL_FTP] button clicked n={n} manifest={manifest_path} host={host} backup={backup_host} port={port} user={username}", flush=True)
            result = start_manual_ftp_download(manifest_path, host, port, username, password, backup_host, save_dir_override=save_dir)
            if result.get("ok"):
                msg = (
                    f"已接收 FTP 参数并启动 Python FTP 下载进程。主机：{result.get('primary_host')}；"
                    f"账号组：{result.get('accounts_count')}；PID：{result.get('pid')}；保存目录：{result.get('download_dir')}。"
                    "大文件下载进度见 ftp_download_status.json / ftp_python_downloader.log；失败时可用 open_winscp_download.bat 接管。"
                )
                inline = (
                    f"✅ FTP 下载已启动。PID：{result.get('pid')}；主机：{result.get('primary_host')}；"
                    f"保存目录：{result.get('download_dir')}。请查看 ftp_download_status.json / ftp_python_downloader.log。"
                )
                return _make_toast("FTP 下载已启动", msg, "success", 15000), inline
            err = str(result.get("error") or result)
            print(f"[MANUAL_FTP][ERROR] {err}", flush=True)
            return _make_toast("FTP 下载启动失败", err, "error", 15000), "❌ FTP 下载启动失败：" + err
        except Exception as exc:
            print(f"[MANUAL_FTP][EXCEPTION] {exc}", flush=True)
            return _make_toast("FTP 下载启动失败", str(exc), "error", 15000), "❌ FTP 下载启动异常：" + str(exc)

    @app.callback(
        Output("domestic-ftp-progress-text", "children"),
        Output("domestic-ftp-progress-bar-inner", "style"),
        Input("domestic-ftp-progress-poll", "n_intervals", allow_optional=True),
        State("domestic-download-action-state", "data-manifest-path", allow_optional=True),
        prevent_initial_call=False,
    )
    def update_ftp_progress(n, manifest_path):
        base_style = {"width": "0%"}
        if not manifest_path:
            return "下载状态：等待任务。", base_style
        st = _read_ftp_download_status(manifest_path)
        status = str(st.get("status") or st.get("ftp_download_status") or "not_started")
        if status == "not_started":
            detail = st.get("message") or "本次会话未开始。"
            return "下载状态：" + str(detail), base_style
        total = st.get("remote_total_known_bytes") or st.get("total_known_bytes") or 0
        done = st.get("downloaded_bytes_total") or st.get("local_bytes_total") or 0
        current_local = st.get("current_local_bytes") or 0
        current_size = st.get("current_remote_size") or 0
        percent = st.get("progress_percent")
        try:
            if percent is None and total and done:
                percent = max(0.0, min(100.0, float(done) / float(total) * 100.0))
            percent_f = float(percent or 0.0)
        except Exception:
            percent_f = 0.0
        width = f"{max(0.0, min(100.0, percent_f)):.1f}%"
        style = {"width": width}
        host = st.get("host") or st.get("ftp_host") or ""
        cur = st.get("current_remote") or ""
        msg_parts = [f"下载状态：{status}"]
        if host:
            msg_parts.append(f"主机：{host}")
        if total:
            msg_parts.append(f"总量：{_format_bytes(total)}")
        if done:
            msg_parts.append(f"已下载：{_format_bytes(done)}")
        if percent_f:
            msg_parts.append(f"进度：{percent_f:.1f}%")
        if current_size or current_local:
            msg_parts.append(f"当前文件：{_format_bytes(current_local)} / {_format_bytes(current_size)}")
        if cur:
            msg_parts.append(f"正在下载：{str(cur)[-90:]}")
        if st.get("space_warning"):
            msg_parts.append("磁盘提示：" + str(st.get("space_warning")))
        if status in {"ftp_download_completed", "download_verified"}:
            style = {"width": "100%"}
            msg_parts.append("下载完成。")
        if status in {"ftp_download_failed", "ftp_download_process_start_failed", "ftp_host_failed"} and st.get("error"):
            msg_parts.append("错误：" + str(st.get("error")))
        return "；".join(msg_parts), style


    @app.callback(
        Output("ui-toast-store", "data", allow_duplicate=True),
        Input("domestic-download-started-btn", "n_clicks"),
        State("domestic-download-action-state", "data-manifest-path"),
        prevent_initial_call=True,
    )
    def confirm_domestic_download_started(n, manifest_path):
        if not n:
            raise PreventUpdate
        try:
            manifest = mark_download_started(manifest_path, mode="manual_handoff")
            msg = "已记录下载开始。下载可能持续数小时或数天；完成后请点击“标记为已完成下载”进行文件真实性检查。"
            if manifest.get("manifest_path"):
                msg += " Manifest：" + str(manifest.get("manifest_path"))
            return _make_toast("下载已开始", msg, "success", 8000)
        except Exception as exc:
            return _make_toast("记录下载开始失败", str(exc), "error", 12000)

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Output("ui-toast-store", "data", allow_duplicate=True),
        Input("domestic-download-completed-btn", "n_clicks"),
        State("domestic-download-action-state", "data-manifest-path"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def confirm_domestic_download_completed(n, manifest_path, store_data):
        if not n:
            raise PreventUpdate
        session = _session_from_store(store_data)
        try:
            manifest = scan_downloaded_files(manifest_path)
            if manifest.get("status") == "download_verified":
                files = manifest.get("validated_files") or []
                first = files[0].get("path") if files else ""
                post = process_download_manifest_for_map(manifest_path)
                if post.get("ok") and post.get("display_tif"):
                    tif = str(post.get("display_tif"))
                    session.latest_result_kind = "uploaded_raster"
                    session.uploaded_result_paths = {"uploaded_tif": tif, "download_manifest": str(manifest_path or "")}
                    session.last_result_paths = session.uploaded_result_paths.copy()
                    session.active_layout_key = None
                    session.add_message("assistant", "下载文件已验证，并已转换/加载为地图栅格预览。若原始文件是 NetCDF/HDF，多时相数据只导出用户指定年份的预览 GeoTIFF，原始文件保留。\n显示文件：" + tif)
                    return _commit_session(session), _make_toast("下载文件已加载", f"发现 {len(files)} 个真实数据文件，已加载到地图：{tif}", "success", 10000)
                session.add_message("assistant", "下载文件已验证，但没有生成可显示的 GeoTIFF。请检查文件是否为有效栅格或 NetCDF/HDF 是否具有经纬度维度。")
                return _commit_session(session), _make_toast("下载已验证但未能显示", str(post.get("error") or first), "warning", 12000)
            return _commit_session(session), _make_toast("仍未发现真实数据文件", "已扫描保存目录，但暂未发现可用 GeoTIFF/ZIP/HDF/NetCDF 等真实数据文件。下载未完成时可稍后再点此按钮。", "warning", 12000)
        except Exception as exc:
            return _commit_session(session), _make_toast("下载文件验证失败", str(exc), "error", 12000)

    @app.callback(
        Output("auth-page", "data", allow_duplicate=True),
        Output("auth-flash", "data", allow_duplicate=True),
        Input("login-register-link", "n_clicks", allow_optional=True),
        Input("login-reset-link", "n_clicks", allow_optional=True),
        Input("register-login-link", "n_clicks", allow_optional=True),
        Input("reset-login-link", "n_clicks", allow_optional=True),
        Input("register-duplicate-reset-btn", "n_clicks", allow_optional=True),
        prevent_initial_call=True,
    )
    def auth_nav(login_reg, login_reset, reg_login, reset_login, duplicate_reset):
        trig = ctx.triggered_id
        if trig == "login-register-link":
            return "register", None
        if trig in {"login-reset-link", "register-duplicate-reset-btn"}:
            return "reset", None
        if trig in {"register-login-link", "reset-login-link"}:
            return "login", None
        raise PreventUpdate


    @app.callback(
        Output("auth-page", "data", allow_duplicate=True),
        Output("auth-flash", "data", allow_duplicate=True),
        Input("register-agreement-link", "n_clicks"),
        prevent_initial_call=True,
    )
    def open_agreement(n):
        if not n:
            raise PreventUpdate
        return "agreement", None


    @app.callback(
        Output("auth-page", "data", allow_duplicate=True),
        Output("auth-agreed", "data", allow_duplicate=True),
        Output("auth-flash", "data", allow_duplicate=True),
        Input("agreement-accept-btn", "n_clicks"),
        Input("agreement-reject-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def agreement_decision(accept_n, reject_n):
        trig = ctx.triggered_id
        if trig == "agreement-accept-btn":
            return "register", True, {"type": "success", "text": "已同意用户协议，可以继续注册。"}
        if trig == "agreement-reject-btn":
            return "register", False, {"type": "error", "text": "不同意用户协议则无法使用该智能体。"}
        raise PreventUpdate

    @app.callback(
        Output("register-code-msg", "children"),
        Input("register-code-btn", "n_clicks"),
        State("register-contact", "value"),
        prevent_initial_call=True,
    )
    def send_register_code(n, contact):
        if not n:
            raise PreventUpdate
        result = simulate_send_code(contact, require_registered=False)
        return html.Span(result.get("message") if result.get("ok") else result.get("error"), className=("auth-msg-ok" if result.get("ok") else "auth-msg-error"))

    @app.callback(
        Output("reset-code-msg", "children"),
        Input("reset-code-btn", "n_clicks"),
        State("reset-contact", "value"),
        prevent_initial_call=True,
    )
    def send_reset_code(n, contact):
        if not n:
            raise PreventUpdate
        result = simulate_send_code(contact, require_registered=True)
        return html.Span(result.get("message") if result.get("ok") else result.get("error"), className=("auth-msg-ok" if result.get("ok") else "auth-msg-error"))

    @app.callback(
        Output("auth-store", "data", allow_duplicate=True),
        Output("auth-page", "data", allow_duplicate=True),
        Output("auth-flash", "data", allow_duplicate=True),
        Output("auth-memory", "data", allow_duplicate=True),
        Input("login-submit", "n_clicks"),
        State("login-name", "value"),
        State("login-password", "value"),
        State("login-remember", "value"),
        State("auth-memory", "data"),
        prevent_initial_call=True,
    )
    def login_submit(n, login_name, password, remember_values, memory):
        if not n:
            raise PreventUpdate
        result = authenticate(login_name or "", password or "")
        if not result.get("ok"):
            return no_update, "login", {"type": "error", "text": result.get("error") or "用户名或者密码错误。"}, memory or {}
        remember = "remember" in (remember_values or [])
        new_memory = {"remember": remember, "username": login_name or "", "password": (password or "") if remember else ""}
        return {"user": result.get("user"), "logged_in_at": int(datetime.now().timestamp())}, "app", {"type": "success", "text": "登录成功。"}, new_memory


    @app.callback(
        Output("auth-page", "data", allow_duplicate=True),
        Output("auth-flash", "data", allow_duplicate=True),
        Output("auth-agreed", "data", allow_duplicate=True),
        Input("register-submit", "n_clicks"),
        State("register-username", "value"),
        State("register-contact", "value"),
        State("register-password", "value"),
        State("register-password2", "value"),
        State("register-code", "value"),
        State("register-agreed", "value"),
        State("auth-agreed", "data"),
        prevent_initial_call=True,
    )
    def register_submit(n, username, contact, password, password2, code, agreed_values, agreed_store):
        if not n:
            raise PreventUpdate
        agreed = bool(agreed_store) or ("yes" in (agreed_values or []))
        result = register_user(username or "", contact or "", password or "", password2 or "", code or "", agreed)
        if result.get("ok"):
            return "login", {"type": "success", "text": "注册成功，请使用新账号登录。"}, False
        return "register", {"type": "error", "text": result.get("error") or "注册失败。", "reason": result.get("reason")}, agreed


    @app.callback(
        Output("auth-page", "data", allow_duplicate=True),
        Output("auth-flash", "data", allow_duplicate=True),
        Input("reset-submit", "n_clicks"),
        State("reset-contact", "value"),
        State("reset-code", "value"),
        State("reset-password", "value"),
        State("reset-password2", "value"),
        prevent_initial_call=True,
    )
    def reset_submit(n, contact, code, password, password2):
        if not n:
            raise PreventUpdate
        result = reset_password(contact or "", code or "", password or "", password2 or "")
        if result.get("ok"):
            return "login", {"type": "success", "text": "密码已重置，请使用新密码登录。"}
        return "reset", {"type": "error", "text": result.get("error") or "找回密码失败。"}

    @app.callback(
        Output("auth-store", "data", allow_duplicate=True),
        Output("auth-page", "data", allow_duplicate=True),
        Output("auth-flash", "data", allow_duplicate=True),
        Input("logout-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def logout(n):
        if not n:
            raise PreventUpdate
        return None, "login", {"type": "success", "text": "已退出登录。"}


    @app.callback(
        Output("agent-version-store", "data", allow_duplicate=True),
        Input("agent-version-radio", "value", allow_optional=True),
        prevent_initial_call=True,
    )
    def choose_agent_version(value):
        if not value:
            raise PreventUpdate
        return normalize_agent_version(value)

    @app.callback(
        Output("pro-payment-store", "data", allow_duplicate=True),
        Input("pro-pay-btn", "n_clicks", allow_optional=True),
        State("auth-store", "data"),
        prevent_initial_call=True,
    )
    def open_pro_payment_window(n, auth_data):
        if not n:
            raise PreventUpdate
        auth_user = (auth_data or {}).get("user") or {}
        return {
            "open": True,
            "user_id": auth_user.get("id"),
            "price": PRO_PRICE_RMB_MONTH,
            "created_at": int(datetime.now().timestamp()),
            "qr": get_payment_qr_payload(),
        }

    @app.callback(
        Output("pro-payment-host", "children"),
        Input("pro-payment-store", "data"),
    )
    def render_pro_payment_window(payload):
        return _render_pro_payment_modal(payload)

    @app.callback(
        Output("pro-payment-store", "data", allow_duplicate=True),
        Input("pro-payment-cancel-btn", "n_clicks", allow_optional=True),
        prevent_initial_call=True,
    )
    def close_pro_payment_window(n):
        if not n:
            raise PreventUpdate
        return None

    @app.callback(
        Output("auth-store", "data", allow_duplicate=True),
        Output("auth-flash", "data", allow_duplicate=True),
        Output("agent-version-store", "data", allow_duplicate=True),
        Output("pro-payment-store", "data", allow_duplicate=True),
        Input("pro-payment-finish-btn", "n_clicks", allow_optional=True),
        State("auth-store", "data"),
        State("pro-payment-store", "data"),
        prevent_initial_call=True,
    )
    def confirm_pro_payment_finished(n, auth_data, payment_payload):
        if not n:
            raise PreventUpdate
        auth_user = (auth_data or {}).get("user") or {}
        result = upgrade_user_to_pro(auth_user.get("id"))
        if not result.get("ok"):
            return no_update, {"type": "error", "text": result.get("error") or "Pro 开通失败。"}, no_update, no_update
        new_auth = dict(auth_data or {})
        new_auth["user"] = result.get("user")
        new_auth["pro_upgraded_at"] = int(datetime.now().timestamp())
        new_auth["payment_confirmed_at"] = int(datetime.now().timestamp())
        return new_auth, {"type": "success", "text": result.get("message") or "Pro 版已开通。"}, "pro", None


    @app.callback(
        Output("app-body", "children"),
        Input("session-store", "data"),
        Input("auth-store", "data"),
        Input("auth-page", "data"),
        Input("auth-agreed", "data"),
        Input("auth-flash", "data"),
        Input("auth-memory", "data"),
        Input("agent-version-store", "data"),
    )
    def render_body(store_data, auth_data, auth_page, auth_agreed, auth_flash, auth_memory, agent_version):
        if not (auth_data and auth_data.get("user")):
            safe_auth_page = auth_page if auth_page in {"login", "register", "reset", "agreement"} else "login"
            return _render_auth_page(safe_auth_page or "login", auth_memory or {}, bool(auth_agreed), auth_flash or {})
        session = _session_from_store(store_data)
        stage_payload = build_stage_payload(session)
        ramp_default = (stage_payload.get("result") or {}).get("palette") or normalize_palette_name(getattr(session, "som_palette", None))
        reverse_default = ["reverse"] if (stage_payload.get("result") or {}).get("reverse") else []
        right_panel = _result_info_panel(session, ramp_default=ramp_default, reverse_default=reverse_default)
        auth_user = (auth_data or {}).get("user") or {}

        chat_panel = html.Div([
            title_block(APP_TITLE, APP_SUBTITLE),
            html.Div(className="login-user-bar v76-login-bar", children=[
                html.Div(className="login-account-line", children=[
                    html.Span("当前用户：" + str(auth_user.get("username") or "--"), className="login-user-name"),
                    html.Span("权限：" + str(auth_user.get("plan_label") or "标准版"), className="login-plan-badge"),
                    html.Button("退出登录", id="logout-btn", type="button", className="logout-btn"),
                ]),
                html.Div(className="login-version-line", children=[
                    _render_agent_version_panel(agent_version, auth_user, session),
                ]),
            ]),
            _history_toolbar(),
            html.Div(id="session-meta", **{"data-session-id": session.session_id}, style={"display": "none"}),
            html.Pre(id="layout-stage-payload", style={"display": "none"}, children=json.dumps(stage_payload, ensure_ascii=False)),
            html.Div(id="chat-history", children=_render_chat(session.chat_history), style={
                "flex": "1",
                "overflowY": "auto",
                "paddingRight": "8px",
                "minHeight": "48vh",
                "maxHeight": "62vh",
                "scrollBehavior": "smooth",
            }),
            html.Div([
                dcc.Textarea(
                    id="chat-input",
                    placeholder="输入你的问题，例如：为我制作成都市土壤有机质图",
                    style={
                        "width": "100%",
                        "height": "92px",
                        "borderRadius": "12px",
                        "padding": "12px",
                        "border": "1px solid #d1d5db",
                        "fontSize": "15px",
                        "resize": "vertical",
                    }
                ),
                html.Div("Enter 发送，Shift+Enter 换行", style={"marginTop": "8px", "fontSize": "12px", "color": "#6b7280"}),
                html.Button("发送", id="send-btn", n_clicks=0, style={
                    "marginTop": "10px",
                    "padding": "10px 18px",
                    "borderRadius": "10px",
                    "border": "none",
                    "background": "#2563eb",
                    "color": "white",
                    "fontSize": "15px",
                    "cursor": "pointer",
                })
            ])
        ], className="chat-panel")

        layouts_for_selector = stage_payload.get("layouts") or []
        if len(layouts_for_selector) > 1:
            layout_tabs = html.Div(
                [
                    html.Div("当前显示图层", className="layout-select-label"),
                    dcc.Dropdown(
                        id="layout-select",
                        className="layout-select",
                        options=[
                            {
                                "label": item.get("tab_title") or item.get("title") or f"布局{idx+1}",
                                "value": item.get("key"),
                            }
                            for idx, item in enumerate(layouts_for_selector)
                        ],
                        value=stage_payload.get("active_layout_key"),
                        clearable=False,
                        searchable=False,
                    ),
                    
                ],
                id="layout-tabs-bar",
                className="layout-tabs-bar layout-select-bar",
            )
        else:
            layout_tabs = html.Div(
                [
                    html.Button(
                        item.get("tab_title") or f"布局{idx+1}",
                        id={"type": "layout-tab-btn", "index": idx},
                        className=("layout-tab-btn active" if item.get("key") == stage_payload.get("active_layout_key") else "layout-tab-btn"),
                        type="button",
                        **{"data-layout-key": item.get("key")},
                    )
                    for idx, item in enumerate(layouts_for_selector)
                ],
                id="layout-tabs-bar",
                className="layout-tabs-bar",
            )

        center_panel = html.Div([
            html.Div("布局工作区", className="panel-headline"),
            layout_tabs,
            html.Div(id="layout-stage", className="layout-stage", style={"minHeight": "640px"}, children=[
                html.Div(id="map-frame-item", className="layout-item map-frame-item", style={"left": "20px", "top": "20px", "width": "calc(100% - 40px)", "height": "600px", "display": "block", "position": "absolute"}, children=[
                    html.Div(className="map-frame-handle", title="拖动地图框"),
                    html.Div(id="leaflet-map", className="leaflet-map"),
                    html.Div(className="resize-handle handle-nw", **{"data-dir": "nw"}),
                    html.Div(className="resize-handle handle-ne", **{"data-dir": "ne"}),
                    html.Div(className="resize-handle handle-sw", **{"data-dir": "sw"}),
                    html.Div(className="resize-handle handle-se", **{"data-dir": "se"}),
                ]),
                html.Div(id="overlay-layer", className="overlay-layer"),
                html.Div(id="layout-context-menu", className="layout-context-menu hidden", children=[
                    html.Button("复制", type="button", **{"data-menu-action": "duplicate"}),
                    html.Button("删除", type="button", **{"data-menu-action": "delete"}),
                    html.Button("置于顶层", type="button", **{"data-menu-action": "top"}),
                    html.Button("置于底层", type="button", **{"data-menu-action": "bottom"}),
                    html.Button("水平居中", type="button", **{"data-menu-action": "align-h"}),
                    html.Button("垂直居中", type="button", **{"data-menu-action": "align-v"}),
                    html.Button("锁定/解锁", type="button", **{"data-menu-action": "lock"}),
                ]),
                html.Div(id="snap-guide-v", className="snap-guide snap-guide-v"),
                html.Div(id="snap-guide-h", className="snap-guide snap-guide-h"),
            ]),
        ], className="center-shell")

        return html.Div([
            _render_ui_notification(session),
            html.Div(chat_panel, className="chat-shell"),
            html.Div(center_panel, className="center-shell-wrap"),
            html.Div(right_panel, className="right-shell"),
        ], className="workspace-grid fullpage-grid")

    @app.callback(
        Output("session-sync-status", "children"),
        Input("session-store", "data"),
        prevent_initial_call=True,
    )
    def persist_session_snapshot(store_data):
        session = _session_from_store(store_data)
        SESSION_STORE.save_active(session.to_dict())
        return "ok"

    @app.callback(
        Output("chat-history", "children", allow_duplicate=True),
        Input("session-store", "data"),
        prevent_initial_call=True,
    )
    def refresh_chat_history_from_session(store_data):
        """Keep the conversation panel synchronized with every backend state update.

        Uploads, local-folder imports, mapping progress, data review, GCP/AOA tasks
        and AI replies all write to session.chat_history.  Earlier builds updated
        the session object but the visible chat panel could remain on an older
        render until the whole page re-rendered.  This callback makes the chat
        history a direct subscriber of session-store, so each completed backend
        step is visible in the conversation record.
        """
        session = _session_from_store(store_data)
        return _render_chat(session.chat_history)

    @app.callback(
        Output("layout-stage-payload", "children", allow_duplicate=True),
        Output("result-kind-box", "children", allow_duplicate=True),
        Input("session-store", "data"),
        Input("map-stage-sync-poll", "n_intervals"),
        prevent_initial_call=True,
    )
    def refresh_map_stage_from_session(store_data, _map_poll_n):
        """Synchronize map preview/current-result panel after upload/import/mapping.

        This ensures that a successfully uploaded raster is immediately published
        to the map workspace once the backend has activated it in session state.
        """
        session = _session_from_store(store_data)
        payload = build_stage_payload(session)
        # Keep a lightweight backend trace for map publication state.  This is
        # intentionally separate from the scientific task audit: it records only
        # whether the current session has a publishable map payload.
        try:
            if getattr(session, "latest_uploaded_files", None) and not payload.get("result"):
                from services.pipeline_step_logger import append_step_record
                task_id = str(getattr(session, "latest_map_preview_audit_task_id", "") or f"map_preview_{session.session_id}")
                session.remember_task("latest_map_preview_audit_task_id", task_id)
                append_step_record(task_id, "map_preview", "running", 50, "地图载荷同步", "当前会话已有上传/导入数据，但尚未生成可显示的地图载荷。", event="map_payload_empty", extra={"layout_count": len(payload.get("layouts") or []), "preview_errors": payload.get("preview_errors") or []})
        except Exception:
            pass
        return json.dumps(payload, ensure_ascii=False), _result_text_from_stage_payload(payload)

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Input("new-session-btn", "n_clicks"),
        Input("load-archive-btn", "n_clicks"),
        State("archive-session-select", "value"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def manage_session_records(new_n, load_n, archive_id, store_data):
        triggered = ctx.triggered_id
        current = _session_from_store(store_data)
        if triggered == "new-session-btn":
            if current.chat_history or current.latest_result_kind:
                SESSION_STORE.archive_active(current.session_id, reason="manual_new_chat")
            return _commit_session(SessionState())
        if triggered == "load-archive-btn":
            if current.chat_history or current.latest_result_kind:
                SESSION_STORE.archive_active(current.session_id, reason="manual_switch_session")
            restored = SESSION_STORE.restore_archive(archive_id)
            if restored:
                session = SessionState.from_dict(restored)
                return _commit_session(session)
        raise PreventUpdate

    @app.callback(
        Output("upload-status", "children"),
        Output("upload-analysis-status", "children"),
        Output("session-store", "data", allow_duplicate=True),
        Output("ui-toast-store", "data", allow_duplicate=True),
        Input("upload-data", "contents"),
        State("upload-data", "filename"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def on_upload(contents, filenames, store_data):
        if not contents or not filenames:
            raise PreventUpdate
        session = _session_from_store(store_data)
        upload_task_id = f"upload_{session.session_id}_{int(time.time() * 1000)}"
        session.remember_task("latest_upload_audit_task_id", upload_task_id)
        try:
            file_names_for_log = filenames if isinstance(filenames, list) else [filenames]
            append_step_record(upload_task_id, "upload", "running", 1, "接收上传", "已收到上传请求，开始保存并识别文件。", event="upload_start", extra={"filenames": file_names_for_log})
        except Exception:
            pass
        try:
            saved_all = save_multiple_uploads(contents, filenames, session.session_id)
        except Exception as exc:
            # Global upload failure, usually Dash base64 or filesystem issue.
            msg = f"上传失败：{exc}"
            try:
                append_step_record(upload_task_id, "upload", "error", 100, "上传失败", msg, event="upload_failed")
            except Exception:
                pass
            session.latest_upload_feedback = msg
            session.add_message("assistant", msg)
            return _render_upload_feedback_box(msg), _render_upload_analysis_box(session), _commit_session(session), _make_toast("上传失败", msg, "error", 12000)

        saved_ok = [x for x in (saved_all or []) if not x.get("upload_failed")]
        failed_items = [x for x in (saved_all or []) if x.get("upload_failed")]
        current_success_paths = {str(x.get("path") or "") for x in saved_ok if x.get("path")}
        try:
            total_seen = max(1, len(saved_ok) + len(failed_items))
            for idx, item in enumerate(saved_ok, start=1):
                append_step_record(
                    upload_task_id, "upload", "running", int(5 + 35 * idx / total_seen),
                    "文件保存完成", f"已成功保存上传文件：{item.get('name') or item.get('original_name')}",
                    event="file_saved", extra={"path": item.get("path"), "size": item.get("size"), "role_guess": item.get("role_guess")},
                )
            for item in failed_items:
                append_step_record(
                    upload_task_id, "upload", "running", 40,
                    "文件保存失败", f"上传失败：{item.get('name') or item.get('original_name')}；原因：{item.get('upload_error')}",
                    event="file_failed", extra={"error": item.get("upload_error")},
                )
        except Exception:
            pass

        # Uploads are session assets, not a replacing batch.
        existing_files = list(getattr(session, "latest_uploaded_files", []) or [])
        merged_files = []
        seen_keys = set()
        for item in existing_files + list(saved_ok or []):
            try:
                key = (str(item.get("path") or ""), str(item.get("name") or ""), str(item.get("size") or ""))
            except Exception:
                key = (str(item),)
            if key in seen_keys or not key[0]:
                continue
            seen_keys.add(key)
            merged_files.append(item)
        session.latest_uploaded_files = merged_files
        if saved_ok:
            session.active_data_source = "user"

        # Role inference and preprocessing run on the full session inventory, so
        # sampling points uploaded earlier remain available when covariates arrive later.
        try:
            append_step_record(upload_task_id, "upload", "running", 45, "角色识别", "开始识别样点、协变量、NC、瓦片、矢量和结果数据。", event="role_inference_start")
        except Exception:
            pass
        role_report = build_role_report(merged_files)
        try:
            summary_for_log = role_report.get("summary") or {}
            append_step_record(upload_task_id, "upload", "running", 58, "角色识别完成", "上传数据角色识别完成。", event="role_inference_done", extra=summary_for_log)
        except Exception:
            pass
        plan = build_covariate_plan(role_report)
        try:
            append_step_record(upload_task_id, "upload", "running", 65, "预处理检查", "开始执行坐标、分辨率、NoData、NC/瓦片转换需求和入模可用性检查。", event="preprocess_start")
        except Exception:
            pass
        preprocess_report = run_upload_preprocessing(session.session_id, merged_files, role_report=role_report, covariate_plan=plan)
        try:
            append_step_record(upload_task_id, "upload", "running", 78, "预处理检查完成", "上传数据预处理检查完成。", event="preprocess_done", extra=(preprocess_report.get("gate") or {} if isinstance(preprocess_report, dict) else {}))
        except Exception:
            pass
        session.data_role_report = role_report
        session.covariate_readiness_plan = plan
        session.preprocess_report = preprocess_report
        session.preprocess_paths = preprocess_report.get("paths") or {}

        if int((role_report.get("summary") or {}).get("sample_count") or 0) > 0:
            session.map_style_prefs = {**(getattr(session, "map_style_prefs", {}) or {}), "show_samples": True}

        preview_ready = _activate_first_uploaded_raster_preview(session, prefer_latest=False)
        try:
            append_step_record(upload_task_id, "upload", "running", 86, "地图预览", "已尝试将可预览栅格发布到布局工作区。" if preview_ready else "当前批次暂未发现可立即预览的栅格。", event="preview_update", extra={"preview_ready": bool(preview_ready), "active_layout_key": getattr(session, "active_layout_key", None)})
        except Exception:
            pass

        sample_count_now = int((role_report.get("summary") or {}).get("sample_count") or 0)
        cov_count_now = int((role_report.get("summary") or {}).get("model_covariate_count") or 0)
        if not _run_upload_review_if_ready(session, trigger_text="dash_upload", defer_until_map_loaded=bool(preview_ready)):
            if sample_count_now <= 0 and cov_count_now > 0:
                session.add_message("assistant", "已接收到环境协变量数据。等你再上传包含经度、纬度和有机质字段的采样点数据后，我会开始完整的数据合理性审查。")
            elif sample_count_now > 0 and cov_count_now <= 0:
                session.add_message("assistant", "已接收到采样点数据。等你再上传至少一个环境协变量栅格，或明确要求使用默认环境协变量后，我会开始完整的数据合理性审查。")

        # One chat message per file, as requested: success and failure are both
        # visible in the conversation history, not only in a toast.
        by_path = _role_item_map(role_report)
        pre_by_path = _preprocess_record_map(preprocess_report)
        for item in failed_items:
            session.add_message("assistant", _format_upload_failure_message(item))
        for item in saved_ok:
            pth = str(item.get("path") or "")
            session.add_message("assistant", _format_upload_success_message(item, by_path.get(pth), pre_by_path.get(pth)))
        for spatial_msg in _format_spatial_alignment_chat_messages(preprocess_report, current_success_paths):
            session.add_message("assistant", spatial_msg)
        if preview_ready:
            _defer_messages_until_map_loaded(session, ["已将上传的栅格数据加载到布局工作区预览；采样点数据会叠加显示在栅格图层上方。"], reason="layer")
        else:
            raster_n = _session_uploaded_raster_count(session)
            if raster_n > 0:
                session.add_message("assistant", f"已识别到 {raster_n} 个栅格文件，但地图预览尚未生成。系统会继续刷新地图载荷；若仍未显示，请查看后台地图预览审计记录。")

        sample_count = int((role_report.get("summary") or {}).get("sample_count") or 0)
        cov_count = int((role_report.get("summary") or {}).get("model_covariate_count") or 0)
        pending_count = int((role_report.get("summary") or {}).get("pending_conversion_count") or 0)
        gate = (preprocess_report.get("gate") or {}) if isinstance(preprocess_report, dict) else {}
        total_count = int((role_report.get("summary") or {}).get("file_count") or len(session.latest_uploaded_files or []))

        msg = (
            f"上传处理完成：本次成功 {len(saved_ok)} 个，失败 {len(failed_items)} 个；"
            f"当前会话共识别文件 {total_count} 个，采样点数据 {sample_count} 个，"
            f"可直接入模环境协变量 {cov_count} 个，待转换/插值数据 {pending_count} 个。"
        )
        if gate.get("message"):
            msg += "\n预处理结论：" + str(gate.get("message"))
        if len(saved_ok) + len(failed_items) > 1:
            if preview_ready:
                _defer_messages_until_map_loaded(session, [msg], reason="layer")
            else:
                session.add_message("assistant", msg)
        session.latest_upload_feedback = msg

        if failed_items and not saved_ok:
            toast = _make_toast("上传失败", f"本次 {len(failed_items)} 个文件均未成功上传。", "error", 12000)
        elif failed_items:
            toast = _make_toast("部分上传成功", f"成功 {len(saved_ok)} 个，失败 {len(failed_items)} 个；失败原因已逐条写入对话历史。", "warning", 10000)
        elif sample_count <= 0:
            toast = _make_toast("上传成功，等待样点", "已保存文件并完成角色识别；尚未识别到包含经纬度和有机质字段的采样点数据。", "warning", 9000)
        else:
            toast_msg = f"已识别采样点数据 {sample_count} 个、可入模环境协变量 {cov_count} 个。"
            if gate.get("message"):
                toast_msg += " " + str(gate.get("message"))
            toast = _make_toast("上传成功", toast_msg, "success", 6000)
        try:
            paths = audit_paths(upload_task_id)
            session.remember_task("latest_upload_audit_paths", paths)
            append_step_record(upload_task_id, "upload", "done", 100, "上传处理完成", msg, event="upload_done", extra={"audit_paths": paths})
            session.add_message("assistant", "后台上传审计已记录：" + str(paths.get("step_audit_csv") or paths.get("step_audit_jsonl")))
        except Exception:
            pass
        spatial_toast = _build_spatial_alignment_toast(preprocess_report, current_success_paths)
        if spatial_toast:
            toast = [spatial_toast, toast]
        return _render_upload_feedback_box(msg), _render_upload_analysis_box(session), _commit_session(session), toast

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Output("layout-stage-payload", "children", allow_duplicate=True),
        Output("result-kind-box", "children", allow_duplicate=True),
        Input("layout-select", "value", allow_optional=True),
        State("session-store", "data"),
        State("layout-stage-payload", "children"),
        prevent_initial_call=True,
    )
    def persist_layout_select(active_key, store_data, stage_json):
        """Persist and immediately publish the current preview layer.

        dcc.Dropdown is a React component, so the map JS cannot reliably catch a
        native ``change`` event. This callback updates both session-store and the
        hidden stage payload that layout_editor.js polls, making the map, hover
        readout and right-side current result switch together.
        """
        if not active_key:
            raise PreventUpdate
        session = _session_from_store(store_data)
        session.active_layout_key = str(active_key)
        if str(active_key).startswith("uploaded_layout_"):
            session.latest_result_kind = "uploaded_raster"
        elif str(active_key) == "uploaded_stack_layout":
            session.latest_result_kind = "uploaded_raster_stack"
        elif str(active_key) == "som_layout":
            session.latest_result_kind = "som_map"
        elif str(active_key) == "gcp_layout":
            session.latest_result_kind = "gcp_map"

        stage_payload = build_stage_payload(session)
        # Defensive: if build_stage_payload fell back for any reason, still patch
        # the currently selected layout directly from the existing hidden payload.
        if stage_payload.get("active_layout_key") != str(active_key):
            try:
                old_payload = json.loads(stage_json or "{}")
                layouts = old_payload.get("layouts") or []
                chosen = next((x for x in layouts if x.get("key") == str(active_key)), None)
                if chosen:
                    stage_payload = old_payload
                    stage_payload["active_layout_key"] = str(active_key)
                    stage_payload["title"] = chosen.get("title") or stage_payload.get("title")
                    stage_payload["result"] = chosen.get("result")
                    stage_payload["result_key"] = chosen.get("result_key") or "empty"
            except Exception:
                pass
        result = stage_payload.get("result") or {}
        kind = result.get("kind") or ""
        if kind == "uploaded_raster":
            result_text = "上传栅格：" + str(result.get("source_name") or "未命名图层")
        elif kind == "uploaded_raster_stack":
            result_text = "上传协变量叠加预览"
        elif kind == "som_map":
            result_text = str(stage_payload.get("title") or result.get("title") or "有机质制图结果")
        elif kind == "gcp_map":
            result_text = str(stage_payload.get("title") or result.get("title") or "不确定性分析结果")
        else:
            result_text = "暂无"
        return _commit_session(session), json.dumps(stage_payload, ensure_ascii=False), result_text


    @app.callback(
        Output("uploaded-file-role-detail", "children", allow_duplicate=True),
        Output("session-store", "data", allow_duplicate=True),
        Input("uploaded-file-role-select", "value", allow_optional=True),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def update_uploaded_file_role_detail(selected_idx, store_data):
        session = _session_from_store(store_data)
        items = ((getattr(session, "data_role_report", {}) or {}).get("items") or [])
        try:
            idx = int(selected_idx)
        except Exception:
            idx = 0
        if idx < 0 or idx >= len(items):
            raise PreventUpdate
        selected = items[idx]
        session.uploaded_role_selected_idx = idx
        # The upload recognition dropdown is also the user's layer selector.
        # When a GeoTIFF covariate is selected, switch the center map to the
        # corresponding uploaded_layout_N so the map and the selected file stay
        # synchronized. CSV sample selections keep the current raster unchanged.
        selected_path = str(selected.get("path") or "")
        if selected_path.lower().endswith((".tif", ".tiff")):
            order = 0
            for f in list(getattr(session, "latest_uploaded_files", []) or []):
                pth = str(f.get("path") or "")
                if not pth.lower().endswith((".tif", ".tiff")):
                    continue
                order += 1
                if pth == selected_path:
                    session.active_layout_key = f"uploaded_layout_{order}"
                    session.latest_result_kind = "uploaded_raster"
                    break
        return _render_upload_role_detail(selected), _commit_session(session)

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Output("chat-input", "value"),
        Output("pending-agent-store", "data"),
        Input("send-btn", "n_clicks"),
        State("chat-input", "value"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def queue_user_message(n_clicks, user_text, store_data):
        """Fast UI callback: clear the input and render the user bubble immediately."""
        if not n_clicks or not user_text or not str(user_text).strip():
            raise PreventUpdate
        session = _session_from_store(store_data)
        _release_finished_model_task_state(session)
        clean_text = str(user_text).strip()
        session.add_message("user", clean_text)
        if _looks_like_stop_task_request(clean_text):
            session.add_message("assistant", "正在停止当前任务……")
        elif _extract_local_paths_from_text(clean_text):
            session.add_message("assistant", "已收到本机数据路径，正在递归识别文件夹及子文件夹中的样点、栅格、NC、瓦片和矢量数据……")
        else:
            session.add_message("assistant", THINKING_MESSAGE + "（智能体对话主控已接收请求，正在进行上下文推理。）")
        pending = {
            "text": clean_text,
            "session_id": session.session_id,
            "nonce": int(time.time() * 1000),
        }
        return _commit_session(session), "", pending

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Output("ui-toast-store", "data", allow_duplicate=True),
        Output("domestic-handoff-store", "data", allow_duplicate=True),
        Input("pending-agent-store", "data"),
        State("session-store", "data"),
        State("agent-version-store", "data"),
        State("auth-store", "data"),
        prevent_initial_call=True,
    )
    def process_pending_agent(pending_data, store_data, agent_version, auth_data):
        if not pending_data or not isinstance(pending_data, dict):
            raise PreventUpdate
        user_text = str(pending_data.get("text") or "").strip()
        if not user_text:
            raise PreventUpdate
        session = _session_from_store(store_data)
        _release_finished_model_task_state(session)
        auth_user = (auth_data or {}).get("user") or {}
        selected_version = normalize_agent_version(agent_version)

        # Replace the temporary “thinking” bubble with the real answer/tool status
        # when this slow callback finishes.
        _drop_pending_thinking_message(session)

        if _looks_like_stop_task_request(user_text):
            msg = _stop_current_user_task(session, user_text)
            # Composite commands are common, e.g. “停止任务，进行不确定性分析”.
            # Stop must be honored first, but a GCP/AOA request cannot use an old
            # completed map after the current mapping run was stopped or failed.
            if re.search(r"不确定性|GCP|AOA|适用域", str(user_text or ""), re.I):
                ok_cur, miss_cur = _has_current_completed_mapping_result(session)
                if not ok_cur:
                    msg += "\n未启动不确定性分析：只能基于当前已完成的制图结果运行；当前缺少：" + "、".join(miss_cur)
            session.add_message("assistant", msg)
            return _commit_session(session), _make_toast("任务已停止", msg, "success", 6000), no_update

        # Local-path data ingestion tool: when the user gives a local file/folder path,
        # import those files into the current session before normal AI reasoning continues.
        # V220: paths are attachments/context, not the end of the request. After import,
        # the same full user sentence is still sent to the AI planner so the real user
        # intent in the remaining clause can be executed, e.g. "these are my paths; now
        # map Chengdu cropland SOM" must start mapping, while "can these data be used"
        # must stay as data review.
        agent_outcome_after_import = None
        local_import_task_id = f"local_import_{session.session_id}_{int(time.time() * 1000)}"
        if _extract_local_paths_from_text(user_text):
            try:
                append_step_record(local_import_task_id, "local_import", "running", 1, "递归识别本机路径", "已收到本机数据路径，开始递归识别文件夹和子文件夹。", event="local_import_start", extra={"user_text": user_text[:500]})
            except Exception:
                pass
        imported_local, failed_local = _import_local_paths_from_user_text(user_text, session)
        local_import_toast = None
        if imported_local or failed_local:
            existing_files = list(getattr(session, "latest_uploaded_files", []) or [])
            merged_files = []
            seen_keys = set()
            for item in existing_files + imported_local:
                key = (str(item.get("path") or ""), str(item.get("name") or ""), str(item.get("size") or ""))
                if not key[0] or key in seen_keys:
                    continue
                seen_keys.add(key)
                merged_files.append(item)
            session.latest_uploaded_files = merged_files
            # Ensure uploaded sample points remain visible as point overlays on top
            # of whichever raster layer is previewed.
            session.map_style_prefs = {**(getattr(session, "map_style_prefs", {}) or {}), "show_samples": True}
            total_local = max(1, len(imported_local) + len(failed_local))
            for idx, item in enumerate(imported_local, start=1):
                session.add_message("assistant", f"已从本机路径导入数据：{item.get('name')}。该数据已登记到当前任务，若为可预览栅格，将加入地图图层队列。")
                try:
                    append_step_record(local_import_task_id, "local_import", "running", int(5 + 55 * idx / total_local), "导入成功", f"已从本机路径导入数据：{item.get('name')}", event="file_imported", extra={"path": item.get("path"), "role_guess": item.get("role_guess")})
                except Exception:
                    pass
            for item in failed_local:
                session.add_message("assistant", f"本机路径导入失败：{item.get('name')}。原因：{item.get('error')}")
                try:
                    append_step_record(local_import_task_id, "local_import", "running", 40, "导入失败", f"本机路径导入失败：{item.get('name')}。", event="file_import_failed", extra={"error": item.get("error")})
                except Exception:
                    pass
            if imported_local:
                try:
                    append_step_record(local_import_task_id, "local_import", "running", 65, "角色识别", "开始识别本机路径导入数据的样点、协变量、NC、瓦片和矢量数据。", event="role_inference_start")
                except Exception:
                    pass
                role_report = build_role_report(session.latest_uploaded_files)
                plan = build_covariate_plan(role_report)
                try:
                    append_step_record(local_import_task_id, "local_import", "running", 75, "预处理检查", "开始执行本机路径数据预处理检查。", event="preprocess_start")
                except Exception:
                    pass
                preprocess_report = run_upload_preprocessing(session.session_id, session.latest_uploaded_files, role_report=role_report, covariate_plan=plan)
                try:
                    append_step_record(local_import_task_id, "local_import", "running", 88, "预处理检查完成", "本机路径数据预处理检查完成。", event="preprocess_done", extra=(preprocess_report.get("gate") or {} if isinstance(preprocess_report, dict) else {}))
                except Exception:
                    pass
                session.data_role_report = role_report
                session.covariate_readiness_plan = plan
                session.preprocess_report = preprocess_report
                session.preprocess_paths = preprocess_report.get("paths") or {}
                if int((role_report.get("summary") or {}).get("sample_count") or 0) > 0:
                    session.map_style_prefs = {**(getattr(session, "map_style_prefs", {}) or {}), "show_samples": True}
                session.active_data_source = "user"
                preview_ready = _activate_first_uploaded_raster_preview(session, prefer_latest=False)
                session.latest_upload_feedback = f"已从本机路径导入 {len(imported_local)} 个文件，失败 {len(failed_local)} 个。"
                try:
                    append_step_record(local_import_task_id, "local_import", "running", 93, "地图预览", "已尝试将本机路径中的可预览栅格发布到布局工作区。" if preview_ready else "本机路径导入完成，当前未发现可立即预览的栅格。", event="preview_update", extra={"preview_ready": bool(preview_ready), "active_layout_key": getattr(session, "active_layout_key", None)})
                except Exception:
                    pass
                if preview_ready:
                    _defer_messages_until_map_loaded(session, ["已将本机路径中的首个可预览栅格加载到布局工作区；采样点数据会叠加显示在栅格图层上方。"], reason="layer")
                else:
                    raster_n = _session_uploaded_raster_count(session)
                    if raster_n > 0:
                        session.add_message("assistant", f"已从本机路径识别到 {raster_n} 个栅格文件，但地图预览尚未生成。系统会继续刷新地图载荷；若仍未显示，请查看后台地图预览审计记录。")
                inv_msg = _uploaded_inventory_brief(session)
                if preview_ready:
                    _defer_messages_until_map_loaded(session, [inv_msg], reason="layer")
                else:
                    session.add_message("assistant", inv_msg)
                _run_upload_review_if_ready(session, trigger_text=user_text, defer_until_map_loaded=bool(preview_ready))
                # If the previous turn requested mapping but was blocked waiting for rasters,
                # importing a raster path now should resume the same mapping task automatically.
                if getattr(session, "pending_mapping_request_text", None) and _session_uploaded_raster_count(session) > 0:
                    resume_text = _pop_pending_mapping_text(session, fallback=user_text)
                    if preview_ready:
                        _defer_messages_until_map_loaded(session, ["✅ 已检测到新的预测栅格，并已接上前面的制图请求；地图预览加载完成后已启动土壤有机质制图流程。"], reason="layer")
                    else:
                        session.add_message("assistant", "✅ 已检测到新的预测栅格，并已接上前面的制图请求；现在继续启动土壤有机质制图流程。")
                    if _has_active_model_task(session):
                        session.add_message("assistant", "当前已有任务正在运行，请等待完成后再启动新的计算。")
                        return _commit_session(session), _make_toast("任务运行中", "已有制图任务正在运行。", "info", 7000), no_update
                    session.has_requested_map = True
                    session.som_map_shown = False
                    session.gcp_shown = False
                    session.rfk_result_paths = {}
                    session.gcp_result_paths = {}
                    session.last_result_paths = {}
                    session.active_layout_key = None
                    session.latest_result_kind = "rfk_running"
                    task_id = start_rfk_task(data_source=session.active_data_source, session_id=session.session_id, request_text=resume_text, user_id=auth_user.get("id"), uploaded_files=session.latest_uploaded_files)
                    session.latest_rfk_task_id = task_id
                    task = TASKS.get(task_id)
                    if task and task.status == "error" and task.error:
                        session.latest_result_kind = None
                        session.add_message("assistant", "❌ 制图启动失败：" + _clean_task_error(task.error))
                        return _commit_session(session), _make_toast("制图失败", _clean_task_error(task.error), "error", 12000), no_update
                    if preview_ready:
                        _defer_messages_until_map_loaded(session, ["✅ 已开始执行土壤有机质制图流程。系统会动态匹配融合CSV列与当前会话已上传栅格，并自动比较候选协变量组合与候选模型；匹配不到栅格的CSV特征只用于训练诊断，不会伪造整幅图。"], reason="layer")
                    else:
                        session.add_message("assistant", "✅ 已开始执行土壤有机质制图流程。系统会动态匹配融合CSV列与当前会话已上传栅格，并自动比较候选协变量组合与候选模型；匹配不到栅格的CSV特征只用于训练诊断，不会伪造整幅图。")
                    return _commit_session(session), _make_toast("继续制图", "已接上前面的CSV和新上传栅格，开始制图。", "info", 7000), no_update
            try:
                paths = audit_paths(local_import_task_id)
                session.remember_task("latest_local_import_audit_paths", paths)
                append_step_record(local_import_task_id, "local_import", "done", 100, "本机数据导入完成", f"成功 {len(imported_local)} 个，失败 {len(failed_local)} 个。", event="local_import_done", extra={"audit_paths": paths})
                session.add_message("assistant", "后台本机路径导入审计已记录：" + str(paths.get("step_audit_csv") or paths.get("step_audit_jsonl")))
            except Exception:
                pass
            local_import_toast = _make_toast("本机数据导入完成", f"成功 {len(imported_local)} 个，失败 {len(failed_local)} 个。", "success" if imported_local else "warning", 7000)
            # V220：导入路径只是完成“把数据交给智能体”这一步；同一句话后半段的真实意图
            # 仍必须由 AI planner 判断。不能因为前半句包含本机路径，就把整条消息截断为“上传完成”。
            if imported_local:
                try:
                    agent_outcome_after_import = execute_agent_turn(user_text, session, agent_version=selected_version, auth_user=auth_user)
                except Exception as exc:
                    msg = f"AI 处理失败：{exc}"
                    session.add_message("assistant", msg)
                    return _commit_session(session), _make_toast("AI 处理失败", msg, "error", 12000), no_update
                imported_mode = str((agent_outcome_after_import or {}).get("mode") or "reply")
                # Only when AI does NOT choose an executable task, and the user is indeed asking
                # for data usability, do the data review here. If AI chose start_rfk, fall through
                # to the common start_rfk executor below.
                if imported_mode not in {"start_rfk", "start_gcp", "start_download_tpdc", "tpdc_search", "start_download_gee", "standardize_spatial", "mapping_source_choice", "download_source_choice"} and (_analysis_intent_requested(user_text) or _is_data_review_only_request(user_text)):
                    reviewed = _perform_requested_data_usability_review(session, user_text)
                    if not reviewed:
                        session.add_message("assistant", "✅ 已完成本机路径数据导入和基础审查。当前指令未要求启动计算任务。")
                    return _commit_session(session), local_import_toast, no_update

        # AI-first trunk:
        # Every user message goes to execute_agent_turn() before any mapping/download routing.
        # The returned mode is the only signal that may start a tool-like backend workflow.
        if agent_outcome_after_import is not None:
            agent_outcome = agent_outcome_after_import
        else:
            try:
                agent_outcome = execute_agent_turn(user_text, session, agent_version=selected_version, auth_user=auth_user)
            except Exception as exc:
                msg = f"AI 处理失败：{exc}"
                session.add_message("assistant", msg)
                return _commit_session(session), _make_toast("AI 处理失败", msg, "error", 12000), no_update
        session.last_route_decision = agent_outcome.get("planner") or {}
        planner = session.last_route_decision if isinstance(session.last_route_decision, dict) else {}
        mode = agent_outcome.get("mode") or "reply"

        # AI-controlled map styling: if the user asks to change color/opacity/layer visibility,
        # apply it directly and answer in chat instead of exposing manual style widgets.
        style_changed, style_msg = _apply_ai_style_instruction(user_text, session)
        if style_changed and mode in {"reply", "followup", "show_uploaded_raster"}:
            session.add_message("assistant", style_msg)
            return _commit_session(session), _make_toast("地图样式已更新", style_msg, "success", 5000), no_update

        if planner.get("blocked_by_plan") and planner.get("required_plan") == "pro":
            session.pending_pro_request_text = user_text
            session.pending_pro_target = request_target_summary(user_text)
        elif mode == "start_rfk":
            session.pending_pro_request_text = None
            session.pending_pro_target = {}

        ds = str(agent_outcome.get("data_source") or "none")
        if ds in {"default", "user", "pro_user", "pro_web", "pro_domestic", "pro_gee"}:
            session.active_data_source = ds

        pro_console_log(
            "ROUTE",
            "AI-first 当前消息路由结果",
            {
                "session_id": session.session_id,
                "mode": mode,
                "selected_version": selected_version,
                "is_pro_user": is_pro_user(auth_user),
                "active_data_source": session.active_data_source,
                "planner": session.last_route_decision,
            },
        )

        session.last_task_type = {
            "start_rfk": "rfk_mapping",
            "start_gcp": "gcp_uncertainty",
            "show_uploaded_raster": "show_uploaded_raster",
            "reply": "general_chat",
            "followup": "general_chat",
            "mapping_source_choice": "mapping_source_choice",
            "download_source_choice": "download_source_choice",
            "start_download_gee": "download_only",
            "start_download_tpdc": "download_only",
            "tpdc_search": "download_only",
            "standardize_spatial": "spatial_standardization",
        }.get(mode, "general_chat")

        if mode == "mapping_source_choice":
            payload = build_source_choice_payload(user_text, session_id=session.session_id)
            session.pending_pro_request_text = user_text
            session.last_task_type = "mapping_source_choice"
            session.add_message(
                "assistant",
                agent_outcome.get("assistant_text") or "请先选择本次制图数据策略：只使用上传数据，或 TPDC 补充。",
            )
            return _commit_session(session), _make_toast("请选择制图数据策略", "请选择：只用上传数据或 TPDC 补充。", "info", 10000), payload

        if mode == "download_source_choice":
            payload = build_source_choice_payload(user_text, session_id=session.session_id, task_type="download")
            session.pending_pro_request_text = user_text
            session.last_task_type = "download_source_choice"
            session.add_message(
                "assistant",
                agent_outcome.get("assistant_text") or "请先选择本次数据下载来源：TPDC。",
            )
            return _commit_session(session), _make_toast("请选择下载数据源", "请选择 TPDC 后继续下载。", "info", 10000), payload

        if mode == "standardize_spatial":
            try:
                payload = planner.get("payload") or {}
                std = standardize_session_rasters(
                    session_id=session.session_id,
                    uploaded_files=list(getattr(session, "latest_uploaded_files", []) or []),
                    instruction_text=user_text,
                    target_crs_text=str(payload.get("target_crs_text") or ""),
                    target_resolution_m=(float(payload.get("target_resolution_m")) if str(payload.get("target_resolution_m") or "").strip() else None),
                    reference_layer=str(payload.get("reference_layer") or ""),
                )
                if not std.get("partial_ok"):
                    msg = build_standardization_chat_message(std)
                    session.add_message("assistant", msg)
                    return _commit_session(session), _make_toast("空间标准化未完成", msg, "warning", 10000), no_update

                # Replace current raster records with standardized paths, while keeping sample points.
                session.latest_uploaded_files = list(std.get("updated_files") or getattr(session, "latest_uploaded_files", []) or [])
                try:
                    append_step_record(local_import_task_id, "local_import", "running", 65, "角色识别", "开始识别本机路径导入数据的样点、协变量、NC、瓦片和矢量数据。", event="role_inference_start")
                except Exception:
                    pass
                role_report = build_role_report(session.latest_uploaded_files)
                plan = build_covariate_plan(role_report)
                try:
                    append_step_record(local_import_task_id, "local_import", "running", 75, "预处理检查", "开始执行本机路径数据预处理检查。", event="preprocess_start")
                except Exception:
                    pass
                preprocess_report = run_upload_preprocessing(session.session_id, session.latest_uploaded_files, role_report=role_report, covariate_plan=plan)
                try:
                    append_step_record(local_import_task_id, "local_import", "running", 88, "预处理检查完成", "本机路径数据预处理检查完成。", event="preprocess_done", extra=(preprocess_report.get("gate") or {} if isinstance(preprocess_report, dict) else {}))
                except Exception:
                    pass
                session.data_role_report = role_report
                session.covariate_readiness_plan = plan
                session.preprocess_report = preprocess_report
                session.preprocess_paths = preprocess_report.get("paths") or {}
                if int((role_report.get("summary") or {}).get("sample_count") or 0) > 0:
                    session.map_style_prefs = {**(getattr(session, "map_style_prefs", {}) or {}), "show_samples": True}
                session.latest_result_kind = "uploaded_raster"
                session.active_layout_key = None
                session.active_data_source = "user"
                msg = build_standardization_chat_message(std)
                session.add_message("assistant", msg)
                return _commit_session(session), _make_toast("空间标准化完成", "已统一坐标系/分辨率/网格，并重新完成数据预处理检查。", "success", 8000), no_update
            except Exception as exc:
                msg = f"空间标准化任务失败：{exc}"
                session.add_message("assistant", msg)
                return _commit_session(session), _make_toast("空间标准化失败", msg, "error", 12000), no_update

        if mode == "start_download_gee":
            try:
                result = run_gee_download_task(user_text, auth_user=auth_user)
                session.last_task_type = "download_only"
                session.last_route_decision = {**planner, "mode": "download_only_gee", "result": result}
                if result.get("ok") and result.get("display_tif"):
                    tif = str(result.get("display_tif"))
                    session.latest_result_kind = "uploaded_raster"
                    session.uploaded_result_paths = {"uploaded_tif": tif, "download_manifest": str(result.get("manifest_path") or "")}
                    session.last_result_paths = session.uploaded_result_paths.copy()
                    session.active_layout_key = None
                    session.add_message("assistant", "已使用 GEE（Google Earth Engine）下载数据，并已转换/加载为地图栅格预览。\nManifest：" + str(result.get("manifest_path") or ""))
                    return _commit_session(session), _make_toast("GEE 数据已下载", "数据已通过 GEE 下载并加载到地图。", "success", 8000), None
                msg = str(result.get("error") or "GEE 下载失败，未生成可显示栅格。")
                session.add_message("assistant", "GEE 数据下载失败：" + msg)
                return _commit_session(session), _make_toast("GEE 下载失败", msg, "error", 12000), None
            except Exception as exc:
                msg = f"GEE 数据下载任务创建失败：{exc}"
                session.add_message("assistant", msg)
                return _commit_session(session), _make_toast("GEE 下载失败", msg, "error", 12000), None

        if mode in {"start_download_tpdc", "tpdc_search"}:
            try:
                # New download instruction must start with a clean progress panel.
                # Do not display a previous manifest/error while the new TPDC browser flow is only being prepared.
                session.uploaded_result_paths = {k: v for k, v in (getattr(session, "uploaded_result_paths", {}) or {}).items() if k != "download_manifest"}
                download_job = prepare_download_only_task(user_text, auth_user=auth_user)
                handoff = download_job.get("handoff") or {}
                manifest = download_job.get("manifest") or {}
                manifest_path = manifest.get("manifest_path")
                launch_result = launch_capture_browser(manifest_path)
                handoff["auto_launch_result"] = launch_result
                handoff["auto_launched"] = bool(launch_result.get("ok"))
                session.last_task_type = "download_only"
                session.last_route_decision = {**planner, "mode": "download_only_tpdc", "download_manifest": manifest_path, "auto_launch_result": launch_result, "download_notified": [], "download_progress_snapshot": {}, "download_progress_compact": {}}
                if launch_result.get("ok"):
                    session.add_message(
                        "assistant",
                        "已直接启动国家青藏高原科学数据中心（TPDC）检索流程：系统会优先使用 Microsoft Edge 打开 TPDC 登录页，并自动填写 .env 中的 TPDC 账号密码；验证码由你在浏览器页面手动输入。登录成功后，系统只会把核心数据关键词填入搜索框。你选择合适数据集并点击下载后，后台 FTP 捕捉器会自动读取 TPDC 弹窗中的主机、端口、用户名和密码，并尝试下载到任务保存目录；无需再手动填写 FTP 参数。"
                        + ("\n任务记录：" + str(manifest_path) if manifest_path else "")
                    )
                    toast = _make_toast("已打开 TPDC", "TPDC 登录页已打开并开始自动填充账号密码。请在浏览器中输入验证码；登录后系统会自动搜索核心关键词。", "info", 9000)
                else:
                    session.add_message(
                        "assistant",
                        "已选择国内数据平台：国家青藏高原科学数据中心（TPDC），但自动打开浏览器失败。"
                        + ("\n错误：" + str(launch_result.get("error")) if launch_result.get("error") else "")
                    )
                    toast = _make_toast("TPDC 打开失败", str(launch_result.get("error") or launch_result), "error", 12000)
                return _commit_session(session), toast, None
            except Exception as exc:
                msg = f"公开数据获取任务创建失败：{exc}"
                session.add_message("assistant", msg)
                return _commit_session(session), _make_toast("下载任务创建失败", msg, "error", 12000), no_update

        preflight_toast = None

        if mode == "start_rfk":
            gate = _mapping_data_gate(session, choice=session.active_data_source)
            if not gate.get("ok"):
                title = gate.get("title") or "数据未就绪"
                msg = gate.get("message") or "当前数据尚未满足制图条件。"
                _set_pending_mapping(session, user_text, msg)
                session.add_message("assistant", msg + "\n我已保留本次制图请求；你继续上传/导入预测栅格后，我会自动接上当前任务继续制图。")
                return _commit_session(session), _make_toast(title, msg, "warning", 10000), no_update

            # Post-agent deterministic safety check. This is not intent routing.
            try:
                preflight = preflight_mapping_region(user_text, session.latest_uploaded_files)
            except Exception as exc:
                preflight = {
                    "ok": False,
                    "blocked": True,
                    "popup": True,
                    "title": "AOI预检失败",
                    "message": f"制图前样点-AOI一致性预检失败：{exc}",
                }
            session.last_route_decision = {**(session.last_route_decision or {}), "aoi_preflight": preflight, "preflight_phase": "after_ai_agent"}
            if preflight.get("blocked"):
                title = preflight.get("title") or "样点与制图区域不匹配"
                msg = preflight.get("message") or "样点数据与用户指定制图区域不匹配，已停止制图。"
                _push_ui_notification(session, title, msg, "error")
                session.add_message("assistant", f"❌ 制图未启动：{msg}")
                session.last_task_type = "preflight_blocked"
                session.latest_result_kind = None
                session.has_requested_map = False
                pro_console_log("PREFLIGHT", "AI-first 后置AOI预检阻止制图任务启动", {"session_id": session.session_id, "preflight": preflight})
                return _commit_session(session), _make_toast(title, msg, "error"), no_update
            elif preflight.get("popup"):
                title = preflight.get("title") or "目标区域样点风险"
                msg = preflight.get("message") or "目标区域样点不足，结果需谨慎解释。"
                _push_ui_notification(session, title, msg, "warning")
                session.add_message("assistant", f"⚠️ AOI 预检提醒：{msg}\n我会继续执行制图，但结果需要按该提醒谨慎解释。")
                preflight_toast = _make_toast(title, msg, "warning")

            # 用户若只有样点而无协变量，自动改为国内 TPDC 补充；不再走 GEE。
            ready_cov_count_for_route = int(((getattr(session, "data_role_report", {}) or {}).get("summary") or {}).get("model_covariate_count") or 0)
            if (
                selected_version == "pro"
                and is_pro_user(auth_user)
                and session.latest_uploaded_files
                and session.active_data_source == "user"
                and ready_cov_count_for_route <= 0
            ):
                session.active_data_source = "pro_domestic"
                pro_console_log(
                    "ROUTE",
                    "检测到用户只有样点、尚无协变量，已改用国内 TPDC 补充数据。",
                    {
                        "session_id": session.session_id,
                        "request_text": user_text,
                        "uploaded_count": len(session.latest_uploaded_files),
                        "ready_covariate_count": ready_cov_count_for_route,
                        "selected_version": selected_version,
                    },
                )

            style_prefs = parse_map_style_preferences(user_text)
            layout_instr = parse_cartography_instruction(user_text)
            # Mapping commands may specify the map scope by wording such as “耕地土壤有机质图”.
            # General questions containing the same words must remain normal AI Q&A, therefore scope
            # inference is injected only after the AI planner has already selected start_rfk.
            if not layout_instr.get("map_scope"):
                inferred_scope = infer_map_scope(user_text)
                layout_instr = {
                    **layout_instr,
                    "map_scope": inferred_scope,
                    "mask_non_cropland": bool(inferred_scope == "cropland"),
                    "non_cropland_color": "#ffffff",
                }
            if layout_instr.get("palette"):
                style_prefs["palette"] = layout_instr.get("palette")
            if style_prefs.get("palette"):
                session.som_palette = style_prefs.get("palette")
            session.map_style_prefs = {**(getattr(session, "map_style_prefs", {}) or {}), **style_prefs}
            session.map_layout_prefs = merge_layout_instruction(getattr(session, "map_layout_prefs", {}) or {}, layout_instr)
            pro_console_log(
                "STYLE",
                "已解析用户制图样式和制图范围指令",
                {"session_id": session.session_id, "style_prefs": style_prefs, "layout_instr": layout_instr, "som_palette": session.som_palette},
            )
            if _has_active_model_task(session):
                session.add_message("assistant", "当前已有任务正在运行，请等待完成后再启动新的计算。")
                return _commit_session(session), (preflight_toast or no_update), no_update
            session.has_requested_map = True
            _start_new_mapping_context(session, user_text)
            session.active_layout_key = None
            session.latest_result_kind = "rfk_running"
            task_id = start_rfk_task(data_source=session.active_data_source, session_id=session.session_id, request_text=user_text, user_id=auth_user.get("id"), uploaded_files=session.latest_uploaded_files)
            session.latest_rfk_task_id = task_id
            task = TASKS.get(task_id)
            if task and task.status == "error" and task.error:
                session.latest_result_kind = None
                _mark_mapping_not_completed(session, "error")
                session.add_message("assistant", task.error)
                return _commit_session(session), _make_toast("制图失败", _clean_task_error(task.error), "error", 12000), no_update
            start_msg = "✅ 已开始执行 土壤有机质制图。\n系统将使用本地行政区划裁剪目标区域；系统会动态累积本会话上传/导入的数据，按融合CSV识别训练特征，并仅用有栅格支撑的特征输出GeoTIFF。"
            if bool(locals().get("preview_ready")) and (imported_local or failed_local):
                _defer_messages_until_map_loaded(session, [start_msg], reason="layer")
            else:
                session.add_message("assistant", start_msg)
            start_toast = _make_toast("开始制图", "已开始制图，正在准备协变量并搜索当前数据适用的模型。", "info", 5000)
            return _commit_session(session), ([preflight_toast, start_toast] if preflight_toast else start_toast), no_update

        if mode == "start_gcp":
            if _has_active_model_task(session):
                session.add_message("assistant", "当前已有任务正在运行，请等待完成后再启动新的计算。")
                return _commit_session(session), no_update, no_update
            ok, missing = _gcp_input_status(session)
            if not ok:
                session.add_message("assistant", "未启动不确定性分析：只能基于当前已完成的制图结果运行。当前缺少：" + "、".join(missing))
                return _commit_session(session), no_update, no_update
            session.gcp_shown = False
            session.gcp_result_paths = {}
            session.latest_result_kind = "gcp_running"
            session.map_style_prefs = {**(getattr(session, "map_style_prefs", {}) or {}), "show_samples": False}
            task_id = start_gcp_task(rfk_result_paths=session.rfk_result_paths.copy(), session_id=session.session_id)
            session.latest_gcp_task_id = task_id
            task = TASKS.get(task_id)
            if task and task.status == "error" and task.error:
                session.latest_result_kind = None
                _mark_mapping_not_completed(session, "error")
                session.add_message("assistant", task.error)
                return _commit_session(session), _make_toast("不确定性分析失败", _clean_task_error(task.error), "error", 12000), no_update
            session.add_message("assistant", agent_outcome.get("assistant_text") or "收到，我将基于当前已完成的制图结果执行地理共形预测（GCP）+ AOA 不确定性与适用域分析。GCP 在本系统中是地理共形预测，不是广义克里金精度。")
            start_toast = _make_toast("开始不确定性分析", "已开始基于制图结果运行地理共形预测（GCP）+ AOA 不确定性分析。", "info", 5000)
            return _commit_session(session), start_toast, no_update

        if mode == "show_uploaded_raster":
            tif = pick_latest_uploaded_tif(session.latest_uploaded_files)
            if tif:
                session.latest_result_kind = "uploaded_raster"
                session.uploaded_result_paths = {"uploaded_tif": tif["path"]}
                session.last_result_paths = session.uploaded_result_paths.copy()
                session.add_message("assistant", agent_outcome.get("assistant_text") or f"已显示你上传的栅格：{tif['name']}。")
            else:
                session.add_message("assistant", build_user_data_unavailable_text())
            return _commit_session(session), no_update, no_update

        # reply / followup / unknown modes: normal AI chat. Do not run mapping/download gates.
        assistant_text = agent_outcome.get("assistant_text") or build_guidance_text()
        session.add_message("assistant", assistant_text)
        return _commit_session(session), no_update, no_update

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Output("ui-toast-store", "data", allow_duplicate=True),
        Input("retry-pro-request-btn", "n_clicks", allow_optional=True),
        State("session-store", "data"),
        State("auth-store", "data"),
        prevent_initial_call=True,
    )
    def retry_pending_pro_request(n, store_data, auth_data):
        if not n:
            raise PreventUpdate
        session = _session_from_store(store_data)
        auth_user = (auth_data or {}).get("user") or {}
        pending_text = (session.pending_pro_request_text or "").strip()
        if not pending_text:
            session.add_message("assistant", "当前没有待继续的 Pro 请求。")
            return _commit_session(session), _make_toast("没有待继续任务", "当前没有待继续的 Pro 请求。", "warning", 8000)
        if not is_pro_user(auth_user):
            session.add_message("assistant", build_standard_limit_message(pending_text, "pro"))
            return _commit_session(session), _make_toast("需要 Pro 权限", "该请求需要 Pro 版权限。", "warning", 8000)
        if _has_active_model_task(session):
            session.add_message("assistant", "当前已有任务正在运行，请等待完成后再继续上次 Pro 请求。")
            return _commit_session(session), _make_toast("任务正在运行", "当前已有任务正在运行，请等待完成后再继续。", "warning", 8000)

        if _looks_like_mapping_request(pending_text, session):
            try:
                preflight = preflight_mapping_region(pending_text, session.latest_uploaded_files)
            except Exception as exc:
                preflight = {"blocked": True, "popup": True, "title": "AOI预检失败", "message": f"制图前样点-AOI一致性预检失败：{exc}"}
            if preflight.get("blocked"):
                title = preflight.get("title") or "样点与制图区域不匹配"
                msg = preflight.get("message") or "样点数据与用户指定制图区域不匹配，已停止制图。"
                _push_ui_notification(session, title, msg, "error")
                return _commit_session(session), _make_toast(title, msg, "error", 12000)

        session.has_requested_map = True
        _start_new_mapping_context(session, pending_text)
        session.active_layout_key = None
        session.active_data_source = "pro_web"
        session.latest_result_kind = "rfk_running"
        task_id = start_rfk_task(data_source="pro_web", session_id=session.session_id, request_text=pending_text, user_id=auth_user.get("id"), uploaded_files=session.latest_uploaded_files)
        session.latest_rfk_task_id = task_id
        session.pending_pro_request_text = None
        session.pending_pro_target = {}
        task = TASKS.get(task_id)
        msg = "已继续执行上次 Pro 请求。我会先执行自动数据准备，再执行土壤有机质制图。"
        if task and task.status == "error" and task.error:
            session.add_message("assistant", task.error)
            return _commit_session(session), _make_toast("制图失败", _clean_task_error(task.error), "error", 12000)
        session.add_message("assistant", msg)
        return _commit_session(session), _make_toast("开始制图", "已继续执行上次 Pro 请求，正在准备协变量并训练模型。", "info", 5000)

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Output("ui-toast-store", "data", allow_duplicate=True),
        Output("domestic-handoff-store", "data", allow_duplicate=True),
        Input("mapping-source-user-btn", "n_clicks", allow_optional=True),
        Input("mapping-source-domestic-btn", "n_clicks", allow_optional=True),
        Input("mapping-source-gee-btn", "n_clicks", allow_optional=True),
        Input("mapping-source-cancel-btn", "n_clicks", allow_optional=True),
        State("domestic-handoff-store", "data", allow_optional=True),
        State("session-store", "data", allow_optional=True),
        State("auth-store", "data", allow_optional=True),
        prevent_initial_call=True,
    )
    def handle_mapping_source_choice(user_n, domestic_n, gee_n, cancel_n, payload, store_data, auth_data):
        trig = ctx.triggered_id
        if trig == "mapping-source-cancel-btn":
            session = _session_from_store(store_data)
            session.add_message("assistant", "已取消本次制图数据源选择。")
            return _commit_session(session), _make_toast("已取消", "本次制图数据源选择已取消。", "warning", 5000), None
        if trig not in {"mapping-source-user-btn", "mapping-source-domestic-btn", "mapping-source-gee-btn"}:
            raise PreventUpdate
        payload = payload or {}
        request_text = payload.get("request_text") or ""
        if not request_text:
            raise PreventUpdate
        choice = "user" if trig == "mapping-source-user-btn" else ("domestic" if trig == "mapping-source-domestic-btn" else "gee")
        data_source = data_source_for_choice(choice)
        session = _session_from_store(store_data)
        auth_user = (auth_data or {}).get("user") or {}

        # V150: 同一个选择弹窗也服务于“单数据下载”。下载任务不做 AOI/样点预检，
        # 也不启动模型；选择 GEE 则直接下载并预览，选择 TPDC 则进入人工/自动接管流程。
        if str((payload or {}).get("task_type") or "").lower() == "download_only":
            if choice == "gee":
                try:
                    result = run_gee_download_task(request_text, auth_user=auth_user)
                    session.last_task_type = "download_only"
                    session.last_route_decision = {"mode": "download_only_gee", "result": result}
                    if result.get("ok") and result.get("display_tif"):
                        tif = str(result.get("display_tif"))
                        session.latest_result_kind = "uploaded_raster"
                        session.uploaded_result_paths = {"uploaded_tif": tif, "download_manifest": str(result.get("manifest_path") or "")}
                        session.last_result_paths = session.uploaded_result_paths.copy()
                        session.active_layout_key = None
                        session.add_message("assistant", "已选择 GEE（Google Earth Engine）下载数据，并已加载为地图栅格预览。\nManifest：" + str(result.get("manifest_path") or ""))
                        return _commit_session(session), _make_toast("GEE 数据已下载", "数据已通过 GEE 下载并加载到地图。", "success", 8000), None
                    msg = str(result.get("error") or "GEE 下载失败，未生成可显示栅格。")
                    session.add_message("assistant", "GEE 数据下载失败：" + msg)
                    return _commit_session(session), _make_toast("GEE 下载失败", msg, "error", 12000), None
                except Exception as exc:
                    msg = f"GEE 数据下载任务创建失败：{exc}"
                    session.add_message("assistant", msg)
                    return _commit_session(session), _make_toast("GEE 下载失败", msg, "error", 12000), None
            try:
                session.uploaded_result_paths = {k: v for k, v in (getattr(session, "uploaded_result_paths", {}) or {}).items() if k != "download_manifest"}
                download_job = prepare_download_only_task(request_text, auth_user=auth_user)
                handoff = download_job.get("handoff") or {}
                manifest = download_job.get("manifest") or {}
                manifest_path = manifest.get("manifest_path")
                launch_result = launch_capture_browser(manifest_path)
                handoff["auto_launch_result"] = launch_result
                handoff["auto_launched"] = bool(launch_result.get("ok"))
                session.last_task_type = "download_only"
                session.last_route_decision = {"mode": "download_only_tpdc", "download_manifest": manifest_path, "auto_launch_result": launch_result, "download_notified": [], "download_progress_snapshot": {}, "download_progress_compact": {}}
                session.add_message("assistant", "已选择国内数据平台：国家青藏高原科学数据中心（TPDC）下载数据。系统会直接打开 TPDC 登录页并自动填写账号密码；验证码由你手动输入，登录后自动检索核心关键词。你进入数据详情页并点击下载后，系统会自动捕捉 FTP 信息并开始下载。")
                toast = _make_toast("已打开 TPDC", "TPDC 登录页已打开并开始自动填充账号密码。请手动输入验证码。", "info", 9000) if launch_result.get("ok") else _make_toast("TPDC 打开失败", str(launch_result.get("error") or launch_result), "error", 12000)
                return _commit_session(session), toast, None
            except Exception as exc:
                msg = f"TPDC 数据下载任务创建失败：{exc}"
                session.add_message("assistant", msg)
                return _commit_session(session), _make_toast("下载任务创建失败", msg, "error", 12000), None

        data_gate = _mapping_data_gate(session, choice=choice)
        if not data_gate.get("ok"):
            msg = data_gate.get("message") or "当前数据尚未满足制图条件。"
            session.add_message("assistant", msg)
            return _commit_session(session), _make_toast(data_gate.get("title") or "数据未就绪", msg, "warning", 10000), None

        # AOI 预检仍然必须先过。
        try:
            preflight = preflight_mapping_region(request_text, session.latest_uploaded_files)
        except Exception as exc:
            preflight = {"blocked": True, "popup": True, "title": "AOI预检失败", "message": f"制图前样点-AOI一致性预检失败：{exc}"}
        if preflight.get("blocked"):
            title = preflight.get("title") or "样点与制图区域不匹配"
            msg = preflight.get("message") or "样点数据与制图区域不匹配，已停止制图。"
            _push_ui_notification(session, title, msg, "error")
            session.last_task_type = "preflight_blocked"
            return _commit_session(session), _make_toast(title, msg, "error", 12000), None
        if _has_active_model_task(session):
            session.add_message("assistant", "当前已有任务正在运行，请等待完成后再启动新的计算。")
            return _commit_session(session), _make_toast("任务正在运行", "当前已有任务正在运行。", "warning", 8000), None
        session.active_data_source = data_source
        session.has_requested_map = True
        _start_new_mapping_context(session, request_text)
        session.active_layout_key = None
        session.latest_result_kind = "rfk_running"
        session.last_task_type = "rfk_mapping"
        task_id = start_rfk_task(data_source=data_source, session_id=session.session_id, request_text=request_text, user_id=auth_user.get("id"), uploaded_files=session.latest_uploaded_files)
        session.latest_rfk_task_id = task_id
        if choice == "user":
            label = "只使用用户上传数据"
            msg = "已选择只使用用户上传数据制图。系统不会额外下载缺失协变量；若用户上传协变量较少，本次结果应作为基线图解释。"
        elif choice == "domestic":
            label = "国内数据平台：国家青藏高原科学数据中心（TPDC）"
            msg = "已选择用户数据优先，并用国内数据平台作为缺失协变量补充来源：国家青藏高原科学数据中心（TPDC）。TPDC 免费数据较多、来源审计更强，但下载较慢，可能需要验证码、人机验证、协议确认或 FTP 参数；系统会在需要时弹出接管窗口。"
        else:
            label = "用户数据/国内数据平台"
            msg = "已开始土壤有机质制图。系统优先使用用户数据，必要时仅使用国内 TPDC 补充数据。"
        session.add_message("assistant", msg + "\n已开始执行土壤有机质制图。")
        return _commit_session(session), _make_toast("开始制图", f"已选择：{label}。制图流程已启动。", "info", 6000), None

    @app.callback(
        Output("session-store", "data", allow_duplicate=True),
        Output("ui-toast-store", "data", allow_duplicate=True),
        Output("domestic-handoff-store", "data", allow_duplicate=True),
        Input("task-poll", "n_intervals"),
        State("session-store", "data"),
        prevent_initial_call=True,
    )
    def poll_tasks(n, store_data):
        session = _session_from_store(store_data)
        changed = False
        toast = no_update
        handoff_payload = no_update

        # RFK task polling must be robust even if the browser store still says
        # "rfk_running" after the backend has already finished. Earlier builds
        # could leave the right progress bar at 80–86% and never load the result
        # map because only intermediate progress snapshots were synced.
        if session.latest_rfk_task_id:
            task = TASKS.get(session.latest_rfk_task_id)
            decision = getattr(session, "last_route_decision", {}) or {}
            if task and task.status in {"queued", "running"}:
                prog_sig = {"task_id": task.task_id, "progress": int(task.progress or 0), "stage": str(task.stage or ""), "status": task.status}
                if decision.get("rfk_progress_snapshot") != prog_sig:
                    decision["rfk_progress_snapshot"] = prog_sig
                    session.last_route_decision = decision
                    changed = True
            if task and task.status == "done":
                already_notified = bool((task.result_paths or {}).get("_ui_rfk_done_notified")) or decision.get("rfk_done_task_id") == task.task_id
                session.som_map_shown = True
                session.latest_result_kind = "som_map"
                _mark_mapping_completed_for_current_context(session, task)
                assigned_paths = _assign_display_map_title(session, dict(task.result_paths or {}))
                try:
                    task.result_paths = dict(assigned_paths)
                except Exception:
                    pass
                session.rfk_result_paths = dict(assigned_paths)
                session.last_result_paths = dict(assigned_paths)
                session.map_style_prefs = {**(getattr(session, "map_style_prefs", {}) or {}), "show_samples": False}
                session.active_layout_key = "som_layout"
                decision["rfk_progress_snapshot"] = {"task_id": task.task_id, "progress": 100, "stage": "已完成", "status": "done"}
                decision["rfk_done_task_id"] = task.task_id
                session.last_route_decision = decision
                if not already_notified:
                    metrics = load_json_if_exists((task.result_paths or {}).get("report_json", ""))
                    if not isinstance(metrics, dict):
                        metrics = {}
                    try:
                        evidence = read_mapping_result(task.result_paths or {}, metrics)
                        if isinstance(evidence.get("report"), dict):
                            metrics = {**evidence.get("report"), **metrics}
                        metrics["_result_reader_evidence"] = {k: evidence.get(k) for k in ["ok", "missing", "report_json", "model_name", "pred_tif", "public_run_code"]}
                        try:
                            pro_console_log("RESULT", "后台报告校验信息", {"report_sha256": evidence.get("report_sha256"), "run_fingerprint": evidence.get("run_fingerprint"), "public_run_code": evidence.get("public_run_code")})
                        except Exception:
                            pass
                    except Exception:
                        pass
                    # Use the cleaned user-facing result paths in the final chat card.
                    for k in ["report_json", "model_report_json", "model_record_txt", "pred_tif", "preview_png", "final_result_dir"]:
                        if (task.result_paths or {}).get(k):
                            metrics[k] = (task.result_paths or {}).get(k)
                    gee_report = load_json_if_exists((task.result_paths or {}).get("gee_formal_report_json", ""))
                    gcp_auto_report = load_json_if_exists((task.result_paths or {}).get("gee_gcp_report_json", ""))
                    if isinstance(gee_report, dict) and gee_report:
                        metrics = {**metrics, "gee_formal_report": gee_report}
                    if isinstance(gcp_auto_report, dict) and gcp_auto_report:
                        metrics = {**metrics, "gee_gcp_report": gcp_auto_report}
                    note = _notification_from_mapping_reports(task.result_paths or {})
                    if note:
                        _push_ui_notification(session, note.get("title") or "系统提醒", note.get("message") or "", note.get("type") or "warning")
                    msg = "制图完成，结果图层已加载到地图。"
                    if note and note.get("message"):
                        msg += "\n" + str(note.get("message"))
                    toast = _make_toast("制图完成", msg, "success", 5000)
                    try:
                        final_text = build_map_completion_text(metrics)
                    except Exception as exc:
                        final_text = "✅ 土壤有机质制图完成\n\n结果图层已加载到地图。结果文件已保存到：" + str((task.result_paths or {}).get("final_result_dir") or (task.result_paths or {}).get("pred_tif") or "结果目录") + f"\n\n结果分析文本生成失败：{exc}"
                    session.add_message("assistant", final_text)
                    try:
                        task.result_paths = {**(task.result_paths or {}), "_ui_rfk_done_notified": True}
                    except Exception:
                        pass
                if not already_notified:
                    changed = True
            elif task and task.status == "error":
                already_notified = bool((task.result_paths or {}).get("_ui_rfk_error_notified")) or decision.get("rfk_error_task_id") == task.task_id
                session.latest_result_kind = None
                _mark_mapping_not_completed(session, "error")
                decision["rfk_progress_snapshot"] = {"task_id": task.task_id, "progress": 100, "stage": "运行失败", "status": "error"}
                decision["rfk_error_task_id"] = task.task_id
                session.last_route_decision = decision
                err_raw = task.error or ""
                handoff = _parse_domestic_handoff_error(err_raw)
                err = _clean_task_error(task.error)
                if handoff:
                    handoff_payload = handoff
                    toast = _make_toast("需要人工下载国内平台数据", "已弹出人工接管窗口。请完成登录/注册/下载真实数据文件后重新提交制图任务。", "warning", 15000)
                else:
                    toast = _make_toast("制图失败", err, "error", 12000)
                if not already_notified:
                    session.add_message("assistant", "❌ 本次制图运行失败。\n原因：" + err)
                    try:
                        task.result_paths = {**(task.result_paths or {}), "_ui_rfk_error_notified": True}
                    except Exception:
                        pass
                changed = True

        if session.latest_gcp_task_id:
            task = TASKS.get(session.latest_gcp_task_id)
            decision = getattr(session, "last_route_decision", {}) or {}
            if task and task.status in {"queued", "running"}:
                prog_sig = {"task_id": task.task_id, "progress": int(task.progress or 0), "stage": str(task.stage or ""), "status": task.status}
                if decision.get("gcp_progress_snapshot") != prog_sig:
                    decision["gcp_progress_snapshot"] = prog_sig
                    session.last_route_decision = decision
                    changed = True
            if task and task.status == "done":
                already_notified = bool((task.result_paths or {}).get("_ui_gcp_done_notified")) or decision.get("gcp_done_task_id") == task.task_id
                if not already_notified:
                    session.gcp_shown = True
                    session.latest_result_kind = "gcp_map"
                    gcp_paths = _assign_uncertainty_display_title(session, task.result_paths.copy())
                    try:
                        task.result_paths = dict(gcp_paths)
                    except Exception:
                        pass
                    session.gcp_result_paths = dict(gcp_paths)
                    session.last_result_paths = dict(gcp_paths)
                    session.map_style_prefs = {**(getattr(session, "map_style_prefs", {}) or {}), "show_samples": False}
                    session.active_layout_key = "gcp_layout"
                    decision["gcp_progress_snapshot"] = {"task_id": task.task_id, "progress": 100, "stage": "已完成", "status": "done"}
                    decision["gcp_done_task_id"] = task.task_id
                    session.last_route_decision = decision
                    metrics = load_json_if_exists(task.result_paths.get("report_json", ""))
                    try:
                        evidence = read_gcp_result(task.result_paths or {}, metrics if isinstance(metrics, dict) else {})
                        if isinstance(evidence.get("report"), dict):
                            metrics = evidence.get("report")
                    except Exception:
                        pass
                    toast = _make_toast("不确定性分析完成", "不确定性分析完成，结果图层已加载。现在可以继续发送下载、分析或其他新指令。", "success", 5000)
                    session.add_message("assistant", build_gcp_completion_text(metrics, result_paths=task.result_paths or {}))
                    try:
                        task.result_paths = {**(task.result_paths or {}), "_ui_gcp_done_notified": True}
                    except Exception:
                        pass
                    session.last_task_type = "general_chat"
                    changed = True
            elif task and task.status == "error":
                already_notified = bool((task.result_paths or {}).get("_ui_gcp_error_notified")) or decision.get("gcp_error_task_id") == task.task_id
                session.latest_result_kind = None
                err = _clean_task_error(task.error)
                decision["gcp_error_task_id"] = task.task_id
                session.last_route_decision = decision
                toast = _make_toast("不确定性分析失败", err, "error", 12000)
                if not already_notified:
                    session.add_message("assistant", "本次不确定性分析运行失败：" + err)
                    try:
                        task.result_paths = {**(task.result_paths or {}), "_ui_gcp_error_notified": True}
                    except Exception:
                        pass
                changed = True

        # TPDC FTP capture/download progress: keep the right panel live and add
        # milestone messages to the chat history. This runs even when no RFK/GCP
        # model task is active.
        try:
            manifest_path = _download_manifest_path_from_session(session)
            decision = getattr(session, "last_route_decision", {}) or {}
            if manifest_path:
                snap = _normalize_ftp_status_for_ui(manifest_path)
                # Reduce noisy full-page rerenders: compare a compact progress signature.
                compact = {
                    "status": snap.get("status"),
                    "watcher": snap.get("watcher"),
                    "ftp_accounts_count": snap.get("ftp_accounts_count"),
                    "host": snap.get("host"),
                    "pid": snap.get("pid"),
                    "progress_percent": round(float(snap.get("progress_percent") or 0.0), 1),
                    "downloaded_bytes": int(float(snap.get("downloaded_bytes") or 0)),
                    "total_bytes": int(float(snap.get("total_bytes") or 0)),
                    "current_index": snap.get("current_index"),
                    "file_count": snap.get("file_count"),
                    "current_remote": snap.get("current_remote"),
                    "download_dir": snap.get("download_dir"),
                    "error": snap.get("error"),
                }
                prev = decision.get("download_progress_compact") if isinstance(decision.get("download_progress_compact"), dict) else {}
                if compact != prev:
                    decision["download_progress_snapshot"] = snap
                    decision["download_progress_compact"] = compact
                    session.last_route_decision = decision
                    changed = True
                notified = set(decision.get("download_notified") or [])
                st = str(snap.get("status") or "")
                watcher = str(snap.get("watcher") or "")
                # Capture success + download process started. This is the main user-facing milestone.
                if (st in {"ftp_credentials_detected", "ftp_download_process_started", "ftp_downloading"} or watcher in {"captured", "download_started", "downloading"}) and "ftp_capture_started" not in notified and not _download_ui_marker_seen(manifest_path, "ftp_capture_started"):
                    msg = "已捕捉到 TPDC 弹窗中的 FTP 主机、端口、用户名和密码，正在自动开始下载数据。"
                    if snap.get("host"):
                        msg += "\n主机：" + str(snap.get("host"))
                    if snap.get("download_dir"):
                        msg += "\n保存位置：" + str(snap.get("download_dir"))
                    if snap.get("pid"):
                        msg += "\n下载进程 PID：" + str(snap.get("pid"))
                    msg += "\n右侧已显示实时下载进度；FTP 临时密码不会在界面中明文显示。"
                    session.add_message("assistant", msg)
                    notified.add("ftp_capture_started")
                    _download_ui_mark_seen(manifest_path, "ftp_capture_started")
                    decision["download_notified"] = sorted(notified)
                    session.last_route_decision = decision
                    toast = _make_toast("已捕捉 FTP，开始下载", "TPDC FTP 信息已识别，后台下载已启动。", "success", 9000)
                    changed = True
                if st in {"ftp_download_completed", "download_verified"} and "ftp_download_completed" not in notified and not _download_ui_marker_seen(manifest_path, "ftp_download_completed"):
                    mapping_intent = _infer_download_mapping_intent_from_session(session, manifest_path)
                    post = finalize_tpdc_download_after_ftp(manifest_path, mapping_intent=mapping_intent)
                    msg = "TPDC 数据下载完成。"
                    if snap.get("download_dir"):
                        msg += "\n原始 FTP 下载目录：" + str(snap.get("download_dir"))
                    if post.get("final_data_dir"):
                        msg += "\n最终数据文件夹：" + str(post.get("final_data_dir"))
                    derived = list(post.get("derived_tifs") or [])
                    final_files = list(post.get("final_files") or [])
                    if post.get("ok"):
                        msg += f"\n已完成下载后处理：保留/整理目标年份数据 {len(final_files)} 个，生成或识别目标区域栅格 {len(derived)} 个。"
                        region_clip = post.get("region_clip") if isinstance(post.get("region_clip"), dict) else {}
                        if region_clip.get("ok"):
                            msg += "\n已按用户指定区域完成裁剪；最终文件夹只保留目标年份、目标区域成果。"
                        elif region_clip and not region_clip.get("skipped"):
                            msg += "\n区域裁剪未完成：" + str(region_clip.get("error") or region_clip.get("reason") or "未生成裁剪结果。")
                    else:
                        msg += "\n下载后处理未完全完成：" + str(post.get("error") or "未生成可用栅格。")
                    if mapping_intent and derived:
                        _register_downloaded_tifs_as_covariates(session, [str(x) for x in derived])
                        session.latest_result_kind = "uploaded_raster"
                        session.uploaded_result_paths = {"uploaded_tif": str(derived[0]), "download_manifest": str(manifest_path or "")}
                        session.last_result_paths = session.uploaded_result_paths.copy()
                        session.active_layout_key = None
                        msg += "\n我判断这次下载属于当前制图工作流，已把下载得到的栅格作为环境协变量加入当前会话，并与已上传样点/协变量一起重新生成预处理计划。"
                        recent_text = " ".join(str(m.get("content") or "") for m in (getattr(session, "chat_history", []) or [])[-12:])
                        if (not _is_data_review_only_request(recent_text)) and re.search(r"制图|建模|开始.*土壤有机质|进行.*土壤有机质", recent_text, re.I) and not _has_active_model_task(session):
                            try:
                                session.has_requested_map = True
                                _start_new_mapping_context(session, "下载数据后自动接入当前数据并执行土壤有机质制图")
                                session.latest_result_kind = "rfk_running"
                                task_id = start_rfk_task(data_source=session.active_data_source, session_id=session.session_id, request_text="下载数据后自动接入当前数据并执行土壤有机质制图", user_id=None, uploaded_files=session.latest_uploaded_files)
                                session.latest_rfk_task_id = task_id
                                msg += "\n检测到你前文需要继续制图，已自动启动土壤有机质制图任务。"
                            except Exception as exc:
                                msg += "\n尝试自动启动制图失败：" + str(exc)
                    elif mapping_intent:
                        msg += "\n我判断这次下载与当前制图工作流有关，但暂未生成可入模栅格。请查看下载数据格式，或让我指定变量/时间层后继续转换。"
                    else:
                        msg += "\n我判断这次是单纯数据下载任务，因此不自动启动制图；最终文件夹中只保留整理后的原始数据和对应栅格。"
                    session.add_message("assistant", msg)
                    notified.add("ftp_download_completed")
                    _download_ui_mark_seen(manifest_path, "ftp_download_completed")
                    decision["download_notified"] = sorted(notified)
                    decision["download_postprocess"] = post
                    session.last_route_decision = decision
                    toast = _make_toast("下载完成", "TPDC 数据已下载完成，并已执行下载后处理。", "success", 10000)
                    changed = True
                if ("failed" in st or st in {"ftp_host_failed"}) and "ftp_download_failed" not in notified and not _download_ui_marker_seen(manifest_path, "ftp_download_failed"):
                    msg = "TPDC FTP 下载失败或中断。"
                    if snap.get("error"):
                        raw_err = str(snap.get("error"))
                        msg += "\n原因：" + raw_err
                        msg += "\n解释：" + _explain_ftp_download_error(raw_err)
                    if snap.get("download_dir"):
                        msg += "\n任务目录：" + str(snap.get("download_dir"))
                    msg += "\n可以重试下载，或使用任务目录中的 WinSCP 接管脚本继续下载。"
                    session.add_message("assistant", msg)
                    notified.add("ftp_download_failed")
                    _download_ui_mark_seen(manifest_path, "ftp_download_failed")
                    decision["download_notified"] = sorted(notified)
                    session.last_route_decision = decision
                    toast = _make_toast("FTP 下载异常", "下载失败或中断，请查看聊天历史和任务目录日志。", "error", 12000)
                    changed = True
        except Exception as exc:
            try:
                pro_console_log("TPDC_PROGRESS", "读取FTP下载进度失败", {"error": str(exc)})
            except Exception:
                pass

        if changed:
            return _commit_session(session), toast, handoff_payload
        raise PreventUpdate

    server = app.server

    @server.get("/__local_file")
    def serve_local_file():
        raw_path = request.args.get("path", "")
        if not raw_path:
            return ("missing path", 400)
        p = Path(raw_path).resolve()
        allowed_roots = [BASE_DIR.resolve(), DATA_DIR.resolve(), Path(r"E:\Agent_DSM").resolve()]
        if not any(str(p).startswith(str(root)) for root in allowed_roots):
            return ("forbidden", 403)
        if not p.exists() or not p.is_file():
            return ("not found", 404)
        return send_file(p)

    @server.post("/__render_overlay")
    def render_overlay():
        payload = request.get_json(silent=True) or {}
        tif_path = payload.get("tif_path")
        if not tif_path:
            return jsonify({"ok": False, "error": "missing tif_path"}), 400
        try:
            bundle = get_overlay_bundle(
                tif_path=tif_path,
                palette=payload.get("palette") or "ArcGIS Pro｜Green Continuous",
                opacity=float(payload.get("opacity") if payload.get("opacity") is not None else 1.0),
                reverse=bool(payload.get("reverse")),
                palette_colors=payload.get("palette_colors"),
                admin_region=payload.get("admin_region") or None,
                clcd_mask_path=payload.get("clcd_mask_path") or None,
                noncropland_mode=payload.get("noncropland_mode") or "fill",
                noncropland_color=payload.get("noncropland_color") or None,
            )
            return jsonify({"ok": True, **bundle})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500



    @server.post("/__client_chat_ack")
    def client_chat_ack():
        """Persist a frontend-confirmed UI event after the browser has actually rendered it.

        Used for events whose completion is only known in the browser, such as a
        Leaflet imageOverlay load after palette/style changes or after a newly
        uploaded raster becomes visible.  This avoids telling the user "已完成"
        before the map has actually refreshed.
        """
        payload = request.get_json(silent=True) or {}
        session_id = str(payload.get("session_id") or "").strip()
        text = str(payload.get("text") or "").strip()
        key = str(payload.get("key") or "").strip()[:160]
        if not session_id or not text:
            return jsonify({"ok": False, "error": "missing session_id or text"}), 400
        try:
            active = SESSION_STORE.read_active(session_id) or {}
            session_data = active.get("session_data") or {}
            if not session_data:
                return jsonify({"ok": True, "persisted": False})
            mem = dict(session_data.get("task_memory") or {})
            seen = set(mem.get("client_chat_ack_keys") or [])
            if key and key in seen:
                return jsonify({"ok": True, "dedup": True})
            session = SessionState.from_dict(session_data)
            session.add_message("assistant", text)
            mem = dict(session.task_memory or {})
            seen = set(mem.get("client_chat_ack_keys") or [])
            if key:
                seen.add(key)
                mem["client_chat_ack_keys"] = sorted(seen)[-300:]
                # Deferred layer/style messages are one-shot: after the browser has
                # rendered the active layer and persisted the confirmation, remove
                # them so later manual layer switches do not replay stale upload or
                # task-start feedback.
                if key.startswith("deferred:pending_layer_loaded_messages:"):
                    mem["pending_layer_loaded_messages"] = []
                if key.startswith("deferred:pending_style_loaded_messages:"):
                    mem["pending_style_loaded_messages"] = []
                session.task_memory = mem
            SESSION_STORE.save_active(session.to_dict())
            return jsonify({"ok": True, "persisted": True})
        except Exception as exc:
            try:
                pro_console_log("CLIENT_ACK", "前端渲染确认写入失败", {"session_id": session_id, "error": str(exc)})
            except Exception:
                pass
            return jsonify({"ok": False, "error": str(exc)}), 500


    @server.post("/__raster_cache")
    def raster_cache_for_frontend():
        """Return a compact Float32 raster cache for zero-roundtrip hover readout.

        V105: Real-time hover should not call the backend on every mouse move.
        This endpoint is called once when a result GeoTIFF is loaded. The browser
        keeps the Float32Array in memory and samples it synchronously while the
        user moves the mouse.
        """
        payload = request.get_json(silent=True) or {}
        tif_path = payload.get("tif_path") or payload.get("path")
        admin_region = payload.get("admin_region") or None
        if not tif_path:
            return jsonify({"ok": False, "error": "missing tif_path"}), 400
        try:
            import base64 as _base64
            import numpy as _np
            import rasterio as _rasterio
            from rasterio.enums import Resampling as _Resampling
            from rasterio.warp import transform_bounds as _transform_bounds

            p = Path(str(tif_path)).resolve()
            allowed_roots = [BASE_DIR.resolve(), DATA_DIR.resolve(), Path(r"E:\Agent_DSM").resolve()]
            if not any(str(p).startswith(str(root)) for root in allowed_roots):
                return jsonify({"ok": False, "error": "forbidden"}), 403
            if not p.exists() or not p.is_file():
                return jsonify({"ok": False, "error": "not found"}), 404

            max_cells = int(os.getenv("WEBGIS_HOVER_CACHE_MAX_CELLS", "900000"))
            with _rasterio.open(p) as src:
                src_w, src_h = int(src.width), int(src.height)
                if src_w <= 0 or src_h <= 0:
                    return jsonify({"ok": False, "error": "empty raster"}), 400
                scale = max(1.0, (src_w * src_h / max(max_cells, 1)) ** 0.5)
                out_w = max(1, int(round(src_w / scale)))
                out_h = max(1, int(round(src_h / scale)))
                raw = src.read(1, out_shape=(out_h, out_w), masked=True, resampling=_Resampling.nearest).astype("float32")
                vals = _np.asarray(raw.filled(_np.nan), dtype="<f4")
                # Treat explicit nodata, GDAL masks and inferred edge background
                # values as NaN; JS reads NaN as NoData. The astype(float32) before
                # filled(np.nan) also fixes integer rasters such as int8.
                valid_mask, nodata_meta = raster_valid_mask(src, vals, raw, admin_region=admin_region)
                vals[~valid_mask] = _np.nan
                vals[~_np.isfinite(vals)] = _np.nan
                bounds = src.bounds
                try:
                    if src.crs:
                        west, south, east, north = _transform_bounds(src.crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top, densify_pts=21)
                    else:
                        west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top
                except Exception:
                    west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top
                b64 = _base64.b64encode(vals.tobytes(order="C")).decode("ascii")
                return jsonify({
                    "ok": True,
                    "tif_path": str(p),
                    "width": out_w,
                    "height": out_h,
                    "source_width": src_w,
                    "source_height": src_h,
                    "downsampled": bool(out_w != src_w or out_h != src_h),
                    "bounds": [[float(south), float(west)], [float(north), float(east)]],
                    "dtype": "float32",
                    "nodata_is_nan": True,
                    "nodata_meta": nodata_meta,
                    "values_b64": b64,
                })
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500


    @server.post("/__sample_raster")
    def sample_raster_value():
        payload = request.get_json(silent=True) or {}
        tif_path = payload.get("tif_path") or payload.get("path")
        if not tif_path:
            return jsonify({"ok": False, "error": "missing tif_path"}), 400
        try:
            lon = float(payload.get("lon"))
            lat = float(payload.get("lat"))
        except Exception:
            return jsonify({"ok": False, "error": "invalid lon/lat"}), 400
        try:
            import numpy as _np
            import rasterio as _rasterio
            from rasterio.warp import transform as _rio_transform
            p = Path(str(tif_path)).resolve()
            allowed_roots = [BASE_DIR.resolve(), DATA_DIR.resolve(), Path(r"E:\Agent_DSM").resolve()]
            if not any(str(p).startswith(str(root)) for root in allowed_roots):
                return jsonify({"ok": False, "error": "forbidden"}), 403
            if not p.exists() or not p.is_file():
                return jsonify({"ok": False, "error": "not found"}), 404
            with _rasterio.open(p) as src:
                x, y = lon, lat
                if src.crs and str(src.crs).upper() not in {"EPSG:4326", "OGC:CRS84"}:
                    xs, ys = _rio_transform("EPSG:4326", src.crs, [lon], [lat])
                    x, y = xs[0], ys[0]
                row, col = src.index(x, y)
                if row < 0 or col < 0 or row >= src.height or col >= src.width:
                    return jsonify({"ok": True, "nodata": True, "message": "当前位置超出栅格范围"})
                arr = src.read(1, window=((row, row + 1), (col, col + 1)), masked=True)
                v = arr[0, 0]
                nodata = bool(_np.ma.is_masked(v))
                if not nodata:
                    try:
                        fv = float(v)
                        if src.nodata is not None and abs(fv - float(src.nodata)) < 1e-9:
                            nodata = True
                    except Exception:
                        nodata = True
                if nodata:
                    return jsonify({"ok": True, "nodata": True, "row": int(row), "col": int(col), "message": "该位置为 NODATA"})
                return jsonify({"ok": True, "nodata": False, "value": float(v), "row": int(row), "col": int(col)})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500


    @server.get("/__style_catalog")
    def style_catalog():
        return jsonify({"ok": True, **get_style_catalog()})


    @server.post("/__vip_upgrade")
    def vip_upgrade():
        payload = request.get_json(silent=True) or {}
        result = upgrade_user_to_pro(payload.get("user_id"))
        return jsonify(result), (200 if result.get("ok") else 400)


    @server.get("/__vip_status")
    def vip_status():
        user_id = request.args.get("user_id", "")
        result = get_user_plan_status(user_id)
        return jsonify(result), (200 if result.get("ok") else 400)


    @server.get("/__pro_usage")
    def pro_usage():
        user_id = request.args.get("user_id", "")
        return jsonify({"ok": True, "records": list_recent_pro_usage(user_id or None, limit=20)})


    @server.get("/__qgis_svg")
    def qgis_svg():
        raw_path = request.args.get("path", "")
        if not raw_path:
            return ("missing path", 400)
        p = Path(raw_path).resolve()
        allowed_roots = [Path(x).resolve() for x in (os.getenv("QGIS_SVG_PATHS", "").split(os.pathsep)) if x.strip()]
        if not allowed_roots or not any(str(p).startswith(str(root)) for root in allowed_roots):
            return ("forbidden", 403)
        if not p.exists() or not p.is_file():
            return ("not found", 404)
        return send_file(p, mimetype="image/svg+xml")

    @server.post("/__heartbeat")
    def _heartbeat():
        payload = request.get_json(silent=True) or {}
        BROWSER_SESSIONS.heartbeat(payload.get("client_id"), payload.get("session_id"))
        return ("", 204)

    @server.post("/__shutdown_intent")
    def _shutdown_intent():
        payload = request.get_json(silent=True) or {}
        BROWSER_SESSIONS.shutdown(payload.get("client_id"), payload.get("session_id"))
        return ("", 204)

    BROWSER_SESSIONS.start_monitor()
    return app
