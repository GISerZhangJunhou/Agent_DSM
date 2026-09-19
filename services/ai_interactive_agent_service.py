from __future__ import annotations

"""
AI-first interactive agent for the formal digital soil mapping application.

Architecture principle:
1. Every user message enters this AI layer first.
2. Normal questions are answered as a normal AI assistant would answer.
3. GIS / mapping / download / uncertainty functions are exposed as tools/actions.
4. Dash only executes the structured `mode` returned by the agent; Dash should not
   use keyword routing before this layer.
"""

import json
import os
import re
from typing import Any, Dict, Literal

from pydantic import BaseModel, Field

from services.agent_service import AgentOutcome
from services.llm_service import _get_llm, compose_agent_reply
from services.result_context_service import build_result_context_text
from services.agent_memory_service import build_agent_memory_payload, build_agent_memory_text, remember_user_turn

try:  # LangChain 1.x / LangGraph path
    from langchain.agents import create_agent
    from langchain.tools import tool
    from langgraph.checkpoint.memory import InMemorySaver
except Exception:  # keep the project importable before dependencies are installed
    create_agent = None  # type: ignore
    tool = None  # type: ignore
    InMemorySaver = None  # type: ignore

_CHECKPOINTER = InMemorySaver() if InMemorySaver is not None else None


class SoilAgentDecision(BaseModel):
    """Structured decision consumed by ui.dashboard."""

    mode: Literal[
        "reply",
        "followup",
        "start_rfk",
        "start_gcp",
        "show_uploaded_raster",
        "mapping_source_choice",
        "download_source_choice",
        "start_download_gee",
        "start_download_tpdc",
        "standardize_spatial",
        "tpdc_search",
    ] = "reply"
    assistant_text: str = Field(default="", description="给用户看的中文回复")
    data_source: Literal[
        "default",       # built-in local default data
        "user",          # uploaded data only
        "pro_user",      # uploaded data only, Pro route
        "pro_domestic",  # TPDC / National Tibetan Plateau Data Center
        "pro_web",       # generic online preparation route
        "none",
    ] = "none"
    use_result_context: bool = False
    reason: str = ""
    payload: Dict[str, Any] = Field(default_factory=dict)


VALID_DATA_SOURCES = {"default", "user", "pro_user", "pro_domestic", "pro_web", "none"}


GCP_TOPIC_WORDS = [
    "不确定性", "不确定性分析", "适用域", "AOA", "aoa", "GCP", "gcp", "GCP+AOA", "gcp+aoa",
    "外推风险", "预测区间", "置信区间", "可靠性分析", "高风险区域", "虚假自信"
]
GCP_EXECUTION_WORDS = [
    "开始", "执行", "运行", "进行", "开展", "启动", "继续", "进一步", "再", "重新", "生成", "输出", "计算", "做", "跑"
]
QUESTION_WORDS = ["什么", "哪些", "为什么", "为何", "怎么", "如何", "吗", "么", "能否", "可否", "是不是", "是否", "?", "？"]
DOWNLOAD_TOPIC_WORDS = ["下载", "获取数据", "数据获取", "数据下载", "TPDC", "tpdc", "青藏高原科学数据中心", "数据中心", "公开数据", "FTP", "ftp"]
DOWNLOAD_EXECUTION_WORDS = ["下载", "获取", "打开", "搜索", "检索", "找", "准备", "开始", "继续", "导入下载"]

MAPPING_TOPIC_WORDS = [
    "制图", "绘图", "画图", "出图", "绘制", "制作", "生成图", "预测图",
    "有机质图", "土壤有机质图", "SOM图", "som图", "重新制图", "再画", "再做一张"
]
MAPPING_EXECUTION_WORDS = ["开始", "执行", "运行", "进行", "生成", "绘制", "制作", "画", "出", "做", "跑", "重新", "再次", "继续", "用", "根据"]
OPTIMIZE_TOPIC_WORDS = ["提升精度", "提高精度", "优化精度", "精度优化", "重新优化", "调参", "重新训练", "重新建模", "优化模型", "更高精度"]
NEW_DATA_WORDS = ["新数据", "新样点", "新的一组", "另一组", "重新上传", "刚上传", "换一套", "这组数据", "这批数据"]


def _has_any_text(text: str, words: list[str]) -> bool:
    return any(w in str(text or "") for w in words)


def _is_explicit_download_request(text: str) -> bool:
    """Detect a current-turn download command without inheriting older GCP context."""
    t = str(text or "").strip()
    if not t or not _has_any_text(t, DOWNLOAD_TOPIC_WORDS):
        return False
    has_question = _has_any_text(t, QUESTION_WORDS) or bool(re.search(r"需要准备什么|怎么用|如何用|有什么要求", t))
    # Questions about downloading should remain in the conversational trunk.
    # Commands such as “下载2022年气温数据/请帮我下载气温数据” are routed to TPDC.
    if has_question:
        return False
    return _has_any_text(t, DOWNLOAD_EXECUTION_WORDS)


def _is_explicit_mapping_or_optimization_request(text: str) -> bool:
    """Detect current-turn mapping/rerun/accuracy-optimization commands.

    This guard is intentionally evaluated before GCP context guards so that after
    an uncertainty-analysis result, commands such as “重新制图需要提升精度” or
    “我又上传了一组数据，开始制图” start a new 制图模型 workflow instead of being
    swallowed by the previous GCP context or routed as general chat.
    """
    t = str(text or "").strip()
    if not t:
        return False
    if _is_explicit_download_request(t):
        return False
    # Do not hijack explicit GCP/AOA commands.
    if _has_any_text(t, GCP_TOPIC_WORDS) and not _has_any_text(t, MAPPING_TOPIC_WORDS + OPTIMIZE_TOPIC_WORDS):
        return False
    has_mapping_topic = _has_any_text(t, MAPPING_TOPIC_WORDS)
    has_optimize_topic = _has_any_text(t, OPTIMIZE_TOPIC_WORDS)
    has_new_data = _has_any_text(t, NEW_DATA_WORDS)
    has_action = _has_any_text(t, MAPPING_EXECUTION_WORDS)
    # Questions about how to improve precision should remain Q&A unless the user
    # requests execution with “重新/优化/提升/开始/运行”.
    has_pure_question = _has_any_text(t, QUESTION_WORDS) and not has_optimize_topic and not has_mapping_topic
    if has_pure_question:
        return False
    return bool((has_mapping_topic and has_action) or has_optimize_topic or (has_new_data and has_mapping_topic))


def _mapping_decision_from_context(user_text: str, session) -> SoilAgentDecision:
    ctx = _session_context(session)
    # Prefer current uploaded/user data after a user uploads or imports another group.
    active = ctx.get("active_data_source") or "user"
    if active not in VALID_DATA_SOURCES or active == "none":
        active = "user" if ctx.get("has_uploaded_files") else "default"
    if active == "user" and ctx.get("has_uploaded_files"):
        active = "pro_user"
    ready = _readiness_for_mapping(session, active)
    if not ready.get("ok") and active == "pro_user":
        ready = _readiness_for_mapping(session, "user")
    if not ready.get("ok") and ctx.get("has_uploaded_files"):
        # Let the dashboard mapping gate provide the precise blocking reason; the
        # important part here is to route the current command as a mapping command,
        # not as GCP continuation or generic chat.
        ds = "pro_user" if str(getattr(session, "active_data_source", "user")) in {"pro_user", "user", "default", "none"} else str(getattr(session, "active_data_source", "user"))
        return SoilAgentDecision(
            mode="start_rfk",
            assistant_text="收到，我将按当前数据和当前指令启动新一轮土壤有机质制图/精度优化流程。系统会先执行数据门控检查；如缺少样点或协变量，会在对话历史中给出明确拦截原因。",
            data_source=(ds if ds in VALID_DATA_SOURCES else "pro_user"),
            use_result_context=False,
            reason="current_turn_mapping_or_accuracy_optimization_after_previous_result",
            payload={"action": "start_new_rfk_mapping_or_optimization", "request_text": str(user_text or ""), "previous_result_kind": ctx.get("latest_result_kind")},
        )
    if not ready.get("ok"):
        return SoilAgentDecision(
            mode="followup",
            assistant_text="我理解你现在要重新制图或提升精度，但当前会话没有识别到满足制图入口的数据。请上传/导入样点和环境协变量，或说明使用默认/TPDC 数据。",
            data_source=(active if active in VALID_DATA_SOURCES else "none"),
            use_result_context=False,
            reason="mapping_or_optimization_requires_data",
            payload={"action": "need_mapping_data", "readiness": ready},
        )
    ds = ready.get("data_source") if ready.get("data_source") in VALID_DATA_SOURCES else active
    return SoilAgentDecision(
        mode="start_rfk",
        assistant_text="收到，我将基于当前上下文启动新一轮土壤有机质制图/精度优化流程。上一轮不确定性分析结果会保留为历史结果，但不会锁定本轮任务。",
        data_source=(ds if ds in VALID_DATA_SOURCES else "user"),
        use_result_context=False,
        reason="current_turn_mapping_or_accuracy_optimization_overrides_previous_gcp_context",
        payload={"action": "start_new_rfk_mapping_or_optimization", "request_text": str(user_text or ""), "previous_result_kind": ctx.get("latest_result_kind")},
    )


def _is_explicit_gcp_execution_request(text: str) -> bool:
    """Return True only for an executable GCP/AOA follow-up, not a concept question.

    This is not a replacement for the AI-first agent. It is a formal-delivery guard
    after the agent has received full session memory: when the user explicitly says
    “进一步进行不确定性分析” after SOM mapping, the system must create a real
    GCP + AOA task instead of leaving the decision as a narrative reply.
    """
    t = str(text or "").strip()
    if not t or not _has_any_text(t, GCP_TOPIC_WORDS):
        return False
    has_action = _has_any_text(t, GCP_EXECUTION_WORDS)
    has_question = _has_any_text(t, QUESTION_WORDS)
    if has_question and not has_action:
        return False
    # Short follow-up commands such as “不确定性分析” are executable when a mapping
    # result exists in memory; general explanatory questions are handled above.
    return has_action or len(t) <= 24


def _postprocess_contextual_decision(decision: SoilAgentDecision, user_text: str, session) -> SoilAgentDecision:
    """Validate an AI planner decision without keyword-escalating it.

    V209 rule: keywords are semantic evidence only.  This function must not turn
    a reply/question into a tool execution merely because words such as “制图”,
    “下载” or “不确定性分析” appear.  It only performs safety validation after the
    AI planner has already selected an executable mode.
    """
    try:
        ctx = _session_context(session)
    except Exception:
        ctx = {}

    # Normal reply/follow-up/show modes stay normal.  Do not promote them to a
    # task by regex or keyword matching.
    if decision.mode in {"reply", "followup", "show_uploaded_raster"}:
        # Do not put a router placeholder into the chat.  The caller will fill
        # blank/placeholder reply text with a real conversational LLM answer.
        decision.use_result_context = bool(decision.use_result_context)
        return decision

    # If AI decided to run GCP/AOA, block it unless the current session has a
    # completed mapping result.  This is a safety gate, not a keyword trigger.
    if decision.mode == "start_gcp":
        if not ctx.get("has_rfk_result"):
            return SoilAgentDecision(
                mode="followup",
                assistant_text="当前会话还没有可用于不确定性分析的已完成制图结果。请先完成本轮制图，再运行地理共形预测（GCP）+ AOA 不确定性与适用域分析。",
                data_source=(ctx.get("active_data_source") if ctx.get("active_data_source") in VALID_DATA_SOURCES else "none") or "none",
                use_result_context=False,
                reason="safety_gate_gcp_requires_completed_current_mapping",
                payload={"action": "need_completed_mapping_before_gcp"},
            )
        decision.use_result_context = True
        decision.payload = {**(decision.payload or {}), "method": "GCP_AOA_formal_uncertainty_analysis"}
        if not decision.assistant_text.strip():
            decision.assistant_text = "收到，我将基于当前已完成的制图结果启动 GCP + AOA 不确定性与适用域分析。"
        return decision

    # If AI decided to run a mapping/download/preprocess task, only normalize
    # missing fields.  Do not infer a task from keywords here.
    if decision.mode in {"start_rfk", "mapping_source_choice"}:
        decision.use_result_context = False
        decision.payload = {**(decision.payload or {}), "action": decision.payload.get("action") or "ai_planner_start_mapping"}
        if decision.data_source not in VALID_DATA_SOURCES:
            decision.data_source = "user" if ctx.get("has_uploaded_files") else "default"
        if not decision.assistant_text.strip():
            decision.assistant_text = "收到，我将按当前会话数据和你的执行意图进入制图任务。"
        return decision

    if decision.mode in {"start_download_tpdc", "start_download_gee", "tpdc_search", "download_source_choice"}:
        decision.use_result_context = False
        if decision.mode in {"start_download_tpdc", "tpdc_search"}:
            decision.data_source = "pro_domestic"
        if not decision.assistant_text.strip():
            decision.assistant_text = "收到，我将按当前数据获取意图准备下载流程。"
        return decision

    if decision.mode == "standardize_spatial":
        decision.use_result_context = False
        if not decision.assistant_text.strip():
            decision.assistant_text = "收到，我将按你的要求准备空间标准化处理。"
        return decision

    return decision


SYSTEM_PROMPT = """
你是数字土壤制图智能体的正式交互主控，面向土壤有机质制图、数据下载、数据预处理、模型搜索、专题制图和不确定性分析。

你的首要职责是保持上下文连续：必须结合当前会话记忆、上传数据、上一轮制图结果、上一轮不确定性结果、地图布局配置和最近对话来理解用户意图。用户说“继续”“用刚才的数据”“这个结果”“把图例放右下角”“做不确定性分析”时，应优先从后端会话上下文中解析指代对象。

必须遵守：
1. 你是唯一的意图主控，不得把关键词当作任务触发器。关键词只能作为语义线索，不能直接启动任务。
2. 普通知识、方法解释、模型解释、结果解读、数据准备咨询、可行性咨询、协变量准备问题，返回 mode=reply。
3. 只有你综合整句语义、上下文、当前任务状态并判断用户明确要求执行任务时，才返回 start_rfk、start_gcp、start_download_tpdc、standardize_spatial、tpdc_search 等动作。
4. 用户消息中如果包含本机路径、文件名、文件夹路径或上传说明，这些内容应视为“用户把数据交给智能体”的上下文材料；不能把路径导入本身当成整条消息的最终意图。必须继续理解路径之后/路径之外的真实指令：若用户说“现在根据我的数据进行/绘制/生成土壤有机质图”，应返回 start_rfk；若用户说“分析这些数据能不能用于制图”，应返回 reply 并进行数据可用性说明；若用户只是问“需要准备哪些协变量”，应返回 reply。
5. 用户说“我要进行某制图，我需要准备哪些数据/协变量/条件？”属于咨询，不属于执行制图；用户说“分析这组数据能不能用”属于数据审查，不属于自动制图。
3. 用户没有指定模型时，进入自动模型搜索和当前数据最优模型选择；RFRK/制图模型 是候选和对照模型，不是唯一固定模型。
4. 用户明确要求 RFRK/制图模型 时，按 RFRK/制图模型 运行；用户要求精度优先时，允许 ExtraTrees、XGBoost、LightGBM、Stacking 等模型参与选择。
5. 正式制图默认输出 GeoTIFF 和带标题、图例、比例尺、指北针、边框的 PNG；用户指定部件时按用户指定输出。
6. GeoTIFF 不包含制图部件；PNG 和交互地图包含制图部件。
7. 用户要求修改地图部件位置或样式时，结合当前布局配置解释并返回可执行意图；不要要求用户重新上传数据。
8. 用户要求数据下载时，默认进入 TPDC 下载流程，结果保存到正式下载目录。
9. 用户要求不确定性分析/GCP/AOA/适用域/外推风险时，必须基于当前已完成制图结果返回 mode=start_gcp；如果没有制图结果，说明需要先完成制图。
10. NC/NetCDF、瓦片数据、全国数据裁剪、CRS 统一、分辨率统一、栅格对齐都属于正式数据处理流程。
11. 全部回答、日志和报告使用正式交付表述。
12. 最终必须返回 SoilAgentDecision 结构。
13. 当 mode=reply 或 followup 时，assistant_text 必须是可以直接展示给用户的完整自然语言回答。不得只返回空字符串、不得只写“已收到/正在分析/请继续说明”，也不得把 reason 当作回答。
14. 普通问候、闲聊、与制图无关的问题，都应像正常大语言模型一样直接回答，同时保持必要的专业边界。

禁止模式：不得仅因“制图/下载/不确定性/协变量/数据”等词出现就执行工具；不得把问句、咨询、解释、准备建议、可行性分析误判为执行任务。
回答风格：中文、正式、清楚、可执行；不要像关键词机器人，要给出真正的用户可读回复。
"""


def _recent_history_text(session, limit: int = 10) -> str:
    recent_history_fn = getattr(session, "recent_history", None)
    try:
        recent_history = recent_history_fn(limit) if callable(recent_history_fn) else []
    except Exception:
        recent_history = []
    lines = []
    for item in recent_history[-limit:]:
        role = item.get("role", "")
        content = str(item.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _session_context(session) -> Dict[str, Any]:
    role_report = getattr(session, "data_role_report", {}) or {}
    cov_plan = getattr(session, "covariate_readiness_plan", {}) or {}
    preprocess = getattr(session, "preprocess_report", {}) or {}
    gate = preprocess.get("gate") or {}
    return {
        "session_id": getattr(session, "session_id", None),
        "has_uploaded_files": bool(getattr(session, "latest_uploaded_files", [])),
        "uploaded_file_count": len(getattr(session, "latest_uploaded_files", []) or []),
        "has_training_sample": bool(gate.get("has_training_sample")),
        "ready_covariate_count": int(gate.get("ready_covariate_count") or 0),
        "ready_covariates": preprocess.get("ready_covariates") or cov_plan.get("ready") or [],
        "missing_covariates": preprocess.get("missing_default_covariates") or cov_plan.get("missing") or [],
        "preprocess_status": preprocess.get("status"),
        "preprocess_gate": gate,
        "upload_summary": role_report.get("summary") or {},
        "active_data_source": getattr(session, "active_data_source", "default"),
        "has_rfk_result": bool(getattr(session, "has_rfk_result", False)),
        "has_gcp_result": bool(getattr(session, "has_gcp_result", False)),
        "latest_result_kind": getattr(session, "latest_result_kind", None),
        "last_task_type": getattr(session, "last_task_type", None),
        "recent_history_text": _recent_history_text(session, 20),
        "agent_memory": build_agent_memory_payload(session, history_limit=20),
    }


def _result_context(session) -> str:
    try:
        return build_result_context_text(
            rfk_paths=getattr(session, "rfk_result_paths", {}) or {},
            gcp_paths=getattr(session, "gcp_result_paths", {}) or {},
        )
    except Exception:
        return ""


def _normalise_data_source(value: str | None, session) -> str:
    v = str(value or "").strip().lower()
    aliases = {
        "": "none",
        "none": "none",
        "default": "default",
        "local": "default",
        "builtin": "default",
        "built-in": "default",
        "内置": "default",
        "默认": "default",
        "user": "user",
        "uploaded": "user",
        "上传": "user",
        "pro_user": "pro_user",
        "tpdc": "pro_domestic",
        "domestic": "pro_domestic",
        "pro_domestic": "pro_domestic",
        "国家青藏高原科学数据中心": "pro_domestic",
        "青藏高原科学数据中心": "pro_domestic",
        "web": "pro_web",
        "pro_web": "pro_web",
    }
    if v in aliases:
        return aliases[v]
    active = str(getattr(session, "active_data_source", "default") or "default")
    return active if active in VALID_DATA_SOURCES else "default"


def _readiness_for_mapping(session, data_source: str) -> Dict[str, Any]:
    ctx = _session_context(session)
    ds = _normalise_data_source(data_source, session)
    has_uploaded = bool(ctx["has_uploaded_files"])
    has_sample = bool(ctx["has_training_sample"])
    ready_cov = int(ctx["ready_covariate_count"] or 0)

    if ds in {"none", "default"}:
        return {
            "ok": True,
            "data_source": "default",
            "message": "可以使用电脑内置默认数据进行基线制图。",
            "context": ctx,
        }
    if ds in {"pro_domestic", "pro_web"}:
        ok = (not has_uploaded) or has_sample
        return {
            "ok": ok,
            "data_source": ds,
            "message": "可以使用在线平台补齐环境协变量。" if ok else "已上传文件，但尚未识别到包含经纬度和 SOM/有机质字段的训练样点。",
            "context": ctx,
        }
    if ds in {"user", "pro_user"}:
        ok = has_uploaded and has_sample and ready_cov > 0
        return {
            "ok": ok,
            "data_source": ds,
            "message": "上传样点和上传协变量已满足制图入口。" if ok else "只使用上传数据时，需要训练样点和至少一个可入模环境协变量。",
            "context": ctx,
        }
    return {"ok": False, "data_source": ds, "message": f"未知数据源：{ds}", "context": ctx}


def _decision_to_outcome(d: SoilAgentDecision) -> Dict[str, Any]:
    planner = {
        "agent_type": "ai_first_langchain_agent",
        "reason": d.reason,
        "use_result_context": d.use_result_context,
        "payload": d.payload,
        "data_source": d.data_source,
    }
    return AgentOutcome(
        mode=d.mode,
        assistant_text=d.assistant_text,
        data_source=d.data_source if d.data_source in VALID_DATA_SOURCES else "none",
        planner=planner,
    ).to_dict()


def _is_placeholder_reply(text: str | None) -> bool:
    t = str(text or "").strip()
    if not t:
        return True
    placeholders = [
        "我已收到你的问题",
        "请继续说明你的目标",
        "当前可以围绕数据准备",
        "我理解了，但没有形成可执行动作",
        "已收到，正在结合当前任务上下文分析",
    ]
    return any(p in t for p in placeholders) or len(t) <= 4


def _ensure_conversational_reply(decision: SoilAgentDecision, user_text: str, session, agent_version: str, auth_user: dict[str, Any] | None = None) -> SoilAgentDecision:
    """Make reply/followup modes behave like a real LLM answer, not a router log.

    The structured planner decides whether a tool should run.  If it decides
    not to run a tool, the user still deserves an actual assistant response.
    Therefore an empty/placeholder assistant_text is filled by the conversational
    LLM path.  This does not trigger any task and does not use keyword routing.
    """
    if decision.mode not in {"reply", "followup"}:
        return decision
    if not _is_placeholder_reply(decision.assistant_text):
        return decision
    ctx = _session_context(session)
    result_ctx = _result_context(session)
    try:
        reply = compose_agent_reply(
            user_text=user_text,
            extra_context=(
                "当前智能体会话上下文，仅用于理解用户问题，不得据此自动启动任务："
                + json.dumps({
                    "session_context": ctx,
                    "agent_memory": build_agent_memory_payload(session, history_limit=24),
                    "agent_version": agent_version,
                    "user_id": (auth_user or {}).get("id"),
                    "planner_reason": decision.reason,
                    "planner_payload": decision.payload,
                }, ensure_ascii=False)[:6000]
            ),
            result_context=result_ctx,
            chat_history=getattr(session, "recent_history", lambda n=12: [])(12),
        )
    except Exception:
        reply = ""
    if reply and str(reply).strip():
        decision.assistant_text = str(reply).strip()
    else:
        # Last-resort for non-task mode: still provide a direct natural reply,
        # not a task status.  This branch should be rare and makes failures visible.
        decision.assistant_text = "你好，我在。你可以直接问我数据准备、制图流程、结果解释，也可以发起具体任务；我会先理解你的完整意图，再决定是否需要执行工具。"
    if decision.reason:
        decision.reason += "; conversational_reply_generated"
    else:
        decision.reason = "conversational_reply_generated"
    return decision


def _build_tools(session):
    if tool is None:
        return []

    @tool
    def inspect_current_session() -> dict:
        """Inspect uploaded files, preprocessing state, active data source, mapping results, and recent conversation."""
        return _session_context(session)

    @tool
    def check_mapping_readiness(data_source: str = "default") -> dict:
        """Check whether a selected data source can be used for soil organic matter mapping."""
        return _readiness_for_mapping(session, data_source)

    @tool
    def prepare_rfk_mapping(
        data_source: str = "default",
        region: str = "",
        year: str = "",
        covariates: str = "default",
        notes: str = "",
    ) -> dict:
        """Prepare to start 制图模型 soil organic matter mapping after the user explicitly asks to create/run a map."""
        ready = _readiness_for_mapping(session, data_source)
        if not ready.get("ok"):
            return {
                "mode": "followup",
                "assistant_text": ready.get("message"),
                "data_source": ready.get("data_source", "none"),
                "payload": {"region": region, "year": year, "covariates": covariates, "notes": notes},
            }
        return {
            "mode": "start_rfk",
            "assistant_text": "收到，我将启动土壤有机质制图任务。系统会自动进行数据检查、模型搜索和结果输出；右侧会显示实时进度，完成后可以继续进行结果解读或不确定性分析。",
            "data_source": ready.get("data_source", "default"),
            "payload": {"region": region, "year": year, "covariates": covariates, "notes": notes},
        }

    @tool
    def prepare_gcp_uncertainty(notes: str = "") -> dict:
        """Prepare to start GCP uncertainty analysis after the user explicitly requests uncertainty/reliability mapping."""
        ctx = _session_context(session)
        if not ctx.get("has_rfk_result"):
            return {
                "mode": "followup",
                "assistant_text": "当前会话还没有可用于不确定性分析的制图结果。请先完成一次土壤有机质制图，再运行 GCP + AOA 不确定性与适用域分析。",
                "data_source": ctx.get("active_data_source") or "default",
                "payload": {"notes": notes},
            }
        return {
            "mode": "start_gcp",
            "assistant_text": "收到，我将基于当前制图结果启动 GCP + AOA 不确定性与适用域分析。",
            "data_source": ctx.get("active_data_source") or "default",
            "payload": {"notes": notes},
        }

    @tool
    def prepare_public_data_download(data_source: str = "none", query: str = "", notes: str = "") -> dict:
        """Prepare a public data download action. data_source should be tpdc/pro_domestic; if unknown, ask UI to show source choice."""
        ds = _normalise_data_source(data_source, session)
        if ds == "pro_domestic":
            return {
                "mode": "start_download_tpdc",
                "assistant_text": "收到，我将启动国家青藏高原科学数据中心（TPDC）数据准备流程。",
                "data_source": "pro_domestic",
                "payload": {"query": query, "notes": notes},
            }
        return {
            "mode": "download_source_choice",
            "assistant_text": "请先选择本次数据下载来源：国家青藏高原科学数据中心 TPDC。",
            "data_source": "none",
            "payload": {"query": query, "notes": notes},
        }

    @tool
    def prepare_spatial_standardization(
        target_crs_text: str = "",
        target_resolution_m: float | None = None,
        reference_layer: str = "",
        notes: str = "",
    ) -> dict:
        """Prepare a raster spatial standardization task. Use this when the user asks to unify CRS, EPSG, resolution, resample, reproject, or align raster grids."""
        ctx = _session_context(session)
        if not ctx.get("has_uploaded_files"):
            return {
                "mode": "followup",
                "assistant_text": "当前会话还没有可处理的上传或导入栅格。请先上传/导入环境协变量，或告诉我本机数据路径。",
                "data_source": ctx.get("active_data_source") or "user",
                "payload": {"target_crs_text": target_crs_text, "target_resolution_m": target_resolution_m, "reference_layer": reference_layer, "notes": notes},
            }
        return {
            "mode": "standardize_spatial",
            "assistant_text": "收到，我将把当前环境协变量统一到指定坐标系、分辨率和像元网格。",
            "data_source": ctx.get("active_data_source") or "user",
            "payload": {
                "target_crs_text": target_crs_text,
                "target_resolution_m": target_resolution_m,
                "reference_layer": reference_layer,
                "notes": notes,
            },
        }

    @tool
    def prepare_tpdc_search(query: str = "", notes: str = "") -> dict:
        """Prepare to open TPDC, auto-fill login credentials from .env, wait for user captcha, and search the short data keyword only."""
        return {
            "mode": "tpdc_search",
            "assistant_text": "收到，我将打开国家青藏高原科学数据中心（TPDC），自动填写账号密码；验证码由你手动输入。登录完成后我会只用核心关键词检索数据。",
            "data_source": "pro_domestic",
            "payload": {"query": query, "notes": notes},
        }

    @tool
    def web_search_optional(query: str) -> list[str]:
        """Optionally search the web when the user explicitly wants current/external information. Returns empty if disabled/unavailable."""
        if os.getenv("ENABLE_WEB_SEARCH", "0") != "1":
            return []
        try:
            from services.web_research_service import search_web
            return search_web(query) or []
        except Exception:
            return []

    return [
        inspect_current_session,
        check_mapping_readiness,
        prepare_rfk_mapping,
        prepare_gcp_uncertainty,
        prepare_public_data_download,
        prepare_spatial_standardization,
        prepare_tpdc_search,
        web_search_optional,
    ]


def _extract_tool_decision(result: Any) -> SoilAgentDecision | None:
    """If the agent used a tool that returned a mode dict, convert it to SoilAgentDecision.

    This makes the UI robust even when the model returns a normal final sentence after
    calling a tool instead of filling structured_response perfectly.
    """
    try:
        messages = result.get("messages", []) if isinstance(result, dict) else []
    except Exception:
        messages = []
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if isinstance(content, dict) and content.get("mode"):
            try:
                return SoilAgentDecision.model_validate(content)
            except Exception:
                pass
        if isinstance(content, str) and '"mode"' in content:
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and parsed.get("mode"):
                    return SoilAgentDecision.model_validate(parsed)
            except Exception:
                continue
    return None


def _plain_structured_fallback(user_text: str, session, agent_version: str, auth_user: dict[str, Any] | None = None) -> SoilAgentDecision:
    """LLM-first fallback when LangChain Agent runtime is unavailable.

    It still asks the LLM to make a structured decision; it never uses keyword routing.
    """
    model = _get_llm()
    ctx = _session_context(session)
    result_ctx = _result_context(session)
    if model is None:
        return SoilAgentDecision(
            mode="reply",
            assistant_text=(
                "AI 交互主干还没有启动：未初始化大模型。请检查 .env 中 ENABLE_QWEN、"
                "DASHSCOPE_API_KEY、DASHSCOPE_BASE_URL、QWEN_TEXT_MODEL。当前不会回退到关键词路由。"
            ),
            data_source="none",
            reason="llm_not_available",
        )
    prompt = (
        SYSTEM_PROMPT
        + "\n\n后端会话上下文 JSON：\n"
        + json.dumps({
            "session_context": ctx,
            "agent_memory": build_agent_memory_payload(session, history_limit=24),
            "result_context_preview": result_ctx[:3000],
            "agent_version": agent_version,
            "user_id": (auth_user or {}).get("id"),
        }, ensure_ascii=False)
        + "\n\n用户当前输入：\n"
        + user_text
        + "\n\n请只输出符合 SoilAgentDecision 的 JSON。"
    )
    try:
        structured_model = model.with_structured_output(SoilAgentDecision)
        raw = structured_model.invoke(prompt)
        if isinstance(raw, SoilAgentDecision):
            return _ensure_conversational_reply(raw, user_text, session, agent_version, auth_user)
        if isinstance(raw, dict):
            return _ensure_conversational_reply(SoilAgentDecision.model_validate(raw), user_text, session, agent_version, auth_user)
    except Exception:
        pass

    # Last fallback: normal AI answer, no task execution.
    reply = compose_agent_reply(
        user_text=user_text,
        extra_context="当前后端会话上下文：" + json.dumps(ctx, ensure_ascii=False)[:3000],
        result_context=result_ctx,
        chat_history=getattr(session, "recent_history", lambda n=10: [])(10),
    )
    return SoilAgentDecision(
        mode="reply",
        assistant_text=reply or "你好，我在。你可以直接问我任何问题；如果涉及制图、下载或不确定性分析，我会先理解你的完整意图，再决定是否需要执行任务。",
        data_source="none",
        reason="plain_llm_fallback_conversational_reply",
    )


def _looks_like_information_or_question(text: str) -> bool:
    t = str(text or "")
    lower = t.lower()
    question_words = ["什么", "哪些", "为什么", "为何", "怎么", "如何", "吗", "么", "能不能", "需要", "准备", "?", "？"]
    info_words = ["协变量", "环境变量", "环境因子", "数据", "原理", "区别", "作用", "解释", "建议", "精度", "模型", "流程", "方法", "为什么"]
    execute_words = ["开始", "执行", "运行", "生成", "绘制", "出图", "制图", "下载", "处理", "统一", "重采样", "不确定性分析", "GCP", "AOA", "适用域", "重新制图", "提升精度", "提高精度", "优化精度", "调参", "重新训练", "重新建模"]
    # Explicit execution still goes through structured decision.
    if any(w in t for w in execute_words) and not any(w in t for w in ["需要哪些", "需要准备", "怎么", "如何", "为什么", "吗", "？", "?"]):
        return False
    return any(w in t for w in question_words) or any(w in t for w in info_words)


def _conversation_reply_decision(user_text: str, session, agent_version: str, auth_user: dict[str, Any] | None = None) -> SoilAgentDecision:
    """Use the LLM as a conversational assistant for normal Q&A.

    This restores the V176 AI-first conversational behavior: normal questions are
    answered directly with conversation memory, and only explicit task execution
    enters the structured tool-decision path.
    """
    ctx = _session_context(session)
    result_ctx = _result_context(session)
    reply = compose_agent_reply(
        user_text=user_text,
        extra_context=(
            "当前智能体任务上下文："
            + json.dumps({
                "session_context": ctx,
                "agent_memory": build_agent_memory_payload(session, history_limit=24),
                "agent_version": agent_version,
                "user_id": (auth_user or {}).get("id"),
            }, ensure_ascii=False)[:6000]
        ),
        result_context=result_ctx,
        chat_history=getattr(session, "recent_history", lambda n=12: [])(12),
    )
    if not reply:
        reply = "你好，我在。你可以直接问我任何问题；如果涉及制图、下载或不确定性分析，我会先理解你的完整意图，再决定是否需要执行任务。"
    return SoilAgentDecision(mode="reply", assistant_text=reply, data_source="none", reason="conversation_first")

def execute_ai_interactive_turn(
    user_text: str,
    session,
    agent_version: str = "standard",
    auth_user: dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """AI-first conversational entry point.

    V184 uses a conversation-first AI trunk for normal questions and a structured task decision path for explicit workflow execution. The LangChain tool agent remains available by setting ENABLE_LANGCHAIN_TOOL_AGENT=1.
    """
    print(f"[AI-FIRST] 收到用户消息，进入智能体上下文推理 | session={getattr(session, 'session_id', 'default')} | text={str(user_text)[:80]!r}")
    if os.getenv("ENABLE_AI_INTERACTIVE_AGENT", "1") != "1":
        decision = _plain_structured_fallback(user_text, session, agent_version, auth_user)
        decision = _postprocess_contextual_decision(decision, user_text, session)
        decision = _ensure_conversational_reply(decision, user_text, session, agent_version, auth_user)
        outcome = _decision_to_outcome(decision)
        print(f"[AI-FIRST] 结构化推理完成 | mode={outcome.get('mode')} | reason=ENABLE_AI_INTERACTIVE_AGENT_OFF")
        return outcome

    # V209: no keyword-direct routing.  Every message is planned by the AI
    # structured decision layer.  Regex/keyword functions may remain as safety
    # validators elsewhere, but they must not be used here to start a task.
    # Structured decision is used for all turns: normal Q&A returns mode=reply;
    # explicit task execution returns the relevant task mode.
    if os.getenv("ENABLE_LANGCHAIN_TOOL_AGENT", "0") != "1":
        decision = _plain_structured_fallback(user_text, session, agent_version, auth_user)
        decision = _postprocess_contextual_decision(decision, user_text, session)
        decision = _ensure_conversational_reply(decision, user_text, session, agent_version, auth_user)
        outcome = _decision_to_outcome(decision)
        remember_user_turn(session, user_text, outcome)
        print(f"[AI-FIRST] 任务决策完成 | mode={outcome.get('mode')} | data_source={outcome.get('data_source')} | reason={outcome.get('planner', {}).get('reason')}")
        return outcome

    model = _get_llm()
    if model is None:
        decision = _plain_structured_fallback(user_text, session, agent_version, auth_user)
        decision = _postprocess_contextual_decision(decision, user_text, session)
        decision = _ensure_conversational_reply(decision, user_text, session, agent_version, auth_user)
        outcome = _decision_to_outcome(decision)
        print(f"[AI-FIRST] LLM不可用，已使用安全回退 | mode={outcome.get('mode')}")
        return outcome

    if create_agent is None:
        decision = _plain_structured_fallback(user_text, session, agent_version, auth_user)
        decision = _postprocess_contextual_decision(decision, user_text, session)
        decision = _ensure_conversational_reply(decision, user_text, session, agent_version, auth_user)
        outcome = _decision_to_outcome(decision)
        print(f"[AI-FIRST] LangChain Agent不可用，已使用结构化推理 | mode={outcome.get('mode')}")
        return outcome

    ctx = _session_context(session)
    result_ctx = _result_context(session)
    try:
        print("[AI-FIRST] 启用 LangChain 工具智能体运行。")
        agent = create_agent(
            model=model,
            tools=_build_tools(session),
            system_prompt=SYSTEM_PROMPT,
            response_format=SoilAgentDecision,
            checkpointer=_CHECKPOINTER,
        )
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "后端会话上下文 JSON：\n" + json.dumps({
                            "session_context": ctx,
                            "agent_memory": build_agent_memory_payload(session, history_limit=24),
                            "agent_memory_text": build_agent_memory_text(session, history_limit=24)[:6000],
                            "result_context_preview": result_ctx[:4000],
                            "agent_version": agent_version,
                            "user_id": (auth_user or {}).get("id"),
                        }, ensure_ascii=False),
                    },
                    {"role": "user", "content": user_text},
                ]
            },
            config={"configurable": {"thread_id": getattr(session, "session_id", "default")}},
        )

        structured = result.get("structured_response") if isinstance(result, dict) else None
        if isinstance(structured, SoilAgentDecision):
            decision = structured
        elif isinstance(structured, dict):
            decision = SoilAgentDecision.model_validate(structured)
        else:
            decision = _extract_tool_decision(result)
            if decision is None:
                messages = result.get("messages", []) if isinstance(result, dict) else []
                content = getattr(messages[-1], "content", "") if messages else ""
                decision = SoilAgentDecision(
                    mode="reply",
                    assistant_text=str(content or "我理解了，但没有形成可执行动作。"),
                    data_source="none",
                    reason="no_structured_response",
                )
        decision = _postprocess_contextual_decision(decision, user_text, session)
        decision = _ensure_conversational_reply(decision, user_text, session, agent_version, auth_user)
        outcome = _decision_to_outcome(decision)
        remember_user_turn(session, user_text, outcome)
        return outcome
    except Exception as exc:
        # Still keep AI-first behavior. Do not fall back to legacy keyword routing unless explicitly configured there.
        decision = _plain_structured_fallback(user_text, session, agent_version, auth_user)
        decision = _postprocess_contextual_decision(decision, user_text, session)
        if decision.reason:
            decision.reason += f"; create_agent_error={exc!r}"
        else:
            decision.reason = f"create_agent_error={exc!r}"
        outcome = _decision_to_outcome(decision)
        remember_user_turn(session, user_text, outcome)
        return outcome
