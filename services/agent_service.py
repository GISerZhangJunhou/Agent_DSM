from __future__ import annotations

import os

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from core.query_normalizer import is_probably_soil_mapping_query, normalize_query
from services.knowledge_service import _fallback_answer  # type: ignore
from services.llm_service import compose_agent_reply, plan_agent_action
from services.result_context_service import build_result_context_text
from services.agent_memory_service import build_agent_memory_payload, remember_user_turn
from services.web_research_service import search_web
from services.vip_service import build_standard_limit_message, is_pro_user, needs_pro_auto_data, normalize_agent_version

try:
    from services.aoi_preflight_service import detect_requested_aoi_from_local_admin
except Exception:  # geopandas or local admin files may be unavailable during lightweight routing
    detect_requested_aoi_from_local_admin = None  # type: ignore


QUESTION_HINTS = ["什么", "哪些", "为何", "为什么", "吗", "么", "如何", "怎么", "咋", "能否", "可否", "?", "？"]
SEARCH_HINTS = ["查", "搜索", "搜一下", "帮我搜", "资料", "文献", "论文", "参考", "最新", "联网", "research"]
MAP_ACTIONS = [
    "绘制", "制图", "出图", "生成图", "生成一张图", "画图", "画一张图", "预测图", "分布图",
    "开始制图", "开始计算", "运行制图", "重新计算", "重新制图", "帮我做图", "给我做图", "跑一下制图",
    "制作", "制作图", "为我制作", "帮我制作", "给我制作", "做一张图", "做图", "为我做", "给我来一张图"
]
GCP_ACTIONS = [
    "不确定性分析", "做不确定性", "生成不确定性图", "给我不确定性图",
    "适用域分析", "AOA", "aoa", "GCP", "gcp", "GCP+AOA", "gcp+aoa",
    "跑一下gcp", "运行gcp", "计算gcp", "重新做gcp",
    "重新做不确定性", "重新计算不确定性", "给我可靠性分析",
    "外推风险", "高风险区域", "虚假自信", "预测区间"
]
MAP_TOPICS = ["土壤有机质", "有机质", "土壤有机碳", "som", "soc", "成都市", "成都"]
RESULT_REF_WORDS = [
    "这个结果", "本次结果", "这次结果", "当前结果", "刚才的结果", "这个图", "这张图", "本次制图", "这次制图", "刚刚的图"
]
ANALYSIS_WORDS = [
    "分析结果", "为我分析结果", "帮我分析结果", "解释结果", "分析这张图", "怎么看这个结果", "如何解读结果",
    "解读这张图", "结合结果分析", "根据结果分析", "分析这次结果", "为我分析", "分析一下结果", "帮我看看结果", "评价结果"
]
RELIABILITY_WORDS = ["可靠吗", "可靠性", "不确定性", "稳不稳", "风险", "可信", "靠谱吗", "置信区间", "预测区间"]
SHOW_RASTER_WORDS = ["显示这个tif", "看看我上传的栅格", "打开我这个图", "显示我上传的图", "看这个栅格", "预览我上传的", "显示我上传的栅格"]
USER_DATA_WORDS = ["用我的数据", "用我上传的", "按我这个", "使用我的数据", "用这个csv", "用这个tif", "按我上传的数据"]
COVARIATE_CHOICE_WORDS = [
    "默认协变量", "默认环境协变量", "默认环境变量", "图中的环境协变量", "内置环境协变量",
    "全部协变量", "所有协变量", "全部环境协变量", "所有环境协变量",
    "协变量", "环境变量", "环境因子", "变量选择", "特征选择",
    "DEM", "dem", "pH", "ph", "BD", "CEC", "clay", "sand", "silt", "gravel", "porosity",
    "LULC", "CLCD", "NDVI", "EVI", "NPP", "ChinaCP", "Management", "BareSoil", "VegPhenology",
    "高程", "土壤容重", "阳离子交换量", "黏粒", "粘粒", "砂粒", "粉粒", "砾石", "孔隙度", "土地利用", "土地覆盖", "裸土", "作物", "气候", "水热", "植被物候",
]


def _has_covariate_choice(text: str) -> bool:
    text = text or ""
    return any(w in text for w in COVARIATE_CHOICE_WORDS)


def _covariate_choice_prompt(text: str) -> str:
    return (
        "开始制图前，需要先确认环境协变量。你可以直接选择：\n\n"
        "1）使用默认环境协变量：BD、CEC、DEM、gravel、pH、porosity、sand、silt、clay、CLCD、BareSoil、ChinaCP、Management、SeasonalClimateWater、VegPhenology（V161起默认不用LULCcd）。\n"
        "2）自定义协变量，例如：只用 DEM、pH、sand、clay。\n"
        "3）使用全部协变量。\n\n"
        "请把区域、年份和协变量一起说清楚，例如：使用默认环境协变量，绘制 2021 年温江区土壤有机质图。"
    )


@dataclass
class AgentPlan:
    action: str = "chat"
    use_web_search: bool = False
    use_result_context: bool = False
    explicit_compute: bool = False
    reply_focus: str = "normal"
    data_source: str = "default"
    reason: str = "默认自由聊天"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentOutcome:
    mode: str = "reply"  # reply | start_rfk | start_gcp | show_uploaded_raster | followup
    assistant_text: str = ""
    data_source: str = "default"
    planner: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


VALID_ACTIONS = {"chat", "clarify", "run_rfk", "run_gcp", "show_uploaded_raster"}


def _contains_any(text: str, words: List[str]) -> bool:
    return any(w in text for w in words)


def _has_admin_region_mention(text: str) -> bool:
    """Return True when user text contains a province/city/county name from local admin data.

    This lets follow-up commands such as “那为我绘制邛崃市的” remain executable even
    when the user omits “土壤有机质图” because it is already the current task topic.
    """
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
    # Fallback for common Chinese admin suffixes. Use at least 2 Chinese chars before suffix.
    return bool(__import__("re").search(r"[\u4e00-\u9fff]{2,}(省|市|区|县|旗|州|盟)", text))


def _history_has_mapping_topic(session_context: Dict[str, Any] | None = None) -> bool:
    ctx = session_context or {}
    if ctx.get("has_uploaded_files") or ctx.get("last_task_type") in {"rfk_mapping", "preflight_blocked"}:
        return True
    recent = str(ctx.get("recent_history_text") or "")
    return _contains_any(recent.lower(), [w.lower() for w in MAP_TOPICS + ["制图", "绘制", "土壤", "rfk"]])


def _has_question_intent(text: str) -> bool:
    return _contains_any(text, QUESTION_HINTS)


def _is_preparation_question(text: str) -> bool:
    t = str(text or "")
    return bool(__import__("re").search(
        r"(需要|应该|要|准备|提供|收集).{0,12}(哪些|什么|哪类|多少|环境协变量|环境变量|数据|资料)|"
        r"(哪些|什么|哪类).{0,12}(环境协变量|环境变量|数据|资料).{0,12}(需要|准备|提供|收集)|"
        r"(需要准备哪些|需要哪些数据|准备什么数据|要准备什么|需要什么环境)",
        t,
    ))


def _is_explicit_map_request(text: str, session_context: Dict[str, Any] | None = None) -> bool:
    t = str(text or "")
    tl = t.lower()
    # Questions about required data/covariates are advisory Q&A, not execution commands.
    if _is_preparation_question(t) and not __import__("re").search(r"(现在|直接|立即|马上|开始).{0,6}(制图|建模|预测|绘制|运行)", t):
        return False
    if not _contains_any(t, MAP_ACTIONS):
        return False
    # Full command: “绘制2021年温江区土壤有机质图”.
    if _contains_any(tl, [w.lower() for w in MAP_TOPICS]):
        return True
    # Contextual command: “那为我绘制邛崃市的”. The AOI name + current mapping context is enough.
    if _has_admin_region_mention(t) and _history_has_mapping_topic(session_context):
        return True
    return False


def _is_explicit_gcp_request(text: str) -> bool:
    return _contains_any(text, GCP_ACTIONS) or (
        _contains_any(text, RELIABILITY_WORDS) and _contains_any(text, ["生成", "做", "跑", "计算", "分析", "图"])
    )


def _wants_uploaded_raster_preview(text: str) -> bool:
    return _contains_any(text.lower(), [w.lower() for w in SHOW_RASTER_WORDS])


def _needs_web_by_heuristic(text: str) -> bool:
    if _contains_any(text, SEARCH_HINTS):
        return True
    if is_probably_soil_mapping_query(text) and (
        "文献" in text or "论文" in text or "一般" in text or "业内" in text or "常用" in text or "推荐" in text
    ):
        return True
    return False


def _needs_result_context_by_heuristic(text: str, has_rfk_result: bool, has_gcp_result: bool) -> bool:
    if not (has_rfk_result or has_gcp_result):
        return False
    if _contains_any(text, RESULT_REF_WORDS + ANALYSIS_WORDS + RELIABILITY_WORDS):
        return True
    if _contains_any(text, ["精度", "rmse", "r2", "mae", "高值区", "低值区", "空间分布"]):
        return True
    if _contains_any(text, ["这次", "当前", "刚刚", "本次"]) and _has_question_intent(text):
        return True
    return False


def _fallback_plan(user_text: str, session_context: Dict[str, Any] | None = None) -> AgentPlan:
    text = normalize_query(user_text)
    session_context = session_context or {}
    has_rfk_result = bool(session_context.get("has_rfk_result", False))
    has_gcp_result = bool(session_context.get("has_gcp_result", False))
    active_data_source = session_context.get("active_data_source", "default")
    data_source = "user" if _contains_any(text, USER_DATA_WORDS) else (active_data_source if active_data_source in {"default", "user"} else "default")

    if _is_explicit_map_request(text, session_context):
        return AgentPlan(action="run_rfk", explicit_compute=True, data_source=data_source, reason="用户明确要求执行制图")
    if _is_explicit_gcp_request(text):
        return AgentPlan(action="run_gcp", explicit_compute=True, use_result_context=True, data_source=data_source, reason="用户明确要求执行不确定性分析")
    if _wants_uploaded_raster_preview(text):
        return AgentPlan(action="show_uploaded_raster", reason="用户要求预览上传栅格")
    return AgentPlan(
        action="chat",
        use_web_search=_needs_web_by_heuristic(text),
        use_result_context=_needs_result_context_by_heuristic(text, has_rfk_result, has_gcp_result),
        explicit_compute=False,
        reply_focus="result_interpretation" if _needs_result_context_by_heuristic(text, has_rfk_result, has_gcp_result) else ("teaching" if is_probably_soil_mapping_query(text) else "normal"),
        data_source=data_source,
        reason="回退到启发式自由对话",
    )


def _validate_plan(raw: Dict[str, Any] | None, user_text: str, session_context: Dict[str, Any] | None = None) -> AgentPlan | None:
    if not isinstance(raw, dict):
        return None
    action = str(raw.get("action", "chat"))
    if action not in VALID_ACTIONS:
        return None
    plan = AgentPlan(
        action=action,
        use_web_search=bool(raw.get("use_web_search", False)),
        use_result_context=bool(raw.get("use_result_context", False)),
        explicit_compute=bool(raw.get("explicit_compute", False)),
        reply_focus=str(raw.get("reply_focus", "normal") or "normal"),
        data_source=str(raw.get("data_source", "default") or "default"),
        reason=str(raw.get("reason", "LLM agent planner") or "LLM agent planner"),
    )
    text = normalize_query(user_text)

    # 硬保护：没有明确执行意图时，禁止 LLM 擅自跑计算。
    if plan.action == "run_rfk" and not _is_explicit_map_request(text, session_context):
        return _fallback_plan(user_text, session_context)
    if plan.action == "run_gcp" and not _is_explicit_gcp_request(text):
        return _fallback_plan(user_text, session_context)
    return plan


def plan_agent_turn(user_text: str, session_context: Dict[str, Any] | None = None, result_context_text: str = "", chat_history=None) -> AgentPlan:
    llm_raw = plan_agent_action(user_text, session_context=session_context, result_context=result_context_text, chat_history=chat_history)
    plan = _validate_plan(llm_raw, user_text, session_context=session_context)
    if plan is not None:
        return plan
    return _fallback_plan(user_text, session_context=session_context)


def _build_runtime_context(session) -> tuple[Dict[str, Any], str]:
    role_report = getattr(session, "data_role_report", {}) or {}
    cov_plan = getattr(session, "covariate_readiness_plan", {}) or {}
    preprocess = getattr(session, "preprocess_report", {}) or {}
    session_context = {
        "has_rfk_result": bool(getattr(session, "has_rfk_result", False)),
        "has_gcp_result": bool(getattr(session, "has_gcp_result", False)),
        "active_data_source": getattr(session, "active_data_source", "default"),
        "last_task_type": getattr(session, "last_task_type", None),
        "latest_result_kind": getattr(session, "latest_result_kind", None),
        "has_uploaded_files": bool(getattr(session, "latest_uploaded_files", [])),
        "upload_summary": role_report.get("summary") or {},
        "ready_covariates": preprocess.get("ready_covariates") or cov_plan.get("ready") or [],
        "missing_covariates": preprocess.get("missing_default_covariates") or cov_plan.get("missing") or [],
        "preprocess_gate": preprocess.get("gate") or {},
        "preprocess_status": preprocess.get("status"),
        "recent_history_text": "\n".join(str(m.get("content", "")) for m in getattr(session, "recent_history", lambda n=20: [])(20)),
        "agent_memory": build_agent_memory_payload(session, history_limit=20),
    }
    result_context_text = build_result_context_text(
        rfk_paths=getattr(session, "rfk_result_paths", {}),
        gcp_paths=getattr(session, "gcp_result_paths", {}),
    )
    return session_context, result_context_text

def execute_agent_turn(user_text: str, session, agent_version: str = "standard", auth_user: dict[str, Any] | None = None) -> Dict[str, Any]:
    # AI-first trunk: every user message should enter the conversational Agent before
    # any legacy keyword planner. The legacy planner is kept only as an explicit
    # emergency fallback by setting ALLOW_LEGACY_AGENT_FALLBACK=1.
    if os.getenv("ENABLE_AI_INTERACTIVE_AGENT", "1") == "1":
        try:
            from services.ai_interactive_agent_service import execute_ai_interactive_turn
            return execute_ai_interactive_turn(user_text, session, agent_version=agent_version, auth_user=auth_user)
        except Exception as exc:
            print(f"[AI-FIRST] interactive agent failed: {exc!r}")
            if os.getenv("ALLOW_LEGACY_AGENT_FALLBACK", "0") != "1":
                return AgentOutcome(
                    mode="reply",
                    assistant_text=(
                        "智能体交互主干启动失败，系统未切换到旧版路由。"
                        "请检查 LangChain/LangGraph 依赖和大模型配置。"
                        f"\n错误：{exc}"
                    ),
                    data_source="none",
                    planner={"agent_type": "ai_first_error", "error": repr(exc)},
                ).to_dict()
            print("[AI-FIRST] ALLOW_LEGACY_AGENT_FALLBACK=1, fallback to legacy keyword planner.")

    text = normalize_query(user_text)
    session_context, result_context_text = _build_runtime_context(session)
    history_for_llm = (session.recent_history(10)[:-1] if getattr(session, "chat_history", None) else [])
    plan = plan_agent_turn(text, session_context=session_context, result_context_text=result_context_text, chat_history=history_for_llm)
    selected_version = normalize_agent_version(agent_version)
    dev_force_pro = os.getenv("PRO_DEV_FORCE_PRO_TEST", "0") == "1"
    effective_is_pro_user = is_pro_user(auth_user) or dev_force_pro

    if plan.action == "run_rfk":
        has_uploaded_files = bool(getattr(session, "latest_uploaded_files", []))
        needs_pro = needs_pro_auto_data(text, has_uploaded_files=has_uploaded_files, selected_version=selected_version)
        if needs_pro and not effective_is_pro_user:
            return AgentOutcome(
                mode="followup",
                assistant_text=build_standard_limit_message(text, selected_version),
                data_source=plan.data_source,
                planner={**plan.to_dict(), "blocked_by_plan": True, "required_plan": "pro"},
            ).to_dict()
        if needs_pro:
            plan.data_source = "pro_web"
            plan.reason = "Pro 版自动联网数据准备后执行制图"

    if plan.action == "run_gcp" and selected_version == "pro" and not effective_is_pro_user:
        return AgentOutcome(
            mode="followup",
            assistant_text=build_standard_limit_message(text, selected_version),
            data_source=plan.data_source,
            planner={**plan.to_dict(), "blocked_by_plan": True, "required_plan": "pro"},
        ).to_dict()

    if plan.action == "run_gcp" and not session_context.get("has_rfk_result"):
        return AgentOutcome(
            mode="followup",
            assistant_text="可以执行不确定性分析，但当前会话还没有可用的制图结果。请先完成一次土壤有机质制图，我再基于制图结果继续运行 GCP + AOA 分析。",
            data_source=plan.data_source,
            planner=plan.to_dict(),
        ).to_dict()

    if plan.action == "run_rfk":
        if os.getenv("PRO_COVARIATE_ASK_BEFORE_RUN", "1") == "1" and not _has_covariate_choice(text):
            return AgentOutcome(
                mode="followup",
                assistant_text=_covariate_choice_prompt(text),
                data_source=plan.data_source,
                planner={**plan.to_dict(), "blocked_by_covariate_choice": True},
            ).to_dict()
        start_text = "收到，我现在开始执行土壤有机质制图任务。系统会进行数据检查、模型搜索、结果输出和精度报告生成；右侧会显示实时进度。"
        if plan.data_source == "pro_web":
            start_text = "收到，已启用自动数据准备流程。我会先根据目标年份和地区整理可用协变量数据，然后执行土壤有机质制图任务。"
        return AgentOutcome(
            mode="start_rfk",
            assistant_text=start_text,
            data_source=plan.data_source,
            planner=plan.to_dict(),
        ).to_dict()

    if plan.action == "run_gcp":
        return AgentOutcome(
            mode="start_gcp",
            assistant_text="收到，我现在开始执行不确定性分析。右侧会显示实时进度，完成后我会结合区间和指标继续解释。",
            data_source=plan.data_source,
            planner=plan.to_dict(),
        ).to_dict()

    if plan.action == "show_uploaded_raster":
        return AgentOutcome(
            mode="show_uploaded_raster",
            assistant_text="我将显示你上传的栅格，右侧会给出预览。",
            data_source=plan.data_source,
            planner=plan.to_dict(),
        ).to_dict()

    if plan.action == "clarify":
        return AgentOutcome(
            mode="followup",
            assistant_text="你可以继续直接说你的目标。例如你是想让我解释当前结果、帮你查资料，还是现在就开始制图/做不确定性分析？",
            data_source=plan.data_source,
            planner=plan.to_dict(),
        ).to_dict()

    if (not session_context.get("has_rfk_result") and not session_context.get("has_gcp_result")) and _contains_any(text, RESULT_REF_WORDS + ANALYSIS_WORDS + RELIABILITY_WORDS):
        return AgentOutcome(
            mode="reply",
            assistant_text="如果你说的是当前这次制图结果，那我这边还没有可供解读的结果。你可以先让我开始制图；如果你是在问一般情况下怎么判断结果可不可靠，我也可以先直接给你讲判断思路。",
            data_source=plan.data_source,
            planner=plan.to_dict(),
        ).to_dict()

    web_snippets: List[str] = []
    if plan.use_web_search:
        web_snippets = search_web(text)

    reply_context = {
        "planner_reason": plan.reason,
        "reply_focus": plan.reply_focus,
        "active_data_source": plan.data_source,
        "session_context": session_context,
    }
    extra_context_parts: List[str] = [f"Agent上下文：{reply_context}"]
    if getattr(session, "ai_covariate_review", None):
        extra_context_parts.append("上传数据审查结果：" + str(getattr(session, "ai_covariate_review", ""))[:3000])
    if getattr(session, "preprocess_report", None):
        extra_context_parts.append("上传数据预处理状态：" + str((getattr(session, "preprocess_report", {}) or {}).get("gate", {})))
    if plan.use_result_context and result_context_text:
        extra_context_parts.append(result_context_text)
    extra_context = "\n\n".join(part for part in extra_context_parts if part)

    reply = compose_agent_reply(
        user_text=text,
        extra_context=extra_context,
        web_snippets=web_snippets,
        result_context=result_context_text if plan.use_result_context else "",
        chat_history=history_for_llm,
    )

    if not reply:
        if web_snippets:
            reply = "我查到了一些相关资料，先给你一个简要结论：\n" + "\n".join(f"- {s}" for s in web_snippets[:4])
        else:
            reply = _fallback_answer(text)

    return AgentOutcome(
        mode="reply",
        assistant_text=reply,
        data_source=plan.data_source,
        planner=plan.to_dict(),
    ).to_dict()
