from dataclasses import dataclass, asdict
from typing import Any, Dict

from core.query_normalizer import normalize_query, is_probably_soil_mapping_query
from services.llm_service import route_with_llm_json


@dataclass
class RouteDecision:
    task_type: str = "general_chat"
    need_compute: bool = False
    need_web: bool = False
    need_result_panel: bool = False
    need_progress: bool = False
    data_source: str = "none"
    followup_required: bool = False
    use_context_result: bool = False
    result_panel_mode: str = "empty"
    reason: str = "默认普通聊天"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


VALID_TASK_TYPES = {"general_chat", "web_knowledge", "rfk_mapping", "gcp_uncertainty"}
VALID_PANEL_MODES = {"empty", "rfk_progress", "rfk_result", "gcp_progress", "gcp_result"}


QUESTION_HINTS = ["什么", "哪些", "为何", "为什么", "吗", "么", "如何", "怎么", "咋", "能否", "可否", "?", "？"]
SEARCH_HINTS = ["查", "搜索", "搜一下", "帮我搜", "资料", "文献", "论文", "参考", "最新", "联网"]
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
MAP_TOPICS = ["土壤有机质", "有机质", "som", "成都市", "成都"]
RESULT_REF_WORDS = [
    "这个结果", "本次结果", "这次结果", "当前结果", "刚才的结果", "这个图", "这张图", "本次制图", "这次制图", "刚刚的图"
]
ANALYSIS_WORDS = [
    "分析结果", "为我分析结果", "帮我分析结果", "解释结果", "分析这张图", "怎么看这个结果", "如何解读结果",
    "解读这张图", "结合结果分析", "根据结果分析", "分析这次结果", "为我分析", "分析一下结果", "帮我看看结果", "评价结果"
]
KNOWLEDGE_WORDS = [
    "常用环境变量", "环境变量有哪些", "协变量", "环境因子", "影响因素", "如何提升", "怎么提升", "如何优化", "怎么优化",
    "改进建议", "如何提高", "提升精度", "提高精度", "别人怎么做", "一般怎么做", "原理", "区别", "作用", "含义"
]
RELIABILITY_WORDS = ["可靠吗", "可靠性", "不确定性", "稳不稳", "风险", "可信", "靠谱吗", "置信区间", "预测区间"]


def _contains_any(text: str, words: list[str]) -> bool:
    return any(w in text for w in words)


def _has_question_intent(text: str) -> bool:
    return _contains_any(text, QUESTION_HINTS)


def _is_explicit_map_request(text: str) -> bool:
    return _contains_any(text, MAP_ACTIONS) and _contains_any(text.lower(), [w.lower() for w in MAP_TOPICS])


def _is_explicit_gcp_request(text: str) -> bool:
    return _contains_any(text, GCP_ACTIONS) or (
        _contains_any(text, RELIABILITY_WORDS) and _contains_any(text, ["生成", "做", "跑", "计算", "分析", "图"])
    )


def _looks_like_information_request(text: str) -> bool:
    if _contains_any(text, SEARCH_HINTS + KNOWLEDGE_WORDS + ANALYSIS_WORDS):
        return True
    if _has_question_intent(text) and is_probably_soil_mapping_query(text):
        return True
    return False


def _base_decision(task_type: str, data_source: str = "none", use_context_result: bool = False, reason: str = "") -> RouteDecision:
    if task_type == "general_chat":
        return RouteDecision(task_type=task_type, reason=reason or "普通聊天或概念解释")
    if task_type == "web_knowledge":
        return RouteDecision(task_type=task_type, need_web=True, data_source="none", use_context_result=use_context_result, reason=reason or "需要联网知识问答")
    if task_type == "rfk_mapping":
        return RouteDecision(task_type=task_type, need_compute=True, need_result_panel=True, need_progress=True, data_source=data_source, result_panel_mode="rfk_progress", reason=reason or "需要重新执行制图")
    if task_type == "gcp_uncertainty":
        return RouteDecision(task_type=task_type, need_compute=True, need_result_panel=True, need_progress=True, data_source=data_source, use_context_result=True, result_panel_mode="gcp_progress", reason=reason or "需要重新执行GCP不确定性分析")
    return RouteDecision(reason="未知任务类型，已回退为普通聊天")


def _validate_decision(raw: dict | None) -> RouteDecision | None:
    if not isinstance(raw, dict):
        return None
    task_type = raw.get("task_type")
    mode = raw.get("result_panel_mode", "empty")
    if task_type not in VALID_TASK_TYPES or mode not in VALID_PANEL_MODES:
        return None
    try:
        return RouteDecision(
            task_type=task_type,
            need_compute=bool(raw.get("need_compute", False)),
            need_web=bool(raw.get("need_web", False)),
            need_result_panel=bool(raw.get("need_result_panel", False)),
            need_progress=bool(raw.get("need_progress", False)),
            data_source=raw.get("data_source", "none"),
            followup_required=bool(raw.get("followup_required", False)),
            use_context_result=bool(raw.get("use_context_result", False)),
            result_panel_mode=mode,
            reason=str(raw.get("reason", "LLM 路由判断")),
        )
    except Exception:
        return None


def route_user_request(user_text: str, context: dict | None = None, llm_enabled: bool = True) -> dict:
    text = normalize_query(user_text)
    lower = text.lower()
    context = context or {}
    has_rfk_result = bool(context.get("has_rfk_result", False))
    active_data_source = context.get("active_data_source", "default")
    recent_history = context.get("recent_history") or []

    explicit_user_data = ["用我的数据", "用我上传的", "按我这个", "使用我的数据", "用这个csv", "用这个tif", "按我上传的数据"]
    data_source = "user" if _contains_any(text, explicit_user_data) else (active_data_source if active_data_source in {"default", "user"} else "default")

    asks_about_current_result = _contains_any(text, RESULT_REF_WORDS) or (has_rfk_result and _contains_any(text, ["这次", "当前", "刚刚", "本次"]))

    # 1) 只有“明确执行命令”才触发计算，避免把普通问答误判成制图任务
    if _is_explicit_gcp_request(text):
        d = _base_decision("gcp_uncertainty", data_source=data_source, reason="用户明确要求执行不确定性分析")
        if not has_rfk_result:
            d.followup_required = True
            d.need_compute = False
            d.need_result_panel = False
            d.need_progress = False
            d.use_context_result = False
            d.result_panel_mode = "empty"
            d.reason = "当前还没有本次制图结果，需先完成一次制图"
        return d.to_dict()

    if _is_explicit_map_request(text):
        return _base_decision("rfk_mapping", data_source=data_source, reason="用户明确要求执行土壤有机质制图").to_dict()

    # 2) 对当前结果的追问，优先按“聊天/知识问答”处理，而不是自动重算
    if asks_about_current_result and (_contains_any(text, RELIABILITY_WORDS) or _contains_any(text, ANALYSIS_WORDS) or _has_question_intent(text)):
        return _base_decision("web_knowledge", use_context_result=True, reason="用户在追问当前结果的解释、可靠性或改进建议").to_dict()

    # 3) 搜资料 / 方法 / 变量 / 文献 / 提升建议
    if _looks_like_information_request(text):
        use_context = asks_about_current_result or ("结合这次结果" in text) or ("结合当前结果" in text) or ("精度" in text and has_rfk_result)
        return _base_decision("web_knowledge", use_context_result=use_context, reason="用户在自由提问、搜资料或咨询方法建议").to_dict()

    # 4) 让 LLM 辅助判断边界情况，但仍禁止它把普通问答升级成计算任务，除非文本里有明确执行意图
    if llm_enabled:
        llm_raw = route_with_llm_json(text, {
            "has_rfk_result": has_rfk_result,
            "has_gcp_result": bool(context.get("has_gcp_result", False)),
            "active_data_source": data_source,
            "last_task_type": context.get("last_task_type")
        }, chat_history=recent_history)
        llm_decision = _validate_decision(llm_raw)
        if llm_decision is not None:
            if llm_decision.task_type == "rfk_mapping" and not _is_explicit_map_request(text):
                llm_decision = _base_decision("web_knowledge" if is_probably_soil_mapping_query(text) else "general_chat", use_context_result=asks_about_current_result, reason="改为自由问答，避免把普通聊天误判成制图任务")
            if llm_decision.task_type == "gcp_uncertainty" and not _is_explicit_gcp_request(text):
                llm_decision = _base_decision("web_knowledge", use_context_result=asks_about_current_result or has_rfk_result, reason="改为结果问答，避免把普通聊天误判成不确定性计算")
            if isinstance(llm_decision, RouteDecision):
                if llm_decision.task_type == "gcp_uncertainty" and not has_rfk_result:
                    llm_decision.followup_required = True
                    llm_decision.need_compute = False
                    llm_decision.need_result_panel = False
                    llm_decision.need_progress = False
                    llm_decision.use_context_result = False
                    llm_decision.result_panel_mode = "empty"
                    llm_decision.reason = "当前还没有本次制图结果，需先完成一次制图"
                if llm_decision.task_type in {"rfk_mapping", "gcp_uncertainty"}:
                    llm_decision.data_source = data_source
                return llm_decision.to_dict()

    return _base_decision("general_chat", reason="按自由聊天处理").to_dict()
