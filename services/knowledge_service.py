from services.llm_service import answer_knowledge_question, summarize_web_research
from services.web_research_service import search_web
from core.query_normalizer import normalize_query, is_probably_soil_mapping_query

LOCAL_KNOWLEDGE = {
    "dsm_basic": "数字土壤制图（DSM）是利用土壤样点、环境变量和空间模型来预测土壤属性空间分布的一类方法。常见环境变量包括地形、气候、遥感、土地利用、母质和土壤质地等。",
    "som_role": "土壤有机质会影响土壤肥力、团聚体结构、保水保肥能力和作物生长环境，是评价耕地质量的重要指标之一。",
    "gcp_basic": "GCP 指地理共形预测（Geographic Conformal Prediction），用于基于校准/验证残差构建预测区间，表达预测结果的不确定性。它不是广义克里金精度，也不是克里金模型。",
    "improve_accuracy": "提升制图精度通常可从样点质量、环境变量质量、变量筛选、空间验证方式和高不确定性区域补样这几方面入手。",
    "soil_dsm_variables": "数字土壤制图里常见环境变量通常包括地形变量、气候变量、遥感光谱与植被指数、土壤质地与理化性质、土地利用/覆盖、母质与地质背景，以及反映水热条件和管理活动的变量。"
}


def _fallback_answer(text: str) -> str:
    lower = text.lower().strip()
    if lower in {"?", "？", "啥", "什么", "什么意思"}:
        return "你可以直接告诉我具体需求，例如：为我绘制成都市土壤有机质图，或者问我这个结果可靠吗、怎么提升精度、这张图该怎么解读。"
    if ("土壤有机质" in text or "有机质" in text) and ("作用" in text or "意义" in text or "用途" in text):
        return LOCAL_KNOWLEDGE["som_role"]
    if "gcp" in lower or "不确定性" in text:
        return LOCAL_KNOWLEDGE["gcp_basic"]
    if is_probably_soil_mapping_query(text) and ("环境变量" in text or "协变量" in text or "环境因子" in text):
        return LOCAL_KNOWLEDGE["soil_dsm_variables"]
    if "数字土壤制图" in text or "dsm" in lower:
        return LOCAL_KNOWLEDGE["dsm_basic"]
    return "我需要更具体一点的问题。你可以直接问我：为我绘图、这个结果可靠吗、结合本次结果分析、数字土壤制图常用环境变量有哪些。"


def answer_general_chat(user_text: str, extra_context: str = "", chat_history=None) -> str:
    text = normalize_query(user_text)
    llm_answer = answer_knowledge_question(text, extra_context=extra_context, chat_history=chat_history)
    return llm_answer or _fallback_answer(text)


def answer_web_knowledge(user_text: str, extra_context: str = "", chat_history=None) -> str:
    text = normalize_query(user_text)
    snippets = search_web(text)
    if snippets:
        llm_summary = summarize_web_research(text, snippets, extra_context=extra_context, chat_history=chat_history)
        if llm_summary:
            return llm_summary
        joined = "\n".join(f"- {s}" for s in snippets[:5])
        return f"我查到了一些可参考的资料，整理如下：\n{joined}"

    llm_answer = answer_knowledge_question(text, extra_context=extra_context, chat_history=chat_history)
    if llm_answer:
        return llm_answer

    if is_probably_soil_mapping_query(text) and ("环境变量" in text or "协变量" in text or "环境因子" in text):
        return LOCAL_KNOWLEDGE["soil_dsm_variables"]
    return LOCAL_KNOWLEDGE["improve_accuracy"]
