from core.query_normalizer import normalize_query
from services.llm_service import classify_intent_with_llm


def _contains_any(text: str, words: list[str]) -> bool:
    return any(w in text for w in words)


def route_intent(user_text: str, llm_enabled: bool = True) -> str:
    text = normalize_query(user_text)
    lower = text.lower()

    map_words = ["绘制", "生成图", "制图", "出图", "分布图", "预测图", "给我一张图", "画一张图", "绘图", "显示图"]
    map_topics = ["土壤有机质", "有机质", "som", "成都市", "成都"]
    reliability_words = ["可靠", "不确定性", "可信", "区间", "稳定", "风险", "稳不稳"]
    model_words = ["什么模型", "用了什么模型", "为何用", "为什么选", "rfk", "gcp"]
    metrics_words = ["r2", "rmse", "mae", "精度多少", "精度怎么样", "指标", "精度指标"]
    improve_words = ["如何提升", "怎么提升", "如何优化", "怎么优化", "改进建议", "怎么改进", "如何提高", "提升精度", "提高精度", "别人怎么做"]
    dsm_words = ["dsm", "数字土壤制图", "协变量", "环境变量", "空间验证", "采样策略", "kriging", "随机森林"]
    som_words = ["土壤有机质作用", "有机质作用", "土壤有机质有什么作用", "有机质有什么用", "土壤有机质的意义"]
    user_data_words = ["用我的数据", "用我上传的", "按我这个", "不要默认数据", "使用我的数据", "用这个csv", "用这个tif"]
    show_raster_words = ["显示这个tif", "看看我上传的栅格", "打开我这个图", "显示我上传的图", "看这个栅格"]

    if _contains_any(text, user_data_words):
        return "use_user_data"
    if _contains_any(text, show_raster_words):
        return "show_uploaded_raster"
    if _contains_any(text, map_words) and _contains_any(lower, [w.lower() for w in map_topics]):
        return "render_map"
    if _contains_any(text, reliability_words):
        return "reliability_analysis"
    if _contains_any(lower, model_words):
        return "explain_model"
    if _contains_any(lower, metrics_words):
        return "ask_accuracy_metrics"
    if _contains_any(text, improve_words):
        return "improvement_advice"
    if _contains_any(lower, dsm_words):
        return "dsm_knowledge"
    if _contains_any(text, som_words):
        return "som_knowledge"

    if llm_enabled:
        llm_intent = classify_intent_with_llm(text)
        if llm_intent:
            return llm_intent

    return "small_talk"
