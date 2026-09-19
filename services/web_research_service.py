from config.settings import ENABLE_WEB_RESEARCH, SEARCH_RESULT_LIMIT
from core.query_normalizer import infer_domain, is_probably_soil_mapping_query, normalize_query


NEGATIVE_AUTO_TERMS = [
    "Mitsubishi", "Eclipse", "Talon", "Plymouth", "Laser", "4G63", "DSMtuners", "engine", "brake", "wiring"
]

RELEVANT_HINTS = [
    "数字土壤制图", "土壤有机质", "环境变量", "协变量", "环境因子", "遥感", "地形", "气候", "土壤", "SOM"
]


def build_search_queries(user_text: str) -> list[str]:
    text = normalize_query(user_text)
    queries: list[str] = []
    if is_probably_soil_mapping_query(text):
        base = text.replace("DSM", "数字土壤制图").replace("dsm", "数字土壤制图")
        negatives = " ".join(f"-{t}" for t in NEGATIVE_AUTO_TERMS)
        queries.append(f'{base} 数字土壤制图 土壤有机质 {negatives}')
        if "环境变量" in text or "环境因子" in text or "协变量" in text:
            queries.append(f'数字土壤制图 土壤有机质 常用环境变量 协变量 遥感 地形 气候 {negatives}')
        if "精度" in text or "优化" in text or "提升" in text:
            queries.append(f'数字土壤制图 土壤有机质 精度提升 优化 协变量 采样 验证 {negatives}')
    else:
        queries.append(text)
    seen = set()
    out = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out[:3]


def _score_result(text: str) -> int:
    merged = text.lower()
    score = 0
    for hint in RELEVANT_HINTS:
        if hint.lower() in merged:
            score += 2
    if any(t.lower() in merged for t in [x.lower() for x in NEGATIVE_AUTO_TERMS]):
        score -= 5
    if "soil" in merged or "digital soil mapping" in merged:
        score += 2
    return score


def search_web(query: str):
    if not ENABLE_WEB_RESEARCH:
        return []
    try:
        from duckduckgo_search import DDGS
        out = []
        queries = build_search_queries(query)
        with DDGS() as ddgs:
            for q in queries:
                for item in ddgs.text(q, max_results=SEARCH_RESULT_LIMIT):
                    title = item.get("title", "")
                    body = item.get("body", "")
                    href = item.get("href", "")
                    merged = " | ".join(x for x in [title, body, href] if x)
                    if not merged:
                        continue
                    out.append({"text": merged, "score": _score_result(merged)})
        out.sort(key=lambda x: x["score"], reverse=True)
        return [x["text"] for x in out if x["score"] >= 0][:SEARCH_RESULT_LIMIT]
    except Exception:
        return []
