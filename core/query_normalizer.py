import re
from typing import Iterable

COMMON_REPLACEMENTS = {
    "土朗有机质": "土壤有机质",
    "土嚷有机质": "土壤有机质",
    "土壤有机值": "土壤有机质",
    "有机值": "有机质",
    "有机资": "有机质",
    "不确丁性": "不确定性",
    "不确定形": "不确定性",
    "不确定醒": "不确定性",
    "成度": "成都",
    "成都是": "成都市",
    "靠不靠谱": "是否可靠",
    "稳不稳": "是否可靠",
    "准不准": "精度怎么样",
    "出一张图": "绘制一张图",
    "环静变量": "环境变量",
    "环境因孑": "环境因子",
    "环竟变量": "环境变量",
    "协表量": "协变量",
    "制囹": "制图",
    "土壤右机质": "土壤有机质",
    "制图任努": "制图任务",
    "靠普": "靠谱",
}

SOIL_MAPPING_TERMS = {
    "数字土壤制图", "土壤有机质", "有机质", "环境变量", "环境因子", "协变量", "遥感", "地形",
    "气候", "土地利用", "母质", "土壤质地", "空间验证", "不确定性", "成都市", "成都",
    "som", "rfk", "gcp", "制图", "出图", "预测图", "分布图", "精度", "结果", "图件",
}

AUTO_DSM_TERMS = {
    "mitsubishi", "eclipse", "talon", "plymouth", "laser", "4g63", "brembo", "engine",
    "brake", "caliper", "dsmtuners", "galant", "vr-4", "fuse", "wiring", "plug",
}


def _replace_many(text: str, mapping: dict[str, str]) -> str:
    out = text
    for k, v in mapping.items():
        out = out.replace(k, v)
    return out


def _normalize_space(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def infer_domain(text: str) -> str:
    lower = (text or "").lower()
    soil_score = sum(1 for t in SOIL_MAPPING_TERMS if t in lower or t in text)
    auto_score = sum(1 for t in AUTO_DSM_TERMS if t in lower)
    if soil_score == 0 and auto_score == 0:
        return "unknown"
    return "soil_dsm" if soil_score >= auto_score else "auto_dsm"


def is_probably_soil_mapping_query(text: str) -> bool:
    domain = infer_domain(text)
    if domain == "soil_dsm":
        return True
    lower = (text or "").lower()
    return any(k in lower or k in text for k in ["dsm 常用", "环境变量", "土壤", "有机质", "协变量", "数字土壤制图"])


def enrich_soil_dsm_terms(text: str) -> str:
    s = text
    if "dsm" in s.lower() and "数字土壤制图" not in s:
        if is_probably_soil_mapping_query(s):
            s = re.sub(r"\b[dD][sS][mM]\b", "数字土壤制图", s)
            s = s.replace("DSM", "数字土壤制图").replace("dsm", "数字土壤制图")
    if "常用环境变量" in s and "数字土壤制图" not in s and is_probably_soil_mapping_query(s):
        s = f"数字土壤制图 {s}"
    return s


def normalize_query(text: str) -> str:
    if not text:
        return ""
    s = _normalize_space(text)
    s = _replace_many(s, COMMON_REPLACEMENTS)
    s = enrich_soil_dsm_terms(s)
    s = re.sub(r"[，,。；;]{2,}", "，", s)
    s = _normalize_space(s)
    return s
