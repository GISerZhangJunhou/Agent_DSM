from __future__ import annotations

import json
from pathlib import Path
from typing import Any

UNKNOWN_MODEL_LABEL = "未读取到本轮模型名称"

_MODEL_KEYS = [
    "display_model_name", "model_name", "selected_model", "best_model",
    "model_algorithm", "algorithm", "model_type", "estimator", "learner",
    "final_model", "chosen_model", "training_model",
]

def _clean_model_name(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        for k in _MODEL_KEYS:
            if value.get(k):
                return _clean_model_name(value.get(k))
        if value.get("name"):
            return _clean_model_name(value.get("name"))
        return ""
    if isinstance(value, (list, tuple)):
        for item in value:
            s = _clean_model_name(item)
            if s:
                return s
        return ""
    s = str(value).strip()
    if not s or s.lower() in {"none", "null", "nan", "na"}:
        return ""
    aliases = {
        "RANDOMFOREST": "Random Forest",
        "RANDOM_FOREST": "Random Forest",
        "RF": "Random Forest",
        "RFK": "RFK",
        "RANDOM FOREST KRIGING": "RFK",
        "RANDOM_FOREST_KRIGING": "RFK",
        "XGB": "XGBoost",
        "XGBOOST": "XGBoost",
        "LGBM": "LightGBM",
        "LIGHTGBM": "LightGBM",
        "CATBOOST": "CatBoost",
        "KRIGING": "Kriging",
    }
    return aliases.get(s.upper(), s)

def _read_json(path: str | Path | None) -> dict[str, Any]:
    try:
        if not path:
            return {}
        p = Path(path)
        if not p.exists() or not p.is_file():
            return {}
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def extract_model_name_from_report(report: dict[str, Any] | None) -> str:
    if not isinstance(report, dict):
        return ""
    for k in _MODEL_KEYS:
        s = _clean_model_name(report.get(k))
        if s:
            return s
    nested_keys = ["model", "best_estimator", "selected_estimator", "pipeline", "metadata", "training", "summary", "target"]
    for nk in nested_keys:
        v = report.get(nk)
        if isinstance(v, dict):
            s = extract_model_name_from_report(v)
            if s:
                return s
    # Some reports store algorithm inside rfk sub-dict. Only use it when explicitly present.
    for nk in ["rfk", "rf", "xgboost", "lightgbm"]:
        v = report.get(nk)
        if isinstance(v, dict):
            s = _clean_model_name(v.get("algorithm") or v.get("model_name") or v.get("model_type"))
            if s:
                return s
    return ""

def extract_model_name(result_paths: dict[str, Any] | None = None, report: dict[str, Any] | None = None) -> str:
    s = extract_model_name_from_report(report)
    if s:
        return s
    paths = result_paths or {}
    if isinstance(paths, dict):
        for k in ["display_model_name", "model_name", "selected_model", "best_model", "model_algorithm", "algorithm", "model_type"]:
            s = _clean_model_name(paths.get(k))
            if s:
                return s
        for key in ["report_json", "model_report_json", "formal_model_report_json", "manifest", "manifest_path"]:
            obj = _read_json(paths.get(key))
            s = extract_model_name_from_report(obj)
            if s:
                return s
        try:
            pred = paths.get("pred_tif") or paths.get("prediction_tif")
            if pred:
                obj = _read_json(Path(pred).parent / "formal_model_report.json")
                s = extract_model_name_from_report(obj)
                if s:
                    return s
        except Exception:
            pass
    return ""

def model_label(result_paths: dict[str, Any] | None = None, report: dict[str, Any] | None = None) -> str:
    return extract_model_name(result_paths, report) or UNKNOWN_MODEL_LABEL
