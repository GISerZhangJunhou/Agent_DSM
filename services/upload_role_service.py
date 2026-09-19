from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import rasterio
except Exception:  # pragma: no cover
    rasterio = None

from services.llm_service import compose_agent_reply, get_llm_health

DEFAULT_COVARIATES = [
    "DEM", "slope", "aspect_sin", "aspect_cos", "TPI", "TRI", "BD", "CEC", "pH",
    "sand", "silt", "clay", "gravel", "porosity", "CLCD", "BareSoil",
    "ChinaCP", "Management", "VegPhenology", "SeasonalClimateWater",
]

COVARIATE_CN_LABELS: dict[str, str] = {
    "DEM": "数字高程模型",
    "slope": "坡度",
    "aspect_sin": "坡向正弦项",
    "aspect_cos": "坡向余弦项",
    "TPI": "地形位置指数",
    "TRI": "地形崎岖度",
    "BD": "土壤容重",
    "CEC": "阳离子交换量",
    "pH": "土壤 pH",
    "sand": "砂粒含量",
    "silt": "粉粒含量",
    "clay": "黏粒含量",
    "gravel": "砾石含量",
    "porosity": "孔隙度",
    "CLCD": "中国土地覆盖数据",
    "LULCcd": "土地利用/土地覆盖",
    "BareSoil": "裸土暴露度",
    "ChinaCP": "复种/作物强度",
    "Management": "农业管理因子",
    "VegPhenology": "植被物候",
    "SeasonalClimateWater": "季节性气候水分因子",
}


def covariate_display_name(cov: str | None, with_code: bool = True) -> str:
    """Return a user-facing Chinese covariate name.

    `with_code=False` is used in the welcome prompt to keep the first message
    compact; `with_code=True` is useful in upload feedback where the original
    filename may contain the English covariate code.
    """
    key = str(cov or "").strip()
    cn = COVARIATE_CN_LABELS.get(key, key or "未知协变量")
    if with_code and key and key != cn:
        return f"{cn}（{key}）"
    return cn


DERIVED_FROM_DEM = {"slope", "aspect_sin", "aspect_cos", "TPI", "TRI"}

COV_ALIASES: dict[str, list[str]] = {
    "DEM": ["dem", "elevation", "高程", "海拔"],
    "BD": ["bd", "bulk_density", "容重", "土壤容重"],
    "CEC": ["cec", "阳离子交换", "cation"],
    "pH": ["ph", "pH", "soilph"],
    "sand": ["sand", "砂", "砂粒"],
    "silt": ["silt", "粉砂", "粉粒"],
    "clay": ["clay", "黏土", "粘土", "黏粒"],
    "gravel": ["gravel", "砾石", "砾"],
    "porosity": ["porosity", "孔隙", "孔隙度"],
    "CLCD": ["clcd"],
    "LULCcd": ["lulc", "landcover", "land_cover", "土地覆盖", "土地利用"],
    "BareSoil": ["baresoil", "bare_soil", "裸土"],
    "ChinaCP": ["chinacp", "复种", "cropping", "crop_intensity"],
    "Management": ["management", "管理", "human", "人为"],
    "VegPhenology": ["vegphenology", "vegphenologymodis", "phenology", "modis", "ndvi", "evi", "vegetation", "植被", "物候"],
    "SeasonalClimateWater": ["seasonalclimatewater", "climate", "precip", "temperature", "temp", "water", "气候", "降水", "温度", "水热"],
}

SAMPLE_TARGET_KEYS = ["有机质", "有机质含量", "土壤有机质", "土壤有机质含量", "som", "SOM", "som_value", "om", "OM", "om_value", "soc", "SOC", "organic_matter", "soil_organic_matter", "organic matter", "soil organic matter", "有机碳", "土壤有机碳"]
LON_KEYS = ["lon", "lng", "long", "longitude", "经度", "东经", "x", "x_4326", "lon_wgs84", "sample_lon"]
LAT_KEYS = ["lat", "latitude", "纬度", "北纬", "y", "y_4326", "lat_wgs84", "sample_lat"]


def _norm(s: Any) -> str:
    return re.sub(r"[\s_\-()（）]+", "", str(s or "").strip().lower())


def _contains_any(name: str, words: list[str]) -> bool:
    n = _norm(name)
    return any(_norm(w) in n for w in words if w)


def match_covariate_name(name: str) -> str | None:
    """Match an uploaded filename to a canonical covariate name.

    Longer / more specific aliases are evaluated first. This avoids generic
    tokens from hiding an explicit filename such as
    ``Chengdu_VegPhenology_MODIS_2019_2021_250m.tif``.
    """
    text = _norm(name)
    if not text:
        return None
    pairs: list[tuple[str, str]] = []
    for cov, aliases in COV_ALIASES.items():
        for alias in aliases:
            a = _norm(alias)
            if a:
                pairs.append((cov, a))
    pairs.sort(key=lambda x: len(x[1]), reverse=True)
    for cov, alias in pairs:
        if alias in text:
            return cov
    return None


def _read_table_header(path: Path) -> tuple[list[str], int | None]:
    if pd is None:
        return [], None
    try:
        if path.suffix.lower() in {".csv", ".txt"}:
            for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "cp936"):
                try:
                    df = pd.read_csv(path, nrows=20, encoding=enc, sep=None, engine="python")
                    return [str(c) for c in df.columns], None
                except Exception:
                    continue
        elif path.suffix.lower() in {".xls", ".xlsx"}:
            df = pd.read_excel(path, nrows=20)
            return [str(c) for c in df.columns], None
    except Exception:
        pass
    return [], None


def _classify_table(path: Path, item: dict) -> dict:
    columns = (item.get("validation") or {}).get("columns") or []
    if not columns:
        columns, _ = _read_table_header(path)
    cols_norm = [_norm(c) for c in columns]
    has_lon = any(_norm(k) in cols_norm for k in LON_KEYS)
    has_lat = any(_norm(k) in cols_norm for k in LAT_KEYS)
    has_target = any(_norm(k) in cols_norm for k in SAMPLE_TARGET_KEYS)
    if has_lon and has_lat and has_target:
        return {"role": "sample_points", "label": "土壤有机质训练样点", "confidence": 0.96, "can_model": "training_target", "reason": "识别到经纬度字段和土壤有机质/SOM目标字段。", "matched_covariate": None}
    if has_lon and has_lat:
        cov = match_covariate_name(path.name + " " + " ".join(columns))
        return {"role": "covariate_table_points", "label": "点状环境观测表", "confidence": 0.72, "can_model": "after_interpolation", "reason": "识别到经纬度，但没有SOM目标字段；需要插值/栅格化后才能作为全域协变量。", "matched_covariate": cov}
    return {"role": "unknown_table", "label": "未知表格", "confidence": 0.45, "can_model": False, "reason": "未识别到完整经纬度与目标变量组合，需要用户确认。", "matched_covariate": None}


def _raster_basic_meta(path: Path) -> dict:
    meta = {}
    if rasterio is None:
        return {"rasterio_available": False}
    try:
        with rasterio.open(path) as src:
            meta = {
                "crs": str(src.crs) if src.crs else None,
                "width": int(src.width),
                "height": int(src.height),
                "count": int(src.count),
                "bounds": [float(src.bounds.left), float(src.bounds.bottom), float(src.bounds.right), float(src.bounds.top)],
                "resolution": [float(abs(src.res[0])), float(abs(src.res[1]))],
                "nodata": None if src.nodata is None else float(src.nodata) if isinstance(src.nodata, (int, float)) else str(src.nodata),
            }
            try:
                import numpy as np
                arr = src.read(1, masked=True)
                valid = np.asarray(~np.ma.getmaskarray(arr), dtype=bool)
                total = int(valid.size)
                meta["valid_ratio"] = float(valid.sum() / total) if total else None
            except Exception:
                pass
    except Exception as exc:
        meta = {"read_error": str(exc)}
    return meta


def _classify_raster(path: Path, item: dict) -> dict:
    lower = path.name.lower()
    meta = _raster_basic_meta(path)
    if any(k in lower for k in ["mask", "cropland", "admin", "boundary", "clip", "掩膜", "边界", "耕地"]):
        return {"role": "mask_raster", "label": "mask / 裁剪审计层", "confidence": 0.88, "can_model": False, "reason": "文件名显示为mask/边界/耕地掩膜，只能用于裁剪或审计，不进入模型特征。", "matched_covariate": None, "spatial_meta": meta}
    if any(k in lower for k in ["som", "soc", "organic", "prediction", "pred", "有机质", "预测"]):
        return {"role": "target_prediction_raster", "label": "已有预测结果图", "confidence": 0.82, "can_model": False, "reason": "文件名疑似SOM/有机质预测结果，不能作为协变量，避免目标泄漏。", "matched_covariate": None, "spatial_meta": meta}
    cov = match_covariate_name(path.name)
    role = "covariate_categorical" if cov in {"CLCD", "LULCcd"} else "covariate_raster"
    label = "分类型环境协变量" if role == "covariate_categorical" else "连续型环境协变量栅格"
    conf = 0.92 if cov else 0.68
    return {"role": role, "label": label, "confidence": conf, "can_model": True, "reason": (f"按文件名匹配为默认协变量：{cov}。" if cov else "可读取为空间栅格，但未匹配到默认协变量名，建议用户确认变量含义。"), "matched_covariate": cov, "spatial_meta": meta}


def _classify_vector(path: Path, item: dict) -> dict:
    lower = path.name.lower()
    if any(k in lower for k in ["admin", "boundary", "aoi", "行政", "边界", "区划"]):
        return {"role": "admin_boundary", "label": "行政边界 / AOI", "confidence": 0.82, "can_model": False, "reason": "矢量边界用于AOI匹配、裁剪或审计，不进入模型。", "matched_covariate": None}
    return {"role": "vector_unknown", "label": "矢量数据", "confidence": 0.55, "can_model": "after_rasterization", "reason": "矢量数据需确认用途；若为面状环境变量，需要栅格化后才能入模。", "matched_covariate": None}


def classify_file(item: dict) -> dict:
    path = Path(str(item.get("path") or ""))
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt", ".xls", ".xlsx"}:
        cls = _classify_table(path, item)
    elif suffix in {".tif", ".tiff"}:
        cls = _classify_raster(path, item)
    elif suffix in {".nc", ".hdf", ".h5", ".hdf5"}:
        cov = match_covariate_name(path.name)
        cls = {"role": "multidimensional_covariate", "label": "多维环境协变量", "confidence": 0.76 if cov else 0.62, "can_model": "after_conversion", "reason": "NetCDF/HDF需要提取变量与时间窗口并转换为GeoTIFF后才能入模。", "matched_covariate": cov}
    elif suffix in {".shp", ".gpkg", ".geojson", ".json"}:
        cls = _classify_vector(path, item)
    elif suffix == ".zip":
        cls = {"role": "archive_package", "label": "压缩数据包", "confidence": 0.6, "can_model": False, "reason": "压缩包本身不入模，系统会识别解压后的内部文件。", "matched_covariate": None}
    else:
        cls = {"role": "unknown", "label": "未知类型", "confidence": 0.3, "can_model": False, "reason": "不支持或无法识别的文件类型，需要用户确认。", "matched_covariate": None}
    original_name = item.get("original_name") or item.get("name") or path.name
    cls.update({
        "name": original_name,
        "display_name": original_name,
        "stored_name": item.get("name") or path.name,
        "path": str(path),
        "size": item.get("size"),
    })
    return cls


def build_role_report(files: list[dict]) -> dict:
    items = []
    for f in files or []:
        try:
            cls = classify_file(f)
            f["role_inference"] = cls
            items.append(cls)
        except Exception as exc:
            items.append({"name": f.get("name"), "path": f.get("path"), "role": "unknown", "label": "识别失败", "confidence": 0, "can_model": False, "reason": str(exc)})
    return {"items": items, "summary": summarize_role_items(items)}


def summarize_role_items(items: list[dict]) -> dict:
    def count(pred): return sum(1 for x in items if pred(x))
    return {
        "file_count": len(items),
        "sample_count": count(lambda x: x.get("role") == "sample_points"),
        "model_covariate_count": count(lambda x: bool(x.get("can_model") is True)),
        "pending_conversion_count": count(lambda x: str(x.get("can_model")) in {"after_conversion", "after_interpolation", "after_rasterization"}),
        "mask_or_boundary_count": count(lambda x: x.get("role") in {"mask_raster", "admin_boundary"}),
        "blocked_count": count(lambda x: x.get("can_model") is False and x.get("role") not in {"sample_points", "admin_boundary", "mask_raster", "archive_package"}),
    }


def build_covariate_plan(role_report: dict) -> dict:
    items = role_report.get("items") or []
    provided: dict[str, list[dict]] = {}
    for it in items:
        cov = it.get("matched_covariate")
        if cov and (it.get("can_model") is True or str(it.get("can_model")) in {"after_conversion"}):
            provided.setdefault(cov, []).append({"file": it.get("name"), "role": it.get("role"), "path": it.get("path")})
    has_dem = bool(provided.get("DEM"))
    plan_items = []
    for cov in DEFAULT_COVARIATES:
        if cov in DERIVED_FROM_DEM:
            status = "derivable" if has_dem else "missing"
            source = "由 DEM 自动派生" if has_dem else "需要 DEM 后派生"
            action = "自动派生" if has_dem else "补充 DEM"
        elif cov in provided:
            status = "ready"
            source = "用户上传"
            action = "直接使用"
        else:
            status = "missing"
            source = "缺失"
            action = "可选择 TPDC 补充、GEE 补充或跳过/仅用现有数据"
        plan_items.append({"covariate": cov, "status": status, "source": source, "action": action, "provided_files": provided.get(cov, [])})
    missing = [x["covariate"] for x in plan_items if x["status"] == "missing"]
    ready = [x["covariate"] for x in plan_items if x["status"] in {"ready", "derivable"}]
    return {"default_covariates": DEFAULT_COVARIATES, "items": plan_items, "ready": ready, "missing": missing, "provided": provided}


def _rule_based_review(role_report: dict, plan: dict) -> str:
    """Build the factual part of the upload review.

    This is intentionally phrased as an AI-facing data review instead of a
    rigid “缺失清单”. Missing default variables are recommendations, not hard
    blockers, because the user may still run a baseline map with the uploaded
    covariates.
    """
    s = role_report.get("summary") or {}
    ready = plan.get("ready") or []
    missing = plan.get("missing") or []
    ready_cn = [covariate_display_name(x, with_code=True) for x in ready]
    missing_cn = [covariate_display_name(x, with_code=True) for x in missing]
    lines = [
        "上传数据审查结果：",
        f"- 已识别文件 {s.get('file_count', 0)} 个，其中训练样点 {s.get('sample_count', 0)} 个、可直接入模环境协变量栅格 {s.get('model_covariate_count', 0)} 个、待转换/插值数据 {s.get('pending_conversion_count', 0)} 个。",
        f"- 当前可用于制图或可由 DEM 自动派生的协变量：{('、'.join(ready_cn) if ready_cn else '暂无')}。",
    ]
    if missing:
        lines.append(f"- 建议补充的默认协变量：{('、'.join(missing_cn))}。这些不是硬性缺失项，不影响先用现有上传数据做基线制图。")
    else:
        lines.append("- 默认协变量清单已基本覆盖，暂不需要额外补充。")
    if not s.get("sample_count"):
        lines.append("- 需要先上传包含经度、纬度和有机质字段的采样点数据，才可以进入 RFK 制图。")
    if missing:
        lines.append("- 若希望结果更完整，可从国家青藏高原科学数据中心或 Google Earth Engine 补充建议项；也可以直接使用当前上传数据先做一次基线制图。")
    lines.append("- 说明：行政边界、掩膜、已有预测图、样点表中的普通实验室属性默认不作为全域环境协变量入模。")
    return "\n".join(lines)


def _extract_covariate_fact_summary(role_report: dict, plan: dict) -> dict:
    """Extract hard facts for AI review, including sample table diagnostics.

    The LLM must not infer sample row counts or SOM fields from filenames/chat.
    These facts come from validation/preprocessing only and are used both in the
    prompt and in contradiction checks.
    """
    items = role_report.get("items") or []
    summary = role_report.get("summary") or {}
    preprocess = plan.get("preprocess") if isinstance(plan, dict) else {}
    preprocess = preprocess or {}
    uploaded_cov_files = [
        str(x.get("display_name") or x.get("name") or "")
        for x in items
        if x.get("can_model") is True and x.get("role") in {"covariate_raster", "covariate_categorical"}
    ]
    ready = list(plan.get("ready") or [])
    missing = list(plan.get("missing") or [])

    sample_details = []
    # Prefer preprocessing records because they contain cleaned sample counts.
    for r in preprocess.get("sample_records") or []:
        mf = r.get("matched_fields") or {}
        sample_details.append({
            "name": r.get("name"),
            "ready": bool(r.get("ready")),
            "status": r.get("status"),
            "raw_rows": r.get("row_count_raw"),
            "valid_rows": r.get("row_count_cleaned"),
            "dropped_rows": r.get("dropped_count"),
            "duplicate_xy_count": r.get("duplicate_xy_count"),
            "lon_field": mf.get("lon"),
            "lat_field": mf.get("lat"),
            "som_field": mf.get("som"),
            "som_min": r.get("som_min"),
            "som_max": r.get("som_max"),
            "message": r.get("message"),
        })
    # Also include upload-validation facts in case preprocessing did not run.
    for x in items:
        if x.get("role") != "sample_points":
            continue
        if any(str(d.get("name")) == str(x.get("name")) for d in sample_details):
            continue
        val = {}
        # role_report item does not always carry validation; find common fields if present.
        mf = x.get("matched_fields") or {}
        sample_details.append({
            "name": x.get("display_name") or x.get("name"),
            "ready": True,
            "status": "recognized",
            "raw_rows": x.get("rows"),
            "valid_rows": x.get("rows"),
            "lon_field": mf.get("lon"),
            "lat_field": mf.get("lat"),
            "som_field": mf.get("som"),
        })

    ready_sample_rows = [int(d.get("valid_rows") or 0) for d in sample_details if d.get("ready")]
    total_ready_sample_rows = sum(ready_sample_rows)
    return {
        "file_count": int(summary.get("file_count") or len(items)),
        "sample_count": int(summary.get("sample_count") or 0),
        "sample_details": sample_details,
        "ready_sample_row_count": int(total_ready_sample_rows),
        "has_ready_sample": bool(total_ready_sample_rows > 0 or summary.get("sample_count")),
        "uploaded_covariate_file_count": len(uploaded_cov_files),
        "uploaded_covariate_files": uploaded_cov_files,
        "ready_or_derivable_covariates": ready,
        "ready_or_derivable_count": len(ready),
        "recommended_covariates": missing,
        "recommended_count": len(missing),
        # Backward-compatible keys used by the contradiction guard.
        "missing_covariates": missing,
        "missing_count": len(missing),
    }

def _llm_review_contradicts_facts(text: str, facts: dict) -> bool:
    """Reject LLM reviews that invent different counts or miss confirmed sample fields."""
    if not text:
        return False
    t = str(text)
    expected = int(facts.get("uploaded_covariate_file_count") or 0)
    patterns = [
        r"上传(?:了|的)?\s*(\d+)\s*个[^\n，。；]*协变量",
        r"(\d+)\s*个[^\n，。；]*协变量",
        r"可(?:直接)?入模(?:环境)?协变量[：: ]*(\d+)\s*个",
    ]
    for pat in patterns:
        for m in re.finditer(pat, t):
            try:
                n = int(m.group(1))
            except Exception:
                continue
            if n != expected and n != int(facts.get("ready_or_derivable_count") or -1):
                return True
    ready_set = set(facts.get("ready_or_derivable_covariates") or [])
    for cov in ["DEM", "BD", "CEC", "pH", "sand", "silt", "clay", "gravel", "porosity", "CLCD", "BareSoil", "ChinaCP", "Management", "SeasonalClimateWater", "VegPhenology"]:
        if cov in ready_set and re.search(rf"缺失[^。；\n]*{re.escape(cov)}|{re.escape(cov)}[^。；\n]*缺失", t, flags=re.I):
            return True

    rows = int(facts.get("ready_sample_row_count") or 0)
    if rows >= 2:
        # Hard blockers contradicted by the local parser.
        bad_phrases = [
            "仅含1个空间点", "仅含 1 个空间点", "只有1个空间点", "只有 1 个空间点",
            "没有发现土壤有机质", "未发现土壤有机质", "没有SOM", "无SOM", "缺少SOM", "缺少 SOM",
            "无法作为因变量", "不能作为因变量", "无法作为目标变量", "不能作为目标变量",
        ]
        if any(p in t for p in bad_phrases):
            return True
        # If the LLM states a sample count that is not close to the parsed count, reject.
        for m in re.finditer(r"(?:有效样点|空间点|样点)[^\d]{0,6}(\d+)\s*个", t):
            try:
                n = int(m.group(1))
                if n not in {rows, int(facts.get("sample_count") or -1)} and n < max(10, rows * 0.5):
                    return True
            except Exception:
                pass
    return False

def _compact_fact_review(facts: dict) -> str:
    rec = facts.get("recommended_covariates") or []
    rec_text = "、".join(covariate_display_name(x, with_code=True) for x in rec) if rec else "暂无"
    sample_bits = []
    for d in facts.get("sample_details") or []:
        name = d.get("name") or "样点文件"
        rows = d.get("valid_rows") or d.get("raw_rows") or 0
        lonf, latf, somf = d.get("lon_field"), d.get("lat_field"), d.get("som_field")
        if d.get("ready"):
            sample_bits.append(f"{name} 已识别为训练样点：有效样点 {rows} 个，坐标字段 {lonf}/{latf}，有机质字段 {somf}。")
        else:
            sample_bits.append(f"{name} 暂未通过样点检查：{d.get('message') or d.get('status') or '字段不完整'}。")
    sample_text = " ".join(sample_bits) if sample_bits else f"训练样点 {facts.get('sample_count', 0)} 个。"
    return (
        f"我已经完成上传数据审查：本会话共识别 {facts['file_count']} 个文件，其中{sample_text}"
        f"上传的可入模协变量栅格 {facts['uploaded_covariate_file_count']} 个；"
        f"当前可用或可由 DEM 派生的协变量共 {facts['ready_or_derivable_count']} 项。"
        f"建议补充项：{rec_text}。这些只是提升完整性的建议，不是制图硬性门槛；"
        "如果训练样点和至少一个全域协变量已经就绪，可以先做一次基线制图。"
    )

def generate_ai_covariate_review(role_report: dict, plan: dict, chat_history: list[dict] | None = None) -> str:
    """Generate a natural AI-style upload review after sample + covariates are both present."""
    facts = _extract_covariate_fact_summary(role_report, plan)
    if os.getenv("ENABLE_QWEN", "1") != "1" or not get_llm_health().get("api_key_exists"):
        return _compact_fact_review(facts) + "\n当前未检测到可用 Qwen API Key，因此这部分由系统根据文件硬事实生成。"
    payload = {"hard_facts": facts, "role_report": role_report, "covariate_plan": plan}
    prompt = (
        "你是数字土壤制图的数据审查员。请像正常 AI 助手一样自然分析用户上传的数据，不要输出固定模板、不要出现‘规则引擎’或‘硬规则’字样。"
        "必须严格服从 hard_facts 中的数量、样点有效行数、SOM字段、文件清单、可用协变量和建议补充项；如果有建议补充项，只能称为建议，不要说成硬性缺失。"
        "请判断：样点是否可用于 RFK 制图、协变量是否基本合理、还需要用户注意什么、下一步可以怎么做。"
        "回答要分层但简洁，最多 6 条。\n"
        + json.dumps(payload, ensure_ascii=False)[:9000]
    )
    try:
        text = compose_agent_reply("请审查我上传的土壤有机质制图数据是否合理", extra_context=prompt, chat_history=chat_history)
        text = (text or "").strip()
        if not text or _llm_review_contradicts_facts(text, facts):
            return _compact_fact_review(facts) + "\n本次数据审查已根据本地文件识别结果校正，并按实际文件状态输出结论。"
        return text
    except Exception:
        return _compact_fact_review(facts) + "\n数据审查补充分析暂不可用，已基于本地文件检查结果给出以上结论。"

def write_upload_analysis(session_id: str, role_report: dict, plan: dict, ai_review: str) -> dict[str, str]:
    """Write upload-analysis reports to a flat Chinese-named report folder."""
    root = Path(os.getenv("PRO_REPORT_ROOT", r"E:\Agent_DSM\runs"))
    out_dir = root / "上传数据分析报告"
    out_dir.mkdir(parents=True, exist_ok=True)
    short_id = str(session_id or "session")[:8]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{stamp}_{short_id}"
    role_path = out_dir / f"角色_{prefix}.json"
    plan_path = out_dir / f"计划_{prefix}.json"
    ai_path = out_dir / f"数据审查_{prefix}.txt"
    role_path.write_text(json.dumps(role_report, ensure_ascii=False, indent=2), encoding="utf-8")
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    ai_path.write_text(ai_review, encoding="utf-8")
    return {
        "report_dir": str(out_dir),
        "角色报告": str(role_path),
        "协变量计划": str(plan_path),
        "数据审查": str(ai_path),
    }
