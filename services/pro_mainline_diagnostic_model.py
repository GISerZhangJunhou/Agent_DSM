from __future__ import annotations

"""
PRO 主线诊断模型。

目的：验证“用户上传样点 -> PRO公开数据获取/记录 -> 生成模型输入 -> 训练 -> 输出GeoTIFF/报告”这条主线能否跑通。

重要边界：
- 这是 diagnostic test，不是正式土壤有机质制图模型。
- 当前只把公开网页/下载结果作为 provenance 和全局可用性信息记录；如果没有真实栅格协变量，
  则使用样点坐标派生的临时协变量生成诊断图。
- 输出文件会明确标记 mainline_diagnostic_test=true，防止误用为科研结果。
"""

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from utils.pro_console import pro_console_log

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import rasterio
    from rasterio.transform import from_origin
except Exception:  # pragma: no cover
    rasterio = None
    from_origin = None

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
except Exception:  # pragma: no cover
    RandomForestRegressor = None
    KFold = None
    cross_val_predict = None
    r2_score = None
    mean_squared_error = None
    mean_absolute_error = None


ENCODING_CANDIDATES = ["utf-8-sig", "utf-8", "gb18030", "gbk", "cp936", "gb2312", "big5", "cp950", "utf-16", "utf-16-le", "utf-16-be", "latin1"]

# V167: one robust schema dictionary for all PRO table entry points.  The upload
# validator already recognised OM, but the formal RFK pipeline used a smaller
# alias table here, so a fused CSV with columns like OM/lon/lat could pass upload
# validation and still fail at model start.  Keep this list deliberately broad
# for target/coordinate/auxiliary fields, while environmental covariates remain
# dynamic: every numeric non-core, non-leakage column is handled downstream.
FIELD_ALIASES = {
    "lon": [
        "lon", "longitude", "long", "lng", "x", "x_4326", "lon_wgs84", "longitude_wgs84",
        "sample_lon", "sample_longitude", "经度", "东经", "样点经度", "采样点经度",
    ],
    "lat": [
        "lat", "latitude", "y", "y_4326", "lat_wgs84", "latitude_wgs84",
        "sample_lat", "sample_latitude", "纬度", "北纬", "样点纬度", "采样点纬度",
    ],
    "som": [
        "som", "SOM", "som_value", "som_gkg", "som_g_kg", "som_g/kg", "som含量",
        "om", "OM", "om_value", "om_gkg", "om_g_kg", "om_g/kg", "om含量",
        "soil_organic_matter", "soil organic matter", "organic_matter", "organic matter",
        "organicmatter", "土壤有机质", "土壤有机质含量", "有机质", "有机质含量", "有机质gkg", "有机质g/kg",
        "soc", "SOC", "soil_organic_carbon", "organic_carbon", "有机碳", "土壤有机碳",
        "target_som", "target_om", "目标有机质",
    ],
    "sample_id": ["sample_id", "sampleid", "id", "ID", "fid", "FID", "objectid", "编号", "样点编号", "点号", "样本编号"],
    "year": ["year", "sample_year", "output_year", "年份", "采样年份", "制图年份", "调查年份", "sample_date", "date", "采样时间"],
    "depth": ["depth", "sample_depth", "采样深度", "土层深度", "深度", "depth_cm"],
}

REQUIRED_CORE_FIELDS = ["lon", "lat", "som"]


def _emit(stage: str, message: str, payload: Any | None = None, task_id: str | None = None) -> None:
    pro_console_log(stage, message, payload=payload, task_id=task_id)


def read_sample_table(sample_path: str | Path) -> tuple["pd.DataFrame", dict[str, Any]]:
    if pd is None:
        raise RuntimeError("缺少 pandas，无法读取样点表。")
    p = Path(sample_path)
    if not p.exists():
        raise FileNotFoundError(f"样点文件不存在：{p}")
    suffix = p.suffix.lower()
    meta: dict[str, Any] = {"path": str(p), "suffix": suffix, "encoding": None}
    if suffix in {".xls", ".xlsx"}:
        df = pd.read_excel(p)
        meta["encoding"] = "excel"
    elif suffix in {".csv", ".txt"}:
        try:
            from services.sample_reader import _read_csv_with_fallback
            df, read_meta = _read_csv_with_fallback(p, nrows=None)
            meta.update(read_meta or {})
        except Exception as exc:
            raise RuntimeError(f"样点CSV读取失败，已尝试多编码读取，错误：{exc}") from exc
    else:
        raise RuntimeError(f"主线诊断模型当前只支持 CSV/Excel 点样本：{suffix}")
    df.columns = [str(c).strip() for c in df.columns]
    meta["columns"] = list(df.columns)
    meta["row_count_raw"] = int(len(df))
    return df, meta


def _normalize_name(s: str) -> str:
    # Normalize headers such as "OM (g/kg)", "土壤有机质(g/kg)", "lon_wgs84".
    # Do not transliterate Chinese; just remove separators and unit punctuation.
    return re.sub(r"[\s_\-()（）\[\]【】{}:：/\\.%％]+", "", str(s or "").strip()).lower()


def _find_schema_field(columns: list[str], aliases: list[str]) -> str | None:
    # Exact normalized match first; this prevents short aliases such as x/y/OM
    # from accidentally matching long environmental names.
    norm_to_raw: dict[str, str] = {}
    for raw in columns:
        norm_to_raw.setdefault(_normalize_name(raw), raw)
    for alias in aliases:
        key = _normalize_name(alias)
        if key and key in norm_to_raw:
            return norm_to_raw[key]

    # Containment fallback for descriptive headers like "土壤有机质含量(g/kg)".
    # Aliases of length <= 2 are exact-only because "om" or "x" are too broad.
    alias_norms = sorted({_normalize_name(a) for a in aliases if len(_normalize_name(a)) > 2}, key=len, reverse=True)
    for raw in columns:
        nr = _normalize_name(raw)
        for key in alias_norms:
            if key and key in nr:
                return raw
    return None


def infer_core_fields(columns: list[str]) -> dict[str, str]:
    columns = [str(c).strip() for c in columns]
    matched: dict[str, str] = {}
    for key, aliases in FIELD_ALIASES.items():
        hit = _find_schema_field(columns, aliases)
        if hit:
            matched[key] = hit

    missing = [k for k in REQUIRED_CORE_FIELDS if k not in matched]
    if missing:
        debug = {
            "missing": missing,
            "columns": columns,
            "aliases": {k: FIELD_ALIASES[k] for k in REQUIRED_CORE_FIELDS},
            "note": "V167 支持 OM/SOM/有机质/土壤有机质等目标字段，以及 lon/lat/经度/纬度等坐标字段。",
        }
        raise RuntimeError("样点表缺少核心字段：" + ", ".join(missing) + "；字段识别诊断：" + json.dumps(debug, ensure_ascii=False))
    return matched


def _to_numeric(series):
    # 兼容中文CSV里带单位、空格、百分号等情况。
    return pd.to_numeric(series.astype(str).str.replace("%", "", regex=False).str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False), errors="coerce")


def build_model_table(df: "pd.DataFrame", matched: dict[str, str], manifest: dict[str, Any] | None = None) -> tuple["pd.DataFrame", list[str], dict[str, Any]]:
    m = manifest or {}
    records = m.get("records") or m.get("data_sources") or []
    usable_outputs = m.get("usable_outputs") or {}
    out = pd.DataFrame()
    out["lon"] = _to_numeric(df[matched["lon"]])
    out["lat"] = _to_numeric(df[matched["lat"]])
    out["som"] = _to_numeric(df[matched["som"]])
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["lon", "lat", "som"]).copy()
    if out.empty:
        raise RuntimeError("样点表读取成功，但 lon/lat/som 转数值后没有有效记录。")

    # 坐标派生临时协变量。只用于诊断模型链路。
    lon = out["lon"].to_numpy(dtype="float64")
    lat = out["lat"].to_numpy(dtype="float64")
    lon0, lat0 = float(np.nanmean(lon)), float(np.nanmean(lat))
    lon_span = max(float(np.nanmax(lon) - np.nanmin(lon)), 1e-6)
    lat_span = max(float(np.nanmax(lat) - np.nanmin(lat)), 1e-6)
    out["coord_lon_norm"] = (out["lon"] - lon0) / lon_span
    out["coord_lat_norm"] = (out["lat"] - lat0) / lat_span
    out["coord_dist_center"] = np.sqrt(out["coord_lon_norm"] ** 2 + out["coord_lat_norm"] ** 2)
    out["coord_lon_lat_interaction"] = out["coord_lon_norm"] * out["coord_lat_norm"]
    out["coord_sin_lon"] = np.sin(np.deg2rad(out["lon"]))
    out["coord_cos_lat"] = np.cos(np.deg2rad(out["lat"]))

    # 全局数据可用性特征。它本身不能提供空间差异，但可证明PRO下载/探测信息进入了建模表。
    # 为避免纯常数被模型完全忽略，加入与坐标的轻微交互项，仍明确标记为 diagnostic proxy。
    record_count = float(len(records))
    processed_count = float(usable_outputs.get("processed_file_count") or 0)
    saved_count = float(sum(1 for r in records if str(r.get("status")) in {"saved", "downloaded", "parsed", "available"}))
    out["pro_record_count"] = record_count
    out["pro_processed_file_count"] = processed_count
    out["pro_saved_source_count"] = saved_count
    out["pro_record_x_lon"] = record_count * out["coord_lon_norm"]
    out["pro_saved_x_lat"] = saved_count * out["coord_lat_norm"]

    feature_cols = [c for c in out.columns if c not in {"som"}]
    meta = {
        "valid_row_count": int(len(out)),
        "feature_columns": feature_cols,
        "manifest_record_count": int(record_count),
        "manifest_processed_file_count": int(processed_count),
        "warning": "当前为PRO主线诊断特征：若没有真实栅格协变量，则主要使用坐标派生特征和PRO数据可用性记录，不代表正式制图精度。",
    }
    return out, feature_cols, meta


def train_and_predict_grid(model_table: "pd.DataFrame", feature_cols: list[str], out_dir: Path, target: dict[str, Any] | None = None, task_id: str | None = None) -> dict[str, str]:
    if RandomForestRegressor is None or rasterio is None or from_origin is None:
        raise RuntimeError("缺少 sklearn 或 rasterio，无法执行主线诊断模型。")
    out_dir.mkdir(parents=True, exist_ok=True)
    X = model_table[feature_cols].to_numpy(dtype="float64")
    y = model_table["som"].to_numpy(dtype="float64")
    if len(y) < 5:
        raise RuntimeError(f"有效样点数过少：{len(y)}。主线连通性诊断至少需要 5 个点。")

    n_estimators = int(os.getenv("PRO_DIAG_RF_TREES", "80"))
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, min_samples_leaf=1, n_jobs=1)
    _emit("MODEL", "[PRO-DIAG] 开始训练临时随机森林模型", {"rows": len(y), "features": feature_cols, "trees": n_estimators}, task_id=task_id)

    # 简单CV，只用于确认评估链路可运行。样点太少时自动降级。
    cv_pred = None
    metrics = {}
    if len(y) >= 8 and KFold is not None and cross_val_predict is not None:
        n_splits = min(5, len(y))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        try:
            cv_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
            metrics = {
                "cv_type": f"KFold({n_splits}) diagnostic-test",
                "r2": float(r2_score(y, cv_pred)) if r2_score else None,
                "rmse": float(math.sqrt(mean_squared_error(y, cv_pred))) if mean_squared_error else None,
                "mae": float(mean_absolute_error(y, cv_pred)) if mean_absolute_error else None,
            }
        except Exception as exc:
            metrics = {"cv_type": "failed", "cv_error": str(exc)}
    else:
        metrics = {"cv_type": "skipped", "reason": "样点数不足或sklearn CV组件不可用"}

    model.fit(X, y)

    # 输出训练表和OOF表。
    train_csv = out_dir / "pro_diagnostic_model_training_table.csv"
    model_table.to_csv(train_csv, index=False, encoding="utf-8-sig")
    oof = model_table[["lon", "lat", "som"]].copy()
    if cv_pred is not None:
        oof["pred"] = cv_pred
        oof["residual"] = oof["som"] - oof["pred"]
    else:
        pred_train = model.predict(X)
        oof["pred"] = pred_train
        oof["residual"] = oof["som"] - oof["pred"]
    oof_csv = out_dir / "strict_groupkfold_oof.csv"
    oof.to_csv(oof_csv, index=False, encoding="utf-8-sig")

    # 建一个小格网，保证能在WebGIS叠加预览。范围来自样点包络。
    lon_min, lon_max = float(model_table["lon"].min()), float(model_table["lon"].max())
    lat_min, lat_max = float(model_table["lat"].min()), float(model_table["lat"].max())
    lon_pad = max((lon_max - lon_min) * 0.12, 0.02)
    lat_pad = max((lat_max - lat_min) * 0.12, 0.02)
    lon_min -= lon_pad; lon_max += lon_pad; lat_min -= lat_pad; lat_max += lat_pad
    max_dim = int(os.getenv("PRO_DIAG_GRID_SIZE", "180"))
    max_dim = max(40, min(max_dim, 360))
    aspect = max((lon_max - lon_min) / max(lat_max - lat_min, 1e-6), 0.2)
    width = max(40, min(max_dim, int(max_dim * min(aspect, 2.0))))
    height = max(40, min(max_dim, int(width / aspect) if aspect > 0 else max_dim))
    xs = np.linspace(lon_min, lon_max, width)
    ys = np.linspace(lat_max, lat_min, height)  # north -> south
    xx, yy = np.meshgrid(xs, ys)

    lon0 = float(model_table["lon"].mean())
    lat0 = float(model_table["lat"].mean())
    lon_span = max(float(model_table["lon"].max() - model_table["lon"].min()), 1e-6)
    lat_span = max(float(model_table["lat"].max() - model_table["lat"].min()), 1e-6)
    grid_df = pd.DataFrame({"lon": xx.ravel(), "lat": yy.ravel()})
    grid_df["coord_lon_norm"] = (grid_df["lon"] - lon0) / lon_span
    grid_df["coord_lat_norm"] = (grid_df["lat"] - lat0) / lat_span
    grid_df["coord_dist_center"] = np.sqrt(grid_df["coord_lon_norm"] ** 2 + grid_df["coord_lat_norm"] ** 2)
    grid_df["coord_lon_lat_interaction"] = grid_df["coord_lon_norm"] * grid_df["coord_lat_norm"]
    grid_df["coord_sin_lon"] = np.sin(np.deg2rad(grid_df["lon"]))
    grid_df["coord_cos_lat"] = np.cos(np.deg2rad(grid_df["lat"]))
    grid_df["pro_record_count"] = float(model_table["pro_record_count"].iloc[0]) if "pro_record_count" in model_table else 0.0
    grid_df["pro_processed_file_count"] = float(model_table["pro_processed_file_count"].iloc[0]) if "pro_processed_file_count" in model_table else 0.0
    grid_df["pro_saved_source_count"] = float(model_table["pro_saved_source_count"].iloc[0]) if "pro_saved_source_count" in model_table else 0.0
    grid_df["pro_record_x_lon"] = grid_df["pro_record_count"] * grid_df["coord_lon_norm"]
    grid_df["pro_saved_x_lat"] = grid_df["pro_saved_source_count"] * grid_df["coord_lat_norm"]

    pred = model.predict(grid_df[feature_cols].to_numpy(dtype="float64")).reshape((height, width)).astype("float32")
    xres = (lon_max - lon_min) / max(width - 1, 1)
    yres = (lat_max - lat_min) / max(height - 1, 1)
    transform = from_origin(lon_min - xres / 2, lat_max + yres / 2, xres, yres)
    pred_tif = out_dir / "pro_diagnostic_som_prediction.tif"
    with rasterio.open(
        pred_tif,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
        compress="lzw",
    ) as dst:
        dst.write(pred, 1)

    report = {
        "mainline_diagnostic_test": True,
        "not_for_scientific_use": True,
        "message": "该结果只用于验证PRO主线是否能够完成：上传样点→公开数据记录→建模表→训练→输出GeoTIFF。结果用于正式流程自检，请结合完整制图输出使用。",
        "target": target or {},
        "rows": int(len(model_table)),
        "features": feature_cols,
        "metrics": metrics,
        "prediction_grid": {"width": width, "height": height, "crs": "EPSG:4326", "bounds": [lon_min, lat_min, lon_max, lat_max]},
        "outputs": {},
    }
    report_json = out_dir / "pro_diagnostic_accuracy_report.json"
    manifest = out_dir / "rfk_manifest.json"
    outputs = {
        "pred_tif": str(pred_tif),
        "report_json": str(report_json),
        "manifest": str(manifest),
        "strict_oof_csv": str(oof_csv),
        "training_table_csv": str(train_csv),
    }
    report["outputs"] = outputs
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_payload = {
        "mainline_diagnostic_test": True,
        "created_at": int(time.time()),
        "created_at_text": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "RandomForestRegressor diagnostic test",
        "target": target or {},
        "outputs": outputs,
        "warning": report["message"],
    }
    manifest.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _emit("MODEL", "[PRO-DIAG] 临时制图模型完成", outputs, task_id=task_id)
    return outputs


def run_pro_mainline_diagnostic_model(manifest_path: str | Path, target: dict[str, Any] | None = None, task_id: str | None = None) -> dict[str, str]:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"PRO数据manifest不存在：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = manifest.get("spec") or {}
    sample_path = spec.get("sample_path") or manifest.get("sample_path")
    if not sample_path:
        raise RuntimeError("manifest中没有 sample_path，无法进行主线诊断建模。")
    work_dir = manifest_path.parent.parent
    out_dir = work_dir / "diagnostic_model_outputs"
    _emit("MODEL", "[PRO-DIAG] 开始主线诊断建模", {"manifest": str(manifest_path), "sample_path": sample_path, "out_dir": str(out_dir)}, task_id=task_id)
    df, read_meta = read_sample_table(sample_path)
    fields = infer_core_fields(list(df.columns))
    _emit("SAMPLE", "[PRO-DIAG] 样点表读取成功", {**read_meta, "matched_fields": fields}, task_id=task_id)
    model_table, feature_cols, table_meta = build_model_table(df, fields, manifest)
    _emit("MODEL", "[PRO-DIAG] 建模表生成成功", table_meta, task_id=task_id)
    return train_and_predict_grid(model_table, feature_cols, out_dir, target=target, task_id=task_id)
