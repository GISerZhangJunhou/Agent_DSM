from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from services.model_label_service import model_label, UNKNOWN_MODEL_LABEL


def sha256_file(path: str | Path | None) -> str:
    try:
        if not path:
            return ""
        p = Path(path)
        if not p.exists() or not p.is_file():
            return ""
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def read_json(path: str | Path | None) -> dict[str, Any]:
    try:
        if not path:
            return {}
        p = Path(path)
        if not p.exists() or not p.is_file():
            return {}
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            obj.setdefault("_source_json", str(p))
            obj.setdefault("_source_sha256", sha256_file(p))
            return obj
    except Exception:
        return {}
    return {}


def _first_existing_path(paths: dict[str, Any], keys: list[str]) -> str:
    for k in keys:
        v = paths.get(k)
        if v and Path(str(v)).exists():
            return str(v)
    return ""


def _candidate_report_json(paths: dict[str, Any]) -> str:
    p = _first_existing_path(paths, ["report_json", "model_report_json", "formal_model_report_json"])
    if p:
        return p
    pred = paths.get("pred_tif") or paths.get("prediction_tif") or paths.get("internal_pred_tif")
    if pred:
        try:
            base = Path(str(pred)).parent
            for name in ["formal_model_report.json", "model_report.json", "report.json"]:
                cand = base / name
                if cand.exists():
                    return str(cand)
        except Exception:
            pass
    final_dir = paths.get("final_result_dir")
    if final_dir:
        try:
            for name in ["formal_model_report.json", "模型报告.json", "model_report.json"]:
                cand = Path(str(final_dir)) / name
                if cand.exists():
                    return str(cand)
        except Exception:
            pass
    return ""


def raster_stats(tif_path: str | Path | None) -> dict[str, Any]:
    if not tif_path:
        return {}
    try:
        import numpy as np
        import rasterio
        p = Path(tif_path)
        if not p.exists():
            return {}
        with rasterio.open(p) as ds:
            arr = ds.read(1, masked=True).astype("float64")
            vals = arr.compressed()
            if vals.size == 0:
                return {"path": str(p), "valid_count": 0, "width": int(ds.width), "height": int(ds.height), "crs": str(ds.crs)}
            return {
                "path": str(p),
                "valid_count": int(vals.size),
                "width": int(ds.width),
                "height": int(ds.height),
                "crs": str(ds.crs),
                "min": float(np.nanmin(vals)),
                "max": float(np.nanmax(vals)),
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals)),
                "nodata": None if ds.nodata is None else float(ds.nodata),
            }
    except Exception as exc:
        return {"path": str(tif_path), "error": str(exc)}


def _metrics_dict(report: dict[str, Any]) -> dict[str, Any]:
    for key in ["metrics", "main_result", "formal_result", "validation_metrics", "model_metrics"]:
        obj = report.get(key)
        if isinstance(obj, dict) and obj:
            return obj
    return {}


def read_mapping_result(result_paths: dict[str, Any] | None = None, report: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read the current mapping run from disk/result paths.

    This is the single evidence gate for post-mapping analysis.  It never
    fabricates accuracy, model name, or spatial statistics.  Callers should only
    ask the AI to analyze the payload when ok=True.
    """
    paths = dict(result_paths or {})
    rep = dict(report or {}) if isinstance(report, dict) else {}
    report_json = str(paths.get("report_json") or paths.get("model_report_json") or rep.get("report_json") or rep.get("model_report_json") or "")
    if not (report_json and Path(report_json).exists()):
        report_json = _candidate_report_json({**paths, **rep})
    disk_report = read_json(report_json)
    if disk_report:
        merged_report = {**disk_report, **rep}
    else:
        merged_report = rep
    if report_json:
        merged_report.setdefault("report_json", report_json)
        merged_report.setdefault("report_sha256", sha256_file(report_json))
    pred_tif = str(paths.get("pred_tif") or paths.get("prediction_tif") or merged_report.get("pred_tif") or merged_report.get("prediction_tif") or merged_report.get("internal_pred_tif") or "")
    metrics = _metrics_dict(merged_report)
    missing: list[str] = []
    if not report_json or not Path(report_json).exists():
        missing.append("formal_model_report.json/report_json")
    if not metrics:
        missing.append("metrics")
    model_name = model_label(paths, merged_report)
    if model_name == UNKNOWN_MODEL_LABEL:
        missing.append("model_name")
    out = {
        "ok": bool(report_json and Path(report_json).exists() and metrics),
        "missing": missing,
        "result_paths": paths,
        "report": merged_report,
        "report_json": report_json,
        # Full SHA256 stays available to backend/audit code but should not be rendered
        # in user-facing chat cards.  Use public_run_code for any short run label.
        "report_sha256": sha256_file(report_json),
        "public_run_code": str(paths.get("public_run_code") or merged_report.get("public_run_code") or ""),
        "run_fingerprint": merged_report.get("run_fingerprint") or paths.get("run_fingerprint"),
        "model_name": model_name,
        "metrics": metrics,
        "pred_tif": pred_tif,
        "prediction_stats": raster_stats(pred_tif),
        "map_scope": paths.get("map_scope") or merged_report.get("map_scope") or merged_report.get("output_scope") or "未读取到制图范围",
        "target_unit": (merged_report.get("target") or {}).get("unit") if isinstance(merged_report.get("target"), dict) else (merged_report.get("target_unit") or "g/kg"),
    }
    return out


def read_gcp_result(result_paths: dict[str, Any] | None = None, report: dict[str, Any] | None = None) -> dict[str, Any]:
    paths = dict(result_paths or {})
    rep = dict(report or {}) if isinstance(report, dict) else {}
    report_json = str(paths.get("report_json") or rep.get("report_json") or "")
    if not (report_json and Path(report_json).exists()):
        for key in ["gcp_report_json", "uncertainty_report_json"]:
            val = paths.get(key) or rep.get(key)
            if val and Path(str(val)).exists():
                report_json = str(val)
                break
    disk_report = read_json(report_json)
    merged_report = {**disk_report, **rep} if disk_report else rep
    if report_json:
        merged_report.setdefault("report_json", report_json)
        merged_report.setdefault("report_sha256", sha256_file(report_json))
    sample_metrics = merged_report.get("sample_metrics") or merged_report.get("metrics") or {}
    if not isinstance(sample_metrics, dict):
        sample_metrics = {}
    width_tif = str(paths.get("width_tif") or (merged_report.get("outputs") or {}).get("width_tif") or "")
    missing: list[str] = []
    if not report_json or not Path(report_json).exists():
        missing.append("GCP_AOA指标报告.json/report_json")
    if not sample_metrics:
        missing.append("sample_metrics")
    return {
        "ok": bool(report_json and Path(report_json).exists() and sample_metrics),
        "missing": missing,
        "result_paths": paths,
        "report": merged_report,
        "report_json": report_json,
        "report_sha256": sha256_file(report_json),
        "public_run_code": str(paths.get("public_run_code") or merged_report.get("public_run_code") or ""),
        "sample_metrics": sample_metrics,
        "grid_statistics": merged_report.get("grid_statistics") or {},
        "width_tif": width_tif,
        "width_stats": raster_stats(width_tif),
        "previous_model_name": paths.get("previous_model_name") or merged_report.get("previous_model_name") or UNKNOWN_MODEL_LABEL,
        "target_unit": merged_report.get("target_unit") or "g/kg",
    }
