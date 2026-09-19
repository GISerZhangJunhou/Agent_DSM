from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
except Exception:  # pragma: no cover
    rasterio = None
    reproject = None
    Resampling = None

try:
    from services.upload_role_service import SAMPLE_TARGET_KEYS, LON_KEYS, LAT_KEYS, DEFAULT_COVARIATES, DERIVED_FROM_DEM
except Exception:  # pragma: no cover
    SAMPLE_TARGET_KEYS = ["有机质", "有机质含量", "土壤有机质", "土壤有机质含量", "som", "SOM", "om", "OM", "soc", "SOC", "organic_matter", "soil_organic_matter"]
    LON_KEYS = ["lon", "lng", "long", "longitude", "经度", "东经", "x", "x_4326", "lon_wgs84", "sample_lon"]
    LAT_KEYS = ["lat", "latitude", "纬度", "北纬", "y", "y_4326", "lat_wgs84", "sample_lat"]
    DEFAULT_COVARIATES = []
    DERIVED_FROM_DEM = set()

try:
    from services.download_postprocess_service import convert_multidimensional_to_tif
except Exception:  # pragma: no cover
    convert_multidimensional_to_tif = None  # type: ignore


def _root() -> Path:
    return Path(os.getenv("PRO_PREPROCESS_ROOT", os.getenv("PRO_REPORT_ROOT", r"E:\Agent_DSM\runs")))


def _safe_stem(name: str) -> str:
    import re
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(name or "file")).strip(" ._")
    return (s[:80] or "file")


def _norm(s: Any) -> str:
    import re
    return re.sub(r"[\s_\-()（）]+", "", str(s or "").strip().lower())


def _find_col(cols: list[str], keys: list[str]) -> str | None:
    norm_map = {_norm(c): c for c in cols}
    for k in keys:
        if _norm(k) in norm_map:
            return norm_map[_norm(k)]
    # more permissive contains matching
    for c in cols:
        cn = _norm(c)
        if any(_norm(k) and _norm(k) in cn for k in keys):
            return c
    return None


def _read_table(path: Path):
    if pd is None:
        raise RuntimeError("pandas 不可用，无法预处理表格。")
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    last = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "cp936"):
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python")
        except Exception as exc:
            last = exc
    raise RuntimeError(f"表格读取失败：{last}")


def _preprocess_sample(item: dict, out_dir: Path) -> dict:
    path = Path(str(item.get("path") or ""))
    rec: dict[str, Any] = {"name": item.get("original_name") or item.get("name") or path.name, "input_path": str(path), "role": "sample_points"}
    try:
        df = _read_table(path)
        cols = [str(c) for c in df.columns]
        lon_col = _find_col(cols, LON_KEYS)
        lat_col = _find_col(cols, LAT_KEYS)
        som_col = _find_col(cols, SAMPLE_TARGET_KEYS)
        rec["matched_fields"] = {"lon": lon_col, "lat": lat_col, "som": som_col}
        if not (lon_col and lat_col and som_col):
            rec.update({"status": "failed", "ready": False, "message": "样点缺少 lon/lat/有机质 字段，不能用于制图。"})
            return rec
        work = df.copy()
        for c in (lon_col, lat_col, som_col):
            work[c] = pd.to_numeric(work[c].astype(str).str.replace("％", "%", regex=False).str.replace("%", "", regex=False).str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False), errors="coerce")
        before = len(work)
        work = work.dropna(subset=[lon_col, lat_col, som_col])
        work = work[(work[lon_col] >= -180) & (work[lon_col] <= 180) & (work[lat_col] >= -90) & (work[lat_col] <= 90)]
        # Do not silently delete target outliers; only report broad diagnostics.
        dup_count = int(work.duplicated(subset=[lon_col, lat_col]).sum())
        cleaned = out_dir / "cleaned_samples.csv"
        work.to_csv(cleaned, index=False, encoding="utf-8-sig")
        rec.update({
            "status": "ready",
            "ready": True,
            "cleaned_path": str(cleaned),
            "row_count_raw": int(before),
            "row_count_cleaned": int(len(work)),
            "dropped_count": int(before - len(work)),
            "duplicate_xy_count": dup_count,
            "som_min": float(work[som_col].min()) if len(work) else None,
            "som_max": float(work[som_col].max()) if len(work) else None,
            "message": f"样点预处理完成：有效样点 {len(work)} 个，删除无效记录 {before-len(work)} 条。",
        })
    except Exception as exc:
        rec.update({"status": "failed", "ready": False, "message": f"样点预处理失败：{exc}"})
    return rec


def _preprocess_raster(item: dict, out_dir: Path) -> dict:
    path = Path(str(item.get("path") or ""))
    role_inf = item.get("role_inference") or {}
    cov = role_inf.get("matched_covariate")
    role = role_inf.get("role") or "covariate_raster"
    rec: dict[str, Any] = {
        "name": item.get("original_name") or item.get("name") or path.name,
        "input_path": str(path),
        "role": role,
        "matched_covariate": cov,
        "can_enter_model": bool(role_inf.get("can_model") is True),
    }
    if role in {"mask_raster", "admin_boundary", "target_prediction_raster"} or role_inf.get("can_model") is not True:
        rec.update({"status": "excluded", "ready": False, "message": "该文件不作为模型协变量；仅可用于显示、裁剪、审计或参考。"})
        return rec
    if rasterio is None:
        rec.update({"status": "pending", "ready": False, "message": "rasterio 不可用，无法完成栅格预处理。"})
        return rec
    try:
        with rasterio.open(path) as src:
            meta = src.meta.copy()
            arr = src.read(1, masked=True)
            mask = np.ma.getmaskarray(arr) if np is not None else None
            total = int(arr.size)
            valid_count = int((~mask).sum()) if mask is not None else total
            valid_ratio = float(valid_count / total) if total else None
            # At upload stage we do not know final AOI/resolution yet. Create a normalized copy
            # preserving CRS/transform, and leave final clipping/resampling to model build stage.
            std_name = _safe_stem(cov or path.stem) + "_std.tif"
            std_path = out_dir / std_name
            shutil.copy2(path, std_path)
            rec.update({
                "status": "ready",
                "ready": True,
                "standardized_path": str(std_path),
                "crs": str(src.crs) if src.crs else None,
                "width": int(src.width),
                "height": int(src.height),
                "band_count": int(src.count),
                "resolution": [float(abs(src.res[0])), float(abs(src.res[1]))],
                "bounds": [float(src.bounds.left), float(src.bounds.bottom), float(src.bounds.right), float(src.bounds.top)],
                "valid_ratio": valid_ratio,
                "message": "栅格初步预处理完成：已复制到标准化协变量目录；最终裁剪/重采样将在制图建模阶段统一执行。",
            })
            if not src.crs:
                rec["status"] = "warning"
                rec["ready"] = False
                rec["message"] = "栅格缺少 CRS，暂不建议入模。请先定义投影。"
            elif valid_ratio is not None and valid_ratio < 0.2:
                rec["status"] = "warning"
                rec["message"] += " 但有效像元比例偏低，需谨慎。"
    except Exception as exc:
        rec.update({"status": "failed", "ready": False, "message": f"栅格预处理失败：{exc}"})
    return rec


def _preprocess_multidim(item: dict, out_dir: Path) -> dict:
    path = Path(str(item.get("path") or ""))
    role_inf = item.get("role_inference") or {}
    rec = {
        "name": item.get("original_name") or item.get("name") or path.name,
        "input_path": str(path),
        "role": role_inf.get("role") or "multidimensional_covariate",
        "matched_covariate": role_inf.get("matched_covariate"),
        "status": "pending_conversion",
        "ready": False,
        "message": "NC/HDF 数据已识别，系统将尝试自动提取可用变量并转换为 GeoTIFF。",
    }
    if convert_multidimensional_to_tif is None:
        rec["message"] = "NC/HDF 数据已识别，但当前转换组件不可用。"
        return rec
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        converted = convert_multidimensional_to_tif(path, out_dir, requested_year=None, data_name=rec.get("matched_covariate") or path.stem)
        if converted and converted.get("ok") and converted.get("display_tif"):
            tif_path = str(converted.get("display_tif"))
            rec.update({
                "status": "ready",
                "ready": True,
                "can_enter_model": True,
                "standardized_path": tif_path,
                "converted_tif": tif_path,
                "conversion": converted,
                "message": "NC/HDF 已自动转换为 GeoTIFF，可作为环境协变量继续检查和入模。",
            })
        else:
            rec.update({
                "status": "pending_conversion",
                "ready": False,
                "conversion": converted,
                "message": "NC/HDF 自动转换未完成，需要用户确认变量、时间或数据维度。" + (" 原因：" + str((converted or {}).get("error")) if isinstance(converted, dict) and converted.get("error") else ""),
            })
    except Exception as exc:
        rec.update({"status": "failed", "ready": False, "message": f"NC/HDF 自动转换失败：{exc}"})
    return rec



def _float_close(a: float, b: float, tol: float = 1e-8) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def _transform_close(a, b, tol: float = 1e-8) -> bool:
    try:
        aa = tuple(float(x) for x in a[:6])
        bb = tuple(float(x) for x in b[:6])
        return all(_float_close(x, y, tol) for x, y in zip(aa, bb))
    except Exception:
        return False


def _grid_matches_src_ref(src, ref) -> bool:
    """Strict grid match: same CRS, shape and affine transform.

    In this application “中心对齐” means that the pixel-center grid is identical:
    the same projection, pixel size, raster size and origin/transform. This is
    stronger than visual overlap and is the safe precondition for stacking model
    covariates cell-by-cell.
    """
    try:
        same_crs = bool(src.crs and ref.crs and src.crs == ref.crs)
        same_shape = int(src.width) == int(ref.width) and int(src.height) == int(ref.height)
        same_transform = _transform_close(src.transform, ref.transform)
        return bool(same_crs and same_shape and same_transform)
    except Exception:
        return False


def _alignment_action(initial_crs_match: bool, initial_grid_match: bool) -> str:
    if initial_crs_match and initial_grid_match:
        return "none"
    if not initial_crs_match and not initial_grid_match:
        return "reprojected_and_aligned"
    if not initial_crs_match:
        return "reprojected"
    return "aligned"


def _action_cn(action: str) -> str:
    return {
        "none": "无需处理",
        "reference": "对齐基准",
        "aligned": "自动对齐",
        "reprojected": "自动统一坐标系",
        "reprojected_and_aligned": "自动统一坐标系并对齐网格",
    }.get(str(action or ""), str(action or "空间处理"))


def _align_covariate_records(cov_records: list[dict], cov_dir: Path) -> dict:
    """Auto-unify CRS and pixel grid for uploaded model covariate rasters.

    The first ready covariate becomes the reference grid. Later rasters are
    reprojected/resampled onto that reference grid when CRS, shape or affine
    transform differ. Records are updated in-place so downstream preview and
    modeling can use the standardized/aligned GeoTIFF.
    """
    summary = {"reference": None, "checked": 0, "actions": [], "failed": []}
    if rasterio is None or reproject is None or Resampling is None:
        return summary

    ready = [r for r in cov_records if r.get("ready") and r.get("standardized_path")]
    if len(ready) <= 0:
        return summary

    ref_rec = None
    ref_src_ctx = None
    try:
        for rec in ready:
            p = Path(str(rec.get("standardized_path") or rec.get("input_path") or ""))
            if not p.exists():
                continue
            src = rasterio.open(p)
            if src.crs:
                ref_rec = rec
                ref_src_ctx = src
                break
            src.close()
        if ref_rec is None or ref_src_ctx is None:
            return summary

        ref = ref_src_ctx
        ref_name = str(ref_rec.get("name") or Path(ref_rec.get("input_path") or "").name)
        ref_cov = str(ref_rec.get("matched_covariate") or "")
        summary["reference"] = ref_name
        ref_rec["alignment"] = {
            "status": "reference",
            "action": "reference",
            "reference_name": ref_name,
            "message": f"作为本批环境协变量的空间对齐基准：{ref_name}。",
        }
        ref_profile = ref.profile.copy()
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = int(ref.width)
        ref_height = int(ref.height)

        for rec in ready:
            if rec is ref_rec:
                continue
            summary["checked"] += 1
            in_path = Path(str(rec.get("standardized_path") or rec.get("input_path") or ""))
            if not in_path.exists():
                continue
            name = str(rec.get("name") or in_path.name)
            cov = str(rec.get("matched_covariate") or in_path.stem)
            try:
                with rasterio.open(in_path) as src:
                    initial_crs_match = bool(src.crs and ref_crs and src.crs == ref_crs)
                    initial_grid_match = _grid_matches_src_ref(src, ref)
                    action = _alignment_action(initial_crs_match, initial_grid_match)
                    if action == "none":
                        rec["alignment"] = {
                            "status": "ok",
                            "action": "none",
                            "reference_name": ref_name,
                            "initial_crs_match": True,
                            "initial_grid_match": True,
                            "message": f"空间检查通过：{name} 与基准栅格坐标系和像元中心网格一致。",
                        }
                        continue

                    dst_dtype = src.dtypes[0] if str(rec.get("role") or "") == "covariate_categorical" or cov in {"CLCD", "LULCcd"} else "float32"
                    dst_nodata = src.nodata
                    if dst_nodata is None:
                        dst_nodata = 0 if dst_dtype.startswith("uint") else -9999.0
                    out_name = _safe_stem(cov or in_path.stem) + "_aligned.tif"
                    out_path = cov_dir / out_name
                    profile = ref_profile.copy()
                    profile.update({
                        "driver": "GTiff",
                        "height": ref_height,
                        "width": ref_width,
                        "count": 1,
                        "crs": ref_crs,
                        "transform": ref_transform,
                        "dtype": dst_dtype,
                        "nodata": dst_nodata,
                        "compress": "deflate",
                    })
                    resampling = Resampling.nearest if (str(rec.get("role") or "") == "covariate_categorical" or cov in {"CLCD", "LULCcd"}) else Resampling.bilinear
                    with rasterio.open(out_path, "w", **profile) as dst:
                        reproject(
                            source=rasterio.band(src, 1),
                            destination=rasterio.band(dst, 1),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            src_nodata=src.nodata,
                            dst_transform=ref_transform,
                            dst_crs=ref_crs,
                            dst_nodata=dst_nodata,
                            resampling=resampling,
                        )
                    rec["original_standardized_path"] = str(in_path)
                    rec["standardized_path"] = str(out_path)
                    rec["aligned_path"] = str(out_path)
                    rec["crs"] = str(ref_crs) if ref_crs else None
                    rec["width"] = ref_width
                    rec["height"] = ref_height
                    rec["resolution"] = [float(abs(ref.res[0])), float(abs(ref.res[1]))]
                    rec["bounds"] = [float(ref.bounds.left), float(ref.bounds.bottom), float(ref.bounds.right), float(ref.bounds.top)]
                    rec["ready"] = True
                    rec["status"] = "ready"
                    rec["alignment"] = {
                        "status": "success",
                        "action": action,
                        "reference_name": ref_name,
                        "initial_crs_match": initial_crs_match,
                        "initial_grid_match": initial_grid_match,
                        "output_path": str(out_path),
                        "message": f"{_action_cn(action)}成功：{name} 已统一到 {ref_name} 的坐标系、分辨率、范围和像元中心网格。",
                    }
                    summary["actions"].append({"name": name, "covariate": cov, "action": action, "status": "success"})
            except Exception as exc:
                rec["ready"] = False
                rec["status"] = "warning"
                rec["alignment"] = {
                    "status": "failed",
                    "action": "align_failed",
                    "reference_name": ref_name,
                    "message": f"自动空间对齐失败：{name} 未能统一到 {ref_name}。原因：{exc}",
                }
                summary["failed"].append({"name": name, "error": str(exc)})
    finally:
        try:
            if ref_src_ctx is not None:
                ref_src_ctx.close()
        except Exception:
            pass
    return summary

def run_upload_preprocessing(session_id: str, uploaded_files: list[dict], role_report: dict | None = None, covariate_plan: dict | None = None) -> dict:
    out_root = _root() / "preprocess" / str(session_id)
    sample_dir = out_root / "samples"
    cov_dir = out_root / "standardized_covariates"
    other_dir = out_root / "other"
    for d in (sample_dir, cov_dir, other_dir):
        d.mkdir(parents=True, exist_ok=True)

    sample_records = []
    cov_records = []
    pending_records = []
    excluded_records = []

    for item in uploaded_files or []:
        role = ((item.get("role_inference") or {}).get("role") or "").lower()
        path = Path(str(item.get("path") or ""))
        if not path.exists():
            excluded_records.append({"name": item.get("name"), "status": "missing_file", "ready": False, "message": "上传文件不存在或路径失效。"})
            continue
        if role == "sample_points":
            sample_records.append(_preprocess_sample(item, sample_dir))
        elif role in {"covariate_raster", "covariate_categorical"}:
            cov_records.append(_preprocess_raster(item, cov_dir))
        elif role == "multidimensional_covariate":
            rec = _preprocess_multidim(item, cov_dir)
            if rec.get("ready") and rec.get("standardized_path"):
                rec["role"] = "covariate_raster"
                cov_records.append(rec)
            else:
                pending_records.append(rec)
        else:
            # Keep copies of spatial non-model data for visibility/reference, but do not model.
            rec = {"name": item.get("original_name") or item.get("name") or path.name, "input_path": str(path), "role": role or "unknown", "status": "excluded", "ready": False, "message": "不作为模型协变量；可显示或作为参考。"}
            excluded_records.append(rec)

    spatial_alignment_summary = _align_covariate_records(cov_records, cov_dir)

    ready_covariates = []
    for r in cov_records:
        if r.get("ready") and r.get("matched_covariate"):
            ready_covariates.append(str(r.get("matched_covariate")))
    has_dem = "DEM" in ready_covariates
    derived_ready = sorted(list(DERIVED_FROM_DEM)) if has_dem else []
    sample_ready = any(r.get("ready") for r in sample_records)
    cov_ready_count = sum(1 for r in cov_records if r.get("ready"))
    can_baseline = bool(sample_ready and cov_ready_count > 0)

    report = {
        "session_id": session_id,
        "status": "ready_for_baseline" if can_baseline else "needs_data",
        "preprocess_root": str(out_root),
        "cleaned_sample_path": next((r.get("cleaned_path") for r in sample_records if r.get("cleaned_path")), None),
        "standardized_covariate_dir": str(cov_dir),
        "sample_records": sample_records,
        "covariate_records": cov_records,
        "spatial_alignment": spatial_alignment_summary,
        "pending_conversion_records": pending_records,
        "excluded_records": excluded_records,
        "ready_covariates": sorted(set(ready_covariates + derived_ready)),
        "derived_from_dem": derived_ready,
        "missing_default_covariates": [c for c in DEFAULT_COVARIATES if c not in set(ready_covariates + derived_ready)],
        "can_run_user_only_baseline": can_baseline,
        "gate": {
            "has_training_sample": sample_ready,
            "ready_covariate_count": cov_ready_count,
            "pending_conversion_count": len(pending_records),
            "message": "可以使用用户上传数据做一次基线制图。" if can_baseline else "当前尚不满足直接制图：需要至少一个训练样点文件和至少一个可用全域协变量栅格。",
        },
    }
    paths = {
        "preprocess_report_json": str(out_root / "preprocess_report.json"),
        "session_state_json": str(out_root / "session_state.json"),
    }
    (out_root / "preprocess_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # session_state is a compact machine-readable handoff for context-aware chat.
    session_state = {
        "session_id": session_id,
        "uploaded_summary": (role_report or {}).get("summary") or {},
        "covariate_plan": covariate_plan or {},
        "preprocess_gate": report.get("gate"),
        "ready_covariates": report.get("ready_covariates"),
        "missing_default_covariates": report.get("missing_default_covariates"),
        "preprocess_paths": paths,
    }
    (out_root / "session_state.json").write_text(json.dumps(session_state, ensure_ascii=False, indent=2), encoding="utf-8")
    report["paths"] = paths
    return report


def summarize_preprocess_for_user(report: dict | None) -> str:
    if not report:
        return "数据预处理尚未执行。"
    gate = report.get("gate") or {}
    ready = report.get("ready_covariates") or []
    missing = report.get("missing_default_covariates") or []
    return (
        f"预处理状态：{report.get('status')}\n"
        f"训练样点：{'已就绪' if gate.get('has_training_sample') else '缺失/未通过'}；"
        f"可用协变量：{gate.get('ready_covariate_count', 0)} 个；待转换：{gate.get('pending_conversion_count', 0)} 个。\n"
        f"已就绪协变量：{('、'.join(ready) if ready else '暂无')}\n"
        f"缺失默认协变量：{('、'.join(missing) if missing else '无')}\n"
        f"结论：{gate.get('message') or ''}"
    )
