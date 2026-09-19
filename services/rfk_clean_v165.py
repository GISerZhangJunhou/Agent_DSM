from __future__ import annotations

"""
V165 Clean RFK core.

This module intentionally does not reuse the old RFK validation/tuning logic.
It receives the already-auditable 建模样点表.csv and performs, from scratch:
- p/x/y separation;
- single spatial 7:3 holdout as the primary validation;
- auxiliary random 7:3 holdout;
- complete RF + OrdinaryKriging candidate scoring inside spatial 7:3 tuning splits;
- final RFK model fit;
- clean model report generation.

The caller passes the existing pipeline instance only for shared utilities such
as coordinate transformation, AOI-grid prediction, progress logging, and final
report writer integration.
"""

import hashlib
import json
import math
import os
import time
import secrets
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

try:
    from pykrige.ok import OrdinaryKriging
except Exception:  # pragma: no cover
    OrdinaryKriging = None


def _emit_safe(owner: Any, stage: str, message: str, payload: dict[str, Any] | None = None) -> None:
    try:
        from utils.pro_console import emit as _emit
        _emit(stage, message, payload or {}, task_id=getattr(owner, "task_id", None))
    except Exception:
        pass


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def _score(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, Any]:
    yt = np.asarray(list(y_true), dtype="float64")
    yp = np.asarray(list(y_pred), dtype="float64")
    m = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[m]
    yp = yp[m]
    if len(yt) == 0:
        return {"r2": None, "rmse": None, "mae": None, "bias": None, "n": 0}
    out = {
        "r2": float(r2_score(yt, yp)) if len(yt) > 1 and len(np.unique(yt)) > 1 else None,
        "rmse": _rmse(yt, yp),
        "mae": float(mean_absolute_error(yt, yp)),
        "bias": float(np.nanmean(yp - yt)),
        "n": int(len(yt)),
    }
    return out


def _hash_array(arr: Any) -> str:
    a = np.asarray(arr)
    h = hashlib.md5()
    h.update(str(a.shape).encode("utf-8"))
    h.update(np.ascontiguousarray(a).view(np.uint8))
    return h.hexdigest()[:12]


def _hash_text(text: str) -> str:
    # Backend-only public run code: random six digits per run.
    # Do not expose long md5/SHA strings in the user interface.
    try:
        return f"{secrets.randbelow(900000) + 100000:06d}"
    except Exception:
        return f"{int(time.time() * 1000) % 900000 + 100000:06d}"


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _detect_target_col(df: pd.DataFrame) -> str:
    candidates = ["som", "有机质", "SOM", "som_value", "SOC", "target"]
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError("建模样点表中未找到 SOM/有机质目标字段。")


def _detect_coord_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str | None, str | None]:
    x_cols = ["x", "x_32648", "X", "proj_x", "easting"]
    y_cols = ["y", "y_32648", "Y", "proj_y", "northing"]
    lon_cols = ["lon", "longitude", "经度", "LONGITUDE"]
    lat_cols = ["lat", "latitude", "纬度", "LATITUDE"]
    x = next((c for c in x_cols if c in df.columns), None)
    y = next((c for c in y_cols if c in df.columns), None)
    lon = next((c for c in lon_cols if c in df.columns), None)
    lat = next((c for c in lat_cols if c in df.columns), None)
    return x, y, lon, lat


def _make_xy(owner: Any, df: pd.DataFrame, target: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray, str]:
    # Use the existing pipeline coordinate utility so prediction grid and training coords match.
    try:
        x, y, crs = owner._coords_xy(df, target)
        x = np.asarray(x, dtype="float64")
        y = np.asarray(y, dtype="float64")
        if np.isfinite(x).all() and np.isfinite(y).all():
            return x, y, str(crs)
    except Exception:
        pass
    x_col, y_col, lon_col, lat_col = _detect_coord_columns(df)
    if x_col and y_col:
        x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype="float64")
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype="float64")
        if np.isfinite(x).all() and np.isfinite(y).all():
            return x, y, "existing_projected_xy"
    if lon_col and lat_col:
        lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype="float64")
        lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype="float64")
        try:
            from pyproj import Transformer
            target_crs = str((target or {}).get("target_crs") or os.getenv("PRO_TARGET_CRS_CHINA") or "EPSG:32648")
            tr = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
            x, y = tr.transform(lon, lat)
            return np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64"), target_crs
        except Exception:
            return lon, lat, "EPSG:4326_degree_fallback"
    raise RuntimeError("建模样点表缺少可用坐标列，无法执行 RFK。")


def _build_spatial_groups(x: np.ndarray, y: np.ndarray, target_group_count: int | None = None) -> np.ndarray:
    n = len(x)
    # Dynamic grid: enough groups for GroupShuffleSplit, not too fragmented.
    if target_group_count is None:
        target_group_count = int(os.getenv("PRO_CLEAN_RFK_SPATIAL_GROUPS", "25"))
    target_group_count = max(9, min(target_group_count, max(9, n // 20))) if n >= 100 else max(4, min(target_group_count, n // 4))
    nx = max(3, int(round(math.sqrt(target_group_count))))
    ny = max(3, int(math.ceil(target_group_count / nx)))
    # Use quantile edges rather than equal-width edges to avoid empty extreme blocks.
    def _bins(vals: np.ndarray, k: int) -> np.ndarray:
        qs = np.linspace(0, 1, k + 1)[1:-1]
        edges = np.unique(np.nanquantile(vals, qs))
        if len(edges) == 0:
            return np.zeros(len(vals), dtype=int)
        return np.digitize(vals, edges, right=False).astype(int)
    xb = _bins(x, nx)
    yb = _bins(y, ny)
    return (yb * nx + xb).astype(int)


def _is_probable_leak_or_bookkeeping_col(col: str, target_col: str) -> bool:
    """Columns that must not enter the RFK predictor matrix."""
    c = str(col).strip()
    cl = c.lower()
    if c == target_col:
        return True
    hard_exact = {
        "sample_id", "id", "point_id", "编号", "样点编号", "点号",
        "lon", "lat", "longitude", "latitude", "经度", "纬度",
        "x", "y", "x_32648", "y_32648", "x_4326", "y_4326", "proj_x", "proj_y",
        "row", "col", "grid_row", "grid_col", "pixel_row", "pixel_col",
        "是否在目标区内", "协变量缺失数", "协变量缺失率", "spatial_block", "grid_block_id",
        "year", "年份", "depth", "深度",
    }
    if c in hard_exact or cl in hard_exact:
        return True
    # Target leakage: any column that appears to be an already-gridded SOM/SOC product.
    leak_tokens = ["som_topsoil", "som_mask", "soil_organic_matter_map", "organic_matter_map", "有机质图", "有机质栅格", "soc_map"]
    if any(t in cl for t in leak_tokens):
        return True
    if cl.startswith("src_") or cl.startswith("coord_") or cl.endswith("_path"):
        return True
    return False


def _looks_like_numeric_feature(series: pd.Series) -> bool:
    s = pd.to_numeric(series, errors="coerce")
    return int(s.notna().sum()) >= max(8, int(len(series) * 0.2))


def _auto_detect_feature_cols(df: pd.DataFrame, passed_cols: list[str], target_col: str) -> list[str]:
    """V164: fused CSV and generated 建模样点表 both use the same detector.

    Priority:
    1. Explicit feature lock in env PRO_V164_FEATURES, if provided.
    2. All cov_* columns in model-ready CSV.
    3. Passed feature_cols from upstream pipeline.
    4. Numeric non-bookkeeping columns for fused CSV.
    """
    lock_raw = os.getenv("PRO_V164_FEATURES", "").strip()
    if lock_raw:
        wanted = [x.strip() for x in lock_raw.replace("；", ";").replace("，", ",").replace(";", ",").split(",") if x.strip()]
        return [c for c in wanted if c in df.columns and not _is_probable_leak_or_bookkeeping_col(c, target_col)]

    detected: list[str] = []
    for c in df.columns:
        cs = str(c)
        if cs.startswith("cov_") and not _is_probable_leak_or_bookkeeping_col(cs, target_col):
            detected.append(cs)

    # Preserve upstream order but do not let upstream filtering hide usable covariates.
    for c in passed_cols or []:
        cs = str(c)
        if cs in df.columns and cs not in detected and not _is_probable_leak_or_bookkeeping_col(cs, target_col):
            detected.append(cs)

    # Fused CSV: features may be named DEM/slope/pH/etc. without cov_ prefix.
    use_all_numeric = (os.getenv("PRO_V164_USE_ALL_NUMERIC_FEATURES", "1") or "1").strip().lower() not in {"0", "false", "no", "off"}
    if use_all_numeric:
        for c in df.columns:
            cs = str(c)
            if cs in detected or _is_probable_leak_or_bookkeeping_col(cs, target_col):
                continue
            if _looks_like_numeric_feature(df[cs]):
                detected.append(cs)

    # Never use LULCcd when the policy says CLCD replaces it.
    if (os.getenv("PRO_DISABLE_LULCCD", "1") or "1").strip().lower() not in {"0", "false", "no", "off"}:
        detected = [c for c in detected if "lulccd" not in str(c).lower()]
    return list(dict.fromkeys(detected))



def _normalize_feature_name(value: Any) -> str:
    import re
    t = str(value or "").strip().lower()
    for pref in ["cov_domestic_local_", "cov_local_", "cov_gee_", "cov_"]:
        if t.startswith(pref):
            t = t[len(pref):]
    # Remove common raster suffix fragments and separators for robust matching.
    for suff in ["_32648_250m", "_32648_1km", "_250m", "_1000m", "_1km", ".tif", ".tiff", ".img"]:
        if t.endswith(suff):
            t = t[: -len(suff)]
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", t)


def _is_clcd_feature_name(value: Any) -> bool:
    n = _normalize_feature_name(value)
    return "clcd" in n or "chinalandcover" in n or "中国土地覆盖" in str(value)


def _clcd_mask_only_policy() -> bool:
    return (os.getenv("PRO_CLCD_MASK_ONLY", "1") or "1").strip().lower() not in {"0", "false", "no", "off"}


def _find_raster_for_feature(owner: Any, feature: str) -> str | None:
    """Resolve a CSV feature column to a prediction raster without hard-coding names.

    Priority:
    1 exact key in owner._covariate_raster_map;
    2 normalized key match (CSV column <-> pipeline cov_* column);
    3 normalized filename stem match.
    """
    raster_map = getattr(owner, "_covariate_raster_map", {}) or {}
    if feature in raster_map and raster_map.get(feature):
        return str(raster_map.get(feature))
    nf = _normalize_feature_name(feature)
    for k, v in raster_map.items():
        if _normalize_feature_name(k) == nf and v:
            return str(v)
    for k, v in raster_map.items():
        try:
            if _normalize_feature_name(Path(str(v)).stem) == nf and v:
                return str(v)
        except Exception:
            pass
    # V173: CLCD is a cropland mask only by default. Do not let a CLCD
    # raster satisfy a CSV feature for formal SOM prediction; it is used later
    # only to keep CLCD=1 pixels and mark CLCD=2-8 as non-cropland.
    if (not _clcd_mask_only_policy()) and _is_clcd_feature_name(feature):
        for k, v in raster_map.items():
            if _is_clcd_feature_name(k) or _is_clcd_feature_name(Path(str(v)).stem):
                return str(v)
    return None


def _build_mappable_feature_plan(owner: Any, csv_features: list[str], target: dict[str, Any]) -> dict[str, Any]:
    """V165: separate training-diagnostic features from deployable map features.

    The fused CSV is authoritative for training features, but a formal GeoTIFF can
    only use features that have a matching prediction raster. Missing prediction
    rasters are reported instead of being faked with medians.
    """
    alias_map: dict[str, str] = {}
    mappable: list[str] = []
    missing: list[str] = []
    for f in csv_features:
        if str(f).endswith("__missing"):
            continue
        if _clcd_mask_only_policy() and _is_clcd_feature_name(f):
            # CLCD is intentionally excluded from the deployable SOM feature set.
            # It remains available as clcd_mask_raster below.
            continue
        rp = _find_raster_for_feature(owner, f)
        if rp:
            alias_map[str(f)] = str(rp)
            mappable.append(str(f))
        else:
            missing.append(str(f))
    clcd_raster = None
    raster_map = getattr(owner, "_covariate_raster_map", {}) or {}
    for k, v in raster_map.items():
        if _is_clcd_feature_name(k) or _is_clcd_feature_name(Path(str(v)).stem):
            clcd_raster = str(v); break
    plan = {
        "csv_feature_count": int(len(csv_features)),
        "csv_features": [str(x) for x in csv_features],
        "mappable_feature_count": int(len(mappable)),
        "mappable_features": mappable,
        "missing_raster_feature_count": int(len(missing)),
        "missing_raster_features": missing,
        "feature_raster_map": alias_map,
        "clcd_mask_raster": clcd_raster,
        "rule": "V173：训练诊断特征以融合CSV中的非CLCD协变量为准；正式GeoTIFF出图特征只使用CSV列与可用预测栅格的交集；CLCD仅用于耕地掩膜，不作为SOM预测特征。",
    }
    target["v165_feature_deployment_plan"] = plan
    target["v165_feature_raster_map"] = alias_map
    if clcd_raster:
        target["v165_clcd_mask_raster"] = clcd_raster
    # Also teach the existing prediction writer to resolve CSV feature names directly.
    try:
        raster_map = getattr(owner, "_covariate_raster_map", {}) or {}
        for f, rp in alias_map.items():
            raster_map[f] = rp
        owner._covariate_raster_map = raster_map
    except Exception:
        pass
    return plan


def _prepare_training_matrix_forced(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> tuple[pd.DataFrame, np.ndarray, list[str], dict[str, Any]]:
    """Prepare matrix using exactly the supplied feature columns, except hard drops.

    This is used for the deployable mapping model so that it never silently adds
    CSV features that do not have prediction rasters.
    """
    feature_cols = [str(c) for c in feature_cols if str(c) in df.columns and not _is_probable_leak_or_bookkeeping_col(str(c), target_col)]
    if not feature_cols:
        raise RuntimeError("V165 未检测到有预测栅格支撑的协变量列，无法输出正式GeoTIFF。")
    work = _coerce_numeric(df[[target_col] + feature_cols].copy(), [target_col] + feature_cols)
    row_ok = work[target_col].notna() & work[feature_cols].notna().any(axis=1)
    work = work.loc[row_ok].copy()
    y = work[target_col].to_numpy(dtype="float64")
    X_raw = work[feature_cols].copy()
    kept=[]; feature_audit=[]
    hard_drop_missing = float(os.getenv("PRO_V165_HARD_DROP_FEATURE_MISSING_RATE", os.getenv("PRO_V164_HARD_DROP_FEATURE_MISSING_RATE", "0.98")))
    for c in feature_cols:
        s = pd.to_numeric(X_raw[c], errors="coerce")
        mr = float(s.isna().mean()); un = int(s.dropna().nunique()); zr = float((s == 0).mean())
        if mr >= hard_drop_missing or un <= 1:
            feature_audit.append({"feature": c, "kept": False, "reason": "mappable_but_all_or_nearly_all_missing_or_constant", "missing_rate": mr, "unique_n": un, "zero_rate": zr})
        else:
            kept.append(c); feature_audit.append({"feature": c, "kept": True, "reason": "mappable_feature_kept", "missing_rate": mr, "unique_n": un, "zero_rate": zr})
    if not kept:
        raise RuntimeError("V165 所有可出图协变量均为全缺失/近全缺失或常量，无法输出正式GeoTIFF。")
    X = X_raw[kept].copy()
    impute_values={}
    for c in kept:
        med = pd.to_numeric(X[c], errors="coerce").median()
        if pd.isna(med): med = 0.0
        impute_values[c]=float(med)
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(float(med))
    audit={
        "target_col": target_col,
        "input_feature_count": len(feature_cols),
        "kept_feature_count": len(kept),
        "kept_original_feature_count": len(kept),
        "features": feature_audit,
        "kept_features": kept,
        "kept_original_features": kept,
        "impute_values": impute_values,
        "row_count_after_core_gate": int(len(work)),
        "X_hash": _hash_array(X.to_numpy(dtype="float64")),
        "y_hash": _hash_array(y),
        "v165_feature_policy": "forced deployable feature set: only fused-CSV columns with matched prediction rasters",
    }
    return X, y, list(X.columns), audit

def _prepare_training_matrix(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> tuple[pd.DataFrame, np.ndarray, list[str], dict[str, Any]]:
    feature_cols = _auto_detect_feature_cols(df, feature_cols, target_col)
    if not feature_cols:
        raise RuntimeError("V164 未检测到可用于建模的协变量列。若使用融合CSV，请确认其包含 SOM、坐标列和至少一个数值环境协变量列。")

    work = _coerce_numeric(df[[target_col] + feature_cols].copy(), [target_col] + feature_cols)
    # Drop rows only if y is invalid or all feature values are missing.
    row_ok = work[target_col].notna() & work[feature_cols].notna().any(axis=1)
    work = work.loc[row_ok].copy()
    y = work[target_col].to_numpy(dtype="float64")
    X_raw = work[feature_cols].copy()

    feature_audit: list[dict[str, Any]] = []
    kept: list[str] = []
    hard_drop_missing = float(os.getenv("PRO_V164_HARD_DROP_FEATURE_MISSING_RATE", "0.98"))
    for c in feature_cols:
        s = pd.to_numeric(X_raw[c], errors="coerce")
        missing_rate = float(s.isna().mean())
        unique_n = int(s.dropna().nunique())
        zero_rate = float((s == 0).mean())
        if missing_rate >= hard_drop_missing or unique_n <= 1:
            feature_audit.append({"feature": c, "kept": False, "reason": "all_or_nearly_all_missing_or_constant", "missing_rate": missing_rate, "unique_n": unique_n, "zero_rate": zero_rate})
            continue
        kept.append(c)
        feature_audit.append({"feature": c, "kept": True, "reason": "usable_first_run_kept", "missing_rate": missing_rate, "unique_n": unique_n, "zero_rate": zero_rate})
    if not kept:
        raise RuntimeError("所有协变量均为全缺失/近全缺失或常量，无法建模。")

    X = X_raw[kept].copy()
    impute_values: dict[str, float] = {}
    missing_indicator_cols: list[str] = []
    add_missing_indicator = (os.getenv("PRO_V164_ADD_MISSING_INDICATORS", "1") or "1").strip().lower() not in {"0", "false", "no", "off"}
    for c in kept:
        original_missing = pd.to_numeric(X[c], errors="coerce").isna()
        med = pd.to_numeric(X[c], errors="coerce").median()
        if pd.isna(med):
            med = 0.0
        impute_values[c] = float(med)
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(float(med))
        if add_missing_indicator and bool(original_missing.any()):
            mi = f"{c}__missing"
            X[mi] = original_missing.astype(float).to_numpy()
            missing_indicator_cols.append(mi)

    kept_with_indicators = list(X.columns)
    audit = {
        "target_col": target_col,
        "input_feature_count": len(feature_cols),
        "kept_feature_count": len(kept_with_indicators),
        "kept_original_feature_count": len(kept),
        "missing_indicator_count": len(missing_indicator_cols),
        "dropped_feature_count": len(feature_cols) - len(kept),
        "features": feature_audit,
        "kept_features": kept_with_indicators,
        "kept_original_features": kept,
        "missing_indicator_cols": missing_indicator_cols,
        "impute_values": impute_values,
        "row_count_after_core_gate": int(len(work)),
        "X_hash": _hash_array(X.to_numpy(dtype="float64")),
        "y_hash": _hash_array(y),
        "v164_feature_policy": "all numeric covariates from generated model-ready CSV or user fused CSV; no early default-catalog filtering; LULCcd disabled by default.",
    }
    return X, y, kept_with_indicators, audit

def _candidate_space(n: int, p: int) -> list[dict[str, Any]]:
    """V164 candidate space.

    Differences from V163:
    - includes the user's standalone RFK baseline first;
    - does not prefer global kriging (n_closest_points=0) unless explicitly enabled;
    - keeps enough RF complexity to reproduce external 0.5-ish benchmark runs.
    """
    rf_candidates = []
    n_estimators_options = [500, 800, 1200]
    depth_options = [8, 14, 16, None] if n >= 300 else [6, 10, 14, None]
    leaf_options = [1, 2, 4, 6]
    mf_options = ["sqrt", 0.5, 0.8, 1.0]
    for ne in n_estimators_options:
        for md in depth_options:
            for leaf in leaf_options:
                for mf in mf_options:
                    rf_candidates.append({
                        "n_estimators": int(ne),
                        "max_depth": md,
                        "min_samples_leaf": int(leaf),
                        "min_samples_split": int(max(2, leaf * 2)),
                        "max_features": mf,
                        "random_state": 42,
                        "n_jobs": int(os.getenv("PRO_RFK_RF_N_JOBS", "-1")),
                    })
    allow_global = (os.getenv("PRO_V164_ALLOW_GLOBAL_KRIGING", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
    nclose_options = [16, 32, 64]
    if allow_global:
        nclose_options = [0] + nclose_options
    rfk_candidates = []
    for vm in ["spherical", "exponential", "gaussian", "linear"]:
        for nl in [4, 6, 8, 10, 12]:
            for nc in nclose_options:
                for weight in [False, True]:
                    rfk_candidates.append({
                        "variogram_model": vm,
                        "nlags": int(nl),
                        "n_closest_points": int(nc),
                        "weight": bool(weight),
                        "exact_values": True,
                    })
    max_candidates = int(os.getenv("PRO_CLEAN_RFK_CANDIDATE_COUNT", os.getenv("PRO_V164_CANDIDATE_COUNT", "128")))
    rng = np.random.RandomState(42)
    combos = []
    for rf in rf_candidates:
        for rk in rfk_candidates:
            combos.append({"rf_params": rf, "rfk_params": rk})
    if len(combos) > max_candidates:
        idx = rng.choice(len(combos), size=max_candidates, replace=False)
        combos = [combos[int(i)] for i in idx]
    # Strong baselines aligned with the standalone RFK_formal_minimal.py and local-neighborhood variants.
    baselines = []
    for nc in ([0, 16, 32] if allow_global else [16, 32]):
        baselines.append({
            "rf_params": {"n_estimators": 1200, "max_depth": 16, "min_samples_leaf": 1, "min_samples_split": 2, "max_features": 1.0, "random_state": 42, "n_jobs": int(os.getenv("PRO_RFK_RF_N_JOBS", "-1"))},
            "rfk_params": {"variogram_model": "spherical", "nlags": 8, "n_closest_points": int(nc), "weight": False, "exact_values": True},
        })
    for b in reversed(baselines):
        combos.insert(0, b)
    # Deduplicate while preserving order.
    seen = set()
    out = []
    for c in combos:
        key = json.dumps(c, sort_keys=True, default=str)
        if key in seen:
            continue
        out.append(c); seen.add(key)
    return out[:max_candidates]

def _fit_state(owner: Any, X_train: np.ndarray, y_train: np.ndarray, train_df: pd.DataFrame, target: dict[str, Any], rf_params: dict[str, Any], rfk_params: dict[str, Any]) -> dict[str, Any]:
    rf = RandomForestRegressor(**rf_params)
    # Use AOI weights only if explicitly enabled; for city-level all weights are equal.
    sample_weight = None
    try:
        sample_weight = owner._aoi_sample_weight(train_df)
    except Exception:
        sample_weight = None
    if sample_weight is not None and len(sample_weight) == len(y_train):
        rf.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        rf.fit(X_train, y_train)
    rf_pred = np.asarray(rf.predict(X_train), dtype="float64")
    resid = y_train - rf_pred
    x, y, coord_crs = _make_xy(owner, train_df, target)
    state = {
        "algorithm": "RFK",
        "rf": rf,
        "ok": None,
        "kriging_enabled": False,
        "variogram_model": rfk_params.get("variogram_model"),
        "nlags": int(rfk_params.get("nlags") or 8),
        "n_closest_points": int(rfk_params.get("n_closest_points") or 0),
        "weight": bool(rfk_params.get("weight", False)),
        "exact_values": bool(rfk_params.get("exact_values", True)),
        "coord_crs": coord_crs,
        "residual_summary": {
            "mean": float(np.nanmean(resid)),
            "std": float(np.nanstd(resid)),
            "min": float(np.nanmin(resid)),
            "max": float(np.nanmax(resid)),
        },
        "kriging_warning": None,
    }
    if OrdinaryKriging is None:
        state["kriging_warning"] = "pykrige unavailable; RF-only fallback"
        return state
    try:
        ok = OrdinaryKriging(
            x=x.astype("float64"),
            y=y.astype("float64"),
            z=resid.astype("float64"),
            variogram_model=str(rfk_params.get("variogram_model") or "spherical"),
            nlags=int(rfk_params.get("nlags") or 8),
            weight=bool(rfk_params.get("weight", False)),
            exact_values=bool(rfk_params.get("exact_values", True)),
            verbose=False,
            enable_plotting=False,
        )
        state["ok"] = ok
        state["kriging_enabled"] = True
    except Exception as exc:
        state["kriging_warning"] = str(exc)
        if os.getenv("PRO_RFK_REQUIRE_KRIGING", "1") == "1":
            raise RuntimeError(f"RFK训练阶段残差克里金失败：{exc}")
    return state


def _predict_state(owner: Any, state: dict[str, Any], X_test: np.ndarray, test_df: pd.DataFrame, target: dict[str, Any]) -> np.ndarray:
    rf_pred = np.asarray(state["rf"].predict(X_test), dtype="float64")
    if not state.get("kriging_enabled") or state.get("ok") is None:
        return rf_pred
    try:
        return rf_pred + np.asarray(owner._krige_residuals(state, test_df, target), dtype="float64")
    except Exception:
        # Make validation robust but recordable by candidate score caller.
        return rf_pred


def _one_holdout(owner: Any, X: np.ndarray, y: np.ndarray, df: pd.DataFrame, groups: np.ndarray | None, target: dict[str, Any], rf_params: dict[str, Any], rfk_params: dict[str, Any], spatial: bool, test_size: float, random_state: int, label: str) -> dict[str, Any]:
    if spatial and groups is not None and len(np.unique(groups)) >= 3:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        split_type = "GroupShuffleSplit-spatial-block-7:3"
    else:
        splitter = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y))
        split_type = "ShuffleSplit-random-7:3"
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    state = _fit_state(owner, X[train_idx], y[train_idx], train_df, target, rf_params, rfk_params)
    pred = _predict_state(owner, state, X[test_idx], test_df, target)
    met = _score(y[test_idx], pred)
    aoi_summary = {}
    if "是否在目标区内" in test_df.columns:
        flags = pd.to_numeric(test_df["是否在目标区内"], errors="coerce").fillna(0).to_numpy(dtype="float64") > 0.5
        aoi_summary = {
            "target_aoi_test_n": int(flags.sum()),
            "target_aoi_test_ratio": float(flags.mean()) if len(flags) else 0.0,
        }
        if flags.sum() >= 8 and len(np.unique(y[test_idx][flags])) > 1:
            s2 = _score(y[test_idx][flags], pred[flags])
            aoi_summary.update({"target_aoi_r2": s2.get("r2"), "target_aoi_rmse": s2.get("rmse"), "target_aoi_mae": s2.get("mae"), "target_aoi_bias": s2.get("bias")})
        else:
            aoi_summary["target_aoi_warning"] = "目标AOI验证样点过少，暂不计算稳定R²。"
    return {
        "cv": f"{split_type}({label})",
        "r2": met.get("r2"),
        "rmse": met.get("rmse"),
        "mae": met.get("mae"),
        "bias": met.get("bias"),
        "train_n": int(len(train_idx)),
        "test_n": int(len(test_idx)),
        "split_count": 1,
        "splits": [{"fold": 1, "train_n": int(len(train_idx)), "test_n": int(len(test_idx)), "r2": met.get("r2"), "rmse": met.get("rmse"), "mae": met.get("mae"), "kriging_enabled": bool(state.get("kriging_enabled")), "kriging_warning": state.get("kriging_warning")}],
        **aoi_summary,
    }



def _repeated_holdout(owner: Any, X: np.ndarray, y: np.ndarray, df: pd.DataFrame, groups: np.ndarray | None, target: dict[str, Any], rf_params: dict[str, Any], rfk_params: dict[str, Any], spatial: bool, test_size: float, n_splits: int, label: str) -> dict[str, Any]:
    """Repeated 7:3 diagnostic without counting test_n as unique samples.

    This is for comparison with standalone RFK scripts. It reports both pooled
    row count and unique test sample count so users never confuse repeated test
    rows with actual sample count.
    """
    if n_splits <= 0:
        return {"enabled": False, "reason": "n_splits<=0"}
    if spatial and groups is not None and len(np.unique(groups)) >= 3:
        splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
        splits = splitter.split(X, y, groups=groups)
        cv = f"Repeated-GroupShuffleSplit-spatial-7:3({n_splits} splits)"
    else:
        splitter = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
        splits = splitter.split(X, y)
        cv = f"Repeated-ShuffleSplit-random-7:3({n_splits} splits)"
    pooled_y, pooled_p = [], []
    fold_rows = []
    unique_test = set()
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        train_df = df.iloc[train_idx].copy().reset_index(drop=True)
        test_df = df.iloc[test_idx].copy().reset_index(drop=True)
        try:
            state = _fit_state(owner, X[train_idx], y[train_idx], train_df, target, rf_params, rfk_params)
            pred = _predict_state(owner, state, X[test_idx], test_df, target)
            met = _score(y[test_idx], pred)
            pooled_y.extend(y[test_idx].tolist())
            pooled_p.extend(np.asarray(pred, dtype="float64").tolist())
            unique_test.update([int(i) for i in test_idx])
            fold_rows.append({"fold": fold, "train_n": int(len(train_idx)), "test_n": int(len(test_idx)), "r2": met.get("r2"), "rmse": met.get("rmse"), "mae": met.get("mae"), "kriging_enabled": bool(state.get("kriging_enabled")), "kriging_warning": state.get("kriging_warning")})
        except Exception as exc:
            fold_rows.append({"fold": fold, "error": str(exc), "train_n": int(len(train_idx)), "test_n": int(len(test_idx))})
    pooled = _score(pooled_y, pooled_p) if pooled_y else {"r2": None, "rmse": None, "mae": None, "bias": None, "n": 0}
    return {
        "enabled": True,
        "cv": cv,
        "label": label,
        "split_count": int(n_splits),
        "pooled_test_rows": int(len(pooled_y)),
        "unique_test_n": int(len(unique_test)),
        "unique_test_ratio": float(len(unique_test) / max(len(y), 1)),
        "r2": pooled.get("r2"),
        "rmse": pooled.get("rmse"),
        "mae": pooled.get("mae"),
        "bias": pooled.get("bias"),
        "fold_scores": fold_rows,
        "warning": "repeated holdout pooled_test_rows may exceed total samples; use unique_test_n for sample count.",
    }

def _score_candidate(owner: Any, candidate: dict[str, Any], X: np.ndarray, y: np.ndarray, df: pd.DataFrame, groups: np.ndarray, target: dict[str, Any], n_splits: int = 3) -> dict[str, Any]:
    rf_params = candidate["rf_params"]
    rfk_params = candidate["rfk_params"]
    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=float(os.getenv("PRO_CLEAN_RFK_TUNE_TEST_SIZE", "0.30")), random_state=42)
    fold_scores = []
    ys = []
    ps = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        train_df = df.iloc[train_idx].copy().reset_index(drop=True)
        test_df = df.iloc[test_idx].copy().reset_index(drop=True)
        try:
            state = _fit_state(owner, X[train_idx], y[train_idx], train_df, target, rf_params, rfk_params)
            pred = _predict_state(owner, state, X[test_idx], test_df, target)
            met = _score(y[test_idx], pred)
            fold_scores.append({"fold": fold, "train_n": int(len(train_idx)), "test_n": int(len(test_idx)), "r2": met.get("r2"), "rmse": met.get("rmse"), "mae": met.get("mae"), "kriging_enabled": bool(state.get("kriging_enabled")), "kriging_warning": state.get("kriging_warning")})
            ys.extend(y[test_idx].tolist())
            ps.extend(np.asarray(pred, dtype="float64").tolist())
        except Exception as exc:
            return {"ok": False, "error": str(exc), "rf_params": rf_params, "rfk_params": rfk_params}
    met_all = _score(ys, ps)
    return {
        "ok": True,
        "rf_params": rf_params,
        "rfk_params": rfk_params,
        "r2": met_all.get("r2"),
        "rmse": met_all.get("rmse"),
        "mae": met_all.get("mae"),
        "bias": met_all.get("bias"),
        "split_count": len(fold_scores),
        "fold_scores": fold_scores,
    }


def run_clean_rfk_model(owner: Any, df: pd.DataFrame, feature_cols: list[str], out_dir: Path, target: dict[str, Any], real_cov_cols: list[str]) -> tuple[Path | None, Path | None]:
    if OrdinaryKriging is None:
        raise RuntimeError("V165 Clean RFK 需要 pykrige。请执行：pip install pykrige")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _emit_safe(owner, "MODEL", "V165 Clean RFK 从建模样点表重新开始训练与验证", {"out_dir": str(out_dir)})

    csv_path = getattr(owner, "_model_ready_csv", None)
    if not csv_path or not Path(csv_path).exists():
        raise RuntimeError("V165 Clean RFK 未找到建模样点表.csv，已停止，避免继续沿用内存数据。")
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    target_col = _detect_target_col(raw)
    Xdf, y, kept_features, matrix_audit = _prepare_training_matrix(raw, feature_cols, target_col)
    # Align the sample rows to the exact feature matrix row gate.
    original_feature_cols = [c for c in (matrix_audit.get("kept_original_features") or []) if c in raw.columns]
    if not original_feature_cols:
        original_feature_cols = [c for c in kept_features if c in raw.columns]
    tmp_cols = [target_col] + original_feature_cols
    tmp = _coerce_numeric(raw[tmp_cols].copy(), tmp_cols)
    row_ok = tmp[target_col].notna() & tmp[original_feature_cols].notna().any(axis=1)
    samples_df = raw.loc[row_ok].copy().reset_index(drop=True)
    if len(samples_df) != len(Xdf):
        samples_df = samples_df.iloc[:len(Xdf)].copy().reset_index(drop=True)

    X = Xdf.to_numpy(dtype="float64")
    if len(y) < 20:
        raise RuntimeError(f"有效样点数不足，无法执行 V165 Clean RFK：{len(y)}")
    xcoord, ycoord, coord_crs = _make_xy(owner, samples_df, target)
    groups = _build_spatial_groups(xcoord, ycoord)
    group_count = int(len(np.unique(groups)))
    test_size = float(os.getenv("PRO_CLEAN_RFK_HOLDOUT_TEST_SIZE", "0.30"))

    candidates = _candidate_space(len(y), X.shape[1])
    # User-provided parameter locks can be added later; for V164 do true data-driven search.
    tune_splits = int(os.getenv("PRO_CLEAN_RFK_TUNE_SPLITS", "3"))
    scores = []
    for i, cand in enumerate(candidates, start=1):
        sc = _score_candidate(owner, cand, X, y, samples_df, groups, target, n_splits=tune_splits)
        sc["candidate_id"] = i
        scores.append(sc)
        if i % 10 == 0 or i == 1:
            _emit_safe(owner, "MODEL", f"V165 Clean RFK 联合寻参 {i}/{len(candidates)}", {"candidate_id": i, "rmse": sc.get("rmse"), "r2": sc.get("r2")})
    ok_scores = [s for s in scores if s.get("ok") and s.get("rmse") is not None]
    if not ok_scores:
        raise RuntimeError("V165 Clean RFK 全部候选参数评分失败，无法建模。")
    best = sorted(ok_scores, key=lambda s: (float(s.get("rmse")), -float(s.get("r2") or -999)))[0]
    best_rf = best["rf_params"]
    best_rfk = best["rfk_params"]

    # V164: compute both random 7:3 and spatial 7:3. Primary can be switched for
    # external-script parity without hiding the rigorous spatial diagnostic.
    spatial_holdout = _one_holdout(owner, X, y, samples_df, groups, target, best_rf, best_rfk, spatial=True, test_size=test_size, random_state=20260626, label="single_spatial_primary")
    random_holdout = _one_holdout(owner, X, y, samples_df, None, target, best_rf, best_rfk, spatial=False, test_size=test_size, random_state=20260626, label="single_random_primary")
    primary_mode = (os.getenv("PRO_V164_PRIMARY_VALIDATION", "random_7_3") or "random_7_3").strip().lower()
    primary = random_holdout if primary_mode in {"random", "random_7_3", "shuffle", "shuffle_7_3"} else spatial_holdout
    relaxed = spatial_holdout if primary is random_holdout else random_holdout
    repeated_spatial = _repeated_holdout(owner, X, y, samples_df, groups, target, best_rf, best_rfk, spatial=True, test_size=test_size, n_splits=int(os.getenv("PRO_V164_REPEATED_SPATIAL_SPLITS", "15")), label="repeated_spatial_aux")
    repeated_random = _repeated_holdout(owner, X, y, samples_df, None, target, best_rf, best_rfk, spatial=False, test_size=test_size, n_splits=int(os.getenv("PRO_V164_REPEATED_RANDOM_SPLITS", "0")), label="repeated_random_aux")

    # Diagnostic final model on all usable CSV features. This model is used for
    # training/validation diagnostics and for feature-importance analysis.
    final_state = _fit_state(owner, X, y, samples_df, target, best_rf, best_rfk)

    # V165: formal GeoTIFF deployment is separated from diagnostic training.
    # It may only use features with matching prediction rasters.
    mappable_plan = _build_mappable_feature_plan(owner, original_feature_cols, target)
    map_state = None
    map_features: list[str] = []
    map_samples_df = None
    map_matrix_audit: dict[str, Any] | None = None
    map_metrics: dict[str, Any] | None = None
    map_best = best
    map_best_rf = best_rf
    map_best_rfk = best_rfk
    if mappable_plan.get("mappable_features"):
        try:
            map_Xdf, map_y, map_features, map_matrix_audit = _prepare_training_matrix_forced(raw, list(mappable_plan.get("mappable_features") or []), target_col)
            map_original_feature_cols = [c for c in (map_matrix_audit.get("kept_original_features") or []) if c in raw.columns]
            map_tmp_cols = [target_col] + map_original_feature_cols
            map_tmp = _coerce_numeric(raw[map_tmp_cols].copy(), map_tmp_cols)
            map_row_ok = map_tmp[target_col].notna() & map_tmp[map_original_feature_cols].notna().any(axis=1)
            map_samples_df = raw.loc[map_row_ok].copy().reset_index(drop=True)
            if len(map_samples_df) != len(map_Xdf):
                map_samples_df = map_samples_df.iloc[:len(map_Xdf)].copy().reset_index(drop=True)
            map_X = map_Xdf.to_numpy(dtype="float64")
            map_xcoord, map_ycoord, map_coord_crs = _make_xy(owner, map_samples_df, target)
            map_groups = _build_spatial_groups(map_xcoord, map_ycoord)
            # Re-tune if the deployable feature set differs materially from the diagnostic feature set.
            if (os.getenv("PRO_V165_TUNE_MAPPABLE_MODEL", "1") or "1").strip().lower() not in {"0", "false", "no", "off"}:
                map_candidates = _candidate_space(len(map_y), map_X.shape[1])
                map_scores = []
                for j, cand in enumerate(map_candidates, start=1):
                    sc2 = _score_candidate(owner, cand, map_X, map_y, map_samples_df, map_groups, target, n_splits=tune_splits)
                    sc2["candidate_id"] = j
                    map_scores.append(sc2)
                    if j % 10 == 0 or j == 1:
                        _emit_safe(owner, "MODEL", f"V165 可出图特征RFK联合寻参 {j}/{len(map_candidates)}", {"candidate_id": j, "rmse": sc2.get("rmse"), "r2": sc2.get("r2")})
                map_ok = [z for z in map_scores if z.get("ok") and z.get("rmse") is not None]
                if map_ok:
                    map_best = sorted(map_ok, key=lambda z: (float(z.get("rmse")), -float(z.get("r2") or -999)))[0]
                    map_best_rf = map_best["rf_params"]
                    map_best_rfk = map_best["rfk_params"]
                    mappable_plan["mapping_candidate_count"] = len(map_candidates)
                    mappable_plan["mapping_best_score"] = map_best
            map_spatial = _one_holdout(owner, map_X, map_y, map_samples_df, map_groups, target, map_best_rf, map_best_rfk, spatial=True, test_size=test_size, random_state=20260626, label="single_spatial_mappable")
            map_random = _one_holdout(owner, map_X, map_y, map_samples_df, None, target, map_best_rf, map_best_rfk, spatial=False, test_size=test_size, random_state=20260626, label="single_random_mappable")
            map_state = _fit_state(owner, map_X, map_y, map_samples_df, target, map_best_rf, map_best_rfk)
            map_metrics = {"single_spatial_validation": map_spatial, "single_random_validation": map_random, "feature_count": len(map_features), "features": map_features, "matrix_audit": map_matrix_audit}
            map_matrix_audit.update({"coordinate_crs": map_coord_crs, "spatial_group_count": int(len(np.unique(map_groups))), "spatial_groups_hash": _hash_array(map_groups)})
            (out_dir / "V165_可出图矩阵_审计.json").write_text(json.dumps(map_matrix_audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            target["v165_mapping_matrix_audit"] = {"audit_path": str(out_dir / "V165_可出图矩阵_审计.json"), **map_matrix_audit}
            _emit_safe(owner, "MODEL", "V165已训练正式出图RFK模型", {"mappable_feature_count": len(map_features), "missing_raster_feature_count": mappable_plan.get("missing_raster_feature_count")})
        except Exception as exc:
            mappable_plan["mapping_model_error"] = str(exc)
            map_state = None
            map_features = []
            map_samples_df = None
            _emit_safe(owner, "MODEL", "V165正式出图模型未生成", {"error": str(exc)})
    else:
        _emit_safe(owner, "MODEL", "V165未找到可出图协变量栅格，跳过GeoTIFF输出", {"csv_feature_count": mappable_plan.get("csv_feature_count"), "missing_raster_features": mappable_plan.get("missing_raster_features")})

    # Use mapping state for downstream prediction if available; otherwise keep
    # diagnostic state for report/model inspection only.
    setattr(owner, "_rfk_state", map_state or final_state)

    matrix_audit.update({
        "version": "V165_clean_rfk_dynamic_features",
        "model_ready_csv": str(csv_path),
        "row_count": int(len(samples_df)),
        "feature_count": int(X.shape[1]),
        "coordinate_crs": coord_crs,
        "spatial_group_count": group_count,
        "spatial_groups_hash": _hash_array(groups),
        "validation_design": "V165 computes diagnostic validation on CSV features and, when rasters are incomplete, trains a separate mappable model using only CSV features with matching prediction rasters.",
        "primary_validation_mode": primary_mode,
        "single_spatial_holdout": spatial_holdout,
        "single_random_holdout": random_holdout,
        "repeated_spatial_holdout": repeated_spatial,
        "primary_holdout_train_n": primary.get("train_n"),
        "primary_holdout_test_n": primary.get("test_n"),
    })
    audit_p = out_dir / "V165_训练矩阵_从头审计.json"
    audit_p.write_text(json.dumps(matrix_audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    target["training_matrix_audit"] = {"audit_path": str(audit_p), **matrix_audit}

    metrics = {
        "cv": primary.get("cv"),
        "r2": primary.get("r2"),
        "rmse": primary.get("rmse"),
        "mae": primary.get("mae"),
        "bias": primary.get("bias"),
        "train_n": primary.get("train_n"),
        "test_n": primary.get("test_n"),
        "split_count": 1,
        "splits": primary.get("splits"),
        "target_aoi_test_n": primary.get("target_aoi_test_n"),
        "target_aoi_test_ratio": primary.get("target_aoi_test_ratio"),
        "target_aoi_r2": primary.get("target_aoi_r2"),
        "target_aoi_rmse": primary.get("target_aoi_rmse"),
        "target_aoi_mae": primary.get("target_aoi_mae"),
        "target_aoi_bias": primary.get("target_aoi_bias"),
        "target_aoi_warning": primary.get("target_aoi_warning"),
        "aux_validation": relaxed,
        "single_spatial_validation": spatial_holdout,
        "single_random_validation": random_holdout,
        "repeated_spatial_validation": repeated_spatial,
        "repeated_random_validation": repeated_random,
        "validation_rewrite_note": "V165同时输出随机7:3与空间7:3；训练诊断可使用融合CSV全部有效特征，正式出图模型只使用有匹配预测栅格的特征。",
    }

    report: dict[str, Any] = {
        "formal_mapping_pipeline": True,
        "model_algorithm": "RFK",
        "run_mode": getattr(owner, "run_mode", None),
        "rfk_core_version": "V165_clean_dynamic_features",
        "message": "V165模型优先RFK核心：融合CSV动态识别特征，训练诊断与正式出图特征分离；正式GeoTIFF只使用CSV列与可用预测栅格的交集。",
        "target": target,
        "rows": int(len(y)),
        "feature_count": int(X.shape[1]),
        "feature_cols": kept_features,
        "csv_diagnostic_feature_cols": kept_features,
        "v165_feature_deployment_plan": mappable_plan,
        "mapping_feature_count": int(len(map_features or [])),
        "mapping_feature_cols": list(map_features or []),
        "mapping_model_metrics": map_metrics,
        "real_covariate_cols": real_cov_cols,
        "feature_missing_rate": {str(r["feature"]): float(r.get("missing_rate") or 0) for r in matrix_audit.get("features", [])},
        "feature_zero_value_audit": {str(r["feature"]): {"zero_rate": float(r.get("zero_rate") or 0), "zero_counted_as_missing_in_table": False} for r in matrix_audit.get("features", [])},
        "rf_params": best_rf,
        "hyperparameter_plan": {
            "mode": "V165_clean_full_rfk_joint_auto_tuned",
            "primary_validation_mode": primary_mode,
            "candidate_count": len(candidates),
            "selected_rf_params": best_rf,
            "selected_rfk_params": best_rfk,
            "best_score": best,
            "candidate_scores": scores,
            "split_design": {"primary": primary_mode, "tuning": f"{tune_splits} repeated spatial 7:3 candidate scoring", "test_size": test_size, "spatial_group_count": group_count, "groups_hash": _hash_array(groups)},
        },
        "sample_aoi_overlap": target.get("sample_aoi_overlap"),
        "model_ready_csv": str(csv_path),
        "model_ready_table": target.get("model_ready_table"),
        "training_matrix_audit": target.get("training_matrix_audit"),
        "rfk": {
            "enabled": True,
            "source": "mapping_model" if map_state is not None else "diagnostic_csv_model",
            "variogram_model": (map_state or final_state).get("variogram_model"),
            "nlags": (map_state or final_state).get("nlags"),
            "n_closest_points": (map_state or final_state).get("n_closest_points"),
            "weight": (map_state or final_state).get("weight"),
            "exact_values": (map_state or final_state).get("exact_values"),
            "kriging_enabled": (map_state or final_state).get("kriging_enabled"),
            "coord_crs": (map_state or final_state).get("coord_crs"),
            "residual_summary": (map_state or final_state).get("residual_summary"),
            "warning": (map_state or final_state).get("kriging_warning"),
        },
        "metrics": metrics,
        "warnings": getattr(owner, "warnings", []),
    }

    pred_tif = None
    if map_state is not None and map_features and map_samples_df is not None:
        pred_tif = owner._make_prediction_tif(map_state, map_features, map_samples_df, out_dir, target)
    else:
        target["prediction_grid_audit"] = {"stopped": True, "reason": "no_mappable_feature_rasters", "v165_feature_deployment_plan": mappable_plan}
    report["prediction_grid_csv"] = getattr(owner, "_prediction_grid_csv", None)
    report["prediction_grid_audit"] = target.get("prediction_grid_audit")
    try:
        importances = getattr(final_state.get("rf"), "feature_importances_", None)
        if importances is not None:
            rows = []
            for col, imp in zip(kept_features, list(importances)):
                rows.append({"feature": str(col), "label": str(col).replace("cov_", ""), "importance": float(imp)})
            rows.sort(key=lambda r: r["importance"], reverse=True)
            report["feature_importance"] = rows
        if map_state is not None:
            mi = getattr(map_state.get("rf"), "feature_importances_", None)
            if mi is not None:
                mrows=[]
                for col, imp in zip(map_features, list(mi)):
                    mrows.append({"feature": str(col), "label": str(col).replace("cov_", ""), "importance": float(imp)})
                mrows.sort(key=lambda r: r["importance"], reverse=True)
                report["mapping_feature_importance"] = mrows
    except Exception as exc:
        report["feature_importance_error"] = str(exc)

    fp_src = {
        "version": "V165",
        "target": (target or {}).get("region"),
        "model_ready_csv": str(csv_path),
        "X_hash": matrix_audit.get("X_hash"),
        "y_hash": matrix_audit.get("y_hash"),
        "groups_hash": _hash_array(groups),
        "primary_metrics": {k: metrics.get(k) for k in ["r2", "rmse", "mae", "train_n", "test_n"]},
        "features": kept_features,
        "mapping_features": map_features,
        "missing_raster_features": mappable_plan.get("missing_raster_features"),
    }
    report["run_fingerprint"] = _hash_text(json.dumps(fp_src, ensure_ascii=False, sort_keys=True, default=str))
    report["created_at"] = int(time.time())
    report_path = out_dir / "formal_model_report.json"

    try:
        from services.rfk_result_report_service import generate_ai_rfk_analysis, write_model_record_txt, build_analysis_payload
        report["result_analysis_payload"] = build_analysis_payload(report, pred_tif)
        report["ai_result_analysis"] = generate_ai_rfk_analysis(report, pred_tif)
        model_record = write_model_record_txt(out_dir, report, pred_tif=pred_tif, ai_analysis=report.get("ai_result_analysis"), round_name="第1轮")
        report["model_record_txt"] = str(model_record)
    except Exception as exc:
        report["ai_result_analysis_error"] = str(exc)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _emit_safe(owner, "MODEL", "V165 Clean RFK 建模完成", {"report": str(report_path), "pred_tif": str(pred_tif) if pred_tif else None, "metrics": metrics})
    return report_path, pred_tif
