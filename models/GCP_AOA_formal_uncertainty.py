#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
成都市土壤有机质 地理共形预测（GCP）+ AOA 正式不确定性分析
================================================

功能：
1. 基于上一轮制图结果构建地理共形预测（Geographic Conformal Prediction, GCP）区间；
2. 基于训练样本支持程度计算适用域与不相似性指数；
3. 将预测区间宽度与适用域结果组合为四类风险分区；
4. 输出 GeoTIFF、CSV、JSON 和正式分析报告。

输入来自上一轮制图任务的 manifest、预测栅格和校准/验证样点表。
输出目录采用 config.paths 中的正式不确定性分析目录。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:
    cKDTree = None  # type: ignore

warnings.filterwarnings("ignore", category=UserWarning)

# Ensure the project root is importable when this script is launched from the
# models directory by the background task runner.
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from services.formal_map_png_service import create_formal_continuous_png, create_formal_class_png
except Exception:  # pragma: no cover
    create_formal_continuous_png = None
    create_formal_class_png = None

try:
    from config.paths import UNCERTAINTY_RESULT_ROOT, timestamp_name
except Exception:
    UNCERTAINTY_RESULT_ROOT = Path(r"E:\Agent_DSM\不确定性分析结果")

    def timestamp_name(suffix: str = "") -> str:
        base = datetime.now().strftime("%Y%m%d_%H%M")
        return f"{base}_{suffix}" if suffix else base


@dataclass(frozen=True)
class Config:
    alpha: float = float(os.getenv("AGENT_GCP_ALPHA", "0.10"))
    neighbors: int = int(os.getenv("AGENT_GCP_NEIGHBORS", "60"))
    bandwidth_scale: float = float(os.getenv("AGENT_GCP_BANDWIDTH_SCALE", "0.60"))
    min_bandwidth: float = float(os.getenv("AGENT_GCP_MIN_BANDWIDTH", "200.0"))
    ccb_bins: int = int(os.getenv("AGENT_GCP_CCB_BINS", "10"))
    aoa_k_train: int = int(os.getenv("AGENT_AOA_TRAIN_K", "2"))
    aoa_outlier_quantile: float = float(os.getenv("AGENT_AOA_OUTLIER_QUANTILE", "0.95"))
    width_threshold_quantile: float = float(os.getenv("AGENT_RISK_WIDTH_QUANTILE", "0.75"))
    nodata: float = -9999.0

    true_candidates: Tuple[str, ...] = (
        "som_value", "y_true", "true", "target", "obs", "observed", "有机质"
    )
    pred_candidates: Tuple[str, ...] = (
        "strict_pred_final", "pred_final", "rfk_pred_final", "repeated_pred_final",
        "full_fit_rfk_pred", "full_fit_pred_final", "full_fit_rf_pred", "pred", "prediction"
    )
    x_candidates: Tuple[str, ...] = ("x_32648", "x", "utm_x", "easting", "投影x", "经度", "lon")
    y_candidates: Tuple[str, ...] = ("y_32648", "y", "utm_y", "northing", "投影y", "纬度", "lat")


CFG = Config()


def log(msg: str) -> None:
    print(str(msg), flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def pick_first_existing(df: pd.DataFrame, candidates: Iterable[str], role: str) -> str:
    col_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if str(c).lower() in col_map:
            return col_map[str(c).lower()]
    lower_cols = list(col_map.keys())
    for c in candidates:
        hits = [cc for cc in lower_cols if str(c).lower() in cc]
        if hits:
            return col_map[hits[0]]
    raise ValueError(f"无法识别 {role} 列。当前列名：{list(df.columns)}")


def infer_columns(df: pd.DataFrame) -> Dict[str, str]:
    return {
        "true_col": pick_first_existing(df, CFG.true_candidates, "真实值"),
        "pred_col": pick_first_existing(df, CFG.pred_candidates, "预测值"),
        "x_col": pick_first_existing(df, CFG.x_candidates, "X坐标"),
        "y_col": pick_first_existing(df, CFG.y_candidates, "Y坐标"),
    }


def weighted_quantile(values: np.ndarray, quantile: float, sample_weight: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)
    m = np.isfinite(values) & np.isfinite(sample_weight) & (sample_weight > 0)
    values = values[m]
    sample_weight = sample_weight[m]
    if values.size == 0:
        return float("nan")
    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]
    cum_weight = np.cumsum(sample_weight)
    cutoff = quantile * cum_weight[-1]
    idx = int(np.searchsorted(cum_weight, cutoff, side="left"))
    idx = min(max(idx, 0), len(values) - 1)
    return float(values[idx])


def local_qhat(calib_xy: np.ndarray, calib_resid: np.ndarray, target_xy: np.ndarray,
               alpha: float, neighbors: int, bandwidth_scale: float, min_bandwidth: float) -> np.ndarray:
    n_cal = calib_xy.shape[0]
    k = int(min(max(5, neighbors), max(1, n_cal)))
    out = np.full(target_xy.shape[0], np.nan, dtype=float)
    if n_cal == 0 or target_xy.shape[0] == 0:
        return out

    if cKDTree is not None:
        tree = cKDTree(calib_xy)
        dists, idxs = tree.query(target_xy, k=k, workers=-1)
        if k == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]
        for i in range(target_xy.shape[0]):
            nn_d = np.asarray(dists[i], dtype=float)
            nn_r = calib_resid[np.asarray(idxs[i], dtype=int)]
            bw = max(float(min_bandwidth), float(np.nanmax(nn_d) * bandwidth_scale))
            w = np.exp(-0.5 * (nn_d / bw) ** 2)
            out[i] = weighted_quantile(nn_r, 1.0 - alpha, w)
        return out

    for i, pt in enumerate(target_xy):
        d = np.sqrt(np.sum((calib_xy - pt) ** 2, axis=1))
        nn_idx = np.argpartition(d, k - 1)[:k]
        nn_d = d[nn_idx]
        nn_r = calib_resid[nn_idx]
        bw = max(float(min_bandwidth), float(np.nanmax(nn_d) * bandwidth_scale))
        w = np.exp(-0.5 * (nn_d / bw) ** 2)
        out[i] = weighted_quantile(nn_r, 1.0 - alpha, w)
    return out


def compute_interval_metrics(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float, ccb_bins: int) -> Dict[str, float]:
    width = upper - lower
    valid = np.isfinite(y_true) & np.isfinite(lower) & np.isfinite(upper)
    if not np.any(valid):
        return {"PICP": np.nan, "MPIW": np.nan, "NMPIW": np.nan, "QCP": np.nan, "IntervalScore": np.nan}
    y_true = y_true[valid]
    lower = lower[valid]
    upper = upper[valid]
    width = width[valid]
    covered = (y_true >= lower) & (y_true <= upper)
    picp = float(np.mean(covered))
    mpiw = float(np.mean(width))
    yrange = float(np.nanmax(y_true) - np.nanmin(y_true))
    nmpiw = float(mpiw / yrange) if yrange > 0 else np.nan
    nominal = 1.0 - alpha
    if len(y_true) < ccb_bins:
        qcp = float(abs(picp - nominal))
    else:
        try:
            bins = pd.qcut(pd.Series(width), q=ccb_bins, duplicates="drop")
            bin_df = pd.DataFrame({"covered": covered.astype(float), "bin": bins})
            bin_cov = bin_df.groupby("bin", observed=False)["covered"].mean().values
            qcp = float(np.mean(np.abs(bin_cov - nominal)))
        except Exception:
            qcp = float(abs(picp - nominal))
    under = np.maximum(lower - y_true, 0.0)
    over = np.maximum(y_true - upper, 0.0)
    interval_score = width + (2.0 / alpha) * under + (2.0 / alpha) * over
    return {
        "PICP": picp,
        "MPIW": mpiw,
        "NMPIW": nmpiw,
        "QCP": qcp,
        "IntervalScore": float(np.mean(interval_score)),
        "mean_width": mpiw,
        "median_width": float(np.median(width)),
    }


def candidate_feature_columns(df: pd.DataFrame, cols: Dict[str, str], manifest: Dict[str, Any]) -> List[str]:
    exclude_tokens = [
        cols.get("true_col"), cols.get("pred_col"), cols.get("x_col"), cols.get("y_col"),
        "resid", "residual", "fold", "group", "id", "name", "label", "geometry", "covered",
        "lower", "upper", "qhat", "interval", "pred", "true", "obs", "target", "有机质"
    ]
    exclude = {str(x).lower() for x in exclude_tokens if x}
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    features = []
    manifest_features = (
        manifest.get("features")
        or manifest.get("feature_cols")
        or manifest.get("real_covariate_cols")
        or (manifest.get("outputs") or {}).get("features")
        or manifest.get("params", {}).get("features")
        or []
    )
    manifest_features = [str(x) for x in manifest_features if str(x) in df.columns]
    if manifest_features:
        features = manifest_features
    else:
        for c in numeric_cols:
            lc = str(c).lower()
            if lc in exclude:
                continue
            if any(tok and str(tok).lower() in lc for tok in exclude_tokens if isinstance(tok, str)):
                continue
            features.append(str(c))
    if not features:
        features = [cols["x_col"], cols["y_col"]]
    return features


def standardize_matrix(x: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    if mean is None:
        mean = np.nanmean(x, axis=0)
    if std is None:
        std = np.nanstd(x, axis=0)
    std = np.where((~np.isfinite(std)) | (std <= 0), 1.0, std)
    z = (x - mean) / std
    z[~np.isfinite(z)] = 0.0
    return z, mean, std


def feature_weights_from_manifest(features: List[str], manifest: Dict[str, Any]) -> np.ndarray:
    weights = np.ones(len(features), dtype=float)
    # 支持多种可能的变量重要性结构
    items = manifest.get("feature_importance") or manifest.get("feature_importances") or manifest.get("metrics", {}).get("feature_importance")
    imp_map: Dict[str, float] = {}
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                name = str(item.get("feature") or item.get("name") or item.get("variable") or "")
                val = item.get("importance") or item.get("value") or item.get("weight")
                try:
                    if name:
                        imp_map[name] = float(val)
                except Exception:
                    pass
    elif isinstance(items, dict):
        for k, v in items.items():
            try:
                imp_map[str(k)] = float(v)
            except Exception:
                pass
    if imp_map:
        weights = np.array([max(0.0, float(imp_map.get(f, 0.0))) for f in features], dtype=float)
        if np.nansum(weights) <= 0:
            weights = np.ones(len(features), dtype=float)
    weights = weights / np.nanmean(weights) if np.nanmean(weights) > 0 else np.ones(len(features), dtype=float)
    return np.sqrt(weights)


def compute_aoa_train(df: pd.DataFrame, features: List[str], cols: Dict[str, str], manifest: Dict[str, Any]) -> Dict[str, Any]:
    x = df[features].apply(pd.to_numeric, errors="coerce").values.astype(float)
    z, mean, std = standardize_matrix(x)
    weights = feature_weights_from_manifest(features, manifest)
    zw = z * weights
    n = zw.shape[0]
    if n <= 2:
        di_train = np.zeros(n, dtype=float)
        threshold = 0.0
        mean_nn_dist = 1.0
    else:
        if cKDTree is not None:
            tree = cKDTree(zw)
            d, _ = tree.query(zw, k=min(max(2, CFG.aoa_k_train), n), workers=-1)
            if d.ndim == 1:
                nn = d
            else:
                nn = d[:, 1]
        else:
            nn = []
            for i in range(n):
                d = np.sqrt(np.sum((zw - zw[i]) ** 2, axis=1))
                d[i] = np.inf
                nn.append(np.nanmin(d))
            nn = np.asarray(nn, dtype=float)
        mean_nn_dist = float(np.nanmean(nn[np.isfinite(nn) & (nn > 0)])) if np.any(np.isfinite(nn) & (nn > 0)) else 1.0
        di_train = nn / mean_nn_dist
        q1, q3 = np.nanpercentile(di_train, [25, 75])
        iqr = q3 - q1
        non_outlier = di_train <= (q3 + 1.5 * iqr)
        if np.any(non_outlier):
            threshold = float(np.nanmax(di_train[non_outlier]))
        else:
            threshold = float(np.nanpercentile(di_train, CFG.aoa_outlier_quantile * 100))
    return {
        "features": features,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": weights.tolist(),
        "train_weighted_matrix": zw,
        "train_di": di_train,
        "mean_nn_dist": mean_nn_dist,
        "threshold": threshold,
        "mode": "weighted_covariate_space" if features != [cols["x_col"], cols["y_col"]] else "coordinate_space",
    }


def compute_grid_di_by_coordinate(grid_xy: np.ndarray, calib_xy: np.ndarray, train_di_ref: Dict[str, Any]) -> np.ndarray:
    # 当没有完整预测协变量矩阵时，使用空间位置支持度作为安全回退。
    if calib_xy.shape[0] == 0 or grid_xy.shape[0] == 0:
        return np.full(grid_xy.shape[0], np.nan, dtype=float)
    if cKDTree is not None:
        tree = cKDTree(calib_xy)
        d, _ = tree.query(grid_xy, k=1, workers=-1)
    else:
        d = np.empty(grid_xy.shape[0], dtype=float)
        for i, pt in enumerate(grid_xy):
            d[i] = float(np.nanmin(np.sqrt(np.sum((calib_xy - pt) ** 2, axis=1))))
    denom = float(np.nanmean(d[np.isfinite(d) & (d > 0)])) if np.any(np.isfinite(d) & (d > 0)) else 1.0
    return d / denom


def resolve_grid_covariate_csv(manifest: Dict[str, Any], pred_tif: Path) -> Path | None:
    """Find per-pixel prediction covariate table produced by the mapping workflow."""
    outputs = manifest.get("outputs") or {}
    candidates = []
    for key in [
        "prediction_grid_covariates_csv", "grid_covariates_csv", "prediction_grid_csv", "gee_grid_covariates_csv",
        "prediction_covariates_csv", "predictor_grid_csv", "covariate_grid_csv",
    ]:
        val = outputs.get(key) or manifest.get(key)
        if val:
            candidates.append(Path(str(val)))
    try:
        d = pred_tif.parent
        candidates.extend([
            d / "gee_prediction_grid_covariates.csv",
            d / "prediction_grid_covariates.csv",
            d / "grid_covariates.csv",
            d / "predict_covariates.csv",
        ])
    except Exception:
        pass
    for cand in candidates:
        try:
            if cand.exists() and cand.is_file():
                return cand
        except Exception:
            pass
    return None


def compute_grid_di_by_covariates(grid_df: pd.DataFrame, features: List[str], aoa_train: Dict[str, Any], expected_n: int) -> tuple[np.ndarray | None, str]:
    """Compute AOA DI in the same weighted covariate space as training samples.

    Returns None when the prediction grid table cannot be aligned to valid pixels.
    """
    if grid_df is None or not len(grid_df):
        return None, "missing_grid_covariate_table"
    available = [f for f in features if f in grid_df.columns]
    if len(available) < max(3, int(0.6 * len(features))):
        return None, f"insufficient_matching_features:{len(available)}/{len(features)}"
    if len(grid_df) != int(expected_n):
        # 只接受与有效预测像元一一对应的表，避免行列错配导致错误 AOA。
        return None, f"row_count_mismatch:{len(grid_df)}!={expected_n}"
    x = grid_df[available].apply(pd.to_numeric, errors="coerce").values.astype(float)
    # 对未匹配特征采用训练均值补齐，保证维度与权重一致。
    full = np.zeros((x.shape[0], len(features)), dtype=float)
    mean = np.asarray(aoa_train.get("mean"), dtype=float)
    std = np.asarray(aoa_train.get("std"), dtype=float)
    weights = np.asarray(aoa_train.get("weights"), dtype=float)
    train_w = np.asarray(aoa_train.get("train_weighted_matrix"), dtype=float)
    feature_to_idx = {f: i for i, f in enumerate(features)}
    full[:] = mean.reshape(1, -1)
    for j, f in enumerate(available):
        full[:, feature_to_idx[f]] = x[:, j]
    z = (full - mean.reshape(1, -1)) / np.where(std <= 0, 1.0, std).reshape(1, -1)
    z[~np.isfinite(z)] = 0.0
    zw = z * weights.reshape(1, -1)
    if cKDTree is not None:
        tree = cKDTree(train_w)
        d, _ = tree.query(zw, k=1, workers=-1)
    else:
        d = np.empty(zw.shape[0], dtype=float)
        for i, row in enumerate(zw):
            d[i] = float(np.nanmin(np.sqrt(np.sum((train_w - row) ** 2, axis=1))))
    denom = float(aoa_train.get("mean_nn_dist") or 1.0)
    denom = denom if np.isfinite(denom) and denom > 0 else 1.0
    return d / denom, f"weighted_covariate_space_grid_features:{len(available)}/{len(features)}"


def save_png_continuous(arr: np.ndarray, out_path: Path, cmap_name: str = "viridis") -> str:
    """Save a transparent PNG preview for a continuous raster array."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        a = np.asarray(arr, dtype=float)
        valid = np.isfinite(a)
        if not np.any(valid):
            return ""
        lo, hi = np.nanpercentile(a[valid], [2, 98])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(a[valid])), float(np.nanmax(a[valid]))
        scaled = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
        cmap = plt.get_cmap(cmap_name)
        rgba = (cmap(scaled) * 255).astype("uint8")
        rgba[~valid, 3] = 0
        ensure_dir(out_path.parent)
        Image.fromarray(rgba, mode="RGBA").save(out_path)
        return str(out_path)
    except Exception:
        return ""


def save_png_risk_class(arr: np.ndarray, out_path: Path) -> str:
    """Save the four-class GCP+AOA risk map as a transparent PNG."""
    try:
        import numpy as np
        from PIL import Image
        a = np.asarray(arr)
        rgba = np.zeros((a.shape[0], a.shape[1], 4), dtype="uint8")
        colors = {
            1: (46, 160, 67, 255),     # 高可信
            2: (247, 183, 49, 255),    # 域内高不确定
            3: (220, 53, 69, 255),     # 外推高风险
            4: (138, 43, 226, 255),    # 虚假自信
        }
        for k, c in colors.items():
            rgba[a == k] = c
        ensure_dir(out_path.parent)
        Image.fromarray(rgba, mode="RGBA").save(out_path)
        return str(out_path)
    except Exception:
        return ""


def array_to_tif(arr: np.ndarray, out_path: Path, profile: Dict[str, Any], nodata: float = -9999.0) -> Path:
    ensure_dir(out_path.parent)
    out_profile = profile.copy()
    out_profile.update(driver="GTiff", dtype="float32", count=1, compress="lzw", nodata=nodata)
    a = np.asarray(arr, dtype="float32").copy()
    a[~np.isfinite(a)] = nodata
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(a, 1)
    return out_path


def risk_classification(width_arr: np.ndarray, aoa_inside_arr: np.ndarray, valid_mask: np.ndarray, width_quantile: float) -> Tuple[np.ndarray, float, Dict[str, int]]:
    vals = width_arr[valid_mask & np.isfinite(width_arr)]
    if vals.size == 0:
        threshold = float("nan")
    else:
        threshold = float(np.nanquantile(vals, width_quantile))
    out = np.full(width_arr.shape, np.nan, dtype="float32")
    width_wide = width_arr > threshold
    aoa_inside = aoa_inside_arr == 1
    # 1 高可信：区间窄 + AOA内
    # 2 域内高不确定：区间宽 + AOA内
    # 3 外推高风险：区间宽 + AOA外
    # 4 虚假自信：区间窄 + AOA外
    out[valid_mask & (~width_wide) & aoa_inside] = 1
    out[valid_mask & width_wide & aoa_inside] = 2
    out[valid_mask & width_wide & (~aoa_inside)] = 3
    out[valid_mask & (~width_wide) & (~aoa_inside)] = 4
    counts = {str(k): int(np.nansum(out == k)) for k in [1, 2, 3, 4]}
    return out, threshold, counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GCP + AOA formal uncertainty analysis")
    parser.add_argument("--manifest", dest="manifest_path", default=None, help="上一轮制图 manifest 路径")
    parser.add_argument("--pred_tif", dest="pred_tif", default=None, help="上一轮预测栅格路径")
    parser.add_argument("--strict_oof_csv", dest="strict_oof_csv", default=None, help="上一轮 OOF/校准样点表路径")
    parser.add_argument("--out_dir", dest="out_dir", default=None, help="不确定性分析输出目录")
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> Tuple[Dict[str, Any], Path, Path, Path]:
    manifest: Dict[str, Any] = {}
    if args.manifest_path:
        manifest_path = Path(args.manifest_path)
        if manifest_path.exists():
            manifest = read_json(manifest_path)
        else:
            raise FileNotFoundError(f"未找到制图清单文件：{manifest_path}")
    else:
        manifest_path = Path("")
    outputs = manifest.get("outputs") or {}
    pred_tif = Path(args.pred_tif or outputs.get("pred_tif") or outputs.get("prediction_tif") or "")
    oof_csv = Path(args.strict_oof_csv or outputs.get("strict_oof_csv") or outputs.get("oof_csv") or outputs.get("calibration_csv") or "")
    if not pred_tif.exists():
        raise FileNotFoundError(f"未找到预测栅格：{pred_tif}")
    if not oof_csv.exists():
        raise FileNotFoundError(f"未找到校准样点表：{oof_csv}")
    out_dir = Path(args.out_dir) if args.out_dir else Path(UNCERTAINTY_RESULT_ROOT) / timestamp_name("GCP_AOA")
    return manifest, pred_tif, oof_csv, ensure_dir(out_dir)


def main() -> None:
    args = parse_args()
    manifest, pred_tif, oof_csv, out_dir = resolve_inputs(args)
    report_dir = ensure_dir(out_dir / "01_指标报告")
    map_dir = ensure_dir(out_dir / "02_不确定性与适用域图")
    table_dir = ensure_dir(out_dir / "03_样点与统计表")
    state_dir = ensure_dir(out_dir / "_state")

    log("启动地理共形预测（GCP）+ AOA 不确定性与适用域分析")
    log(f"预测栅格：{pred_tif}")
    log(f"校准样点表：{oof_csv}")
    log("[STEP] 读取校准样点与预测残差")

    df = pd.read_csv(oof_csv)
    cols = infer_columns(df)
    y_true = pd.to_numeric(df[cols["true_col"]], errors="coerce").values.astype(float)
    y_pred = pd.to_numeric(df[cols["pred_col"]], errors="coerce").values.astype(float)
    xs = pd.to_numeric(df[cols["x_col"]], errors="coerce").values.astype(float)
    ys = pd.to_numeric(df[cols["y_col"]], errors="coerce").values.astype(float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(xs) & np.isfinite(ys)
    df = df.loc[valid].reset_index(drop=True)
    y_true, y_pred, xs, ys = y_true[valid], y_pred[valid], xs[valid], ys[valid]
    xy_calib = np.column_stack([xs, ys])
    abs_resid = np.abs(y_true - y_pred)

    log("[STEP] 计算样点级 GCP 局地非一致性阈值")
    qhat_sample = local_qhat(xy_calib, abs_resid, xy_calib, CFG.alpha, CFG.neighbors, CFG.bandwidth_scale, CFG.min_bandwidth)
    lower_sample = y_pred - qhat_sample
    upper_sample = y_pred + qhat_sample
    metrics = compute_interval_metrics(y_true, lower_sample, upper_sample, CFG.alpha, CFG.ccb_bins)

    log("[STEP] 构建样点 AOA 特征空间")
    features = candidate_feature_columns(df, cols, manifest)
    aoa_train = compute_aoa_train(df, features, cols, manifest)
    sample_out = pd.DataFrame({
        "x": xs,
        "y": ys,
        "y_true": y_true,
        "pred": y_pred,
        "abs_residual": abs_resid,
        "qhat": qhat_sample,
        "interval_lower": lower_sample,
        "interval_upper": upper_sample,
        "interval_width": upper_sample - lower_sample,
        "covered": ((y_true >= lower_sample) & (y_true <= upper_sample)).astype(int),
        "AOA_DI_train": aoa_train["train_di"],
        "AOA_inside_train": (aoa_train["train_di"] <= aoa_train["threshold"]).astype(int),
    })
    sample_csv = table_dir / "GCP_AOA_样点区间与适用域审计.csv"
    sample_out.to_csv(sample_csv, index=False, encoding="utf-8-sig")

    log("[STEP] 读取预测栅格并计算像元级 GCP 区间")
    with rasterio.open(pred_tif) as src:
        center_arr = src.read(1).astype("float32")
        profile = src.profile.copy()
        nodata = src.nodata
        if nodata is not None:
            center_arr = np.where(center_arr == nodata, np.nan, center_arr)
    valid_grid = np.isfinite(center_arr)
    rows, cols_grid = np.where(valid_grid)
    grid_x, grid_y = xy(profile["transform"], rows, cols_grid, offset="center")
    grid_xy = np.column_stack([np.asarray(grid_x, dtype=float), np.asarray(grid_y, dtype=float)])

    qhat_grid = local_qhat(xy_calib, abs_resid, grid_xy, CFG.alpha, CFG.neighbors, CFG.bandwidth_scale, CFG.min_bandwidth)
    lower_arr = np.full(center_arr.shape, np.nan, dtype="float32")
    upper_arr = np.full(center_arr.shape, np.nan, dtype="float32")
    width_arr = np.full(center_arr.shape, np.nan, dtype="float32")
    qhat_arr = np.full(center_arr.shape, np.nan, dtype="float32")
    qhat_arr[rows, cols_grid] = qhat_grid.astype("float32")
    vals = center_arr[rows, cols_grid].astype(float)
    lower_arr[rows, cols_grid] = (vals - qhat_grid).astype("float32")
    upper_arr[rows, cols_grid] = (vals + qhat_grid).astype("float32")
    width_arr[rows, cols_grid] = (2.0 * qhat_grid).astype("float32")

    log("[STEP] 计算 AOA 适用域")
    grid_cov_csv = resolve_grid_covariate_csv(manifest, pred_tif)
    di_grid = None
    aoa_grid_mode = ""
    if grid_cov_csv is not None:
        try:
            log(f"[STEP] 读取预测像元协变量矩阵：{grid_cov_csv}")
            grid_df = pd.read_csv(grid_cov_csv)
            di_grid, aoa_grid_mode = compute_grid_di_by_covariates(grid_df, features, aoa_train, expected_n=len(grid_xy))
        except Exception as exc:
            di_grid = None
            aoa_grid_mode = f"grid_covariate_space_failed:{exc}"
    if di_grid is None:
        log("[STEP] 未获得可对齐的预测像元协变量矩阵，采用坐标支持度计算 AOA")
        di_grid = compute_grid_di_by_coordinate(grid_xy, xy_calib, aoa_train)
        if aoa_train["mode"] == "weighted_covariate_space":
            xy_train_di = compute_grid_di_by_coordinate(xy_calib, xy_calib, aoa_train)
            aoa_threshold_grid = float(np.nanpercentile(xy_train_di[np.isfinite(xy_train_di)], 95)) if np.any(np.isfinite(xy_train_di)) else 1.0
            aoa_grid_mode = aoa_grid_mode or "coordinate_supported_grid_DI_with_weighted_training_AOA_reference"
        else:
            aoa_threshold_grid = float(aoa_train["threshold"])
            aoa_grid_mode = aoa_grid_mode or "coordinate_space"
    else:
        aoa_threshold_grid = float(aoa_train["threshold"])
    di_arr = np.full(center_arr.shape, np.nan, dtype="float32")
    di_arr[rows, cols_grid] = di_grid.astype("float32")
    aoa_inside_arr = np.full(center_arr.shape, np.nan, dtype="float32")
    aoa_inside_arr[valid_grid] = (di_arr[valid_grid] <= aoa_threshold_grid).astype("float32")

    log("[STEP] 生成 GCP + AOA 综合风险分区")
    risk_arr, width_threshold, risk_counts = risk_classification(width_arr, aoa_inside_arr, valid_grid, CFG.width_threshold_quantile)

    log("[STEP] 写出 GeoTIFF 与正式 PNG 制图结果")
    center_tif = str(array_to_tif(center_arr, map_dir / "GCP中心预测图.tif", profile, CFG.nodata))
    lower_tif = str(array_to_tif(lower_arr, map_dir / "GCP预测区间下界.tif", profile, CFG.nodata))
    upper_tif = str(array_to_tif(upper_arr, map_dir / "GCP预测区间上界.tif", profile, CFG.nodata))
    width_tif = str(array_to_tif(width_arr, map_dir / "GCP预测区间宽度.tif", profile, CFG.nodata))
    qhat_tif = str(array_to_tif(qhat_arr, map_dir / "GCP局地非一致性阈值.tif", profile, CFG.nodata))
    aoa_di_tif = str(array_to_tif(di_arr, map_dir / "AOA不相似性指数_DI.tif", profile, CFG.nodata))
    aoa_inside_tif = str(array_to_tif(aoa_inside_arr, map_dir / "AOA适用域二值图.tif", profile, CFG.nodata))
    risk_class_tif = str(array_to_tif(risk_arr, map_dir / "GCP_AOA综合风险分区.tif", profile, CFG.nodata))

    if create_formal_continuous_png is not None:
        width_png = create_formal_continuous_png(
            width_tif, map_dir / "GCP预测区间宽度.png",
            title="土壤有机质 GCP 预测区间宽度图", legend_title="区间宽度（g/kg）", ramp_name="magma"
        )
        aoa_di_png = create_formal_continuous_png(
            aoa_di_tif, map_dir / "AOA不相似性指数_DI.png",
            title="土壤有机质 AOA 不相似性指数图", legend_title="DI 指数", ramp_name="viridis"
        )
    else:
        width_png = save_png_continuous(width_arr, map_dir / "GCP预测区间宽度.png", cmap_name="magma")
        aoa_di_png = save_png_continuous(di_arr, map_dir / "AOA不相似性指数_DI.png", cmap_name="viridis")

    if create_formal_class_png is not None:
        risk_class_png = create_formal_class_png(
            risk_class_tif, map_dir / "GCP_AOA综合风险分区.png",
            title="土壤有机质 GCP + AOA 综合风险分区图", legend_title="风险分区"
        )
    else:
        risk_class_png = save_png_risk_class(risk_arr, map_dir / "GCP_AOA综合风险分区.png")

    paths = {
        "center_tif": center_tif,
        "lower_tif": lower_tif,
        "upper_tif": upper_tif,
        "width_tif": width_tif,
        "qhat_tif": qhat_tif,
        "aoa_di_tif": aoa_di_tif,
        "aoa_inside_tif": aoa_inside_tif,
        "risk_class_tif": risk_class_tif,
        "width_png": width_png,
        "aoa_di_png": aoa_di_png,
        "risk_class_png": risk_class_png,
        "sample_csv": str(sample_csv),
        "sample_audit_csv": str(sample_csv),
        "report_json": str(report_dir / "GCP_AOA指标报告.json"),
        "report_txt": str(report_dir / "GCP_AOA分析报告.txt"),
    }

    n_valid = int(valid_grid.sum())
    aoa_inside_count = int(np.nansum(aoa_inside_arr == 1))
    aoa_outside_count = int(np.nansum(aoa_inside_arr == 0))
    risk_labels = {
        "1": "高可信区域：区间窄且位于 AOA 内",
        "2": "域内高不确定区域：区间宽且位于 AOA 内",
        "3": "外推高风险区域：区间宽且位于 AOA 外",
        "4": "虚假自信区域：区间窄但位于 AOA 外",
    }
    summary = {
        "method": "geographic_conformal_prediction_plus_aoa",
        "method_cn": "地理共形预测（GCP）+ 适用域（AOA）分析",
        "gcp_full_name": "Geographic Conformal Prediction",
        "gcp_cn_name": "地理共形预测",
        "gcp_not_kriging": True,
        "alpha": CFG.alpha,
        "nominal_coverage": 1.0 - CFG.alpha,
        "gcp_params": {
            "neighbors": CFG.neighbors,
            "bandwidth_scale": CFG.bandwidth_scale,
            "min_bandwidth": CFG.min_bandwidth,
        },
        "aoa_params": {
            "sample_feature_mode": aoa_train["mode"],
            "grid_aoa_mode": aoa_grid_mode,
            "feature_count_for_sample_aoa": len(features),
            "feature_names_for_sample_aoa": features,
            "sample_aoa_threshold": float(aoa_train["threshold"]),
            "grid_aoa_threshold": float(aoa_threshold_grid),
            "width_threshold_quantile": CFG.width_threshold_quantile,
            "width_threshold": width_threshold,
        },
        "inputs": {
            "manifest": str(args.manifest_path or ""),
            "pred_tif": str(pred_tif),
            "calibration_csv": str(oof_csv),
            "grid_covariate_csv": str(grid_cov_csv or ""),
        },
        "sample_metrics": metrics,
        "grid_statistics": {
            "valid_pixel_count": n_valid,
            "AOA_inside_pixel_count": aoa_inside_count,
            "AOA_outside_pixel_count": aoa_outside_count,
            "AOA_inside_ratio": float(aoa_inside_count / n_valid) if n_valid else np.nan,
            "risk_class_counts": risk_counts,
            "risk_class_labels": risk_labels,
        },
        "outputs": paths,
    }
    try:
        from services.gcp_result_report_service import generate_ai_gcp_analysis
        _ai_text = generate_ai_gcp_analysis(summary, result_paths=paths)
        if _ai_text:
            summary["ai_result_analysis"] = _ai_text
        else:
            summary["ai_result_analysis_status"] = "not_generated_no_template_fallback"
    except Exception as exc:
        summary["ai_result_analysis_error"] = str(exc)
    write_json(report_dir / "GCP_AOA指标报告.json", summary)
    write_json(state_dir / "gcp_aoa_manifest.json", {
        "model": "GCP_AOA",
        "mode": "formal_delivery",
        "metrics": metrics,
        "outputs": paths,
        "summary": summary,
    })

    txt = f"""成都市土壤有机质 地理共形预测（GCP）+ AOA 不确定性与适用域分析报告
=======================================================

一、输入数据
- 预测栅格：{pred_tif}
- 校准样点表：{oof_csv}
- 有效校准样点数：{len(df)}
- 有效预测像元数：{n_valid}

二、地理共形预测（GCP）预测区间指标
- 名义覆盖率：{1.0 - CFG.alpha:.2f}
- PICP：{metrics.get('PICP', np.nan):.6f}
- MPIW：{metrics.get('MPIW', np.nan):.6f}
- NMPIW：{metrics.get('NMPIW', np.nan):.6f}
- QCP：{metrics.get('QCP', np.nan):.6f}
- Interval Score：{metrics.get('IntervalScore', np.nan):.6f}

三、AOA 适用域分析
- 样点 AOA 特征空间：{aoa_train['mode']}
- 像元 AOA 计算模式：{aoa_grid_mode}
- 样点 AOA 特征数量：{len(features)}
- AOA 内像元比例：{(aoa_inside_count / n_valid if n_valid else np.nan):.6f}
- AOA 内像元数：{aoa_inside_count}
- AOA 外像元数：{aoa_outside_count}

四、GCP + AOA 综合风险分区
- 区间宽度分界分位数：{CFG.width_threshold_quantile:.2f}
- 区间宽度阈值：{width_threshold:.6f}
- 1 高可信区域：{risk_counts.get('1', 0)} 像元
- 2 域内高不确定区域：{risk_counts.get('2', 0)} 像元
- 3 外推高风险区域：{risk_counts.get('3', 0)} 像元
- 4 虚假自信区域：{risk_counts.get('4', 0)} 像元

五、结果文件
- GCP 预测区间宽度（g/kg）：{paths['width_tif']}
- AOA 不相似性指数：{paths['aoa_di_tif']}
- AOA 适用域二值图：{paths['aoa_inside_tif']}
- GCP + AOA 综合风险分区：{paths['risk_class_tif']}
- GCP 区间宽度 PNG：{paths.get('width_png', '')}
- AOA 不相似性指数 PNG：{paths.get('aoa_di_png', '')}
- 综合风险分区 PNG：{paths.get('risk_class_png', '')}
- 样点审计表：{paths['sample_csv']}

说明：AOA 优先采用模型训练样点中的协变量特征空间。若制图结果提供逐像元协变量矩阵，像元端适用域图采用同一加权协变量空间计算；若未提供可对齐矩阵，则采用坐标支持度作为稳定输出，并在本报告中记录计算模式。
"""
    report_txt = report_dir / "GCP_AOA分析报告.txt"
    report_txt.write_text(txt, encoding="utf-8")
    log(f"GCP + AOA 分析完成：{report_txt}")


if __name__ == "__main__":
    main()
