#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
成都市土壤有机质 GCP 正式运行版
================================
用途：
1. 基于已定版 RFK 预测结果构建 GeoCP-like 空间加权共形预测区间；
2. 输出简洁指标报告：
   - 预测区间覆盖率（PICP）
   - 平均区间宽度（MPIW）
   - 归一化平均区间宽度（NMPIW）
   - 条件覆盖偏差（CCB）
   - 区间评分（Interval Score）
3. 输出全区不确定性图。

输出目录：
E:/Agent_DSM/GCP_OUT

仅输出两类最终成果：
- 01_指标报告
- 02_不确定性图
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 固定配置
# =============================================================================

@dataclass(frozen=True)
class Config:
    rfk_out_root: Path = Path(r"E:/Agent_DSM/RFK_OUT")
    out_root: Path = Path(r"E:/Agent_DSM/GCP_OUT")

    # 已调优的 GCP 参数文件；如不存在则使用默认参数
    tuned_config_json: Path = Path(r"E:/Agent_DSM/GCP_OUT/_state/gcp_config.json")

    # 默认参数（若 tuned_config_json 不存在或读取失败，则使用这里）
    alpha: float = 0.10
    neighbors: int = 60
    bandwidth_scale: float = 0.6
    min_bandwidth: float = 200.0

    # CCB 的分箱数
    ccb_bins: int = 10

    # 列名候选
    true_candidates: Tuple[str, ...] = (
        "som_value", "y_true", "true", "target", "obs", "observed"
    )
    pred_candidates: Tuple[str, ...] = (
        "strict_pred_final",
        "pred_final",
        "rfk_pred_final",
        "repeated_pred_final",
        "full_fit_rfk_pred",
        "full_fit_pred_final",
        "full_fit_rf_pred",
        "pred",
    )
    x_candidates: Tuple[str, ...] = ("x_32648", "x", "utm_x", "easting")
    y_candidates: Tuple[str, ...] = ("y_32648", "y", "utm_y", "northing")


CFG = Config()
REPORT_DIR = CFG.out_root / "01_指标报告"
MAP_DIR = CFG.out_root / "02_不确定性图"
INTERNAL_DIR = CFG.out_root / "_internal"
STATE_DIR = CFG.out_root / "_state"


# =============================================================================
# 工具函数
# =============================================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pick_first_existing(df: pd.DataFrame, candidates: Tuple[str, ...], role: str) -> str:
    col_map = {str(c).lower(): c for c in df.columns}

    # 先精确匹配
    for c in candidates:
        if c.lower() in col_map:
            return col_map[c.lower()]

    # 再模糊匹配
    lower_cols = list(col_map.keys())
    for c in candidates:
        hits = [cc for cc in lower_cols if c.lower() in cc]
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


def load_rfk_manifest(manifest_path: Path | None = None) -> Dict[str, object]:
    if manifest_path is None:
        manifest_path = CFG.rfk_out_root / "_state" / "rfk_manifest.json"
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"未找到 RFK manifest：{manifest_path}")
    log(f"读取 RFK manifest：{manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))

def load_tuned_params() -> Tuple[float, int, float, float]:
    alpha = CFG.alpha
    neighbors = CFG.neighbors
    bandwidth_scale = CFG.bandwidth_scale
    min_bandwidth = CFG.min_bandwidth

    if CFG.tuned_config_json.exists():
        try:
            cfg = json.loads(CFG.tuned_config_json.read_text(encoding="utf-8"))
            alpha = float(cfg.get("alpha", alpha))
            sel = cfg.get("selected_tuned_params", {})
            neighbors = int(sel.get("neighbors", neighbors))
            bandwidth_scale = float(sel.get("bandwidth_scale", bandwidth_scale))
            min_bandwidth = float(sel.get("min_bandwidth", min_bandwidth))
        except Exception:
            pass

    return alpha, neighbors, bandwidth_scale, min_bandwidth


def weighted_quantile(values: np.ndarray, quantile: float, sample_weight: np.ndarray) -> float:
    """
    计算加权分位数
    """
    values = np.asarray(values, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)

    m = np.isfinite(values) & np.isfinite(sample_weight) & (sample_weight > 0)
    values = values[m]
    sample_weight = sample_weight[m]

    if len(values) == 0:
        return np.nan

    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    cum_weight = np.cumsum(sample_weight)
    cutoff = quantile * cum_weight[-1]
    idx = np.searchsorted(cum_weight, cutoff, side="left")
    idx = min(max(idx, 0), len(values) - 1)
    return float(values[idx])


def compute_local_qhat(
    calib_xy: np.ndarray,
    calib_resid: np.ndarray,
    target_xy: np.ndarray,
    alpha: float,
    neighbors: int,
    bandwidth_scale: float,
    min_bandwidth: float,
) -> np.ndarray:
    """
    GeoCP-like 局地 qhat：
    对每个目标点，使用校准点距离加权后的绝对残差分位数作为 qhat。
    """
    n_cal = calib_xy.shape[0]
    k = int(min(max(5, neighbors), n_cal))
    out = np.full(target_xy.shape[0], np.nan, dtype=float)

    for i, pt in enumerate(target_xy):
        d = np.sqrt(np.sum((calib_xy - pt) ** 2, axis=1))
        nn_idx = np.argpartition(d, k - 1)[:k]
        nn_d = d[nn_idx]
        nn_r = calib_resid[nn_idx]

        base_bw = np.nanmax(nn_d) * float(bandwidth_scale)
        bw = max(float(min_bandwidth), float(base_bw))
        w = np.exp(-0.5 * (nn_d / bw) ** 2)

        q = weighted_quantile(nn_r, 1.0 - alpha, w)
        out[i] = q

    return out


def compute_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
    ccb_bins: int,
) -> Dict[str, float]:
    width = upper - lower
    covered = (y_true >= lower) & (y_true <= upper)

    # 1) 预测区间覆盖率
    picp = float(np.mean(covered))

    # 2) 平均区间宽度
    mpiw = float(np.mean(width))

    # 3) 归一化平均区间宽度
    y_range = float(np.nanmax(y_true) - np.nanmin(y_true))
    nmpiw = float(mpiw / y_range) if y_range > 0 else np.nan

    # 4) 条件覆盖偏差 CCB：按区间宽度分箱
    nominal = 1.0 - alpha
    if len(y_true) < ccb_bins:
        ccb = float(abs(picp - nominal))
    else:
        width_series = pd.Series(width)
        try:
            bins = pd.qcut(width_series, q=ccb_bins, duplicates="drop")
            bin_df = pd.DataFrame({"covered": covered.astype(float), "bin": bins})
            bin_cov = bin_df.groupby("bin", observed=False)["covered"].mean().values
            ccb = float(np.mean(np.abs(bin_cov - nominal)))
        except Exception:
            ccb = float(abs(picp - nominal))

    # 5) 区间评分 Interval Score
    under = np.maximum(lower - y_true, 0.0)
    over = np.maximum(y_true - upper, 0.0)
    interval_score = width + (2.0 / alpha) * under + (2.0 / alpha) * over
    is_mean = float(np.mean(interval_score))

    return {
        "PICP": picp,
        "MPIW": mpiw,
        "NMPIW": nmpiw,
        "CCB": ccb,
        "IntervalScore": is_mean,
        "mean_width": mpiw,
        "median_width": float(np.median(width)),
    }



def ensure_parent_dir(path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def build_timestamp_path(path: Path | str) -> Path:
    p = Path(path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return p.with_name(f"{p.stem}_{ts}{p.suffix}")


def get_writable_output_path(path: Path | str) -> Path:
    """
    优先使用固定输出名；
    若目标文件已存在，则尝试删除；
    删除失败（常见于 ArcGIS/QGIS/网页占用）时，自动改为带时间戳的新文件名。
    """
    p = ensure_parent_dir(path)
    if not p.exists():
        return p
    try:
        p.unlink()
        return p
    except Exception:
        return build_timestamp_path(p)


def safe_float32_array(arr: np.ndarray, nodata_value: float = -9999.0) -> np.ndarray:
    out = np.asarray(arr, dtype="float32").copy()
    out[~np.isfinite(out)] = nodata_value
    return out


def array_to_tif(arr: np.ndarray, out_path: Path, profile: dict) -> Path:
    """
    将数组安全写出为 tif，返回真实写出的路径。
    """
    real_out_path = get_writable_output_path(out_path)

    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        compress="lzw",
        nodata=-9999.0,
    )

    write_arr = safe_float32_array(arr, nodata_value=-9999.0)

    with rasterio.open(real_out_path, "w", **out_profile) as dst:
        dst.write(write_arr, 1)

    return Path(real_out_path)

def write_concise_report(metrics: Dict[str, float], params: Dict[str, float]) -> None:
    report_json = {
        "method": "GeoCP-like local conformal prediction",
        "alpha": params["alpha"],
        "neighbors": params["neighbors"],
        "bandwidth_scale": params["bandwidth_scale"],
        "min_bandwidth": params["min_bandwidth"],
        "metrics": metrics,
    }
    with open(REPORT_DIR / "GCP指标报告.json", "w", encoding="utf-8") as f:
        json.dump(report_json, f, ensure_ascii=False, indent=2)

    txt = f"""成都市土壤有机质 GCP 指标报告
==============================
方法：GeoCP-like 空间加权共形预测

固定参数：
alpha            = {params['alpha']}
neighbors        = {params['neighbors']}
bandwidth_scale  = {params['bandwidth_scale']}
min_bandwidth    = {params['min_bandwidth']}

核心指标：
PICP          = {metrics['PICP']:.6f}
MPIW          = {metrics['MPIW']:.6f}
NMPIW         = {metrics['NMPIW']:.6f}
CCB           = {metrics['CCB']:.6f}
IntervalScore = {metrics['IntervalScore']:.6f}

说明：
- PICP：预测区间覆盖率
- MPIW：平均区间宽度
- NMPIW：归一化平均区间宽度
- CCB：条件覆盖偏差
- IntervalScore：区间评分（越小越好）
"""
    (REPORT_DIR / "GCP指标报告.txt").write_text(txt, encoding="utf-8")


# =============================================================================
# 启动参数
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GCP formal from RFK")
    parser.add_argument("--manifest", dest="manifest_path", default=None, help="本次 RFK manifest 路径")
    parser.add_argument("--pred_tif", dest="pred_tif", default=None, help="本次 RFK 预测图路径")
    parser.add_argument("--strict_oof_csv", dest="strict_oof_csv", default=None, help="本次 RFK strict OOF CSV 路径")
    return parser.parse_args()


# =============================================================================
# 主流程
# =============================================================================

def main() -> None:
    args = parse_args()
    ensure_dir(CFG.out_root)
    ensure_dir(REPORT_DIR)
    ensure_dir(MAP_DIR)
    ensure_dir(INTERNAL_DIR)
    ensure_dir(STATE_DIR)

    log("开始 GCP 正式精简建模 ...")
    alpha, neighbors, bandwidth_scale, min_bandwidth = load_tuned_params()
    log(
        f"参数：alpha={alpha}, neighbors={neighbors}, "
        f"bandwidth_scale={bandwidth_scale}, min_bandwidth={min_bandwidth}"
    )

    manifest_override = Path(args.manifest_path) if args.manifest_path else None
    rfk_manifest = load_rfk_manifest(manifest_override)
    rfk_outputs = rfk_manifest.get("outputs", {})
    calibration_csv = Path(args.strict_oof_csv) if args.strict_oof_csv else Path(rfk_outputs["strict_oof_csv"])
    center_pred_tif = Path(args.pred_tif) if args.pred_tif else Path(rfk_outputs["pred_tif"])

    log(f"读取本次 RFK 结果：strict_oof_csv={calibration_csv} | pred_tif={center_pred_tif}")

    if not calibration_csv.exists():
        raise FileNotFoundError(f"未找到 RFK strict OOF 文件：{calibration_csv}")
    if not center_pred_tif.exists():
        raise FileNotFoundError(f"未找到 RFK 中心预测图：{center_pred_tif}")

    calib_df = pd.read_csv(calibration_csv)
    calib_cols = infer_columns(calib_df)

    calib_true = calib_df[calib_cols["true_col"]].values.astype(float)
    calib_pred = calib_df[calib_cols["pred_col"]].values.astype(float)
    calib_x = calib_df[calib_cols["x_col"]].values.astype(float)
    calib_y = calib_df[calib_cols["y_col"]].values.astype(float)
    calib_xy = np.column_stack([calib_x, calib_y])
    calib_abs_resid = np.abs(calib_true - calib_pred)

    eval_df = calib_df.copy()
    eval_cols = calib_cols
    eval_true = calib_true.copy()
    eval_pred = calib_pred.copy()
    eval_x = calib_x.copy()
    eval_y = calib_y.copy()
    eval_xy = calib_xy.copy()

    # 3. 样点级区间与指标
    log("计算样点级 GCP 区间与指标 ...")
    qhat_eval = compute_local_qhat(
        calib_xy=calib_xy,
        calib_resid=calib_abs_resid,
        target_xy=eval_xy,
        alpha=alpha,
        neighbors=neighbors,
        bandwidth_scale=bandwidth_scale,
        min_bandwidth=min_bandwidth,
    )

    eval_lower = eval_pred - qhat_eval
    eval_upper = eval_pred + qhat_eval

    metrics = compute_metrics(
        y_true=eval_true,
        lower=eval_lower,
        upper=eval_upper,
        alpha=alpha,
        ccb_bins=CFG.ccb_bins,
    )

    sample_df = pd.DataFrame(
        {
            "x_32648": eval_x,
            "y_32648": eval_y,
            "y_true": eval_true,
            "pred_center": eval_pred,
            "qhat": qhat_eval,
            "interval_lower": eval_lower,
            "interval_upper": eval_upper,
            "interval_width": eval_upper - eval_lower,
            "covered": ((eval_true >= eval_lower) & (eval_true <= eval_upper)).astype(int),
        }
    )
    sample_df.to_csv(INTERNAL_DIR / "gcp_sample_intervals.csv", index=False, encoding="utf-8-sig")

    params = {
        "alpha": alpha,
        "neighbors": neighbors,
        "bandwidth_scale": bandwidth_scale,
        "min_bandwidth": min_bandwidth,
    }
    write_concise_report(metrics, params)
    log(f"指标报告输出完成：{REPORT_DIR / 'GCP指标报告.txt'}")

    # 4. 输出全区不确定性图
    log("开始输出全区不确定性图 ...")
    with rasterio.open(center_pred_tif) as src:
        center_arr = src.read(1).astype("float32")
        profile = src.profile.copy()
        nodata = src.nodata
        if nodata is not None:
            center_arr = np.where(center_arr == nodata, np.nan, center_arr)

    valid_mask = np.isfinite(center_arr)
    rows, cols = np.where(valid_mask)
    xs, ys = xy(profile["transform"], rows, cols, offset="center")
    grid_xy = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])

    qhat_grid = compute_local_qhat(
        calib_xy=calib_xy,
        calib_resid=calib_abs_resid,
        target_xy=grid_xy,
        alpha=alpha,
        neighbors=neighbors,
        bandwidth_scale=bandwidth_scale,
        min_bandwidth=min_bandwidth,
    )

    lower_arr = np.full(center_arr.shape, np.nan, dtype="float32")
    upper_arr = np.full(center_arr.shape, np.nan, dtype="float32")
    width_arr = np.full(center_arr.shape, np.nan, dtype="float32")
    qhat_arr = np.full(center_arr.shape, np.nan, dtype="float32")

    center_vals = center_arr[rows, cols].astype(float)
    lower_arr[rows, cols] = (center_vals - qhat_grid).astype("float32")
    upper_arr[rows, cols] = (center_vals + qhat_grid).astype("float32")
    width_arr[rows, cols] = (2.0 * qhat_grid).astype("float32")
    qhat_arr[rows, cols] = qhat_grid.astype("float32")

    center_tif = build_timestamp_path(MAP_DIR / "GCP中心预测图.tif")
    lower_tif = build_timestamp_path(MAP_DIR / "GCP下界图.tif")
    upper_tif = build_timestamp_path(MAP_DIR / "GCP上界图.tif")
    width_tif = build_timestamp_path(MAP_DIR / "GCP不确定性图_区间宽度.tif")
    qhat_tif = build_timestamp_path(MAP_DIR / "GCP局地qhat图.tif")

    center_tif = array_to_tif(center_arr, center_tif, profile)
    lower_tif = array_to_tif(lower_arr, lower_tif, profile)
    upper_tif = array_to_tif(upper_arr, upper_tif, profile)
    width_tif = array_to_tif(width_arr, width_tif, profile)
    qhat_tif = array_to_tif(qhat_arr, qhat_tif, profile)

    # 5. 内部元数据
    meta = {
        "method": "GeoCP-like local conformal prediction",
        "alpha": alpha,
        "neighbors": neighbors,
        "bandwidth_scale": bandwidth_scale,
        "min_bandwidth": min_bandwidth,
        "inputs": {
            "calibration_csv": str(calibration_csv),
            "evaluation_csv": str(calibration_csv),
            "center_pred_tif": str(center_pred_tif),
        },
        "metrics": metrics,
        "n_calibration": int(len(calib_df)),
        "n_evaluation": int(len(eval_df)),
        "n_valid_pixels": int(valid_mask.sum()),
        "outputs": {
            "report_txt": str(REPORT_DIR / "GCP指标报告.txt"),
            "report_json": str(REPORT_DIR / "GCP指标报告.json"),
            "center_tif": str(center_tif),
            "lower_tif": str(lower_tif),
            "upper_tif": str(upper_tif),
            "width_tif": str(width_tif),
            "qhat_tif": str(qhat_tif),
        },
    }
    with open(INTERNAL_DIR / "gcp_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(STATE_DIR / "gcp_manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": "GCP",
            "mode": "formal_fixed",
            "rfk_upstream_manifest": rfk_manifest,
            "params": {
                "alpha": alpha,
                "neighbors": neighbors,
                "bandwidth_scale": bandwidth_scale,
                "min_bandwidth": min_bandwidth,
            },
            "metrics": metrics,
            "outputs": meta["outputs"],
        }, f, ensure_ascii=False, indent=2)

    log("全部完成。")
    log(f"- 指标报告：{REPORT_DIR / 'GCP指标报告.txt'}")
    log(f"- 不确定性图：{width_tif}")


if __name__ == "__main__":
    main()