#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
成都市土壤有机质 GCP 可搜索正式版
====================================
目标：
1. 直接读取 RFK 输出的 manifest、strict_groupkfold_oof.csv、中心预测图；
2. 支持 alpha / neighbors / bandwidth_scale / min_bandwidth 参数搜索；
3. 记住历史最优参数；
4. latest_run 每次覆盖；只有当新结果优于历史最优时，才提升为 current_best。

说明：
- 无法保证每次重算都一定更好；
- 代码通过“只晋升更优结果”的策略，保证 current_best 不会被更差结果覆盖。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
import math
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 配置
# =============================================================================

@dataclass(frozen=True)
class SearchConfig:
    # 简化搜索：只保留少量稳定候选组合，便于定位问题。
    n_iter: int = 5
    random_seed: int = 42
    alpha_choices: Tuple[float, ...] = (0.10, 0.15)
    neighbors_choices: Tuple[int, ...] = (45, 60, 80)
    bandwidth_scale_choices: Tuple[float, ...] = (0.6, 0.8)
    min_bandwidth_choices: Tuple[float, ...] = (200.0, 300.0)
    promote_only_if_better: bool = True


@dataclass(frozen=True)
class Config:
    rfk_out_root: Path = Path(r"E:\Agent_DSM\RFK_OUT")
    out_root: Path = Path(r"E:\Agent_DSM\GCP_OUT")
    ccb_bins: int = 10
    search: SearchConfig = SearchConfig()

    true_candidates: Tuple[str, ...] = (
        "som_value", "y_true", "true", "target", "obs", "observed"
    )
    pred_candidates: Tuple[str, ...] = (
        "strict_pred_final", "pred_final", "rfk_pred_final", "pred", "prediction"
    )
    x_candidates: Tuple[str, ...] = ("x_32648", "x", "utm_x", "easting")
    y_candidates: Tuple[str, ...] = ("y_32648", "y", "utm_y", "northing")


CFG = Config()

LATEST_DIR = CFG.out_root / "latest_run"
CURRENT_DIR = CFG.out_root / "current_best"
STATE_DIR = CFG.out_root / "_state"
TRIAL_DIR = CFG.out_root / "_trials"


# =============================================================================
# 通用函数
# =============================================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_float(v: object) -> float:
    return float(v) if v is not None else math.nan


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def pick_first_existing(df: pd.DataFrame, candidates: Sequence[str], role: str) -> str:
    col_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in col_map:
            return col_map[c.lower()]
    for c in candidates:
        hits = [cc for cc in col_map.keys() if c.lower() in cc]
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


# =============================================================================
# GCP 核心
# =============================================================================

def weighted_quantile(values: np.ndarray, quantile: float, sample_weight: np.ndarray) -> float:
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

        out[i] = weighted_quantile(nn_r, 1.0 - alpha, w)
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
    nominal = 1.0 - alpha

    picp = float(np.mean(covered))
    mpiw = float(np.mean(width))
    y_range = float(np.nanmax(y_true) - np.nanmin(y_true))
    nmpiw = float(mpiw / y_range) if y_range > 0 else np.nan

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

    under = np.maximum(lower - y_true, 0.0)
    over = np.maximum(y_true - upper, 0.0)
    interval_score = width + (2.0 / alpha) * under + (2.0 / alpha) * over
    is_mean = float(np.mean(interval_score))

    # 综合目标：越大越好
    composite_score = float(
        -is_mean
        - 4.0 * abs(picp - nominal)
        - 1.0 * ccb
        - 0.2 * nmpiw
    )

    return {
        "PICP": picp,
        "MPIW": mpiw,
        "NMPIW": nmpiw,
        "CCB": ccb,
        "IntervalScore": is_mean,
        "CompositeScore": composite_score,
        "nominal_coverage": nominal,
        "mean_width": mpiw,
        "median_width": float(np.median(width)),
    }


def objective_key(metrics: Dict[str, float]) -> Tuple[float, float, float]:
    return (
        safe_float(metrics.get("CompositeScore")),
        -safe_float(metrics.get("IntervalScore")),
        -abs(safe_float(metrics.get("PICP")) - safe_float(metrics.get("nominal_coverage"))),
    )


def is_better(candidate: Dict[str, float], incumbent: Optional[Dict[str, float]]) -> bool:
    if incumbent is None:
        return True
    return objective_key(candidate) > objective_key(incumbent)


# =============================================================================
# I/O
# =============================================================================


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

def load_rfk_manifest() -> Dict[str, object]:
    # 优先读取 latest_run，确保 GCP 默认接“本次最新 RFK 结果”；若不存在则回退 current_best
    latest_manifest = CFG.rfk_out_root / "latest_run" / "_state" / "rfk_manifest.json"
    current_manifest = CFG.rfk_out_root / "current_best" / "_state" / "rfk_manifest.json"
    if latest_manifest.exists():
        return load_json(latest_manifest)
    if current_manifest.exists():
        return load_json(current_manifest)
    raise FileNotFoundError("未找到 RFK manifest。请先运行 RFK_searchable_pipeline.py")


def load_previous_best_state() -> Optional[Dict[str, object]]:
    state_file = STATE_DIR / "gcp_best_state.json"
    if not state_file.exists():
        return None
    return load_json(state_file)


def load_rfk_inputs() -> Tuple[pd.DataFrame, Dict[str, str], Path, Dict[str, object]]:
    manifest = load_rfk_manifest()
    outputs = manifest["outputs"]
    oof_path = Path(outputs["strict_oof_csv"])
    center_pred_tif = Path(outputs["pred_tif"])
    if not oof_path.exists():
        raise FileNotFoundError(f"未找到 RFK strict OOF 文件：{oof_path}")
    if not center_pred_tif.exists():
        raise FileNotFoundError(f"未找到 RFK 全区预测图：{center_pred_tif}")

    df = pd.read_csv(oof_path)
    cols = infer_columns(df)
    return df, cols, center_pred_tif, manifest


# =============================================================================
# 搜索与输出
# =============================================================================

def build_search_space() -> List[Dict[str, object]]:
    """
    简化 GCP 参数搜索：改为少量可解释候选，而不是大范围随机采样。
    """
    candidates: List[Dict[str, object]] = [
        {
            "name": "baseline",
            "alpha": 0.10,
            "neighbors": 60,
            "bandwidth_scale": 0.6,
            "min_bandwidth": 200.0,
        },
        {
            "name": "wider_interval",
            "alpha": 0.15,
            "neighbors": 60,
            "bandwidth_scale": 0.6,
            "min_bandwidth": 200.0,
        },
        {
            "name": "more_neighbors",
            "alpha": 0.10,
            "neighbors": 80,
            "bandwidth_scale": 0.6,
            "min_bandwidth": 200.0,
        },
        {
            "name": "smoother_bandwidth",
            "alpha": 0.10,
            "neighbors": 60,
            "bandwidth_scale": 0.8,
            "min_bandwidth": 300.0,
        },
        {
            "name": "balanced_alt",
            "alpha": 0.15,
            "neighbors": 45,
            "bandwidth_scale": 0.8,
            "min_bandwidth": 300.0,
        },
    ]
    return candidates[: CFG.search.n_iter]


def evaluate_candidate(
    params: Dict[str, object],
    data_df: pd.DataFrame,
    cols: Dict[str, str],
) -> Tuple[Dict[str, object], pd.DataFrame]:
    y_true = data_df[cols["true_col"]].values.astype(float)
    y_pred = data_df[cols["pred_col"]].values.astype(float)
    x = data_df[cols["x_col"]].values.astype(float)
    y = data_df[cols["y_col"]].values.astype(float)
    xy_pts = np.column_stack([x, y])

    abs_resid = np.abs(y_true - y_pred)
    qhat = compute_local_qhat(
        calib_xy=xy_pts,
        calib_resid=abs_resid,
        target_xy=xy_pts,
        alpha=float(params["alpha"]),
        neighbors=int(params["neighbors"]),
        bandwidth_scale=float(params["bandwidth_scale"]),
        min_bandwidth=float(params["min_bandwidth"]),
    )
    lower = y_pred - qhat
    upper = y_pred + qhat
    metrics = compute_metrics(y_true=y_true, lower=lower, upper=upper, alpha=float(params["alpha"]), ccb_bins=CFG.ccb_bins)
    metrics["search_name"] = str(params.get("name", "trial"))
    metrics.update({
        "alpha": float(params["alpha"]),
        "neighbors": int(params["neighbors"]),
        "bandwidth_scale": float(params["bandwidth_scale"]),
        "min_bandwidth": float(params["min_bandwidth"]),
    })

    sample_df = pd.DataFrame({
        cols["x_col"]: x,
        cols["y_col"]: y,
        "y_true": y_true,
        "pred_center": y_pred,
        "qhat": qhat,
        "interval_lower": lower,
        "interval_upper": upper,
        "interval_width": upper - lower,
        "covered": ((y_true >= lower) & (y_true <= upper)).astype(int),
    })
    return metrics, sample_df


def write_trial_table(trials: List[Dict[str, object]], out_path: Path) -> None:
    pd.DataFrame(trials).to_csv(out_path, index=False, encoding="utf-8-sig")


def write_concise_report(metrics: Dict[str, object], params: Dict[str, object], report_dir: Path) -> None:
    payload = {
        "method": "GeoCP-like local conformal prediction",
        "objective": "CompositeScore",
        "params": params,
        "metrics": metrics,
    }
    write_json(report_dir / "GCP指标报告.json", payload)

    txt = f"""成都市土壤有机质 GCP 搜索报告
==============================
方法：GeoCP-like 空间加权共形预测
搜索目标：CompositeScore 最大化

最优参数：
alpha            = {params['alpha']}
neighbors        = {params['neighbors']}
bandwidth_scale  = {params['bandwidth_scale']}
min_bandwidth    = {params['min_bandwidth']}

核心指标：
PICP            = {metrics['PICP']:.6f}
MPIW            = {metrics['MPIW']:.6f}
NMPIW           = {metrics['NMPIW']:.6f}
CCB             = {metrics['CCB']:.6f}
IntervalScore   = {metrics['IntervalScore']:.6f}
CompositeScore  = {metrics['CompositeScore']:.6f}

说明：
- latest_run 每次都会被覆盖；
- current_best 只有在新结果更优时才会被覆盖；
- 本脚本直接读取 RFK 的 current_best / latest_run manifest。
"""
    (report_dir / "GCP指标报告.txt").write_text(txt, encoding="utf-8")


def output_fullarea_maps(
    center_pred_tif: Path,
    calib_xy: np.ndarray,
    calib_resid: np.ndarray,
    params: Dict[str, object],
    map_dir: Path,
) -> Dict[str, str]:
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
        calib_resid=calib_resid,
        target_xy=grid_xy,
        alpha=float(params["alpha"]),
        neighbors=int(params["neighbors"]),
        bandwidth_scale=float(params["bandwidth_scale"]),
        min_bandwidth=float(params["min_bandwidth"]),
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

    center_tif = map_dir / "GCP中心预测图.tif"
    lower_tif = map_dir / "GCP下界图.tif"
    upper_tif = map_dir / "GCP上界图.tif"
    width_tif = map_dir / "GCP不确定性图_区间宽度.tif"
    qhat_tif = map_dir / "GCP局地qhat图.tif"

    center_tif = array_to_tif(center_arr, center_tif, profile)
    lower_tif = array_to_tif(lower_arr, lower_tif, profile)
    upper_tif = array_to_tif(upper_arr, upper_tif, profile)
    width_tif = array_to_tif(width_arr, width_tif, profile)
    qhat_tif = array_to_tif(qhat_arr, qhat_tif, profile)

    return {
        "center_tif": str(center_tif),
        "lower_tif": str(lower_tif),
        "upper_tif": str(upper_tif),
        "width_tif": str(width_tif),
        "qhat_tif": str(qhat_tif),
    }


def build_manifest(
    metrics: Dict[str, object],
    params: Dict[str, object],
    map_paths: Dict[str, str],
    root_dir: Path,
    rfk_manifest: Dict[str, object],
) -> Dict[str, object]:
    return {
        "model": "GCP",
        "method": "GeoCP-like local conformal prediction",
        "rfk_upstream_manifest": rfk_manifest,
        "params": params,
        "metrics": metrics,
        "outputs": {
            "report_txt": str(root_dir / "01_指标报告" / "GCP指标报告.txt"),
            "report_json": str(root_dir / "01_指标报告" / "GCP指标报告.json"),
            "sample_intervals_csv": str(root_dir / "_internal" / "gcp_sample_intervals.csv"),
            **map_paths,
        },
    }


def promote_or_keep(latest_metrics: Dict[str, object], previous_state: Optional[Dict[str, object]]) -> Tuple[bool, Dict[str, object]]:
    prev_metrics = previous_state.get("metrics") if previous_state else None
    improved = is_better(latest_metrics, prev_metrics)
    should_promote = improved or (not CFG.search.promote_only_if_better)
    state_payload = {
        "metrics": latest_metrics if should_promote or previous_state is None else previous_state["metrics"],
        "manifest_path": str(CURRENT_DIR / "_state" / "gcp_manifest.json") if should_promote else previous_state["manifest_path"],
        "last_attempt_manifest_path": str(LATEST_DIR / "_state" / "gcp_manifest.json"),
        "improved": improved,
    }
    return should_promote, state_payload


# =============================================================================
# 主流程
# =============================================================================

def main() -> None:
    log("开始 GCP 参数搜索与正式建模 ...")
    ensure_dir(CFG.out_root)
    ensure_dir(STATE_DIR)
    ensure_dir(TRIAL_DIR)
    reset_dir(LATEST_DIR)
    for sub in ["01_指标报告", "02_不确定性图", "_internal", "_state"]:
        ensure_dir(LATEST_DIR / sub)
    ensure_dir(CURRENT_DIR)

    data_df, cols, center_pred_tif, rfk_manifest = load_rfk_inputs()
    previous_state = load_previous_best_state()
    previous_metrics = previous_state.get("metrics") if previous_state else None
    if previous_metrics:
        log(f"检测到历史最优 CompositeScore = {previous_metrics['CompositeScore']:.6f}")

    search_candidates = build_search_space()
    best_metrics: Optional[Dict[str, object]] = None
    best_sample_df: Optional[pd.DataFrame] = None
    trials: List[Dict[str, object]] = []

    for i, params in enumerate(search_candidates, start=1):
        log(f"[GCP] trial {i}/{len(search_candidates)}: {params}")
        metrics, sample_df = evaluate_candidate(params=params, data_df=data_df, cols=cols)
        metrics["trial_id"] = i
        trials.append(metrics)
        if is_better(metrics, best_metrics):
            best_metrics = metrics
            best_sample_df = sample_df
            log(
                f"当前最佳 trial={i} | CompositeScore={metrics['CompositeScore']:.6f} | "
                f"PICP={metrics['PICP']:.6f}"
            )

    if best_metrics is None or best_sample_df is None:
        raise RuntimeError("GCP 搜索失败，没有获得有效候选结果。")

    write_trial_table(trials, TRIAL_DIR / "gcp_search_trials.csv")

    latest_report_dir = LATEST_DIR / "01_指标报告"
    latest_map_dir = LATEST_DIR / "02_不确定性图"
    latest_internal_dir = LATEST_DIR / "_internal"
    latest_state_dir = LATEST_DIR / "_state"

    best_params = {
        "alpha": best_metrics["alpha"],
        "neighbors": best_metrics["neighbors"],
        "bandwidth_scale": best_metrics["bandwidth_scale"],
        "min_bandwidth": best_metrics["min_bandwidth"],
    }
    write_concise_report(best_metrics, best_params, latest_report_dir)
    log("GCP指标报告输出完成。")
    best_sample_df.to_csv(latest_internal_dir / "gcp_sample_intervals.csv", index=False, encoding="utf-8-sig")

    calib_xy = np.column_stack([
        data_df[cols["x_col"]].values.astype(float),
        data_df[cols["y_col"]].values.astype(float),
    ])
    calib_resid = np.abs(
        data_df[cols["true_col"]].values.astype(float) - data_df[cols["pred_col"]].values.astype(float)
    )
    log("开始输出全区 GCP 不确定性图 ...")
    map_paths = output_fullarea_maps(
        center_pred_tif=center_pred_tif,
        calib_xy=calib_xy,
        calib_resid=calib_resid,
        params=best_params,
        map_dir=latest_map_dir,
    )
    latest_manifest = build_manifest(best_metrics, best_params, map_paths, LATEST_DIR, rfk_manifest)
    write_json(latest_state_dir / "gcp_manifest.json", latest_manifest)

    should_promote, state_payload = promote_or_keep(best_metrics, previous_state)
    if should_promote:
        if CURRENT_DIR.exists():
            shutil.rmtree(CURRENT_DIR)
        shutil.copytree(LATEST_DIR, CURRENT_DIR)
        state_payload["manifest_path"] = str(CURRENT_DIR / "_state" / "gcp_manifest.json")
        log("本次 GCP 结果优于历史最优，已覆盖 current_best。")
    else:
        log("本次 GCP 结果未优于历史最优，仅覆盖 latest_run，保留 current_best。")

    write_json(STATE_DIR / "gcp_best_state.json", state_payload)
    log(f"GCP 完成。latest_run 清单：{LATEST_DIR / '_state' / 'gcp_manifest.json'}")
    log(f"GCP 当前正式清单：{CURRENT_DIR / '_state' / 'gcp_manifest.json'}")


if __name__ == "__main__":
    main()
