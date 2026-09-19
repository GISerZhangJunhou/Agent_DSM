#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
成都市土壤有机质 RFK 可搜索正式版
====================================
目标：
1. 支持 RF 参数、Kriging 参数、特征子集搜索；
2. 输出可直接供 GCP 使用的 strict_groupkfold_oof.csv 与 manifest；
3. 记住历史最优参数；
4. latest_run 每次覆盖；只有当新结果优于历史最优时，才提升为 current_best。

说明：
- 无法保证每次重算都一定更好；
- 代码通过“只晋升更优结果”的策略，保证 current_best 不会被更差结果覆盖。
"""

from __future__ import annotations

import json
import math
import shutil
import warnings
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import rasterio
from pykrige.ok import OrdinaryKriging
from rasterio.warp import Resampling, reproject
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 配置
# =============================================================================

@dataclass(frozen=True)
class SearchConfig:
    # 简化搜索：少量、稳定、可解释的候选，避免过重的随机搜索。
    n_iter: int = 6
    target_r2: float = 0.60
    promote_only_if_better: bool = True
    random_seed: int = 42
    metric_name: str = "strict_final_r2"
    tie_breaker_name: str = "strict_final_rmse"
    topk_choices: Tuple[int, ...] = (8, 11, 14)


@dataclass(frozen=True)
class Config:
    csv_path: Path = Path(r"E:/Agent_DSM/your_training.csv")
    out_root: Path = Path(r"E:/Agent_DSM/RFK_OUT")

    target_col: str = "som_value"
    x_col: str = "x_32648"
    y_col: str = "y_32648"
    strict_group_col: str = "spatial_block"

    id_cols: Tuple[str, ...] = (
        "point_id", "longitude", "latitude", "x_32648", "y_32648", "grid_block_id"
    )
    leak_cols: Tuple[str, ...] = ("SOM_topsoil_0_30m_masked_32648_1000m",)

    fine_grid_nx: int = 3
    fine_grid_ny: int = 3
    repeated_splits: int = 10
    strict_splits: int = 5
    test_size: float = 0.30
    base_random_seed: int = 42

    # 候选特征池：默认在样点表中自动推断；若非空则强制限制在该集合内。
    candidate_features: Tuple[str, ...] = tuple()

    # 栅格目录
    raster_root: Path = Path(r"E:/Agent_DSM/your_covariate_rasters")
    forced_paths: Dict[str, str] = field(default_factory=lambda: {
        "pH": r"E:/Agent_DSM/your_covariate_rasters/pH.tif",
        "sin_aspect": r"E:/Agent_DSM/your_covariate_rasters/sin_aspect_32648_1km.tif",
        "tpi_3x3": r"E:/Agent_DSM/your_covariate_rasters/tpi_3x3_32648_1km.tif",
    })

    search: SearchConfig = SearchConfig()


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


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def score_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "medae": float(median_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
    }


def safe_float(v: object) -> float:
    return float(v) if v is not None else math.nan


def create_fine_spatial_groups(df: pd.DataFrame) -> np.ndarray:
    x = df[CFG.x_col].values.astype(float)
    y = df[CFG.y_col].values.astype(float)

    x_edges = np.linspace(np.nanmin(x), np.nanmax(x), CFG.fine_grid_nx + 1)
    y_edges = np.linspace(np.nanmin(y), np.nanmax(y), CFG.fine_grid_ny + 1)

    x_bin = np.digitize(x, x_edges[1:-1], right=False)
    y_bin = np.digitize(y, y_edges[1:-1], right=False)
    return (y_bin * CFG.fine_grid_nx + x_bin).astype(int)


def split_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    code_cols, cont_cols = [], []
    for c in X.columns:
        nunique = X[c].nunique(dropna=True)
        if pd.api.types.is_integer_dtype(X[c]) or nunique <= 20:
            code_cols.append(c)
        else:
            cont_cols.append(c)
    return code_cols, cont_cols


def make_preprocessor(X_train: pd.DataFrame, code_cols: List[str], cont_cols: List[str]) -> Dict[str, object]:
    fill_map: Dict[str, object] = {}
    for c in code_cols:
        mode = X_train[c].mode(dropna=True)
        fill_map[c] = mode.iloc[0] if len(mode) else 0
    for c in cont_cols:
        fill_map[c] = float(X_train[c].median(skipna=True))
    return {"code_cols": code_cols, "cont_cols": cont_cols, "fill_map": fill_map}


def apply_preprocessor(X: pd.DataFrame, state: Dict[str, object]) -> pd.DataFrame:
    X2 = X.copy()
    for c, v in state["fill_map"].items():
        if c in X2.columns:
            X2[c] = X2[c].fillna(v)
    return X2


def objective_key(metrics: Dict[str, float]) -> Tuple[float, float, float]:
    # 主目标：strict_final_r2 越大越好
    # 次目标：strict_final_rmse 越小越好
    # 再次目标：final_r2 越大越好
    return (
        safe_float(metrics.get("strict_final_r2")),
        -safe_float(metrics.get("strict_final_rmse")),
        safe_float(metrics.get("final_r2")),
    )


def is_better(candidate: Dict[str, float], incumbent: Optional[Dict[str, float]]) -> bool:
    if incumbent is None:
        return True
    return objective_key(candidate) > objective_key(incumbent)


# =============================================================================
# 数据与特征
# =============================================================================

def load_training_table() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    df = pd.read_csv(CFG.csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    excluded = set(CFG.id_cols) | set(CFG.leak_cols) | {CFG.target_col, CFG.strict_group_col}
    numeric_cols = [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]

    if CFG.candidate_features:
        candidate_features = [c for c in CFG.candidate_features if c in numeric_cols]
    else:
        candidate_features = numeric_cols

    if len(candidate_features) < 6:
        raise ValueError("可搜索候选特征过少，至少需要 6 个数值特征。")

    X = df[candidate_features].copy()
    return df, X, candidate_features


def get_feature_ranking(df_full: pd.DataFrame, X_full: pd.DataFrame) -> List[str]:
    X0 = X_full.copy()
    for c in X0.columns:
        if X0[c].isna().any():
            if pd.api.types.is_integer_dtype(X0[c]) or X0[c].nunique(dropna=True) <= 20:
                mode = X0[c].mode(dropna=True)
                X0[c] = X0[c].fillna(mode.iloc[0] if len(mode) else 0)
            else:
                X0[c] = X0[c].fillna(float(X0[c].median(skipna=True)))

    pilot = RandomForestRegressor(
        n_estimators=600,
        max_depth=16,
        max_features="sqrt",
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=CFG.base_random_seed,
        n_jobs=-1,
    )
    pilot.fit(X0, df_full[CFG.target_col].values.astype(float))
    ranking = pd.Series(pilot.feature_importances_, index=X0.columns).sort_values(ascending=False)
    return ranking.index.tolist()


def choose_feature_subset(ranked_features: Sequence[str], topk: int) -> List[str]:
    topk = min(max(6, int(topk)), len(ranked_features))
    return list(ranked_features[:topk])


# =============================================================================
# RFK 核心
# =============================================================================

def fit_rfk(
    train_df: pd.DataFrame,
    X_train: pd.DataFrame,
    rf_params: Dict[str, object],
    variogram_model: str,
    nlags: int,
) -> Tuple[RandomForestRegressor, Dict[str, object], OrdinaryKriging]:
    y_train = train_df[CFG.target_col].values.astype(float)
    code_cols, cont_cols = split_feature_types(X_train)
    prep_state = make_preprocessor(X_train, code_cols, cont_cols)
    X_train_p = apply_preprocessor(X_train, prep_state)

    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_train_p, y_train)
    rf_pred = rf.predict(X_train_p)
    resid = y_train - rf_pred

    ok = OrdinaryKriging(
        x=train_df[CFG.x_col].values.astype(float),
        y=train_df[CFG.y_col].values.astype(float),
        z=resid.astype(float),
        variogram_model=variogram_model,
        nlags=int(nlags),
        verbose=False,
        enable_plotting=False,
    )
    return rf, prep_state, ok


def predict_rfk(
    rf: RandomForestRegressor,
    prep_state: Dict[str, object],
    ok: OrdinaryKriging,
    val_df: pd.DataFrame,
    X_val: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_val_p = apply_preprocessor(X_val, prep_state)
    rf_pred = rf.predict(X_val_p).astype(float)
    krig_pred, _ = ok.execute(
        "points",
        val_df[CFG.x_col].values.astype(float),
        val_df[CFG.y_col].values.astype(float),
    )
    krig_pred = np.asarray(krig_pred, dtype=float)
    final_pred = rf_pred + krig_pred
    return rf_pred, krig_pred, final_pred


def run_one_split(
    df_full: pd.DataFrame,
    X_full: pd.DataFrame,
    features: Sequence[str],
    rf_params: Dict[str, object],
    variogram_model: str,
    nlags: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    train_df = df_full.iloc[train_idx].copy().reset_index(drop=False)
    val_df = df_full.iloc[val_idx].copy().reset_index(drop=False)
    X_train = X_full.iloc[train_idx][list(features)].copy().reset_index(drop=True)
    X_val = X_full.iloc[val_idx][list(features)].copy().reset_index(drop=True)

    rf, prep_state, ok = fit_rfk(train_df, X_train, rf_params, variogram_model, nlags)
    rf_pred, krig_pred, final_pred = predict_rfk(rf, prep_state, ok, val_df, X_val)

    out = val_df.copy()
    out["pred_final"] = final_pred
    out["pred_rf"] = rf_pred
    out["pred_krig"] = krig_pred
    metrics = score_dict(val_df[CFG.target_col].values.astype(float), final_pred)
    return out, metrics


def run_repeated_spatial_holdout(
    df_full: pd.DataFrame,
    X_full: pd.DataFrame,
    features: Sequence[str],
    rf_params: Dict[str, object],
    variogram_model: str,
    nlags: int,
    random_seed: int,
) -> Dict[str, float]:
    fine_groups = create_fine_spatial_groups(df_full)
    splitter = GroupShuffleSplit(
        n_splits=CFG.repeated_splits,
        test_size=CFG.test_size,
        random_state=int(random_seed),
    )

    oof_sum = np.zeros(len(df_full), dtype=float)
    oof_count = np.zeros(len(df_full), dtype=int)
    fold_r2: List[float] = []

    for train_idx, val_idx in splitter.split(X_full, df_full[CFG.target_col], groups=fine_groups):
        pred_df, metrics = run_one_split(
            df_full=df_full,
            X_full=X_full,
            features=features,
            rf_params=rf_params,
            variogram_model=variogram_model,
            nlags=nlags,
            train_idx=train_idx,
            val_idx=val_idx,
        )
        preds = pred_df["pred_final"].values.astype(float)
        oof_sum[val_idx] += preds
        oof_count[val_idx] += 1
        fold_r2.append(metrics["r2"])

    valid = oof_count > 0
    final_pred = np.full(len(df_full), np.nan, dtype=float)
    final_pred[valid] = oof_sum[valid] / oof_count[valid]
    met = score_dict(df_full.loc[valid, CFG.target_col].values.astype(float), final_pred[valid])
    return {
        "final_r2": met["r2"],
        "final_rmse": met["rmse"],
        "final_mae": met["mae"],
        "final_medae": met["medae"],
        "final_bias": met["bias"],
        "coverage_points": int(valid.sum()),
        "mean_fold_r2": float(np.mean(fold_r2)),
        "std_fold_r2": float(np.std(fold_r2)),
    }


def run_strict_groupkfold(
    df_full: pd.DataFrame,
    X_full: pd.DataFrame,
    features: Sequence[str],
    rf_params: Dict[str, object],
    variogram_model: str,
    nlags: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    groups = df_full[CFG.strict_group_col].values
    splitter = GroupKFold(n_splits=CFG.strict_splits)
    oof_pred = np.full(len(df_full), np.nan, dtype=float)
    oof_rf = np.full(len(df_full), np.nan, dtype=float)
    oof_krig = np.full(len(df_full), np.nan, dtype=float)
    fold_r2: List[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        splitter.split(X_full, df_full[CFG.target_col], groups=groups), start=1
    ):
        pred_df, metrics = run_one_split(
            df_full=df_full,
            X_full=X_full,
            features=features,
            rf_params=rf_params,
            variogram_model=variogram_model,
            nlags=nlags,
            train_idx=train_idx,
            val_idx=val_idx,
        )
        oof_pred[val_idx] = pred_df["pred_final"].values.astype(float)
        oof_rf[val_idx] = pred_df["pred_rf"].values.astype(float)
        oof_krig[val_idx] = pred_df["pred_krig"].values.astype(float)
        fold_r2.append(metrics["r2"])
        log(f"strict GroupKFold | fold={fold_idx}/{CFG.strict_splits} | r2={metrics['r2']:.6f}")

    valid = np.isfinite(oof_pred)
    met = score_dict(df_full.loc[valid, CFG.target_col].values.astype(float), oof_pred[valid])
    oof_df = df_full.copy()
    oof_df["strict_pred_final"] = oof_pred
    oof_df["strict_pred_rf"] = oof_rf
    oof_df["strict_pred_krig"] = oof_krig
    oof_df["strict_abs_error"] = np.abs(oof_df[CFG.target_col].values.astype(float) - oof_pred)
    oof_df["fold_valid"] = valid.astype(int)

    summary = {
        "strict_final_r2": met["r2"],
        "strict_final_rmse": met["rmse"],
        "strict_final_mae": met["mae"],
        "strict_final_medae": met["medae"],
        "strict_final_bias": met["bias"],
        "strict_coverage_points": int(valid.sum()),
        "strict_mean_fold_r2": float(np.mean(fold_r2)),
        "strict_std_fold_r2": float(np.std(fold_r2)),
    }
    return summary, oof_df


def run_single_holdout(
    df_full: pd.DataFrame,
    X_full: pd.DataFrame,
    features: Sequence[str],
    rf_params: Dict[str, object],
    variogram_model: str,
    nlags: int,
    random_seed: int,
) -> Dict[str, float]:
    fine_groups = create_fine_spatial_groups(df_full)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=CFG.test_size,
        random_state=int(random_seed),
    )
    train_idx, val_idx = next(splitter.split(X_full, df_full[CFG.target_col], groups=fine_groups))
    pred_df, _ = run_one_split(
        df_full=df_full,
        X_full=X_full,
        features=features,
        rf_params=rf_params,
        variogram_model=variogram_model,
        nlags=nlags,
        train_idx=train_idx,
        val_idx=val_idx,
    )
    y_true = pred_df[CFG.target_col].values.astype(float)
    y_pred = pred_df["pred_final"].values.astype(float)
    met = score_dict(y_true, y_pred)
    return {
        "single_holdout_r2": met["r2"],
        "single_holdout_rmse": met["rmse"],
        "single_holdout_mae": met["mae"],
        "single_holdout_medae": met["medae"],
        "single_holdout_bias": met["bias"],
    }


def fit_full_model(
    df_full: pd.DataFrame,
    X_full: pd.DataFrame,
    features: Sequence[str],
    rf_params: Dict[str, object],
    variogram_model: str,
    nlags: int,
    model_dir: Path,
) -> Dict[str, object]:
    y = df_full[CFG.target_col].values.astype(float)
    X_sel = X_full[list(features)].copy()
    code_cols, cont_cols = split_feature_types(X_sel)
    prep_state = make_preprocessor(X_sel, code_cols, cont_cols)
    X_p = apply_preprocessor(X_sel, prep_state)

    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_p, y)
    rf_pred = rf.predict(X_p)
    resid = y - rf_pred

    train_resid_df = df_full[[CFG.x_col, CFG.y_col]].copy()
    train_resid_df["residual"] = resid
    train_resid_df.to_csv(model_dir / "rfk_training_residuals.csv", index=False, encoding="utf-8-sig")

    bundle = {
        "rf_model": rf,
        "preprocessor": prep_state,
        "features": list(features),
        "rf_params": rf_params,
        "variogram_model": variogram_model,
        "nlags": int(nlags),
    }
    joblib.dump(bundle, model_dir / "rf_model.joblib")
    return bundle


# =============================================================================
# 栅格出图
# =============================================================================

def resolve_feature_path(feature: str) -> Path:
    if feature in CFG.forced_paths:
        p = Path(CFG.forced_paths[feature])
        if p.exists():
            return p

    candidates = [
        CFG.raster_root / f"{feature}.tif",
        CFG.raster_root / f"{feature}_32648_1km.tif",
    ]
    for p in candidates:
        if p.exists():
            return p

    matches = [p for p in CFG.raster_root.glob("*.tif") if p.stem.lower() == feature.lower()]
    if matches:
        return matches[0]

    raise FileNotFoundError(f"未找到特征栅格：{feature}")


def choose_resampling(feature: str) -> Resampling:
    return Resampling.nearest if feature in {"LULCcd"} else Resampling.bilinear


def read_and_align_raster(path: Path, dst_profile: Optional[dict], resampling: Resampling):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)

        if dst_profile is None:
            profile = src.profile.copy()
            profile.update(dtype="float32", count=1, compress="lzw", nodata=np.nan)
            return arr, profile

        dst = np.full((dst_profile["height"], dst_profile["width"]), np.nan, dtype="float32")
        reproject(
            source=arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_profile["transform"],
            dst_crs=dst_profile["crs"],
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=resampling,
        )
        return dst, dst_profile.copy()


def build_feature_stack(features: Sequence[str]) -> Tuple[Dict[str, np.ndarray], dict]:
    feature_paths = {f: resolve_feature_path(f) for f in features}
    reference_feature = list(features)[-1]
    ref_path = feature_paths[reference_feature]
    ref_arr, ref_profile = read_and_align_raster(ref_path, None, choose_resampling(reference_feature))

    feature_arrays = {reference_feature: ref_arr}
    for feat, path in feature_paths.items():
        if feat == reference_feature:
            continue
        arr, _ = read_and_align_raster(path, ref_profile, choose_resampling(feat))
        feature_arrays[feat] = arr
    return feature_arrays, ref_profile


def build_valid_mask(feature_arrays: Dict[str, np.ndarray], features: Sequence[str]) -> np.ndarray:
    mask = np.ones_like(next(iter(feature_arrays.values())), dtype=bool)
    for feat in features:
        mask &= np.isfinite(feature_arrays[feat])
    return mask


def array_to_tif(arr: np.ndarray, out_path: Path, profile: dict) -> None:
    out_profile = profile.copy()
    out_profile.update(driver="GTiff", dtype="float32", count=1, compress="lzw", nodata=np.nan)
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(arr.astype("float32"), 1)


def output_prediction_map(bundle: Dict[str, object], map_dir: Path, model_dir: Path) -> Path:
    features = bundle["features"]
    log("开始输出全区 RFK 预测图 ...")
    feature_arrays, ref_profile = build_feature_stack(features)
    valid_mask = build_valid_mask(feature_arrays, features)
    rows, cols = np.where(valid_mask)
    log(f"有效像元数 = {int(valid_mask.sum())}")

    X_grid = pd.DataFrame({f: feature_arrays[f][rows, cols] for f in features})
    X_grid_p = apply_preprocessor(X_grid, bundle["preprocessor"])
    rf_model = bundle["rf_model"]
    rf_pred = rf_model.predict(X_grid_p).astype(float)

    resid_df = pd.read_csv(model_dir / "rfk_training_residuals.csv")
    ok = OrdinaryKriging(
        x=resid_df[CFG.x_col].values.astype(float),
        y=resid_df[CFG.y_col].values.astype(float),
        z=resid_df["residual"].values.astype(float),
        variogram_model=bundle["variogram_model"],
        nlags=int(bundle["nlags"]),
        verbose=False,
        enable_plotting=False,
    )

    xs, ys = rasterio.transform.xy(ref_profile["transform"], rows, cols, offset="center")
    krig_pred, _ = ok.execute("points", np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))
    krig_pred = np.asarray(krig_pred, dtype=float)
    final_pred = rf_pred + krig_pred

    pred_arr = np.full(valid_mask.shape, np.nan, dtype="float32")
    pred_arr[rows, cols] = final_pred.astype("float32")

    pred_tif = map_dir / "RFK最终预测图.tif"
    array_to_tif(pred_arr, pred_tif, ref_profile)
    return pred_tif


# =============================================================================
# 搜索
# =============================================================================

def build_search_space(ranked_features: Sequence[str]) -> List[Dict[str, object]]:
    """
    简化 RFK 参数搜索：不再做大范围随机采样，只评估少量稳定候选。
    这样更容易排查问题，也能明显缩短单次运行时间。
    """
    topk_mid = min(11, len(ranked_features))
    topk_small = min(8, len(ranked_features))
    topk_large = min(14, len(ranked_features))

    candidates: List[Dict[str, object]] = [
        {
            "name": "baseline",
            "n_estimators": 800,
            "max_depth": 16,
            "max_features": 1.0,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "bootstrap": True,
            "variogram_model": "spherical",
            "nlags": 8,
            "topk": topk_mid,
        },
        {
            "name": "lighter_forest",
            "n_estimators": 500,
            "max_depth": 12,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "bootstrap": True,
            "variogram_model": "spherical",
            "nlags": 6,
            "topk": topk_small,
        },
        {
            "name": "deeper_forest",
            "n_estimators": 1000,
            "max_depth": 20,
            "max_features": 0.8,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "bootstrap": True,
            "variogram_model": "spherical",
            "nlags": 10,
            "topk": topk_mid,
        },
        {
            "name": "robust_leaf2",
            "n_estimators": 800,
            "max_depth": 16,
            "max_features": 0.8,
            "min_samples_leaf": 2,
            "min_samples_split": 4,
            "bootstrap": True,
            "variogram_model": "spherical",
            "nlags": 8,
            "topk": topk_mid,
        },
        {
            "name": "gaussian_try",
            "n_estimators": 800,
            "max_depth": 16,
            "max_features": 1.0,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "bootstrap": True,
            "variogram_model": "gaussian",
            "nlags": 8,
            "topk": topk_mid,
        },
        {
            "name": "more_features",
            "n_estimators": 800,
            "max_depth": 16,
            "max_features": 1.0,
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "bootstrap": True,
            "variogram_model": "spherical",
            "nlags": 8,
            "topk": topk_large,
        },
    ]

    deduped: List[Dict[str, object]] = []
    seen = set()
    for p in candidates:
        key = tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in sorted(p.items()) if k != "name")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped[: CFG.search.n_iter]


def evaluate_candidate(
    trial_id: int,
    params: Dict[str, object],
    df_full: pd.DataFrame,
    X_full: pd.DataFrame,
    ranked_features: Sequence[str],
) -> Tuple[Dict[str, object], pd.DataFrame]:
    features = choose_feature_subset(ranked_features, int(params["topk"]))
    rf_params = {
        "n_estimators": int(params["n_estimators"]),
        "max_depth": None if params["max_depth"] is None else int(params["max_depth"]),
        "max_features": params["max_features"],
        "min_samples_leaf": int(params["min_samples_leaf"]),
        "min_samples_split": int(params["min_samples_split"]),
        "bootstrap": bool(params["bootstrap"]),
        "n_jobs": -1,
        "random_state": int(CFG.search.random_seed),
    }
    variogram_model = str(params["variogram_model"])
    nlags = int(params["nlags"])

    repeated_summary = run_repeated_spatial_holdout(
        df_full=df_full,
        X_full=X_full,
        features=features,
        rf_params=rf_params,
        variogram_model=variogram_model,
        nlags=nlags,
        random_seed=CFG.search.random_seed,
    )
    strict_summary, strict_oof = run_strict_groupkfold(
        df_full=df_full,
        X_full=X_full,
        features=features,
        rf_params=rf_params,
        variogram_model=variogram_model,
        nlags=nlags,
    )
    single_summary = run_single_holdout(
        df_full=df_full,
        X_full=X_full,
        features=features,
        rf_params=rf_params,
        variogram_model=variogram_model,
        nlags=nlags,
        random_seed=CFG.search.random_seed,
    )

    metrics: Dict[str, object] = {}
    metrics.update(repeated_summary)
    metrics.update(strict_summary)
    metrics.update(single_summary)
    metrics["trial_id"] = int(trial_id)
    metrics["n_features"] = int(len(features))
    metrics["features"] = features
    metrics["rf_params"] = rf_params
    metrics["variogram_model"] = variogram_model
    metrics["nlags"] = nlags
    metrics["search_name"] = str(params.get("name", f"trial_{trial_id}"))
    metrics["reached_target"] = bool(metrics["strict_final_r2"] >= CFG.search.target_r2)
    return metrics, strict_oof


# =============================================================================
# 状态、报告、输出
# =============================================================================

def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_previous_best_state() -> Optional[Dict[str, object]]:
    state_file = STATE_DIR / "rfk_best_state.json"
    if not state_file.exists():
        return None
    return json.loads(state_file.read_text(encoding="utf-8"))


def write_trial_table(trials: List[Dict[str, object]], out_path: Path) -> None:
    records: List[Dict[str, object]] = []
    for t in trials:
        r = dict(t)
        r["features"] = ", ".join(r.get("features", []))
        r["rf_params"] = json.dumps(r.get("rf_params", {}), ensure_ascii=False)
        records.append(r)
    pd.DataFrame(records).to_csv(out_path, index=False, encoding="utf-8-sig")


def write_concise_report(summary: Dict[str, object], report_dir: Path) -> None:
    simple = {
        "model": "RFK",
        "search_metric": "strict_final_r2",
        "target_r2": CFG.search.target_r2,
        "features": summary["features"],
        "rf_params": summary["rf_params"],
        "variogram_model": summary["variogram_model"],
        "nlags": summary["nlags"],
        "main_result": {
            "method": "重复空间7:3",
            "r2": summary["final_r2"],
            "rmse": summary["final_rmse"],
            "mae": summary["final_mae"],
        },
        "strict_diagnosis": {
            "method": "strict GroupKFold",
            "r2": summary["strict_final_r2"],
            "rmse": summary["strict_final_rmse"],
            "mae": summary["strict_final_mae"],
        },
        "single_holdout": {
            "method": "单次空间7:3",
            "r2": summary["single_holdout_r2"],
            "rmse": summary["single_holdout_rmse"],
            "mae": summary["single_holdout_mae"],
        },
    }
    write_json(report_dir / "RFK精度报告.json", simple)

    txt = f"""成都市土壤有机质 RFK 搜索报告
==============================
搜索目标：strict_final_r2 最大化
目标阈值：{CFG.search.target_r2}

最优特征（{len(summary['features'])}个）：
{", ".join(summary['features'])}

最优 RF 参数：
{json.dumps(summary['rf_params'], ensure_ascii=False)}

最优 Kriging 参数：
variogram_model = {summary['variogram_model']}
nlags = {summary['nlags']}

正式主结果（重复空间7:3）：
R²   = {summary['final_r2']:.6f}
RMSE = {summary['final_rmse']:.6f}
MAE  = {summary['final_mae']:.6f}

严格诊断（strict GroupKFold）：
R²   = {summary['strict_final_r2']:.6f}
RMSE = {summary['strict_final_rmse']:.6f}
MAE  = {summary['strict_final_mae']:.6f}

单次留档（空间7:3）：
R²   = {summary['single_holdout_r2']:.6f}
RMSE = {summary['single_holdout_rmse']:.6f}
MAE  = {summary['single_holdout_mae']:.6f}

说明：
- latest_run 每次都会被覆盖；
- current_best 只有在新结果更优时才会被覆盖；
- GCP 读取本次 RFK 输出的 manifest 与 strict_groupkfold_oof.csv。
"""
    (report_dir / "RFK精度报告.txt").write_text(txt, encoding="utf-8")


def promote_or_keep(
    latest_manifest: Dict[str, object],
    latest_metrics: Dict[str, object],
    previous_state: Optional[Dict[str, object]],
) -> Tuple[bool, Dict[str, object]]:
    prev_metrics = previous_state.get("metrics") if previous_state else None
    improved = is_better(latest_metrics, prev_metrics)
    should_promote = improved or (not CFG.search.promote_only_if_better)

    state_payload = {
        "metrics": latest_metrics if should_promote or previous_state is None else previous_state["metrics"],
        "manifest_path": str(CURRENT_DIR / "_state" / "rfk_manifest.json") if should_promote else previous_state["manifest_path"],
        "last_attempt_manifest_path": str(LATEST_DIR / "_state" / "rfk_manifest.json"),
        "improved": improved,
    }
    return should_promote, state_payload


def build_manifest(summary: Dict[str, object], pred_tif: Path, root_dir: Path) -> Dict[str, object]:
    return {
        "model": "RFK",
        "target_col": CFG.target_col,
        "x_col": CFG.x_col,
        "y_col": CFG.y_col,
        "strict_group_col": CFG.strict_group_col,
        "features": summary["features"],
        "rf_params": summary["rf_params"],
        "variogram_model": summary["variogram_model"],
        "nlags": summary["nlags"],
        "metrics": {
            "final_r2": summary["final_r2"],
            "final_rmse": summary["final_rmse"],
            "strict_final_r2": summary["strict_final_r2"],
            "strict_final_rmse": summary["strict_final_rmse"],
        },
        "outputs": {
            "report_txt": str(root_dir / "01_精度报告" / "RFK精度报告.txt"),
            "report_json": str(root_dir / "01_精度报告" / "RFK精度报告.json"),
            "pred_tif": str(pred_tif),
            "strict_oof_csv": str(root_dir / "03_GCP输入" / "strict_groupkfold_oof.csv"),
            "rf_model": str(root_dir / "_internal" / "rf_model.joblib"),
            "training_residuals": str(root_dir / "_internal" / "rfk_training_residuals.csv"),
        },
    }


# =============================================================================
# 主流程
# =============================================================================

def main() -> None:
    ensure_dir(CFG.out_root)
    ensure_dir(STATE_DIR)
    ensure_dir(TRIAL_DIR)
    reset_dir(LATEST_DIR)
    for sub in ["01_精度报告", "02_预测图", "03_GCP输入", "_internal", "_state"]:
        ensure_dir(LATEST_DIR / sub)
    ensure_dir(CURRENT_DIR)

    log("开始 RFK 参数搜索与正式建模 ...")
    df_full, X_full, _ = load_training_table()
    ranked_features = get_feature_ranking(df_full, X_full)
    search_candidates = build_search_space(ranked_features)

    previous_state = load_previous_best_state()
    previous_metrics = previous_state.get("metrics") if previous_state else None
    if previous_metrics:
        log(f"检测到历史最优 strict_final_r2 = {previous_metrics['strict_final_r2']:.6f}")

    best_trial_metrics: Optional[Dict[str, object]] = None
    best_trial_oof: Optional[pd.DataFrame] = None
    all_trials: List[Dict[str, object]] = []

    for trial_id, params in enumerate(search_candidates, start=1):
        log(f"[RFK] trial {trial_id}/{len(search_candidates)}: {params}")
        metrics, strict_oof = evaluate_candidate(
            trial_id=trial_id,
            params=params,
            df_full=df_full,
            X_full=X_full,
            ranked_features=ranked_features,
        )
        all_trials.append(metrics)
        if is_better(metrics, best_trial_metrics):
            best_trial_metrics = metrics
            best_trial_oof = strict_oof
            log(
                f"当前最佳 trial={trial_id} | strict_final_r2={metrics['strict_final_r2']:.6f} | "
                f"final_r2={metrics['final_r2']:.6f}"
            )
        if metrics["reached_target"]:
            log(f"已达到目标阈值 {CFG.search.target_r2:.2f}，提前结束搜索。")
            break

    if best_trial_metrics is None or best_trial_oof is None:
        raise RuntimeError("RFK 搜索失败，没有获得有效候选结果。")

    write_trial_table(all_trials, TRIAL_DIR / "rfk_search_trials.csv")

    # 生成 latest_run 正式产物
    latest_report_dir = LATEST_DIR / "01_精度报告"
    latest_map_dir = LATEST_DIR / "02_预测图"
    latest_gcp_input_dir = LATEST_DIR / "03_GCP输入"
    latest_model_dir = LATEST_DIR / "_internal"
    latest_state_dir = LATEST_DIR / "_state"

    best_features = best_trial_metrics["features"]
    best_rf_params = best_trial_metrics["rf_params"]
    best_variogram_model = best_trial_metrics["variogram_model"]
    best_nlags = int(best_trial_metrics["nlags"])

    bundle = fit_full_model(
        df_full=df_full,
        X_full=X_full,
        features=best_features,
        rf_params=best_rf_params,
        variogram_model=best_variogram_model,
        nlags=best_nlags,
        model_dir=latest_model_dir,
    )
    pred_tif = output_prediction_map(bundle, latest_map_dir, latest_model_dir)
    best_trial_oof.to_csv(latest_gcp_input_dir / "strict_groupkfold_oof.csv", index=False, encoding="utf-8-sig")
    write_concise_report(best_trial_metrics, latest_report_dir)
    log("RFK精度报告输出完成。")
    latest_manifest = build_manifest(best_trial_metrics, pred_tif, LATEST_DIR)
    write_json(latest_state_dir / "rfk_manifest.json", latest_manifest)

    should_promote, state_payload = promote_or_keep(latest_manifest, best_trial_metrics, previous_state)

    if should_promote:
        if CURRENT_DIR.exists():
            shutil.rmtree(CURRENT_DIR)
        shutil.copytree(LATEST_DIR, CURRENT_DIR)
        state_payload["manifest_path"] = str(CURRENT_DIR / "_state" / "rfk_manifest.json")
        log("本次 RFK 结果优于历史最优，已覆盖 current_best。")
    else:
        log("本次 RFK 结果未优于历史最优，仅覆盖 latest_run，保留 current_best。")

    write_json(STATE_DIR / "rfk_best_state.json", state_payload)
    log(f"RFK 完成。latest_run 清单：{LATEST_DIR / '_state' / 'rfk_manifest.json'}")
    log(f"RFK 当前正式清单：{CURRENT_DIR / '_state' / 'rfk_manifest.json'}")


if __name__ == "__main__":
    main()
