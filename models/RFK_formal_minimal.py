#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
成都市土壤有机质 RFK 正式运行版
================================
用途：
1. 采用固定 best11 特征、固定 RF 参数、固定 Kriging 参数进行正式 RFK 建模；
2. 输出简洁精度报告；
3. 输出全区 RFK 最终预测图（GeoTIFF）。

输出目录：
E:/Agent_DSM/RFK_OUT

最终成果：
- 01_精度报告
- 02_预测图
- 03_GCP输入
- _state manifest
"""

from __future__ import annotations

import json
import os
from datetime import datetime
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import rasterio
from pykrige.ok import OrdinaryKriging
from rasterio.warp import reproject, Resampling
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 固定配置
# =============================================================================

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
    repeated_splits: int = 15
    strict_splits: int = 5
    test_size: float = 0.30
    random_seed: int = 42

    fixed_features: Tuple[str, ...] = (
        "BD",
        "CEC",
        "Chengdu_ChinaCP_Multiyear_2015_2021",
        "Chengdu_SeasonalClimateWater_Vars_2019_2021_250m",
        "gravel",
        "LULCcd",
        "pH",
        "porosity",
        "sand",
        "sin_aspect",
        "tpi_3x3",
    )

    rf_params: Dict[str, object] = None
    variogram_model: str = "spherical"
    nlags: int = 8

    # 栅格目录：优先用已经验证过的 111/03_用到的栅格
    raster_root: Path = Path(r"E:/Agent_DSM/your_covariate_rasters")
    reference_feature: str = "tpi_3x3"

    # 已知显式路径
    forced_paths: Dict[str, str] = None


CFG = Config(
    rf_params={
        "n_estimators": 1200,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "max_features": 1.0,
        "max_depth": 16,
        "n_jobs": -1,
        "random_state": 42,
    },
    forced_paths={
        "pH": r"E:/Agent_DSM/your_covariate_rasters/pH.tif",
        "sin_aspect": r"E:/Agent_DSM/your_covariate_rasters/sin_aspect_32648_1km.tif",
        "tpi_3x3": r"E:/Agent_DSM/your_covariate_rasters/tpi_3x3_32648_1km.tif",
    },
)

REPORT_DIR = CFG.out_root / "01_精度报告"
MAP_DIR = CFG.out_root / "02_预测图"
GCP_INPUT_DIR = CFG.out_root / "03_GCP输入"
MODEL_DIR = CFG.out_root / "_internal"
STATE_DIR = CFG.out_root / "_state"

RF_MODEL_FILE = MODEL_DIR / "rf_model.joblib"
TRAINING_RESIDUALS_FILE = MODEL_DIR / "rfk_training_residuals.csv"
STRICT_OOF_FILE = GCP_INPUT_DIR / "strict_groupkfold_oof.csv"
MANIFEST_FILE = STATE_DIR / "rfk_manifest.json"


# =============================================================================
# 工具函数
# =============================================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: Path) -> None:
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


def create_fine_spatial_groups(df: pd.DataFrame) -> np.ndarray:
    x = df[CFG.x_col].values.astype(float)
    y = df[CFG.y_col].values.astype(float)

    x_edges = np.linspace(np.nanmin(x), np.nanmax(x), CFG.fine_grid_nx + 1)
    y_edges = np.linspace(np.nanmin(y), np.nanmax(y), CFG.fine_grid_ny + 1)

    x_bin = np.digitize(x, x_edges[1:-1], right=False)
    y_bin = np.digitize(y, y_edges[1:-1], right=False)
    groups = y_bin * CFG.fine_grid_nx + x_bin
    return groups.astype(int)


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
    fill_map = {}
    for c in code_cols:
        mode = X_train[c].mode(dropna=True)
        fill_map[c] = mode.iloc[0] if len(mode) else 0
    for c in cont_cols:
        fill_map[c] = float(X_train[c].median(skipna=True))
    return {
        "code_cols": code_cols,
        "cont_cols": cont_cols,
        "fill_map": fill_map,
    }


def apply_preprocessor(X: pd.DataFrame, state: Dict[str, object]) -> pd.DataFrame:
    X2 = X.copy()
    for c, v in state["fill_map"].items():
        if c in X2.columns:
            X2[c] = X2[c].fillna(v)
    return X2


def fit_rfk(train_df: pd.DataFrame, X_train: pd.DataFrame) -> Tuple[RandomForestRegressor, Dict[str, object], OrdinaryKriging]:
    y_train = train_df[CFG.target_col].values.astype(float)
    code_cols, cont_cols = split_feature_types(X_train)
    prep_state = make_preprocessor(X_train, code_cols, cont_cols)
    X_train_p = apply_preprocessor(X_train, prep_state)

    rf = RandomForestRegressor(**CFG.rf_params)
    rf.fit(X_train_p, y_train)
    rf_pred = rf.predict(X_train_p)
    resid = y_train - rf_pred

    ok = OrdinaryKriging(
        x=train_df[CFG.x_col].values.astype(float),
        y=train_df[CFG.y_col].values.astype(float),
        z=resid.astype(float),
        variogram_model=CFG.variogram_model,
        nlags=CFG.nlags,
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
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    train_df = df_full.iloc[train_idx].copy().reset_index(drop=False)
    val_df = df_full.iloc[val_idx].copy().reset_index(drop=False)
    X_train = X_full.iloc[train_idx].copy().reset_index(drop=True)
    X_val = X_full.iloc[val_idx].copy().reset_index(drop=True)

    rf, prep_state, ok = fit_rfk(train_df, X_train)
    rf_pred, krig_pred, final_pred = predict_rfk(rf, prep_state, ok, val_df, X_val)

    out = val_df.copy()
    out["pred_final"] = final_pred
    out["pred_rf"] = rf_pred
    out["pred_krig"] = krig_pred
    metrics = score_dict(val_df[CFG.target_col].values.astype(float), final_pred)
    return out, metrics


def run_repeated_spatial_holdout(df_full: pd.DataFrame, X_full: pd.DataFrame) -> Dict[str, float]:
    fine_groups = create_fine_spatial_groups(df_full)
    splitter = GroupShuffleSplit(
        n_splits=CFG.repeated_splits,
        test_size=CFG.test_size,
        random_state=CFG.random_seed,
    )

    oof_sum = np.full(len(df_full), np.nan, dtype=float)
    oof_count = np.zeros(len(df_full), dtype=int)
    fold_r2 = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        splitter.split(X_full, df_full[CFG.target_col], groups=fine_groups), start=1
    ):
        pred_df, metrics = run_one_split(df_full, X_full, train_idx, val_idx)
        preds = pred_df["pred_final"].values.astype(float)

        if np.isnan(oof_sum[val_idx]).all():
            oof_sum[val_idx] = preds
        else:
            current = oof_sum[val_idx]
            current = np.where(np.isnan(current), preds, current + preds)
            oof_sum[val_idx] = current

        oof_count[val_idx] += 1
        fold_r2.append(metrics["r2"])
        log(f"重复空间 7:3 | fold={fold_idx}/{CFG.repeated_splits} | r2={metrics['r2']:.6f}")

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


def run_strict_groupkfold(df_full: pd.DataFrame, X_full: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    groups = df_full[CFG.strict_group_col].values
    splitter = GroupKFold(n_splits=CFG.strict_splits)
    oof_pred = np.full(len(df_full), np.nan, dtype=float)
    oof_fold = np.full(len(df_full), -1, dtype=int)
    fold_r2 = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        splitter.split(X_full, df_full[CFG.target_col], groups=groups), start=1
    ):
        pred_df, metrics = run_one_split(df_full, X_full, train_idx, val_idx)
        oof_pred[val_idx] = pred_df["pred_final"].values.astype(float)
        oof_fold[val_idx] = fold_idx
        fold_r2.append(metrics["r2"])
        log(f"strict GroupKFold | fold={fold_idx}/{CFG.strict_splits} | r2={metrics['r2']:.6f}")

    valid = np.isfinite(oof_pred)
    met = score_dict(df_full.loc[valid, CFG.target_col].values.astype(float), oof_pred[valid])

    oof_df = df_full.loc[valid, [CFG.x_col, CFG.y_col, CFG.target_col]].copy()
    oof_df["strict_pred_final"] = oof_pred[valid]
    oof_df["strict_fold"] = oof_fold[valid]
    oof_df.to_csv(STRICT_OOF_FILE, index=False, encoding="utf-8-sig")
    log(f"写出 strict_groupkfold_oof：{STRICT_OOF_FILE}")

    return {
        "strict_final_r2": met["r2"],
        "strict_final_rmse": met["rmse"],
        "strict_final_mae": met["mae"],
        "strict_final_medae": met["medae"],
        "strict_final_bias": met["bias"],
        "strict_coverage_points": int(valid.sum()),
        "strict_mean_fold_r2": float(np.mean(fold_r2)),
        "strict_std_fold_r2": float(np.std(fold_r2)),
    }, oof_df


def run_single_holdout(df_full: pd.DataFrame, X_full: pd.DataFrame) -> Dict[str, float]:
    fine_groups = create_fine_spatial_groups(df_full)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=CFG.test_size,
        random_state=CFG.random_seed,
    )
    train_idx, val_idx = next(splitter.split(X_full, df_full[CFG.target_col], groups=fine_groups))
    pred_df, _ = run_one_split(df_full, X_full, train_idx, val_idx)
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


def fit_full_model(df_full: pd.DataFrame, X_full: pd.DataFrame):
    y = df_full[CFG.target_col].values.astype(float)
    code_cols, cont_cols = split_feature_types(X_full)
    prep_state = make_preprocessor(X_full, code_cols, cont_cols)
    X_p = apply_preprocessor(X_full, prep_state)

    rf = RandomForestRegressor(**CFG.rf_params)
    rf.fit(X_p, y)
    rf_pred = rf.predict(X_p)
    resid = y - rf_pred

    train_resid_df = df_full[[CFG.x_col, CFG.y_col]].copy()
    train_resid_df["residual"] = resid
    train_resid_df.to_csv(TRAINING_RESIDUALS_FILE, index=False, encoding="utf-8-sig")

    bundle = {
        "rf_model": rf,
        "preprocessor": prep_state,
        "features": list(X_full.columns),
        "rf_params": CFG.rf_params,
        "variogram_model": CFG.variogram_model,
        "nlags": CFG.nlags,
    }
    joblib.dump(bundle, RF_MODEL_FILE)
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

    # 宽松匹配
    matches = [p for p in CFG.raster_root.glob("*.tif") if p.stem.lower() == feature.lower()]
    if matches:
        return matches[0]

    raise FileNotFoundError(f"未找到特征栅格：{feature}")


def choose_resampling(feature: str) -> Resampling:
    if feature in {"LULCcd"}:
        return Resampling.nearest
    return Resampling.bilinear


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


def build_feature_stack() -> Tuple[Dict[str, np.ndarray], dict]:
    feature_paths = {f: resolve_feature_path(f) for f in CFG.fixed_features}
    ref_path = feature_paths[CFG.reference_feature]
    ref_arr, ref_profile = read_and_align_raster(ref_path, None, choose_resampling(CFG.reference_feature))

    feature_arrays = {CFG.reference_feature: ref_arr}
    for feat, path in feature_paths.items():
        if feat == CFG.reference_feature:
            continue
        arr, _ = read_and_align_raster(path, ref_profile, choose_resampling(feat))
        feature_arrays[feat] = arr
    return feature_arrays, ref_profile


def build_valid_mask(feature_arrays: Dict[str, np.ndarray]) -> np.ndarray:
    mask = np.ones_like(next(iter(feature_arrays.values())), dtype=bool)
    for feat in CFG.fixed_features:
        mask &= np.isfinite(feature_arrays[feat])
    return mask


def apply_preprocessor_to_array_table(X: pd.DataFrame, preprocessor: Dict[str, object]) -> pd.DataFrame:
    X2 = X.copy()
    for c, v in preprocessor["fill_map"].items():
        if c in X2.columns:
            X2[c] = X2[c].fillna(v)
    return X2



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

def output_prediction_map(bundle: Dict[str, object]) -> Path:
    log("开始输出全区 RFK 预测图 ...")
    feature_arrays, ref_profile = build_feature_stack()
    valid_mask = build_valid_mask(feature_arrays)
    rows, cols = np.where(valid_mask)
    n_valid = int(valid_mask.sum())
    log(f"有效像元数 = {n_valid}")

    X_grid = pd.DataFrame({f: feature_arrays[f][rows, cols] for f in CFG.fixed_features})
    X_grid_p = apply_preprocessor_to_array_table(X_grid, bundle["preprocessor"])
    rf_model = bundle["rf_model"]

    rf_pred = rf_model.predict(X_grid_p).astype(float)

    resid_df = pd.read_csv(TRAINING_RESIDUALS_FILE)
    ok = OrdinaryKriging(
        x=resid_df[CFG.x_col].values.astype(float),
        y=resid_df[CFG.y_col].values.astype(float),
        z=resid_df["residual"].values.astype(float),
        variogram_model=CFG.variogram_model,
        nlags=CFG.nlags,
        verbose=False,
        enable_plotting=False,
    )

    xs, ys = rasterio.transform.xy(ref_profile["transform"], rows, cols, offset="center")
    krig_pred, _ = ok.execute("points", np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))
    krig_pred = np.asarray(krig_pred, dtype=float)

    final_pred = rf_pred + krig_pred

    pred_arr = np.full(valid_mask.shape, np.nan, dtype="float32")
    pred_arr[rows, cols] = final_pred.astype("float32")

    pred_tif = build_timestamp_path(MAP_DIR / "RFK最终预测图.tif")
    pred_tif = array_to_tif(pred_arr, pred_tif, ref_profile)
    log(f"预测图输出完成：{pred_tif}")
    return pred_tif


# =============================================================================
# 数据准备
# =============================================================================

def load_training_table() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(CFG.csv_path)
    df.columns = [str(c).strip() for c in df.columns]

    drop_cols = set(CFG.id_cols) | set(CFG.leak_cols) | {CFG.target_col, CFG.strict_group_col}
    feature_cols = [c for c in CFG.fixed_features if c in df.columns]
    if len(feature_cols) != len(CFG.fixed_features):
        miss = [c for c in CFG.fixed_features if c not in df.columns]
        raise ValueError(f"样点表缺少固定特征列：{miss}")

    work_df = df[[*df.columns]].copy()
    X = work_df[feature_cols].copy()
    return work_df, X


# =============================================================================
# 报告输出
# =============================================================================

def write_concise_report(summary: Dict[str, float]) -> None:
    json_path = REPORT_DIR / "RFK精度报告.json"
    txt_path = REPORT_DIR / "RFK精度报告.txt"

    simple = {
        "model": "RFK",
        "scope": "citywide",
        "features": list(CFG.fixed_features),
        "rf_params": CFG.rf_params,
        "variogram_model": CFG.variogram_model,
        "nlags": CFG.nlags,
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

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(simple, f, ensure_ascii=False, indent=2)

    txt = f"""成都市土壤有机质 RFK 精度报告
==============================
模型：RFK
范围：全区单模型

固定特征（11个）：
{", ".join(CFG.fixed_features)}

固定参数：
RF = {json.dumps(CFG.rf_params, ensure_ascii=False)}
Kriging = variogram_model={CFG.variogram_model}, nlags={CFG.nlags}

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
- 正式结论建议采用“重复空间7:3”。
- strict GroupKFold 用于更严格的空间泛化诊断。
"""
    txt_path.write_text(txt, encoding="utf-8")
    log(f"精度报告输出完成：{txt_path}")




def write_manifest(summary: Dict[str, float], pred_tif: Path) -> None:
    manifest = {
        "model": "RFK",
        "mode": "formal_fixed",
        "inputs": {
            "csv_path": str(CFG.csv_path),
            "raster_root": str(CFG.raster_root),
        },
        "params": {
            "features": list(CFG.fixed_features),
            "rf_params": CFG.rf_params,
            "variogram_model": CFG.variogram_model,
            "nlags": CFG.nlags,
            "random_seed": CFG.random_seed,
            "strict_splits": CFG.strict_splits,
            "repeated_splits": CFG.repeated_splits,
        },
        "metrics": summary,
        "outputs": {
            "report_json": str(REPORT_DIR / "RFK精度报告.json"),
            "report_txt": str(REPORT_DIR / "RFK精度报告.txt"),
            "pred_tif": str(pred_tif),
            "strict_oof_csv": str(STRICT_OOF_FILE),
            "training_residuals_csv": str(TRAINING_RESIDUALS_FILE),
            "rf_model_joblib": str(RF_MODEL_FILE),
        },
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

# =============================================================================
# 主程序
# =============================================================================

def main() -> None:
    ensure_dir(CFG.out_root)
    ensure_dir(REPORT_DIR)
    ensure_dir(MAP_DIR)
    ensure_dir(GCP_INPUT_DIR)
    ensure_dir(MODEL_DIR)
    ensure_dir(STATE_DIR)

    log("开始 RFK 正式精简建模 ...")
    log(f"样点 CSV = {CFG.csv_path}")
    log(f"输出目录 = {CFG.out_root}")

    df_full, X_full = load_training_table()
    log(f"固定特征数 = {len(CFG.fixed_features)}")
    log(f"固定变差函数 = {CFG.variogram_model}, nlags = {CFG.nlags}")
    log(f"固定 RF 参数 = {CFG.rf_params}")
    log(f"样本数 = {len(df_full)}, 可用固定特征数 = {X_full.shape[1]}")

    repeated_summary = run_repeated_spatial_holdout(df_full, X_full)
    log(f"重复空间 7:3 完成：final_r2 = {repeated_summary['final_r2']:.6f}")

    strict_summary, _ = run_strict_groupkfold(df_full, X_full)
    log(f"strict GroupKFold 完成：strict_final_r2 = {strict_summary['strict_final_r2']:.6f}")

    single_summary = run_single_holdout(df_full, X_full)
    log(f"单次空间 7:3 留档完成：val_r2 = {single_summary['single_holdout_r2']:.6f}")

    summary = {}
    summary.update(repeated_summary)
    summary.update(strict_summary)
    summary.update(single_summary)

    write_concise_report(summary)

    bundle = fit_full_model(df_full, X_full)
    pred_tif = output_prediction_map(bundle)
    write_manifest(summary, pred_tif)

    log("全部完成。")
    log(f"- 精度报告：{REPORT_DIR / 'RFK精度报告.txt'}")
    log(f"- 预测图：{pred_tif}")
    log(f"- GCP输入：{STRICT_OOF_FILE}")


if __name__ == "__main__":
    main()
