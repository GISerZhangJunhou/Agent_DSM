#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V165 RFK fused-CSV dynamic runner
=========================

Use when the user already has a fused training CSV:
    sample coordinates + SOM/有机质 + environmental covariate columns

Example:
    python models/RFK_V165_from_fused_csv.py ^
      --csv "E:/Agent_DSM/your_training.csv" ^
      --out "E:/Agent_DSM/RFK_V165_OUT" ^
      --raster-root "E:/Agent_DSM/your_covariate_rasters" ^
      --target-col som_value --x-col x_32648 --y-col y_32648

If --raster-root is omitted, the script still trains/evaluates RFK and writes
reports, but cannot create a full-coverage GeoTIFF prediction map.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services import rfk_clean_v165  # noqa: E402

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
except Exception:  # pragma: no cover
    rasterio = None
    reproject = None
    Resampling = None


class FusedCsvOwner:
    def __init__(self, csv_path: Path, out_dir: Path, raster_root: Path | None, target_col: str | None, x_col: str | None, y_col: str | None, lon_col: str | None, lat_col: str | None):
        self._model_ready_csv = str(csv_path)
        self.task_id = None
        self.run_mode = "v165_fused_csv_standalone"
        self.warnings: list[str] = []
        self.out_dir = Path(out_dir)
        self.raster_root = Path(raster_root) if raster_root else None
        self.target_col = target_col
        self.x_col = x_col
        self.y_col = y_col
        self.lon_col = lon_col
        self.lat_col = lat_col
        self._prediction_grid_csv = None

    def _coords_xy(self, df: pd.DataFrame, target: dict[str, Any] | None):
        if self.x_col and self.y_col and self.x_col in df.columns and self.y_col in df.columns:
            return pd.to_numeric(df[self.x_col], errors="coerce").to_numpy(float), pd.to_numeric(df[self.y_col], errors="coerce").to_numpy(float), "existing_projected_xy"
        if self.lon_col and self.lat_col and self.lon_col in df.columns and self.lat_col in df.columns:
            from pyproj import Transformer
            dst = str((target or {}).get("target_crs") or os.getenv("PRO_TARGET_CRS_CHINA") or "EPSG:32648")
            tr = Transformer.from_crs("EPSG:4326", dst, always_xy=True)
            x, y = tr.transform(pd.to_numeric(df[self.lon_col], errors="coerce"), pd.to_numeric(df[self.lat_col], errors="coerce"))
            return np.asarray(x, float), np.asarray(y, float), dst
        # Let core auto-detect as fallback.
        raise RuntimeError("owner has no explicit coordinate columns")

    def _aoi_sample_weight(self, train_df: pd.DataFrame):
        return None

    def _krige_residuals(self, state: dict[str, Any], df: pd.DataFrame, target: dict[str, Any] | None):
        x, y, _ = rfk_clean_v165._make_xy(self, df, target)
        nc = int(state.get("n_closest_points") or 0)
        if nc > 0:
            z, _ = state["ok"].execute("points", x.astype(float), y.astype(float), n_closest_points=nc)
        else:
            z, _ = state["ok"].execute("points", x.astype(float), y.astype(float))
        return np.asarray(z, dtype="float64")

    def _resolve_raster(self, feature: str) -> Path | None:
        if not self.raster_root or not self.raster_root.exists():
            return None
        name = str(feature).replace("cov_domestic_local_", "").replace("cov_", "")
        candidates = [
            self.raster_root / f"{name}.tif",
            self.raster_root / f"{feature}.tif",
            self.raster_root / f"{name}_32648_1km.tif",
            self.raster_root / f"{name}_32648_250m.tif",
        ]
        for p in candidates:
            if p.exists():
                return p
        low = name.lower()
        matches = [p for p in self.raster_root.glob("*.tif") if p.stem.lower() == low or low in p.stem.lower() or p.stem.lower() in low]
        return matches[0] if matches else None

    def _choose_resampling(self, feature: str):
        if Resampling is None:
            return None
        fl = str(feature).lower()
        if any(k in fl for k in ["clcd", "lulc", "landcover", "land_use", "classification", "class"]):
            return Resampling.nearest
        return Resampling.bilinear

    def _read_aligned(self, path: Path, dst_profile: dict | None, resampling):
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

    def _make_prediction_tif(self, final_state: dict[str, Any], kept_features: list[str], samples_df: pd.DataFrame, out_dir: Path, target: dict[str, Any] | None):
        if rasterio is None or self.raster_root is None or not self.raster_root.exists():
            self.warnings.append("未提供 raster_root 或 rasterio 不可用：仅输出精度报告，不输出预测图。")
            return None
        # Ignore generated missing indicators for raster prediction; they are 0 for grid cells.
        base_features = [f for f in kept_features if not str(f).endswith("__missing")]
        paths = {f: self._resolve_raster(f) for f in base_features}
        paths = {f: p for f, p in paths.items() if p is not None and Path(p).exists()}
        if not paths:
            self.warnings.append("未在 raster_root 中找到可匹配的特征栅格：仅输出精度报告。")
            return None
        # Reference = first matched raster.
        first_feature = next(iter(paths))
        ref_arr, ref_profile = self._read_aligned(paths[first_feature], None, self._choose_resampling(first_feature))
        arrays = {first_feature: ref_arr}
        for f, p in paths.items():
            if f == first_feature:
                continue
            arrays[f], _ = self._read_aligned(p, ref_profile, self._choose_resampling(f))
        mask = np.ones_like(ref_arr, dtype=bool)
        for f in paths:
            mask &= np.isfinite(arrays[f])
        rows, cols = np.where(mask)
        if len(rows) == 0:
            self.warnings.append("预测栅格没有共同有效像元：仅输出精度报告。")
            return None
        X_grid = pd.DataFrame({f: arrays[f][rows, cols] for f in paths})
        # Reconstruct full model feature matrix order. Missing feature columns are filled with training medians/0.
        full = pd.DataFrame(index=X_grid.index)
        audit = (target or {}).get("training_matrix_audit") or {}
        impute = audit.get("impute_values") or {}
        for f in kept_features:
            if str(f).endswith("__missing"):
                full[f] = 0.0
            elif f in X_grid.columns:
                full[f] = pd.to_numeric(X_grid[f], errors="coerce").fillna(float(impute.get(f, 0.0)))
            else:
                full[f] = float(impute.get(f, 0.0))
        rf_pred = final_state["rf"].predict(full.to_numpy(dtype="float64"))
        xs, ys = rasterio.transform.xy(ref_profile["transform"], rows, cols, offset="center")
        pts_df = pd.DataFrame({self.x_col or "x": np.asarray(xs, float), self.y_col or "y": np.asarray(ys, float)})
        # Force coordinate columns used by _coords_xy for standalone prediction.
        oldx, oldy = self.x_col, self.y_col
        self.x_col, self.y_col = pts_df.columns[0], pts_df.columns[1]
        try:
            resid = self._krige_residuals(final_state, pts_df, target)
        finally:
            self.x_col, self.y_col = oldx, oldy
        pred = rf_pred + resid
        out_arr = np.full(mask.shape, np.nan, dtype="float32")
        out_arr[rows, cols] = pred.astype("float32")
        out_tif = Path(out_dir) / "V165_RFK_有机质图.tif"
        profile = ref_profile.copy(); profile.update(driver="GTiff", dtype="float32", count=1, compress="lzw", nodata=np.nan)
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(out_arr, 1)
        grid_csv = Path(out_dir) / "V165_预测网格抽样表.csv"
        X_grid.assign(row=rows, col=cols, pred=pred).to_csv(grid_csv, index=False, encoding="utf-8-sig")
        self._prediction_grid_csv = str(grid_csv)
        target["prediction_grid_audit"] = {"raster_root": str(self.raster_root), "matched_features": list(paths.keys()), "valid_count": int(mask.sum()), "path": str(out_tif)}
        return out_tif


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="融合后的训练CSV：含SOM/有机质、坐标、环境协变量")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--raster-root", default="", help="可选：用于生成预测图的协变量栅格目录")
    ap.add_argument("--target-col", default="", help="目标列名，如 som_value/有机质")
    ap.add_argument("--x-col", default="", help="投影X列，如 x_32648")
    ap.add_argument("--y-col", default="", help="投影Y列，如 y_32648")
    ap.add_argument("--lon-col", default="", help="经度列")
    ap.add_argument("--lat-col", default="", help="纬度列")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    work_csv = out_dir / "建模样点表.csv"
    # Keep the user CSV untouched, copy as the model-ready entrance.
    pd.read_csv(csv_path).to_csv(work_csv, index=False, encoding="utf-8-sig")
    owner = FusedCsvOwner(work_csv, out_dir, Path(args.raster_root) if args.raster_root else None, args.target_col or None, args.x_col or None, args.y_col or None, args.lon_col or None, args.lat_col or None)
    # Feature detection is handled by rfk_clean_v165.
    target = {"region": "fused_csv", "target_crs": "EPSG:32648", "fused_csv_source": str(csv_path)}
    report, pred = rfk_clean_v165.run_clean_rfk_model(owner, pd.DataFrame(), [], out_dir, target, [])
    print("V165 RFK 完成")
    print("report_json=", report)
    print("pred_tif=", pred)


if __name__ == "__main__":
    main()
