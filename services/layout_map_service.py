from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None

from config.settings import DATA_DIR

try:
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.features import geometry_mask
except Exception:  # pragma: no cover
    rasterio = None
    transform_bounds = None
    geometry_mask = None


DEFAULT_CENTER = {"lat": 30.67, "lng": 104.06, "zoom": 9}
DEFAULT_BOUNDS = [[29.95, 103.15], [31.35, 104.95]]
TDT_TOKEN = "cb252a30348a792ba38092c14e30993d"

STYLED_DIR = DATA_DIR / "styled_cache"
STYLED_DIR.mkdir(parents=True, exist_ok=True)

# V79: low-resolution GEE grids are kept georeferenced but rendered as larger PNGs
# so the web map does not look like a tiny 70x51 thumbnail when opened directly.
OVERLAY_RENDER_MIN_PX = int(os.getenv("WEBGIS_OVERLAY_RENDER_MIN_PX", "1024"))
OVERLAY_RESAMPLE_METHOD = os.getenv("WEBGIS_OVERLAY_RESAMPLE_METHOD", "nearest").strip().lower()
WEBGIS_LEGEND_RANGE_MODE = os.getenv("WEBGIS_LEGEND_RANGE_MODE", "robust").strip().lower()
RESULT_OVERLAY_OPACITY = float(os.getenv("WEBGIS_RESULT_OVERLAY_OPACITY", "0.92"))
SAMPLE_POINT_MAX_COUNT = int(os.getenv("WEBGIS_SAMPLE_POINT_MAX_COUNT", "800"))
BASEMAP_AUTO_HIDE_AFTER_RESULT = os.getenv("WEBGIS_BASEMAP_AUTO_HIDE_AFTER_RESULT", "1").strip() != "0"
WEBGIS_RESULT_GAMMA = float(os.getenv("WEBGIS_RESULT_GAMMA", "0.85"))

# Maximum number of uploaded/derived GeoTIFF layers exposed to the workspace selector.
# V236 keeps all typical covariate layers available so switching does not wait
# for the server to rebuild a new layer after the user changes the dropdown.
UPLOAD_PREVIEW_LAYER_MAX_COUNT = int(os.getenv("WEBGIS_UPLOAD_PREVIEW_LAYER_MAX_COUNT", "200"))

# ArcGIS Pro 风格近似色带（网页可直接消费）
COLOR_RAMPS: dict[str, list[str]] = {
    "ArcGIS Pro｜Yellow Green Blue": ["#ffffd9", "#edf8b1", "#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#225ea8", "#0c2c84"],
    "ArcGIS Pro｜Yellow Orange Red": ["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026"],
    "ArcGIS Pro｜Green Continuous": ["#f7fcf5", "#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476", "#41ab5d", "#238b45", "#005a32"],
    "ArcGIS Pro｜Blue Continuous": ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"],
    "ArcGIS Pro｜Terrain": ["#543005", "#8c510a", "#bf812d", "#dfc27d", "#f6e8c3", "#c7eae5", "#80cdc1", "#35978f", "#01665e"],
    "ArcGIS Pro｜Viridis": ["#440154", "#472c7a", "#3b518b", "#2c718e", "#21908d", "#27ad81", "#5cc863", "#aadc32", "#fde725"],
    "ArcGIS Pro｜RdYlGn": ["#a50026", "#d73027", "#f46d43", "#fdae61", "#fee08b", "#d9ef8b", "#a6d96a", "#66bd63", "#1a9850", "#006837"],
    "ArcGIS Pro｜Spectral": ["#9e0142", "#d53e4f", "#f46d43", "#fdae61", "#fee08b", "#e6f598", "#abdda4", "#66c2a5", "#3288bd", "#5e4fa2"],
    "ArcGIS Pro｜Cyan Purple": ["#e0ecf4", "#bfd3e6", "#9ebcda", "#8c96c6", "#8c6bb1", "#88419d", "#6e016b"],
    "ArcGIS Pro｜Heat": ["#ffffb2", "#fed976", "#feb24c", "#fd8d3c", "#f03b20", "#bd0026"],
    "ArcGIS Pro｜Gray Continuous": ["#f7f7f7", "#d9d9d9", "#bdbdbd", "#969696", "#737373", "#525252", "#252525"],
    "ArcGIS Pro｜Brown Orange": ["#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#8c2d04"],
}

_SIMPLE_TO_RAMP = {
    "green": "ArcGIS Pro｜Green Continuous",
    "blue": "ArcGIS Pro｜Blue Continuous",
    "purple": "ArcGIS Pro｜Cyan Purple",
    "orange": "ArcGIS Pro｜Yellow Orange Red",
    "yellow": "ArcGIS Pro｜Brown Orange",
    "red": "ArcGIS Pro｜Heat",
    "heat": "ArcGIS Pro｜Heat",
    "terrain": "ArcGIS Pro｜Terrain",
    "viridis": "ArcGIS Pro｜Viridis",
    "spectral": "ArcGIS Pro｜Spectral",
    "rdylgn": "ArcGIS Pro｜RdYlGn",
    "cyan_purple": "ArcGIS Pro｜Cyan Purple",
    "gray": "ArcGIS Pro｜Gray Continuous",
    "grey": "ArcGIS Pro｜Gray Continuous",
}


def palette_options() -> list[dict[str, str]]:
    return [{"label": name, "value": name} for name in COLOR_RAMPS.keys()]


def normalize_palette_name(name: str | None) -> str:
    if not name:
        return "ArcGIS Pro｜Green Continuous"
    return _SIMPLE_TO_RAMP.get(name, name if name in COLOR_RAMPS else "ArcGIS Pro｜Green Continuous")


UPLOAD_LAYER_PALETTE_CYCLE = [
    "ArcGIS Pro｜Green Continuous",
    "ArcGIS Pro｜Blue Continuous",
    "ArcGIS Pro｜Yellow Orange Red",
    "ArcGIS Pro｜Cyan Purple",
    "ArcGIS Pro｜Terrain",
    "ArcGIS Pro｜Viridis",
    "ArcGIS Pro｜Spectral",
    "ArcGIS Pro｜Heat",
]


def _upload_layer_palette(idx: int) -> str:
    try:
        return UPLOAD_LAYER_PALETTE_CYCLE[(max(1, int(idx)) - 1) % len(UPLOAD_LAYER_PALETTE_CYCLE)]
    except Exception:
        return "ArcGIS Pro｜Green Continuous"


def _preprocess_covariate_record_map(session) -> dict[str, dict]:
    out = {}
    report = getattr(session, "preprocess_report", {}) or {}
    for rec in report.get("covariate_records") or []:
        p = str(rec.get("input_path") or "")
        if p:
            out[p] = rec
    return out


def _display_tif_from_preprocess(original_path: str, record: dict | None) -> str:
    if record:
        for key in ("aligned_path", "standardized_path"):
            pth = str(record.get(key) or "")
            if pth and Path(pth).exists():
                return pth
    return original_path


def _hex_to_rgb(v: str) -> tuple[int, int, int]:
    v = v.lstrip("#")
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


def _build_lut(colors: list[str], size: int = 256) -> np.ndarray:
    rgb = np.array([_hex_to_rgb(c) for c in colors], dtype=np.float32)
    xs = np.linspace(0.0, 1.0, len(rgb))
    q = np.linspace(0.0, 1.0, size)
    lut = np.zeros((size, 3), dtype=np.uint8)
    for i in range(3):
        lut[:, i] = np.interp(q, xs, rgb[:, i]).astype(np.uint8)
    return lut


def _safe_transform_bounds(src) -> tuple[float, float, float, float]:
    if transform_bounds is None or src.crs is None:
        return (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
    try:
        return transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
    except Exception:
        return (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)


def _edge_mask(shape: tuple[int, int], edge_width: int | None = None) -> np.ndarray:
    """Boolean mask for raster edge cells used by preview background inference."""
    h, w = shape
    ew = edge_width or max(1, min(8, h // 30 if h >= 30 else 1, w // 30 if w >= 30 else 1))
    edge = np.zeros((h, w), dtype=bool)
    edge[:ew, :] = True
    edge[-ew:, :] = True
    edge[:, :ew] = True
    edge[:, -ew:] = True
    return edge


def _infer_edge_background_values(arr: np.ndarray, valid: np.ndarray) -> list[float]:
    """Infer constant rectangular background / mask values from raster edges.

    Many uploaded covariate rasters are clipped to an administrative boundary but
    still stored inside a rectangular GeoTIFF extent. If the producer did not set
    the GeoTIFF nodata tag, the outside-boundary fill value is rendered as a
    normal class/value. This helper finds exact or nearly exact values that
    dominate the raster edge. Continuous low/high edge backgrounds are handled
    later by `_infer_edge_background_components`.
    """
    try:
        if arr.ndim != 2 or arr.size == 0:
            return []
        edge_mask = _edge_mask(arr.shape)
        edge_vals = arr[edge_mask & valid & np.isfinite(arr)]
        if edge_vals.size < 16:
            return []
        rounded = np.round(edge_vals.astype("float64"), 6)
        uniq, counts = np.unique(rounded, return_counts=True)
        if uniq.size == 0:
            return []
        order = np.argsort(counts)[::-1][:6]
        out: list[float] = []
        total_edge = float(edge_vals.size)
        common_sentinels = {0.0, -9999.0, 9999.0, -32768.0, 32767.0, -2147483648.0, 2147483647.0}
        valid_total = float(max(1, np.sum(valid)))
        for i in order:
            cand = float(uniq[i])
            edge_ratio = float(counts[i]) / total_edge
            close = np.isclose(arr, cand, rtol=0, atol=max(1e-6, abs(cand) * 1e-6))
            global_ratio = float(np.sum(close & valid)) / valid_total
            sentinel_like = cand in common_sentinels or abs(cand) > 1e20
            if sentinel_like or edge_ratio >= 0.22 or (edge_ratio >= 0.12 and global_ratio >= 0.08):
                out.append(cand)
        return out
    except Exception:
        return []



def _infer_edge_background_components(arr: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Infer non-constant edge-connected mask components.

    Some clipped GeoTIFFs have an outside-boundary background that is not exactly
    one value after resampling/alignment, so exact nodata detection is not enough.
    We conservatively remove only large components connected to the raster edge
    and located in the extreme low or high tail of the value distribution. This
    fixes the visible rectangular mask while keeping interior cells with the same
    numeric range.
    """
    try:
        if arr.ndim != 2 or arr.size == 0 or not np.any(valid):
            return np.zeros(arr.shape, dtype=bool), []
        finite = valid & np.isfinite(arr)
        vals = arr[finite].astype("float64")
        if vals.size < 100:
            return np.zeros(arr.shape, dtype=bool), []
        edge = _edge_mask(arr.shape)
        edge_vals = arr[edge & finite].astype("float64")
        if edge_vals.size < 24:
            return np.zeros(arr.shape, dtype=bool), []
        out = np.zeros(arr.shape, dtype=bool)
        notes: list[str] = []
        total_valid = float(max(1, np.sum(finite)))
        total_edge = float(max(1, np.sum(edge & finite)))

        # Candidate tails. q02/q98 catch sentinel-like stretched backgrounds;
        # q08/q92 catch smoother resampled backgrounds. All candidates must form
        # a large edge-connected component, so legitimate interior extremes stay.
        pct_candidates = [
            ("low", 2.0), ("low", 5.0), ("low", 10.0),
            ("high", 90.0), ("high", 95.0), ("high", 98.0),
        ]
        for side, pct in pct_candidates:
            thr = float(np.nanpercentile(vals, pct))
            if side == "low":
                cand = finite & (arr <= thr)
            else:
                cand = finite & (arr >= thr)
            if not np.any(cand):
                continue
            comp = _edge_connected_region(cand)
            comp_count = int(np.sum(comp))
            if comp_count <= 0:
                continue
            comp_ratio = comp_count / total_valid
            edge_ratio = float(np.sum(comp & edge)) / total_edge
            if comp_ratio >= 0.10 and edge_ratio >= 0.20:
                out |= comp
                notes.append(f"edge_{side}_p{int(pct)}")
                # Once a strong low/high background is found, do not keep widening
                # the same tail too aggressively.
                if side == "low" and pct >= 5.0:
                    pass
        return out, notes
    except Exception:
        return np.zeros(arr.shape, dtype=bool), []



def _edge_connected_region(mask: np.ndarray) -> np.ndarray:
    """Return cells in *mask* connected to the raster edge using 4-neighbourhood."""
    try:
        from collections import deque
        if mask.ndim != 2 or not np.any(mask):
            return np.zeros(mask.shape, dtype=bool)
        h, w = mask.shape
        seen = np.zeros(mask.shape, dtype=bool)
        q = deque()

        def add(r: int, c: int) -> None:
            if r < 0 or c < 0 or r >= h or c >= w:
                return
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

        for c in range(w):
            add(0, c)
            add(h - 1, c)
        for r in range(h):
            add(r, 0)
            add(r, w - 1)
        while q:
            r, c = q.popleft()
            add(r - 1, c)
            add(r + 1, c)
            add(r, c - 1)
            add(r, c + 1)
        return seen
    except Exception:
        return np.zeros(mask.shape, dtype=bool)



def _split_env_paths(value: str) -> list[str]:
    import re
    return [x.strip().strip('"').strip("'") for x in re.split(r"[;,\n]+", value or "") if x.strip()]


def _strip_admin_suffix(name: str) -> str:
    s = str(name or "").strip()
    for suf in ["特别行政区", "壮族自治区", "维吾尔自治区", "回族自治区", "自治区", "自治州", "地区", "盟", "省", "市", "县", "区", "旗"]:
        if s.endswith(suf) and len(s) > len(suf):
            return s[: -len(suf)]
    return s


def _same_admin_name(a: str | None, b: str | None) -> bool:
    a = str(a or "").strip()
    b = str(b or "").strip()
    if not a or not b:
        return False
    aa = {a, _strip_admin_suffix(a)}
    bb = {b, _strip_admin_suffix(b)}
    aa = {x for x in aa if x}
    bb = {x for x in bb if x}
    return bool(aa & bb) or a in b or b in a


def _admin_vector_paths(level: str) -> list[Path]:
    keys = {
        "county": ["PRO_LOCAL_ADMIN_COUNTY_VECTOR", "PRO_ADMIN_COUNTY_VECTOR"],
        "city": ["PRO_LOCAL_ADMIN_CITY_VECTOR", "PRO_ADMIN_CITY_VECTOR"],
        "province": ["PRO_LOCAL_ADMIN_PROVINCE_VECTOR", "PRO_ADMIN_PROVINCE_VECTOR"],
    }.get(level, [])
    paths: list[Path] = []
    for key in keys:
        for s in _split_env_paths(os.getenv(key, "")):
            p = Path(s)
            if p.exists() and p.is_file():
                paths.append(p)
    root = os.getenv("PRO_LOCAL_ADMIN_VECTOR_DIR", "").strip() or os.getenv("PRO_ADMIN_VECTOR_DIR", "").strip()
    if root:
        d = Path(root)
        if d.exists() and d.is_dir():
            exact = {"county": "县.shp", "city": "市.shp", "province": "省.shp"}.get(level)
            if exact and (d / exact).exists():
                paths.append(d / exact)
            keywords = {"county": ["县", "区"], "city": ["市"], "province": ["省"]}.get(level, [])
            excludes = {"十段线", "九段线", "界线", "南海诸岛"}
            for shp in sorted(d.glob("*.shp")):
                if shp.stem in excludes:
                    continue
                if any(k in shp.stem for k in keywords):
                    paths.append(shp)
    out, seen = [], set()
    for p in paths:
        k = str(p.resolve()) if p.exists() else str(p)
        if k not in seen:
            seen.add(k); out.append(p)
    return out


def _name_fields(columns: list[str], level: str) -> list[str]:
    env_key = {
        "county": "PRO_LOCAL_ADMIN_COUNTY_NAME_FIELD",
        "city": "PRO_LOCAL_ADMIN_CITY_NAME_FIELD",
        "province": "PRO_LOCAL_ADMIN_PROVINCE_NAME_FIELD",
    }.get(level, "PRO_LOCAL_ADMIN_NAME_FIELD")
    env_fields = _split_env_paths(os.getenv(env_key, "")) + _split_env_paths(os.getenv("PRO_LOCAL_ADMIN_NAME_FIELD", ""))
    defaults = {
        "county": ["县名", "区名", "县", "区", "NAME", "name", "county", "district", "Name"],
        "city": ["市名", "市", "地市", "地级市", "NAME", "name", "city", "Name"],
        "province": ["省名", "省", "NAME", "name", "province", "Name"],
    }.get(level, ["NAME", "name", "Name"])
    seen, out = set(), []
    for f in env_fields + defaults:
        if f in columns and f not in seen:
            out.append(f); seen.add(f)
    return out


def _region_from_uploaded_samples(session) -> str | None:
    try:
        for item in list(getattr(session, "latest_uploaded_files", []) or []):
            val = item.get("validation") or {}
            reg = val.get("region_inference") or {}
            if reg.get("ok") and reg.get("region"):
                return str(reg.get("region"))
    except Exception:
        pass
    return None


def _admin_boundary_inside_mask(src, region_name: str | None, out_shape: tuple[int, int] | None = None, out_transform=None) -> tuple[np.ndarray | None, dict]:
    """Rasterize a matched local administrative boundary to the source grid.

    This is used only for preview transparency. If no local admin vector is
    configured, the function returns None and the normal nodata/background
    inference remains in effect.
    """
    meta = {"enabled": bool(region_name), "region": region_name, "applied": False}
    if not region_name or gpd is None or geometry_mask is None:
        meta["reason"] = "未配置区域名称、geopandas 或 rasterio.features，跳过行政边界透明处理。"
        return None, meta
    try:
        for level in ["county", "city", "province"]:
            for shp in _admin_vector_paths(level):
                try:
                    gdf = gpd.read_file(shp)
                    if gdf.empty:
                        continue
                    fields = _name_fields(list(gdf.columns), level)
                    if not fields:
                        continue
                    matched = None
                    for field in fields:
                        sel = gdf[gdf[field].astype(str).map(lambda x: _same_admin_name(x, region_name))]
                        if not sel.empty:
                            matched = sel
                            break
                    if matched is None or matched.empty:
                        continue
                    if matched.crs is None:
                        matched = matched.set_crs("EPSG:4326", allow_override=True)
                    if src.crs is not None:
                        matched = matched.to_crs(src.crs)
                    geoms = [geom for geom in matched.geometry if geom is not None and not geom.is_empty]
                    if not geoms:
                        continue
                    shape = out_shape or (src.height, src.width)
                    transform = out_transform if out_transform is not None else src.transform
                    inside = geometry_mask(geoms, out_shape=shape, transform=transform, invert=True, all_touched=True)
                    if inside is not None and np.any(inside):
                        meta.update({"applied": True, "level": level, "path": str(shp), "matched_count": int(len(matched))})
                        return np.asarray(inside, dtype=bool), meta
                except Exception as exc:
                    meta["last_error"] = str(exc)
                    continue
    except Exception as exc:
        meta["error"] = str(exc)
    meta.setdefault("reason", "未匹配到可用本地行政边界。")
    return None, meta



def _infer_edge_modal_background_components(arr: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Mask edge-connected modal/background values more aggressively.

    This targets uploaded covariates whose outside-AOI background is encoded as
    a normal value such as 0, while the GeoTIFF nodata tag is absent or different.
    Only the edge-connected component is removed, so interior cells with the same
    value are preserved.
    """
    try:
        if arr.ndim != 2 or arr.size == 0 or not np.any(valid):
            return np.zeros(arr.shape, dtype=bool), []
        finite = valid & np.isfinite(arr)
        edge = _edge_mask(arr.shape)
        edge_vals = arr[edge & finite].astype("float64")
        if edge_vals.size < 16:
            return np.zeros(arr.shape, dtype=bool), []
        vals = arr[finite].astype("float64")
        total_valid = float(max(1, np.sum(finite)))
        total_edge = float(max(1, edge_vals.size))
        out = np.zeros(arr.shape, dtype=bool)
        notes: list[str] = []
        # Round by data spread, but keep small integer/categorical values exact.
        spread = float(np.nanmax(vals) - np.nanmin(vals)) if vals.size else 0.0
        decimals = 6 if spread < 1000 else 3
        rounded = np.round(edge_vals, decimals)
        uniq, counts = np.unique(rounded, return_counts=True)
        if uniq.size == 0:
            return out, notes
        order = np.argsort(counts)[::-1][:8]
        global_min = float(np.nanmin(vals))
        global_max = float(np.nanmax(vals))
        for i in order:
            cand = float(uniq[i])
            edge_ratio = float(counts[i]) / total_edge
            # Candidate must be common on the outer edge or a common sentinel.
            sentinel_like = cand in {0.0, -9999.0, 9999.0, -32768.0, 32767.0} or abs(cand) > 1e20
            if edge_ratio < 0.08 and not sentinel_like:
                continue
            atol = max(1e-6, abs(cand) * 1e-6, spread * 1e-7)
            close = finite & np.isclose(arr, cand, rtol=0, atol=atol)
            comp = _edge_connected_region(close)
            comp_count = int(np.sum(comp))
            if comp_count <= 0:
                continue
            comp_ratio = comp_count / total_valid
            comp_edge_ratio = float(np.sum(comp & edge)) / total_edge
            # Remove when it is a visible edge-connected block. This is designed
            # for background rectangles, not small legitimate border pixels.
            if (comp_edge_ratio >= 0.12 and comp_ratio >= 0.02) or sentinel_like:
                out |= comp
                label = "edge_modal_value_" + ("0" if abs(cand) < 1e-12 else str(round(cand, 6)))
                notes.append(label)
        return out, notes
    except Exception:
        return np.zeros(arr.shape, dtype=bool), []

def raster_valid_mask(src, arr_float: np.ndarray, masked_arr=None, admin_region: str | None = None) -> tuple[np.ndarray, dict]:
    """Return a robust valid-pixel mask for web preview and hover readout.

    Besides formal nodata and GDAL masks, this masks common untagged background
    values inferred from raster edges so uploaded covariates do not show a
    rectangular mask block in the browser.
    """
    arr = np.asarray(arr_float, dtype="float32")
    valid = np.isfinite(arr)
    try:
        if masked_arr is not None:
            valid &= ~np.ma.getmaskarray(masked_arr)
    except Exception:
        pass
    nodata_values: list[float] = []
    try:
        if src.nodata is not None:
            nd = float(src.nodata)
            nodata_values.append(nd)
            valid &= ~np.isclose(arr, nd, rtol=0, atol=max(1e-6, abs(nd) * 1e-6))
    except Exception:
        pass
    inferred = _infer_edge_background_values(arr, valid)
    for nd in inferred:
        try:
            close = np.isclose(arr, nd, rtol=0, atol=max(1e-6, abs(nd) * 1e-6)) & valid
            # Only remove the edge-connected background component. If the same
            # numeric value appears inside the study area, keep it as a real
            # observation/covariate value.
            bg = _edge_connected_region(close)
            if np.any(bg):
                valid &= ~bg
                nodata_values.append(float(nd))
        except Exception as exc:
            try:
                payload.setdefault("preview_errors", []).append({
                    "name": tif_item.get("name") or Path(str(tif_item.get("path") or "")).name,
                    "path": str(tif_item.get("path") or ""),
                    "error": str(exc)[:500],
                })
            except Exception:
                pass
            continue
    component_bg, component_notes = _infer_edge_background_components(arr, valid)
    if np.any(component_bg):
        valid &= ~component_bg
    modal_bg, modal_notes = _infer_edge_modal_background_components(arr, valid)
    if np.any(modal_bg):
        valid &= ~modal_bg
        component_notes = list(component_notes or []) + list(modal_notes or [])

    admin_meta = {}
    out_shape = arr.shape
    out_transform = getattr(src, "transform", None)
    try:
        if out_shape != (int(src.height), int(src.width)):
            from affine import Affine
            out_transform = src.transform * Affine.scale(src.width / float(out_shape[1]), src.height / float(out_shape[0]))
    except Exception:
        out_transform = getattr(src, "transform", None)
    admin_inside, admin_meta = _admin_boundary_inside_mask(src, admin_region, out_shape=out_shape, out_transform=out_transform)
    if admin_inside is not None and np.any(admin_inside):
        # Apply only when it overlaps the current valid raster. This prevents a
        # misconfigured boundary from blanking the whole preview.
        before = int(np.sum(valid))
        overlap = int(np.sum(valid & admin_inside))
        if before > 0 and overlap / float(before) >= 0.05:
            valid &= admin_inside
            admin_meta["applied"] = True
            admin_meta["valid_before"] = before
            admin_meta["valid_after"] = int(np.sum(valid))
        else:
            admin_meta["applied"] = False
            admin_meta["reason"] = "行政边界与当前栅格有效区重叠过少，未用于透明处理。"

    return valid, {
        "formal_nodata": None if getattr(src, "nodata", None) is None else float(src.nodata),
        "inferred_background_values": inferred,
        "inferred_background_components": component_notes,
        "admin_boundary_mask": admin_meta,
        "nodata_values": nodata_values,
    }


def _cache_key(tif_path: str, palette: str, opacity: float, reverse: bool, palette_colors: list[str] | None = None, clcd_mask_path: str | None = None, *extra_tokens: str) -> str:
    p = Path(tif_path)
    color_sig = ",".join(palette_colors or [])
    mask_sig = ""
    try:
        if clcd_mask_path and Path(clcd_mask_path).exists():
            mp = Path(clcd_mask_path)
            mask_sig = f"|mask:{mp.resolve()}:{mp.stat().st_mtime_ns}"
    except Exception:
        mask_sig = f"|mask:{clcd_mask_path}" if clcd_mask_path else ""
    sig = f"v176rednoncrop|{p.resolve()}|{p.stat().st_mtime_ns if p.exists() else 0}|{palette}|{color_sig}|{opacity:.3f}|{int(reverse)}|{OVERLAY_RESAMPLE_METHOD}|{WEBGIS_LEGEND_RANGE_MODE}{mask_sig}"
    return hashlib.md5(sig.encode("utf-8")).hexdigest()


def local_file_url(path: str | Path) -> str:
    return f"/__local_file?path={quote(str(path))}"



def _resize_rgba_for_web(rgba: np.ndarray) -> tuple[np.ndarray, dict]:
    """Upscale very small rasters for readable browser display.

    The geographic bounds remain unchanged; only the PNG pixel dimensions are
    enlarged. This is a visual-display operation, not a scientific resampling
    step.
    """
    try:
        h, w = rgba.shape[:2]
        if max(h, w) >= OVERLAY_RENDER_MIN_PX:
            return rgba, {"display_resampled": False, "source_png_size": [w, h], "display_png_size": [w, h]}
        factor = max(1, int(np.ceil(OVERLAY_RENDER_MIN_PX / max(h, w))))
        new_size = (int(w * factor), int(h * factor))
        img = Image.fromarray(rgba, mode="RGBA")
        resampling_ns = getattr(Image, "Resampling", Image)
        resampling = getattr(resampling_ns, "BILINEAR", Image.BILINEAR)
        if OVERLAY_RESAMPLE_METHOD in {"nearest", "nearest_neighbor", "none"}:
            resampling = getattr(resampling_ns, "NEAREST", Image.NEAREST)
        img = img.resize(new_size, resample=resampling)
        arr = np.asarray(img, dtype=np.uint8)
        return arr, {
            "display_resampled": True,
            "source_png_size": [w, h],
            "display_png_size": [new_size[0], new_size[1]],
            "display_resample_factor": factor,
        }
    except Exception:
        h, w = rgba.shape[:2]
        return rgba, {"display_resampled": False, "source_png_size": [w, h], "display_png_size": [w, h], "display_resample_error": True}


def _field_lookup(columns: list[str]) -> dict[str, str]:
    return {str(x).strip().lower(): x for x in columns}


def _pick_coord_fields(columns: list[str]) -> tuple[str | None, str | None, str | None]:
    fields = _field_lookup(columns)
    lon_key = (
        fields.get("lon")
        or fields.get("longitude")
        or fields.get("x")
        or fields.get("经度")
        or fields.get("jd")
    )
    lat_key = (
        fields.get("lat")
        or fields.get("latitude")
        or fields.get("y")
        or fields.get("纬度")
        or fields.get("wd")
    )
    som_key = (
        fields.get("som")
        or fields.get("soc")
        or fields.get("有机质")
        or fields.get("土壤有机质")
        or fields.get("soil_organic_matter")
        or fields.get("organic_matter")
    )
    return lon_key, lat_key, som_key


def _points_from_rows(rows, columns: list[str], max_points: int = SAMPLE_POINT_MAX_COUNT) -> list[dict]:
    lon_key, lat_key, som_key = _pick_coord_fields(columns)
    if not lon_key or not lat_key:
        return []
    rows = list(rows)
    step = max(1, int(np.ceil(max(len(rows), 1) / max(max_points, 1))))
    points: list[dict] = []
    for i, row in enumerate(rows):
        if i % step != 0:
            continue
        try:
            lon = float(row.get(lon_key, ""))
            lat = float(row.get(lat_key, ""))
            if not np.isfinite(lon) or not np.isfinite(lat):
                continue
            item = {"lon": lon, "lat": lat}
            if som_key and row.get(som_key, "") not in (None, ""):
                try:
                    item["som"] = float(row.get(som_key, ""))
                except Exception:
                    item["som"] = str(row.get(som_key, ""))
            points.append(item)
            if len(points) >= max_points:
                break
        except Exception:
            continue
    return points


def _read_sample_points_table(path: str | None, max_points: int = SAMPLE_POINT_MAX_COUNT) -> list[dict]:
    """Read uploaded lon/lat/SOM sample points for immediate web-map preview.

    This is intentionally independent from model output. The user should see
    uploaded sample points on the basemap immediately after upload, before any
    GEE extraction or modelling task starts.
    """
    if not path:
        return []
    p = Path(path)
    if not p.exists() or not p.is_file():
        return []
    suffix = p.suffix.lower()
    try:
        if suffix in {".csv", ".txt"}:
            # Try common encodings without failing the whole map payload.
            last_rows = []
            for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
                try:
                    with p.open("r", encoding=enc, newline="", errors="ignore") as f:
                        reader = csv.DictReader(f)
                        if not reader.fieldnames:
                            continue
                        rows = [row for row in reader]
                        pts = _points_from_rows(rows, list(reader.fieldnames), max_points=max_points)
                        if pts:
                            return pts
                        last_rows = rows
                except Exception:
                    continue
            return _points_from_rows(last_rows, list(last_rows[0].keys()) if last_rows else [], max_points=max_points)
        if suffix in {".xlsx", ".xls"} and pd is not None:
            df = pd.read_excel(p)
            if df is None or df.empty:
                return []
            records = df.replace({np.nan: ""}).to_dict(orient="records")
            return _points_from_rows(records, [str(c) for c in df.columns], max_points=max_points)
    except Exception:
        return []
    return []


def _read_sample_points_csv(csv_path: str | None, max_points: int = SAMPLE_POINT_MAX_COUNT) -> list[dict]:
    """Backward-compatible alias used by older result-map code paths."""
    return _read_sample_points_table(csv_path, max_points=max_points)


def _uploaded_sample_points_from_session(session, max_points: int = SAMPLE_POINT_MAX_COUNT) -> dict:
    """Return point payload from the most recent valid uploaded sample table.

    V191 keeps the original sample-point layer independent from raster previews.
    It first trusts preprocessing/role-inference records, then falls back to the
    session file inventory. This prevents a later covariate upload from hiding
    or replacing the previously uploaded sample points.
    """
    candidates: list[tuple[str, str]] = []

    # Cleaned sample generated by preprocessing is preferred when available.
    try:
        prep = getattr(session, "preprocess_report", {}) or {}
        paths = prep.get("paths") or {}
        cleaned = paths.get("cleaned_sample_path")
        if cleaned:
            candidates.append((str(cleaned), Path(str(cleaned)).name))
        for rec in prep.get("sample_records") or []:
            for key in ("cleaned_path", "input_path", "path"):
                val = rec.get(key)
                if val:
                    candidates.append((str(val), rec.get("name") or Path(str(val)).name))
    except Exception:
        pass

    # Role report uses the full accumulated session inventory after every upload.
    try:
        report = getattr(session, "data_role_report", {}) or {}
        for rec in report.get("items") or []:
            if rec.get("role") == "sample_points":
                val = rec.get("path")
                if val:
                    candidates.append((str(val), rec.get("display_name") or rec.get("name") or Path(str(val)).name))
    except Exception:
        pass

    # Fall back to raw upload inventory.
    files = list(getattr(session, "latest_uploaded_files", []) or [])
    for item in reversed(files):
        validation = item.get("validation") or {}
        path = validation.get("path") or item.get("path")
        role_inf = item.get("role_inference") or {}
        role = role_inf.get("role") or item.get("role_guess") or ""
        suffix = Path(str(path or "")).suffix.lower()
        if suffix not in {".csv", ".txt", ".xlsx", ".xls"}:
            continue
        if role and "sample" not in str(role).lower() and "tabular" not in str(role).lower():
            # Do not discard outright: older uploads may have weak role labels.
            # The reader itself verifies lon/lat/SOM fields before returning.
            pass
        candidates.append((str(path), item.get("original_name") or item.get("name") or Path(str(path)).name))

    seen: set[str] = set()
    for path, name in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        suffix = Path(str(path)).suffix.lower()
        if suffix not in {".csv", ".txt", ".xlsx", ".xls"}:
            continue
        points = _read_sample_points_table(path, max_points=max_points)
        if points:
            return {
                "points": points,
                "point_count": len(points),
                "source_name": name or Path(path).name,
                "source_path": str(path),
                "display_default": True,
            }
    return {"points": [], "point_count": 0, "source_name": None, "source_path": None, "display_default": False}



def _parse_hex_color_rgba_local(value: str | None, default: str) -> tuple[int, int, int, int]:
    raw = str(value or default or "").strip()
    if raw.startswith("#"):
        raw = raw[1:]
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    try:
        return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16), 255
    except Exception:
        return (216, 27, 96, 255)


def _find_clcd_mask_for_prediction(tif_path: str | Path) -> str | None:
    """V176: locate the binary CLCD mask next to a prediction GeoTIFF.

    The browser overlay is generated from the scientific SOM GeoTIFF. That
    GeoTIFF correctly stores non-cropland as NoData, but the user-facing browser
    preview must still paint AOI-internal non-cropland red. This helper finds the
    binary mask needed for that compositing step, even when result_paths did not
    explicitly carry clcd_mask_tif.
    """
    try:
        p = Path(tif_path)
        dirs = [p.parent]
        try:
            dirs.append(p.parent.parent)
        except Exception:
            pass
        names = [
            "目标地类掩膜.tif",
            "CLCD目标地类掩膜_1目标0其他.tif",
            "耕地掩膜.tif",
            "CLCD耕地掩膜_1耕地0非耕地.tif",
            "clcd_cropland_mask.tif",
            "cropland_mask.tif",
        ]
        for d in dirs:
            for name in names:
                cand = d / name
                if cand.exists():
                    return str(cand)
            for cand in d.glob("*目标地类*掩膜*.tif"):
                return str(cand)
            for cand in d.glob("*耕地*掩膜*.tif"):
                return str(cand)
            for cand in d.glob("*crop*mask*.tif"):
                return str(cand)
            for cand in d.glob("*landcover*mask*.tif"):
                return str(cand)
    except Exception:
        pass
    return None


def _read_mask_on_prediction_grid(mask_path: str | None, src) -> np.ndarray | None:
    """Read binary CLCD mask on the prediction raster grid.

    Expected mask values: 1=cropland, 0=non-cropland inside AOI, 255=outside AOI.
    If the mask grid differs from the prediction grid, reproject with nearest
    neighbor so display alignment follows the scientific raster grid.
    """
    if not mask_path:
        return None
    try:
        mp = Path(mask_path)
        if not mp.exists():
            return None
        with rasterio.open(mp) as ms:
            if ms.width == src.width and ms.height == src.height and ms.crs == src.crs and ms.transform == src.transform:
                return ms.read(1)
            from rasterio.warp import reproject, Resampling
            dst = np.full((src.height, src.width), 255, dtype="uint8")
            reproject(
                source=ms.read(1),
                destination=dst,
                src_transform=ms.transform,
                src_crs=ms.crs,
                dst_transform=src.transform,
                dst_crs=src.crs,
                resampling=Resampling.nearest,
                src_nodata=ms.nodata if ms.nodata is not None else 255,
                dst_nodata=255,
            )
            return dst
    except Exception:
        return None

def get_overlay_bundle(tif_path: str, palette: str, opacity: float = 1.0, reverse: bool = False, palette_colors: list[str] | None = None, admin_region: str | None = None, clcd_mask_path: str | None = None, noncropland_mode: str = "fill", noncropland_color: str | None = None) -> dict:
    if rasterio is None:
        raise RuntimeError("缺少 rasterio，无法生成结果图层。")
    p = Path(tif_path)
    if not p.exists():
        raise FileNotFoundError(f"未找到结果栅格：{tif_path}")

    normalized_palette = normalize_palette_name(palette)
    effective_colors = list(palette_colors or COLOR_RAMPS.get(normalized_palette, COLOR_RAMPS["ArcGIS Pro｜Green Continuous"]))
    effective_mask_path = clcd_mask_path or _find_clcd_mask_for_prediction(p)
    key = _cache_key(str(p), normalized_palette, opacity, reverse, effective_colors + ([f"admin:{admin_region}"] if admin_region else []), effective_mask_path, f"noncrop_mode:{noncropland_mode}", f"noncrop_color:{noncropland_color or ''}")
    out_png = STYLED_DIR / f"{key}.png"
    meta_json = STYLED_DIR / f"{key}.json"

    if out_png.exists() and meta_json.exists():
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
        meta["overlay_url"] = local_file_url(out_png)
        return meta

    with rasterio.open(p) as src:
        raw = src.read(1, masked=True).astype("float32")
        arr = np.asarray(raw.filled(np.nan), dtype="float32")
        valid, nodata_meta = raster_valid_mask(src, arr, raw, admin_region=admin_region)
        clcd_display_mask = _read_mask_on_prediction_grid(effective_mask_path, src)
        if not np.any(valid):
            raise RuntimeError("结果栅格没有有效像元，无法生成叠加图层。")
        actual_zmin = float(np.nanmin(arr[valid]))
        actual_zmax = float(np.nanmax(arr[valid]))
        if WEBGIS_LEGEND_RANGE_MODE in {"percentile", "robust", "p2p98"}:
            zmin = float(np.nanpercentile(arr[valid], 2))
            zmax = float(np.nanpercentile(arr[valid], 98))
        else:
            zmin = actual_zmin
            zmax = actual_zmax
        if zmax <= zmin:
            zmax = zmin + 1e-6
        vals = np.clip((arr - zmin) / (zmax - zmin), 0.0, 1.0)
        try:
            gamma = float(WEBGIS_RESULT_GAMMA)
        except Exception:
            gamma = 1.0
        if gamma > 0 and abs(gamma - 1.0) > 1e-6:
            vals = np.power(vals, gamma)
        idx = np.zeros(arr.shape, dtype=np.uint8)
        idx[valid] = np.round(vals[valid] * 255).astype(np.uint8)
        colors = list(effective_colors)
        if reverse:
            colors.reverse()
        lut = _build_lut(colors)
        rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
        rgba[..., :3] = lut[idx]
        rgba[..., 3] = np.where(valid, int(max(0.05, min(1.0, opacity)) * 255), 0).astype(np.uint8)
        rgba[~valid, :3] = 0
        # V176: browser SOM overlay must visually match the CLCD cropland mask.
        # Cropland (CLCD=1) keeps the continuous green SOM ramp. AOI-internal
        # non-cropland (CLCD=2-8 or invalid, encoded as 0 in the binary mask) is
        # painted solid red. AOI outside (255) stays transparent. This is display
        # compositing only; the underlying GeoTIFF still stores non-cropland as NoData.
        if clcd_display_mask is not None and clcd_display_mask.shape == rgba.shape[:2]:
            if str(noncropland_mode or "fill").lower() in {"transparent", "mask", "hide"}:
                rgba[clcd_display_mask == 0] = (0, 0, 0, 0)
            else:
                non_color = _parse_hex_color_rgba_local(noncropland_color or os.getenv("PRO_CLCD_NON_CROPLAND_COLOR", "#E53935"), "#E53935")
                rgba[clcd_display_mask == 0] = non_color
            rgba[clcd_display_mask == 255] = (0, 0, 0, 0)
        valid_count = int(np.sum(valid))
        total_count = int(valid.size)
        bounds = _safe_transform_bounds(src)

    rgba, display_meta = _resize_rgba_for_web(rgba)
    Image.fromarray(rgba, mode="RGBA").save(out_png)
    west, south, east, north = bounds
    meta = {
        "tif_path": str(p),
        "bounds": [[south, west], [north, east]],
        "zmin": actual_zmin,
        "zmax": actual_zmax,
        "style_zmin": zmin,
        "style_zmax": zmax,
        "palette": (palette if palette_colors else normalized_palette),
        "palette_colors": colors,
        "admin_region": admin_region,
        "opacity": opacity,
        "reverse": reverse,
        "overlay_path": str(out_png),
        "overlay_url": local_file_url(out_png),
        "valid_cell_count": valid_count,
        "total_cell_count": total_count,
        "valid_cell_ratio": (valid_count / total_count) if total_count else None,
        "nodata_meta": nodata_meta,
        "clcd_mask_path": str(effective_mask_path or ""),
        "display_policy": ("V196: CLCD=1 cropland keeps the continuous SOM ramp; CLCD=0 non-cropland is rendered as transparent when the user requests cropland-only mapping, otherwise it is shown with the configured contrast color; AOI outside stays transparent."),
        "noncropland_mode": noncropland_mode,
        **display_meta,
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta




def _result_display_title(paths: dict | None, default: str) -> str:
    p = paths or {}
    title = str(p.get("display_map_title") or p.get("map_title") or p.get("map_title_base") or "").strip()
    return title or default

def _uncertainty_display_title(session, default: str = "土壤有机质不确定性分析图") -> str:
    gcp = getattr(session, "gcp_result_paths", {}) or {}
    if gcp.get("display_map_title") or gcp.get("map_title"):
        return str(gcp.get("display_map_title") or gcp.get("map_title"))
    rfk = getattr(session, "rfk_result_paths", {}) or {}
    base = str(rfk.get("display_map_title") or rfk.get("map_title") or rfk.get("map_title_base") or "").strip()
    if base:
        return base.replace("图", "不确定性分析图") if base.endswith("图") else (base + "不确定性分析图")
    return default

def build_stage_payload(session) -> dict:
    prefs = getattr(session, "map_style_prefs", {}) or {}
    try:
        preferred_opacity = float(prefs.get("opacity") if prefs.get("opacity") is not None else RESULT_OVERLAY_OPACITY)
    except Exception:
        preferred_opacity = RESULT_OVERLAY_OPACITY
    preferred_opacity = max(0.05, min(1.0, preferred_opacity))
    preferred_palette = normalize_palette_name(prefs.get("palette") or None) if prefs.get("palette") else None
    payload = {
        "session_id": getattr(session, "session_id", ""),
        "title": "成都市土壤有机质专题图",
        "map": {
            "tdt_token": TDT_TOKEN,
            "default_center": DEFAULT_CENTER,
            "default_bounds": DEFAULT_BOUNDS,
            "basemap_auto_hide_after_result": BASEMAP_AUTO_HIDE_AFTER_RESULT,
            "has_uploaded_data": bool(getattr(session, "latest_uploaded_files", []) or getattr(session, "uploaded_result_paths", {}).get("uploaded_tif")),
            "show_basemap": prefs.get("show_basemap"),
            "show_samples": prefs.get("show_samples"),
        },
        "result": None,
        "uploaded_samples": _uploaded_sample_points_from_session(session),
        "palette_options": palette_options(),
        "layout_preferences": {**(getattr(session, "map_style_prefs", {}) or {}), **(getattr(session, "map_layout_prefs", {}) or {})},
        "result_key": "empty",
        "layouts": [],
        "active_layout_key": None,
        "preview_errors": [],
        "pending_layer_loaded_messages": list(((getattr(session, "task_memory", {}) or {}).get("pending_layer_loaded_messages") or [])),
        "pending_style_loaded_messages": list(((getattr(session, "task_memory", {}) or {}).get("pending_style_loaded_messages") or [])),
    }

    # Fail-safe promotion: if a backend task has finished but the browser store
    # has not yet been updated by the polling callback, still show the generated
    # result.  GCP/AOA completion has priority over the earlier SOM map; otherwise
    # every payload rebuild would force the workspace back to the SOM layout.
    try:
        from services.task_service import TASKS  # lazy import to avoid cycles
        gcp_task_id = getattr(session, "latest_gcp_task_id", None)
        if gcp_task_id:
            gcp_task = TASKS.get(gcp_task_id)
            if gcp_task and gcp_task.status == "done" and isinstance(gcp_task.result_paths, dict) and gcp_task.result_paths.get("width_tif"):
                session.gcp_shown = True
                session.latest_result_kind = "gcp_map"
                session.gcp_result_paths = dict(gcp_task.result_paths or {})
                session.last_result_paths = dict(gcp_task.result_paths or {})
                try:
                    session.map_style_prefs = {**(getattr(session, "map_style_prefs", {}) or {}), "show_samples": False}
                except Exception:
                    pass
                # V239：GCP 完成时只自动切换一次。随后用户手动/指令切换到
                # 有机质图或上传协变量图层时，build_stage_payload 不能每次轮询都
                # 把 active_layout_key 强行改回 gcp_layout。
                try:
                    mem = getattr(session, "task_memory", {}) or {}
                    key = "auto_switched_gcp_layout_task_id"
                    if mem.get(key) != str(gcp_task_id) and getattr(session, "active_layout_key", None) in {None, "", "gcp_running"}:
                        session.active_layout_key = "gcp_layout"
                    mem[key] = str(gcp_task_id)
                    session.task_memory = mem
                except Exception:
                    if getattr(session, "active_layout_key", None) in {None, "", "gcp_running"}:
                        session.active_layout_key = "gcp_layout"
        task_id = getattr(session, "latest_rfk_task_id", None)
        if task_id and not bool(getattr(session, "gcp_shown", False)) and getattr(session, "latest_result_kind", None) not in {"gcp_running", "gcp_map"}:
            task = TASKS.get(task_id)
            if task and task.status == "done" and isinstance(task.result_paths, dict) and task.result_paths.get("pred_tif"):
                session.som_map_shown = True
                session.latest_result_kind = "som_map"
                session.rfk_result_paths = dict(task.result_paths or {})
                session.last_result_paths = dict(task.result_paths or {})
                try:
                    session.map_style_prefs = {**(getattr(session, "map_style_prefs", {}) or {}), "show_samples": False}
                except Exception:
                    pass
                try:
                    mem = getattr(session, "task_memory", {}) or {}
                    key = "auto_switched_som_layout_task_id"
                    if mem.get(key) != str(task_id) and getattr(session, "active_layout_key", None) in {None, "", "rfk_running"}:
                        session.active_layout_key = "som_layout"
                    mem[key] = str(task_id)
                    session.task_memory = mem
                except Exception:
                    if getattr(session, "active_layout_key", None) in {None, "", "rfk_running"}:
                        session.active_layout_key = "som_layout"
    except Exception:
        pass

    layouts = []
    uploaded_admin_region = _region_from_uploaded_samples(session)

    if getattr(session, "som_map_shown", False) and session.rfk_result_paths.get("pred_tif") and Path(session.rfk_result_paths.get("pred_tif")).exists():
        palette = preferred_palette or normalize_palette_name(getattr(session, "som_palette", None))
        clcd_mask_for_display = session.rfk_result_paths.get("clcd_mask_tif") or _find_clcd_mask_for_prediction(session.rfk_result_paths.get("pred_tif"))
        layout_prefs = getattr(session, "map_layout_prefs", {}) or {}
        map_scope = str(layout_prefs.get("map_scope") or (session.rfk_result_paths.get("map_scope") if isinstance(session.rfk_result_paths, dict) else "") or "full_domain")
        # For any target land-cover map (cropland/forest/grassland/etc.), non-target
        # land-cover cells remain transparent.  Palette changes affect only the
        # visible target-class prediction raster unless the user explicitly asks to
        # recolor non-target land-cover cells.
        noncropland_mode = "transparent" if map_scope in {"cropland", "landcover_class"} or bool(layout_prefs.get("mask_by_landcover")) else "fill"
        noncropland_color = str(layout_prefs.get("non_cropland_color") or os.getenv("PRO_CLCD_NON_CROPLAND_COLOR", "#E53935"))
        overlay = get_overlay_bundle(
            session.rfk_result_paths.get("pred_tif"),
            palette=palette,
            opacity=preferred_opacity,
            reverse=False,
            clcd_mask_path=clcd_mask_for_display,
            noncropland_mode=noncropland_mode,
            noncropland_color=noncropland_color,
        )
        map_title = _result_display_title(session.rfk_result_paths, "土壤有机质图")
        # Final formal maps hide training/sample points by default.  Sample
        # points remain available during upload/covariate preview, but should not
        # sit on top of the deliverable SOM map unless the user explicitly asks.
        layouts.append({
            "key": "som_layout",
            "tab_title": map_title,
            "title": map_title,
            "result": {**overlay, "legend_title": "有机质含量（g/kg）", "kind": "som_map", "sample_points": [], "sample_point_count": 0},
            "result_key": _cache_key(session.rfk_result_paths.get("pred_tif"), palette, overlay["opacity"], overlay["reverse"], None, clcd_mask_for_display, f"noncrop_mode:{noncropland_mode}", f"noncrop_color:{noncropland_color}"),
        })

    if getattr(session, "gcp_shown", False) and session.gcp_result_paths.get("width_tif") and Path(session.gcp_result_paths.get("width_tif")).exists():
        palette = preferred_palette or normalize_palette_name(getattr(session, "gcp_palette", None))
        overlay = get_overlay_bundle(session.gcp_result_paths.get("width_tif"), palette=palette, opacity=preferred_opacity, reverse=False)
        gcp_title = _uncertainty_display_title(session)
        layouts.append({
            "key": "gcp_layout",
            "tab_title": gcp_title,
            "title": gcp_title,
            "result": {**overlay, "legend_title": "区间宽度（g/kg）", "kind": "gcp_map", "sample_points": [], "sample_point_count": 0},
            "result_key": _cache_key(session.gcp_result_paths.get("width_tif"), palette, overlay["opacity"], overlay["reverse"]),
        })

    # Uploaded GeoTIFFs are kept in upload order. Each one can be inspected
    # individually, and when multiple covariate rasters exist a cumulative stack
    # layout is also provided so users can check alignment/coverage visually.
    uploaded_tifs = []
    pre_map = _preprocess_covariate_record_map(session)
    for item in list(getattr(session, "latest_uploaded_files", []) or []):
        original_pth = str(item.get("path") or "")
        if original_pth.lower().endswith((".tif", ".tiff")) and Path(original_pth).exists():
            role_inf = item.get("role_inference") or {}
            role_code = role_inf.get("role") or item.get("role_guess") or "uploaded_raster"
            role = role_inf.get("label") or item.get("role_guess") or "上传栅格"
            rec = pre_map.get(original_pth)
            display_pth = _display_tif_from_preprocess(original_pth, rec)
            uploaded_tifs.append({
                "path": display_pth,
                "original_path": original_pth,
                "name": item.get("original_name") or item.get("name") or Path(original_pth).name,
                "role": role,
                "role_code": role_code,
                "alignment": (rec or {}).get("alignment") if rec else None,
            })
    if getattr(session, "uploaded_result_paths", {}).get("uploaded_tif"):
        pth = str(session.uploaded_result_paths.get("uploaded_tif"))
        if Path(pth).exists() and pth not in {x["path"] for x in uploaded_tifs}:
            uploaded_tifs.append({"path": pth, "original_path": pth, "name": Path(pth).name, "role": "上传/下载栅格", "role_code": "uploaded_result"})

    stack_overlays = []
    max_preview_layers = max(1, int(UPLOAD_PREVIEW_LAYER_MAX_COUNT))
    if len(uploaded_tifs) > max_preview_layers:
        try:
            payload["preview_errors"].append(f"当前识别到 {len(uploaded_tifs)} 个可预览栅格；出于浏览器性能保护，本次发布前 {max_preview_layers} 个图层。可通过 WEBGIS_UPLOAD_PREVIEW_LAYER_MAX_COUNT 调整。")
        except Exception:
            pass
    for idx, tif_item in enumerate(uploaded_tifs[:max_preview_layers], start=1):
        try:
            palette = preferred_palette or _upload_layer_palette(idx)
            overlay = get_overlay_bundle(tif_item["path"], palette=palette, opacity=preferred_opacity, reverse=False, admin_region=uploaded_admin_region)
            display_name = str(tif_item.get("name") or Path(tif_item["path"]).name)
            overlay_with_name = {
                **overlay,
                "source_name": display_name,
                "source_role": tif_item["role"],
                "source_role_code": tif_item.get("role_code"),
                "original_tif_path": tif_item.get("original_path"),
                "alignment": tif_item.get("alignment"),
                "stack_order": idx,
                "stack_opacity": 0.38,
            }
            stack_overlays.append(overlay_with_name)
            layouts.append({
                "key": f"uploaded_layout_{idx}",
                "tab_title": display_name,
                "title": f"用户上传数据预览：{display_name}",
                "result": {**overlay_with_name, "legend_title": "栅格值", "kind": "uploaded_raster"},
                "result_key": _cache_key(tif_item["path"], palette, overlay["opacity"], overlay["reverse"], overlay.get("palette_colors")),
            })
        except Exception:
            continue

    # Do not create a cumulative multi-raster stack by default. Users found the
    # mixed stack visually confusing, and it can make one layer's background
    # appear to belong to another layer. The center selector now previews one
    # uploaded raster at a time; AI commands can later enable derived overlays.

    payload["layouts"] = layouts

    active_key = getattr(session, "active_layout_key", None)
    valid_keys = {item["key"] for item in layouts}
    if active_key not in valid_keys:
        if getattr(session, "latest_result_kind", None) == "gcp_map" and "gcp_layout" in valid_keys:
            active_key = "gcp_layout"
        elif getattr(session, "latest_result_kind", None) == "som_map" and "som_layout" in valid_keys:
            active_key = "som_layout"
        elif any(item.get("key") == "uploaded_layout_1" for item in layouts):
            # When the user has uploaded covariate rasters but no modeled result yet,
            # preview the first uploaded file instead of the cumulative stack. The
            # center dropdown can then switch to any other uploaded file.
            active_key = "uploaded_layout_1"
        elif layouts:
            active_key = layouts[-1]["key"]
        else:
            active_key = None

    payload["active_layout_key"] = active_key
    active_layout = next((item for item in layouts if item["key"] == active_key), None)
    if active_layout:
        payload["title"] = active_layout["title"]
        payload["result"] = active_layout["result"]
        payload["result_key"] = active_layout["result_key"]
        if (payload.get("result") or {}).get("kind") in {"som_map", "gcp_map"}:
            payload["map"]["show_samples"] = False

    return payload
