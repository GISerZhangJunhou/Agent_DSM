from __future__ import annotations

import json
import math
import os
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.mask import mask as rasterio_mask
    from rasterio.warp import transform_geom
except Exception:  # pragma: no cover
    rasterio = None
    from_origin = None
    rasterio_mask = None
    transform_geom = None

try:
    import fiona
except Exception:  # pragma: no cover
    fiona = None

try:
    import xarray as xr
except Exception:  # pragma: no cover
    xr = None

RASTER_EXTS = {".tif", ".tiff", ".img", ".vrt"}
MULTIDIM_EXTS = {".nc", ".hdf", ".h5", ".hdf5"}
ARCHIVE_EXTS = {".zip"}


def _load_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_name(s: Any) -> str:
    txt = str(s or "data").strip()
    txt = re.sub(r"[\\/:*?\"<>|\s]+", "_", txt)
    return txt.strip("_")[:80] or "data"


def _band_year_from_tags(ds: Any, band: int) -> int | None:
    texts: list[str] = []
    try:
        texts.extend([str(v) for v in (ds.tags(band) or {}).values()])
    except Exception:
        pass
    try:
        desc = ds.descriptions[band - 1]
        if desc:
            texts.append(str(desc))
    except Exception:
        pass
    try:
        # GDAL NetCDF often stores time as metadata on bands.
        texts.append(str(ds.tags().get(f"NETCDF_DIM_time_VALUES", "")))
    except Exception:
        pass
    joined = " ".join(texts)
    m = re.search(r"(19\d{2}|20\d{2})", joined)
    if m:
        return int(m.group(1))
    return None


def _choose_band(ds: Any, requested_year: int | None) -> tuple[int, str]:
    count = int(getattr(ds, "count", 1) or 1)
    if count <= 1:
        return 1, "single_band"
    if requested_year:
        for b in range(1, count + 1):
            yr = _band_year_from_tags(ds, b)
            if yr == requested_year:
                return b, f"band_year_match:{requested_year}"
    return 1, "fallback_first_band_no_year_match"


def _write_single_band_tif(src_ds: Any, arr: np.ndarray, out_tif: Path, band_index: int, reason: str) -> dict[str, Any]:
    if rasterio is None:
        raise RuntimeError("rasterio 不可用，无法写 GeoTIFF。")
    profile = src_ds.profile.copy()
    profile.update({
        "driver": "GTiff",
        "height": int(arr.shape[0]),
        "width": int(arr.shape[1]),
        "count": 1,
        "dtype": str(arr.dtype),
        "compress": "deflate",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    })
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(arr, 1)
        try:
            dst.set_band_description(1, f"selected_band_{band_index}")
            dst.update_tags(conversion_reason=reason)
        except Exception:
            pass
    return {"ok": True, "path": str(out_tif), "method": "rasterio", "selected_band": band_index, "reason": reason}


def _convert_with_rasterio(path: Path, out_tif: Path, requested_year: int | None) -> dict[str, Any] | None:
    if rasterio is None:
        return None
    errors: list[str] = []
    candidates = [str(path)]
    try:
        with rasterio.open(path) as root:
            subdatasets = list(getattr(root, "subdatasets", []) or [])
            if subdatasets:
                candidates = subdatasets
    except Exception as exc:
        errors.append(f"root_open_failed:{exc}")
    for cand in candidates:
        try:
            with rasterio.open(cand) as ds:
                if getattr(ds, "width", 0) <= 0 or getattr(ds, "height", 0) <= 0:
                    continue
                band, reason = _choose_band(ds, requested_year)
                arr = ds.read(band)
                if arr.ndim != 2:
                    arr = np.asarray(arr).squeeze()
                if arr.ndim != 2:
                    errors.append(f"not_2d:{cand}")
                    continue
                meta = _write_single_band_tif(ds, arr, out_tif, band, reason)
                meta["source_dataset"] = cand
                meta["errors"] = errors[-5:]
                return meta
        except Exception as exc:
            errors.append(f"candidate_failed:{cand}:{exc}")
    return {"ok": False, "error": "; ".join(errors[-6:]) or "rasterio_conversion_failed"}


def _find_xy_names(ds: Any) -> tuple[str | None, str | None]:
    coords = list(getattr(ds, "coords", {}).keys())
    lon_names = ["lon", "longitude", "x", "经度"]
    lat_names = ["lat", "latitude", "y", "纬度"]
    lon = next((n for n in lon_names if n in coords), None)
    lat = next((n for n in lat_names if n in coords), None)
    if lon and lat:
        return lon, lat
    for n in coords:
        lo = str(n).lower()
        if lon is None and ("lon" in lo or lo == "x"):
            lon = n
        if lat is None and ("lat" in lo or lo == "y"):
            lat = n
    return lon, lat


def _convert_with_xarray(path: Path, out_tif: Path, requested_year: int | None) -> dict[str, Any] | None:
    if xr is None or rasterio is None or from_origin is None:
        return None
    try:
        ds = xr.open_dataset(path, decode_times=True)
    except Exception as exc:
        return {"ok": False, "error": f"xarray_open_failed:{exc}"}
    try:
        lon_name, lat_name = _find_xy_names(ds)
        if not lon_name or not lat_name:
            return {"ok": False, "error": "xarray_no_lon_lat_coords"}
        data_var = None
        for name, da in ds.data_vars.items():
            dims = set(map(str, da.dims))
            if lon_name in dims and lat_name in dims and np.issubdtype(da.dtype, np.number):
                # Avoid obvious QC/bounds variables.
                low = str(name).lower()
                if any(x in low for x in ["qc", "quality", "flag", "bounds", "bnds"]):
                    continue
                data_var = name
                break
        if not data_var:
            return {"ok": False, "error": "xarray_no_suitable_data_variable"}
        da = ds[data_var]
        selected_reason = "no_time_dimension"
        time_dim = next((d for d in da.dims if "time" in str(d).lower()), None)
        if time_dim and requested_year:
            try:
                years = da[time_dim].dt.year
                sub = da.where(years == int(requested_year), drop=True)
                if int(sub.sizes.get(time_dim, 0)) > 0:
                    da = sub
                    selected_reason = f"time_year_filter:{requested_year}"
                    # A multi-temporal year cannot be displayed as multiple frames in this app yet.
                    # Use mean for map preview and keep raw file unchanged.
                    if int(da.sizes.get(time_dim, 0)) > 1:
                        da = da.mean(dim=time_dim, skipna=True)
                        selected_reason += ":mean_within_year"
                    else:
                        da = da.isel({time_dim: 0})
                else:
                    da = da.isel({time_dim: 0})
                    selected_reason = "fallback_first_time_no_requested_year"
            except Exception:
                da = da.isel({time_dim: 0})
                selected_reason = "fallback_first_time_year_parse_failed"
        elif time_dim:
            da = da.isel({time_dim: 0})
            selected_reason = "fallback_first_time_no_requested_year"
        # Squeeze any remaining singleton dimensions.
        for d in list(da.dims):
            if d not in {lat_name, lon_name} and int(da.sizes.get(d, 0)) == 1:
                da = da.isel({d: 0})
        if set(da.dims) != {lat_name, lon_name}:
            # Try to average over non-spatial leftovers.
            for d in list(da.dims):
                if d not in {lat_name, lon_name}:
                    da = da.mean(dim=d, skipna=True)
        da = da.transpose(lat_name, lon_name)
        arr = np.asarray(da.values)
        if arr.ndim != 2:
            return {"ok": False, "error": f"xarray_selected_data_not_2d:{arr.shape}"}
        lons = np.asarray(ds[lon_name].values, dtype=float)
        lats = np.asarray(ds[lat_name].values, dtype=float)
        if lons.ndim != 1 or lats.ndim != 1:
            return {"ok": False, "error": "xarray_lon_lat_not_1d"}
        # GeoTIFF expects north-up. Flip if latitude is ascending south->north.
        if lats[0] < lats[-1]:
            arr = np.flipud(arr)
            lats = lats[::-1]
        xres = abs(float(lons[1] - lons[0])) if len(lons) > 1 else 0.01
        yres = abs(float(lats[0] - lats[1])) if len(lats) > 1 else 0.01
        west = float(np.nanmin(lons)) - xres / 2.0
        north = float(np.nanmax(lats)) + yres / 2.0
        transform = from_origin(west, north, xres, yres)
        profile = {
            "driver": "GTiff",
            "height": int(arr.shape[0]),
            "width": int(arr.shape[1]),
            "count": 1,
            "dtype": str(arr.dtype),
            "crs": "EPSG:4326",
            "transform": transform,
            "compress": "deflate",
            "tiled": True,
            "BIGTIFF": "IF_SAFER",
            "nodata": np.nan if np.issubdtype(arr.dtype, np.floating) else None,
        }
        out_tif.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(arr, 1)
            dst.set_band_description(1, str(data_var))
            dst.update_tags(source_variable=str(data_var), conversion_reason=selected_reason)
        return {"ok": True, "path": str(out_tif), "method": "xarray", "source_variable": str(data_var), "reason": selected_reason}
    finally:
        try:
            ds.close()
        except Exception:
            pass


def convert_multidimensional_to_tif(path: str | Path, out_dir: str | Path, requested_year: int | None = None, data_name: str | None = None) -> dict[str, Any]:
    p = Path(path)
    out = Path(out_dir) / f"{_safe_name(data_name or p.stem)}_{requested_year or 'selected'}_preview.tif"
    first = _convert_with_rasterio(p, out, requested_year)
    if first and first.get("ok"):
        return first
    second = _convert_with_xarray(p, out, requested_year)
    if second and second.get("ok"):
        if first:
            second["rasterio_attempt"] = first
        return second
    return {"ok": False, "error": "NetCDF/HDF 转 GeoTIFF 失败", "rasterio_attempt": first, "xarray_attempt": second}


def _extract_zip(path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    try:
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(out_dir)
        for p in out_dir.rglob("*"):
            if p.is_file():
                files.append(p)
    except Exception:
        pass
    return files


def _candidate_files_from_manifest(manifest: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for rec in manifest.get("validated_files") or []:
        p = rec.get("path")
        if p:
            paths.append(Path(str(p)))
    for key in ["local_save_dir", "manual_download_dir"]:
        d = manifest.get(key)
        if d and Path(str(d)).exists():
            for p in Path(str(d)).rglob("*"):
                if p.is_file() and p.suffix.lower() in (RASTER_EXTS | MULTIDIM_EXTS | ARCHIVE_EXTS):
                    paths.append(p)
    # De-duplicate while preserving order.
    seen = set()
    out: list[Path] = []
    for p in paths:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def process_download_manifest_for_map(manifest_path: str | Path | None) -> dict[str, Any]:
    """Prepare a verified downloaded file for WebGIS display.

    - GeoTIFF: use directly.
    - ZIP: extract and search for GeoTIFF/NetCDF.
    - NetCDF/HDF: convert the requested year to a single GeoTIFF for map display.

    The raw source file is never deleted. For multi-year data, only the requested
    year is materialized into the preview GeoTIFF; other years are not exported.
    """
    manifest = _load_json(manifest_path)
    if not manifest:
        return {"ok": False, "error": "未找到 download_manifest.json。"}
    requested_year = manifest.get("year") or manifest.get("requested_year")
    try:
        requested_year = int(requested_year) if requested_year else None
    except Exception:
        requested_year = None
    data_name = manifest.get("data_name") or "downloaded_data"
    base_dir = Path(str(manifest.get("local_save_dir") or Path(str(manifest_path)).parent))
    display_dir = base_dir / "_display"
    display_dir.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_files_from_manifest(manifest)
    extracted: list[Path] = []
    for p in list(candidates):
        if p.suffix.lower() == ".zip":
            extracted.extend(_extract_zip(p, display_dir / f"extract_{_safe_name(p.stem)}"))
    candidates.extend(extracted)
    attempts: list[dict[str, Any]] = []
    for p in candidates:
        ext = p.suffix.lower()
        if ext in RASTER_EXTS:
            manifest.update({
                "display_ready": True,
                "display_tif": str(p),
                "display_source_file": str(p),
                "postprocess_status": "display_ready_existing_raster",
            })
            if manifest_path:
                _save_json(manifest_path, manifest)
            return {"ok": True, "display_tif": str(p), "manifest": manifest, "method": "existing_raster"}
        if ext in MULTIDIM_EXTS:
            res = convert_multidimensional_to_tif(p, display_dir, requested_year, str(data_name))
            attempts.append({"source": str(p), **res})
            if res.get("ok") and res.get("path"):
                manifest.update({
                    "display_ready": True,
                    "display_tif": str(res.get("path")),
                    "display_source_file": str(p),
                    "postprocess_status": "converted_multidimensional_to_geotiff",
                    "postprocess_note": f"已从多时相数据中提取/聚合 {requested_year or '首个可用时段'}，仅生成该时段的预览 GeoTIFF；原始多时相文件保留在 raw 目录。",
                    "postprocess_attempts": attempts,
                })
                if manifest_path:
                    _save_json(manifest_path, manifest)
                return {"ok": True, "display_tif": str(res.get("path")), "manifest": manifest, "method": res.get("method"), "conversion": res}
    manifest.update({"display_ready": False, "postprocess_status": "no_displayable_raster", "postprocess_attempts": attempts})
    if manifest_path:
        _save_json(manifest_path, manifest)
    return {"ok": False, "error": "未发现可显示的 GeoTIFF，且 NetCDF/HDF 转换失败或无可用变量。", "attempts": attempts, "manifest": manifest}



def _is_helper_path(path: Path) -> bool:
    parts = {str(x).lower() for x in path.parts}
    name = path.name.lower()
    helper_names = {
        "ftp_download_status.json", "ftp_python_downloader.log", "ftp_accounts_detected.json",
        "open_winscp_download.bat", "winscp_download_script.txt", "open_ftp_client.bat",
        "download_manifest.json",
    }
    if name in helper_names:
        return True
    if any(x in parts for x in {"meta", "ftp_tickets", "_display", "__pycache__"}):
        return True
    return False


def _download_data_candidates(root: Path) -> list[Path]:
    exts = RASTER_EXTS | MULTIDIM_EXTS | ARCHIVE_EXTS | {".grib", ".grb", ".grb2", ".gz", ".tar"}
    out: list[Path] = []
    if not root.exists():
        return out
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if _is_helper_path(p):
            continue
        if p.suffix.lower() in exts:
            out.append(p)
    return sorted(out, key=lambda x: (len(x.parts), str(x).lower()))


def _files_matching_year(files: list[Path], requested_year: int | None) -> list[Path]:
    if not requested_year:
        return files
    year = str(int(requested_year))
    explicit = [p for p in files if re.search(rf"(?<!\d){re.escape(year)}(?!\d)", p.name)]
    # If the server has explicit target-year files, keep only those. Otherwise
    # keep the original multi-year files, because TPDC often stores all years in
    # one NetCDF/HDF file and the year is selected during conversion.
    return explicit or files


def _unique_dest(dest_dir: Path, name: str) -> Path:
    dest = dest_dir / name
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    for i in range(2, 9999):
        cand = dest_dir / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
    return dest_dir / f"{stem}_{int(time.time())}{suffix}"


def _move_or_copy_data_file(src: Path, dest_dir: Path, *, move: bool = True) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = _unique_dest(dest_dir, src.name)
    if src.resolve() == dest.resolve():
        return dest
    if move:
        shutil.move(str(src), str(dest))
    else:
        shutil.copy2(str(src), str(dest))
    return dest


def _infer_mode_from_manifest(manifest: dict[str, Any], mapping_intent: bool | None) -> str:
    if mapping_intent is True:
        return "mapping"
    if mapping_intent is False:
        return "download_only"
    text = " ".join(str(manifest.get(k) or "") for k in ["request_text", "data_name", "task_type"]).lower()
    if any(x in text for x in ["制图", "建模", "入模", "协变量", "土壤有机质", "som", "rfk", "结合"]):
        return "mapping"
    return "download_only"




def _admin_root() -> Path:
    return Path(os.getenv("DSM_ADMIN_BOUNDARY_ROOT", r"E:\Agent_DSM\行政区划"))


def _normalize_region_text(value: Any) -> str:
    return re.sub(r"[\s_\-·•]+", "", str(value or "").strip())


def _region_aliases(region: Any) -> list[str]:
    txt = _normalize_region_text(region)
    if not txt:
        return []
    aliases = {txt}
    for suf in ["市", "区", "县", "自治县", "省"]:
        if txt.endswith(suf) and len(txt) > len(suf):
            aliases.add(txt[: -len(suf)])
    if txt == "成都":
        aliases.add("成都市")
    if txt == "成都市":
        aliases.add("成都")
    return [x for x in aliases if x]


def _candidate_admin_shps() -> list[Path]:
    root = _admin_root()
    names = ["县.shp", "区县.shp", "县级.shp", "市.shp", "地级市.shp", "省.shp"]
    out = [root / n for n in names if (root / n).exists()]
    if root.exists():
        for p in root.rglob("*.shp"):
            if p not in out:
                out.append(p)
    return out


def _load_region_geoms(region: Any) -> tuple[list[dict], Any, str] | None:
    if fiona is None:
        return None
    aliases = set(_region_aliases(region))
    if not aliases:
        return None
    for shp in _candidate_admin_shps():
        try:
            with fiona.open(shp, "r") as src:
                crs = src.crs_wkt or src.crs
                geoms: list[dict] = []
                matched_name = ""
                for feat in src:
                    props = dict(feat.get("properties") or {})
                    values = [_normalize_region_text(v) for v in props.values() if v is not None]
                    if any(a in values or any(a and a in v for v in values) for a in aliases):
                        geom = feat.get("geometry")
                        if geom:
                            geoms.append(geom)
                            matched_name = next((v for v in values if v), matched_name)
                if geoms:
                    return geoms, crs, f"{shp}|{matched_name}"
        except Exception:
            continue
    return None


def _clip_tif_to_region(src_tif: Path, region: Any, out_tif: Path) -> dict[str, Any]:
    if rasterio is None or rasterio_mask is None or transform_geom is None:
        return {"ok": False, "error": "rasterio/fiona 裁剪组件不可用"}
    region_info = _load_region_geoms(region)
    if not region_info:
        return {"ok": False, "error": f"未找到可用于裁剪的行政区边界：{region}"}
    geoms, src_crs, match_source = region_info
    try:
        with rasterio.open(src_tif) as ds:
            target_crs = ds.crs
            use_geoms = geoms
            if target_crs and src_crs:
                use_geoms = [transform_geom(src_crs, target_crs, g) for g in geoms]
            arr, transform = rasterio_mask(ds, use_geoms, crop=True, nodata=ds.nodata, filled=True)
            profile = ds.profile.copy()
            profile.update({
                "height": int(arr.shape[1]),
                "width": int(arr.shape[2]),
                "transform": transform,
                "compress": "deflate",
                "tiled": True,
                "BIGTIFF": "IF_SAFER",
            })
            out_tif.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_tif, "w", **profile) as dst:
                dst.write(arr)
                try:
                    dst.update_tags(region_clip=str(region), region_boundary_source=str(match_source), postprocess="year_region_filtered")
                except Exception:
                    pass
        return {"ok": True, "path": str(out_tif), "region": str(region), "boundary_source": str(match_source)}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "source": str(src_tif)}


def _clip_final_tifs_to_region(tifs: list[str], final_dir: Path, region: Any) -> dict[str, Any]:
    if not region:
        return {"ok": False, "skipped": True, "reason": "未指定区域"}
    clipped: list[str] = []
    attempts: list[dict[str, Any]] = []
    clip_dir = final_dir / "区域裁剪结果"
    for item in tifs:
        src = Path(str(item))
        if not src.exists() or src.suffix.lower() not in RASTER_EXTS:
            continue
        out = clip_dir / f"{_safe_name(src.stem)}_{_safe_name(region)}_clip{src.suffix}"
        res = _clip_tif_to_region(src, region, out)
        attempts.append(res)
        if res.get("ok") and res.get("path"):
            clipped.append(str(res.get("path")))
            if _is_truth(os.getenv("TPDC_DELETE_UNCLIPPED_TIF_AFTER_REGION_CLIP", "1"), "1"):
                try:
                    # Only remove intermediate TIFs in final_dir; never remove raw source NC/HDF.
                    if final_dir in src.parents and clip_dir not in src.parents:
                        src.unlink(missing_ok=True)
                except Exception:
                    pass
    return {"ok": bool(clipped), "clipped_tifs": clipped, "attempts": attempts}


def _is_truth(value: Any, default: str = "0") -> bool:
    text = str(value if value is not None else default).strip().lower()
    return text in {"1", "true", "yes", "y", "on", "启用", "是"}


def _delete_unselected_download_data(candidates: list[Path], selected: list[Path], final_dir: Path) -> list[str]:
    if not _is_truth(os.getenv("TPDC_DELETE_UNSELECTED_RAW_DATA", "1"), "1"):
        return []
    keep = set()
    for p in selected:
        try:
            keep.add(str(p.resolve()))
        except Exception:
            keep.add(str(p))
    deleted: list[str] = []
    for p in candidates:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key in keep or final_dir in p.parents:
            continue
        if p.exists() and p.is_file() and p.suffix.lower() in (RASTER_EXTS | MULTIDIM_EXTS | ARCHIVE_EXTS | {".grib", ".grb", ".grb2", ".gz", ".tar"}):
            try:
                p.unlink()
                deleted.append(str(p))
            except Exception as exc:
                deleted.append(f"删除失败：{p} -> {exc}")
    return deleted

def finalize_tpdc_download_after_ftp(
    manifest_path: str | Path | None,
    *,
    mapping_intent: bool | None = None,
    move_raw_to_final: bool = True,
) -> dict[str, Any]:
    """Clean and postprocess a completed TPDC FTP download.

    User-facing rule:
    - The raw FTP working area may contain logs/tickets/progress JSON.
    - The final data folder contains only data deliverables: target-year NC/HDF
      originals and generated GeoTIFF rasters.
    - If the task is only a download task, we do not start modeling. If the task
      is part of a mapping conversation, the returned GeoTIFFs can be registered
      as covariates by the Dash session.
    """
    manifest = _load_json(manifest_path)
    if not manifest:
        return {"ok": False, "error": "未找到 download_manifest.json。"}
    if manifest.get("tpdc_finalized") and manifest.get("final_data_dir"):
        return {"ok": True, "already_finalized": True, "manifest": manifest, "final_data_dir": manifest.get("final_data_dir"), "derived_tifs": manifest.get("derived_tifs") or []}

    requested_year = manifest.get("year") or manifest.get("requested_year")
    try:
        requested_year = int(requested_year) if requested_year else None
    except Exception:
        requested_year = None
    data_name = manifest.get("data_name") or "TPDC下载数据"
    target_region = manifest.get("region") or manifest.get("requested_region") or (manifest.get("request") or {}).get("region")
    base_dir = Path(str(manifest.get("local_save_dir") or (Path(str(manifest_path)).parent.parent if manifest_path else ".")))
    status = {}
    for status_path in [base_dir / "ftp_download_status.json", Path(str(manifest.get("ftp_download_status_path") or ""))]:
        try:
            if status_path and status_path.exists():
                status = _load_json(status_path)
                break
        except Exception:
            pass
    roots = []
    if status.get("download_root"):
        roots.append(Path(str(status.get("download_root"))))
    roots.append(base_dir)
    # De-duplicate roots.
    uniq_roots: list[Path] = []
    seen = set()
    for r in roots:
        try:
            key = str(r.resolve())
        except Exception:
            key = str(r)
        if key not in seen and r.exists():
            uniq_roots.append(r)
            seen.add(key)

    candidates: list[Path] = []
    for r in uniq_roots:
        candidates.extend(_download_data_candidates(r))
    # Drop files already in the final directory from a previous partial run.
    final_dir = base_dir / "下载结果"
    candidates = [p for p in candidates if final_dir not in p.parents]
    selected = _files_matching_year(candidates, requested_year)

    final_dir.mkdir(parents=True, exist_ok=True)
    deleted_unselected = _delete_unselected_download_data(candidates, selected, final_dir)
    moved_files: list[str] = []
    extracted_files: list[Path] = []
    for src in selected:
        try:
            if src.suffix.lower() == ".zip":
                extracted = _extract_zip(src, final_dir / f"解压_{_safe_name(src.stem)}")
                extracted_files.extend([p for p in extracted if p.suffix.lower() in (RASTER_EXTS | MULTIDIM_EXTS)])
                # Keep the archive out of the final deliverable folder unless it
                # is the only data representation.
                continue
            dst = _move_or_copy_data_file(src, final_dir, move=move_raw_to_final)
            moved_files.append(str(dst))
        except Exception as exc:
            moved_files.append(f"移动/复制失败：{src} -> {exc}")
    for src in extracted_files:
        try:
            if src.exists():
                dst = _move_or_copy_data_file(src, final_dir, move=True)
                moved_files.append(str(dst))
        except Exception as exc:
            moved_files.append(f"解压文件整理失败：{src} -> {exc}")

    final_files = _download_data_candidates(final_dir)
    derived_tifs: list[str] = []
    conversion_attempts: list[dict[str, Any]] = []
    for src in list(final_files):
        if src.suffix.lower() in MULTIDIM_EXTS:
            res = convert_multidimensional_to_tif(src, final_dir, requested_year, str(data_name))
            conversion_attempts.append({"source": str(src), **(res or {})})
            if res and res.get("ok") and res.get("path"):
                derived_tifs.append(str(res.get("path")))
        elif src.suffix.lower() in RASTER_EXTS:
            derived_tifs.append(str(src))

    region_clip = _clip_final_tifs_to_region(derived_tifs, final_dir, target_region) if target_region else {"ok": False, "skipped": True, "reason": "未指定区域"}
    if region_clip.get("ok") and region_clip.get("clipped_tifs"):
        derived_tifs = list(region_clip.get("clipped_tifs") or [])

    workflow_mode = _infer_mode_from_manifest(manifest, mapping_intent)
    manifest.update({
        "tpdc_finalized": True,
        "workflow_mode": workflow_mode,
        "final_data_dir": str(final_dir),
        "final_data_files": [str(p) for p in _download_data_candidates(final_dir)],
        "final_raw_files": moved_files,
        "derived_tifs": derived_tifs,
        "target_region": str(target_region or ""),
        "target_year": requested_year,
        "region_clip": region_clip,
        "deleted_unselected_data_files": deleted_unselected,
        "download_postprocess_status": "ready_for_mapping" if workflow_mode == "mapping" and derived_tifs else "download_only_ready" if workflow_mode == "download_only" else "postprocess_incomplete",
        "download_postprocess_attempts": conversion_attempts,
        "download_postprocess_note": "最终数据文件夹仅保留目标年份、目标区域裁剪后的可交付数据；FTP账号、日志、进度文件保留在任务根目录/元数据目录。",
    })
    if manifest_path:
        _save_json(manifest_path, manifest)
    return {
        "ok": bool(final_files or derived_tifs),
        "workflow_mode": workflow_mode,
        "final_data_dir": str(final_dir),
        "final_files": [str(p) for p in _download_data_candidates(final_dir)],
        "derived_tifs": derived_tifs,
        "region_clip": region_clip,
        "deleted_unselected_data_files": deleted_unselected,
        "conversion_attempts": conversion_attempts,
        "manifest": manifest,
    }
