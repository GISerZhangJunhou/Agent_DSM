from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

try:
    import ee
except Exception:  # pragma: no cover
    ee = None

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None

from services.domestic_download_service import parse_download_intent, raw_save_dir, manifest_path_for
from services.aoi_preflight_service import detect_requested_aoi_from_local_admin
from services.download_postprocess_service import process_download_manifest_for_map


def _init_ee() -> None:
    if ee is None:
        raise RuntimeError("缺少 earthengine-api，无法使用 GEE（Google Earth Engine）下载数据。")
    project = os.getenv("PRO_GEE_PROJECT") or os.getenv("GEE_PROJECT") or "fit-territory-472114-b0"
    ee.Initialize(project=project)


def _same_admin_name(a: str, b: str) -> bool:
    a = str(a or "").strip()
    b = str(b or "").strip()
    if not a or not b:
        return False
    def strip(x: str) -> str:
        for suf in ["特别行政区", "壮族自治区", "维吾尔自治区", "回族自治区", "自治区", "自治州", "地区", "省", "市", "县", "区", "旗"]:
            if x.endswith(suf) and len(x) > len(suf):
                return x[:-len(suf)]
        return x
    return a == b or strip(a) == strip(b) or a in b or b in a


def _aoi_bbox_from_local_admin(request_text: str) -> tuple[list[float], dict[str, Any]]:
    """Return [west, south, east, north] from local admin vector, or Chengdu fallback."""
    aoi = detect_requested_aoi_from_local_admin(request_text)
    if aoi.get("ok") and gpd is not None:
        path = aoi.get("path")
        field = aoi.get("field")
        region = aoi.get("region")
        try:
            gdf = gpd.read_file(path)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326", allow_override=True)
            else:
                gdf = gdf.to_crs("EPSG:4326")
            if field in gdf.columns:
                sel = gdf[gdf[field].astype(str).map(lambda x: _same_admin_name(x, region))]
            else:
                sel = gdf.iloc[[int(aoi.get("geometry_index", 0))]]
            if not sel.empty:
                minx, miny, maxx, maxy = sel.total_bounds.tolist()
                return [float(minx), float(miny), float(maxx), float(maxy)], {"source": "local_admin_vector", **aoi}
        except Exception as exc:
            return [103.15, 29.95, 104.95, 31.35], {"source": "fallback_chengdu_bbox", "aoi_error": str(exc), "aoi": aoi}
    # Conservative default for Chengdu workflows.
    return [103.15, 29.95, 104.95, 31.35], {"source": "fallback_chengdu_bbox", "aoi": aoi}


def _year_range(year: int | None) -> tuple[str, str]:
    y = int(year or 2020)
    return f"{y}-01-01", f"{y+1}-01-01"


def _build_image(intent, region_geom):
    data_name = str(intent.data_name or "DEM")
    year = intent.year or 2020
    start, end = _year_range(year)
    if data_name == "DEM":
        return ee.Image("USGS/SRTMGL1_003").select("elevation").rename("DEM"), {"gee_asset": "USGS/SRTMGL1_003", "temporal_type": "static"}
    if data_name in {"NDVI", "EVI"}:
        band = data_name
        img = (ee.ImageCollection("MODIS/061/MOD13Q1")
               .filterBounds(region_geom)
               .filterDate(start, end)
               .select(band)
               .mean()
               .multiply(0.0001)
               .rename(band))
        return img, {"gee_asset": "MODIS/061/MOD13Q1", "date_start": start, "date_end": end, "temporal_type": "dynamic"}
    if data_name == "NPP":
        img = (ee.ImageCollection("MODIS/061/MOD17A3HGF")
               .filterBounds(region_geom)
               .filterDate(start, end)
               .select("Npp")
               .mean()
               .multiply(0.0001)
               .rename("NPP"))
        return img, {"gee_asset": "MODIS/061/MOD17A3HGF", "date_start": start, "date_end": end, "temporal_type": "dynamic"}
    if data_name in {"降水", "气象"}:
        img = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
               .filterBounds(region_geom)
               .filterDate(start, end)
               .select("precipitation")
               .sum()
               .rename("precipitation"))
        return img, {"gee_asset": "UCSB-CHG/CHIRPS/DAILY", "date_start": start, "date_end": end, "temporal_type": "dynamic"}
    if data_name == "气温":
        img = (ee.ImageCollection("MODIS/061/MOD11A2")
               .filterBounds(region_geom)
               .filterDate(start, end)
               .select("LST_Day_1km")
               .mean()
               .multiply(0.02)
               .subtract(273.15)
               .rename("LST_C"))
        return img, {"gee_asset": "MODIS/061/MOD11A2", "date_start": start, "date_end": end, "temporal_type": "dynamic"}
    if data_name == "LULC":
        # 2020 uses WorldCover v100; 2021 uses v200. Other years fall back to 2020/2021 nearest.
        collection = "ESA/WorldCover/v100" if int(year) <= 2020 else "ESA/WorldCover/v200"
        img = ee.ImageCollection(collection).first().select("Map").rename("LULC")
        return img, {"gee_asset": collection, "date_start": start, "date_end": end, "temporal_type": "quasi_dynamic"}
    # Safe default: SRTM DEM, with clear manifest note. This prevents a blank task.
    return ee.Image("USGS/SRTMGL1_003").select("elevation").rename("DEM"), {"gee_asset": "USGS/SRTMGL1_003", "temporal_type": "static", "fallback_reason": f"GEE 下载暂未内置 {data_name}，已回退 DEM。"}


def run_gee_download_task(request_text: str, auth_user: dict[str, Any] | None = None) -> dict[str, Any]:
    intent = parse_download_intent(request_text)
    out_dir = raw_save_dir(intent)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path_for(intent)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "task_type": "download_only",
        "platform_scope": "gee",
        "platform_name": "GEE（Google Earth Engine）",
        "request_text": request_text,
        "data_name": intent.data_name,
        "region": intent.region,
        "year": intent.year,
        "temporal_type": intent.temporal_type,
        "local_save_dir": str(out_dir),
        "manifest_path": str(manifest_path),
        "ready_for_modeling": False,
        "created_at": int(time.time()),
    }
    try:
        _init_ee()
        bbox, aoi_meta = _aoi_bbox_from_local_admin(request_text)
        geom = ee.Geometry.Rectangle(bbox, proj="EPSG:4326", geodesic=False)
        image, image_meta = _build_image(intent, geom)
        scale = int(intent.requested_resolution_meters or os.getenv("GEE_DOWNLOAD_SCALE_M", "250"))
        filename_base = f"{intent.year or 'unknown'}_{intent.region}_{intent.data_name}_GEE"
        filename = re.sub(r"[\\/:*?\"<>|\s]+", "_", filename_base)[:100] + ".zip"
        url = image.clip(geom).getDownloadURL({
            "name": Path(filename).stem,
            "region": geom,
            "scale": scale,
            "crs": "EPSG:4326",
            "fileFormat": "GeoTIFF",
        })
        manifest.update({"status": "download_started", "gee_download_url_generated": True, "aoi": aoi_meta, "bbox": bbox, "scale_m": scale, **image_meta})
        _save_manifest(manifest_path, manifest)
        local_zip = out_dir / filename
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(local_zip, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        manifest.update({
            "status": "download_verified",
            "ready_for_modeling": True,
            "download_mode": "gee_getDownloadURL",
            "validated_files": [{"path": str(local_zip), "size": local_zip.stat().st_size, "validation": "gee_geotiff_zip"}],
        })
        _save_manifest(manifest_path, manifest)
        post = process_download_manifest_for_map(manifest_path)
        manifest = _load_manifest(manifest_path)
        return {"ok": True, "manifest": manifest, "manifest_path": str(manifest_path), "postprocess": post, "display_tif": post.get("display_tif") if post.get("ok") else None}
    except Exception as exc:
        manifest.update({"status": "download_failed", "error": str(exc), "ready_for_modeling": False})
        _save_manifest(manifest_path, manifest)
        return {"ok": False, "error": str(exc), "manifest": manifest, "manifest_path": str(manifest_path)}


def _save_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
