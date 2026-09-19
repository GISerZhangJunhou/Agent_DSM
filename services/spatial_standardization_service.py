from __future__ import annotations

"""AI-callable spatial standardization tools.

This service standardizes uploaded/downloaded raster covariates to a common CRS,
resolution, extent and pixel-center grid.  It is intentionally task-oriented:
Dash/Agent calls one function and receives a fact-grounded report that can be
shown in chat history.
"""

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import rasterio  # type: ignore
    from rasterio.enums import Resampling  # type: ignore
    from rasterio.transform import from_origin  # type: ignore
    from rasterio.warp import calculate_default_transform, reproject, transform_bounds  # type: ignore
except Exception:  # pragma: no cover
    rasterio = None  # type: ignore
    Resampling = None  # type: ignore
    from_origin = None  # type: ignore
    calculate_default_transform = None  # type: ignore
    reproject = None  # type: ignore
    transform_bounds = None  # type: ignore

try:
    from pyproj import CRS  # type: ignore
except Exception:  # pragma: no cover
    CRS = None  # type: ignore

from config.settings import BASE_DIR

RASTER_EXTS = {".tif", ".tiff", ".img", ".vrt"}

CHINA_ALBERS_CGCS2000_PROJ4 = (
    "+proj=aea +lat_0=0 +lon_0=105 +lat_1=25 +lat_2=47 "
    "+x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"
)
KRASOVSKY_ALBERS_PROJ4 = (
    "+proj=aea +lat_0=0 +lon_0=105 +lat_1=25 +lat_2=47 "
    "+x_0=0 +y_0=0 +ellps=krass +units=m +no_defs"
)

CRS_ALIASES: dict[str, str] = {
    "wgs84": "EPSG:4326",
    "wgs 84": "EPSG:4326",
    "epsg4326": "EPSG:4326",
    "经纬度": "EPSG:4326",
    "cgcs2000": "EPSG:4490",
    "cgcs 2000": "EPSG:4490",
    "国家2000": "EPSG:4490",
    "国家大地2000": "EPSG:4490",
    "2000国家大地坐标系": "EPSG:4490",
    "web mercator": "EPSG:3857",
    "web墨卡托": "EPSG:3857",
    "墨卡托": "EPSG:3857",
    "北京54": "EPSG:4214",
    "beijing54": "EPSG:4214",
    "西安80": "EPSG:4610",
    "xian80": "EPSG:4610",
    "cgcs2000 albers": CHINA_ALBERS_CGCS2000_PROJ4,
    "cgcs 2000 albers": CHINA_ALBERS_CGCS2000_PROJ4,
    "cgcs2000阿尔伯斯": CHINA_ALBERS_CGCS2000_PROJ4,
    "cgcs2000 albers等面积": CHINA_ALBERS_CGCS2000_PROJ4,
    "中国阿尔伯斯": CHINA_ALBERS_CGCS2000_PROJ4,
    "中国albers": CHINA_ALBERS_CGCS2000_PROJ4,
    "阿尔伯斯等面积": CHINA_ALBERS_CGCS2000_PROJ4,
    "albers": CHINA_ALBERS_CGCS2000_PROJ4,
    "克拉索夫斯基阿尔伯斯": KRASOVSKY_ALBERS_PROJ4,
    "krasovsky albers": KRASOVSKY_ALBERS_PROJ4,
}

CATEGORICAL_HINTS = [
    "lulc", "lucc", "clcd", "land", "cover", "土地", "利用", "覆盖", "分类", "class",
    "type", "soiltype", "母质", "地类", "管理分区",
]


def _safe_name(name: str, max_len: int = 140) -> str:
    name = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", str(name or "layer"))
    return (name.strip("._-") or "layer")[:max_len]


def _is_raster_item(item: dict[str, Any]) -> bool:
    path = Path(str(item.get("path") or ""))
    if path.suffix.lower() in RASTER_EXTS:
        return True
    validation = item.get("validation") or {}
    return str(validation.get("file_type") or "").lower() == "raster"


def _raster_items(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for item in files or []:
        path = Path(str(item.get("path") or ""))
        if _is_raster_item(item) and path.exists():
            out.append(item)
    return out


def _find_reference_raster(files: list[dict[str, Any]], ref_name: str | None = None) -> dict[str, Any] | None:
    rasters = _raster_items(files)
    if not rasters:
        return None
    ref = str(ref_name or "").strip().lower()
    if ref:
        for item in rasters:
            name = str(item.get("name") or Path(str(item.get("path") or "")).name).lower()
            path = str(item.get("path") or "").lower()
            if ref in name or ref in path:
                return item
    return rasters[0]


def parse_resolution_meters(text: str) -> float | None:
    t = str(text or "")
    # Prefer phrases near 分辨率/像元大小.  Then fall back to any explicit m/km.
    patterns = [
        r"(?:分辨率|像元大小|空间分辨率|resolution)[^0-9]{0,12}(\d+(?:\.\d+)?)\s*(公里|千米|km|KM|Km)",
        r"(?:分辨率|像元大小|空间分辨率|resolution)[^0-9]{0,12}(\d+(?:\.\d+)?)\s*(米|m|M)",
        r"(\d+(?:\.\d+)?)\s*(公里|千米|km|KM|Km)",
        r"(\d+(?:\.\d+)?)\s*(米|m|M)",
    ]
    for pat in patterns:
        m = re.search(pat, t, flags=re.I)
        if not m:
            continue
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit in {"公里", "千米", "km"}:
            val *= 1000.0
        if 0 < val <= 1000000:
            return val
    return None


def parse_reference_name(text: str) -> str | None:
    t = str(text or "")
    for pat in [
        r"以\s*([^，。；;\n]+?\.(?:tif|tiff|img|vrt))\s*为基准",
        r"对齐到\s*([^，。；;\n]+?\.(?:tif|tiff|img|vrt))",
        r"参照\s*([^，。；;\n]+?\.(?:tif|tiff|img|vrt))",
        r"基准(?:图层|栅格|文件)?\s*(?:为|是|:|：)?\s*([^，。；;\n]+?\.(?:tif|tiff|img|vrt))",
    ]:
        m = re.search(pat, t, flags=re.I)
        if m:
            return Path(m.group(1).strip().strip('"\'“”‘’')).name
    # common short names
    for k in ["DEM", "BD", "CEC", "pH", "LULC", "CLCD"]:
        if re.search(r"以\s*" + re.escape(k) + r"\s*为基准", t, flags=re.I):
            return k
    return None


def parse_crs_from_text(text: str, fallback_crs: Any = None) -> tuple[Any, str, str]:
    """Return (crs, label, note). crs may be None if nothing requested."""
    if CRS is None:
        return None, "", "pyproj 未安装，无法解析坐标系。"
    t = str(text or "").strip()
    # EPSG:xxxx / EPSG xxxx / 只写 4490 is intentionally not accepted unless near 坐标系.
    m = re.search(r"EPSG\s*[:：]?\s*(\d{4,6})", t, flags=re.I)
    if m:
        code = f"EPSG:{m.group(1)}"
        return CRS.from_user_input(code), code, "按用户给出的 EPSG 代码解析。"
    m = re.search(r"(?:坐标系|投影|crs)[^0-9A-Za-z]{0,10}(\d{4,6})", t, flags=re.I)
    if m:
        code = f"EPSG:{m.group(1)}"
        return CRS.from_user_input(code), code, "按用户给出的 EPSG 代码解析。"
    compact = re.sub(r"[\s_\-]+", "", t.lower())
    # More specific aliases first.
    for alias, value in sorted(CRS_ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True):
        alias_compact = re.sub(r"[\s_\-]+", "", alias.lower())
        if alias_compact and alias_compact in compact:
            crs = CRS.from_user_input(value)
            return crs, alias, "按常用坐标系名称/别名解析。"
    if fallback_crs is not None:
        return CRS.from_user_input(fallback_crs), "参考图层坐标系", "用户未指定坐标系，沿用参考图层坐标系。"
    return None, "", "用户未指定坐标系。"


def _resampling_for_item(item: dict[str, Any]) -> Any:
    name = str(item.get("name") or Path(str(item.get("path") or "")).name).lower()
    if any(h.lower() in name for h in CATEGORICAL_HINTS):
        return Resampling.nearest
    validation = item.get("validation") or {}
    dtype = str(validation.get("dtype") or "").lower()
    if dtype.startswith(("uint", "int")) and any(x in name for x in ["class", "lulc", "clcd", "lucc"]):
        return Resampling.nearest
    return Resampling.bilinear


def _target_grid_from_reference(ref_path: Path, target_crs: Any, target_res_m: float | None) -> tuple[Any, Any, int, int, tuple[float, float, float, float]]:
    with rasterio.open(ref_path) as ref:
        ref_crs = ref.crs
        ref_bounds = ref.bounds
        if target_crs is None:
            target_crs = ref_crs
        if str(target_crs) == str(ref_crs):
            bounds = (ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top)
        else:
            bounds = transform_bounds(ref_crs, target_crs, ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top, densify_pts=21)
        # If no explicit resolution, keep reference resolution after possible CRS transform.
        if target_res_m is None:
            if str(target_crs) == str(ref_crs):
                xres = abs(ref.transform.a)
                yres = abs(ref.transform.e)
            else:
                transform, width, height = calculate_default_transform(ref_crs, target_crs, ref.width, ref.height, *ref.bounds)
                xres = abs(transform.a)
                yres = abs(transform.e)
        else:
            xres = yres = float(target_res_m)
        left, bottom, right, top = bounds
        width = max(1, int(math.ceil((right - left) / xres)))
        height = max(1, int(math.ceil((top - bottom) / yres)))
        transform = from_origin(left, top, xres, yres)
        return target_crs, transform, width, height, bounds


def _standardize_one(item: dict[str, Any], target_crs: Any, target_transform: Any, width: int, height: int, out_dir: Path) -> dict[str, Any]:
    src_path = Path(str(item.get("path") or ""))
    name = str(item.get("name") or src_path.name)
    out_path = out_dir / (_safe_name(src_path.stem) + "_标准化.tif")
    resampling = _resampling_for_item(item)
    with rasterio.open(src_path) as src:
        nodata = src.nodata
        dst_nodata = -9999.0
        dst = np.full((height, width), dst_nodata, dtype="float32")
        src_nodata = nodata
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=target_transform,
            dst_crs=target_crs,
            dst_nodata=dst_nodata,
            resampling=resampling,
        )
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            crs=target_crs,
            transform=target_transform,
            dtype="float32",
            nodata=dst_nodata,
            compress="deflate",
            tiled=False,
            BIGTIFF="IF_SAFER",
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst_ds:
            dst_ds.write(dst, 1)
    return {
        "name": name,
        "source_path": str(src_path),
        "standardized_path": str(out_path),
        "resampling": "nearest" if resampling == Resampling.nearest else "bilinear",
        "ok": True,
    }


def standardize_session_rasters(
    session_id: str,
    uploaded_files: list[dict[str, Any]],
    instruction_text: str = "",
    target_crs_text: str | None = None,
    target_resolution_m: float | None = None,
    reference_layer: str | None = None,
) -> dict[str, Any]:
    """Standardize uploaded rasters and return updated file records.

    The function does not require a UI.  It is safe for Agent/Dash calls.
    """
    if rasterio is None or CRS is None:
        return {"ok": False, "error": "缺少 rasterio 或 pyproj，无法执行空间标准化。"}
    files = list(uploaded_files or [])
    rasters = _raster_items(files)
    if not rasters:
        return {"ok": False, "error": "当前会话没有可标准化的栅格协变量。"}
    ref_name = reference_layer or parse_reference_name(instruction_text)
    ref_item = _find_reference_raster(files, ref_name)
    if not ref_item:
        return {"ok": False, "error": "没有找到参考栅格，无法建立目标网格。"}
    ref_path = Path(str(ref_item.get("path") or ""))
    with rasterio.open(ref_path) as ref:
        fallback_crs = ref.crs
    crs_text = target_crs_text or instruction_text
    target_crs, crs_label, crs_note = parse_crs_from_text(crs_text, fallback_crs=fallback_crs)
    res_m = target_resolution_m if target_resolution_m is not None else parse_resolution_meters(instruction_text)
    if target_crs is None:
        target_crs = fallback_crs
    # Warn if user asks for meter resolution in geographic CRS.
    target_crs_obj = CRS.from_user_input(target_crs)
    degree_warning = ""
    if res_m and target_crs_obj.is_geographic:
        degree_warning = "目标坐标系是经纬度坐标系，单位为度；系统不会把米直接当作度。已改为沿用参考图层分辨率。建议使用投影坐标系执行米级标准化。"
        res_m = None
    target_crs, transform, width, height, bounds = _target_grid_from_reference(ref_path, target_crs, res_m)
    out_dir = Path(BASE_DIR) / "runs" / "空间标准化" / str(session_id) / time.strftime("%Y%m%d_%H%M%S")
    results: list[dict[str, Any]] = []
    updated_files: list[dict[str, Any]] = []
    by_source_path: dict[str, dict[str, Any]] = {}
    for item in files:
        p = Path(str(item.get("path") or ""))
        if not _is_raster_item(item) or not p.exists():
            updated_files.append(item)
            continue
        try:
            rec = _standardize_one(item, target_crs, transform, width, height, out_dir)
            results.append(rec)
            new_item = dict(item)
            new_item.setdefault("original_path", str(item.get("path") or ""))
            new_item["path"] = rec["standardized_path"]
            new_item["standardized"] = True
            new_item["standardization"] = {
                "target_crs": target_crs_obj.to_string(),
                "target_crs_wkt": CRS.from_user_input(target_crs).to_wkt(),
                "target_resolution_m": res_m,
                "reference_layer": str(ref_item.get("name") or Path(str(ref_item.get("path") or "")).name),
                "target_width": width,
                "target_height": height,
                "target_bounds": list(bounds),
                "source_path": rec["source_path"],
                "resampling": rec["resampling"],
            }
            # refresh validation metadata in a lightweight way
            try:
                with rasterio.open(rec["standardized_path"]) as ds:
                    validation = dict(new_item.get("validation") or {})
                    validation.update({
                        "ok": True,
                        "file_type": "raster",
                        "path": rec["standardized_path"],
                        "width": ds.width,
                        "height": ds.height,
                        "crs": str(ds.crs),
                        "bounds": list(ds.bounds),
                        "resolution": [abs(ds.transform.a), abs(ds.transform.e)],
                        "nodata": ds.nodata,
                        "stage": "standardized",
                        "message": "已完成空间标准化，可用于后续分析。",
                    })
                    new_item["validation"] = validation
            except Exception:
                pass
            updated_files.append(new_item)
            by_source_path[str(p)] = new_item
        except Exception as exc:
            results.append({"name": item.get("name") or p.name, "source_path": str(p), "ok": False, "error": str(exc)})
            updated_files.append(item)
    ok_count = sum(1 for r in results if r.get("ok"))
    fail_count = len(results) - ok_count
    manifest = {
        "ok": ok_count > 0 and fail_count == 0,
        "partial_ok": ok_count > 0,
        "session_id": session_id,
        "reference_layer": str(ref_item.get("name") or ref_path.name),
        "target_crs_label": crs_label,
        "target_crs": CRS.from_user_input(target_crs).to_string(),
        "target_crs_wkt": CRS.from_user_input(target_crs).to_wkt(),
        "target_resolution_m": res_m,
        "target_width": width,
        "target_height": height,
        "target_bounds": list(bounds),
        "output_dir": str(out_dir),
        "results": results,
        "warnings": [x for x in [crs_note, degree_warning] if x],
        "created_at": int(time.time()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "空间标准化报告.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    manifest["updated_files"] = updated_files
    return manifest


def build_standardization_chat_message(result: dict[str, Any]) -> str:
    if not result.get("partial_ok"):
        return "空间标准化未完成：" + str(result.get("error") or "没有成功处理的栅格。")
    ok_items = [r for r in result.get("results", []) if r.get("ok")]
    fail_items = [r for r in result.get("results", []) if not r.get("ok")]
    lines = [
        "已完成空间标准化处理。",
        f"参考图层：{result.get('reference_layer')}",
        f"目标坐标系：{result.get('target_crs_label') or result.get('target_crs')}",
        f"目标分辨率：{result.get('target_resolution_m') or '沿用参考图层'} 米",
        f"目标网格：{result.get('target_width')} × {result.get('target_height')}",
        f"成功：{len(ok_items)} 个；失败：{len(fail_items)} 个。",
    ]
    if ok_items:
        shown = "、".join(str(x.get("name")) for x in ok_items[:8])
        if len(ok_items) > 8:
            shown += f" 等 {len(ok_items)} 个"
        lines.append("已标准化图层：" + shown)
    if fail_items:
        lines.append("失败图层：" + "；".join(f"{x.get('name')}：{x.get('error')}" for x in fail_items[:5]))
    warns = [w for w in result.get("warnings", []) if w]
    if warns:
        lines.append("提示：" + "；".join(warns[:3]))
    if result.get("output_dir"):
        lines.append("标准化结果已保存到本机文件夹：" + str(result.get("output_dir")))
    return "\n".join(lines)
