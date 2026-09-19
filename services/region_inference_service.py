# -*- coding: utf-8 -*-
"""Region inference utilities for uploaded lon/lat samples.

V71 scope:
- Infer at least province-level region from uploaded CSV coordinates.
- Infer selected city-level regions when simple bbox rules are reliable enough for workflow routing.
- Prefer official admin boundary overlay if user later provides vectors through env vars:
  PRO_ADMIN_PROVINCE_VECTOR / PRO_ADMIN_CITY_VECTOR.

This module is deliberately conservative: it does not claim legal/official boundaries unless a
real admin vector overlay is configured. Bbox inference is for workflow routing and user prompts.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


@dataclass
class RegionTextMatch:
    region: str | None = None
    level: str | None = None  # country | province | city
    province: str | None = None
    city: str | None = None
    source: str | None = None
    matched_text: str | None = None


# Bboxes are approximate and only used when no official admin boundary vector is configured.
# lon_min, lat_min, lon_max, lat_max.
PROVINCE_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "北京市": (115.4, 39.4, 117.6, 41.1),
    "天津市": (116.7, 38.5, 118.1, 40.3),
    "河北省": (113.4, 36.0, 119.9, 42.6),
    "山西省": (110.2, 34.5, 114.6, 40.8),
    "内蒙古自治区": (97.0, 37.3, 126.1, 53.6),
    "辽宁省": (118.8, 38.7, 125.8, 43.5),
    "吉林省": (121.6, 40.8, 131.3, 46.4),
    "黑龙江省": (121.1, 43.4, 135.1, 53.6),
    "上海市": (120.8, 30.6, 122.2, 31.9),
    "江苏省": (116.3, 30.7, 121.9, 35.2),
    "浙江省": (118.0, 27.0, 123.0, 31.4),
    "安徽省": (114.8, 29.4, 119.7, 34.7),
    "福建省": (115.8, 23.5, 120.7, 28.4),
    "江西省": (113.5, 24.4, 118.5, 30.1),
    "山东省": (114.8, 34.3, 122.8, 38.4),
    "河南省": (110.2, 31.3, 116.7, 36.4),
    "湖北省": (108.3, 29.0, 116.2, 33.3),
    "湖南省": (108.7, 24.6, 114.3, 30.1),
    "广东省": (109.6, 20.1, 117.3, 25.6),
    "广西壮族自治区": (104.4, 20.8, 112.1, 26.4),
    "海南省": (108.5, 18.0, 111.2, 20.3),
    "重庆市": (105.2, 28.1, 110.2, 32.4),
    "四川省": (97.3, 26.0, 108.6, 34.4),
    "贵州省": (103.5, 24.5, 109.6, 29.3),
    "云南省": (97.5, 21.1, 106.2, 29.3),
    "西藏自治区": (78.4, 26.8, 99.1, 36.5),
    "陕西省": (105.5, 31.7, 111.3, 39.6),
    "甘肃省": (92.3, 32.6, 108.7, 42.8),
    "青海省": (89.4, 31.6, 103.1, 39.2),
    "宁夏回族自治区": (104.1, 35.2, 107.7, 39.4),
    "新疆维吾尔自治区": (73.3, 34.2, 96.4, 49.2),
    "台湾省": (119.2, 21.8, 122.1, 25.4),
    "香港特别行政区": (113.8, 22.1, 114.4, 22.6),
    "澳门特别行政区": (113.5, 22.1, 113.7, 22.3),
}

# Conservative city bboxes for commonly tested areas. If a city is not listed, province inference still works.
CITY_BBOXES: dict[str, dict[str, Any]] = {
    "成都市": {"province": "四川省", "bbox": (102.8, 29.9, 105.2, 31.5), "aliases": ["成都", "成都市"]},
    "绵阳市": {"province": "四川省", "bbox": (103.8, 30.7, 105.7, 33.1), "aliases": ["绵阳", "绵阳市"]},
    "德阳市": {"province": "四川省", "bbox": (103.7, 30.5, 105.1, 31.9), "aliases": ["德阳", "德阳市"]},
    "眉山市": {"province": "四川省", "bbox": (102.8, 29.4, 104.5, 30.4), "aliases": ["眉山", "眉山市"]},
    "雅安市": {"province": "四川省", "bbox": (102.0, 28.8, 103.6, 30.9), "aliases": ["雅安", "雅安市"]},
    "重庆市": {"province": "重庆市", "bbox": (105.2, 28.1, 110.2, 32.4), "aliases": ["重庆", "重庆市"]},
    "北京市": {"province": "北京市", "bbox": (115.4, 39.4, 117.6, 41.1), "aliases": ["北京", "北京市"]},
    "上海市": {"province": "上海市", "bbox": (120.8, 30.6, 122.2, 31.9), "aliases": ["上海", "上海市"]},
    "广州市": {"province": "广东省", "bbox": (112.9, 22.5, 114.1, 23.9), "aliases": ["广州", "广州市"]},
    "深圳市": {"province": "广东省", "bbox": (113.7, 22.4, 114.7, 22.9), "aliases": ["深圳", "深圳市"]},
    "武汉市": {"province": "湖北省", "bbox": (113.6, 29.9, 115.1, 31.4), "aliases": ["武汉", "武汉市"]},
    "西安市": {"province": "陕西省", "bbox": (107.6, 33.6, 109.8, 34.8), "aliases": ["西安", "西安市"]},
    "昆明市": {"province": "云南省", "bbox": (102.1, 24.3, 103.7, 26.5), "aliases": ["昆明", "昆明市"]},
    "贵阳市": {"province": "贵州省", "bbox": (106.0, 26.0, 107.3, 27.2), "aliases": ["贵阳", "贵阳市"]},
}

PROVINCE_ALIASES: dict[str, list[str]] = {
    "北京市": ["北京", "北京市"], "天津市": ["天津", "天津市"], "河北省": ["河北", "河北省"],
    "山西省": ["山西", "山西省"], "内蒙古自治区": ["内蒙古", "内蒙古自治区"], "辽宁省": ["辽宁", "辽宁省"],
    "吉林省": ["吉林", "吉林省"], "黑龙江省": ["黑龙江", "黑龙江省"], "上海市": ["上海", "上海市"],
    "江苏省": ["江苏", "江苏省"], "浙江省": ["浙江", "浙江省"], "安徽省": ["安徽", "安徽省"],
    "福建省": ["福建", "福建省"], "江西省": ["江西", "江西省"], "山东省": ["山东", "山东省"],
    "河南省": ["河南", "河南省"], "湖北省": ["湖北", "湖北省"], "湖南省": ["湖南", "湖南省"],
    "广东省": ["广东", "广东省"], "广西壮族自治区": ["广西", "广西壮族自治区"], "海南省": ["海南", "海南省"],
    "重庆市": ["重庆", "重庆市"], "四川省": ["四川", "四川省"], "贵州省": ["贵州", "贵州省"],
    "云南省": ["云南", "云南省"], "西藏自治区": ["西藏", "西藏自治区"], "陕西省": ["陕西", "陕西省"],
    "甘肃省": ["甘肃", "甘肃省"], "青海省": ["青海", "青海省"], "宁夏回族自治区": ["宁夏", "宁夏回族自治区"],
    "新疆维吾尔自治区": ["新疆", "新疆维吾尔自治区"], "台湾省": ["台湾", "台湾省"],
    "香港特别行政区": ["香港", "香港特别行政区"], "澳门特别行政区": ["澳门", "澳门特别行政区"],
}


def _inside(lon: float, lat: float, bbox: tuple[float, float, float, float]) -> bool:
    return bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]


def detect_region_from_text(text: str) -> RegionTextMatch:
    text = text or ""
    if any(k in text for k in ["全国", "中国", "全国尺度"]):
        return RegionTextMatch(region="全国", level="country", province=None, city=None, source="request_text", matched_text="全国/中国")
    # City first because municipalities/provinces may share substrings.
    for city, meta in CITY_BBOXES.items():
        for alias in meta.get("aliases", []):
            if alias and alias in text:
                return RegionTextMatch(region=city, level="city", province=meta.get("province"), city=city, source="request_text", matched_text=alias)
    for prov, aliases in PROVINCE_ALIASES.items():
        for alias in aliases:
            if alias and alias in text:
                level = "city" if prov in {"北京市", "天津市", "上海市", "重庆市"} else "province"
                return RegionTextMatch(region=prov, level=level, province=prov, city=prov if level == "city" else None, source="request_text", matched_text=alias)
    return RegionTextMatch()


def _infer_by_bbox(samples: Any) -> dict[str, Any]:
    if pd is None:
        return {"ok": False, "method": "bbox", "message": "pandas不可用，无法区域推断。"}
    if samples is None or len(samples) == 0:
        return {"ok": False, "method": "bbox", "message": "样点为空，无法区域推断。"}
    work = samples[["lon", "lat"]].copy()
    work["lon"] = pd.to_numeric(work["lon"], errors="coerce")
    work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
    work = work.dropna()
    n = int(len(work))
    if n == 0:
        return {"ok": False, "method": "bbox", "message": "样点坐标无有效数值，无法区域推断。"}

    province_hits: list[str] = []
    city_hits: list[str] = []
    for lon, lat in zip(work["lon"].tolist(), work["lat"].tolist()):
        # city first for more specific result.
        city_hit = None
        for city, meta in CITY_BBOXES.items():
            if _inside(float(lon), float(lat), meta["bbox"]):
                city_hit = city
                break
        city_hits.append(city_hit or "未匹配")
        prov_hit = None
        # If city matched, use its province. This reduces bbox overlap ambiguity.
        if city_hit:
            prov_hit = CITY_BBOXES[city_hit]["province"]
        else:
            for prov, bbox in PROVINCE_BBOXES.items():
                if _inside(float(lon), float(lat), bbox):
                    prov_hit = prov
                    break
        province_hits.append(prov_hit or "未匹配")

    prov_counts = pd.Series(province_hits).value_counts().to_dict()
    city_counts = pd.Series(city_hits).value_counts().to_dict()
    prov_top = max(prov_counts.items(), key=lambda kv: kv[1]) if prov_counts else ("未匹配", 0)
    city_valid_counts = {k: v for k, v in city_counts.items() if k != "未匹配"}
    city_top = max(city_valid_counts.items(), key=lambda kv: kv[1]) if city_valid_counts else (None, 0)

    min_ratio = float(os.getenv("PRO_REGION_DOMINANCE_RATIO", "0.8"))
    city_ratio = float(city_top[1]) / n if city_top[0] else 0.0
    prov_ratio = float(prov_top[1]) / n if prov_top[0] and prov_top[0] != "未匹配" else 0.0

    selected_region = None
    selected_level = None
    selected_province = None
    selected_city = None
    selected_ratio = 0.0

    if city_top[0] and city_ratio >= min_ratio:
        selected_region = city_top[0]
        selected_level = "city"
        selected_city = city_top[0]
        selected_province = CITY_BBOXES[selected_city]["province"]
        selected_ratio = city_ratio
    elif prov_top[0] and prov_top[0] != "未匹配" and prov_ratio >= min_ratio:
        selected_region = prov_top[0]
        selected_level = "province"
        selected_province = prov_top[0]
        selected_ratio = prov_ratio
    elif prov_top[0] and prov_top[0] != "未匹配":
        selected_region = prov_top[0]
        selected_level = "province_weak"
        selected_province = prov_top[0]
        selected_ratio = prov_ratio

    return {
        "ok": bool(selected_region),
        "method": "bbox_builtin_v71",
        "sample_count": n,
        "region": selected_region,
        "level": selected_level,
        "province": selected_province,
        "city": selected_city,
        "dominance_ratio": selected_ratio,
        "province_counts": prov_counts,
        "city_counts": city_counts,
        "lon_min": float(work["lon"].min()),
        "lon_max": float(work["lon"].max()),
        "lat_min": float(work["lat"].min()),
        "lat_max": float(work["lat"].max()),
        "message": f"根据样点经纬度推断主要区域为 {selected_region}，占比 {selected_ratio:.3f}。" if selected_region else "未能根据内置bbox推断主要区域。",
        "warning": "bbox推断仅用于流程路由；若需正式行政边界，请配置PRO_ADMIN_PROVINCE_VECTOR/PRO_ADMIN_CITY_VECTOR。",
    }


def infer_region_from_samples(samples: Any) -> dict[str, Any]:
    """Infer sample region. Currently uses optional admin vector overlay if available; otherwise bbox.

    Returns a JSON-serializable dict.
    """
    # Reserved official vector overlay interface. Keep it soft-fail so app remains portable.
    city_vec = os.getenv("PRO_ADMIN_CITY_VECTOR", "").strip()
    prov_vec = os.getenv("PRO_ADMIN_PROVINCE_VECTOR", "").strip()
    if city_vec or prov_vec:
        try:
            return _infer_by_admin_vectors(samples, prov_vec=prov_vec, city_vec=city_vec)
        except Exception as exc:
            result = _infer_by_bbox(samples)
            result.setdefault("warnings", [])
            result["admin_vector_error"] = str(exc)
            result["warning"] = f"行政边界矢量叠加失败，已回退内置bbox推断：{exc}"
            return result
    return _infer_by_bbox(samples)


def _infer_by_admin_vectors(samples: Any, prov_vec: str = "", city_vec: str = "") -> dict[str, Any]:
    """Optional admin-boundary overlay. Requires geopandas/shapely and user-provided vectors.

    Expected env fields:
    - PRO_ADMIN_PROVINCE_NAME_FIELD, default: province/name/省/省名/NAME
    - PRO_ADMIN_CITY_NAME_FIELD, default: city/name/市/市名/NAME
    """
    try:
        import geopandas as gpd
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"geopandas不可用：{exc}")
    if pd is None:
        raise RuntimeError("pandas不可用")
    work = samples[["lon", "lat"]].copy()
    work["lon"] = pd.to_numeric(work["lon"], errors="coerce")
    work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
    work = work.dropna().reset_index(drop=True)
    if work.empty:
        raise RuntimeError("样点坐标无有效值")
    pts = gpd.GeoDataFrame(work, geometry=gpd.points_from_xy(work["lon"], work["lat"]), crs="EPSG:4326")

    def pick_name_field(gdf, env_key, candidates):
        env_val = os.getenv(env_key, "").strip()
        if env_val and env_val in gdf.columns:
            return env_val
        for c in candidates:
            if c in gdf.columns:
                return c
        return None

    out: dict[str, Any] = {"ok": False, "method": "admin_vector", "sample_count": int(len(pts))}
    if city_vec and Path(city_vec).exists():
        cities = gpd.read_file(city_vec).to_crs("EPSG:4326")
        name_field = pick_name_field(cities, "PRO_ADMIN_CITY_NAME_FIELD", ["city", "市", "市名", "NAME", "name", "地市", "地级市"])
        if name_field:
            join = gpd.sjoin(pts, cities[[name_field, "geometry"]], how="left", predicate="within")
            vc = join[name_field].fillna("未匹配").value_counts()
            top_name = str(vc.index[0])
            top_count = int(vc.iloc[0])
            ratio = top_count / max(len(join), 1)
            if top_name != "未匹配" and ratio >= float(os.getenv("PRO_REGION_DOMINANCE_RATIO", "0.8")):
                out.update({"ok": True, "region": top_name, "city": top_name, "level": "city", "dominance_ratio": ratio, "city_counts": vc.to_dict(), "message": f"根据行政区矢量叠加推断主要城市为 {top_name}，占比 {ratio:.3f}。"})
                return out
    if prov_vec and Path(prov_vec).exists():
        provs = gpd.read_file(prov_vec).to_crs("EPSG:4326")
        name_field = pick_name_field(provs, "PRO_ADMIN_PROVINCE_NAME_FIELD", ["province", "省", "省名", "NAME", "name", "省级"])
        if name_field:
            join = gpd.sjoin(pts, provs[[name_field, "geometry"]], how="left", predicate="within")
            vc = join[name_field].fillna("未匹配").value_counts()
            top_name = str(vc.index[0])
            top_count = int(vc.iloc[0])
            ratio = top_count / max(len(join), 1)
            if top_name != "未匹配":
                out.update({"ok": True, "region": top_name, "province": top_name, "level": "province", "dominance_ratio": ratio, "province_counts": vc.to_dict(), "message": f"根据行政区矢量叠加推断主要省份为 {top_name}，占比 {ratio:.3f}。"})
                return out
    raise RuntimeError("行政区矢量存在但未能完成有效区域推断，请检查字段名与坐标系。")


def region_conflict(request_region: str | None, inferred: dict[str, Any]) -> bool:
    if not request_region or not inferred or not inferred.get("region"):
        return False
    inf_region = str(inferred.get("region"))
    inf_prov = str(inferred.get("province") or "")
    inf_city = str(inferred.get("city") or "")
    # user region is compatible with inferred city/province.
    if request_region == inf_region or request_region == inf_prov or request_region == inf_city:
        return False
    if request_region in inf_region or inf_region in request_region:
        return False
    return True


def apply_region_to_target(target: dict[str, Any], region_name: str, level: str | None = None, source: str | None = None, inference: dict[str, Any] | None = None) -> dict[str, Any]:
    target["region"] = region_name
    target["region_source"] = source or target.get("region_source") or "unknown"
    if inference is not None:
        target["region_inference"] = inference

    # Preserve the administrative level explicitly.  V168 resolution policy needs to
    # distinguish county/district, prefecture-level city, province and country; the
    # previous boolean is_city flag was not expressive enough.
    level_norm = str(level or "").strip().lower()
    region_s = str(region_name or "").strip()
    if not level_norm:
        if region_s in {"全国", "中国"}:
            level_norm = "country"
        elif any(k in region_s for k in ["县", "区", "旗"]):
            level_norm = "county"
        elif region_s.endswith("市") and region_s not in {"北京市", "天津市", "上海市", "重庆市"}:
            level_norm = "city"
        elif region_s in {"北京市", "天津市", "上海市", "重庆市"}:
            # Keep compatibility with older city CRS handling for municipalities.
            level_norm = "city"
        else:
            level_norm = "province"
    target["region_level"] = level_norm
    target["aoi_level"] = level_norm

    is_city = False
    if level_norm in {"city", "county"}:
        is_city = True
    elif any(k in region_s for k in ["市", "县", "区"]):
        # Provinces ending in 自治区/特别行政区 are not treated as city.
        is_city = not any(k in region_s for k in ["自治区", "特别行政区"])
    target["is_city"] = bool(is_city)

    # IMPORTANT: region recognition must not silently force 250 m/1000 m resolution.
    # V168 applies regional defaults only inside the formal auto-resolution policy,
    # and only when all user-provided raster/CSV scale evidence is 30 m or finer.
    # Keep user/coarsest/pending-auto resolution intact; only apply legacy defaults
    # when explicitly requested through PRO_REGION_DEFAULT_RESOLUTION=1.
    if os.getenv("PRO_REGION_DEFAULT_RESOLUTION", "0") == "1" and not target.get("resolution_m"):
        target["resolution_m"] = int(os.getenv("PRO_CITY_RESOLUTION_M", "250")) if is_city else int(os.getenv("PRO_PROVINCE_RESOLUTION_M", "1000"))
        target["resolution_source"] = "legacy_region_default"
    else:
        target.setdefault("resolution_source", "auto_from_user_covariates")

    city_crs = os.getenv("PRO_TARGET_CRS_CITY", "EPSG:32648")
    china_albers = os.getenv(
        "PRO_TARGET_CRS_CHINA",
        "+proj=aea +lat_0=0 +lon_0=105 +lat_1=25 +lat_2=47 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs",
    )
    target["target_crs"] = city_crs if is_city else china_albers
    return target
