from __future__ import annotations

"""
GEE cloud sampler: user sample table -> Earth Engine covariates -> model-ready columns.

V77 scope:
- Keep the V75 standard covariate loop stable.
- Add an administrative-boundary mask band for formal prediction clipping.
- Add a cropland mask band, by default from ESA WorldCover class 40.
- Keep masks as diagnostic / output-control columns; the downstream model excludes mask columns
  from model predictors so the model is not trained on AOI membership flags.

Important design:
- GEE is used to sample covariates and masks at points. Python still trains the model and writes
  GeoTIFFs.
- If a strict official administrative boundary is required, configure a GEE FeatureCollection asset
  through PRO_GEE_ADMIN_ASSET and set PRO_GEE_ADMIN_REQUIRE=1. The built-in public GAUL route is a
  practical default, but name matching may vary by region.
"""

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import ee
except Exception:  # pragma: no cover
    ee = None


LogFn = Callable[[str, str, Any | None], None]


@dataclass
class GeeSamplerResult:
    ok: bool
    columns: list[str]
    dataframe: Any | None
    message: str
    meta: dict[str, Any]


# Minimal Chinese -> GAUL English alias table for the regions currently supported by the app's
# text/bbox router. Users can override everything with PRO_GEE_ADMIN_ASSET and field settings.
GAUL_REGION_ALIASES: dict[str, dict[str, Any]] = {
    "成都市": {"level": "city", "adm1": ["Sichuan Sheng", "Sichuan"], "adm2": ["Chengdu", "Chengdu Shi", "Chengdu City"]},
    "绵阳市": {"level": "city", "adm1": ["Sichuan Sheng", "Sichuan"], "adm2": ["Mianyang", "Mianyang Shi", "Mianyang City"]},
    "德阳市": {"level": "city", "adm1": ["Sichuan Sheng", "Sichuan"], "adm2": ["Deyang", "Deyang Shi", "Deyang City"]},
    "眉山市": {"level": "city", "adm1": ["Sichuan Sheng", "Sichuan"], "adm2": ["Meishan", "Meishan Shi", "Meishan City"]},
    "雅安市": {"level": "city", "adm1": ["Sichuan Sheng", "Sichuan"], "adm2": ["Ya'an", "Yaan", "Ya'an Shi", "Yaan Shi"]},
    "重庆市": {"level": "province", "adm1": ["Chongqing", "Chongqing Shi"]},
    "北京市": {"level": "province", "adm1": ["Beijing", "Beijing Shi"]},
    "上海市": {"level": "province", "adm1": ["Shanghai", "Shanghai Shi"]},
    "广州市": {"level": "city", "adm1": ["Guangdong Sheng", "Guangdong"], "adm2": ["Guangzhou", "Guangzhou Shi", "Guangzhou City"]},
    "深圳市": {"level": "city", "adm1": ["Guangdong Sheng", "Guangdong"], "adm2": ["Shenzhen", "Shenzhen Shi", "Shenzhen City"]},
    "武汉市": {"level": "city", "adm1": ["Hubei Sheng", "Hubei"], "adm2": ["Wuhan", "Wuhan Shi", "Wuhan City"]},
    "西安市": {"level": "city", "adm1": ["Shaanxi Sheng", "Shaanxi"], "adm2": ["Xi'an", "Xian", "Xi'an Shi", "Xian Shi"]},
    "昆明市": {"level": "city", "adm1": ["Yunnan Sheng", "Yunnan"], "adm2": ["Kunming", "Kunming Shi", "Kunming City"]},
    "贵阳市": {"level": "city", "adm1": ["Guizhou Sheng", "Guizhou"], "adm2": ["Guiyang", "Guiyang Shi", "Guiyang City"]},
    "四川省": {"level": "province", "adm1": ["Sichuan Sheng", "Sichuan"]},
    "河北省": {"level": "province", "adm1": ["Hebei Sheng", "Hebei"]},
    "广东省": {"level": "province", "adm1": ["Guangdong Sheng", "Guangdong"]},
    "湖北省": {"level": "province", "adm1": ["Hubei Sheng", "Hubei"]},
    "陕西省": {"level": "province", "adm1": ["Shaanxi Sheng", "Shaanxi"]},
    "云南省": {"level": "province", "adm1": ["Yunnan Sheng", "Yunnan"]},
    "贵州省": {"level": "province", "adm1": ["Guizhou Sheng", "Guizhou"]},
}


def _truthy(value: str | None, default: str = "0") -> bool:
    return str(value if value is not None else default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _split_env_list(raw: str | None) -> list[str]:
    out: list[str] = []
    for x in re.split(r"[,;，；|]", raw or ""):
        x = x.strip()
        if x:
            out.append(x)
    return out


def _strip_chinese_admin_suffix(name: str) -> str:
    s = str(name or "").strip()
    for suffix in ["特别行政区", "壮族自治区", "维吾尔自治区", "回族自治区", "自治区", "省", "市", "地区", "盟"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


class GeeSampler:
    """Small/medium table GEE sampler.

    Profiles:
    - minimal: elevation, annual NDVI mean, annual EVI mean.
    - standard: minimal + terrain + vegetation seasonal stats + CHIRPS precipitation.
    - extended: standard + MODIS LST and optional land-cover bands.
    - thesis: covariate stack aligned to the user's opening report: terrain derivatives, climate-water, vegetation/NPP, and soil-background proxies where GEE assets are available.

    Masks:
    - cov_gee_admin_mask: 1 inside resolved administrative boundary, 0 outside.
    - cov_gee_cropland_mask: 1 for cropland pixels, 0 otherwise.
    """

    MASK_COLUMNS = {"cov_gee_admin_mask", "cov_gee_cropland_mask", "cov_gee_worldcover_class"}

    def __init__(self, project_id: str | None = None, log_fn: LogFn | None = None):
        self.project_id = project_id or os.getenv("PRO_GEE_PROJECT") or os.getenv("GEE_PROJECT") or "fit-territory-472114-b0"
        self.log_fn = log_fn
        self.max_getinfo_samples = int(os.getenv("PRO_GEE_GETINFO_MAX_SAMPLES", "5000"))
        self.scale = int(os.getenv("PRO_GEE_SAMPLE_SCALE", os.getenv("PRO_CITY_RESOLUTION_M", "250")))
        self.modis_collection = os.getenv("PRO_GEE_MODIS_COLLECTION", "MODIS/061/MOD13Q1")
        self.dem_image = os.getenv("PRO_GEE_DEM_IMAGE", "USGS/SRTMGL1_003")
        self.profile = (os.getenv("PRO_GEE_COVARIATE_PROFILE", "thesis") or "thesis").strip().lower()
        self.last_stack_meta: dict[str, Any] = {}

    def _log(self, stage: str, msg: str, payload: Any | None = None) -> None:
        if self.log_fn:
            self.log_fn(stage, msg, payload)

    def initialize(self) -> None:
        if ee is None:
            raise RuntimeError("缺少 earthengine-api，请先执行 pip install earthengine-api，并完成 earthengine authenticate。")
        self._log("GEE", "开始初始化 Earth Engine", {"project": self.project_id})
        ee.Initialize(project=self.project_id)
        self._log("GEE", "Earth Engine 初始化成功", {"project": self.project_id})

    def _build_feature_collection(self, samples_df: Any):
        if ee is None:
            raise RuntimeError("缺少 earthengine-api。")
        features = []
        for i, row in samples_df.reset_index(drop=True).iterrows():
            lon = float(row["lon"])
            lat = float(row["lat"])
            som = float(row["som"])
            sample_id = int(row["sample_id"]) if "sample_id" in samples_df.columns else int(i)
            features.append(
                ee.Feature(
                    ee.Geometry.Point([lon, lat]),
                    {"sample_id": sample_id, "lon": lon, "lat": lat, "som": som},
                )
            )
        self._log("GEE", "样点 FeatureCollection 构建成功", {"sample_count": len(features)})
        return ee.FeatureCollection(features)

    def _annual_mod13(self, aoi: Any, year: int):
        start = f"{year}-01-01"
        # Earth Engine filterDate uses an exclusive end date. Use Jan 1 of the
        # following year so the requested calendar year is complete.
        end = f"{int(year)+1}-01-01"
        mod13 = (
            ee.ImageCollection(self.modis_collection)
            .filterDate(start, end)
            .filterBounds(aoi)
            .select(["NDVI", "EVI"])
        )
        count = int(mod13.size().getInfo())
        self._log("GEE", "MODIS 影像检索完成", {"collection": self.modis_collection, "year": year, "count": count})
        if count <= 0:
            raise RuntimeError(f"GEE未检索到 {year} 年目标区域内的 MOD13Q1 数据。")
        return mod13, count

    def _or_filter(self, field: str, values: list[str]):
        values = [str(v) for v in values if str(v).strip()]
        if not values:
            return None
        filters = [ee.Filter.eq(field, v) for v in values]
        if len(filters) == 1:
            return filters[0]
        return ee.Filter.Or(*filters)

    def _count_fc(self, fc: Any) -> int:
        try:
            return int(fc.size().getInfo())
        except Exception:
            return 0

    def _target_region_aliases(self, target: dict[str, Any] | None) -> dict[str, Any]:
        target = target or {}
        region = str(target.get("region") or "").strip()
        province = str(target.get("province") or target.get("region_inference", {}).get("province") or "").strip()
        city = str(target.get("city") or target.get("region_inference", {}).get("city") or "").strip()
        explicit_value = os.getenv("PRO_GEE_ADMIN_NAME_VALUE", "").strip()
        aliases: dict[str, Any] = {"region": region, "province": province, "city": city, "custom_values": _split_env_list(explicit_value)}
        if region in GAUL_REGION_ALIASES:
            aliases.update(GAUL_REGION_ALIASES[region])
        elif city and city in GAUL_REGION_ALIASES:
            aliases.update(GAUL_REGION_ALIASES[city])
        elif province and province in GAUL_REGION_ALIASES:
            aliases.update(GAUL_REGION_ALIASES[province])
        else:
            base = _strip_chinese_admin_suffix(region)
            # For custom GEE assets using Chinese attribute names, exact Chinese aliases are still useful.
            aliases.setdefault("level", "city" if (region.endswith("市") or target.get("is_city")) else "province")
            aliases.setdefault("adm1", [region, base] if region else [])
            aliases.setdefault("adm2", [region, base] if region else [])
        return aliases

    def _geometry_from_custom_admin_asset(self, target: dict[str, Any] | None, meta: dict[str, Any]):
        asset = (os.getenv("PRO_GEE_ADMIN_ASSET") or os.getenv("PRO_GEE_ADMIN_FC") or "").strip()
        # Backward-compatible convenience: allow PRO_ADMIN_CITY_VECTOR/PRO_ADMIN_PROVINCE_VECTOR
        # to point to a GEE FeatureCollection asset. Local paths remain handled only by the Python
        # region inference module; they are not auto-uploaded to GEE.
        target = target or {}
        if not asset:
            region_level = "city" if bool(target.get("is_city")) else "province"
            candidate = os.getenv("PRO_ADMIN_CITY_VECTOR" if region_level == "city" else "PRO_ADMIN_PROVINCE_VECTOR", "").strip()
            if candidate.startswith(("projects/", "users/")) or "/assets/" in candidate:
                asset = candidate
        if not asset:
            return None, meta

        aliases = self._target_region_aliases(target)
        field_raw = (
            os.getenv("PRO_GEE_ADMIN_NAME_FIELD")
            or os.getenv("PRO_ADMIN_CITY_NAME_FIELD" if bool(target.get("is_city")) else "PRO_ADMIN_PROVINCE_NAME_FIELD")
            or ""
        ).strip()
        field_candidates = _split_env_list(field_raw) or [
            "name", "NAME", "Name", "市", "市名", "地市", "地级市", "city", "province", "省", "省名", "ADM1_NAME", "ADM2_NAME"
        ]
        value_candidates = aliases.get("custom_values") or []
        if not value_candidates:
            region = str(aliases.get("region") or "").strip()
            city = str(aliases.get("city") or "").strip()
            province = str(aliases.get("province") or "").strip()
            value_candidates = [x for x in [region, city, province, _strip_chinese_admin_suffix(region)] if x]
            value_candidates += list(aliases.get("adm2") or []) + list(aliases.get("adm1") or [])
        value_candidates = list(dict.fromkeys([str(v).strip() for v in value_candidates if str(v).strip()]))

        fc0 = ee.FeatureCollection(asset)
        best_fc = None
        best_count = 0
        best_field = None
        for field in field_candidates:
            flt = self._or_filter(field, value_candidates)
            if flt is None:
                continue
            fc = fc0.filter(flt)
            count = self._count_fc(fc)
            if count > best_count:
                best_fc = fc
                best_count = count
                best_field = field
            if count > 0:
                break
        meta.update({
            "source": "custom_gee_asset",
            "asset": asset,
            "field_candidates": field_candidates,
            "value_candidates": value_candidates,
            "matched_field": best_field,
            "matched_count": int(best_count),
        })
        if best_fc is not None and best_count > 0:
            meta.update({"ok": True, "message": f"已从自定义GEE行政区资产匹配 {best_count} 个要素。"})
            return best_fc.geometry(), meta
        meta.update({"ok": False, "message": "自定义GEE行政区资产未匹配到目标区域。"})
        return None, meta

    def _geometry_from_public_gaul(self, target: dict[str, Any] | None, meta: dict[str, Any]):
        aliases = self._target_region_aliases(target)
        region = aliases.get("region") or ""
        if not region or str(region) in {"全国", "中国", "样点包络范围"}:
            meta.update({"source": "public_gaul", "ok": False, "message": "目标区域不是可裁剪的省/市行政区。"})
            return None, meta
        level = str(aliases.get("level") or ("city" if (target or {}).get("is_city") else "province"))
        level1_asset = os.getenv("PRO_GEE_GAUL_LEVEL1", "FAO/GAUL_SIMPLIFIED_500m/2015/level1")
        level2_asset = os.getenv("PRO_GEE_GAUL_LEVEL2", "FAO/GAUL_SIMPLIFIED_500m/2015/level2")

        fc = None
        matched_count = 0
        matched_level = None
        applied_filters: dict[str, Any] = {"adm1": aliases.get("adm1") or [], "adm2": aliases.get("adm2") or []}

        if level == "city":
            base = ee.FeatureCollection(level2_asset).filter(ee.Filter.eq("ADM0_NAME", "China"))
            adm2_filter = self._or_filter("ADM2_NAME", list(aliases.get("adm2") or []))
            adm1_filter = self._or_filter("ADM1_NAME", list(aliases.get("adm1") or []))
            cand = base
            if adm2_filter is not None:
                cand = cand.filter(adm2_filter)
            if adm1_filter is not None:
                cand = cand.filter(adm1_filter)
            count = self._count_fc(cand)
            if count > 0:
                fc, matched_count, matched_level = cand, count, "level2"

        if fc is None:
            base = ee.FeatureCollection(level1_asset).filter(ee.Filter.eq("ADM0_NAME", "China"))
            adm1_filter = self._or_filter("ADM1_NAME", list(aliases.get("adm1") or []))
            cand = base.filter(adm1_filter) if adm1_filter is not None else base
            count = self._count_fc(cand)
            if count > 0:
                fc, matched_count, matched_level = cand, count, "level1"

        meta.update({
            "source": "public_gaul",
            "level1_asset": level1_asset,
            "level2_asset": level2_asset,
            "region": region,
            "requested_level": level,
            "matched_level": matched_level,
            "matched_count": int(matched_count),
            "applied_filters": applied_filters,
        })
        if fc is not None and matched_count > 0:
            meta.update({"ok": True, "message": f"已从公共GAUL行政区数据匹配 {matched_count} 个要素。"})
            return fc.geometry(), meta
        meta.update({"ok": False, "message": "公共GAUL行政区数据未匹配到目标区域。可配置PRO_GEE_ADMIN_ASSET解决。"})
        return None, meta

    def resolve_admin_geometry(self, target: dict[str, Any] | None, fallback_geometry: Any | None = None):
        """Return (geometry, meta). Geometry is None when disabled or unresolved."""
        meta: dict[str, Any] = {"enabled": _truthy(os.getenv("PRO_GEE_USE_ADMIN_BOUNDARY"), "1")}
        if ee is None:
            raise RuntimeError("缺少 earthengine-api。")
        if not meta["enabled"]:
            meta.update({"ok": False, "message": "PRO_GEE_USE_ADMIN_BOUNDARY=0，未启用行政边界。"})
            return None, meta

        source = (os.getenv("PRO_GEE_ADMIN_SOURCE", "auto") or "auto").strip().lower()
        errors: list[str] = []
        if source in {"auto", "asset", "gee_asset", "custom"}:
            try:
                geom, meta_asset = self._geometry_from_custom_admin_asset(target, dict(meta))
                if geom is not None and meta_asset.get("ok"):
                    return geom, meta_asset
                errors.append(str(meta_asset.get("message") or "custom asset not matched"))
            except Exception as exc:
                errors.append(f"custom_asset_error: {exc}")

        if source in {"auto", "gaul", "public_gaul", "public"}:
            try:
                geom, meta_gaul = self._geometry_from_public_gaul(target, dict(meta))
                if geom is not None and meta_gaul.get("ok"):
                    return geom, meta_gaul
                errors.append(str(meta_gaul.get("message") or "public GAUL not matched"))
            except Exception as exc:
                errors.append(f"public_gaul_error: {exc}")

        meta.update({
            "ok": False,
            "source": source,
            "errors": errors,
            "message": "未解析到可用行政边界，已回退样点/区域bbox；输出报告会标记admin_boundary_applied=false。",
        })
        if _truthy(os.getenv("PRO_GEE_ADMIN_REQUIRE"), "0"):
            raise RuntimeError("PRO_GEE_ADMIN_REQUIRE=1，但未解析到可用行政边界：" + "；".join(errors))
        return None, meta

    def resolve_target_bounds(self, target: dict[str, Any] | None, fallback_bounds: dict[str, float]) -> tuple[dict[str, float], str, dict[str, Any]]:
        """Use the resolved admin geometry bounds when available; otherwise use fallback bounds."""
        if ee is None:
            raise RuntimeError("缺少 earthengine-api。")
        fb = fallback_bounds
        fallback_geom = ee.Geometry.Rectangle(
            [float(fb["lon_min"]), float(fb["lat_min"]), float(fb["lon_max"]), float(fb["lat_max"])],
            "EPSG:4326",
            False,
        )
        geom, meta = self.resolve_admin_geometry(target, fallback_geom)
        if geom is None:
            return fallback_bounds, "fallback_bbox_no_admin_boundary", meta
        coords = geom.bounds(maxError=1000).coordinates().getInfo()[0]
        xs = [float(p[0]) for p in coords]
        ys = [float(p[1]) for p in coords]
        pad = float(os.getenv("PRO_GEE_ADMIN_BOUNDS_PAD_DEG", os.getenv("PRO_GEE_REGION_BBOX_PAD_DEG", "0.005")))
        bounds = {
            "lon_min": min(xs) - pad,
            "lon_max": max(xs) + pad,
            "lat_min": min(ys) - pad,
            "lat_max": max(ys) + pad,
        }
        return bounds, f"admin_boundary:{meta.get('source')}:{meta.get('matched_level') or 'custom'}", meta

    def _resolve_target_aoi(self, samples_fc: Any, target: dict[str, Any] | None):
        fallback_aoi = samples_fc.geometry().bounds()
        admin_geom, admin_meta = self.resolve_admin_geometry(target, fallback_aoi)
        if admin_geom is not None:
            return admin_geom, admin_geom, admin_meta
        return fallback_aoi, None, admin_meta

    def _build_cropland_mask_image(self, year: int, aoi: Any, warnings: list[str], dataset_meta: dict[str, Any]):
        source = (os.getenv("PRO_GEE_CROPLAND_MASK_SOURCE", "worldcover") or "worldcover").strip().lower()
        if source in {"custom", "asset", "gee_asset"} and os.getenv("PRO_GEE_CROPLAND_ASSET", "").strip():
            asset = os.getenv("PRO_GEE_CROPLAND_ASSET", "").strip()
            band = os.getenv("PRO_GEE_CROPLAND_BAND", "").strip()
            img = ee.Image(asset)
            if band:
                img = img.select(band)
            values = _split_env_list(os.getenv("PRO_GEE_CROPLAND_VALUES", "1")) or ["1"]
            mask = None
            for v in values:
                try:
                    part = img.eq(float(v))
                except Exception:
                    part = img.eq(v)
                mask = part if mask is None else mask.Or(part)
            dataset_meta["cropland_mask"] = {"source": "custom_asset", "asset": asset, "band": band, "values": values}
            return mask.clip(aoi).unmask(0), dataset_meta["cropland_mask"]

        # Default: ESA WorldCover. v100=2020, v200=2021. For later years, use 2021 as nearest stable mask.
        wc_collection = os.getenv("PRO_GEE_WORLDCOVER_COLLECTION", "").strip()
        wc_version_year = 2021 if int(year) >= 2021 else 2020
        if not wc_collection:
            wc_collection = "ESA/WorldCover/v200" if wc_version_year >= 2021 else "ESA/WorldCover/v100"
        wc = ee.ImageCollection(wc_collection).first().select("Map").clip(aoi)
        cropland_value = float(os.getenv("PRO_GEE_WORLDCOVER_CROPLAND_CLASS", "40"))
        raw = wc.eq(cropland_value).unmask(0).rename("cropland_raw")
        mask_mode = (os.getenv("PRO_GEE_CROPLAND_MASK_MODE", "fraction") or "fraction").strip().lower()
        meta = {
            "source": "ESA WorldCover",
            "collection": wc_collection,
            "mask_year": wc_version_year,
            "requested_year": int(year),
            "class_value": cropland_value,
            "mask_mode": mask_mode,
        }
        if mask_mode in {"fraction", "fractional", "soft", "area_fraction"}:
            # Important: sampling a 10 m categorical mask only at the 250/500 m cell centre
            # creates salt-and-pepper NoData holes in the final map. Use area fraction at
            # the modelling/output scale so a 250/500 m cell is kept when it contains
            # cropland, not only when its exact centre is cropland.
            max_pixels = int(os.getenv("PRO_GEE_CROPLAND_REDUCE_MAX_PIXELS", "4096"))
            scale = int(os.getenv("PRO_GEE_CROPLAND_FRACTION_SCALE", str(self.scale)))
            try:
                mask = raw.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=max_pixels).reproject(crs=raw.projection(), scale=scale)
            except Exception as exc:
                warnings.append(f"WorldCover耕地比例掩膜构建失败，回退中心点二值掩膜：{exc}")
                mask = raw
                meta["mask_mode"] = "center_binary_fallback"
            meta.update({"fraction_scale": scale, "reduce_max_pixels": max_pixels})
        else:
            mask = raw
            meta["mask_mode"] = "center_binary"
        dataset_meta["cropland_mask"] = meta
        return mask.rename("cov_gee_cropland_mask"), meta

    def _build_stack(self, samples_fc: Any, year: int, target: dict[str, Any] | None = None):
        if ee is None:
            raise RuntimeError("缺少 earthengine-api。")
        aoi, admin_geom, admin_meta = self._resolve_target_aoi(samples_fc, target)
        profile = self.profile
        if profile not in {"minimal", "standard", "extended", "thesis", "auto"}:
            profile = "standard"
        if profile == "auto":
            profile = "thesis"

        bands: list[Any] = []
        band_names: list[str] = []
        warnings: list[str] = []
        dataset_meta: dict[str, Any] = {
            "profile": profile,
            "year": int(year),
            "scale": int(self.scale),
            "datasets": [],
            "warnings": warnings,
            "admin_boundary": admin_meta,
            "temporal_policy": "dynamic GEE collections are filtered by requested calendar year; static DEM/soil products are marked as static/background.",
            "date_ranges": {},
        }

        def add(img: Any, name: str, dataset: str):
            bands.append(img.rename(name))
            band_names.append(name)
            dataset_meta["datasets"].append({"band": name, "dataset": dataset})

        def add_optional(img: Any, name: str, dataset: str, check: bool = False) -> bool:
            """Add optional GEE image band without making the whole stack fragile."""
            try:
                candidate = img.rename(name)
                if check or _truthy(os.getenv("PRO_GEE_VALIDATE_OPTIONAL_ASSETS"), "1"):
                    # Force early validation for optional assets. If an asset id/band is unavailable,
                    # skip only that band instead of failing later during sampleRegions.getInfo().
                    candidate.bandNames().getInfo()
                add(img, name, dataset)
                return True
            except Exception as exc:
                warnings.append(f"可选GEE协变量 {name} 构建失败，已跳过：{exc}")
                return False

        def first_existing_band(asset_id: str, preferred: list[str] | None = None):
            """Load a single topsoil-like band from an optional GEE image asset."""
            try:
                img = ee.Image(asset_id)
                names = list(img.bandNames().getInfo())
                preferred = preferred or []
                chosen = None
                for b in preferred:
                    if b in names:
                        chosen = b
                        break
                if chosen is None:
                    # Most OpenLandMap soil products expose depth bands like b0/b10/b30/b60/b100/b200.
                    for b in names:
                        bs = str(b).lower()
                        if bs in {"b0", "b0_5cm", "b0_5", "0", "0cm"} or bs.startswith("b0"):
                            chosen = b
                            break
                if chosen is None and names:
                    chosen = names[0]
                if chosen is None:
                    raise RuntimeError("asset has no bands")
                return img.select(chosen), chosen
            except Exception as exc:
                warnings.append(f"可选GEE资产 {asset_id} 无法读取，已跳过：{exc}")
                return None, None

        if os.getenv("PRO_GEE_USE_DEM", "1") == "1":
            dem = ee.Image(self.dem_image).select("elevation")
            add(dem, "cov_gee_elevation", self.dem_image)
            if profile in {"standard", "extended", "thesis"} and os.getenv("PRO_GEE_USE_TERRAIN", "1") == "1":
                terrain = ee.Terrain.products(dem)
                add(terrain.select("slope"), "cov_gee_slope", self.dem_image + ":ee.Terrain.slope")
                aspect = terrain.select("aspect")
                aspect_rad = aspect.multiply(math.pi / 180.0)
                add(aspect_rad.sin(), "cov_gee_sin_aspect", self.dem_image + ":ee.Terrain.aspect_sin")
                add(aspect_rad.cos(), "cov_gee_cos_aspect", self.dem_image + ":ee.Terrain.aspect_cos")
                if profile in {"thesis", "extended"} and _truthy(os.getenv("PRO_GEE_USE_TERRAIN_EXTRA"), "1"):
                    try:
                        k3 = ee.Kernel.square(radius=1, units="pixels", normalize=False)
                        k5 = ee.Kernel.square(radius=2, units="pixels", normalize=False)
                        mean3 = dem.reduceNeighborhood(ee.Reducer.mean(), k3)
                        max3 = dem.reduceNeighborhood(ee.Reducer.max(), k3)
                        min3 = dem.reduceNeighborhood(ee.Reducer.min(), k3)
                        max5 = dem.reduceNeighborhood(ee.Reducer.max(), k5)
                        min5 = dem.reduceNeighborhood(ee.Reducer.min(), k5)
                        add_optional(dem.subtract(mean3), "cov_gee_tpi_3x3", self.dem_image + ":TPI_3x3")
                        add_optional(max3.subtract(min3), "cov_gee_roughness_3x3", self.dem_image + ":roughness_3x3")
                        add_optional(max5.subtract(min5), "cov_gee_relief_5x5", self.dem_image + ":relief_5x5")
                        laplacian = ee.Kernel.fixed(3, 3, [[0, 1, 0], [1, -4, 1], [0, 1, 0]])
                        add_optional(dem.convolve(laplacian), "cov_gee_curvature_laplacian", self.dem_image + ":laplacian_curvature_proxy")
                        slope_rad = terrain.select("slope").multiply(math.pi / 180.0)
                        twi_proxy = slope_rad.tan().add(0.001).pow(-1).log()
                        add_optional(twi_proxy, "cov_gee_twi_proxy", self.dem_image + ":slope_based_twi_proxy")
                    except Exception as exc:
                        warnings.append(f"扩展地形因子构建失败，已跳过：{exc}")

        if os.getenv("PRO_GEE_USE_MODIS_NDVI", "1") == "1" or os.getenv("PRO_GEE_USE_MODIS_EVI", "1") == "1":
            mod13, count = self._annual_mod13(aoi, int(year))
            ndvi_ic = mod13.select("NDVI").map(lambda img: img.multiply(0.0001).copyProperties(img, ["system:time_start"]))
            evi_ic = mod13.select("EVI").map(lambda img: img.multiply(0.0001).copyProperties(img, ["system:time_start"]))
            if os.getenv("PRO_GEE_USE_MODIS_NDVI", "1") == "1":
                add(ndvi_ic.mean(), "cov_gee_ndvi_mean", self.modis_collection)
                if profile in {"standard", "extended", "thesis"} and os.getenv("PRO_GEE_USE_MODIS_SEASONAL", "1") == "1":
                    add(ndvi_ic.reduce(ee.Reducer.stdDev()), "cov_gee_ndvi_std", self.modis_collection)
                    add(ndvi_ic.max().subtract(ndvi_ic.min()), "cov_gee_ndvi_range", self.modis_collection)
            if os.getenv("PRO_GEE_USE_MODIS_EVI", "1") == "1":
                add(evi_ic.mean(), "cov_gee_evi_mean", self.modis_collection)
                if profile in {"standard", "extended", "thesis"} and os.getenv("PRO_GEE_USE_MODIS_SEASONAL", "1") == "1":
                    add(evi_ic.reduce(ee.Reducer.stdDev()), "cov_gee_evi_std", self.modis_collection)
                    add(evi_ic.max().subtract(evi_ic.min()), "cov_gee_evi_range", self.modis_collection)
            dataset_meta["mod13_count"] = int(count)
            dataset_meta["date_ranges"]["MOD13Q1"] = {"start": f"{int(year)}-01-01", "end_exclusive": f"{int(year)+1}-01-01"}

        if profile in {"standard", "extended", "thesis"} and os.getenv("PRO_GEE_USE_CHIRPS_PRECIP", "1") == "1":
            try:
                chirps = ee.ImageCollection(os.getenv("PRO_GEE_CHIRPS_COLLECTION", "UCSB-CHG/CHIRPS/DAILY")).filterBounds(aoi)
                annual = chirps.filterDate(f"{int(year)}-01-01", f"{int(year)+1}-01-01").select("precipitation").sum()
                growing = chirps.filterDate(f"{int(year)}-04-01", f"{int(year)}-11-01").select("precipitation").sum()
                add(annual, "cov_gee_precip_annual", "UCSB-CHG/CHIRPS/DAILY")
                add(growing, "cov_gee_precip_growing", "UCSB-CHG/CHIRPS/DAILY")
                dataset_meta["date_ranges"]["CHIRPS"] = {"annual": [f"{int(year)}-01-01", f"{int(year)+1}-01-01"], "growing": [f"{int(year)}-04-01", f"{int(year)}-11-01"]}
            except Exception as exc:
                warnings.append(f"CHIRPS降水变量构建失败，已跳过：{exc}")

        if profile in {"thesis", "extended"} and _truthy(os.getenv("PRO_GEE_USE_TERRACLIMATE"), "1"):
            try:
                tc = (
                    ee.ImageCollection(os.getenv("PRO_GEE_TERRACLIMATE_COLLECTION", "IDAHO_EPSCOR/TERRACLIMATE"))
                    .filterDate(f"{int(year)}-01-01", f"{int(year)+1}-01-01")
                    .filterBounds(aoi)
                )
                tc_count = int(tc.size().getInfo())
                if tc_count > 0:
                    tmmx = tc.select("tmmx").mean().multiply(0.1)
                    tmmn = tc.select("tmmn").mean().multiply(0.1)
                    tmean = tmmx.add(tmmn).divide(2.0)
                    trange = tmmx.subtract(tmmn)
                    pr = tc.select("pr").sum()
                    pet = tc.select("pet").sum().multiply(0.1)
                    add_optional(tmean, "cov_gee_temp_mean", "IDAHO_EPSCOR/TERRACLIMATE:tmmx_tmmn")
                    add_optional(trange, "cov_gee_temp_range", "IDAHO_EPSCOR/TERRACLIMATE:tmmx_minus_tmmn")
                    add_optional(pet, "cov_gee_pet_annual", "IDAHO_EPSCOR/TERRACLIMATE:pet")
                    add_optional(pr.subtract(pet), "cov_gee_climate_water_balance", "IDAHO_EPSCOR/TERRACLIMATE:pr_minus_pet")
                    dataset_meta["terraclimate_count"] = int(tc_count)
                    dataset_meta["date_ranges"]["TerraClimate"] = {"start": f"{int(year)}-01-01", "end_exclusive": f"{int(year)+1}-01-01"}
                else:
                    warnings.append(f"TerraClimate在 {year} 年目标区域影像数为0，已跳过气温/PET变量。")
            except Exception as exc:
                warnings.append(f"TerraClimate气候水热变量构建失败，已跳过：{exc}")

        if profile in {"thesis", "extended"} and _truthy(os.getenv("PRO_GEE_USE_MODIS_NPP"), "1"):
            try:
                npp_coll = (
                    ee.ImageCollection(os.getenv("PRO_GEE_MODIS_NPP_COLLECTION", "MODIS/061/MOD17A3HGF"))
                    .filterDate(f"{year}-01-01", f"{int(year)+1}-01-01")
                    .filterBounds(aoi)
                )
                npp_count = int(npp_coll.size().getInfo())
                if npp_count > 0:
                    npp = npp_coll.select("Npp").mean().multiply(0.0001)
                    add_optional(npp, "cov_gee_npp", "MODIS/061/MOD17A3HGF:Npp")
                    dataset_meta["mod17_count"] = int(npp_count)
                    dataset_meta["date_ranges"]["MOD17A3HGF"] = {"start": f"{int(year)}-01-01", "end_exclusive": f"{int(year)+1}-01-01"}
                else:
                    warnings.append(f"MOD17A3HGF在 {year} 年目标区域影像数为0，已跳过NPP。")
            except Exception as exc:
                warnings.append(f"MODIS NPP变量构建失败，已跳过：{exc}")

        if profile in {"thesis", "extended"} and _truthy(os.getenv("PRO_GEE_USE_OPENLANDMAP_SOIL"), "1"):
            soil_assets = {
                "cov_gee_sand": (os.getenv("PRO_GEE_SOIL_SAND_ASSET", "OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02"), 1.0),
                "cov_gee_clay": (os.getenv("PRO_GEE_SOIL_CLAY_ASSET", "OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02"), 1.0),
                "cov_gee_ph": (os.getenv("PRO_GEE_SOIL_PH_ASSET", "OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02"), 0.1),
                "cov_gee_bulk_density": (os.getenv("PRO_GEE_SOIL_BD_ASSET", "OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02"), 1.0),
                "cov_gee_cec": (os.getenv("PRO_GEE_SOIL_CEC_ASSET", "OpenLandMap/SOL/SOL_CEC-CLAY__CMOL-KG_USDA-3A1A1A_M/v02"), 1.0),
            }
            soil_imgs: dict[str, Any] = {}
            for name, (asset_id, scale_factor) in soil_assets.items():
                img, band = first_existing_band(asset_id, preferred=["b0", "b0_5cm", "b10", "b30"])
                if img is not None:
                    img2 = img.multiply(float(scale_factor)) if float(scale_factor) != 1.0 else img
                    if add_optional(img2, name, f"{asset_id}:{band}"):
                        soil_imgs[name] = img2
            try:
                if "cov_gee_sand" in soil_imgs and "cov_gee_clay" in soil_imgs:
                    silt = ee.Image.constant(100).subtract(soil_imgs["cov_gee_sand"]).subtract(soil_imgs["cov_gee_clay"])
                    add_optional(silt, "cov_gee_silt", "OpenLandMap:100-sand-clay")
            except Exception as exc:
                warnings.append(f"粉粒含量silt代理变量构建失败，已跳过：{exc}")

        if profile in {"thesis", "extended"} and os.getenv("PRO_GEE_USE_MODIS_LST", "1") == "1":
            try:
                lst_coll = (
                    ee.ImageCollection(os.getenv("PRO_GEE_MODIS_LST_COLLECTION", "MODIS/061/MOD11A2"))
                    .filterDate(f"{int(year)}-01-01", f"{int(year)+1}-01-01")
                    .filterBounds(aoi)
                )
                lst_count = int(lst_coll.size().getInfo())
                if lst_count > 0:
                    add(lst_coll.select("LST_Day_1km").mean().multiply(0.02).subtract(273.15), "cov_gee_lst_day_mean", "MODIS/061/MOD11A2")
                    add(lst_coll.select("LST_Night_1km").mean().multiply(0.02).subtract(273.15), "cov_gee_lst_night_mean", "MODIS/061/MOD11A2")
                    dataset_meta["mod11_count"] = int(lst_count)
                    dataset_meta["date_ranges"]["MOD11A2"] = {"start": f"{int(year)}-01-01", "end_exclusive": f"{int(year)+1}-01-01"}
                else:
                    warnings.append(f"MOD11A2在 {year} 年目标区域影像数为0，已跳过LST。")
            except Exception as exc:
                warnings.append(f"MODIS LST变量构建失败，已跳过：{exc}")

        # V77 administrative mask. This is not a predictor; downstream excludes it from feature_cols.
        if admin_geom is not None and _truthy(os.getenv("PRO_GEE_CLIP_TO_ADMIN"), "1"):
            try:
                admin_mask = ee.Image.constant(1).clip(admin_geom).unmask(0)
                add(admin_mask, "cov_gee_admin_mask", f"admin_boundary:{admin_meta.get('source')}")
            except Exception as exc:
                warnings.append(f"行政边界mask构建失败，已跳过：{exc}")

        # V77 cropland mask. Enabled by PRO_GEE_MASK_TO_CROPLAND=1. WorldCover class 40 by default.
        if _truthy(os.getenv("PRO_GEE_MASK_TO_CROPLAND"), "1"):
            try:
                cropland_mask, cropland_meta = self._build_cropland_mask_image(int(year), aoi, warnings, dataset_meta)
                if _truthy(os.getenv("PRO_GEE_ADD_WORLDCOVER_CLASS"), "0") and cropland_meta.get("source") == "ESA WorldCover":
                    wc = ee.ImageCollection(cropland_meta["collection"]).first().select("Map").clip(aoi)
                    add(wc, "cov_gee_worldcover_class", cropland_meta["collection"])
                add(cropland_mask, "cov_gee_cropland_mask", str(cropland_meta))
            except Exception as exc:
                msg = f"耕地掩膜变量构建失败，已跳过：{exc}"
                if _truthy(os.getenv("PRO_GEE_CROPLAND_REQUIRE"), "0"):
                    raise RuntimeError(msg)
                warnings.append(msg)

        if not bands:
            raise RuntimeError("GEE协变量栈为空，请至少启用 PRO_GEE_USE_DEM 或 PRO_GEE_USE_MODIS_NDVI。")
        stack = bands[0]
        for img in bands[1:]:
            stack = stack.addBands(img)
        stack = stack.unmask(-9999)
        self.last_stack_meta = {**dataset_meta, "bands": band_names}
        self._log("GEE", "协变量影像栈构建成功", {"profile": profile, "bands": band_names, "warnings": warnings, "admin_boundary": admin_meta, "cropland_mask": dataset_meta.get("cropland_mask")})
        return stack, band_names, self.last_stack_meta

    def _fc_to_dataframe(self, fc: Any):
        info = fc.getInfo()
        rows = []
        for feature in info.get("features", []):
            rows.append(feature.get("properties", {}))
        if pd is None:
            raise RuntimeError("缺少 pandas。")
        return pd.DataFrame(rows)

    def _clean_covariate_dataframe(self, out_df: Any, cov_cols: list[str]) -> tuple[Any, list[str], dict[str, int]]:
        for c in cov_cols:
            if c in out_df.columns:
                out_df[c] = pd.to_numeric(out_df[c], errors="coerce")
                out_df.loc[out_df[c] == -9999, c] = pd.NA
        existing = [c for c in cov_cols if c in out_df.columns]
        existing = [c for c in existing if int(out_df[c].notna().sum()) > 0]
        valid_counts = {c: int(out_df[c].notna().sum()) for c in existing}
        return out_df, existing, valid_counts


    def sample_points_grid_to_dataframe(self, grid_df: Any, year: int, width: int, height: int, target: dict[str, Any] | None = None) -> GeeSamplerResult:
        """Sample GEE covariates for a pre-built prediction grid.

        V82: used by the projected-meter output path. Python builds the grid in
        EPSG:32648 or another target CRS, stores x/y for GeoTIFF writing, and sends
        only the lon/lat cell centers to GEE. This avoids pretending that a lon/lat
        width-height grid has a real 250 m pixel size.
        """
        if pd is None:
            raise RuntimeError("缺少 pandas。")
        if ee is None:
            raise RuntimeError("缺少 earthengine-api。")
        required = {"grid_id", "row", "col", "lon", "lat"}
        missing = sorted(required - set(grid_df.columns))
        if missing:
            raise ValueError(f"投影预测网格缺少字段：{missing}")

        work = grid_df.copy().reset_index(drop=True)
        n = len(work)
        if n != int(width) * int(height):
            self._log("GEE", "投影预测网格点数与width*height不完全一致", {"rows": n, "width": int(width), "height": int(height)})
        if n > self.max_getinfo_samples:
            raise RuntimeError(
                f"预测网格点数 {n} 超过 PRO_GEE_GETINFO_MAX_SAMPLES={self.max_getinfo_samples}。"
                "请增大 PRO_GEE_FORMAL_OUTPUT_RESOLUTION_M，或使用异步Export.table流程。"
            )

        chunk_size = int(os.getenv("PRO_GEE_GETINFO_CHUNK_SIZE", "5000"))
        chunk_size = max(500, min(chunk_size, max(n, 500)))
        self.initialize()

        features = []
        property_cols = [c for c in ["grid_id", "row", "col", "lon", "lat", "x", "y"] if c in work.columns]
        for rec in work[property_cols].to_dict("records"):
            lon = float(rec["lon"])
            lat = float(rec["lat"])
            props = {}
            for k, v in rec.items():
                if pd.isna(v):
                    continue
                if k in {"grid_id", "row", "col"}:
                    props[k] = int(v)
                else:
                    props[k] = float(v)
            features.append(ee.Feature(ee.Geometry.Point([lon, lat]), props))

        full_fc = ee.FeatureCollection(features)
        lon_min = float(pd.to_numeric(work["lon"], errors="coerce").min())
        lon_max = float(pd.to_numeric(work["lon"], errors="coerce").max())
        lat_min = float(pd.to_numeric(work["lat"], errors="coerce").min())
        lat_max = float(pd.to_numeric(work["lat"], errors="coerce").max())
        bounds = {"lon_min": lon_min, "lon_max": lon_max, "lat_min": lat_min, "lat_max": lat_max}
        self._log("GEE", "投影预测网格 FeatureCollection 构建成功", {"width": int(width), "height": int(height), "points": n, "lonlat_bounds": bounds, "chunk_size": chunk_size})
        stack, cov_cols, stack_meta = self._build_stack(full_fc, int(year), target=target)
        self._log("GEE", "开始 sampleRegions 抽取投影预测网格协变量", {"scale": self.scale, "points": n, "chunk_size": chunk_size})

        dfs = []
        if n <= chunk_size:
            sampled = stack.sampleRegions(collection=full_fc, properties=property_cols, scale=self.scale, geometries=False)
            dfs.append(self._fc_to_dataframe(sampled))
        else:
            total_chunks = int(math.ceil(n / chunk_size))
            for chunk_idx, start in enumerate(range(0, n, chunk_size), start=1):
                end = min(start + chunk_size, n)
                chunk_fc = ee.FeatureCollection(features[start:end])
                self._log("GEE", "投影预测网格分块抽取中", {"chunk": chunk_idx, "total_chunks": total_chunks, "start": start, "end": end, "points": end - start})
                sampled = stack.sampleRegions(collection=chunk_fc, properties=property_cols, scale=self.scale, geometries=False)
                part = self._fc_to_dataframe(sampled)
                if part is not None and not part.empty:
                    dfs.append(part)

        if not dfs:
            return GeeSamplerResult(False, [], pd.DataFrame(), "GEE投影预测网格 sampleRegions 返回空表。", {})
        out_df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        if out_df.empty:
            return GeeSamplerResult(False, [], out_df, "GEE投影预测网格 sampleRegions 返回空表。", {})
        if "grid_id" in out_df.columns:
            out_df = out_df.sort_values("grid_id").reset_index(drop=True)
        out_df, existing, valid_counts = self._clean_covariate_dataframe(out_df, cov_cols)
        self._log("GEE", "GEE投影预测网格协变量抽取完成", {"columns": existing, "valid_counts": valid_counts, "rows": len(out_df)})
        return GeeSamplerResult(
            True,
            existing,
            out_df,
            "GEE投影预测网格协变量抽取成功。",
            {"valid_counts": valid_counts, "rows": int(len(out_df)), "year": int(year), "project": self.project_id, "width": int(width), "height": int(height), "bounds": bounds, "chunk_size": chunk_size, "grid_mode": "projected_meter", "stack": stack_meta},
        )

    def sample_grid_to_dataframe(self, bounds: dict[str, float], year: int, width: int, height: int, target: dict[str, Any] | None = None) -> GeeSamplerResult:
        """Build a lon/lat prediction grid, sample GEE covariates/masks, and return a dataframe.

        V80: supports chunked getInfo sampling. Earlier builds capped the grid at about 5,000
        points, which made the WebGIS result look like scattered patches. Chunking keeps the
        same Python-model workflow while allowing a denser preview grid.
        """
        if pd is None:
            raise RuntimeError("缺少 pandas。")
        if ee is None:
            raise RuntimeError("缺少 earthengine-api。")
        width = int(width)
        height = int(height)
        if width <= 1 or height <= 1:
            raise ValueError(f"预测网格尺寸非法：width={width}, height={height}")
        n = width * height
        if n > self.max_getinfo_samples:
            raise RuntimeError(
                f"预测网格点数 {n} 超过 PRO_GEE_GETINFO_MAX_SAMPLES={self.max_getinfo_samples}。"
                "请调小 PRO_GEE_PRED_GRID_WIDTH，或改用异步Export.table流程。"
            )
        chunk_size = int(os.getenv("PRO_GEE_GETINFO_CHUNK_SIZE", "5000"))
        chunk_size = max(500, min(chunk_size, max(n, 500)))
        lon_min = float(bounds["lon_min"]); lon_max = float(bounds["lon_max"])
        lat_min = float(bounds["lat_min"]); lat_max = float(bounds["lat_max"])
        if lon_max <= lon_min or lat_max <= lat_min:
            raise ValueError(f"预测网格范围非法：{bounds}")
        self.initialize()

        features = []
        for row in range(height):
            lat = lat_max - (lat_max - lat_min) * row / max(height - 1, 1)
            for col in range(width):
                lon = lon_min + (lon_max - lon_min) * col / max(width - 1, 1)
                pid = row * width + col
                features.append(
                    ee.Feature(
                        ee.Geometry.Point([float(lon), float(lat)]),
                        {"grid_id": int(pid), "row": int(row), "col": int(col), "lon": float(lon), "lat": float(lat)},
                    )
                )

        full_fc = ee.FeatureCollection(features)
        self._log("GEE", "预测网格 FeatureCollection 构建成功", {"width": width, "height": height, "points": n, "bounds": bounds, "chunk_size": chunk_size})
        stack, cov_cols, stack_meta = self._build_stack(full_fc, int(year), target=target)
        self._log("GEE", "开始 sampleRegions 抽取预测网格协变量", {"scale": self.scale, "points": n, "chunk_size": chunk_size})

        dfs = []
        if n <= chunk_size:
            sampled = stack.sampleRegions(
                collection=full_fc,
                properties=["grid_id", "row", "col", "lon", "lat"],
                scale=self.scale,
                geometries=False,
            )
            dfs.append(self._fc_to_dataframe(sampled))
        else:
            total_chunks = int(math.ceil(n / chunk_size))
            for chunk_idx, start in enumerate(range(0, n, chunk_size), start=1):
                end = min(start + chunk_size, n)
                chunk_fc = ee.FeatureCollection(features[start:end])
                self._log("GEE", "预测网格分块抽取中", {"chunk": chunk_idx, "total_chunks": total_chunks, "start": start, "end": end, "points": end - start})
                sampled = stack.sampleRegions(
                    collection=chunk_fc,
                    properties=["grid_id", "row", "col", "lon", "lat"],
                    scale=self.scale,
                    geometries=False,
                )
                part = self._fc_to_dataframe(sampled)
                if part is not None and not part.empty:
                    dfs.append(part)

        if not dfs:
            return GeeSamplerResult(False, [], pd.DataFrame(), "GEE预测网格 sampleRegions 返回空表。", {})
        out_df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        if out_df.empty:
            return GeeSamplerResult(False, [], out_df, "GEE预测网格 sampleRegions 返回空表。", {})
        if "grid_id" in out_df.columns:
            out_df = out_df.sort_values("grid_id").reset_index(drop=True)
        out_df, existing, valid_counts = self._clean_covariate_dataframe(out_df, cov_cols)
        self._log("GEE", "GEE预测网格协变量抽取完成", {"columns": existing, "valid_counts": valid_counts, "rows": len(out_df)})
        return GeeSamplerResult(
            True,
            existing,
            out_df,
            "GEE预测网格协变量抽取成功。",
            {"valid_counts": valid_counts, "rows": int(len(out_df)), "year": int(year), "project": self.project_id, "width": width, "height": height, "bounds": bounds, "chunk_size": chunk_size, "stack": stack_meta},
        )

    def sample_to_dataframe(self, samples_df: Any, year: int, target: dict[str, Any] | None = None) -> GeeSamplerResult:
        if pd is None:
            raise RuntimeError("缺少 pandas。")
        required = {"lon", "lat", "som"}
        missing = sorted(required - set(samples_df.columns))
        if missing:
            raise ValueError(f"GEE样点表缺少字段：{missing}")
        work = samples_df.copy().reset_index(drop=True)
        if "sample_id" not in work.columns:
            work["sample_id"] = range(len(work))
        if len(work) > self.max_getinfo_samples:
            raise RuntimeError(
                f"当前样点数 {len(work)} 超过 PRO_GEE_GETINFO_MAX_SAMPLES={self.max_getinfo_samples}。"
                "正式版应改为 Export.table 异步导出。"
            )
        self.initialize()
        fc = self._build_feature_collection(work)
        stack, cov_cols, stack_meta = self._build_stack(fc, int(year), target=target)
        self._log("GEE", "开始 sampleRegions 抽取样点协变量", {"scale": self.scale})
        sampled = stack.sampleRegions(
            collection=fc,
            properties=["sample_id", "lon", "lat", "som"],
            scale=self.scale,
            geometries=False,
        )
        out_df = self._fc_to_dataframe(sampled)
        if out_df.empty:
            return GeeSamplerResult(False, [], out_df, "GEE sampleRegions 返回空表。", {})
        if "sample_id" in out_df.columns:
            out_df = out_df.sort_values("sample_id").reset_index(drop=True)
        out_df, existing, valid_counts = self._clean_covariate_dataframe(out_df, cov_cols)
        self._log("GEE", "GEE样点协变量抽取完成", {"columns": existing, "valid_counts": valid_counts, "rows": len(out_df)})
        return GeeSamplerResult(True, existing, out_df, "GEE协变量抽取成功。", {"valid_counts": valid_counts, "rows": int(len(out_df)), "year": int(year), "project": self.project_id, "stack": stack_meta})
