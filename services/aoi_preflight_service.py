# -*- coding: utf-8 -*-
"""Strict AOI/sample compatibility preflight for PRO formal mapping.

Purpose:
- The requested mapping AOI must come from the user's text and local admin shapefiles.
- Uploaded samples are training samples only; they must not silently redefine the AOI.
- If the requested AOI is outside the uploaded samples' dominant province/city, stop before
  starting the long mapping task and let Dash show a toast popup instead of a chat reply.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None

from services.sample_reader import load_sample_points


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


def _split_env_paths(value: str) -> list[str]:
    return [x.strip().strip('"').strip("'") for x in re.split(r"[;,\n]+", value or "") if x.strip()]


def _candidate_admin_roots() -> list[Path]:
    """Candidate folders for built-in administrative boundaries.

    The user commonly keeps county-level boundaries in E:\\Agent_DSM\\行政区划.
    Environment variables still have highest priority, but the AI agent should
    work out-of-the-box with this project convention.
    """
    roots: list[Path] = []
    for key in ["PRO_LOCAL_ADMIN_VECTOR_DIR", "PRO_ADMIN_VECTOR_DIR"]:
        for s in _split_env_paths(os.getenv(key, "")):
            if s:
                roots.append(Path(s))
    # Common project/local conventions on Windows and inside the project folder.
    roots.extend([
        Path(r"E:\Agent_DSM\行政区划"),
        Path(r"E:\Agent_DSM\xingzhengquhua_shp"),
        Path.cwd() / "行政区划",
        Path.cwd().parent / "行政区划",
        Path.cwd() / "xingzhengquhua_shp",
        Path.cwd().parent / "xingzhengquhua_shp",
    ])
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            key = str(root.resolve()) if root.exists() else str(root)
        except Exception:
            key = str(root)
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


def _admin_path(level: str) -> Path | None:
    keys = {
        "county": ["PRO_LOCAL_ADMIN_COUNTY_VECTOR", "PRO_ADMIN_COUNTY_VECTOR"],
        "city": ["PRO_LOCAL_ADMIN_CITY_VECTOR", "PRO_ADMIN_CITY_VECTOR"],
        "province": ["PRO_LOCAL_ADMIN_PROVINCE_VECTOR", "PRO_ADMIN_PROVINCE_VECTOR"],
    }[level]
    for key in keys:
        for s in _split_env_paths(os.getenv(key, "")):
            p = Path(s)
            if p.exists() and p.is_file():
                return p
    exact_names = {
        "county": ["县.shp", "县级.shp", "区县.shp", "县级行政区.shp", "区县行政区.shp"],
        "city": ["市.shp", "地市.shp", "市级.shp", "市级行政区.shp"],
        "province": ["省.shp", "省级.shp", "省级行政区.shp"],
    }[level]
    keywords = {
        "county": ["县", "区", "县级", "区县"],
        "city": ["市", "地市", "市级"],
        "province": ["省", "省级"],
    }[level]
    excludes = {"十段线", "九段线", "界线", "南海诸岛", "点", "线"}
    for d in _candidate_admin_roots():
        if not d.exists() or not d.is_dir():
            continue
        for nm in exact_names:
            p = d / nm
            if p.exists() and p.is_file():
                return p
        shp_files = sorted(d.rglob("*.shp"))
        # Prefer shallow and semantically named files.
        scored: list[tuple[int, Path]] = []
        for p in shp_files:
            stem = p.stem
            if any(ex in stem for ex in excludes):
                continue
            score = 0
            if any(k in stem for k in keywords):
                score += 20
            if level == "county" and any(k in stem for k in ["县", "区县", "county", "district"]):
                score += 10
            if level == "city" and any(k in stem for k in ["市", "city"]):
                score += 10
            if level == "province" and any(k in stem for k in ["省", "province"]):
                score += 10
            score -= len(p.relative_to(d).parts)
            if score > 0:
                scored.append((score, p))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]
    return None

def _name_fields(columns: list[str], level: str) -> list[str]:
    env_key = {
        "county": "PRO_LOCAL_ADMIN_COUNTY_NAME_FIELD",
        "city": "PRO_LOCAL_ADMIN_CITY_NAME_FIELD",
        "province": "PRO_LOCAL_ADMIN_PROVINCE_NAME_FIELD",
    }[level]
    env_fields = _split_env_paths(os.getenv(env_key, "")) + _split_env_paths(os.getenv("PRO_LOCAL_ADMIN_NAME_FIELD", ""))
    defaults = {
        "county": ["县名", "区名", "县", "区", "NAME", "name", "county", "district"],
        "city": ["市名", "市", "地市", "地级市", "NAME", "name", "city"],
        "province": ["省名", "省", "NAME", "name", "province"],
    }[level]
    seen, out = set(), []
    for f in env_fields + defaults:
        if f in columns and f not in seen:
            out.append(f); seen.add(f)
    return out


def _read_admin(level: str):
    if gpd is None:
        raise RuntimeError("geopandas不可用，无法执行本地行政区划预检。")
    p = _admin_path(level)
    if not p:
        raise RuntimeError(f"未找到{level}级行政区划文件。请检查 E:\\Agent_DSM\\xingzhengquhua_shp。")
    gdf = gpd.read_file(p)
    if gdf.empty:
        raise RuntimeError(f"行政区划文件为空：{p}")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    else:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf, p


def _infer_parent_admin_from_geometry(geom, level: str) -> tuple[str | None, str | None]:
    """Infer (province, city) for a county/city geometry by spatial containment.

    Some county-level shapefiles only have county names. This helper prevents a
    county request such as “温江区” from being rejected merely because the county
    table lacks province/city attributes.
    """
    province = None
    city = None
    if geom is None or gpd is None:
        return province, city
    probe = geom.representative_point()
    for parent_level in (["city", "province"] if level == "county" else ["province"]):
        try:
            parent_gdf, _ = _read_admin(parent_level)
            fields = _name_fields(list(parent_gdf.columns), parent_level)
            if not fields:
                fields = [c for c in parent_gdf.columns if c != "geometry"]
            if not fields:
                continue
            mask = parent_gdf.geometry.contains(probe) | parent_gdf.geometry.touches(probe)
            sel = parent_gdf.loc[mask]
            if sel.empty:
                # Some boundary precision mismatches: fall back to max intersection.
                tmp = parent_gdf.copy()
                try:
                    tmp["__area"] = tmp.geometry.intersection(geom).area
                    sel = tmp.sort_values("__area", ascending=False).head(1)
                except Exception:
                    sel = parent_gdf.iloc[[]]
            if sel.empty:
                continue
            row = sel.iloc[0]
            val = ""
            for f in fields:
                v = str(row.get(f, "") or "").strip()
                if v and v.lower() not in {"nan", "none"}:
                    val = v
                    break
            if parent_level == "city":
                city = val or city
            elif parent_level == "province":
                province = val or province
        except Exception:
            continue
    return province, city


def detect_requested_aoi_from_local_admin(request_text: str) -> dict[str, Any]:
    """Detect the most specific AOI in user text using local admin tables.

    Robustness notes:
    - Search county/city/province, and prefer the most specific match.
    - If configured name fields are absent, scan all non-geometry columns.
    - Support user text such as “成都市温江区” where the sample data may cover
      the parent city but the target AOI is a county/district subset.
    """
    text = request_text or ""
    if not text.strip():
        return {"ok": False, "message": "请求文本为空。"}
    candidates: list[dict[str, Any]] = []
    level_weight = {"county": 3000, "city": 2000, "province": 1000}
    for level in ["county", "city", "province"]:
        try:
            gdf, path = _read_admin(level)
        except Exception as exc:
            continue
        fields = _name_fields(list(gdf.columns), level)
        if not fields:
            fields = [c for c in gdf.columns if c != "geometry"]
        # Parent fields can be present in county tables; otherwise geometry-based
        # inference below fills them.
        city_fields = _name_fields(list(gdf.columns), "city")
        prov_fields = _name_fields(list(gdf.columns), "province")
        for field in fields:
            if field not in gdf.columns or field == "geometry":
                continue
            series = gdf[field].dropna().astype(str).str.strip()
            for idx, raw in series.items():
                raw = str(raw or "").strip()
                if not raw or raw.lower() in {"nan", "none"} or raw in {"全国", "中国"}:
                    continue
                # Ignore pure code fields.
                if re.fullmatch(r"\d+(\.0)?", raw):
                    continue
                aliases = [raw]
                short = _strip_admin_suffix(raw)
                if short and short != raw and len(short) >= 2:
                    aliases.append(short)
                # A row named “温江” should match user text “温江区”; and vice versa.
                expanded = []
                for a in aliases:
                    expanded.append(a)
                    if level == "county" and len(a) >= 2:
                        expanded.extend([a + "区", a + "县", a + "市", a + "旗"])
                    elif level == "city" and len(a) >= 2:
                        expanded.append(a + "市")
                    elif level == "province" and len(a) >= 2:
                        expanded.append(a + "省")
                aliases = list(dict.fromkeys([a for a in expanded if a]))
                hit = None
                for a in aliases:
                    if a and a in text:
                        hit = a
                        break
                if not hit:
                    continue
                row_gdf = gdf.loc[[idx]]
                row = row_gdf.iloc[0]
                city = None
                province = None
                for cf in city_fields:
                    if cf in gdf.columns:
                        v = str(row.get(cf, "") or "").strip()
                        if v and v.lower() not in {"nan", "none"}:
                            city = v
                            break
                for pf in prov_fields:
                    if pf in gdf.columns:
                        v = str(row.get(pf, "") or "").strip()
                        if v and v.lower() not in {"nan", "none"}:
                            province = v
                            break
                if not city or not province:
                    try:
                        p2, c2 = _infer_parent_admin_from_geometry(row_gdf.geometry.iloc[0], level)
                        province = province or p2
                        city = city or c2
                    except Exception:
                        pass
                candidates.append({
                    "ok": True,
                    "level": level,
                    "region": raw,
                    "county": raw if level == "county" else None,
                    "city": city if city else (raw if level == "city" else None),
                    "province": province if province else (raw if level == "province" else None),
                    "path": str(path),
                    "field": field,
                    "matched_alias": hit,
                    "geometry_index": int(idx) if isinstance(idx, int) else str(idx),
                    "score": level_weight[level] + len(hit),
                })
    if not candidates:
        return {"ok": False, "message": "未能从用户指令中匹配到本地行政区划名称。"}
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[0]

def _points_from_samples(sample_path: str | Path):
    if pd is None:
        raise RuntimeError("pandas不可用，无法读取样点。")
    samples, _, meta = load_sample_points(sample_path)
    if not {"lon", "lat"}.issubset(set(samples.columns)):
        raise RuntimeError("样点缺少 lon/lat 字段。")
    work = samples[["lon", "lat"]].copy()
    work["lon"] = pd.to_numeric(work["lon"], errors="coerce")
    work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
    work = work.dropna(subset=["lon", "lat"])
    if work.empty:
        raise RuntimeError("样点坐标无有效数值。")
    pts = gpd.GeoDataFrame(work, geometry=gpd.points_from_xy(work["lon"], work["lat"]), crs="EPSG:4326")
    return pts, meta


def _dominant_admin_for_points(pts, level: str) -> dict[str, Any]:
    gdf, path = _read_admin(level)
    fields = _name_fields(list(gdf.columns), level)
    if not fields:
        return {"ok": False, "message": f"{path} 缺少{level}名称字段。"}
    name_field = fields[0]
    use = gdf[[name_field, "geometry"]].copy()
    joined = gpd.sjoin(pts, use, how="left", predicate="within")
    vc = joined[name_field].fillna("未匹配").astype(str).value_counts()
    if vc.empty:
        return {"ok": False, "path": str(path), "field": name_field, "message": "空间叠加无结果。"}
    name = str(vc.index[0])
    count = int(vc.iloc[0])
    total = int(len(pts))
    ratio = count / max(total, 1)
    return {
        "ok": name != "未匹配",
        "level": level,
        "region": name if name != "未匹配" else None,
        "count": count,
        "total": total,
        "ratio": float(ratio),
        "path": str(path),
        "field": name_field,
        "counts": {str(k): int(v) for k, v in vc.head(10).items()},
    }


def _requested_geometry(aoi: dict[str, Any]):
    level = str(aoi.get("level") or "")
    if level not in {"county", "city", "province"}:
        return None, None
    gdf, path = _read_admin(level)
    field = aoi.get("field")
    region = str(aoi.get("region") or "")
    if field not in gdf.columns:
        fields = _name_fields(list(gdf.columns), level)
        field = fields[0] if fields else None
    if not field:
        return None, path
    mask = gdf[field].astype(str).str.strip().map(lambda x: _same_admin_name(x, region))
    sel = gdf.loc[mask]
    if sel.empty:
        return None, path
    return sel.geometry.unary_union, path


def preflight_mapping_region(request_text: str, uploaded_files: list[dict] | None) -> dict[str, Any]:
    """Return {blocked, popup/title/message, ...} before starting mapping.

    Blocking rule:
    - If target province differs from the dominant sample province: block.
    - If target is city/county and target city differs from dominant sample city, block.
    - If target AOI has zero points but parent province/city is compatible: warn, do not block.
    """
    if os.getenv("PRO_STRICT_AOI_PREFLIGHT", "1") != "1":
        return {"ok": True, "blocked": False, "message": "严格AOI预检未启用。"}
    if gpd is None:
        return {"ok": False, "blocked": True, "popup": True, "title": "行政区划预检失败", "message": "geopandas不可用，无法在制图前判断样点与AOI是否匹配。"}
    sample_path = None
    for f in (uploaded_files or []):
        p = str(f.get("path") or "")
        if p.lower().endswith((".csv", ".txt", ".xls", ".xlsx")):
            sample_path = p
            break
    if not sample_path:
        fallback = os.getenv("PRO_SAMPLE_PATH") or os.getenv("PRO_TEST_SAMPLE_PATH")
        sample_path = fallback or None
    if not sample_path:
        return {"ok": True, "blocked": False, "message": "未检测到上传样点，跳过AOI-样点预检。"}

    aoi = detect_requested_aoi_from_local_admin(request_text)
    if not aoi.get("ok"):
        if os.getenv("PRO_LOCAL_ADMIN_REQUIRE", "1") == "1":
            return {"ok": False, "blocked": True, "popup": True, "title": "未识别到制图区域", "message": "没有在本地行政区划中识别到用户指定的制图区域。请明确写出省/市/区县名称，例如：2021年浙江省土壤有机质图。"}
        return {"ok": True, "blocked": False, "message": aoi.get("message")}

    pts, sample_meta = _points_from_samples(sample_path)
    sample_prov = _dominant_admin_for_points(pts, "province")
    sample_city = _dominant_admin_for_points(pts, "city")
    min_dom = float(os.getenv("PRO_AOI_PREFLIGHT_DOMINANCE_RATIO", "0.50"))

    target_level = str(aoi.get("level") or "")
    target_region = str(aoi.get("region") or "")
    target_prov = str(aoi.get("province") or "")
    target_city = str(aoi.get("city") or "")
    sample_prov_name = str(sample_prov.get("region") or "")
    sample_city_name = str(sample_city.get("region") or "")

    # Province mismatch: Chengdu/Sichuan samples cannot map Zhejiang.
    if target_prov and sample_prov.get("ok") and sample_prov.get("ratio", 0) >= min_dom and not _same_admin_name(target_prov, sample_prov_name):
        return {
            "ok": True,
            "blocked": True,
            "popup": True,
            "title": "样点与制图区域不匹配",
            "message": f"上传样点主要落在{sample_prov_name}，但用户指定制图区域为{target_region}（{target_prov}）。二者跨省不匹配，已停止制图。请改为绘制样点所在区域，或上传{target_region}对应样点。",
            "target_aoi": aoi,
            "sample_province": sample_prov,
            "sample_city": sample_city,
        }

    # City/county mismatch within or across provinces: Chengdu samples cannot map Hangzhou/Ning波/浙江某县.
    if target_level in {"city", "county"} and target_city and sample_city.get("ok") and sample_city.get("ratio", 0) >= min_dom and not _same_admin_name(target_city, sample_city_name):
        # Direct county request may be outside a known city; city field from 县.shp is decisive.
        return {
            "ok": True,
            "blocked": True,
            "popup": True,
            "title": "样点与制图区域不匹配",
            "message": f"上传样点主要落在{sample_city_name}，但用户指定制图区域属于{target_city}（{target_region}）。二者不属于同一市域，已停止制图。请上传{target_region}或{target_city}样点，或改绘样点所在区域。",
            "target_aoi": aoi,
            "sample_province": sample_prov,
            "sample_city": sample_city,
        }

    geom, geom_path = _requested_geometry(aoi)
    inside_count = None
    total_count = int(len(pts))
    inside_ratio = None
    if geom is not None:
        inside_mask = pts.geometry.within(geom) | pts.geometry.touches(geom)
        inside_count = int(inside_mask.sum())
        inside_ratio = inside_count / max(total_count, 1)
    if inside_count == 0:
        return {
            "ok": True,
            "blocked": False,
            "popup": True,
            "title": "目标区内样点为 0",
            "message": f"用户指定制图区域为{target_region}，但该AOI内没有检测到上传样点。父级区域未发现硬冲突，因此允许继续，但结果属于区内外推，不能直接作为正式交付图。",
            "target_aoi": aoi,
            "sample_province": sample_prov,
            "sample_city": sample_city,
            "inside_count": inside_count,
            "total_count": total_count,
            "inside_ratio": inside_ratio,
        }

    return {
        "ok": True,
        "blocked": False,
        "popup": False,
        "message": f"AOI预检通过：目标区域 {target_region}，样点主要省份 {sample_prov_name or '未知'}，样点主要城市 {sample_city_name or '未知'}。",
        "target_aoi": aoi,
        "sample_province": sample_prov,
        "sample_city": sample_city,
        "inside_count": inside_count,
        "total_count": total_count,
        "inside_ratio": inside_ratio,
    }
