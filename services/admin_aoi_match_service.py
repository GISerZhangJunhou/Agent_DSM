# -*- coding: utf-8 -*-
"""Built-in administrative boundary AOI matching service.

This module is used by the formal mapping agent to resolve user AOI instructions
against local administrative boundary data. The built-in project convention is:

    E:\\Agent_DSM\\行政区划

The folder is expected to contain province/city/county shapefiles such as
省.shp, 市.shp and 县.shp. Users should not need to upload administrative
boundaries for routine mapping tasks.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None

ADMIN_ROOT = Path(os.getenv("PRO_LOCAL_ADMIN_VECTOR_DIR", r"E:\Agent_DSM\行政区划"))

LEVEL_FILES = {
    "province": ["省.shp", "省级.shp", "省级行政区.shp"],
    "city": ["市.shp", "市级.shp", "市级行政区.shp", "地市.shp"],
    "county": ["县.shp", "县级.shp", "区县.shp", "县级行政区.shp", "区县行政区.shp"],
}

NAME_FIELDS = {
    "province": ["省名", "省", "NAME", "name", "province", "省级"],
    "city": ["市名", "市", "地市", "地级市", "NAME", "name", "city"],
    "county": ["县名", "区名", "县", "区", "县级", "区县", "NAME", "name", "county", "district"],
}

ADMIN_SUFFIXES = [
    "特别行政区", "壮族自治区", "回族自治区", "维吾尔自治区", "自治区",
    "自治州", "地区", "盟", "省", "市", "县", "区", "旗",
]


@dataclass
class AOIMatchResult:
    ok: bool
    level: str | None = None
    name: str | None = None
    display_name: str | None = None
    path: str | None = None
    name_field: str | None = None
    feature_count: int = 0
    message: str = ""
    candidates: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_admin_name(name: str | None) -> str:
    s = str(name or "").strip()
    s = re.sub(r"[\s　]+", "", s)
    s = s.replace("（", "(").replace("）", ")")
    for suf in ADMIN_SUFFIXES:
        if s.endswith(suf) and len(s) > len(suf):
            return s[: -len(suf)]
    return s


def same_admin_name(query: str | None, candidate: str | None) -> bool:
    q = str(query or "").strip()
    c = str(candidate or "").strip()
    if not q or not c:
        return False
    qn = normalize_admin_name(q)
    cn = normalize_admin_name(c)
    return q == c or qn == cn or (qn and qn in c) or (cn and cn in q)


def _admin_path(level: str) -> Path | None:
    env_keys = {
        "province": ["PRO_LOCAL_ADMIN_PROVINCE_VECTOR", "PRO_ADMIN_PROVINCE_VECTOR"],
        "city": ["PRO_LOCAL_ADMIN_CITY_VECTOR", "PRO_ADMIN_CITY_VECTOR"],
        "county": ["PRO_LOCAL_ADMIN_COUNTY_VECTOR", "PRO_ADMIN_COUNTY_VECTOR"],
    }.get(level, [])
    for k in env_keys:
        v = os.getenv(k, "").strip().strip('"').strip("'")
        if v and Path(v).exists():
            return Path(v)
    roots = [ADMIN_ROOT, Path(r"E:\Agent_DSM\行政区划"), Path.cwd() / "行政区划", Path.cwd().parent / "行政区划"]
    seen: set[str] = set()
    for root in roots:
        if not root or str(root) in seen:
            continue
        seen.add(str(root))
        if not root.exists():
            continue
        for nm in LEVEL_FILES.get(level, []):
            p = root / nm
            if p.exists():
                return p
        # Fallback: semantic search.
        keywords = {"province": ["省"], "city": ["市", "地市"], "county": ["县", "区县"]}.get(level, [])
        candidates = []
        for p in root.rglob("*.shp"):
            stem = p.stem
            if any(x in stem for x in ["十段线", "九段线", "线", "点"]):
                continue
            score = sum(1 for x in keywords if x in stem)
            if score:
                candidates.append((score, len(p.relative_to(root).parts), p))
        if candidates:
            candidates.sort(key=lambda x: (-x[0], x[1]))
            return candidates[0][2]
    return None


def _read_layer(level: str):
    if gpd is None:
        raise RuntimeError("geopandas 不可用，无法读取内置行政区划数据。")
    p = _admin_path(level)
    if not p:
        raise RuntimeError(f"未找到{level}级内置行政区划文件，请检查 {ADMIN_ROOT}。")
    gdf = gpd.read_file(p)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    else:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf, p


def _find_name_field(gdf, level: str) -> str | None:
    cols = [c for c in gdf.columns if c != "geometry"]
    for f in NAME_FIELDS.get(level, []):
        if f in cols:
            return f
    # Prefer text-like columns with non-empty values.
    for c in cols:
        try:
            sample = gdf[c].dropna().astype(str).head(5).tolist()
            if sample and any(re.search(r"[\u4e00-\u9fffA-Za-z]", x) for x in sample):
                return c
        except Exception:
            continue
    return cols[0] if cols else None


def match_aoi_from_text(text: str) -> dict[str, Any]:
    """Resolve AOI mentioned in user instruction to built-in admin boundary.

    Search order is county -> city -> province so the most specific AOI wins.
    """
    text = text or ""
    candidates: list[dict[str, Any]] = []
    for level in ["county", "city", "province"]:
        try:
            gdf, path = _read_layer(level)
            field = _find_name_field(gdf, level)
            if not field:
                continue
            for idx, row in gdf.iterrows():
                raw = str(row.get(field, "") or "").strip()
                if not raw:
                    continue
                if same_admin_name(text, raw):
                    score = {"county": 30, "city": 20, "province": 10}[level]
                    # Longer exact names in text get a small boost.
                    if raw in text:
                        score += 5
                    candidates.append({
                        "level": level,
                        "name": raw,
                        "display_name": raw,
                        "path": str(path),
                        "name_field": field,
                        "index": int(idx) if isinstance(idx, int) else str(idx),
                        "score": score,
                    })
        except Exception as exc:
            candidates.append({"level": level, "error": str(exc), "score": -1})
    valid = [x for x in candidates if x.get("score", 0) > 0]
    if not valid:
        return AOIMatchResult(False, message="未从内置行政区划中匹配到用户指定的制图区域。", candidates=candidates[-8:]).to_dict()
    valid.sort(key=lambda x: x.get("score", 0), reverse=True)
    best = valid[0]
    return AOIMatchResult(
        True,
        level=best.get("level"),
        name=best.get("name"),
        display_name=best.get("display_name"),
        path=best.get("path"),
        name_field=best.get("name_field"),
        feature_count=1,
        message=f"已匹配内置行政区划：{best.get('display_name')}（{best.get('level')}）。",
        candidates=valid[:10],
    ).to_dict()


def write_aoi_match_report(text: str, out_path: str | Path) -> dict[str, Any]:
    result = match_aoi_from_text(text)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
