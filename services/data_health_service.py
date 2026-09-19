from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

SOM_CANDIDATES = [
    "som", "SOM", "som_value", "som_gkg", "som_g_kg",
    "om", "OM", "om_value", "om_gkg", "om_g_kg",
    "soil_organic_matter", "organic_matter", "organic matter", "soil organic matter",
    "有机质", "有机质含量", "土壤有机质", "土壤有机质含量",
    "soc", "SOC", "soil_organic_carbon", "有机碳", "土壤有机碳",
]
LON_CANDIDATES = ["lon", "lon_wgs84", "longitude", "lng", "long", "经度", "东经", "x", "x_4326", "sample_lon"]
LAT_CANDIDATES = ["lat", "lat_wgs84", "latitude", "纬度", "北纬", "y", "y_4326", "sample_lat"]


def _norm_col(x):
    import re
    return re.sub(r"[\s_\-()（）\[\]【】{}:：/\\.%％]+", "", str(x or "").strip()).lower()


def _find_col(cols, candidates):
    norm_map = {}
    for c in cols:
        norm_map.setdefault(_norm_col(c), c)
    for c in candidates:
        key = _norm_col(c)
        if key in norm_map:
            return norm_map[key]
    # Contains fallback only for aliases longer than 2 chars; OM/x/y are exact-only.
    keys = sorted({_norm_col(c) for c in candidates if len(_norm_col(c)) > 2}, key=len, reverse=True)
    for raw in cols:
        nr = _norm_col(raw)
        for key in keys:
            if key and key in nr:
                return raw
    return None


def _to_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace("％", "%", regex=False).str.replace("%", "", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False),
        errors="coerce",
    )


def inspect_csv(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"ok": False, "message": f"文件不存在：{p}"}
    try:
        from services.sample_reader import _read_csv_with_fallback
        df, meta = _read_csv_with_fallback(p, nrows=5000)
    except Exception as e:
        return {"ok": False, "message": f"CSV 读取失败：{e}"}

    som_col = _find_col(df.columns, SOM_CANDIDATES)
    lon_col = _find_col(df.columns, LON_CANDIDATES)
    lat_col = _find_col(df.columns, LAT_CANDIDATES)
    problems = []
    warnings = []
    if not som_col:
        problems.append("缺少 SOM/OM/有机质字段，可用字段名如 OM、SOM、som、有机质、土壤有机质、organic_matter。")
    if not lon_col or not lat_col:
        problems.append("缺少经纬度字段，可用字段名如 lon/lat、经度/纬度。")
    else:
        lon = _to_numeric(df[lon_col])
        lat = _to_numeric(df[lat_col])
        bad_coord = int((lon.isna() | lat.isna()).sum())
        if bad_coord:
            warnings.append(f"有 {bad_coord} 条坐标无法解析。")
        # 成都大致范围，避免误用投影坐标或外地坐标。
        in_cd = lon.between(102.8, 105.2) & lat.between(29.7, 31.6)
        if in_cd.notna().any() and float(in_cd.mean()) < 0.6:
            warnings.append("多数坐标不在成都附近，请确认经纬度坐标系或研究区。")
    if som_col:
        y = _to_numeric(df[som_col])
        miss_rate = float(y.isna().mean()) if len(y) else 1.0
        if miss_rate > 0.2:
            warnings.append(f"有机质字段缺失率较高：{miss_rate:.1%}。")
        if y.notna().sum() > 5:
            q1, q3 = y.quantile(0.25), y.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outliers = int(((y < q1 - 3 * iqr) | (y > q3 + 3 * iqr)).sum())
                if outliers:
                    warnings.append(f"检测到 {outliers} 个可能异常有机质值，建议核查。")
    ok = not problems
    return {
        "ok": ok,
        "rows_checked": int(len(df)),
        "encoding": meta.get("encoding"),
        "separator": meta.get("sep"),
        "encoding_score": meta.get("encoding_score"),
        "som_col": str(som_col) if som_col else None,
        "lon_col": str(lon_col) if lon_col else None,
        "lat_col": str(lat_col) if lat_col else None,
        "problems": problems,
        "warnings": warnings,
        "message": "数据体检通过。" if ok else "数据体检未通过。",
    }


def build_health_text(report: dict[str, Any]) -> str:
    lines = ["数据体检结果：" + str(report.get("message", ""))]
    if report.get("encoding"):
        extra = f"，分隔符：{report.get('separator')}" if report.get("separator") else ""
        lines.append(f"- CSV编码：{report.get('encoding')}{extra}")
    if report.get("som_col"):
        lines.append(f"- 有机质字段：{report['som_col']}")
    if report.get("lon_col") and report.get("lat_col"):
        lines.append(f"- 坐标字段：{report['lon_col']} / {report['lat_col']}")
    for x in report.get("problems") or []:
        lines.append("- 问题：" + str(x))
    for x in report.get("warnings") or []:
        lines.append("- 提醒：" + str(x))
    return "\n".join(lines)
