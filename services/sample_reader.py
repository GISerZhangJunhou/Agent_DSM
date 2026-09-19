from __future__ import annotations

"""统一上传样点/栅格读取与校验模块。

放在正式上传服务层，而不是 PRO 诊断层：
- CSV 自动尝试多种编码和分隔符；
- Excel 读取基础字段；
- TIF/TIFF 校验基础空间元数据；
- 返回统一 validation dict，供前端反馈、控制台日志、PRO/RFK流程复用。
"""

import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import rasterio
except Exception:  # pragma: no cover
    rasterio = None

CSV_ENCODINGS = ["utf-8-sig", "utf-8", "gb18030", "gbk", "cp936", "gb2312", "big5", "cp950", "utf-16", "utf-16-le", "utf-16-be", "latin1"]
CSV_SEPARATORS = [None, ",", "\t", ";", "|"]

FIELD_ALIASES = {
    "lon": ["lon", "lng", "long", "longitude", "x", "x_4326", "经度", "东经", "lon_wgs84", "longitude_wgs84", "sample_lon", "样点经度", "采样点经度"],
    "lat": ["lat", "latitude", "y", "y_4326", "纬度", "北纬", "lat_wgs84", "latitude_wgs84", "sample_lat", "样点纬度", "采样点纬度"],
    "som": ["som", "SOM", "som_value", "som_gkg", "som_g_kg", "som_g/kg", "SOM(g/kg)", "soil_organic_matter", "organic_matter", "organic matter", "soil organic matter", "om", "OM", "om_value", "om_gkg", "om_g_kg", "om_g/kg", "OM(g/kg)", "soc", "SOC", "soc_%", "soc_g_kg", "soil_organic_carbon", "organic_carbon", "g/kg", "g C/kg", "有机质", "土壤有机质", "有机质含量", "土壤有机质含量", "有机质g/kg", "有机碳", "土壤有机碳", "SOC_%"],
    "sample_id": ["sample_id", "sampleid", "id", "ID", "fid", "FID", "objectid", "编号", "样点编号", "点号", "样本编号"],
    "year": ["year", "sample_year", "output_year", "年份", "采样年份", "制图年份", "调查年份"],
    "depth": ["depth", "sample_depth", "采样深度", "土层深度", "深度", "depth_cm"],
}



def _valid_year(y: int | None) -> bool:
    if y is None:
        return False
    try:
        min_y = int(os.getenv("PRO_MIN_VALID_YEAR", "1980"))
        max_y = int(os.getenv("PRO_MAX_VALID_YEAR", "2035"))
        return min_y <= int(y) <= max_y
    except Exception:
        return False


def _infer_year_from_filename(path: Path) -> tuple[int | None, str | None]:
    """Infer year from the uploaded filename only, not parent folders."""
    matches = re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", path.name)
    for m in matches:
        y = int(m)
        if _valid_year(y):
            return y, f"uploaded_filename:{path.name}"
    return None, None


def _infer_year_from_table(df: Any, matched: dict[str, str]) -> tuple[int | None, str | None]:
    if pd is None or df is None:
        return None, None
    candidate_cols: list[str] = []
    if matched.get("year"):
        candidate_cols.append(matched["year"])
    for col in list(getattr(df, "columns", [])):
        col_s = str(col).strip()
        col_l = col_s.lower()
        if any(k in col_l or k in col_s for k in ["year", "年份", "采样年份", "调查年份", "date", "日期", "采样时间", "sample_date", "time"]):
            if col_s not in candidate_cols:
                candidate_cols.append(col_s)
    for col in candidate_cols:
        try:
            vals = df[col].dropna().astype(str).head(2000)
        except Exception:
            continue
        if vals.empty:
            continue
        years: list[int] = []
        for v in vals:
            m = re.search(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", v)
            if m:
                y = int(m.group(1))
                if _valid_year(y):
                    years.append(y)
        if not years:
            nums = pd.to_numeric(vals, errors="coerce").dropna().astype(int)
            years = [int(x) for x in nums.tolist() if _valid_year(int(x))]
        if years:
            vc = pd.Series(years).value_counts()
            top_year = int(vc.index[0])
            ratio = float(vc.iloc[0]) / max(len(vals), 1)
            if ratio >= float(os.getenv("PRO_YEAR_FIELD_DOMINANCE_RATIO", "0.8")):
                return top_year, f"sample_table_field:{col}"
    return None, None


def _infer_region_for_upload(df: Any, matched: dict[str, str]) -> dict[str, Any] | None:
    """Infer approximate upload region for user-facing feedback only."""
    if pd is None or df is None:
        return None
    lon_col, lat_col = matched.get("lon"), matched.get("lat")
    if not lon_col or not lat_col:
        return None
    try:
        work = pd.DataFrame({
            "lon": pd.to_numeric(df[lon_col], errors="coerce"),
            "lat": pd.to_numeric(df[lat_col], errors="coerce"),
        }).dropna()
        if work.empty:
            return None
        from services.region_inference_service import infer_region_from_samples
        return infer_region_from_samples(work)
    except Exception as exc:
        return {"ok": False, "message": f"上传区域推断失败：{exc}"}


def _norm_col(x: Any) -> str:
    return str(x).strip().lower().replace(" ", "").replace("_", "")


def _find_field(columns: list[Any], aliases: list[str]) -> str | None:
    direct = {_norm_col(c): c for c in columns}
    for alias in aliases:
        key = _norm_col(alias)
        if key in direct:
            return str(direct[key])
    # 宽松包含匹配：用于“土壤有机质(g/kg)”这类列名。
    for alias in aliases:
        key = _norm_col(alias)
        for col in columns:
            if key and key in _norm_col(col):
                return str(col)
    return None


def _encoding_candidates_for_file(path: Path) -> list[str]:
    """Return ordered encoding candidates, including optional detector hints.

    中文 CSV 常见来源包括 Excel/Windows 的 GBK/GB18030、UTF-8-SIG、Big5/CP950、
    UTF-16 等。不能把第一个能读出来的结果当作正确结果，因为 latin1 等编码
    几乎总能“读成功”，但会产生乱码字段名，导致样点字段识别失败。
    """
    candidates: list[str] = []
    raw = b""
    try:
        raw = path.read_bytes()[:65536]
    except Exception:
        raw = b""
    # BOM 优先。
    if raw.startswith(b"\xef\xbb\xbf"):
        candidates.append("utf-8-sig")
    elif raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        candidates.extend(["utf-16", "utf-16-le", "utf-16-be"])
    # 可选编码探测库：有就用，没有不影响。
    for mod_name in ("charset_normalizer", "chardet"):
        try:
            if mod_name == "charset_normalizer":
                from charset_normalizer import from_bytes  # type: ignore
                best = from_bytes(raw).best() if raw else None
                enc = getattr(best, "encoding", None) if best else None
            else:
                import chardet  # type: ignore
                enc = (chardet.detect(raw) or {}).get("encoding") if raw else None
            if enc:
                candidates.append(str(enc).lower().replace("-sig", "-sig"))
        except Exception:
            continue
    candidates.extend(CSV_ENCODINGS)
    seen: set[str] = set()
    out: list[str] = []
    alias = {"gb2312": "gb18030", "gb_18030": "gb18030", "windows-936": "gbk", "ms936": "gbk", "ascii": "utf-8"}
    for enc in candidates:
        e = alias.get(str(enc).strip().lower(), str(enc).strip().lower())
        if e and e not in seen:
            out.append(e)
            seen.add(e)
    return out


def _text_mojibake_penalty(text: str) -> float:
    if not text:
        return 0.0
    bad_chars = "�ÃÂÊËÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿº¹²³¼½¾¿"
    bad = sum(text.count(ch) for ch in bad_chars)
    # GBK 被 latin1/cp1252 误读时，字段名里会出现大量高位拉丁符号。
    high_latin = sum(1 for ch in text if 0x00A0 <= ord(ch) <= 0x00FF)
    replacement = text.count("\ufffd")
    return bad * 8.0 + high_latin * 2.0 + replacement * 20.0


def _csv_candidate_score(df: Any, encoding: str, sep: Any) -> tuple[float, dict[str, Any]]:
    columns = [str(c).strip() for c in list(getattr(df, "columns", []))]
    header_text = "|".join(columns)
    matched = _match_sample_fields(columns)
    core_hits = sum(1 for k in ("lon", "lat", "som") if matched.get(k))
    score = 0.0
    score += min(len(columns), 20) * 0.8
    score += core_hits * 40.0
    if matched.get("lon") and matched.get("lat"):
        score += 30.0
    if matched.get("som"):
        score += 40.0
    cjk = sum(1 for ch in header_text if "\u4e00" <= ch <= "\u9fff")
    score += min(cjk, 30) * 2.5
    penalty = _text_mojibake_penalty(header_text)
    score -= penalty
    if len(columns) <= 1:
        score -= 80.0
    if len(columns) == 1 and any(s in columns[0] for s in [",", "\t", ";", "|"]):
        score -= 120.0
    # 检查核心字段数值可解析性，避免“字段名勉强匹配但内容乱码/错列”。
    numeric_valid: dict[str, int] = {}
    if pd is not None and hasattr(df, "__getitem__"):
        for key, col in matched.items():
            if key not in {"lon", "lat", "som"}:
                continue
            try:
                vals = pd.to_numeric(
                    df[col].astype(str).str.replace("%", "", regex=False).str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False),
                    errors="coerce",
                )
                valid = int(vals.notna().sum())
                numeric_valid[key] = valid
                if valid > 0:
                    score += min(valid, 50) * 0.25
            except Exception:
                pass
    # latin1 只作为最后兜底；如果没有完整核心字段，强烈降权。
    if str(encoding).lower() in {"latin1", "latin-1", "iso-8859-1"} and core_hits < 3:
        score -= 100.0
    meta = {
        "matched_fields": matched,
        "core_hits": core_hits,
        "mojibake_penalty": round(float(penalty), 3),
        "numeric_valid": numeric_valid,
        "column_count": len(columns),
        "sep": "auto" if sep is None else sep,
    }
    return score, meta


def _read_csv_with_fallback(path: Path, nrows: int | None = None) -> tuple[Any, dict[str, Any]]:
    candidates = _encoding_candidates_for_file(path)
    if pd is None:
        errors = []
        best = None
        for enc in candidates:
            try:
                with path.open("r", encoding=enc, newline="") as f:
                    sample = f.read(8192)
                    f.seek(0)
                    try:
                        dialect = csv.Sniffer().sniff(sample)
                        sep = dialect.delimiter
                    except Exception:
                        sep = ","
                    reader = csv.reader(f, delimiter=sep)
                    columns = next(reader)
                    rows = list(reader) if nrows is None else [row for _, row in zip(range(nrows), reader)]
                table = {"columns": columns, "rows": rows}
                score, score_meta = _csv_candidate_score(type("Mini", (), {"columns": columns})(), enc, sep)
                if best is None or score > best[0]:
                    best = (score, table, {"encoding": enc, "sep": sep, "engine": "csv", "encoding_score": score, "encoding_diagnostics": score_meta, "attempted_encodings": candidates})
            except Exception as exc:
                errors.append(f"{enc}: {exc}")
        if best is not None:
            return best[1], best[2]
        raise RuntimeError("CSV读取失败，已尝试编码：" + "; ".join(errors[-8:]))

    errors: list[str] = []
    attempts: list[dict[str, Any]] = []
    best_df = None
    best_meta: dict[str, Any] | None = None
    best_score = float("-inf")
    for enc in candidates:
        for sep in CSV_SEPARATORS:
            try:
                kwargs: dict[str, Any] = {"encoding": enc, "nrows": nrows}
                if sep is None:
                    kwargs.update({"sep": None, "engine": "python"})
                else:
                    kwargs.update({"sep": sep})
                try:
                    df = pd.read_csv(path, **kwargs)
                except UnicodeDecodeError:
                    raise
                except TypeError:
                    df = pd.read_csv(path, encoding=enc, nrows=nrows, sep=sep or ",")
                if df is None or (getattr(df, "empty", False) and len(list(getattr(df, "columns", []))) == 0):
                    raise ValueError("读取得到空表")
                df.columns = [str(c).strip() for c in df.columns]
                columns = [str(c) for c in df.columns]
                if len(columns) == 1 and any(s in columns[0] for s in [",", "\t", ";", "|"]):
                    # 不直接失败，参与评分但会被强烈降权。
                    pass
                score, score_meta = _csv_candidate_score(df, enc, sep)
                attempts.append({
                    "encoding": enc,
                    "sep": "auto" if sep is None else sep,
                    "score": round(float(score), 3),
                    "core_hits": score_meta.get("core_hits"),
                    "mojibake_penalty": score_meta.get("mojibake_penalty"),
                    "columns": columns[:12],
                })
                if score > best_score:
                    best_score = score
                    best_df = df
                    best_meta = {
                        "encoding": enc,
                        "sep": "auto" if sep is None else sep,
                        "engine": "pandas",
                        "encoding_score": round(float(score), 3),
                        "encoding_diagnostics": score_meta,
                    }
            except Exception as exc:
                errors.append(f"encoding={enc}, sep={sep!r}: {exc}")
                continue
    if best_df is not None and best_meta is not None:
        attempts_sorted = sorted(attempts, key=lambda x: float(x.get("score") or -999999), reverse=True)[:8]
        best_meta["attempted_encodings"] = candidates
        best_meta["encoding_attempts_top"] = attempts_sorted
        best_meta["encoding_warning"] = None
        diag = best_meta.get("encoding_diagnostics") or {}
        if diag.get("core_hits", 0) < 3:
            best_meta["encoding_warning"] = "已完成多编码读取，但核心字段未完全识别；请核查字段名是否包含经度、纬度和有机质/SOM。"
        return best_df, best_meta
    raise RuntimeError("CSV读取失败，已尝试多种编码和分隔符。最后错误：" + (errors[-1] if errors else "未知错误"))


def _match_sample_fields(columns: list[Any]) -> dict[str, str]:
    matched: dict[str, str] = {}
    for key, aliases in FIELD_ALIASES.items():
        hit = _find_field(columns, aliases)
        if hit:
            matched[key] = hit
    return matched


def _numeric_extract_for_validation(series: Any):
    if pd is None:
        return series
    return pd.to_numeric(
        series.astype(str).str.replace("％", "%", regex=False).str.replace("%", "", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False),
        errors="coerce",
    )


def validate_tabular_sample(path: str | Path, preview_rows: int = 5) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"ok": False, "stage": "file_exists", "message": f"文件不存在：{p}"}
    if p.stat().st_size == 0:
        return {"ok": False, "stage": "file_size", "message": "文件为空，无法读取。", "path": str(p)}

    suffix = p.suffix.lower()
    try:
        if suffix in {".csv", ".txt"}:
            df, meta = _read_csv_with_fallback(p, nrows=None)
            if pd is not None:
                columns = list(df.columns)
                row_count = int(len(df))
                preview = df.head(preview_rows).astype(str).to_dict(orient="records")
            else:
                columns = df["columns"]
                row_count = len(df["rows"])
                preview = [dict(zip(columns, row)) for row in df["rows"][:preview_rows]]
        elif suffix in {".xls", ".xlsx"}:
            if pd is None:
                return {"ok": False, "stage": "read", "message": "当前环境缺少 pandas，不能读取 Excel。", "path": str(p)}
            df = pd.read_excel(p)
            meta = {"encoding": None, "sep": None, "engine": "pandas_excel"}
            columns = list(df.columns)
            row_count = int(len(df))
            preview = df.head(preview_rows).astype(str).to_dict(orient="records")
        else:
            return {"ok": False, "stage": "file_type", "message": f"不支持的表格类型：{suffix}", "path": str(p)}
    except Exception as exc:
        return {
            "ok": False,
            "stage": "read",
            "path": str(p),
            "file_type": suffix.lstrip("."),
            "message": f"表格读取失败：{exc}",
            "attempted_encodings": _encoding_candidates_for_file(p) if suffix in {".csv", ".txt"} else None,
        }

    problems: list[str] = []
    warnings: list[str] = []
    if row_count <= 0:
        problems.append("表格没有数据行。")
    if len(columns) <= 0:
        problems.append("表格没有字段列。")

    matched = _match_sample_fields(columns)
    missing_core = [k for k in ["lon", "lat", "som"] if k not in matched]
    if missing_core:
        problems.append("缺少核心字段：" + "、".join(missing_core) + "。建议包含 lon/lat/som 或 经度/纬度/土壤有机质。")

    if pd is not None and 'df' in locals() and hasattr(df, "__getitem__"):
        for key in ["lon", "lat", "som"]:
            col = matched.get(key)
            if col:
                vals = _numeric_extract_for_validation(df[col])
                invalid = int(vals.isna().sum())
                valid = int(vals.notna().sum())
                if valid == 0:
                    problems.append(f"字段 {col} 不能解析为数值。")
                elif invalid > 0:
                    warnings.append(f"字段 {col} 有 {invalid} 条无法解析为数值。")
        lon_col, lat_col = matched.get("lon"), matched.get("lat")
        if lon_col and lat_col:
            lon = _numeric_extract_for_validation(df[lon_col])
            lat = _numeric_extract_for_validation(df[lat_col])
            illegal = int(((lon < -180) | (lon > 180) | (lat < -90) | (lat > 90)).sum())
            if illegal > 0:
                warnings.append(f"检测到 {illegal} 条坐标超出经纬度合法范围，请确认是否为投影坐标。")
        som_col = matched.get("som")
        if som_col:
            y = _numeric_extract_for_validation(df[som_col])
            if int(y.notna().sum()) >= 5:
                q1, q3 = y.quantile(0.25), y.quantile(0.75)
                iqr = q3 - q1
                if isinstance(iqr, (int, float)) and math.isfinite(float(iqr)) and iqr > 0:
                    outlier_count = int(((y < q1 - 3 * iqr) | (y > q3 + 3 * iqr)).sum())
                    if outlier_count:
                        warnings.append(f"检测到 {outlier_count} 个可能异常 SOM 值，建议核查。")

    inferred_year, inferred_year_source = _infer_year_from_filename(p)
    if inferred_year is None and pd is not None and 'df' in locals() and hasattr(df, "__getitem__"):
        inferred_year, inferred_year_source = _infer_year_from_table(df, matched)

    region_inference = None
    if pd is not None and 'df' in locals() and hasattr(df, "__getitem__"):
        region_inference = _infer_region_for_upload(df, matched)

    ok = not problems
    return {
        "ok": ok,
        "stage": "validated" if ok else "field_validation",
        "path": str(p),
        "file_type": suffix.lstrip("."),
        "size_bytes": p.stat().st_size,
        "encoding": meta.get("encoding"),
        "separator": meta.get("sep"),
        "encoding_score": meta.get("encoding_score"),
        "encoding_warning": meta.get("encoding_warning"),
        "attempted_encodings": meta.get("attempted_encodings"),
        "encoding_attempts_top": meta.get("encoding_attempts_top"),
        "rows": row_count,
        "columns": [str(c) for c in columns],
        "matched_fields": matched,
        "inferred_year": inferred_year,
        "inferred_year_source": inferred_year_source,
        "region_inference": region_inference,
        "missing_core_fields": missing_core,
        "problems": problems,
        "warnings": warnings,
        "preview": preview,
        "message": "样点表读取并校验通过，可进入建模流程。" if ok else "样点表已读取，但字段/内容校验未通过。",
    }


def validate_raster_file(path: str | Path, check_values: bool = True) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"ok": False, "stage": "file_exists", "message": f"文件不存在：{p}"}
    if p.stat().st_size == 0:
        return {"ok": False, "stage": "file_size", "message": "文件为空，无法读取。", "path": str(p)}
    if rasterio is None:
        return {
            "ok": True,
            "stage": "file_exists_only",
            "path": str(p),
            "file_type": "raster",
            "size_bytes": p.stat().st_size,
            "message": "检测到栅格文件，但当前环境未安装 rasterio，仅完成文件存在性检查。",
            "warnings": ["缺少 rasterio，无法检查 CRS、范围、分辨率与 NoData。"],
        }
    try:
        with rasterio.open(p) as src:
            problems: list[str] = []
            warnings: list[str] = []
            if src.width <= 0 or src.height <= 0:
                problems.append("栅格宽高异常。")
            if src.count <= 0:
                problems.append("栅格没有波段。")
            if src.crs is None:
                problems.append("栅格缺少坐标参考系统 CRS。")
            if src.res is None:
                warnings.append("无法读取栅格分辨率。")
            sample_valid_count = None
            if check_values and src.count > 0:
                try:
                    arr = src.read(1, masked=True, out_shape=(1, min(src.height, 512), min(src.width, 512)))
                    sample_valid_count = int(arr.count())
                    if sample_valid_count == 0:
                        problems.append("抽样检查发现栅格全为 NoData 或无有效像元。")
                except Exception as exc:
                    warnings.append(f"像元抽样检查失败：{exc}")
            ok = not problems
            return {
                "ok": ok,
                "stage": "validated" if ok else "raster_validation",
                "path": str(p),
                "file_type": "raster",
                "driver": src.driver,
                "size_bytes": p.stat().st_size,
                "width": int(src.width),
                "height": int(src.height),
                "band_count": int(src.count),
                "crs": str(src.crs) if src.crs else None,
                "bounds": list(src.bounds),
                "resolution": list(src.res),
                "nodata": src.nodata,
                "sample_valid_count": sample_valid_count,
                "problems": problems,
                "warnings": warnings,
                "message": "栅格读取并校验通过，可用于后续分析。" if ok else "栅格能打开，但元数据/有效像元校验未通过。",
            }
    except Exception as exc:
        return {"ok": False, "stage": "read", "path": str(p), "file_type": "raster", "message": f"栅格文件无法打开：{exc}"}


def validate_uploaded_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".csv", ".txt", ".xls", ".xlsx"}:
        return validate_tabular_sample(p)
    if suffix in {".tif", ".tiff"}:
        return validate_raster_file(p)
    return {
        "ok": False,
        "stage": "file_type",
        "path": str(p),
        "file_type": suffix.lstrip(".") or "unknown",
        "message": f"暂不支持该文件类型：{suffix or '无扩展名'}。支持 CSV/Excel/TIF/TIFF。",
    }


def build_upload_feedback(info: dict[str, Any]) -> str:
    name = info.get("name") or Path(str(info.get("path") or "文件")).name
    validation = info.get("validation") or {}
    if not validation:
        return f"文件已接收：{name}。尚未完成读取校验。"
    lines = [f"文件：{name}"]
    if validation.get("ok"):
        lines.append("状态：读取与校验成功。")
    else:
        lines.append("状态：文件已接收，但读取/校验未通过。")
    msg = validation.get("message")
    if msg:
        lines.append("说明：" + str(msg))
    if validation.get("encoding"):
        score = validation.get("encoding_score")
        score_text = f"；置信评分：{score}" if score is not None else ""
        lines.append(f"编码：{validation.get('encoding')}；分隔符：{validation.get('separator')}{score_text}")
    if validation.get("encoding_warning"):
        lines.append("编码提醒：" + str(validation.get("encoding_warning")))
    if validation.get("rows") is not None:
        lines.append(f"数据规模：{validation.get('rows')} 行，{len(validation.get('columns') or [])} 列。")
    if validation.get("matched_fields"):
        mf = validation.get("matched_fields") or {}
        core = []
        for k in ["lon", "lat", "som"]:
            if mf.get(k):
                core.append(f"{k}={mf[k]}")
        if core:
            lines.append("识别字段：" + "，".join(core))
    if validation.get("inferred_year"):
        src = validation.get("inferred_year_source") or "unknown"
        lines.append(f"推断年份：{validation.get('inferred_year')}（来源：{src}）")
    else:
        # V74：上传数据本身无法判断年份时，不阻止上传；正式制图时若用户也没写年份，再要求补充。
        if validation.get("ok") and validation.get("file_type") in {"csv", "excel"}:
            lines.append("推断年份：未识别；请在提问中明确目标年份。")
    ri = validation.get("region_inference") or {}
    if isinstance(ri, dict) and ri.get("region"):
        ratio = ri.get("dominance_ratio")
        try:
            ratio_text = f"，占比 {float(ratio):.3f}" if ratio is not None else ""
        except Exception:
            ratio_text = ""
        level = ri.get("level") or ""
        lines.append(f"推断区域：{ri.get('region')}（{level}{ratio_text}）")
        if ri.get("warning"):
            lines.append("区域提醒：" + str(ri.get("warning")))
    if validation.get("file_type") == "raster":
        lines.append(f"栅格信息：{validation.get('width')}×{validation.get('height')}，波段数={validation.get('band_count')}，CRS={validation.get('crs')}")
    for p in validation.get("problems") or []:
        lines.append("问题：" + str(p))
    for w in validation.get("warnings") or []:
        lines.append("提醒：" + str(w))
    return "\n".join(lines)



def load_sample_points(sample_path: str | Path) -> tuple[Any, dict[str, str], dict[str, Any]]:
    """Read an uploaded SOM sample table and return normalized lon/lat/som columns.

    This lightweight wrapper is intentionally placed in sample_reader because AOI
    preflight runs before the formal PRO pipeline is started. Importing the full
    modeling pipeline here would make startup fragile and can create circular
    imports.
    """
    if pd is None:
        raise RuntimeError("缺少 pandas，无法读取样点表。")
    p = Path(sample_path)
    if not p.exists():
        raise RuntimeError(f"样点文件不存在：{p}")
    suffix = p.suffix.lower()
    if suffix in {".csv", ".txt"}:
        df, meta = _read_csv_with_fallback(p, nrows=None)
    elif suffix in {".xls", ".xlsx"}:
        df = pd.read_excel(p)
        meta = {"encoding": None, "sep": None, "engine": "pandas_excel"}
    else:
        raise RuntimeError(f"不支持的样点表类型：{suffix}")
    fields = _match_sample_fields(list(df.columns))
    missing = [k for k in ["lon", "lat", "som"] if k not in fields]
    if missing:
        raise RuntimeError("样点表缺少核心字段：" + "、".join(missing) + "。需要 lon/lat/som 或 经度/纬度/土壤有机质。")

    def _to_num(series: Any):
        return pd.to_numeric(
            series.astype(str)
            .str.replace("%", "", regex=False)
            .str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False),
            errors="coerce",
        )

    out = pd.DataFrame()
    out["lon"] = _to_num(df[fields["lon"]])
    out["lat"] = _to_num(df[fields["lat"]])
    out["som"] = _to_num(df[fields["som"]])
    if fields.get("sample_id"):
        out["sample_id"] = df[fields["sample_id"]].astype(str)
    if fields.get("year"):
        out["year"] = _to_num(df[fields["year"]])
    out = out.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=["lon", "lat", "som"]).copy()
    if out.empty:
        raise RuntimeError("样点表读取成功，但 lon/lat/som 转数值后没有有效记录。")
    meta = dict(meta or {})
    inferred_year, inferred_year_source = _infer_year_from_filename(p)
    if inferred_year is None:
        inferred_year, inferred_year_source = _infer_year_from_table(df, fields)
    if inferred_year is not None:
        meta["inferred_year_from_table"] = int(inferred_year)
        meta["inferred_year_source"] = inferred_year_source
    meta.update({"matched_fields": fields, "valid_rows": int(len(out)), "source": "services.sample_reader.load_sample_points"})
    return out, fields, meta
