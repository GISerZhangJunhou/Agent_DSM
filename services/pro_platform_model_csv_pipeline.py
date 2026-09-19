from __future__ import annotations

"""
正式版主线：用户上传样点 -> 国内平台/人工接管下载环境协变量 -> 样点协变量抽取 -> 建模CSV -> 模型训练 -> GeoTIFF制图输出。

目标是形成可交付的最小正式版：样点由用户上传，环境协变量由 GEE 自动获取并直接用于建模。

状态定义：
- reachable: 平台页面/API可访问。
- link_found: 发现候选下载链接。
- file_downloaded: 至少下载了文件。
- raster_standardized: 至少一个栅格被统一到目标 CRS/分辨率/样点范围。
- sample_aligned: 至少一个变量成功抽取到样点表。

正式模式下不再使用坐标伪特征或平台状态伪特征兜底。
V88 起增加“GEE缺失协变量 -> 国内平台补充”分支：GEE无法稳定获取的灌溉、复种指数、耕地利用强度、作物种植模式等变量，可继续尝试从国内平台直链、人工登录捕获文件或本地缓存目录接入；只有真正标准化并抽样成功的栅格才进入正式特征。
"""

import hashlib
import json
import math
import os
import re
import shutil
import time
import secrets
import zipfile
import itertools
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from services.landcover_scope_service import infer_landcover_scope, CLCD_CLASS_LABELS
from urllib.parse import quote, urljoin, urlparse, parse_qsl

import requests

from utils.pro_console import pro_console_log
from services.pro_mainline_diagnostic_model import read_sample_table, infer_core_fields, FIELD_ALIASES
from services.region_inference_service import (
    detect_region_from_text, infer_region_from_samples, region_conflict, apply_region_to_target,
    CITY_BBOXES, PROVINCE_BBOXES
)


def _public_six_digit_run_code() -> str:
    try:
        return f"{secrets.randbelow(900000) + 100000:06d}"
    except Exception:
        return f"{int(time.time() * 1000) % 900000 + 100000:06d}"

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import from_bounds, transform as window_transform
    from rasterio.features import rasterize
except Exception:  # pragma: no cover
    rasterio = None
    Resampling = None
    WarpedVRT = None
    from_bounds = None
    window_transform = None
    rasterize = None

try:
    from pyproj import Transformer
except Exception:  # pragma: no cover
    Transformer = None

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold, GroupShuffleSplit, cross_val_predict
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.base import clone
except Exception:  # pragma: no cover
    RandomForestRegressor = None
    KFold = None
    GroupShuffleSplit = None
    cross_val_predict = None
    r2_score = None
    mean_squared_error = None
    mean_absolute_error = None
    clone = None

try:
    from pykrige.ok import OrdinaryKriging
except Exception:  # pragma: no cover
    OrdinaryKriging = None


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 "
        "DSM-Pro-PlatformPipeline/0.1"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
}

DOWNLOAD_EXTS = (".zip", ".rar", ".7z", ".xls", ".xlsx", ".csv", ".tif", ".tiff", ".nc", ".hdf", ".h5", ".hdf5", ".json", ".geojson")
GEE_OUTPUT_MASK_COLUMNS = {"cov_gee_admin_mask", "cov_gee_cropland_mask", "cov_gee_worldcover_class"}
DOMESTIC_MANUAL_REQUIRED_MARKER = "DOMESTIC_MANUAL_REQUIRED_JSON::"
# 可以直接被 GDAL/rasterio 打开的栅格或子数据集容器。
RASTER_EXTS = {".tif", ".tiff", ".hdf", ".h5", ".hdf5", ".nc"}
TABLE_EXTS = {".csv", ".xls", ".xlsx"}
ARCHIVE_EXTS = {".zip", ".7z", ".rar"}


@dataclass
class PlatformSource:
    id: str
    name: str
    url: str
    role: str
    priority: int = 1
    enabled: bool = True
    js_heavy: bool = False
    api_mode: str = "html"  # html | figshare_api | earthdata_cmr
    group: str = "domestic"  # domestic | auxiliary | foreign


@dataclass
class PlatformStatus:
    platform_id: str
    platform_name: str
    url: str
    role: str
    reachable: bool = False
    http_status: int | None = None
    saved_html: str | None = None
    link_count: int = 0
    download_link_count: int = 0
    downloaded_count: int = 0
    html_page_saved_count: int = 0
    true_raw_downloaded_count: int = 0
    processed_raster_count: int = 0
    processed_table_count: int = 0
    sample_aligned_count: int = 0
    status: str = "queued"
    message: str = ""
    warnings: list[str] = field(default_factory=list)
    downloaded_files: list[str] = field(default_factory=list)
    standardized_rasters: list[str] = field(default_factory=list)
    extracted_columns: list[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    ok: bool
    status: str
    work_dir: str
    model_ready_csv: str
    platform_report_csv: str
    platform_report_json: str
    report_md: str
    model_report_json: str | None
    pred_tif: str | None
    statuses: list[dict[str, Any]]
    warnings: list[str]


def _emit(stage: str, message: str, payload: Any | None = None, task_id: str | None = None) -> None:
    """Emit pipeline status to both console and UI task registry.

    Earlier versions only printed PRO-DEBUG lines to PyCharm. The Dash progress
    bar reads TASKS logs, so long in-process RFK runs could finish or fail while
    the UI stayed at 4%. Importing TASKS lazily avoids a module import cycle.
    """
    if task_id:
        try:
            from services.task_service import TASKS  # lazy import: avoid circular import at module load time
            line = f"[{stage}] {message}"
            if payload is not None:
                try:
                    brief = json.dumps(payload, ensure_ascii=False, default=str)
                    if len(brief) > 900:
                        brief = brief[:900] + "..."
                    line += " | " + brief
                except Exception:
                    pass
            TASKS.append_log(task_id, line)
            return
        except Exception:
            pass
    pro_console_log(stage, message, payload=payload, task_id=task_id)


def _safe_name(s: str, max_len: int = 80) -> str:
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", str(s or "x"))
    return s.strip("._-")[:max_len] or "x"


def _path_len(path: Path | str) -> int:
    return len(str(path))


def _short_region_code(region: Any, max_len: int = 10) -> str:
    name = _safe_name(str(region or "aoi"), max_len=max_len)
    # 保留可读区域名，同时限制目录长度；如果区域名太短或为空，使用稳定哈希兜底。
    if name and name != "aoi":
        return name
    digest = hashlib.md5(str(region or "aoi").encode("utf-8", errors="ignore")).hexdigest()[:8]
    return f"aoi_{digest}"


def _compact_run_dir(out_root: Path, region: Any, year: Any, ts: str) -> Path:
    # 用户可见的结果目录应短而清楚；内部 raw/std/meta/out 保持不变。
    region_code = _short_region_code(region, max_len=int(os.getenv("PRO_RUN_REGION_NAME_MAXLEN", "8")))
    year_code = str(year or "y")
    ts_code = ts.replace("_", "")[-4:]
    return out_root / f"{region_code}_{year_code}_{ts_code}"


def parse_request_basic(request_text: str) -> dict[str, Any]:
    text = request_text or ""
    scope_payload = infer_landcover_scope(text)
    map_scope = str(scope_payload.get("map_scope") or "full_domain")
    # Parse requested year conservatively.  Users often type values such as
    # "20220年" when they mean 2020; a plain 4-digit regex would incorrectly
    # read that as 2022 and then reject all 2020 covariates/land-cover rasters.
    year = None
    year_token_match = re.search(r"(?<!\d)((?:19|20)\d{2,3})(?:\s*年)?", text)
    if year_token_match:
        raw_year = year_token_match.group(1)
        try:
            if len(raw_year) == 4:
                year = int(raw_year)
            elif len(raw_year) == 5 and raw_year.startswith("20"):
                # Example: 20220 -> 2020; 20230 -> 2030.
                candidate = int(raw_year[:2] + raw_year[-2:])
                year = candidate if 1980 <= candidate <= 2035 else int(raw_year[:4])
        except Exception:
            year = None
    region_match = detect_region_from_text(text)

    # 分辨率策略：用户明确给出时优先；否则先留空，后续根据用户协变量
    # 自动采用“最粗/最低空间分辨率”。这样不会把 1000 m 气候数据硬插到 250 m
    # 产生虚假细节。
    res_match = re.search(r"(?:分辨率|像元|空间分辨率)[^0-9]{0,12}([0-9]+(?:\.[0-9]+)?)\s*(米|m|M|km|公里)?", text, flags=re.I)
    explicit_res = None
    if res_match:
        explicit_res = float(res_match.group(1))
        unit = (res_match.group(2) or "m").lower()
        if unit in {"km", "公里"}:
            explicit_res *= 1000.0

    target = {
        "year": year,
        "year_source": "request_text" if year else None,
        "region": region_match.region,
        "region_source": region_match.source,
        "region_text_match": region_match.__dict__,
        "resolution_m": explicit_res,
        "resolution_source": "user_request" if explicit_res else "auto_from_user_covariates",
        "target_crs": os.getenv(
            "PRO_TARGET_CRS_CHINA",
            "+proj=aea +lat_0=0 +lon_0=105 +lat_1=25 +lat_2=47 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs",
        ),
        "is_city": False,
        "map_scope": map_scope,
        "output_scope": map_scope,
        "mask_non_cropland": bool(scope_payload.get("mask_by_landcover")),
        "mask_by_landcover": bool(scope_payload.get("mask_by_landcover")),
        "mask_non_target_landcover": bool(scope_payload.get("mask_non_target_landcover")),
        "landcover_class_code": scope_payload.get("landcover_class_code"),
        "landcover_class_label": scope_payload.get("landcover_class_label"),
        "scope_source": scope_payload.get("scope_source") or "request_text_semantic",
    }
    if region_match.region:
        target = apply_region_to_target(target, region_match.region, level=region_match.level, source="request_text")
    return target


def _valid_year(y: int | None) -> bool:
    if y is None:
        return False
    min_y = int(os.getenv("PRO_MIN_VALID_YEAR", "1980"))
    max_y = int(os.getenv("PRO_MAX_VALID_YEAR", "2035"))
    return min_y <= int(y) <= max_y


def infer_year_from_filename(path: str | Path) -> tuple[int | None, str | None]:
    """只从文件名推断年份，避免把目录名里的 20260516 误判为制图年份。"""
    name = Path(path).name
    matches = re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", name)
    years = [int(x) for x in matches if _valid_year(int(x))]
    if years:
        return years[0], f"uploaded_filename:{name}"
    return None, None


def infer_year_from_dataframe(df: Any) -> tuple[int | None, str | None]:
    """从样点表年份/日期字段推断年份。只在年份占比明显时使用，避免误判。"""
    if pd is None or df is None:
        return None, None
    year_field_keywords = ["year", "年份", "采样年份", "样点年份", "调查年份", "date", "日期", "采样时间", "sample_date", "time"]
    for col in list(df.columns):
        col_s = str(col).strip()
        col_l = col_s.lower()
        if not any(k.lower() in col_l or k in col_s for k in year_field_keywords):
            continue
        vals = df[col].dropna().astype(str).head(2000)
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
            # 纯数字年份字段，例如 2021.0
            nums = pd.to_numeric(vals, errors="coerce").dropna().astype(int)
            years = [int(x) for x in nums.tolist() if _valid_year(int(x))]
        if years:
            vc = pd.Series(years).value_counts()
            top_year = int(vc.index[0])
            ratio = float(vc.iloc[0]) / max(len(vals), 1)
            if ratio >= float(os.getenv("PRO_YEAR_FIELD_DOMINANCE_RATIO", "0.8")):
                return top_year, f"sample_table_field:{col_s}"
    return None, None


def default_platform_sources() -> list[PlatformSource]:
    return [
        PlatformSource("geodata", "国家地球系统科学数据中心 GeoData", "https://www.geodata.cn/aboutus.html", "soil_background/climate/geographic_background", priority=1),
        PlatformSource("ngac", "全国地质资料馆 NGAC", "https://www.ngac.cn/125cms/c/qggnew/index.htm", "geology/parent_material", priority=4),
        PlatformSource("gscloud", "地理空间数据云 GSCloud", "https://www.gscloud.cn/sources/index?pid=1&rootid=1", "remote_sensing/dem/landuse", priority=1),
        PlatformSource("tpdc", "国家青藏高原科学数据中心 TPDC", "https://data.tpdc.ac.cn/home", "climate/hydrology/plateau", priority=3, js_heavy=True),
        PlatformSource("nesdc", "国家生态科学数据中心 NESDC", "https://www.nesdc.org.cn/", "soc_background/ecology", priority=2),
        PlatformSource("noda", "国家综合地球观测数据共享平台 NODA", "https://noda.ac.cn/portal/indexSearch?title=MODIS", "earth_observation/remote_sensing", priority=2),
        PlatformSource("osm", "OpenStreetMap / Overpass API", "https://overpass-api.de/api/interpreter", "vector_context/road_water_distance", priority=5, api_mode="osm_overpass", group="auxiliary"),
        PlatformSource("essd", "Earth System Science Data ESSD", "https://www.earth-system-science-data.net/home.html", "dataset_discovery/doi", priority=20, group="foreign"),
        PlatformSource("figshare", "Figshare", "https://api.figshare.com/v2/articles/search", "paper_supplement/doi_dataset", priority=21, api_mode="figshare_api", group="foreign"),
        PlatformSource("earthdata", "NASA Earthdata CMR", "https://cmr.earthdata.nasa.gov/search/collections.json", "modis/climate/earth_observation", priority=22, api_mode="earthdata_cmr", group="foreign"),
    ]


def _norm_schema_name(s: Any) -> str:
    return re.sub(r"[\s_\-()（）\[\]【】{}:：/\\.%％]+", "", str(s or "").strip()).lower()


def _numeric_extract(series: Any):
    return pd.to_numeric(
        series.astype(str)
        .str.replace("％", "%", regex=False)
        .str.replace("%", "", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False),
        errors="coerce",
    )


_COORD_NORMS = {_norm_schema_name(a) for k in ("lon", "lat") for a in FIELD_ALIASES.get(k, [])}
_TARGET_NORMS = {_norm_schema_name(a) for a in FIELD_ALIASES.get("som", [])}
_AUX_EXACT_NORMS = {
    "sampleid", "id", "fid", "objectid", "index", "序号", "编号", "样点编号", "点号",
    "year", "sampleyear", "outputyear", "targetyear", "年份", "采样年份", "制图年份", "调查年份",
    "depth", "sampledepth", "depthcm", "采样深度", "土层深度", "深度",
    "row", "col", "rows", "cols", "像元行", "像元列",
}
_AUX_CONTAINS_NORMS = {
    "samplelon", "samplelat", "longitude", "latitude", "lonwgs84", "latwgs84",
    "sampleyear", "outputyear", "targetyear", "date", "time", "日期", "时间",
    "crs", "epsg", "projection", "坐标系", "投影", "targetcrs",
    "scale", "resolution", "targetscale", "分辨率", "像元大小", "单位",
}
_TARGET_LEAKAGE_NORMS = {
    "som", "soilorganicmatter", "organicmatter", "om", "soc", "soilorganiccarbon", "organiccarbon",
    "有机质", "土壤有机质", "有机碳", "土壤有机碳", "target", "目标",
    "prediction", "predicted", "predict", "pred", "map", "result", "预测", "结果", "有机质图",
}


def _covariate_column_decision(col: Any, core_cols: set[Any], vals: Any, row_count: int) -> tuple[bool, str]:
    """Decide whether a fused-CSV column is an environmental covariate.

    Rule: after core schema detection, every numeric, non-constant, non-auxiliary,
    non-target-leakage column is retained. This keeps feature names dynamic while
    preventing lon/lat duplicates, years, CRS/scale metadata, target aliases, and
    existing prediction/result columns from entering the RFK model.
    """
    raw = str(col).strip()
    n = _norm_schema_name(raw)
    if col in core_cols or raw in {str(x) for x in core_cols if x is not None}:
        return False, "核心字段/辅助字段"
    if n in _COORD_NORMS or n in _TARGET_NORMS or n in _AUX_EXACT_NORMS:
        return False, "坐标、目标或辅助字段别名"
    if n in {"x", "y", "lon", "lat", "lng", "long"}:
        return False, "坐标字段"
    if any(k and k in n for k in _AUX_CONTAINS_NORMS):
        return False, "年份/坐标系/分辨率/样点元数据字段"
    if any(k and k in n for k in _TARGET_LEAKAGE_NORMS):
        return False, "目标变量或已有预测结果泄漏字段"

    valid_n = int(vals.notna().sum())
    min_valid = max(8, int(row_count * 0.2))
    if valid_n < min_valid:
        return False, f"可数值化记录过少：{valid_n} < {min_valid}"
    try:
        unique_n = int(vals.dropna().nunique())
    except Exception:
        unique_n = 0
    if unique_n <= 1:
        return False, "常数或近似常数字段"
    return True, f"有效数值 {valid_n} 条，唯一值 {unique_n} 个"


def load_sample_points(sample_path: str | Path) -> tuple[Any, dict[str, str], dict[str, Any]]:
    if pd is None:
        raise RuntimeError("缺少 pandas，无法生成统一建模 CSV。")
    df, meta = read_sample_table(sample_path)
    inferred_year, inferred_year_source = infer_year_from_dataframe(df)
    if inferred_year:
        meta["inferred_year_from_table"] = int(inferred_year)
        meta["inferred_year_source"] = inferred_year_source
    fields = infer_core_fields(list(df.columns))
    out = pd.DataFrame()
    out["lon"] = _numeric_extract(df[fields["lon"]])
    out["lat"] = _numeric_extract(df[fields["lat"]])
    out["som"] = _numeric_extract(df[fields["som"]])

    # Fused-CSV support: preserve environmental covariates already present in
    # the user CSV. Feature names are not hard-coded. Formal GeoTIFF still uses
    # only retained CSV features that have matching prediction rasters.
    core_cols = {fields.get("lon"), fields.get("lat"), fields.get("som"), fields.get("sample_id"), fields.get("year"), fields.get("depth")}
    preserved_covariates: list[str] = []
    excluded_columns: list[dict[str, Any]] = []
    for col in df.columns:
        col_s = str(col).strip()
        vals = _numeric_extract(df[col])
        keep, reason = _covariate_column_decision(col, core_cols, vals, len(df))
        if not keep:
            excluded_columns.append({"column": col_s, "reason": reason})
            continue
        safe_name = col_s
        if safe_name in out.columns:
            safe_name = "csv_" + safe_name
        out[safe_name] = vals
        preserved_covariates.append(safe_name)

    out = out.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=["lon", "lat", "som"]).copy()
    if out.empty:
        raise RuntimeError("样点表读取成功，但 lon/lat/som 转数值后没有有效记录。")
    meta.update({
        "schema_inference_version": "V167_auto_csv_schema",
        "matched_fields": fields,
        "valid_rows": int(len(out)),
        "preserved_fused_covariates": preserved_covariates,
        "preserved_fused_covariate_count": len(preserved_covariates),
        "excluded_non_covariate_columns": excluded_columns[:200],
        "excluded_non_covariate_column_count": len(excluded_columns),
    })
    return out, fields, meta


class PlatformModelCsvPipeline:
    def __init__(self, out_root: str | Path, task_id: str | None = None):
        self.out_root = Path(out_root)
        self.task_id = task_id
        # Column -> standardized raster path. Used for prediction-grid sampling of
        # domestic/local raster covariates, so they are not reduced to constants.
        self._covariate_raster_map: dict[str, str] = {}
        # Column -> value/no-data audit collected during raster sampling.
        # This is critical because 0 can be a valid environmental value; it must
        # never be counted as missing unless the raster mask/nodata policy says so.
        self._covariate_value_audit: dict[str, dict[str, Any]] = {}
        # V166: explicit raster files uploaded/imported after a fused CSV are session assets.
        # They must be scanned in addition to the sample CSV parent folder; otherwise
        # the workflow loses context when users upload CSV first and rasters later.
        self._uploaded_covariate_files: list[Path] = []
        try:
            _raw_uploaded = os.getenv("PRO_USER_UPLOADED_COVARIATE_FILES_JSON", "")
            if _raw_uploaded:
                import json as _json
                self._uploaded_covariate_files = [Path(str(x)) for x in (_json.loads(_raw_uploaded) or []) if str(x).strip()]
        except Exception:
            self._uploaded_covariate_files = []
        self.timeout = int(os.getenv("PRO_PLATFORM_HTTP_TIMEOUT", os.getenv("PRO_PUBLIC_HTTP_TIMEOUT", "30")))
        self.max_download_mb = int(os.getenv("PRO_PLATFORM_MAX_DOWNLOAD_MB", "200"))
        self.max_downloads_per_platform = int(os.getenv("PRO_PLATFORM_MAX_DOWNLOADS_PER_SOURCE", "3"))
        self.allow_download = os.getenv("PRO_PLATFORM_ALLOW_DOWNLOAD", "1") == "1"
        self.run_mode = (os.getenv("PRO_RUN_MODE", "formal") or "formal").strip().lower()
        self.is_formal_run = self.run_mode not in {"test", "debug", "diagnostic"}
        self.require_real_raster = os.getenv(
            "PRO_REQUIRE_REAL_RASTER_FOR_MODEL_CSV",
            "1" if self.is_formal_run else "0",
        ) == "1"
        # 正式版禁止用坐标/平台状态伪特征补齐模型；诊断模式才允许兜底。
        self.use_fallback_features = os.getenv(
            "PRO_DIAGNOSTIC_USE_FALLBACK_FEATURES",
            "0" if self.is_formal_run else "1",
        ) == "1"
        # 下载策略：正式主线优先使用 GEE 云端协变量；网页平台探测作为审计/备用，不阻塞制图。
        self.skip_large_file = os.getenv("PRO_SKIP_LARGE_FILE", "1") == "1"
        self.diagnostic_max_file_mb = int(os.getenv("PRO_DIAGNOSTIC_MAX_FILE_MB", str(self.max_download_mb)))
        self.download_reuse = os.getenv("PRO_DOWNLOAD_REUSE", "1") == "1"
        self.download_cache_dir = Path(os.getenv("PRO_DOWNLOAD_CACHE_DIR", str(self.out_root / "_download_cache")))
        self.download_cache_dir.mkdir(parents=True, exist_ok=True)
        # 国内原始文件专项爬取：用于应对“必须证明能从国内平台爬取/下载数据”的审计要求。
        # 说明：GSCloud/NODA 页面可能依赖登录、JS、动态 API 或人工审核。本模块先做 requests 级深度探测、
        # JS/API 候选发现、下载链接发现与可选 seeded URL 下载；若无直链，会明确记录原因，而不伪装成功。
        self.domestic_raw_crawl = os.getenv("PRO_DOMESTIC_RAW_CRAWL", "1") == "1"
        self.domestic_raw_targets = {x.strip().lower() for x in os.getenv("PRO_DOMESTIC_RAW_TARGETS", "geodata,gscloud,nesdc,noda,tpdc,ngac").split(",") if x.strip()}
        self.strict_true_data_download = os.getenv("PRO_STRICT_TRUE_DATA_DOWNLOAD", "1") == "1"
        self.save_html_probe = os.getenv("PRO_SAVE_HTML_PROBE", "1") == "1"
        self.domestic_crawl_max_pages = int(os.getenv("PRO_DOMESTIC_CRAWL_MAX_PAGES", "12"))
        self.domestic_crawl_depth = int(os.getenv("PRO_DOMESTIC_CRAWL_DEPTH", "1"))
        self.domestic_raw_keywords = [x.strip() for x in os.getenv("PRO_DOMESTIC_RAW_KEYWORDS", "MODIS,NDVI,DEM,Landsat,Sentinel,土地利用,下载,数据,影像,产品").split(",") if x.strip()]
        # V108：硬性要求国内平台下载。GEE/本地缓存不能替代“本次从国内平台获得真实数据文件”。
        self.force_domestic_platform_download = os.getenv("PRO_FORCE_DOMESTIC_PLATFORM_DOWNLOAD", "0") == "1"
        self.domestic_true_data_min_downloads = int(os.getenv("PRO_DOMESTIC_TRUE_DATA_MIN_DOWNLOADS", "1"))
        self.domestic_manual_handoff = os.getenv("PRO_DOMESTIC_MANUAL_HANDOFF", "1") == "1"
        self.domestic_validate_download_links = os.getenv("PRO_DOMESTIC_VALIDATE_DOWNLOAD_LINKS", "1") == "1"
        # V88：GEE缺失协变量时，国内平台作为补充数据源，而不是完全跳过。
        # 重点补充开题报告中 GEE 公共目录不一定稳定具备的农田利用/作物制度/国内土壤背景类数据。
        self.domestic_fallback_for_missing_gee = os.getenv("PRO_DOMESTIC_FALLBACK_FOR_MISSING_GEE", "1") == "1"
        self.domestic_fallback_required = os.getenv("PRO_DOMESTIC_FALLBACK_REQUIRED", "0") == "1"
        self.include_domestic_covariates_in_formal_model = os.getenv("PRO_INCLUDE_DOMESTIC_COVARIATES_IN_FORMAL_MODEL", "1") == "1"
        # V94: local curated environmental covariates can be loaded even when GEE succeeds.
        # This supports the delivery pattern where the user uploads samples for a broad area
        # but wants to map a smaller administrative unit using a controlled, auditable raster set.
        self.local_covariates_always_load = os.getenv("PRO_LOCAL_COVARIATES_ALWAYS_LOAD", "1") == "1"
        self._covariate_selection_plan: dict[str, Any] = {"mode": "default", "source": "init"}
        # V96：模型参数不再一套固定参数打天下。
        # 先解析用户显式参数；若用户未给参数，则按目标制图区内样点数做轻量超参数搜索。
        self._current_request_text: str = ""
        self._active_rf_params: dict[str, Any] | None = None
        self._active_rfk_params: dict[str, Any] = {}
        self._hyperparameter_plan: dict[str, Any] = {"mode": "not_resolved"}
        self.domestic_fallback_targets = {x.strip().lower() for x in os.getenv("PRO_DOMESTIC_FALLBACK_TARGETS", "geodata,gscloud,nesdc,noda,tpdc,ngac").split(",") if x.strip()}
        self.domestic_fallback_local_dir = Path(os.getenv("PRO_DOMESTIC_FALLBACK_LOCAL_DIR", r"E:/Agent_DSM/data"))
        self.skip_domestic_web_after_local_success = os.getenv("PRO_SKIP_DOMESTIC_WEB_AFTER_LOCAL_SUCCESS", "1") == "1"
        # Playwright 登录态下载捕获：浏览器中人工登录/点击，后端自动接管真实文件。
        # 注意：这里不在 Dash 任务里强制打开浏览器；默认只“导入” capture 工具已捕获的真实文件，避免阻塞网页任务。
        self.playwright_import_downloads = os.getenv("PRO_PLAYWRIGHT_IMPORT_DOWNLOADS", "1") == "1"
        self.playwright_capture_root = Path(os.getenv("PRO_PLAYWRIGHT_CAPTURE_DIR", str(self.out_root / "manual_domestic_downloads")))
        self.playwright_import_max_age_hours = float(os.getenv("PRO_PLAYWRIGHT_IMPORT_MAX_AGE_HOURS", "24"))
        # GEE 云端协变量抽取：真正产品化路线。默认开启，若本机未认证或缺 earthengine-api 会自动记录失败并继续平台探测。
        self.gee_enable = os.getenv("PRO_GEE_ENABLE", "0") == "1"
        self.gee_require_success = os.getenv(
            "PRO_GEE_REQUIRE_SUCCESS",
            "0",
        ) == "1"
        self.gee_project = os.getenv("PRO_GEE_PROJECT") or os.getenv("GEE_PROJECT") or "fit-territory-472114-b0"
        # V68: GEE正式流程开关。开启后，训练/预测只使用cov_开头的真实协变量，
        # 不再把sample_id、src_*平台状态特征、coord_*诊断特征纳入正式模型。
        self.gee_formal_flow = os.getenv("PRO_GEE_FORMAL_FLOW", "0") == "1"
        # V73: GEE 优先正式流程。GEE 已成功时，国内/国外网页平台探测不再阻塞正式制图。
        self.gee_priority_mode = os.getenv("PRO_GEE_PRIORITY_MODE", "0") == "1"
        self.probe_after_gee_success = os.getenv("PRO_PLATFORM_PROBE_AFTER_GEE_SUCCESS", "0") == "1"
        # V74: 年份策略调整。
        # - 用户在问题中明确说了年份：优先采用用户指定年份。
        # - 上传文件名/表格字段能推断年份且与用户年份不同：只记录提醒，不再默认停止。
        # - 只有用户没有说年份时，才用上传数据可推断年份兜底；若二者都没有，停止并要求用户补充年份。
        # 兼容旧 .env：即使旧配置写了 PRO_YEAR_CONFLICT_POLICY=stop，也不会停止；
        # 若确实想严格停止，需额外设置 PRO_YEAR_STRICT_CONFLICT_STOP=1。
        raw_year_conflict_policy = os.getenv("PRO_YEAR_CONFLICT_POLICY", "warn_keep_request").strip().lower()
        if raw_year_conflict_policy == "stop" and os.getenv("PRO_YEAR_STRICT_CONFLICT_STOP", "0") != "1":
            raw_year_conflict_policy = "warn_keep_request"
        self.year_conflict_policy = raw_year_conflict_policy
        # V71: uploaded sample coordinates drive region inference when user does not explicitly specify region.
        self.region_inference_enable = os.getenv("PRO_REGION_INFERENCE_ENABLE", "1") == "1"
        self.region_conflict_policy = os.getenv("PRO_REGION_CONFLICT_POLICY", "prefer_user_warn").strip().lower()
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        # Earthdata 认证不要挂到全局 session 上：CMR 元数据接口是公开检索，
        # 如果把 EDL Bearer Token 或 Basic Auth 加到 CMR 检索请求，部分环境会直接返回 401。
        # 因此：CMR collection/granule 用 public session；真实文件下载时再按需加认证。
        self.earthdata_token: str | None = None
        self.earthdata_user: str | None = None
        self.earthdata_password: str | None = None
        self._configure_earthdata_auth()
        self.warnings: list[str] = []
        # V87: formal modelling algorithm. Default is true RFK, not plain RandomForest.
        # RFK = RandomForest environmental prediction + OrdinaryKriging of RF residuals.
        self.model_algorithm = (os.getenv("PRO_MODEL_ALGORITHM", "RFK" if self.is_formal_run else "RF") or "RFK").strip().upper()
        self._rfk_state: dict[str, Any] | None = None
        # V83: lightweight conformal/GCP-style uncertainty generated directly from the PRO/GEE model run.
        # It is intentionally stored on the runner so the final GEE prediction writer can output
        # lower/upper/width GeoTIFFs using the same grid, CRS, admin mask and cropland mask.
        self._gee_gcp_calibration: dict[str, Any] | None = None

    def _raster_resolution_meters(self, path: Path) -> float | None:
        """Return approximate pixel size in meters for a raster or raster container."""
        if rasterio is None:
            return None
        try:
            with rasterio.open(path) as src0:
                sub = self._select_subdataset(src0, path) if getattr(src0, "subdatasets", None) else None
            open_target = sub or str(path)
            with rasterio.open(open_target) as src:
                if not src.crs:
                    return None
                rx = abs(float(src.transform.a))
                ry = abs(float(src.transform.e))
                if src.crs.is_geographic:
                    b = src.bounds
                    lat_mid = (float(b.top) + float(b.bottom)) / 2.0
                    m_per_deg_lon = 111320.0 * max(0.2, math.cos(math.radians(lat_mid)))
                    m_per_deg_lat = 111320.0
                    return max(rx * m_per_deg_lon, ry * m_per_deg_lat)
                return max(rx, ry)
        except Exception:
            return None

    def _lulccd_disabled(self) -> bool:
        """LULCcd is disabled by default in V161; CLCD is the land-cover covariate."""
        return (os.getenv("PRO_DISABLE_LULCCD", "1") or "1").strip().lower() not in {"0", "false", "no", "off"}

    def _is_lulccd_name(self, value: Any) -> bool:
        lo = str(value or "").lower().replace("-", "_")
        return ("lulccd" in lo) or ("lulc_cd" in lo) or ("lulc" in lo and "clcd" not in lo)

    def _is_clcd_name(self, value: Any) -> bool:
        lo = str(value or "").lower().replace("-", "_")
        return ("clcd" in lo) or ("china_land_cover" in lo) or ("china land cover" in lo)

    def _clcd_class_map(self) -> dict[int, str]:
        return {
            1: "农田",
            2: "森林",
            3: "灌木",
            4: "草地",
            5: "水体",
            6: "冰雪",
            7: "裸地",
            8: "不透水面",
            9: "湿地",
        }

    def _clcd_dummy_col(self, base_col: str, code: int, label: str) -> str:
        safe_base = str(base_col).strip()
        return f"{safe_base}__CLCD_{int(code)}_{label}"

    def _is_clcd_dummy_col(self, col: Any) -> bool:
        return "__CLCD_" in str(col or "")

    def _clcd_mask_only_policy(self) -> bool:
        """V173: CLCD is a cropland mask only, not a SOM predictor.

        For cropland SOM mapping, CLCD=1 defines where continuous SOM is
        predicted/displayed. CLCD classes 2-8 are non-cropland and are never
        encoded as model features. This prevents the land-cover class itself
        from driving the organic matter prediction over a cropland-only target.
        """
        return (os.getenv("PRO_CLCD_MASK_ONLY", "1") or "1").strip().lower() not in {"0", "false", "no", "off"}

    def _is_clcd_model_feature(self, col: Any) -> bool:
        return self._is_clcd_name(col) or self._is_clcd_dummy_col(col)

    def _is_landcover_mask_path(self, path: Any) -> bool:
        lo = str(path or "").lower().replace("-", "_")
        return any(k in lo for k in [
            "clcd", "landcover", "land_cover", "land use", "landuse", "lulc",
            "worldcover", "土地覆盖", "土地利用", "地表覆盖"
        ])

    def _builtin_landcover_roots(self) -> list[Path]:
        roots: list[Path] = []
        env = os.getenv("PRO_BUILTIN_LANDCOVER_DIR", "").strip()
        if env:
            roots.append(Path(env))
        roots.append(Path(r"E:/Agent_DSM/土地覆盖数据"))
        roots.append(Path(r"E:\Agent_DSM\土地覆盖"))
        roots.append(Path(r"E:\Agent_DSM\landcover"))
        out = []
        seen = set()
        for r in roots:
            key = str(r)
            if key not in seen:
                out.append(r); seen.add(key)
        return out

    def _find_builtin_landcover_raster(self, target: dict[str, Any] | None = None) -> str | None:
        """Locate the built-in 30 m national land-cover raster.

        User-specified land-cover mapping must not silently fall back to full-domain
        output when CLCD in the covariate folder is missing or rejected by year audit.
        This searches E:/Agent_DSM/土地覆盖数据 recursively and prefers rasters that
        match the requested year, then generic land-cover rasters.
        """
        try:
            target_year = None
            try:
                target_year = int((target or {}).get("year"))
            except Exception:
                target_year = None
            candidates: list[Path] = []
            for root in self._builtin_landcover_roots():
                if not root.exists():
                    continue
                for pat in ("*.tif", "*.tiff", "*.img", "*.vrt"):
                    candidates.extend([p for p in root.rglob(pat) if p.is_file()])
            if not candidates:
                return None
            def score(p: Path) -> tuple[int, int, int]:
                name = p.name.lower()
                sc = 0
                if any(k in name for k in ["clcd", "landcover", "land_cover", "lulc", "worldcover"]):
                    sc += 50
                if any(k in str(p) for k in ["土地覆盖", "土地利用", "地表覆盖"]):
                    sc += 40
                if target_year and str(target_year) in name:
                    sc += 100
                if "30" in name or "30m" in name:
                    sc += 10
                try:
                    size = p.stat().st_size
                except Exception:
                    size = 0
                return (sc, min(int(size // 1024), 10_000_000), -len(str(p)))
            best = sorted(candidates, key=score, reverse=True)[0]
            return str(best)
        except Exception as exc:
            try:
                self.warnings.append(f"内置土地覆盖数据检索失败：{exc}")
            except Exception:
                pass
            return None

    def _parse_hex_color_rgba(self, value: Any, default: str) -> tuple[int, int, int, int]:
        """Parse #RRGGBB / RRGGBB for mask preview colors."""
        raw = str(value or default or "").strip()
        if not raw:
            raw = str(default)
        if raw.startswith("#"):
            raw = raw[1:]
        if len(raw) == 3:
            raw = "".join(ch * 2 for ch in raw)
        try:
            r = int(raw[0:2], 16); g = int(raw[2:4], 16); b = int(raw[4:6], 16)
            return int(r), int(g), int(b), 255
        except Exception:
            return (46, 160, 67, 255) if str(default).lower() == "#2ea043" else (216, 27, 96, 255)

    def _write_clcd_binary_mask_outputs(self, out_dir: Path, admin_mask: Any, grid_df: Any,
                                        cropland_mask: Any, rows: Any, cols: Any,
                                        transform: Any, crs: str, target_label: str = "目标地类", target_code: int | None = None) -> dict[str, Any]:
        """Write two-class CLCD target land-cover mask products.

        Values: 1 = target CLCD class; 0 = non-target classes or invalid inside
        AOI; 255 = outside AOI (NoData).  Historically this function was named
        cropland, but V221 supports forest/grassland/etc. with the same binary
        mask convention.
        """
        outputs: dict[str, Any] = {}
        try:
            if np is None or rasterio is None:
                return outputs
            cropland_color = self._parse_hex_color_rgba(os.getenv("PRO_CLCD_CROPLAND_COLOR", "#2EA043"), "#2EA043")
            non_cropland_color = self._parse_hex_color_rgba(os.getenv("PRO_CLCD_NON_CROPLAND_COLOR", "#E53935"), "#E53935")
            target_label = str(target_label or "目标地类")
            target_code_txt = str(target_code) if target_code is not None else "目标"
            mask = np.full(admin_mask.shape, 255, dtype="uint8")
            mask[np.asarray(admin_mask) != 0] = 0
            rr = np.asarray(rows, dtype=int)
            cc = np.asarray(cols, dtype=int)
            cm = np.asarray(cropland_mask, dtype=bool)
            ok = (rr >= 0) & (rr < mask.shape[0]) & (cc >= 0) & (cc < mask.shape[1])
            mask[rr[ok], cc[ok]] = np.where(cm[ok], 1, 0).astype("uint8")
            mask_tif = out_dir / "CLCD目标地类掩膜_1目标0其他.tif"
            with rasterio.open(mask_tif, "w", driver="GTiff", height=mask.shape[0], width=mask.shape[1],
                               count=1, dtype="uint8", crs=crs, transform=transform, nodata=255, compress="lzw") as dst:
                dst.write(mask, 1)
                try:
                    dst.write_colormap(1, {
                        0: non_cropland_color,
                        1: cropland_color,
                        255: (0, 0, 0, 0),
                    })
                except Exception:
                    pass
            outputs["clcd_mask_tif"] = str(mask_tif)
            try:
                from PIL import Image
                rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype="uint8")
                rgba[mask == 0] = non_cropland_color
                rgba[mask == 1] = cropland_color
                rgba[mask == 255] = (0, 0, 0, 0)
                mask_png = out_dir / "CLCD目标地类掩膜_双色预览.png"
                Image.fromarray(rgba, mode="RGBA").save(mask_png)
                outputs["clcd_mask_png"] = str(mask_png)
            except Exception as exc:
                outputs["clcd_mask_png_error"] = str(exc)
            outputs["legend"] = {
                "1": {"label": f"目标地类（CLCD={target_code_txt}，{target_label}）", "color": "#%02X%02X%02X" % cropland_color[:3]},
                "0": {"label": "非目标地类或无效（AOI内）", "color": "#%02X%02X%02X" % non_cropland_color[:3]},
                "255": {"label": "AOI外部NoData", "color": "transparent"},
            }
            outputs["target_class_code"] = target_code
            outputs["target_class_label"] = target_label
            outputs["target_landcover_cell_count"] = int(np.sum(mask == 1))
            outputs["non_target_landcover_cell_count"] = int(np.sum(mask == 0))
            outputs["cropland_cell_count"] = int(np.sum(mask == 1)) if target_code == 1 else None
            outputs["non_cropland_cell_count"] = int(np.sum(mask == 0)) if target_code == 1 else None
            outputs["outside_aoi_cell_count"] = int(np.sum(mask == 255))
        except Exception as exc:
            outputs = {"error": str(exc)}
        return outputs

    def _resolution_level_default_m(self, target: dict[str, Any]) -> tuple[float, str]:
        """Return V168 AOI-level default resolution in meters.

        Rule: county/district = 30 m; prefecture-level city = 250 m; province =
        500 m; country = 1000 m.  This default is only used when all observed
        user data resolutions are 30 m or finer, or when no usable resolution can
        be read from data.
        """
        level = str(
            target.get("aoi_level")
            or target.get("region_level")
            or (target.get("region_inference") or {}).get("level")
            or ""
        ).strip().lower()
        region = str(target.get("region") or "").strip()
        if not level:
            if region in {"全国", "中国"}:
                level = "country"
            elif any(k in region for k in ["县", "区", "旗"]):
                level = "county"
            elif bool(target.get("is_city")) or region.endswith("市"):
                level = "city"
            else:
                level = "province"
        if level in {"county", "district", "区县", "县区"}:
            return float(os.getenv("PRO_AUTO_RES_COUNTY_M", "30")), "county"
        if level in {"country", "national", "nation", "全国", "国级"}:
            return float(os.getenv("PRO_AUTO_RES_COUNTRY_M", "1000")), "country"
        if level in {"province", "省", "省级"}:
            return float(os.getenv("PRO_AUTO_RES_PROVINCE_M", "500")), "province"
        return float(os.getenv("PRO_AUTO_RES_CITY_M", "250")), "city"

    def _infer_csv_declared_resolution_records(self, sample_df: Any | None) -> list[dict[str, Any]]:
        """Read target/modeling scale declared in a fused CSV, e.g. target_scale_m=250.

        A fused training CSV is already a modeling matrix.  If it declares a modeling
        scale, that scale must participate in auto-resolution inference together
        with uploaded rasters; otherwise a single 30 m CLCD mask can incorrectly
        force a city-wide 30 m prediction grid.
        """
        records: list[dict[str, Any]] = []
        if pd is None or sample_df is None:
            return records
        aliases = {
            "target_scale_m", "target_resolution_m", "resolution_m", "scale_m",
            "grid_m", "pixel_size_m", "cell_size_m", "spatial_resolution_m",
            "目标分辨率", "目标分辨率米", "分辨率", "像元大小", "像元尺寸",
        }
        for col in list(getattr(sample_df, "columns", [])):
            col_s = str(col).strip()
            col_key = col_s.lower().replace(" ", "_").replace("-", "_")
            if col_s not in aliases and col_key not in aliases:
                continue
            try:
                vals = pd.to_numeric(sample_df[col], errors="coerce")
                vals = vals[np.isfinite(vals)] if np is not None else vals.dropna()
                if len(vals) <= 0:
                    continue
                # Prefer the modal/median declared scale; most fused CSVs repeat the same value.
                v = float(vals.mode(dropna=True).iloc[0]) if hasattr(vals, "mode") and not vals.mode(dropna=True).empty else float(vals.median())
                if math.isfinite(v) and v > 0:
                    records.append({"source": "fused_csv", "column": col_s, "resolution_m": v})
            except Exception:
                continue
        return records

    def _infer_target_resolution_from_user_rasters(self, roots: list[Path], target: dict[str, Any], sample_df: Any | None = None) -> dict[str, Any]:
        """Infer output resolution when the user did not specify it.

        V168 policy:
        1) explicit user resolution always wins;
        2) otherwise use the coarsest/lowest resolution among user data, including
           raster pixel sizes and fused-CSV declared target scale;
        3) if all usable user data are 30 m or finer, avoid over-detailed large AOI
           output by AOI level: county/district 30 m, city 250 m, province 500 m,
           country 1000 m.
        """
        if target.get("resolution_m"):
            target["resolution_source"] = target.get("resolution_source") or "user_or_existing"
            return target
        files = []
        try:
            files = self._scan_files([r for r in roots if r])
        except Exception:
            files = []
        raster_records = []
        ignored_records = []
        for f in files:
            if f.suffix.lower() not in RASTER_EXTS:
                continue
            if self._lulccd_disabled() and self._is_lulccd_name(f.name):
                ignored_records.append({"file": str(f), "name": f.name, "reason": "LULCcd_disabled_use_CLCD"})
                continue
            res = self._raster_resolution_meters(f)
            if res is not None and math.isfinite(float(res)) and float(res) > 0:
                raster_records.append({"file": str(f), "name": f.name, "resolution_m": float(res)})

        csv_records = self._infer_csv_declared_resolution_records(sample_df)
        all_records = list(raster_records) + list(csv_records)
        level_default, level = self._resolution_level_default_m(target)
        fine_threshold = float(os.getenv("PRO_AUTO_RES_FINE_ONLY_THRESHOLD_M", "31"))

        if all_records:
            coarsest = float(max(r["resolution_m"] for r in all_records))
            if coarsest <= fine_threshold:
                chosen = float(max(1.0, round(level_default)))
                source = f"aoi_level_default_when_all_user_data_fine_{level}"
                rule = (
                    "用户未指定分辨率，且全部用户数据为30米或更细；"
                    "按AOI级别设置输出分辨率：县区30m、地级市250m、省级500m、国级1km。"
                )
            else:
                chosen = float(max(1.0, round(coarsest)))
                source = "coarsest_active_user_data"
                rule = (
                    "用户未指定分辨率，采用用户上传/融合数据中的最低/最粗空间分辨率；"
                    "例如30m、250m、500m同时存在时统一为500m。"
                )
            target["resolution_m"] = chosen
            target["resolution_source"] = source
            target["resolution_audit"] = {
                "rule": rule,
                "chosen_resolution_m": chosen,
                "aoi_level": level,
                "aoi_level_default_m": level_default,
                "fine_only_threshold_m": fine_threshold,
                "observed_resolution_min_m": float(min(r["resolution_m"] for r in all_records)),
                "observed_resolution_max_m": coarsest,
                "raster_count": len(raster_records),
                "rasters": sorted(raster_records, key=lambda x: x["resolution_m"], reverse=True)[:50],
                "csv_declared_resolutions": sorted(csv_records, key=lambda x: x["resolution_m"], reverse=True)[:50],
                "ignored_rasters": ignored_records[:50],
            }
            _emit("PREPROCESS", "已按V168规则自动确定目标分辨率", target["resolution_audit"], task_id=self.task_id)
            return target

        fallback = float(max(1.0, round(level_default)))
        target["resolution_m"] = fallback
        target["resolution_source"] = f"aoi_level_default_no_user_resolution_{level}"
        target["resolution_audit"] = {
            "rule": "未能读取用户数据分辨率，按AOI级别设置输出分辨率：县区30m、地级市250m、省级500m、国级1km。",
            "chosen_resolution_m": fallback,
            "aoi_level": level,
            "aoi_level_default_m": level_default,
            "raster_count": 0,
            "csv_declared_resolutions": [],
            "ignored_rasters": ignored_records[:50],
        }
        _emit("PREPROCESS", "未能从用户数据读取分辨率，已按AOI级别设置目标分辨率", target["resolution_audit"], task_id=self.task_id)
        return target

    def _select_platforms(self, platforms: list[PlatformSource] | None = None) -> list[PlatformSource]:
        """按数据源策略选择平台。

        PRO_SOURCE_MODE:
        - domestic_first 默认：国内优先，国外保留在后面；国外默认只做元数据/连通性，避免卡住主线。
        - domestic_only：只测国内。
        - all：全部平台按优先级审计。
        """
        mode = (os.getenv("PRO_SOURCE_MODE") or os.getenv("PRO_DATA_SOURCE_MODE") or "domestic_first").strip().lower()
        domestic_only_legacy = os.getenv("PRO_DOMESTIC_ONLY", "0") == "1"
        base = [p for p in (platforms or default_platform_sources()) if p.enabled]
        if domestic_only_legacy:
            mode = "domestic_only"
        if os.getenv("PRO_FORCE_DOMESTIC_PLATFORM_DOWNLOAD", "1") == "1":
            mode = "domestic_only"
        if mode in {"user_uploaded_only", "user", "local_only", "uploaded_only"}:
            # 用户已经提供样点和环境协变量时，不再爬取任何外部平台。
            # 否则“本机数据制图”会被拖入网页探测流程，进度条长期停留且用户误以为任务失败。
            selected = []
        elif mode == "domestic_only":
            selected = [p for p in base if p.group == "domestic"]
        elif mode == "all":
            selected = base
        else:
            # 默认国内优先，其次 OSM 等轻量辅助源，国外保留但放后面。
            group_rank = {"domestic": 0, "auxiliary": 1, "foreign": 2}
            selected = sorted(base, key=lambda p: (group_rank.get(p.group, 9), p.priority))
        _emit("SOURCE", "数据源审计清单已确定", {
            "source_mode": mode,
            "legacy_PRO_DOMESTIC_ONLY": domestic_only_legacy,
            "domestic_count": len([p for p in selected if p.group == "domestic"]),
            "auxiliary_count": len([p for p in selected if p.group == "auxiliary"]),
            "foreign_count": len([p for p in selected if p.group == "foreign"]),
            "platform_ids": [p.id for p in selected],
            "foreign_download_mode": os.getenv("PRO_FOREIGN_DOWNLOAD_MODE", "metadata_only"),
        }, task_id=self.task_id)
        return selected

    def run(self, request_text: str, sample_path: str | Path, platforms: list[PlatformSource] | None = None) -> PipelineResult:
        self._current_request_text = request_text or ""
        self._active_rf_params = None
        self._active_rfk_params = {}
        self._hyperparameter_plan = {"mode": "not_resolved"}
        target = parse_request_basic(request_text)
        # V94：优先用本地行政区划属性表识别更小制图区。
        # 典型场景：上传的是成都市样点，但用户只要求绘制“温江区”。
        # 此时训练样点仍可覆盖成都全域，预测输出范围必须裁剪到温江区。
        local_region_text_note: dict[str, Any] | None = None
        try:
            target, local_region_text_note = self._refine_target_by_local_admin_text(request_text, target)
        except Exception as exc:
            local_region_text_note = {"ok": False, "error": str(exc), "message": f"本地行政区划文本识别失败，保留原区域解析：{exc}"}
            self.warnings.append(local_region_text_note["message"])

        # V94：解析用户选择的环境协变量。未指定时采用内置默认协变量清单。
        self._covariate_selection_plan = self._parse_covariate_selection_from_text(request_text)
        target["covariate_selection"] = self._covariate_selection_plan
        year_inference_notes: list[dict[str, Any]] = []

        # V74：年份策略。
        # 用户明确写年份时，直接按用户年份制图；上传文件名若能推断出不同年份，只作为提醒，不默认停止。
        # 用户未写年份时，才从文件名/表格字段推断；仍无法推断则停止并要求补充年份。
        filename_year, filename_year_source = infer_year_from_filename(sample_path)
        request_year = target.get("year")
        if request_year and filename_year and int(request_year) != int(filename_year):
            msg = (
                f"检测到用户请求年份为 {int(request_year)}，"
                f"上传文件名可推断为 {int(filename_year)}（{filename_year_source}）。"
            )
            year_inference_notes.append({
                "method": "year_mismatch_warning",
                "request_year": int(request_year),
                "filename_year": int(filename_year),
                "filename_year_source": filename_year_source,
                "policy": self.year_conflict_policy,
                "message": msg + " 已按用户请求年份继续；若样点年份与制图年份不一致，请自行确认结果解释。",
            })
            if self.year_conflict_policy == "prefer_sample":
                target["year"] = int(filename_year)
                target["year_source"] = filename_year_source + ":conflict_override"
                year_inference_notes.append({
                    "method": "prefer_sample",
                    "year": int(filename_year),
                    "source": target["year_source"],
                    "message": f"年份不一致时按策略采用上传样点年份 {int(filename_year)}。",
                })
            elif self.year_conflict_policy in {"strict_stop", "stop"} and os.getenv("PRO_YEAR_STRICT_CONFLICT_STOP", "0") == "1":
                raise RuntimeError(msg + " 当前已启用严格年份冲突停止。请确认制图年份，或重新上传对应年份样点。")

        # 用户未显式写年份时，先从上传文件名推断，例如“2021样点.csv”。
        if not target.get("year") and filename_year:
            target["year"] = int(filename_year)
            target["year_source"] = filename_year_source
            year_inference_notes.append({
                "method": "filename",
                "year": int(filename_year),
                "source": filename_year_source,
                "message": f"未在请求中识别到年份，已根据上传文件名推断为 {int(filename_year)} 年。",
            })

        samples, fields, sample_meta = load_sample_points(sample_path)
        if filename_year:
            sample_meta["inferred_year_from_filename"] = int(filename_year)
            sample_meta["inferred_year_from_filename_source"] = filename_year_source

        # 若样点表字段能明确推断年份，且用户明确请求了另一个年份，仍只提醒，不默认停止。
        table_year = sample_meta.get("inferred_year_from_table")
        if request_year and table_year and int(request_year) != int(table_year):
            src = sample_meta.get("inferred_year_source") or "sample_table"
            year_inference_notes.append({
                "method": "year_mismatch_warning_table",
                "request_year": int(request_year),
                "table_year": int(table_year),
                "table_year_source": src,
                "policy": self.year_conflict_policy,
                "message": (
                    f"检测到用户请求年份为 {int(request_year)}，样点表字段可推断为 {int(table_year)}（{src}）。"
                    f"已按用户请求年份继续；若样点年份与制图年份不一致，请自行确认结果解释。"
                ),
            })
            if self.year_conflict_policy == "prefer_sample":
                target["year"] = int(table_year)
                target["year_source"] = str(src) + ":conflict_override"
            elif self.year_conflict_policy in {"strict_stop", "stop"} and os.getenv("PRO_YEAR_STRICT_CONFLICT_STOP", "0") == "1":
                raise RuntimeError(
                    f"检测到年份不一致：用户请求年份为 {int(request_year)}，样点表字段推断为 {int(table_year)}。"
                    " 当前已启用严格年份冲突停止。请确认制图年份，或重新上传对应年份样点。"
                )

        # V71：区域自动识别。优先使用用户提问中的区域；若提问未写区域，则根据样点经纬度推断。
        region_inference_notes: list[dict[str, Any]] = []
        if local_region_text_note:
            region_inference_notes.append(local_region_text_note)
        if self.region_inference_enable:
            inferred_region = infer_region_from_samples(samples)
            sample_meta["region_inference"] = inferred_region
            request_region = target.get("region")
            if inferred_region.get("ok"):
                if not request_region:
                    target = apply_region_to_target(
                        target,
                        inferred_region["region"],
                        level=inferred_region.get("level"),
                        source="sample_coordinates",
                        inference=inferred_region,
                    )
                    region_inference_notes.append({
                        "method": inferred_region.get("method"),
                        "region": inferred_region.get("region"),
                        "level": inferred_region.get("level"),
                        "dominance_ratio": inferred_region.get("dominance_ratio"),
                        "message": f"未在请求中识别到明确区域，已根据样点经纬度推断为 {inferred_region.get('region')}。",
                    })
                else:
                    compatible, mismatch_note = self._target_matches_sample_region(target, inferred_region)
                    if not compatible:
                        msg = mismatch_note.get("message") or (
                            f"检测到用户请求区域为 {request_region}，但样点经纬度主要落在 "
                            f"{inferred_region.get('region')}；当前策略为 {self.region_conflict_policy}。"
                        )
                        mismatch_note.update({
                            "method": "sample_target_region_mismatch",
                            "request_region": request_region,
                            "inferred_region": inferred_region.get("region"),
                            "policy": self.region_conflict_policy,
                            "popup": True,
                            "severity": "warning",
                            "message": msg,
                        })
                        target["sample_region_mismatch"] = mismatch_note
                        region_inference_notes.append(mismatch_note)
                        self.warnings.append(msg)
                        if self.region_conflict_policy == "stop":
                            raise RuntimeError(msg + " 请确认制图区域或重新上传样点。")
                        if self.region_conflict_policy == "prefer_sample":
                            target = apply_region_to_target(
                                target,
                                inferred_region["region"],
                                level=inferred_region.get("level"),
                                source="sample_coordinates_conflict_override",
                                inference=inferred_region,
                            )
                        else:
                            # prefer_user_warn: keep user region but preserve inference for audit and UI popup.
                            target["region_inference"] = inferred_region
                    else:
                        target["region_inference"] = inferred_region
                        region_inference_notes.append({
                            "method": "compatible_check",
                            "request_region": request_region,
                            "inferred_region": inferred_region.get("region"),
                            "message": f"用户请求区域与样点分布一致或兼容：{request_region}。",
                        })
            else:
                region_inference_notes.append({
                    "method": inferred_region.get("method"),
                    "message": inferred_region.get("message") or "未能根据样点坐标推断区域。",
                })

        # V96：用本地行政区划统计目标制图区内样点数。
        # 它只用于审计、弹窗提醒和超参数自适应；默认不强制只用区内样点训练，避免区县样点过少导致模型不稳定。
        sample_aoi_overlap = self._sample_aoi_overlap(samples, target)
        target["sample_aoi_overlap"] = sample_aoi_overlap
        if sample_aoi_overlap.get("ok"):
            inside_n = int(sample_aoi_overlap.get("inside_count") or 0)
            total_n = int(sample_aoi_overlap.get("total_count") or 0)
            min_warn = int(os.getenv("PRO_AOI_MIN_SAMPLE_WARNING", "30"))
            if inside_n == 0 and target.get("region"):
                note = {
                    "method": "sample_aoi_overlap_zero",
                    "popup": True,
                    "severity": "warning",
                    "message": f"目标制图区 {target.get('region')} 内没有检测到上传样点；当前样点主要不在制图区内，结果只能视为外推，不建议交付。",
                    **sample_aoi_overlap,
                }
                target["sample_region_mismatch"] = note
                region_inference_notes.append(note)
                self.warnings.append(note["message"])
            elif inside_n < min_warn and total_n >= min_warn and target.get("region"):
                note = {
                    "method": "sample_aoi_overlap_low",
                    "popup": True,
                    "severity": "warning",
                    "message": f"目标制图区 {target.get('region')} 内样点数较少：{inside_n}/{total_n}。模型会按区内样点规模收缩复杂度，但结果仍需谨慎解释。",
                    **sample_aoi_overlap,
                }
                region_inference_notes.append(note)
                self.warnings.append(note["message"])

        if not target.get("region"):
            if os.getenv("PRO_REGION_MISSING_POLICY", "sample_bbox").strip().lower() == "stop":
                raise RuntimeError("未识别到制图区域，且样点坐标无法推断主要省份。请在问题中写明区域，或检查CSV经纬度字段。")
            target["region"] = "样点包络范围"
            target["region_source"] = "sample_bbox_fallback"

        # 若文件名没有年份，再尝试从样点表年份/日期字段推断。
        if not target.get("year") and sample_meta.get("inferred_year_from_table"):
            target["year"] = int(sample_meta["inferred_year_from_table"])
            target["year_source"] = sample_meta.get("inferred_year_source") or "sample_table"
            year_inference_notes.append({
                "method": "sample_table",
                "year": int(target["year"]),
                "source": target["year_source"],
                "message": f"未在请求和文件名中识别到年份，已根据样点表字段推断为 {int(target['year'])} 年。",
            })

        # 默认不允许静默使用当前年份，避免把 2026 之类错误年份传给 GEE/Earthdata。
        if not target.get("year"):
            policy = os.getenv("PRO_YEAR_MISSING_POLICY", "infer_or_stop").strip().lower()
            if policy in {"current", "infer_or_current"}:
                fallback_year = int(os.getenv("PRO_DEFAULT_YEAR", str(time.localtime().tm_year)))
                target["year"] = fallback_year
                target["year_source"] = f"fallback_policy:{policy}"
                year_inference_notes.append({
                    "method": "fallback",
                    "year": fallback_year,
                    "source": target["year_source"],
                    "message": f"未识别到年份，按策略使用默认年份 {fallback_year}。",
                })
            else:
                raise RuntimeError("未识别到制图年份。请在问题中写明年份，例如：2021年成都市土壤有机质图；或将样点文件命名为 2021样点.csv。")

        ts = time.strftime("%Y%m%d_%H%M%S")
        work_dir = _compact_run_dir(self.out_root, target.get("region"), target.get("year") or "unknown", ts)
        raw_dir = work_dir / "raw"
        std_dir = work_dir / "std"
        meta_dir = work_dir / "meta"
        out_dir = work_dir / "out"
        max_win_path = int(os.getenv("PRO_MAX_WINDOWS_PATH_LEN", "240"))
        if os.name == "nt" and _path_len(std_dir) >= max_win_path:
            short_root = Path(os.getenv("PRO_SHORT_OUTPUT_ROOT", r"E:\Agent_DSM\runs"))
            work_dir = _compact_run_dir(short_root, target.get("region"), target.get("year") or "unknown", ts)
            raw_dir = work_dir / "raw"
            std_dir = work_dir / "std"
            meta_dir = work_dir / "meta"
            out_dir = work_dir / "out"
            _emit("PATH", "检测到输出路径过长，已自动切换到短输出目录，避免 WinError 206。", {
                "short_output_root": str(short_root),
                "work_dir": str(work_dir),
                "path_length": _path_len(std_dir),
            }, task_id=self.task_id)
        for p in [raw_dir, std_dir, meta_dir, out_dir]:
            try:
                p.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                if os.name == "nt" and getattr(e, "winerror", None) == 206:
                    raise RuntimeError(
                        "输出路径过长导致 Windows 拒绝创建目录。请把项目解压到短路径，或在 .env 中设置 "
                        r"PRO_OUTPUT_ROOT=E:\Agent_DSM\runs。当前目录：" + str(p)
                    ) from e
                raise

        if year_inference_notes:
            for note in year_inference_notes:
                _emit("YEAR", note["message"], note, task_id=self.task_id)
        if 'region_inference_notes' in locals() and region_inference_notes:
            for note in region_inference_notes:
                _emit("REGION", note.get("message", "区域识别记录"), note, task_id=self.task_id)

        _emit("PIPELINE", "正式制图流程启动：用户样点 + 本地/国内协变量", {
            "request_text": request_text,
            "sample_path": str(sample_path),
            "target": target,
            "year_inference_notes": year_inference_notes,
            "region_inference_notes": region_inference_notes if 'region_inference_notes' in locals() else [],
            "work_dir": str(work_dir),
            "allow_download": self.allow_download,
        }, task_id=self.task_id)
        _emit("COVARIATE", "环境协变量选择方案已确定", self._covariate_selection_plan, task_id=self.task_id)

        samples = samples.reset_index(drop=True).copy()
        if "sample_id" not in samples.columns:
            samples["sample_id"] = range(len(samples))
        _emit("SAMPLE", "样点CSV读取成功，准备与平台数据对齐", sample_meta, task_id=self.task_id)
        sample_check_path = meta_dir / "sample_check.json"
        sample_check_path.write_text(json.dumps(sample_meta, ensure_ascii=False, indent=2), encoding="utf-8")

        platform_list = self._select_platforms(platforms)
        statuses: list[PlatformStatus] = []
        model_df = samples.copy()

        # GEE 云端协变量抽取：不下载大影像，只把样点协变量抽取成建模CSV列。
        if self.gee_enable:
            gee_status = self._run_gee_sampler(samples, model_df, target, out_dir)
            statuses.append(gee_status)
            if self.gee_require_success and not gee_status.extracted_columns:
                raise RuntimeError("PRO_GEE_REQUIRE_SUCCESS=1，但 GEE 未成功抽取任何协变量。")

        # 分辨率最终确定：用户未指定时执行 V168 策略。
        # 规则：显式用户分辨率优先；否则取用户数据中的最低/最粗分辨率；
        # 若全部用户数据为30m或更细，则按AOI级别自动降尺度：县区30m、地级市250m、省级500m、国级1km。
        # 这一步必须发生在样点裁剪窗口和栅格标准化之前。
        cov_roots = []
        try:
            cov_roots.append(Path(sample_path).parent)
        except Exception:
            pass
        try:
            cov_roots.append(self.domestic_fallback_local_dir)
        except Exception:
            pass
        # V166: include explicit uploaded raster file paths and their parents in resolution inference.
        for _p in getattr(self, "_uploaded_covariate_files", []) or []:
            try:
                cov_roots.append(Path(_p))
                cov_roots.append(Path(_p).parent)
            except Exception:
                pass
        target = self._infer_target_resolution_from_user_rasters(cov_roots, target, sample_df=samples)

        # V77：样点范围仍用于本地备用栅格裁剪；正式 GEE 预测会优先解析行政边界并生成边界/耕地 mask。
        sample_bbox = self._sample_target_bbox(samples, target)
        aoi_payload = {"note": "V77：样点范围用于备用裁剪；GEE正式预测优先使用行政边界bounds，并输出cov_gee_admin_mask/cov_gee_cropland_mask控制最终GeoTIFF有效区。", **sample_bbox, **target}
        (meta_dir / "aoi_target_bbox.json").write_text(json.dumps(aoi_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        # V88：GEE成功后不再一概跳过国内平台。
        # 如果开题报告协变量体系里仍存在 GEE 不稳定/无法抽取的变量，则只进入国内补充分支；
        # 若没有缺口，仍保持 V73 的快速制图策略。
        gee_covariate_cols_all_pre = [c for c in model_df.columns if str(c).startswith("cov_gee_")]
        gee_model_cols_pre = [c for c in gee_covariate_cols_all_pre if c not in GEE_OUTPUT_MASK_COLUMNS]
        domestic_fallback_plan = self._prepare_domestic_fallback_plan(gee_model_cols_pre)
        domestic_fallback_needed = bool(
            self.domestic_fallback_for_missing_gee
            and domestic_fallback_plan.get("domestic_priority_missing_covariates")
        )
        if domestic_fallback_needed:
            self._inject_domestic_fallback_keywords(domestic_fallback_plan.get("domestic_keywords") or [])
            original_platform_ids = [p.id for p in platform_list]
            platform_list = [p for p in platform_list if p.group == "domestic" and p.id in self.domestic_fallback_targets]
            _emit("SOURCE", "GEE存在缺失/不稳定协变量，进入国内平台补充模式", {
                "missing_covariates": domestic_fallback_plan.get("missing_covariates"),
                "domestic_priority_missing_covariates": domestic_fallback_plan.get("domestic_priority_missing_covariates"),
                "domestic_keywords": domestic_fallback_plan.get("domestic_keywords"),
                "selected_domestic_platforms": [p.id for p in platform_list],
                "original_platforms": original_platform_ids,
                "local_dir": domestic_fallback_plan.get("local_dir"),
            }, task_id=self.task_id)
            # 本地/人工下载目录优先接入：它是最稳定的“国内平台需要登录/审核时”的工程路径。
            local_status = self._build_local_domestic_status(target)
            statuses.append(local_status)
            try:
                added_cols = self._process_platform_files(
                    PlatformSource("domestic_local", "国内协变量本地缓存/人工下载接管", str(self.domestic_fallback_local_dir), "domestic_fallback/local_raster_cache"),
                    local_status, samples, model_df, target, sample_bbox, std_dir / "domestic_local"
                )
                local_status.extracted_columns.extend(added_cols)
                local_status.sample_aligned_count = len(added_cols)
                if added_cols:
                    local_status.status = "sample_aligned"
                    if self.skip_domestic_web_after_local_success:
                        _emit("SOURCE", "本地内置/人工下载协变量已成功接入，跳过国内网页爬取以避免无效下载和长时间等待", {
                            "local_dir": str(self.domestic_fallback_local_dir),
                            "local_feature_cols": added_cols,
                            "skipped_platforms": [p.id for p in platform_list],
                        }, task_id=self.task_id)
                        platform_list = []
            except Exception as exc:
                local_status.warnings.append(f"本地国内协变量标准化/抽样失败：{exc}")
                _emit("PREPROCESS", "本地国内协变量标准化/抽样失败", {"error": str(exc)}, task_id=self.task_id)
        else:
            domestic_fallback_plan = domestic_fallback_plan if 'domestic_fallback_plan' in locals() else {}

        # V94：即使GEE协变量已成功，也允许接入 E:/Agent_DSM/data 这类本地精选协变量。
        # 用户可以在指令里选择具体协变量；未指定时使用默认环境协变量清单。
        if self.local_covariates_always_load and not any(getattr(x, "platform_id", "") == "domestic_local" for x in statuses):
            if self.domestic_fallback_local_dir.exists():
                local_status = self._build_local_domestic_status(target)
                statuses.append(local_status)
                try:
                    added_cols = self._process_platform_files(
                        PlatformSource("domestic_local", "内置/本地环境协变量", str(self.domestic_fallback_local_dir), "local_curated_covariates"),
                        local_status, samples, model_df, target, sample_bbox, std_dir / "domestic_local"
                    )
                    local_status.extracted_columns.extend(added_cols)
                    local_status.sample_aligned_count = len(added_cols)
                    if added_cols:
                        local_status.status = "sample_aligned"
                    _emit("COVARIATE", "本地默认/用户选择协变量已接入", {
                        "local_dir": str(self.domestic_fallback_local_dir),
                        "added_cols": added_cols,
                        "selection_plan": self._covariate_selection_plan,
                    }, task_id=self.task_id)
                except Exception as exc:
                    local_status.warnings.append(f"本地环境协变量标准化/抽样失败：{exc}")
                    _emit("PREPROCESS", "本地环境协变量标准化/抽样失败", {"error": str(exc)}, task_id=self.task_id)
            else:
                _emit("COVARIATE", "本地环境协变量目录不存在，跳过本地默认协变量", {"local_dir": str(self.domestic_fallback_local_dir)}, task_id=self.task_id)

        skip_platform_probe = bool(
            (not self.force_domestic_platform_download)
            and self.gee_priority_mode
            and self.gee_formal_flow
            and any(c.startswith("cov_gee_") for c in model_df.columns)
            and not self.probe_after_gee_success
            and not domestic_fallback_needed
        )
        if skip_platform_probe:
            _emit("SOURCE", "GEE正式协变量已成功且无国内优先补充缺口，跳过国内/国外平台前台探测以加快正式制图", {
                "gee_priority_mode": self.gee_priority_mode,
                "probe_after_gee_success": self.probe_after_gee_success,
                "platform_ids_skipped": [p.id for p in platform_list],
                "domestic_fallback_plan": domestic_fallback_plan,
            }, task_id=self.task_id)
            platform_list = []

        for src in platform_list:
            status = self._probe_and_download_platform(src, raw_dir / src.id, request_text, {**target, **sample_bbox})
            statuses.append(status)
            # OSM/Overpass 是轻量矢量源，直接派生样点特征，不走栅格标准化。
            if src.api_mode == "osm_overpass":
                try:
                    added_cols = self._process_osm_features(src, status, samples, model_df)
                    status.extracted_columns.extend(added_cols)
                    status.sample_aligned_count = len(added_cols)
                    if added_cols:
                        status.status = "sample_aligned"
                    continue
                except Exception as exc:
                    status.warnings.append(f"OSM特征派生失败：{exc}")
                    _emit("PREPROCESS", f"{src.name} OSM特征派生失败", {"error": str(exc)}, task_id=self.task_id)
            # 扫描下载/保存文件，尝试标准化栅格，并抽取到样点。
            try:
                added_cols = self._process_platform_files(src, status, samples, model_df, target, sample_bbox, std_dir / src.id)
                status.extracted_columns.extend(added_cols)
                status.sample_aligned_count = len(added_cols)
                if added_cols:
                    status.status = "sample_aligned"
            except Exception as exc:
                status.warnings.append(f"标准化/抽样失败：{exc}")
                _emit("PREPROCESS", f"{src.name} 标准化/抽样失败", {"error": str(exc)}, task_id=self.task_id)

        real_covariate_cols = [c for c in model_df.columns if c.startswith("cov_")]
        # V164 fused-CSV covariates are preserved from the uploaded training table and
        # may not start with cov_. They are valid model features, but they cannot by
        # themselves generate a prediction map unless matching rasters are also supplied.
        fused_covariate_cols = [c for c in (sample_meta.get("preserved_fused_covariates") or []) if c in model_df.columns]
        domestic_covariate_cols_all = [c for c in real_covariate_cols if c.startswith("cov_") and not c.startswith("cov_gee_")]
        # V173: CLCD is mask-only and must not make the formal model think it has
        # a usable environmental predictor. Keep CLCD raster in the raster map for
        # cropland masking, but remove it from domestic model feature candidates.
        domestic_covariate_cols = [c for c in domestic_covariate_cols_all if not (self._clcd_mask_only_policy() and self._is_clcd_model_feature(c))]
        domestic_true_download_count = sum(int(getattr(s, "true_raw_downloaded_count", 0) or 0) for s in statuses if str(getattr(s, "platform_id", "")).lower() in self.domestic_fallback_targets)
        if self.force_domestic_platform_download and domestic_true_download_count < self.domestic_true_data_min_downloads:
            payload = self._build_domestic_manual_handoff_payload(
                target=target,
                statuses=statuses,
                raw_dir=raw_dir,
                meta_dir=meta_dir,
                request_text=request_text,
                required_downloads=self.domestic_true_data_min_downloads,
                actual_downloads=domestic_true_download_count,
            )
            handoff_path = meta_dir / "domestic_manual_handoff.json"
            handoff_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            msg = (
                "国内平台数据下载未完成，已停止制图。"
                "系统只把 GeoTIFF/NetCDF/HDF/ZIP/CSV 等真实数据文件计入下载；"
                "HTML页面、登录页、网页按钮地址和普通JS/API界面不会计入。"
                f" 请在弹窗中打开国内平台并完成登录/注册/点击下载，保存到指定目录后重新提交。handoff={handoff_path}"
            )
            raise RuntimeError(DOMESTIC_MANUAL_REQUIRED_MARKER + json.dumps(payload, ensure_ascii=False) + "\n" + msg)
        if domestic_fallback_needed and self.domestic_fallback_required and not domestic_covariate_cols:
            raise RuntimeError("PRO_DOMESTIC_FALLBACK_REQUIRED=1，但国内平台/本地缓存未成功接入任何可抽样协变量。")
        gee_covariate_cols_all = [c for c in real_covariate_cols if c.startswith("cov_gee_")]
        # V77：行政边界/耕地掩膜列只控制输出有效区，不作为模型训练特征。
        gee_covariate_cols = [c for c in gee_covariate_cols_all if c not in GEE_OUTPUT_MASK_COLUMNS]
        has_gee_model_features = bool(self.gee_formal_flow and gee_covariate_cols)
        has_domestic_model_features = bool(domestic_covariate_cols)
        has_fused_csv_features = bool(fused_covariate_cols)
        formal_covariate_model = bool(has_gee_model_features or has_domestic_model_features or has_fused_csv_features)

        # 即使GEE正式模型已可用，也保留平台状态列和mask列写入CSV供诊断；
        # 但正式建模feature_cols只允许真实环境协变量，避免sample_id/src_*/coord_*和mask伪相关。
        if self.use_fallback_features:
            self._append_platform_status_features(model_df, statuses)
            if not formal_covariate_model:
                self._append_coordinate_diagnostic_features(model_df)

        if formal_covariate_model:
            if has_gee_model_features and self.include_domestic_covariates_in_formal_model:
                feature_cols = self._dedupe_strings(list(gee_covariate_cols) + list(domestic_covariate_cols))
            elif has_gee_model_features:
                feature_cols = gee_covariate_cols
            elif has_domestic_model_features:
                # V91: if GEE fails but E:/Agent_DSM/data/local rasters are available,
                # formal mode must still proceed with real raster covariates instead of stopping.
                feature_cols = domestic_covariate_cols
            else:
                # V164: direct fused CSV mode. The CSV already contains extracted
                # environmental covariates at sample locations.
                feature_cols = fused_covariate_cols
            before_selection_cols = list(feature_cols)
            feature_cols = self._filter_feature_cols_by_covariate_selection(feature_cols)
            _emit("MODEL", "启用正式建模特征集：真实环境协变量；mask仅用于输出裁剪，不参与建模", {
                "feature_cols_before_covariate_selection": before_selection_cols,
                "feature_cols": feature_cols,
                "gee_feature_cols": gee_covariate_cols,
                "domestic_feature_cols": domestic_covariate_cols,
                "domestic_feature_cols_all": domestic_covariate_cols_all,
                "fused_csv_feature_cols": fused_covariate_cols,
                "gee_mask_cols": [c for c in gee_covariate_cols_all if c in GEE_OUTPUT_MASK_COLUMNS],
                "include_domestic_covariates_in_formal_model": self.include_domestic_covariates_in_formal_model,
                "feature_source_mode": ("gee_plus_domestic" if (has_gee_model_features and domestic_covariate_cols) else ("gee_only" if has_gee_model_features else ("local_domestic_only" if has_domestic_model_features else "fused_csv_only"))),
                "covariate_selection_plan": self._covariate_selection_plan,
                "clcd_policy": "mask_only_not_model_feature" if self._clcd_mask_only_policy() else "categorical_one_hot",
            }, task_id=self.task_id)
        else:
            feature_cols = [c for c in model_df.columns if c not in {"som"}]
        if self.require_real_raster and not real_covariate_cols and not fused_covariate_cols:
            raise RuntimeError("未从任何平台得到可抽样的真实栅格协变量，也未在融合CSV中检测到环境协变量；PRO_REQUIRE_REAL_RASTER_FOR_MODEL_CSV=1，因此停止建模CSV生成。")

        model_df, feature_cols, model_ready_audit = self._prepare_model_ready_table(model_df, feature_cols, out_dir, target)
        model_ready_csv = out_dir / "建模样点表.csv"
        model_df.to_csv(model_ready_csv, index=False, encoding="utf-8-sig")
        self._model_ready_csv = str(model_ready_csv)
        target["model_ready_table"] = {"path": str(model_ready_csv), **(model_ready_audit or {})}
        _emit("ALIGN", "建模样点表.csv已生成，RFK将以该CSV作为唯一训练入口", {
            "path": str(model_ready_csv),
            "rows": int(len(model_df)),
            "cols": int(len(model_df.columns)),
            "real_covariate_cols": real_covariate_cols,
            "feature_cols": feature_cols,
            "audit": model_ready_audit,
        }, task_id=self.task_id)

        status_rows = [asdict(s) for s in statuses]
        platform_report_json = meta_dir / "data_source_audit_report.json"
        platform_report_csv = meta_dir / "data_source_audit_report.csv"
        platform_report_json.write_text(json.dumps({
            "created_at": int(time.time()),
            "target": target,
            "sample_meta": sample_meta,
            "year_inference_notes": year_inference_notes,
            "region_inference_notes": region_inference_notes if 'region_inference_notes' in locals() else [],
            "model_ready_csv": str(model_ready_csv),
            "real_covariate_count": len(real_covariate_cols),
            "gee_covariate_cols_all": gee_covariate_cols_all if 'gee_covariate_cols_all' in locals() else [],
            "gee_model_feature_cols": gee_covariate_cols if 'gee_covariate_cols' in locals() else [],
            "gee_output_mask_cols": [c for c in real_covariate_cols if c in GEE_OUTPUT_MASK_COLUMNS],
            "domestic_covariate_cols": domestic_covariate_cols if 'domestic_covariate_cols' in locals() else [],
            "fused_csv_covariate_cols": fused_covariate_cols if 'fused_covariate_cols' in locals() else [],
            "domestic_fallback_plan": domestic_fallback_plan if 'domestic_fallback_plan' in locals() else {},
            "statuses": status_rows,
            "warnings": self.warnings,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(status_rows).to_csv(platform_report_csv, index=False, encoding="utf-8-sig")

        model_real_covariate_cols = feature_cols if formal_covariate_model else real_covariate_cols
        _emit("MODEL", "即将启动RFK训练/验证/正式出图阶段", {
            "model_ready_csv": str(model_ready_csv),
            "upstream_feature_count": int(len(feature_cols or [])),
            "upstream_features_head": [str(x) for x in (feature_cols or [])[:20]],
            "real_covariate_count": int(len(real_covariate_cols or [])),
            "fused_csv_covariate_count": int(len(fused_covariate_cols or [])) if 'fused_covariate_cols' in locals() else 0,
            "note": "下一步会由Clean RFK核心重新读取建模样点表，并自动识别融合CSV中的有效训练诊断协变量。"
        }, task_id=self.task_id)
        model_report_json, pred_tif = self._train_formal_model(model_df, feature_cols, out_dir, target, model_real_covariate_cols)
        report_md = meta_dir / "formal_mapping_report.md"
        report_md.write_text(self._build_report(target, sample_meta, statuses, model_ready_csv, model_report_json, pred_tif, real_covariate_cols, feature_cols, [c for c in gee_covariate_cols_all if c in GEE_OUTPUT_MASK_COLUMNS]), encoding="utf-8")

        ok = bool(model_ready_csv.exists() and model_report_json)
        _emit("PIPELINE", "正式制图流程结束", {
            "ok": ok,
            "model_ready_csv": str(model_ready_csv),
            "raw_covariate_count": len(real_covariate_cols),
            "model_feature_count": len(feature_cols),
            "gee_mask_column_count": len([c for c in gee_covariate_cols_all if c in GEE_OUTPUT_MASK_COLUMNS]),
            "domestic_covariate_count": len(domestic_covariate_cols) if 'domestic_covariate_cols' in locals() else 0,
            "model_report_json": str(model_report_json) if model_report_json else None,
            "pred_tif": str(pred_tif) if pred_tif else None,
        }, task_id=self.task_id)
        return PipelineResult(
            ok=ok,
            status="done" if ok else "failed",
            work_dir=str(work_dir),
            model_ready_csv=str(model_ready_csv),
            platform_report_csv=str(platform_report_csv),
            platform_report_json=str(platform_report_json),
            report_md=str(report_md),
            model_report_json=str(model_report_json) if model_report_json else None,
            pred_tif=str(pred_tif) if pred_tif else None,
            statuses=status_rows,
            warnings=self.warnings,
        )

    def _static_download_url_assessment(self, url: str) -> dict[str, Any]:
        """V108：先把“看起来像下载按钮的网页地址”和“真实数据文件地址”分开。

        这是静态判别；最终仍以 HEAD/GET + 文件头分类为准。
        """
        parsed = urlparse(url or "")
        path = (parsed.path or "").lower()
        query = (parsed.query or "").lower()
        suffix = Path(path).suffix.lower()
        full = f"{path}?{query}"
        if suffix in DOWNLOAD_EXTS:
            return {"level": "probable_direct_data_file", "reason": f"URL路径含真实数据扩展名 {suffix}"}
        if any(ext in full for ext in DOWNLOAD_EXTS):
            return {"level": "probable_file_in_query", "reason": "URL参数中包含真实数据文件扩展名"}
        if any(k in full for k in ["download", "downfile", "getfile", "export", "fileid", "file_id", "attachment"]):
            return {"level": "download_interface_or_api_candidate", "reason": "URL含下载/API关键词但没有数据文件扩展名，需用HTTP响应和文件头验证"}
        return {"level": "web_page_or_unknown", "reason": "未发现真实数据扩展名或明确下载API标识"}

    def _build_domestic_manual_handoff_payload(self, target: dict[str, Any], statuses: list[PlatformStatus], raw_dir: Path, meta_dir: Path, request_text: str, required_downloads: int, actual_downloads: int) -> dict[str, Any]:
        platform_list = [p for p in default_platform_sources() if p.group == "domestic" and p.id in self.domestic_fallback_targets]
        custom_urls = [x.strip() for x in re.split(r"[,;\n]+", os.getenv("PRO_DOMESTIC_HANDOFF_PLATFORM_URLS", "")) if x.strip().startswith(("http://", "https://"))]
        platforms = []
        for p in platform_list:
            platforms.append({"id": p.id, "name": p.name, "url": p.url, "role": p.role})
        for i, u in enumerate(custom_urls, start=1):
            platforms.append({"id": f"custom_{i}", "name": f"用户配置国内平台入口 {i}", "url": u, "role": "manual_domestic_download"})
        status_rows = [asdict(x) for x in statuses]
        return {
            "version": "V108",
            "type": "domestic_manual_handoff",
            "title": "需要人工接管国内平台下载",
            "message": "当前任务硬性要求使用国内平台下载真实数据文件。系统未获得足够的真实数据文件，已停止制图，等待用户登录/注册/点击下载后重试。",
            "request_text": request_text,
            "target": target,
            "required_true_downloads": int(required_downloads),
            "actual_true_downloads": int(actual_downloads),
            "capture_dir": str(self.playwright_capture_root / "all"),
            "platform_specific_capture_dir": str(self.playwright_capture_root),
            "raw_probe_dir": str(raw_dir),
            "report_dir": str(meta_dir),
            "platforms": platforms,
            "statuses": status_rows,
            "true_link_rules": [
                "只把 GeoTIFF/TIFF、NetCDF、HDF/H5、ZIP/7Z/RAR、CSV/Excel、GeoJSON 等真实数据文件计入。",
                "Content-Type 为 text/html、文件头为 HTML、登录页、检索页、下载按钮页面、JS/CSS/站点配置 JSON 一律不计入真实下载。",
                "没有后缀但包含 download/export/getfile/fileid 的地址只作为候选下载 API，必须通过响应头和文件头二次验证。",
                "若平台要求注册、登录、验证码或人工审核，请在弹窗中打开平台，由用户手动完成，再把下载文件保存到 capture_dir。",
            ],
            "next_steps": [
                "点击弹窗中的国内平台入口。",
                "如需要注册、登录、验证码或申请权限，由用户在浏览器中完成。",
                "下载 GeoTIFF/NetCDF/HDF/ZIP/CSV 等真实数据文件，不要只复制网页按钮地址。",
                "把文件保存到 capture_dir，或设置 PRO_<平台ID>_RAW_URLS 为真实文件直链。",
                "回到智能体重新发送同一制图请求。",
            ],
        }

    def _expected_thesis_covariates_for_audit(self) -> list[str]:
        """开题报告协变量体系的工程化审计清单。

        这里不是强制所有变量都必须在 GEE 中出现。它用于判断：
        - GEE 已经拿到了哪些正式协变量；
        - 哪些变量更适合转入国内平台补充；
        - 最终报告里必须透明记录缺口，而不是伪装变量齐全。
        """
        base = [
            # 地形地貌
            "cov_gee_elevation", "cov_gee_slope", "cov_gee_sin_aspect", "cov_gee_cos_aspect",
            "cov_gee_tpi_3x3", "cov_gee_roughness_3x3", "cov_gee_relief_5x5",
            "cov_gee_curvature_laplacian", "cov_gee_twi_proxy",
            # 植被/作物生长代理
            "cov_gee_ndvi_mean", "cov_gee_ndvi_std", "cov_gee_ndvi_range",
            "cov_gee_evi_mean", "cov_gee_evi_std", "cov_gee_evi_range",
            "cov_gee_npp",
            # 气候水热
            "cov_gee_precip_annual", "cov_gee_precip_growing", "cov_gee_temp_mean",
            "cov_gee_temp_range", "cov_gee_pet_annual", "cov_gee_climate_water_balance",
            # 土壤背景代理
            "cov_gee_sand", "cov_gee_clay", "cov_gee_silt", "cov_gee_ph",
            "cov_gee_bulk_density", "cov_gee_cec",
        ]
        # 开题报告中明确提出，但 GEE 公共目录通常不稳定具备或不是标准可直接调用产品的变量。
        domestic = [
            "cov_domestic_irrigation",
            "cov_domestic_multiple_cropping_index",
            "cov_domestic_cropland_use_intensity",
            "cov_domestic_crop_planting_pattern",
            "cov_domestic_fvc",
        ]
        return base + domestic

    def _domestic_keywords_for_missing_gee(self, missing_cols: list[str]) -> list[str]:
        """把缺失协变量转换为国内平台爬取/检索关键词。"""
        kw_map = {
            "cov_domestic_irrigation": ["灌溉", "灌溉比例", "CIrrMap", "irrigation", "irrigated cropland"],
            "cov_domestic_multiple_cropping_index": ["复种指数", "multiple cropping", "cropping intensity", "耕地复种"],
            "cov_domestic_cropland_use_intensity": ["耕地利用强度", "土地利用强度", "cropland use intensity", "intensive use"],
            "cov_domestic_crop_planting_pattern": ["作物种植模式", "ChinaCP", "水稻 玉米 小麦", "cropping pattern"],
            "cov_domestic_fvc": ["植被覆盖度", "FVC", "fractional vegetation cover", "MODIS FVC"],
            "cov_gee_sand": ["砂粒", "sand", "土壤质地"],
            "cov_gee_silt": ["粉粒", "silt", "土壤质地"],
            "cov_gee_clay": ["黏粒", "clay", "土壤质地"],
            "cov_gee_ph": ["土壤pH", "soil pH"],
            "cov_gee_bulk_density": ["土壤容重", "bulk density"],
            "cov_gee_cec": ["阳离子交换量", "CEC"],
            "cov_gee_npp": ["NPP", "净初级生产力", "MOD17"],
            "cov_gee_pet_annual": ["潜在蒸散发", "PET", "蒸散发"],
            "cov_gee_temp_mean": ["气温", "温度", "temperature"],
            "cov_gee_precip_annual": ["降水", "precipitation", "CHIRPS"],
        }
        out: list[str] = []
        for c in missing_cols:
            out.extend(kw_map.get(c, []))
        # 保留基础关键词，避免只有英文/过窄词导致国内站点扫不到。
        out.extend(["下载", "数据", "栅格", "GeoTIFF", "tif", "成都市", "四川", "耕地"])
        return self._dedupe_strings([x for x in out if x])

    def _prepare_domestic_fallback_plan(self, gee_model_cols: list[str]) -> dict[str, Any]:
        expected = self._expected_thesis_covariates_for_audit()
        present = set(gee_model_cols or [])
        # GEE 里本来就不稳定的国内/农田利用变量永远进入国内补充候选；
        # 其他 thesis GEE 变量只有在当前环境实际缺失时才进入候选。
        missing = [c for c in expected if c not in present]
        domestic_priority = [c for c in missing if c.startswith("cov_domestic_") or c in {
            "cov_gee_sand", "cov_gee_silt", "cov_gee_clay", "cov_gee_ph", "cov_gee_bulk_density", "cov_gee_cec",
            "cov_gee_npp", "cov_gee_pet_annual", "cov_gee_temp_mean", "cov_gee_precip_annual",
        }]
        keywords = self._domestic_keywords_for_missing_gee(domestic_priority)
        return {
            "enabled": bool(self.domestic_fallback_for_missing_gee),
            "expected_covariates": expected,
            "present_gee_covariates": sorted(present),
            "missing_covariates": missing,
            "domestic_priority_missing_covariates": domestic_priority,
            "domestic_keywords": keywords,
            "local_dir": str(self.domestic_fallback_local_dir),
        }

    def _inject_domestic_fallback_keywords(self, keywords: list[str]) -> None:
        merged = self._dedupe_strings(list(self.domestic_raw_keywords) + list(keywords or []))
        self.domestic_raw_keywords = merged

    def _build_local_domestic_status(self, target: dict[str, Any]) -> PlatformStatus:
        """把本地/半自动下载的国内协变量文件作为一个正式国内数据源接入。"""
        status = PlatformStatus(
            platform_id="domestic_local",
            platform_name="国内协变量本地缓存/人工下载接管",
            url=str(self.domestic_fallback_local_dir),
            role="domestic_fallback/local_raster_cache",
            reachable=self.domestic_fallback_local_dir.exists(),
            status="queued",
            message="检查国内协变量本地缓存目录。",
        )
        if not self.domestic_fallback_local_dir.exists():
            status.status = "skipped"
            status.message = f"本地国内协变量目录不存在：{self.domestic_fallback_local_dir}。如国内平台需登录/人工审核，可把下载的TIF/NC/HDF/ZIP放入该目录。"
            status.warnings.append(status.message)
            return status
        files = self._scan_files([self.domestic_fallback_local_dir])
        status.downloaded_files = [str(f) for f in files]
        status.downloaded_count = len(files)
        status.true_raw_downloaded_count = len(files)
        status.status = "file_downloaded" if files else "skipped"
        status.message = f"已发现本地国内协变量文件 {len(files)} 个。" if files else f"目录存在但未发现可处理文件：{self.domestic_fallback_local_dir}。"
        return status

    def _run_gee_sampler(self, samples: Any, model_df: Any, target: dict[str, Any], out_dir: Path) -> PlatformStatus:
        """调用 GEE 云端协变量抽取，并把结果列写入 model_df。"""
        status = PlatformStatus(
            platform_id="gee",
            platform_name="Google Earth Engine 云端协变量抽取",
            url="earthengine://sampleRegions",
            role="cloud_sampler/elevation/ndvi/evi",
            reachable=False,
            status="queued",
            message="GEE待执行",
        )
        year = target.get("year")
        if not year:
            status.status = "skipped"
            status.message = "未识别到目标年份，跳过GEE协变量抽取。V69正常流程应在run()阶段完成年份推断或停止任务。"
            status.warnings.append(status.message)
            return status
        gee_dir = out_dir / "gee_sampler"
        gee_dir.mkdir(parents=True, exist_ok=True)
        try:
            from services.cloud_sampler.gee_sampler import GeeSampler
            sampler = GeeSampler(
                project_id=self.gee_project,
                log_fn=lambda stage, msg, payload=None: _emit(stage, msg, payload, task_id=self.task_id),
            )
            result = sampler.sample_to_dataframe(samples[["sample_id", "lon", "lat", "som"]].copy(), int(year), target=target)
            status.reachable = True
            status.http_status = 200
            if not result.ok or result.dataframe is None or not result.columns:
                status.status = "error"
                status.message = result.message
                status.warnings.append(result.message)
                return status
            gee_df = result.dataframe.copy()
            out_csv = gee_dir / "gee_model_ready_covariates.csv"
            gee_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
            # Persist the exact GEE covariate plan for reproducibility.
            try:
                (gee_dir / "gee_covariate_plan.json").write_text(
                    json.dumps(result.meta.get("stack", result.meta), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
            status.downloaded_files.append(str(out_csv))
            status.processed_table_count = 1
            added_cols: list[str] = []
            # 按 sample_id 合并，避免 GEE 返回顺序变化。
            if "sample_id" in gee_df.columns and "sample_id" in model_df.columns:
                merge_cols = ["sample_id"] + [c for c in result.columns if c in gee_df.columns]
                tmp = gee_df[merge_cols].drop_duplicates(subset=["sample_id"])
                tmp = tmp.set_index("sample_id")
                for c in result.columns:
                    if c in tmp.columns:
                        model_df[c] = model_df["sample_id"].map(tmp[c])
                        added_cols.append(c)
            else:
                # 兜底：长度相同则按行赋值。
                if len(gee_df) != len(model_df):
                    raise RuntimeError(f"GEE返回行数 {len(gee_df)} 与样点数 {len(model_df)} 不一致，且缺少sample_id，无法安全合并。")
                for c in result.columns:
                    model_df[c] = list(gee_df[c])
                    added_cols.append(c)
            valid_counts = {c: int(pd.to_numeric(model_df[c], errors="coerce").notna().sum()) for c in added_cols}
            status.extracted_columns.extend(added_cols)
            status.sample_aligned_count = len(added_cols)
            status.true_raw_downloaded_count = 0
            status.downloaded_count = 0
            status.status = "sample_aligned" if added_cols else "reachable"
            status.message = f"GEE云端协变量抽取完成，新增协变量列 {len(added_cols)} 个。"
            status.saved_html = str(out_csv)
            _emit("GEE", "GEE协变量已写入建模CSV", {
                "columns": added_cols,
                "valid_counts": valid_counts,
                "out_csv": str(out_csv),
                "note": "GEE路线不下载大影像，只回传样点协变量表。",
            }, task_id=self.task_id)
            return status
        except Exception as exc:
            status.status = "error"
            status.message = f"GEE协变量抽取失败：{exc}"
            status.warnings.append(status.message)
            _emit("GEE", "GEE协变量抽取失败", {"error": str(exc), "project": self.gee_project}, task_id=self.task_id)
            return status

    def _sample_target_bbox(self, samples: Any, target: dict[str, Any]) -> dict[str, Any]:
        lon_min, lon_max = float(samples["lon"].min()), float(samples["lon"].max())
        lat_min, lat_max = float(samples["lat"].min()), float(samples["lat"].max())
        lon_pad = max((lon_max - lon_min) * 0.15, 0.02)
        lat_pad = max((lat_max - lat_min) * 0.15, 0.02)
        bbox4326 = [lon_min - lon_pad, lat_min - lat_pad, lon_max + lon_pad, lat_max + lat_pad]
        target_bbox = None
        if Transformer is not None:
            try:
                tr = Transformer.from_crs("EPSG:4326", target["target_crs"], always_xy=True)
                xs, ys = [], []
                for x in [bbox4326[0], bbox4326[2]]:
                    for y in [bbox4326[1], bbox4326[3]]:
                        xx, yy = tr.transform(x, y)
                        xs.append(xx); ys.append(yy)
                buf = float(target["resolution_m"]) * 3
                target_bbox = [min(xs) - buf, min(ys) - buf, max(xs) + buf, max(ys) + buf]
            except Exception as exc:
                self.warnings.append(f"样点范围投影到目标CRS失败：{exc}")
        return {"bbox_epsg4326": bbox4326, "bbox_target_crs": target_bbox}

    def _probe_and_download_platform(self, src: PlatformSource, out_dir: Path, request_text: str, target: dict[str, Any]) -> PlatformStatus:
        out_dir.mkdir(parents=True, exist_ok=True)
        status = PlatformStatus(platform_id=src.id, platform_name=src.name, url=src.url, role=src.role)
        _emit("SOURCE", f"开始审计数据源：{src.name}", {"url": src.url, "role": src.role}, task_id=self.task_id)
        try:
            foreign_download_mode = os.getenv("PRO_FOREIGN_DOWNLOAD_MODE", "metadata_only").strip().lower()
            if src.group == "foreign" and foreign_download_mode in {"off", "metadata_only", "0", "false"}:
                # 国外平台保留连通性/元数据，不下载真实文件，避免网络/认证/EULA导致主线长时间等待。
                if src.api_mode == "earthdata_cmr":
                    return self._probe_earthdata_cmr(src, out_dir, request_text, target, status, metadata_only=True)
                if src.api_mode == "figshare_api":
                    return self._probe_figshare(src, out_dir, request_text, target, status)

            if self.domestic_raw_crawl and src.id in self.domestic_raw_targets:
                return self._probe_domestic_raw_portal(src, out_dir, request_text, target, status)

            if src.api_mode == "earthdata_cmr":
                return self._probe_earthdata_cmr(src, out_dir, request_text, target, status)
            if src.api_mode == "figshare_api":
                return self._probe_figshare(src, out_dir, request_text, target, status)
            if src.api_mode == "osm_overpass":
                return self._probe_osm_overpass(src, out_dir, request_text, target, status)

            r = self.session.get(src.url, timeout=self.timeout)
            status.http_status = r.status_code
            status.reachable = r.ok
            ct = r.headers.get("content-type", "")
            suffix = ".html" if "html" in ct.lower() else Path(urlparse(src.url).path).suffix or ".dat"
            p = out_dir / (src.id + suffix)
            p.write_bytes(r.content)
            status.saved_html = str(p)
            if not r.ok:
                status.status = "error"
                status.message = f"HTTP {r.status_code}"
                return status
            text = r.text if "html" in ct.lower() or suffix.endswith((".html", ".htm", ".shtml")) else ""
            if "javascript" in text.lower() and len(text) < 2000:
                status.warnings.append("页面可能依赖 JavaScript，requests 只能保存入口页；后续需 Playwright 适配。")
            links = self._extract_links(text, src.url) if text else []
            status.link_count = len(links)
            download_links = [(t, u) for t, u in links if self._is_download_url(u)]
            status.download_link_count = len(download_links)
            status.status = "reachable"
            status.message = f"平台可访问，发现链接 {len(links)} 个，候选下载链接 {len(download_links)} 个。"
            if self.allow_download and download_links:
                for i, (txt, link) in enumerate(download_links[: self.max_downloads_per_platform], start=1):
                    dl = self._download_file(link, out_dir, f"{src.id}_{i}_{_safe_name(txt, 32)}")
                    if dl:
                        status.downloaded_files.append(str(dl))
                        status.downloaded_count += 1
                if status.downloaded_count:
                    status.status = "file_downloaded"
            return status
        except Exception as exc:
            status.status = "error"
            status.message = str(exc)
            _emit("SOURCE", f"数据源审计失败：{src.name}", {"error": str(exc)}, task_id=self.task_id)
            return status

    def _probe_domestic_raw_portal(self, src: PlatformSource, out_dir: Path, request_text: str, target: dict[str, Any], status: PlatformStatus) -> PlatformStatus:
        """GSCloud/NODA 国内平台原始文件专项爬取。

        目标不是保证每次都能绕过登录/JS/审核，而是把“国内平台能否真实发现并下载原始文件”拆成可观测步骤：
        1) 保存入口页；2) 递归爬取少量站内候选页；3) 扫描 HTML/JS 中的下载/API 线索；
        4) 下载可直接访问的小文件/原始文件；5) 支持用户从平台网页复制出的 seeded URL 自动下载。
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        status.status = "probing_raw"
        status.message = "国内平台原始文件专项爬取启动。"
        max_pages = self.domestic_crawl_max_pages
        max_depth = self.domestic_crawl_depth
        same_host = urlparse(src.url).netloc
        visited: set[str] = set()
        queue: list[tuple[str, int, str]] = [(src.url, 0, "entry")]
        all_links: list[tuple[str, str]] = []
        all_download_links: list[tuple[str, str]] = []
        all_api_candidates: list[str] = []
        page_records: list[dict[str, Any]] = []

        # 允许用户把浏览器里复制出来的真实下载 URL 放进 .env，便于先证明“后端可接管下载/标准化”。
        seeded = self._seeded_domestic_urls(src.id)
        for u in seeded:
            all_download_links.append(("seeded_env_url", u))

        _emit("SOURCE", f"{src.name} 国内原始文件专项爬取启动", {
            "url": src.url,
            "max_pages": max_pages,
            "max_depth": max_depth,
            "keywords": self.domestic_raw_keywords,
            "seeded_url_count": len(seeded),
        }, task_id=self.task_id)

        while queue and len(visited) < max_pages:
            url, depth, why = queue.pop(0)
            if url in visited:
                continue
            if urlparse(url).netloc and urlparse(url).netloc != same_host:
                continue
            visited.add(url)
            try:
                r = self.session.get(url, timeout=self.timeout)
                status.http_status = r.status_code
                if r.ok:
                    status.reachable = True
                suffix = Path(urlparse(url).path).suffix
                if not suffix or len(suffix) > 8:
                    suffix = ".html"
                page_path = out_dir / f"page_{len(visited):03d}_{_safe_name(why, 30)}{suffix}"
                page_path.write_bytes(r.content)
                ct = r.headers.get("content-type", "")
                text = r.text if ("html" in ct.lower() or "javascript" in ct.lower() or suffix in {".html", ".htm", ".js"}) else ""
                links = self._extract_links(text, url) if text else []
                script_links = self._extract_script_urls(text, url) if text else []
                api_candidates = self._extract_url_candidates_from_text(text, url) if text else []
                api_candidates.extend(script_links)
                all_api_candidates.extend(api_candidates)
                all_links.extend(links)
                dl = [(t, u) for t, u in links if self._is_download_url(u)]
                # HTML/JS 中非 <a> 的下载/API URL 也要纳入候选。
                for u in api_candidates:
                    if self._is_download_url(u):
                        dl.append(("api_or_js_candidate", u))
                all_download_links.extend(dl)
                page_records.append({
                    "url": url, "depth": depth, "why": why, "status": r.status_code,
                    "saved": str(page_path), "link_count": len(links), "download_link_count": len(dl),
                    "api_candidate_count": len(api_candidates), "content_type": ct,
                })
                # 继续爬少量站内候选页：含关键词、产品、搜索、数据、下载等。
                if depth < max_depth:
                    for txt, u in links:
                        if len(queue) + len(visited) >= max_pages:
                            break
                        if u in visited:
                            continue
                        if urlparse(u).netloc and urlparse(u).netloc != same_host:
                            continue
                        if self._is_download_url(u):
                            continue
                        label = f"{txt} {u}"
                        if self._looks_like_candidate_page(label):
                            queue.append((u, depth + 1, _safe_name(txt or Path(urlparse(u).path).name or "candidate", 30)))
                # 下载/扫描 JS 文件中的 API 线索，但不深追全部 JS，避免拖慢。
                for js_url in script_links[: int(os.getenv("PRO_DOMESTIC_MAX_JS_SCAN", "5"))]:
                    try:
                        jr = self.session.get(js_url, timeout=min(self.timeout, 15))
                        if jr.ok and len(jr.content) < int(os.getenv("PRO_DOMESTIC_MAX_JS_BYTES", "2000000")):
                            js_path = out_dir / f"script_{len(page_records):03d}_{_safe_name(Path(urlparse(js_url).path).name, 40)}.js"
                            js_path.write_bytes(jr.content)
                            js_text = jr.text
                            cands = self._extract_url_candidates_from_text(js_text, js_url)
                            all_api_candidates.extend(cands)
                            for cu in cands:
                                if self._is_download_url(cu):
                                    all_download_links.append(("js_api_candidate", cu))
                    except Exception as exc:
                        status.warnings.append(f"JS扫描失败：{js_url} | {exc}")
            except Exception as exc:
                page_records.append({"url": url, "depth": depth, "why": why, "error": str(exc)})
                status.warnings.append(f"页面探测失败：{url} | {exc}")

        # 去重，保留顺序。
        all_download_links = self._dedupe_pairs(all_download_links)
        all_api_candidates = self._dedupe_strings(all_api_candidates)
        status.link_count = len(self._dedupe_pairs(all_links))
        status.download_link_count = len(all_download_links)
        status.saved_html = str(out_dir / "domestic_raw_probe_report.json")
        (out_dir / "domestic_raw_probe_report.json").write_text(json.dumps({
            "platform": src.id,
            "name": src.name,
            "entry_url": src.url,
            "visited_pages": page_records,
            "download_links": [{"text": t, "url": u, "assessment": self._static_download_url_assessment(u)} for t, u in all_download_links],
            "api_candidates": all_api_candidates[:300],
            "seeded_urls": seeded,
            "note": "若download_links为空，说明requests级爬取未发现公开直链；可能需要登录、JS渲染、人工审核或复制真实下载URL到PRO_GSCLOUD_RAW_URLS/PRO_NODA_RAW_URLS。",
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        # 下载候选原始文件。对 GSCloud/NODA 兼容慢速平台允许慢，但仍受大小阈值/超时策略控制；可用 env 放宽。
        if self.allow_download and all_download_links:
            for i, (txt, link) in enumerate(all_download_links[: self.max_downloads_per_platform], start=1):
                dl = self._download_file(link, out_dir, f"{src.id}_raw_{i}_{_safe_name(txt, 30)}")
                if dl:
                    status.downloaded_files.append(str(dl))
                    status.downloaded_count += 1
                    status.true_raw_downloaded_count += 1

        # Playwright 路线B：导入真实浏览器会话中人工/半自动点击下载捕获的文件。
        pw_imported = self._ingest_playwright_downloads(src.id, out_dir, status)
        if pw_imported:
            status.warnings.append(f"已导入 Playwright 捕获的真实下载文件 {pw_imported} 个。")

        if status.downloaded_count:
            status.status = "file_downloaded"
            status.message = f"国内平台原始文件准备完成：候选链接 {status.download_link_count} 个，真实数据文件 {status.downloaded_count} 个。"
        elif all_download_links:
            status.status = "reachable"
            status.message = f"发现候选下载链接 {status.download_link_count} 个，但下载被大小/超时/认证/网络策略跳过，或仅得到HTML/站点资源。"
        else:
            status.status = "reachable" if status.reachable else "error"
            status.message = f"国内平台可访问，爬取页面 {len(visited)} 个，发现候选下载链接 {status.download_link_count} 个。"

        _emit("SOURCE", f"{src.name} 国内原始文件专项爬取结束", {
            "reachable": status.reachable,
            "visited_pages": len(visited),
            "link_count": status.link_count,
            "download_link_count": status.download_link_count,
            "downloaded_count": status.downloaded_count,
            "true_raw_downloaded_count": status.true_raw_downloaded_count,
            "report": status.saved_html,
        }, task_id=self.task_id)
        return status

    def _ingest_playwright_downloads(self, platform_id: str, out_dir: Path, status: PlatformStatus) -> int:
        """导入 Playwright 工具捕获的真实下载文件。

        使用方式：先运行 tools/playwright_domestic_download_capture.py，人工登录平台并点击下载；
        本函数在流水线运行时把 data/manual_domestic_downloads/<platform>/ 下的真实数据文件复制进
        当前任务目录，并纳入后续标准化/样点抽取流程。
        """
        if not self.playwright_import_downloads:
            return 0
        roots: list[Path] = []
        # 平台专属目录 + all/common 目录，方便一次捕获多个平台文件。
        roots.append(self.playwright_capture_root / platform_id)
        roots.append(self.playwright_capture_root / "all")
        # 允许用户给某个平台单独指定捕获目录。
        env_key = f"PRO_{platform_id.upper()}_PLAYWRIGHT_DIR"
        if os.getenv(env_key):
            roots.append(Path(os.getenv(env_key, "")))
        imported = 0
        max_age_seconds = self.playwright_import_max_age_hours * 3600.0
        now = time.time()
        out_dir.mkdir(parents=True, exist_ok=True)
        for root in roots:
            if not root.exists():
                continue
            for f in sorted(root.rglob("*"), key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True):
                if not f.is_file():
                    continue
                if f.name.endswith((".crdownload", ".tmp", ".part")):
                    continue
                try:
                    age = now - f.stat().st_mtime
                    if max_age_seconds > 0 and age > max_age_seconds:
                        continue
                    classification = self._classify_downloaded_file(f)
                    if not classification.get("is_true_data"):
                        status.warnings.append(f"Playwright捕获文件不是可用原始数据，已跳过：{f.name} | {classification.get('reason')}")
                        continue
                    dst = out_dir / f"playwright_{platform_id}_{imported+1}_{_safe_name(f.name, 60)}"
                    # 保留原扩展名，避免后续 rasterio/pandas 无法识别。
                    if not dst.suffix and f.suffix:
                        dst = dst.with_suffix(f.suffix)
                    shutil.copy2(f, dst)
                    status.downloaded_files.append(str(dst))
                    status.downloaded_count += 1
                    status.true_raw_downloaded_count += 1
                    imported += 1
                    _emit("PLAYWRIGHT", "已导入浏览器捕获的真实下载文件", {
                        "platform": platform_id,
                        "source": str(f),
                        "dest": str(dst),
                        "bytes": dst.stat().st_size,
                        "classification": classification,
                    }, task_id=self.task_id)
                    if imported >= self.max_downloads_per_platform:
                        return imported
                except Exception as exc:
                    status.warnings.append(f"Playwright捕获文件导入失败：{f} | {exc}")
        return imported

    def _seeded_domestic_urls(self, platform_id: str) -> list[str]:
        keys = [f"PRO_{platform_id.upper()}_RAW_URLS", "PRO_DOMESTIC_RAW_URLS"]
        if platform_id == "gscloud":
            keys.insert(0, "PRO_GSCLOUD_RAW_URLS")
        elif platform_id == "noda":
            keys.insert(0, "PRO_NODA_RAW_URLS")
        vals: list[str] = []
        for k in keys:
            raw = os.getenv(k, "")
            vals.extend([x.strip() for x in re.split(r"[,\n;]+", raw) if x.strip().startswith(("http://", "https://"))])
        return self._dedupe_strings(vals)

    def _looks_like_candidate_page(self, label: str) -> bool:
        lo = (label or "").lower()
        if any(k.lower() in lo for k in self.domestic_raw_keywords):
            return True
        return any(k in lo for k in ["source", "product", "dataset", "data", "search", "query", "download", "api", "detail", "catalog", "资源", "产品", "数据", "下载", "检索", "详情"])

    def _extract_script_urls(self, html: str, base_url: str) -> list[str]:
        if not html:
            return []
        urls = []
        for src in re.findall(r"<script[^>]+src=[\"']([^\"']+)[\"']", html, flags=re.I):
            if src and not src.startswith(("javascript:", "#")):
                urls.append(urljoin(base_url, src))
        return self._dedupe_strings(urls)

    def _extract_url_candidates_from_text(self, text: str, base_url: str) -> list[str]:
        if not text:
            return []
        cands: list[str] = []
        # 绝对URL。
        for u in re.findall(r"https?://[^\s'\"<>\\)]+", text):
            cands.append(u.rstrip(".,;"))
        # JS中常见的相对 API / download 路径。
        for u in re.findall(r"[\"']((?:/|\.\./|\./)?[^\"']*(?:download|down|export|file|api|search|query|product|dataset|data|下载)[^\"']*)[\"']", text, flags=re.I):
            if not u or u.startswith(("#", "javascript:")):
                continue
            try:
                cands.append(urljoin(base_url, u))
            except Exception:
                pass
        # 只保留和站点相关或明显文件候选，避免无意义噪声。
        host = urlparse(base_url).netloc
        out = []
        for u in cands:
            pu = urlparse(u)
            if not pu.scheme.startswith("http"):
                continue
            if pu.netloc and pu.netloc != host and not self._is_download_url(u):
                continue
            out.append(u)
        return self._dedupe_strings(out)

    def _dedupe_pairs(self, pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        seen = set(); out = []
        for t, u in pairs:
            if not u or u in seen:
                continue
            seen.add(u); out.append((t, u))
        return out

    def _dedupe_strings(self, vals: list[str]) -> list[str]:
        seen = set(); out = []
        for v in vals:
            if not v or v in seen:
                continue
            seen.add(v); out.append(v)
        return out

    def _probe_osm_overpass(self, src: PlatformSource, out_dir: Path, request_text: str, target: dict[str, Any], status: PlatformStatus) -> PlatformStatus:
        """OpenStreetMap/Overpass 轻量矢量特征源。

        目的不是替代遥感协变量，而是先打通“真实外部数据 -> 样点特征 -> model_ready.csv”。
        默认只查询道路/水系，数据量小、无需登录，适合第一条真实协变量链路诊断。
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        if os.getenv("PRO_OSM_ENABLE", "1") != "1":
            status.status = "skipped"
            status.message = "PRO_OSM_ENABLE=0，已跳过 OSM/Overpass。"
            return status
        bbox = target.get("bbox_epsg4326") or [103.0, 30.0, 105.0, 31.5]
        west, south, east, north = map(float, bbox)
        # 为避免 Overpass 返回过大，设置范围阈值；过大时自动用样点中心小缓冲。
        max_area_deg2 = float(os.getenv("PRO_OSM_MAX_BBOX_DEG2", "3.0"))
        area = max(east - west, 0) * max(north - south, 0)
        if area > max_area_deg2:
            cx, cy = (west + east) / 2, (south + north) / 2
            half = float(os.getenv("PRO_OSM_FALLBACK_HALF_DEG", "0.5"))
            west, east, south, north = cx - half, cx + half, cy - half, cy + half
            status.warnings.append(f"OSM查询范围过大，已缩小到中心附近 {2*half}° 方框。")
        timeout_s = int(os.getenv("PRO_OSM_TIMEOUT", "25"))
        max_ways = int(os.getenv("PRO_OSM_MAX_WAYS", "1200"))
        query = f"""
[out:json][timeout:{timeout_s}];
(
  way["highway"]({south},{west},{north},{east});
  way["waterway"]({south},{west},{north},{east});
  way["natural"="water"]({south},{west},{north},{east});
);
out geom {max_ways};
""".strip()
        _emit("OSM", "开始 Overpass 查询道路/水系矢量", {"bbox": [west, south, east, north], "timeout": timeout_s, "max_ways": max_ways}, task_id=self.task_id)
        r = self.session.post(src.url, data={"data": query}, timeout=timeout_s + 5)
        status.http_status = r.status_code
        status.reachable = r.ok
        p = out_dir / "osm_overpass_roads_water.json"
        p.write_bytes(r.content)
        status.saved_html = str(p)
        if not r.ok:
            status.status = "error"
            status.message = f"Overpass HTTP {r.status_code}"
            return status
        try:
            js = r.json()
            elements = js.get("elements") or []
        except Exception as exc:
            snippet = r.text[:500] if hasattr(r, "text") else ""
            status.status = "error"
            status.message = f"Overpass JSON解析失败：{exc}"
            status.warnings.append(f"Overpass响应不是JSON，前500字符：{snippet}")
            _emit("OSM", "Overpass响应不是JSON", {"status": r.status_code, "content_type": r.headers.get("content-type"), "snippet": snippet[:300]}, task_id=self.task_id)
            return status
        status.link_count = len(elements)
        status.download_link_count = 1
        status.downloaded_count = 1
        status.processed_table_count = 1
        status.status = "file_downloaded"
        status.message = f"Overpass查询成功，elements={len(elements)}；将派生道路/水体距离样点特征。"
        _emit("OSM", "Overpass 查询成功", {"elements": len(elements), "saved": str(p)}, task_id=self.task_id)
        return status

    def _process_osm_features(self, src: PlatformSource, status: PlatformStatus, samples: Any, model_df: Any) -> list[str]:
        if not status.saved_html or not Path(status.saved_html).exists():
            return []
        try:
            data = json.loads(Path(status.saved_html).read_text(encoding="utf-8"))
        except Exception:
            data = json.loads(Path(status.saved_html).read_bytes().decode("utf-8", errors="ignore"))
        elements = data.get("elements") or []
        road_lines, water_lines = [], []
        for el in elements:
            if el.get("type") != "way" or not el.get("geometry"):
                continue
            tags = el.get("tags") or {}
            coords = [(float(pt["lon"]), float(pt["lat"])) for pt in el.get("geometry") or [] if "lon" in pt and "lat" in pt]
            if len(coords) < 2:
                continue
            if "highway" in tags:
                road_lines.append(coords)
            if "waterway" in tags or tags.get("natural") == "water":
                water_lines.append(coords)
        added: list[str] = []
        if road_lines:
            col = "cov_osm_dist_road_m"
            model_df[col] = self._dist_to_lines_m(samples, road_lines)
            added.append(col)
        if water_lines:
            col = "cov_osm_dist_water_m"
            model_df[col] = self._dist_to_lines_m(samples, water_lines)
            added.append(col)
        if added:
            _emit("ALIGN", "OSM矢量特征已抽取到样点", {"columns": added, "road_ways": len(road_lines), "water_ways": len(water_lines)}, task_id=self.task_id)
        else:
            status.warnings.append("Overpass返回成功，但未识别到可用于距离特征的 highway/waterway 线。")
        return added

    def _dist_to_lines_m(self, samples: Any, lines: list[list[tuple[float, float]]]) -> list[float | None]:
        if np is None:
            raise RuntimeError("缺少 numpy，不能计算 OSM 距离特征。")
        lat0 = float(samples["lat"].mean())
        r = 6371000.0
        cos0 = math.cos(math.radians(lat0))
        segs = []
        for line in lines:
            pts = [(math.radians(lon) * r * cos0, math.radians(lat) * r) for lon, lat in line]
            for a, b in zip(pts[:-1], pts[1:]):
                if a != b:
                    segs.append((a[0], a[1], b[0], b[1]))
        max_segments = int(os.getenv("PRO_OSM_MAX_SEGMENTS", "6000"))
        if len(segs) > max_segments:
            step = max(1, len(segs) // max_segments)
            segs = segs[::step][:max_segments]
        if not segs:
            return [None] * len(samples)
        arr = np.asarray(segs, dtype="float64")
        x1, y1, x2, y2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        dx, dy = x2 - x1, y2 - y1
        denom = dx * dx + dy * dy
        denom[denom == 0] = 1.0
        vals: list[float | None] = []
        for lon, lat in zip(samples["lon"], samples["lat"]):
            px = math.radians(float(lon)) * r * cos0
            py = math.radians(float(lat)) * r
            t = ((px - x1) * dx + (py - y1) * dy) / denom
            t = np.clip(t, 0.0, 1.0)
            projx = x1 + t * dx
            projy = y1 + t * dy
            d = np.sqrt((px - projx) ** 2 + (py - projy) ** 2)
            vals.append(float(np.nanmin(d)))
        return vals

    def _probe_figshare(self, src: PlatformSource, out_dir: Path, request_text: str, target: dict[str, Any], status: PlatformStatus) -> PlatformStatus:
        # Figshare API 搜索只做元数据，不直接下载未知附件。
        query = f"soil organic matter {target.get('region') or ''} {target.get('year') or ''}".strip()
        payload = {"search_for": query, "page_size": 5}
        r = self.session.post(src.url, json=payload, timeout=self.timeout)
        status.http_status = r.status_code
        status.reachable = r.ok
        p = out_dir / "figshare_search.json"
        p.write_bytes(r.content)
        status.saved_html = str(p)
        if r.ok:
            try:
                items = r.json()
                status.link_count = len(items) if isinstance(items, list) else 0
            except Exception:
                status.link_count = 0
            status.status = "reachable"
            status.message = f"Figshare API检索成功，item_count={status.link_count}；当前只记录元数据。"
        else:
            status.status = "error"
            status.message = f"HTTP {r.status_code}"
        return status

    def _configure_earthdata_auth(self) -> None:
        """准备 Earthdata 认证信息，但不污染 CMR 元数据检索 session。

        CMR collection/granule 检索通常可以公开访问。之前版本把 Bearer Token
        直接加入 self.session，导致 collection 接口在某些 token/EULA 状态下返回 401。
        新逻辑只保存凭据，真实文件下载时再临时附加 Authorization 或 Basic Auth。
        """
        token = (
            os.getenv("EARTHDATA_TOKEN")
            or os.getenv("NASA_EARTHDATA_TOKEN")
            or os.getenv("NASA_TOKEN")
            or os.getenv("EARTHDATA_BEARER_TOKEN")
            or ""
        ).strip()
        if token.lower().startswith("bearer "):
            token = token.split(None, 1)[1].strip()
        # 防止用户从网页复制时带换行、空格。
        token = "".join(token.split())
        self.earthdata_token = token or None
        self.earthdata_user = os.getenv("EARTHDATA_USERNAME") or os.getenv("NASA_EARTHDATA_USERNAME")
        self.earthdata_password = os.getenv("EARTHDATA_PASSWORD") or os.getenv("NASA_EARTHDATA_PASSWORD")
        _emit("AUTH", "Earthdata 认证配置已读取", {
            "has_token": bool(self.earthdata_token),
            "token_length": len(self.earthdata_token or ""),
            "has_username_password": bool(self.earthdata_user and self.earthdata_password),
            "note": "CMR元数据检索不携带认证头；真实文件下载时再使用认证。"
        }, task_id=self.task_id)

    def _earthdata_download_session(self) -> requests.Session:
        """为 Earthdata 数据文件下载构造独立 session。"""
        sess = requests.Session()
        sess.headers.update(DEFAULT_HEADERS)
        if self.earthdata_token:
            sess.headers.update({"Authorization": f"Bearer {self.earthdata_token}"})
        elif self.earthdata_user and self.earthdata_password:
            sess.auth = (self.earthdata_user, self.earthdata_password)
        return sess

    def _probe_earthdata_cmr(self, src: PlatformSource, out_dir: Path, request_text: str, target: dict[str, Any], status: PlatformStatus, metadata_only: bool = False) -> PlatformStatus:
        """Earthdata 第一版适配：collection 检索 -> granule 检索 -> 可选下载。

        metadata_only=True 时只验证 CMR 检索，不下载 HDF/NC/TIF，避免国外数据源阻塞国内优先主线。
        """
        # 1) collections：保留原来的元数据探测，证明 CMR API 可访问。
        collection_url = self._earthdata_collection_url(request_text, target)
        status.url = collection_url
        _emit("EARTHDATA", "开始 CMR collection 公开检索", {"url": collection_url, "auth_header_sent": False}, task_id=self.task_id)
        # CMR 元数据检索必须使用 public session，不携带 EDL token/basic auth。
        cmr_session = requests.Session()
        cmr_session.headers.update(DEFAULT_HEADERS)
        r = cmr_session.get(collection_url, timeout=self.timeout)
        status.http_status = r.status_code
        status.reachable = r.ok
        collection_path = out_dir / "earthdata_cmr_collections.json"
        collection_path.write_bytes(r.content)
        status.saved_html = str(collection_path)
        if not r.ok:
            status.status = "error"
            status.message = f"CMR collection HTTP {r.status_code}；已使用不带认证头的公开检索请求。"
            status.warnings.append("若仍为401，请检查是否被代理/网络环境改写请求，或先用浏览器访问 CMR API 验证网络。")
            return status
        try:
            collection_items = (r.json().get("feed") or {}).get("entry") or []
        except Exception:
            collection_items = []
        status.link_count = len(collection_items)

        # 2) granules：用 short_name + temporal + bounding_box 找目标区域/年份的真实文件。
        granule_url = self._earthdata_granule_url(target)
        status.warnings.append(f"Earthdata granule_query={granule_url}")
        _emit("EARTHDATA", "开始 CMR granule 公开检索", {"url": granule_url, "auth_header_sent": False}, task_id=self.task_id)
        gr = cmr_session.get(granule_url, timeout=self.timeout)
        granule_path = out_dir / "earthdata_cmr_granules.json"
        granule_path.write_bytes(gr.content)
        if not gr.ok:
            status.status = "reachable"
            status.message = f"CMR collection成功，但granule检索失败 HTTP {gr.status_code}。"
            return status
        try:
            granules = (gr.json().get("feed") or {}).get("entry") or []
        except Exception:
            granules = []
        data_links = self._earthdata_data_links(granules)
        status.download_link_count = len(data_links)
        _emit("EARTHDATA", "CMR granule 检索结果", {"collection_count": len(collection_items), "granule_count": len(granules), "data_link_count": status.download_link_count, "first_data_link": data_links[0] if data_links else None}, task_id=self.task_id)
        status.message = f"CMR检索成功：collection_count={len(collection_items)}，granule_count={len(granules)}，data_link_count={status.download_link_count}。"
        status.status = "reachable"

        if metadata_only:
            status.warnings.append("国外平台当前按 metadata_only 模式处理：已完成 CMR 元数据/链接检索，但跳过真实文件下载。")
            _emit("EARTHDATA", "国外平台 metadata_only，跳过真实数据下载", {"platform": src.id, "data_link_count": status.download_link_count}, task_id=self.task_id)
            return status
        if not self.allow_download:
            return status
        if os.getenv("PRO_EARTHDATA_GRANULE_DOWNLOAD", "1") != "1":
            status.warnings.append("PRO_EARTHDATA_GRANULE_DOWNLOAD=0，已跳过真实文件下载。")
            return status

        # 3) 下载：只下载少量小文件，避免第一次任务被大文件拖垮。
        max_links = int(os.getenv("PRO_EARTHDATA_MAX_GRANULE_DOWNLOADS", "1"))
        downloaded = 0
        for href in data_links:
            if downloaded >= max_links:
                break
            # 优先真正的栅格/科学数据容器。
            if Path(urlparse(href).path.lower()).suffix not in {".hdf", ".h5", ".hdf5", ".nc", ".tif", ".tiff"}:
                continue
            dl = self._download_file(href, out_dir, f"earthdata_granule_{downloaded+1}", session=self._earthdata_download_session())
            if dl:
                status.downloaded_files.append(str(dl))
                status.downloaded_count += 1
                downloaded += 1
        if status.downloaded_count:
            status.status = "file_downloaded"
        elif status.download_link_count:
            status.warnings.append("发现Earthdata数据链接，但未能下载成功。常见原因：未配置EARTHDATA_USERNAME/PASSWORD、token失效、下载链接需要URS跳转授权，或文件超过大小限制。")
        return status

    def _earthdata_collection_url(self, request_text: str, target: dict[str, Any]) -> str:
        keyword = os.getenv("PRO_EARTHDATA_CMR_KEYWORD", "MODIS")
        return "https://cmr.earthdata.nasa.gov/search/collections.json?" + f"keyword={quote(keyword)}&page_size=10"

    def _earthdata_granule_url(self, target: dict[str, Any]) -> str:
        short_name = os.getenv("PRO_EARTHDATA_SHORT_NAME", "MOD13Q1")
        version = os.getenv("PRO_EARTHDATA_VERSION", "061")
        year = int(target.get("year") or time.strftime("%Y"))
        # 默认只取年内一个短窗口，降低下载和检索负担；后续正式版再做全年合成。
        start = os.getenv("PRO_EARTHDATA_TEMPORAL_START", f"{year}-06-01T00:00:00Z")
        end = os.getenv("PRO_EARTHDATA_TEMPORAL_END", f"{year}-08-31T23:59:59Z")
        bbox = target.get("bbox_epsg4326") or None
        # target 没带 bbox 时，从 run() 写入的 sample_bbox 不能直接传入这里；所以从环境变量兜底成都附近。
        bbox_env = os.getenv("PRO_EARTHDATA_BBOX", "103.0,30.0,105.0,31.5")
        bbox_str = ",".join(map(str, bbox)) if bbox else bbox_env
        params = {
            "short_name": short_name,
            "version": version,
            "temporal": f"{start},{end}",
            "bounding_box": bbox_str,
            "downloadable": "true",
            "page_size": os.getenv("PRO_EARTHDATA_GRANULE_PAGE_SIZE", "10"),
            "sort_key": "start_date",
        }
        return "https://cmr.earthdata.nasa.gov/search/granules.json?" + "&".join(f"{k}={quote(str(v))}" for k, v in params.items() if v)

    def _earthdata_data_links(self, granules: list[dict[str, Any]]) -> list[str]:
        out: list[str] = []
        seen = set()
        for g in granules or []:
            for link in g.get("links") or []:
                href = link.get("href") or ""
                if not href or href in seen:
                    continue
                rel = str(link.get("rel") or "").lower()
                title = str(link.get("title") or "").lower()
                inherited = bool(link.get("inherited"))
                if inherited:
                    continue
                suffix = Path(urlparse(href).path.lower()).suffix
                looks_data = suffix in {".hdf", ".h5", ".hdf5", ".nc", ".tif", ".tiff"}
                is_data_rel = "data#" in rel or "download" in title or "data" in title
                is_browse = "browse" in rel or "browse" in title or suffix in {".jpg", ".jpeg", ".png"}
                if (looks_data or is_data_rel) and not is_browse:
                    out.append(href)
                    seen.add(href)
        return out

    def _extract_years_from_filename(self, path: Path) -> list[int]:
        nums = re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", path.stem)
        years = []
        for x in nums:
            try:
                y = int(x)
                if 1980 <= y <= 2100 and y not in years:
                    years.append(y)
            except Exception:
                pass
        return years

    def _is_dynamic_local_covariate_file(self, path: Path) -> bool:
        name = path.stem.lower()
        dynamic_kw = [
            "lulc", "clcd", "landcover", "land_cover", "worldcover", "crop", "cropland", "chinacp",
            "management", "seasonal", "climate", "water", "veg", "phenology", "ndvi", "evi", "npp",
            "fvc", "irrig", "irrigation", "planting", "pattern", "cropping", "利用", "作物", "复种", "灌溉",
        ]
        static_kw = ["dem", "slope", "aspect", "tpi", "tri", "roughness", "relief", "curvature", "bd", "cec", "clay", "sand", "silt", "ph", "gravel", "porosity"]
        if any(k in name for k in dynamic_kw):
            return True
        if any(k in name for k in static_kw):
            return False
        return False

    def _local_file_temporal_check(self, path: Path, target: dict[str, Any], src_id: str) -> tuple[bool, dict[str, Any]]:
        """Audit whether a local fallback file is temporally compatible with the requested year.

        Static/background rasters (DEM, soil texture, pH, BD, CEC etc.) may not encode a year in
        the filename. Dynamic rasters must either contain the requested year or a year range that
        includes it, unless PRO_LOCAL_DYNAMIC_UNDATED_ALLOW=1.
        """
        try:
            target_year = int(target.get("year"))
        except Exception:
            return True, {"policy": "no_target_year", "accepted": True, "path": str(path)}
        years = self._extract_years_from_filename(path)
        is_dynamic = self._is_dynamic_local_covariate_file(path)
        policy = (os.getenv("PRO_LOCAL_COVARIATE_YEAR_POLICY", "target_or_static") or "target_or_static").strip().lower()
        info = {"path": str(path), "target_year": target_year, "filename_years": years, "dynamic": is_dynamic, "policy": policy, "source": src_id}
        if policy in {"off", "none", "ignore"}:
            info.update({"accepted": True, "reason": "temporal_policy_off"})
            return True, info
        # Land-cover rasters used as output masks are allowed even if their filename
        # year differs from the target year.  They are not SOM predictors in
        # mask-only mode; rejecting them would silently disable the requested
        # cropland/forest/grassland/etc. output mask.
        if bool((target or {}).get("mask_by_landcover") or (target or {}).get("mask_non_target_landcover")) and self._is_landcover_mask_path(path):
            info.update({"accepted": True, "reason": "landcover_mask_for_requested_output_scope_year_exempt"})
            return True, info
        if years:
            y0, y1 = min(years), max(years)
            ok = (y0 <= target_year <= y1)
            info.update({"accepted": bool(ok), "reason": "target_year_within_filename_year_or_range" if ok else "target_year_not_in_filename_year_or_range"})
            return bool(ok), info
        # V182: user-provided local covariates should not be discarded merely because
        # a dynamic-looking filename has no year. In the current workflow the user often
        # imports a curated folder (for example Management_Proxy_Vars_250m.tif or
        # VegPhenology_MODIS250m.tif). Dropping these layers makes the model look
        # artificially poor. Default to accepting undated dynamic local layers, but keep
        # the audit reason explicit so the AI can warn the user if needed.
        if is_dynamic and os.getenv("PRO_LOCAL_DYNAMIC_UNDATED_ALLOW", "1") != "1":
            info.update({"accepted": False, "reason": "dynamic_local_file_without_year"})
            return False, info
        info.update({"accepted": True, "reason": "undated_dynamic_allowed_by_user_local_folder" if is_dynamic else "undated_static_or_allowed_background"})
        return True, info

    def _multiband_stack_enabled(self) -> bool:
        """V175: allow one multi-band raster stack to provide many covariates."""
        return (os.getenv("PRO_MULTIBAND_STACK_ENABLE", "1") or "1").strip().lower() not in {"0", "false", "no", "off"}

    def _is_multiband_stack_raster(self, path: Path) -> bool:
        """Return True for ordinary multi-band rasters that can be split into covariates."""
        if rasterio is None:
            return False
        if path.suffix.lower() not in {".tif", ".tiff"}:
            # NetCDF/HDF often expose subdatasets, not a simple DSM covariate stack.
            return False
        try:
            with rasterio.open(path) as src:
                if getattr(src, "subdatasets", None):
                    return False
                return int(getattr(src, "count", 1) or 1) > 1
        except Exception:
            return False

    def _read_multiband_mapping_file(self, mapping_path: Path) -> dict[int, str]:
        """Read a band->feature_name mapping from JSON or CSV/Excel.

        Supported CSV columns: band plus feature_name/name/feature/variable/covariate.
        Supported JSON forms:
          {"1": "DEM", "2": "slope"}
          {"bands": [{"band": 1, "feature_name": "DEM"}, ...]}
        """
        mapping: dict[int, str] = {}
        try:
            suffix = mapping_path.suffix.lower()
            if suffix == ".json":
                obj = json.loads(mapping_path.read_text(encoding="utf-8-sig"))
                if isinstance(obj, dict) and isinstance(obj.get("bands"), list):
                    rows = obj.get("bands") or []
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        b = row.get("band") or row.get("band_index") or row.get("index") or row.get("id")
                        n = row.get("feature_name") or row.get("name") or row.get("feature") or row.get("variable") or row.get("covariate")
                        try:
                            bi = int(b)
                            if bi >= 1 and str(n or "").strip():
                                mapping[bi] = str(n).strip()
                        except Exception:
                            pass
                elif isinstance(obj, dict):
                    for k, v in obj.items():
                        try:
                            bi = int(str(k).strip())
                            if bi >= 1 and str(v or "").strip():
                                mapping[bi] = str(v).strip()
                        except Exception:
                            pass
            elif suffix in TABLE_EXTS and pd is not None:
                if suffix == ".csv":
                    # Band mapping files are small schema tables, not sample tables;
                    # read them directly first so BOM in "band" is handled correctly.
                    try:
                        df = pd.read_csv(mapping_path, encoding="utf-8-sig")
                    except Exception:
                        try:
                            df = pd.read_csv(mapping_path, encoding="utf-8")
                        except Exception:
                            try:
                                from services.sample_reader import _read_csv_with_fallback
                                df, _ = _read_csv_with_fallback(mapping_path, nrows=None)
                            except TypeError:
                                df, _ = _read_csv_with_fallback(mapping_path)
                else:
                    df = pd.read_excel(mapping_path)
                cols_l = {str(c).strip().lower(): c for c in df.columns}
                band_col = None
                for cand in ["band", "band_index", "index", "波段", "波段号", "band_id"]:
                    if cand.lower() in cols_l:
                        band_col = cols_l[cand.lower()]; break
                name_col = None
                for cand in ["feature_name", "name", "feature", "variable", "covariate", "字段名", "变量名", "协变量", "band_name"]:
                    if cand.lower() in cols_l:
                        name_col = cols_l[cand.lower()]; break
                # Allow a simple one-column list of feature names. Band number = row index + 1.
                if band_col is None and name_col is None and len(df.columns) == 1:
                    name_col = df.columns[0]
                    for i, n in enumerate(df[name_col].tolist(), start=1):
                        if str(n or "").strip():
                            mapping[int(i)] = str(n).strip()
                elif band_col is not None and name_col is not None:
                    for _, row in df.iterrows():
                        try:
                            bi = int(row[band_col])
                            nm = str(row[name_col]).strip()
                            if bi >= 1 and nm and nm.lower() != "nan":
                                mapping[bi] = nm
                        except Exception:
                            pass
        except Exception as exc:
            self.warnings.append(f"V175读取多波段band映射失败：{mapping_path} | {exc}")
        return mapping

    def _find_multiband_mapping_file(self, raster_path: Path, files: list[Path]) -> Path | None:
        """Find the companion band mapping table for a multi-band stack."""
        stem = raster_path.stem.lower()
        parent = raster_path.parent.resolve()
        strong_names = {
            f"{stem}_bands", f"{stem}_band", f"{stem}_band_mapping", f"{stem}_bands_mapping",
            f"{stem}.bands", f"{stem}.band_mapping", "band_mapping", "bands", "波段映射", "band_names"
        }
        candidates = []
        for f in files:
            try:
                if f.suffix.lower() not in TABLE_EXTS | {".json"}:
                    continue
                score = 0
                fs = f.stem.lower()
                if f.parent.resolve() == parent:
                    score += 10
                if fs in strong_names:
                    score += 100
                if fs.startswith(stem) and ("band" in fs or "波段" in fs):
                    score += 80
                if "band" in fs or "波段" in fs:
                    score += 20
                if score > 0:
                    candidates.append((score, f))
            except Exception:
                pass
        candidates.sort(key=lambda x: x[0], reverse=True)
        for _, f in candidates:
            m = self._read_multiband_mapping_file(f)
            if m:
                return f
        return None

    def _multiband_feature_names(self, raster_path: Path, files: list[Path]) -> tuple[dict[int, str], dict[str, Any]]:
        """Resolve feature names for every band in a stack."""
        audit: dict[str, Any] = {"raster": str(raster_path), "source": None, "mapping_file": None, "warnings": []}
        names: dict[int, str] = {}
        count = 0
        descriptions: tuple[Any, ...] = tuple()
        try:
            with rasterio.open(raster_path) as src:
                count = int(src.count)
                descriptions = tuple(getattr(src, "descriptions", tuple()) or tuple())
        except Exception as exc:
            audit["warnings"].append(f"无法读取多波段栅格元数据：{exc}")
            return names, audit

        mapping_file = self._find_multiband_mapping_file(raster_path, files)
        if mapping_file:
            names.update(self._read_multiband_mapping_file(mapping_file))
            audit["source"] = "companion_mapping_file"
            audit["mapping_file"] = str(mapping_file)

        env_names = [x.strip() for x in str(os.getenv("PRO_MULTIBAND_BAND_NAMES", "") or "").split(",") if x.strip()]
        if not names and env_names:
            for i, nm in enumerate(env_names, start=1):
                names[int(i)] = nm
            audit["source"] = "PRO_MULTIBAND_BAND_NAMES"

        if not names:
            for i in range(1, count + 1):
                desc = descriptions[i - 1] if i - 1 < len(descriptions) else None
                if desc and str(desc).strip():
                    names[int(i)] = str(desc).strip()
            if names:
                audit["source"] = "geotiff_band_description"

        if not names:
            # Fallback is allowed for engineering completeness, but the report will
            # warn because field matching against fused CSV names is impossible.
            for i in range(1, count + 1):
                names[int(i)] = f"{raster_path.stem}_band_{i:02d}"
            audit["source"] = "fallback_band_index_names"
            audit["warnings"].append("未找到band映射表/description，已使用band_序号命名；建议上传band_mapping.csv。")

        # Clamp to actual band count and de-duplicate names.
        clean: dict[int, str] = {}
        used: set[str] = set()
        for i in range(1, count + 1):
            nm = str(names.get(i) or f"{raster_path.stem}_band_{i:02d}").strip()
            nm = re.sub(r"\s+", "_", nm)
            if not nm:
                nm = f"{raster_path.stem}_band_{i:02d}"
            base = nm
            k = 2
            while nm in used:
                nm = f"{base}_{k}"
                k += 1
            used.add(nm)
            clean[int(i)] = nm
        audit["band_count"] = int(count)
        audit["band_names"] = {str(k): v for k, v in clean.items()}
        return clean, audit

    def _standardize_raster_band(self, src_path: Path, out_dir: Path, target: dict[str, Any], sample_bbox: dict[str, Any], band_index: int = 1, band_name: str | None = None) -> Path | None:
        """Standardize one band from a raster stack to a single-band GeoTIFF."""
        if rasterio is None or WarpedVRT is None or from_bounds is None or window_transform is None:
            raise RuntimeError("缺少 rasterio，不能标准化栅格。")
        bbox = sample_bbox.get("bbox_target_crs")
        if not bbox:
            raise RuntimeError("没有目标CRS下的样点裁剪范围。")
        target_crs = target["target_crs"]
        res = float(target["resolution_m"])
        band_label = _safe_name(str(band_name or f"band_{band_index:02d}"), 48)
        out_path = out_dir / f"{_safe_name(src_path.stem)}_B{int(band_index):02d}_{band_label}_{int(res)}m_std.tif"

        try:
            with rasterio.open(src_path) as probe:
                subdataset = self._select_subdataset(probe, src_path)
        except Exception as exc:
            raise RuntimeError(f"源栅格无法打开，可能GDAL缺少对应驱动或文件需要认证下载：{exc}")
        open_target = subdataset or str(src_path)
        var_name_for_policy = band_name or src_path.name
        with rasterio.open(open_target) as src:
            if not src.crs:
                raise RuntimeError("源栅格缺少 CRS。")
            if int(band_index) < 1 or int(band_index) > int(src.count):
                raise RuntimeError(f"band_index越界：{band_index}，实际band_count={src.count}")
            nodata = -9999.0
            src_nodata = src.nodata
            zero_nodata_honored = self._zero_should_be_nodata(var_name_for_policy, src_nodata)
            resampling = Resampling.nearest if self._is_categorical_or_mask_raster(var_name_for_policy) else Resampling.bilinear
            vrt_opts = {"crs": target_crs, "resolution": res, "resampling": resampling, "nodata": nodata}
            if src_nodata is not None:
                try:
                    nd_float = float(src_nodata)
                    if math.isclose(nd_float, 0.0, rel_tol=0, abs_tol=1e-12) and not zero_nodata_honored:
                        vrt_opts["src_nodata"] = None
                    else:
                        vrt_opts["src_nodata"] = src_nodata
                except Exception:
                    vrt_opts["src_nodata"] = src_nodata
            with WarpedVRT(src, **vrt_opts) as vrt:
                win = from_bounds(*bbox, transform=vrt.transform)
                win = win.round_offsets().round_lengths()
                if win.width <= 1 or win.height <= 1:
                    raise RuntimeError("裁剪窗口为空。")
                data = vrt.read(int(band_index), window=win, masked=False)
                if data.size == 0:
                    raise RuntimeError("读取窗口为空。")
                arr = data.astype("float32")
                invalid = ~np.isfinite(arr) if np is not None else None
                try:
                    invalid = invalid | (np.abs(arr) >= self._extreme_abs_limit())
                except Exception:
                    pass
                try:
                    if src_nodata is not None and not (math.isclose(float(src_nodata), 0.0, rel_tol=0, abs_tol=1e-12) and not zero_nodata_honored):
                        invalid = invalid | np.isclose(arr, float(src_nodata), rtol=0, atol=1e-9)
                except Exception:
                    pass
                if invalid is not None:
                    arr[invalid] = nodata
                scale = float(os.getenv("PRO_RASTER_VALUE_SCALE", "1"))
                offset = float(os.getenv("PRO_RASTER_VALUE_OFFSET", "0"))
                if scale != 1.0 or offset != 0.0:
                    valid = arr != nodata
                    arr[valid] = arr[valid] * scale + offset
                meta = vrt.meta.copy()
                meta.update({"driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1], "count": 1, "dtype": "float32", "crs": target_crs, "transform": window_transform(win, vrt.transform), "nodata": nodata, "compress": "lzw"})
                with rasterio.open(out_path, "w", **meta) as dst:
                    dst.write(arr, 1)
                    try:
                        dst.set_band_description(1, str(band_name or f"band_{band_index}"))
                    except Exception:
                        pass
        _emit("PREPROCESS", "V175多波段栅格单band标准化完成", {"src": str(src_path), "opened": open_target, "band": int(band_index), "feature_name": str(band_name or ""), "out": str(out_path), "target_crs": target_crs, "resolution_m": res, "resampling": ("nearest" if self._is_categorical_or_mask_raster(var_name_for_policy) else "bilinear"), "source_nodata": str(src_nodata) if src_nodata is not None else None, "zero_nodata_honored": bool(zero_nodata_honored), "zero_nodata_policy": os.getenv("PRO_ZERO_NODATA_POLICY", "never")}, task_id=self.task_id)
        return out_path

    def _process_multiband_stack_file(self, src: PlatformSource, raster_path: Path, files: list[Path], samples: Any, model_df: Any, target: dict[str, Any], sample_bbox: dict[str, Any], out_dir: Path) -> tuple[list[str], list[Path], dict[str, Any]]:
        """Split/process a multi-band covariate stack and extract each band to samples."""
        added_cols: list[str] = []
        standardized: list[Path] = []
        band_names, audit = self._multiband_feature_names(raster_path, files)
        audit.update({"platform_id": src.id, "stack_path": str(raster_path), "added_columns": [], "mask_only_columns": [], "warnings": audit.get("warnings") or []})
        if not band_names:
            audit["warnings"].append("未能识别多波段栅格band名称，已跳过。")
            return added_cols, standardized, audit
        try:
            self._v175_multiband_stack_audit = getattr(self, "_v175_multiband_stack_audit", []) or []
        except Exception:
            self._v175_multiband_stack_audit = []

        for band_index, feature_name in band_names.items():
            try:
                # CLCD may be present in the stack. It is retained as a mask raster,
                # but not returned as a formal SOM model feature when mask-only is enabled.
                std = self._standardize_raster_band(raster_path, out_dir, target, sample_bbox, int(band_index), str(feature_name))
                if not std:
                    continue
                standardized.append(std)
                col = f"cov_{src.id}_{_safe_name(str(feature_name), 56)}"
                values = self._sample_raster(std, samples, col)
                valid_count = sum(v is not None for v in values)
                if valid_count <= 0:
                    audit["warnings"].append(f"band {band_index} {feature_name} 标准化成功但样点抽取全为空")
                    continue
                model_df[col] = values
                self._covariate_raster_map[col] = str(std)
                # Also allow direct feature-name lookup in later fused-CSV/grid deployment plans.
                self._covariate_raster_map[str(feature_name)] = str(std)
                if self._is_clcd_model_feature(col) or self._is_clcd_model_feature(feature_name):
                    target["clcd_mask_raster"] = str(std)
                    target["v165_clcd_mask_raster"] = str(std)
                    if self._clcd_mask_only_policy():
                        audit["mask_only_columns"].append(col)
                        continue
                added_cols.append(col)
                audit["added_columns"].append({"band": int(band_index), "feature_name": str(feature_name), "column": col, "std_raster": str(std), "valid_sample_count": int(valid_count)})
            except Exception as exc:
                audit["warnings"].append(f"band {band_index} {feature_name} 处理失败：{exc}")
        try:
            self._v175_multiband_stack_audit.append(audit)
            target["v175_multiband_stack_audit"] = self._v175_multiband_stack_audit
        except Exception:
            pass
        return added_cols, standardized, audit

    def _process_platform_files(self, src: PlatformSource, status: PlatformStatus, samples: Any, model_df: Any, target: dict[str, Any], sample_bbox: dict[str, Any], out_dir: Path) -> list[str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        roots = []
        if status.saved_html:
            roots.append(Path(status.saved_html).parent)
        for f in status.downloaded_files:
            roots.append(Path(f))
        # V166: explicit rasters uploaded/imported in the session may live outside
        # the CSV folder and outside PRO_DOMESTIC_FALLBACK_LOCAL_DIR. Add them directly.
        for _p in getattr(self, "_uploaded_covariate_files", []) or []:
            try:
                roots.append(Path(_p))
            except Exception:
                pass
        files = self._scan_files(roots)
        added_cols: list[str] = []
        for f in files:
            if src.id == "domestic_local" and not self._local_covariate_file_allowed(f):
                status.warnings.append(f"未被当前环境协变量选择方案选中，已跳过：{f.name}")
                continue
            ok_temporal, temporal_info = self._local_file_temporal_check(f, target, src.id)
            if not ok_temporal:
                status.warnings.append(f"按目标年份过滤本地动态协变量：{f.name}；原因={temporal_info.get('reason')}；目标年={temporal_info.get('target_year')}；文件年份={temporal_info.get('filename_years')}")
                _emit("SOURCE", "本地协变量年份不匹配，已跳过", temporal_info, task_id=self.task_id)
                continue
            elif src.id == "domestic_local":
                _emit("SOURCE", "本地协变量年份审计通过", temporal_info, task_id=self.task_id)
            suffix = f.suffix.lower()
            if suffix in ARCHIVE_EXTS:
                for child in self._unpack_archive(f, out_dir / "unpacked" / f.stem):
                    files.append(child)
                continue
            if suffix in RASTER_EXTS:
                try:
                    # V175: a single GeoTIFF may be a multi-band covariate stack.
                    # Each band is treated as one environmental covariate when a
                    # band mapping file, band descriptions, or PRO_MULTIBAND_BAND_NAMES
                    # can provide names. This is the preferred formal DSM input:
                    # sample CSV + multi-band stack + CLCD mask.
                    handled_multiband = False
                    if self._multiband_stack_enabled() and self._is_multiband_stack_raster(f):
                        mb_cols, mb_rasters, mb_audit = self._process_multiband_stack_file(
                            src, f, files, samples, model_df, target, sample_bbox, out_dir
                        )
                        if mb_cols or mb_rasters:
                            handled_multiband = True
                            added_cols.extend(mb_cols)
                            status.standardized_rasters.extend([str(x) for x in mb_rasters])
                            status.processed_raster_count = len(status.standardized_rasters)
                            status.warnings.extend(mb_audit.get("warnings") or [])
                            _emit("COVARIATE", "V175多波段协变量栈已接入", mb_audit, task_id=self.task_id)
                    if handled_multiband:
                        continue

                    std = self._standardize_raster(f, out_dir, target, sample_bbox)
                    if std:
                        status.standardized_rasters.append(str(std))
                        status.processed_raster_count += 1
                        col = f"cov_{src.id}_{_safe_name(f.stem, 24)}"
                        values = self._sample_raster(std, samples, col)
                        valid_count = sum(v is not None for v in values)
                        if valid_count > 0:
                            model_df[col] = values
                            self._covariate_raster_map[col] = str(std)
                            added_cols.append(col)
                        else:
                            status.warnings.append(f"栅格已标准化但样点抽取全为空：{std.name}")
                        status.processed_raster_count = len(status.standardized_rasters)
                except Exception as exc:
                    status.warnings.append(f"栅格处理失败 {f.name}: {exc}")
            elif suffix in TABLE_EXTS:
                try:
                    rows, cols = self._inspect_table(f)
                    status.processed_table_count += 1
                    status.warnings.append(f"表格已识别但未空间化：{f.name}, rows={rows}, cols={cols}")
                except Exception as exc:
                    status.warnings.append(f"表格读取失败 {f.name}: {exc}")
        return added_cols

    def _scan_files(self, roots: list[Path]) -> list[Path]:
        out: list[Path] = []
        seen = set()
        for r in roots:
            if not r.exists():
                continue
            candidates = [r] if r.is_file() else list(r.rglob("*"))
            for f in candidates:
                if not f.is_file():
                    continue
                if f.suffix.lower() in RASTER_EXTS | TABLE_EXTS | ARCHIVE_EXTS:
                    key = str(f.resolve())
                    if key not in seen:
                        out.append(f)
                        seen.add(key)
        return out

    def _unpack_zip(self, z: Path, out_dir: Path) -> list[Path]:
        return self._unpack_archive(z, out_dir)

    def _unpack_archive(self, archive_path: Path, out_dir: Path) -> list[Path]:
        """解压国内平台常见归档文件。zip 直接支持；7z/rar 优先尝试 py7zr/系统 7z。"""
        out_dir.mkdir(parents=True, exist_ok=True)
        children: list[Path] = []
        suffix = archive_path.suffix.lower()
        try:
            if suffix == ".zip":
                with zipfile.ZipFile(archive_path, "r") as zz:
                    zz.extractall(out_dir)
            elif suffix == ".7z":
                try:
                    import py7zr  # type: ignore
                    with py7zr.SevenZipFile(archive_path, mode="r") as zz:
                        zz.extractall(path=out_dir)
                except Exception:
                    # 回退系统 7z。Windows 用户如安装 7-Zip 并加入 PATH，也可直接使用。
                    import subprocess
                    subprocess.run(["7z", "x", str(archive_path), f"-o{out_dir}", "-y"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            elif suffix == ".rar":
                try:
                    import rarfile  # type: ignore
                    with rarfile.RarFile(archive_path) as rr:
                        rr.extractall(out_dir)
                except Exception:
                    import subprocess
                    subprocess.run(["7z", "x", str(archive_path), f"-o{out_dir}", "-y"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                return []
            children = [p for p in out_dir.rglob("*") if p.is_file()]
        except Exception as exc:
            self.warnings.append(f"归档解压失败：{archive_path} | {exc}")
        return children

    def _inspect_table(self, f: Path) -> tuple[int, int]:
        if pd is None:
            raise RuntimeError("缺少 pandas。")
        if f.suffix.lower() == ".csv":
            # 只检查，不强求字段；使用统一多编码读取，避免中文字段乱码。
            try:
                from services.sample_reader import _read_csv_with_fallback
                df, _ = _read_csv_with_fallback(f, nrows=200)
            except Exception as exc:
                raise RuntimeError(f"CSV多编码读取失败：{exc}")
        else:
            df = pd.read_excel(f, nrows=200)
        return int(len(df)), int(len(df.columns))

    def _select_subdataset(self, src: Any, src_path: Path) -> str | None:
        subdatasets = list(getattr(src, "subdatasets", []) or [])
        if not subdatasets:
            return None
        keywords = [k.strip().lower() for k in os.getenv(
            "PRO_RASTER_SUBDATASET_KEYWORDS",
            "NDVI,EVI,LST,precip,temperature,soil_moisture,GPP,NPP,reflectance"
        ).split(",") if k.strip()]
        # MODIS NDVI 常见子数据集名称含 NDVI；优先避免 QA、quality、metadata。
        bad = ["qa", "quality", "qc", "metadata", "view", "angle"]
        scored: list[tuple[int, str]] = []
        for i, sd in enumerate(subdatasets):
            lo = sd.lower()
            score = 0
            for kw in keywords:
                if kw in lo:
                    score += 100
            if any(b in lo for b in bad):
                score -= 50
            score -= i
            scored.append((score, sd))
        scored.sort(reverse=True, key=lambda x: x[0])
        chosen = scored[0][1] if scored else subdatasets[0]
        _emit("PREPROCESS", "检测到栅格容器子数据集，已选择一个子数据集", {
            "src": str(src_path),
            "subdataset_count": len(subdatasets),
            "chosen": chosen,
            "keywords": keywords,
        }, task_id=self.task_id)
        return chosen

    def _covariate_name_lower(self, value: Any) -> str:
        return str(value or "").lower()

    def _is_categorical_or_mask_raster(self, name_or_path: Any) -> bool:
        """Heuristic for categorical/mask rasters where nearest-neighbor is required.

        This is intentionally conservative. Continuous variables such as slope,
        curvature, TWI, NDVI-like indices, management proxies, climate and soil
        properties can contain a real 0, so they are not treated as categorical
        merely because they have integer values.
        """
        lo = self._covariate_name_lower(name_or_path)
        keys = [x.strip().lower() for x in os.getenv(
            "PRO_CATEGORICAL_RASTER_KEYWORDS",
            "lulc,clcd,landcover,land_cover,worldcover,cropland_mask,admin_mask,mask,classification,class"
        ).split(",") if x.strip()]
        return any(k in lo for k in keys)

    def _zero_should_be_nodata(self, name_or_path: Any, nodata_value: Any) -> bool:
        """Return whether value 0 should be considered NoData for this covariate.

        V172 rule: 0 is a valid environmental value by default. Many DSM
        covariates can legitimately be 0, for example slope, distance, index
        residuals, built-up area fraction, water/management proxies or binary
        indicators. Therefore metadata nodata=0 is ignored unless the operator
        explicitly opts into a stricter policy.

        Policy values:
        - never/valid/keep (default): never treat 0 as missing.
        - metadata/honor/always: honor raster metadata; if src.nodata==0 then
          0 is NoData. Use this only when the layer has been manually verified.
        - auto: legacy behavior; categorical/mask/background rasters may honor
          metadata nodata=0. Not recommended as the default.
        """
        try:
            nd = float(nodata_value) if nodata_value is not None else None
        except Exception:
            nd = None
        if nd is None or not math.isfinite(nd) or not math.isclose(nd, 0.0, rel_tol=0, abs_tol=1e-12):
            return False
        policy = (os.getenv("PRO_ZERO_NODATA_POLICY", "never") or "never").strip().lower()
        if policy in {"never", "valid", "keep", "0_valid", "zero_valid"}:
            return False
        if policy in {"metadata", "honor", "always"}:
            return True
        if policy == "auto":
            return self._is_categorical_or_mask_raster(name_or_path)
        return False

    def _extreme_abs_limit(self) -> float:
        """Absolute limit for physically impossible raster values.

        Defaults to 1e30 so only fill-value style artifacts such as +/-3.4e38
        are removed. Ordinary zeros and ordinary negative/positive values are
        retained.
        """
        try:
            v = float(os.getenv("PRO_RASTER_EXTREME_ABS_MAX", "1e30"))
            if math.isfinite(v) and v > 0:
                return v
        except Exception:
            pass
        return 1e30

    def _is_missing_raster_value(self, value: Any, nodata_value: Any, name_or_path: Any = None, masked: bool = False) -> tuple[bool, str | None]:
        """Variable-aware raster missing check.

        V172 rule: 0 is never missing by default. Missing is limited to raster
        mask, explicit NoData (except metadata nodata=0 under the default policy),
        NaN/non-finite values and extreme fill-value artifacts.
        """
        try:
            x = float(value)
        except Exception:
            return True, "non_numeric"
        if masked:
            # rasterio may mask a real zero when a source incorrectly declares
            # nodata=0. Preserve that zero unless the explicit zero policy says
            # otherwise. Other masked values remain missing.
            try:
                if math.isfinite(x) and math.isclose(x, 0.0, rel_tol=0, abs_tol=1e-12) and not self._zero_should_be_nodata(name_or_path, 0.0):
                    return False, None
            except Exception:
                pass
            return True, "masked"
        if not math.isfinite(x):
            return True, "nonfinite"
        if abs(x) >= self._extreme_abs_limit():
            return True, "extreme_fill_value"
        if nodata_value is not None:
            try:
                nd = float(nodata_value)
                if math.isfinite(nd) and math.isclose(x, nd, rel_tol=0, abs_tol=1e-9):
                    if math.isclose(nd, 0.0, rel_tol=0, abs_tol=1e-12) and not self._zero_should_be_nodata(name_or_path, nd):
                        return False, None
                    return True, "nodata"
            except Exception:
                pass
        return False, None

    def _standardize_raster(self, src_path: Path, out_dir: Path, target: dict[str, Any], sample_bbox: dict[str, Any]) -> Path | None:
        if rasterio is None or WarpedVRT is None or from_bounds is None or window_transform is None:
            raise RuntimeError("缺少 rasterio，不能标准化栅格。")
        bbox = sample_bbox.get("bbox_target_crs")
        if not bbox:
            raise RuntimeError("没有目标CRS下的样点裁剪范围。")
        target_crs = target["target_crs"]
        res = float(target["resolution_m"])
        out_path = out_dir / (_safe_name(src_path.stem) + f"_{int(res)}m_std.tif")

        # HDF/H5/NetCDF 常是多子数据集容器，需先选一个科学变量子数据集。
        try:
            with rasterio.open(src_path) as probe:
                subdataset = self._select_subdataset(probe, src_path)
        except Exception as exc:
            raise RuntimeError(f"源栅格无法打开，可能GDAL缺少对应驱动或文件需要认证下载：{exc}")
        open_target = subdataset or str(src_path)

        with rasterio.open(open_target) as src:
            if not src.crs:
                raise RuntimeError("源栅格缺少 CRS。")
            nodata = -9999.0
            src_nodata = src.nodata
            zero_nodata_honored = self._zero_should_be_nodata(src_path.name, src_nodata)
            # Do not let a metadata value of 0 erase meaningful zero-valued
            # continuous covariates. In auto mode, zero-nodata is honored only
            # for categorical/mask rasters; otherwise zeros remain valid values.
            vrt_opts = {
                "crs": target_crs,
                "resolution": res,
                "resampling": (Resampling.nearest if self._is_categorical_or_mask_raster(src_path.name) else Resampling.bilinear),
                "nodata": nodata,
            }
            if src_nodata is not None:
                try:
                    nd_float = float(src_nodata)
                    if math.isclose(nd_float, 0.0, rel_tol=0, abs_tol=1e-12) and not zero_nodata_honored:
                        # Override wrong/unsafe zero-nodata metadata for continuous layers.
                        vrt_opts["src_nodata"] = None
                    else:
                        vrt_opts["src_nodata"] = src_nodata
                except Exception:
                    vrt_opts["src_nodata"] = src_nodata
            with WarpedVRT(src, **vrt_opts) as vrt:
                win = from_bounds(*bbox, transform=vrt.transform)
                win = win.round_offsets().round_lengths()
                if win.width <= 1 or win.height <= 1:
                    raise RuntimeError("裁剪窗口为空。")
                data = vrt.read(1, window=win, masked=False)
                if data.size == 0:
                    raise RuntimeError("读取窗口为空。")
                arr = data.astype("float32")
                # Normalize only true missing pixels to the internal nodata value.
                # Generic 0 values are left untouched.
                invalid = ~np.isfinite(arr) if np is not None else None
                try:
                    extreme_limit = self._extreme_abs_limit()
                    invalid = invalid | (np.abs(arr) >= extreme_limit)
                except Exception:
                    pass
                try:
                    if src_nodata is not None and not (math.isclose(float(src_nodata), 0.0, rel_tol=0, abs_tol=1e-12) and not zero_nodata_honored):
                        invalid = invalid | np.isclose(arr, float(src_nodata), rtol=0, atol=1e-9)
                except Exception:
                    pass
                if invalid is not None:
                    arr[invalid] = nodata
                # 对 MODIS NDVI 等整型缩放产品提供可选 scale。默认不缩放，避免误处理其他变量。
                scale = float(os.getenv("PRO_RASTER_VALUE_SCALE", "1"))
                offset = float(os.getenv("PRO_RASTER_VALUE_OFFSET", "0"))
                if scale != 1.0 or offset != 0.0:
                    valid = arr != nodata
                    arr[valid] = arr[valid] * scale + offset
                meta = vrt.meta.copy()
                meta.update({
                    "driver": "GTiff",
                    "height": arr.shape[0],
                    "width": arr.shape[1],
                    "count": 1,
                    "dtype": "float32",
                    "crs": target_crs,
                    "transform": window_transform(win, vrt.transform),
                    "nodata": nodata,
                    "compress": "lzw",
                })
                with rasterio.open(out_path, "w", **meta) as dst:
                    dst.write(arr, 1)
        _emit("PREPROCESS", "栅格标准化完成", {"src": str(src_path), "opened": open_target, "out": str(out_path), "target_crs": target_crs, "resolution_m": res, "resampling": ("nearest" if self._is_categorical_or_mask_raster(src_path.name) else "bilinear"), "source_nodata": str(src_nodata) if src_nodata is not None else None, "zero_nodata_honored": bool(zero_nodata_honored), "zero_nodata_policy": os.getenv("PRO_ZERO_NODATA_POLICY", "never")}, task_id=self.task_id)
        return out_path

    def _sample_raster(self, raster_path: Path, samples: Any, col: str) -> list[float | None]:
        if rasterio is None or Transformer is None:
            raise RuntimeError("缺少 rasterio/pyproj，不能抽样栅格。")
        vals: list[float | None] = []
        audit = {
            "raster": str(raster_path),
            "column": col,
            "total": 0,
            "valid_count": 0,
            "missing_count": 0,
            "missing_reasons": {},
            "zero_count": 0,
            "zero_valid_count": 0,
            "zero_missing_count": 0,
            "zero_counted_as_missing": False,
            "nodata_value": None,
            "zero_nodata_policy": os.getenv("PRO_ZERO_NODATA_POLICY", "never"),
        }
        with rasterio.open(raster_path) as src:
            audit["nodata_value"] = float(src.nodata) if src.nodata is not None else None
            audit["zero_counted_as_missing"] = bool(self._zero_should_be_nodata(col or raster_path.name, src.nodata))
            tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            coords = [tr.transform(float(lon), float(lat)) for lon, lat in zip(samples["lon"], samples["lat"])]
            bounds = src.bounds
            for xy, v in zip(coords, src.sample(coords, masked=True)):
                audit["total"] += 1
                in_bounds = bool(bounds.left <= xy[0] <= bounds.right and bounds.bottom <= xy[1] <= bounds.top)
                masked = False
                raw0 = None
                if len(v):
                    raw0 = v[0]
                if hasattr(v, "mask") and len(v):
                    try:
                        m = getattr(v, "mask", False)
                        masked = bool(m[0]) if hasattr(m, "__len__") else bool(m)
                    except Exception:
                        masked = False
                # Avoid NumPy's "converting a masked element to nan" warning by
                # never calling float() directly on a masked scalar. If the
                # underlying value is 0 and zero-nodata is disabled, keep it as a
                # valid zero; otherwise the mask is treated as missing.
                try:
                    if masked and raw0 is not None and hasattr(raw0, "data"):
                        x = float(raw0.data)
                    elif raw0 is not None:
                        x = float(raw0)
                    else:
                        x = float("nan")
                except Exception:
                    x = float("nan")
                if masked and in_bounds:
                    try:
                        if math.isfinite(x) and math.isclose(x, 0.0, rel_tol=0, abs_tol=1e-12) and not self._zero_should_be_nodata(col or raster_path.name, 0.0):
                            masked = False
                    except Exception:
                        pass
                try:
                    if math.isfinite(x) and math.isclose(x, 0.0, rel_tol=0, abs_tol=1e-12):
                        audit["zero_count"] += 1
                except Exception:
                    pass
                missing, reason = self._is_missing_raster_value(x, src.nodata, col or raster_path.name, masked=masked or (not in_bounds))
                if not missing:
                    vals.append(x)
                    audit["valid_count"] += 1
                    try:
                        if math.isclose(float(x), 0.0, rel_tol=0, abs_tol=1e-12):
                            audit["zero_valid_count"] += 1
                    except Exception:
                        pass
                else:
                    vals.append(None)
                    audit["missing_count"] += 1
                    audit["missing_reasons"][reason or "missing"] = int(audit["missing_reasons"].get(reason or "missing", 0)) + 1
                    try:
                        if math.isfinite(x) and math.isclose(x, 0.0, rel_tol=0, abs_tol=1e-12):
                            audit["zero_missing_count"] += 1
                    except Exception:
                        pass
        if audit["total"]:
            audit["missing_rate"] = float(audit["missing_count"] / audit["total"])
            audit["zero_rate"] = float(audit["zero_count"] / audit["total"])
        self._covariate_value_audit[col] = audit
        _emit("ALIGN", "栅格值已抽取到样点", {"raster": str(raster_path), "column": col, "valid_count": audit["valid_count"], "missing_count": audit["missing_count"], "zero_count": audit["zero_count"], "zero_valid_count": audit["zero_valid_count"], "zero_missing_count": audit["zero_missing_count"], "zero_counted_as_missing": audit["zero_counted_as_missing"]}, task_id=self.task_id)
        return vals

    def _append_platform_status_features(self, model_df: Any, statuses: list[PlatformStatus]) -> None:
        for s in statuses:
            prefix = f"src_{s.platform_id}"
            model_df[f"{prefix}_reachable"] = 1 if s.reachable else 0
            model_df[f"{prefix}_downloaded_count"] = int(s.downloaded_count)
            model_df[f"{prefix}_raster_count"] = int(s.processed_raster_count)
            model_df[f"{prefix}_table_count"] = int(s.processed_table_count)
        _emit("ALIGN", "已添加平台状态特征到建模CSV", {"platform_count": len(statuses)}, task_id=self.task_id)

    def _append_coordinate_diagnostic_features(self, model_df: Any) -> None:
        lon = model_df["lon"].astype(float)
        lat = model_df["lat"].astype(float)
        lon0, lat0 = float(lon.mean()), float(lat.mean())
        lon_span = max(float(lon.max() - lon.min()), 1e-6)
        lat_span = max(float(lat.max() - lat.min()), 1e-6)
        model_df["coord_lon_norm"] = (lon - lon0) / lon_span
        model_df["coord_lat_norm"] = (lat - lat0) / lat_span
        model_df["coord_dist_center"] = ((model_df["coord_lon_norm"] ** 2 + model_df["coord_lat_norm"] ** 2) ** 0.5)
        model_df["coord_lon_lat_interaction"] = model_df["coord_lon_norm"] * model_df["coord_lat_norm"]

    def _weighted_quantile_1d(self, values: Any, weights: Any, q: float) -> float:
        """Small weighted-quantile helper used by spatial conformal intervals."""
        if np is None:
            return float("nan")
        vals = np.asarray(values, dtype="float64")
        w = np.asarray(weights, dtype="float64")
        ok = np.isfinite(vals) & np.isfinite(w) & (w > 0)
        vals = vals[ok]
        w = w[ok]
        if vals.size == 0:
            return float("nan")
        order = np.argsort(vals)
        vals = vals[order]
        w = w[order]
        cw = np.cumsum(w)
        cutoff = float(q) * float(cw[-1])
        idx = int(np.searchsorted(cw, cutoff, side="left"))
        idx = min(max(idx, 0), vals.size - 1)
        return float(vals[idx])

    def _build_kfold_conformal_calibration(self, y_true: Any, y_oof_pred: Any, out_dir: Path, samples_df: Any | None = None, target: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build conformal/GCP calibration from KFold OOF residuals.

        V83 used a single global residual quantile, therefore the uncertainty-width map was
        spatially constant wherever the final mask was valid. The formal workflow keeps that global conformal
        value as a safe fallback, but also stores calibration-point coordinates and prepares
        local, distance-weighted residual quantiles for spatially varying uncertainty maps.

        This remains an engineering GCP bridge, not a final paper-grade geographically weighted
        conformal predictor. The report explicitly marks the method and all local-quantile knobs.
        """
        if np is None:
            return {}
        alpha = float(os.getenv("PRO_GCP_ALPHA", "0.10"))
        alpha = min(max(alpha, 0.01), 0.50)
        y_true = np.asarray(y_true, dtype="float64")
        y_oof_pred = np.asarray(y_oof_pred, dtype="float64")
        valid = np.isfinite(y_true) & np.isfinite(y_oof_pred)
        if int(valid.sum()) < 8:
            return {}
        yv = y_true[valid]
        pv = y_oof_pred[valid]
        residual = np.abs(yv - pv)
        n = len(residual)
        q_rank = int(math.ceil((n + 1) * (1 - alpha)))
        q_rank = min(max(q_rank, 1), n)
        q_global = float(np.sort(residual)[q_rank - 1])
        lower = pv - q_global
        upper = pv + q_global
        covered = (yv >= lower) & (yv <= upper)
        picp = float(np.mean(covered))
        mean_width = float(np.mean(upper - lower))
        y_range = float(np.nanmax(yv) - np.nanmin(yv)) if len(yv) else float("nan")
        nmpiw = float(mean_width / y_range) if y_range and np.isfinite(y_range) and y_range > 0 else None
        below = yv < lower
        above = yv > upper
        interval_score = (upper - lower) + (2 / alpha) * (lower - yv) * below + (2 / alpha) * (yv - upper) * above

        target = target or {}
        target_crs = str(target.get("target_crs") or os.getenv("PRO_GEE_OUTPUT_CRS") or os.getenv("PRO_TARGET_CRS") or "EPSG:32648").strip()
        spatial_enabled = os.getenv("PRO_GCP_SPATIAL_ENABLE", "1") == "1"
        k_neighbors = int(os.getenv("PRO_GCP_SPATIAL_K", "48"))
        min_neighbors = int(os.getenv("PRO_GCP_SPATIAL_MIN_K", "24"))
        power = float(os.getenv("PRO_GCP_SPATIAL_POWER", "1.5"))
        shrink = float(os.getenv("PRO_GCP_SPATIAL_SHRINKAGE", "0.35"))
        min_scale = float(os.getenv("PRO_GCP_SPATIAL_MIN_SCALE", "0.65"))
        max_scale = float(os.getenv("PRO_GCP_SPATIAL_MAX_SCALE", "1.80"))

        cal_x = None
        cal_y = None
        cal_lon = None
        cal_lat = None
        if samples_df is not None:
            try:
                sdf = samples_df.loc[valid].copy()
                if "lon" in sdf.columns and "lat" in sdf.columns:
                    cal_lon = pd.to_numeric(sdf["lon"], errors="coerce").to_numpy(dtype="float64")
                    cal_lat = pd.to_numeric(sdf["lat"], errors="coerce").to_numpy(dtype="float64")
                    if Transformer is not None:
                        tr = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
                        cal_x, cal_y = tr.transform(cal_lon, cal_lat)
                        cal_x = np.asarray(cal_x, dtype="float64")
                        cal_y = np.asarray(cal_y, dtype="float64")
            except Exception as exc:
                self.warnings.append(f"空间GCP校准点坐标构建失败，已采用全局非一致性阈值：{exc}")
                cal_x = cal_y = cal_lon = cal_lat = None

        has_xy = cal_x is not None and cal_y is not None and np.isfinite(cal_x).sum() >= max(8, min_neighbors)
        report = {
            "method": "spatial_weighted_kfold_conformal_residual" if (spatial_enabled and has_xy) else "global_kfold_conformal_residual",
            "alpha": float(alpha),
            "nominal_coverage": float(1 - alpha),
            "sample_count": int(n),
            "residual_quantile": q_global,
            "global_residual_quantile": q_global,
            "mean_width": mean_width,
            "PICP": picp,
            "MPIW": mean_width,
            "NMPIW": nmpiw,
            "CCB": float(abs(picp - (1 - alpha))),
            "IntervalScore": float(np.mean(interval_score)),
            "spatial_enabled": bool(spatial_enabled and has_xy),
            "spatial_config": {
                "target_crs": target_crs,
                "k_neighbors": int(k_neighbors),
                "min_neighbors": int(min_neighbors),
                "power": float(power),
                "shrinkage_to_global": float(shrink),
                "min_scale_vs_global_q": float(min_scale),
                "max_scale_vs_global_q": float(max_scale),
            },
            "note": "自动不确定性分析：全局共形q作为安全底座，同时用校准点OOF残差的空间邻域加权分位数生成空间变化的区间宽度。该结果采用空间加权共形预测流程，建议结合空间验证结果进行解释。",
        }
        try:
            cal_csv = out_dir / "gee_gcp_kfold_calibration_points.csv"
            payload = {
                "som": yv,
                "oof_pred": pv,
                "abs_residual": residual,
                "global_lower": lower,
                "global_upper": upper,
                "global_covered": covered.astype(int),
            }
            if cal_lon is not None and cal_lat is not None:
                payload["lon"] = cal_lon
                payload["lat"] = cal_lat
            if cal_x is not None and cal_y is not None:
                payload["x"] = cal_x
                payload["y"] = cal_y
            pd.DataFrame(payload).to_csv(cal_csv, index=False, encoding="utf-8-sig")
            report["calibration_points_csv"] = str(cal_csv)
        except Exception:
            pass
        return {
            "q": q_global,
            "alpha": alpha,
            "metrics": report,
            "spatial_enabled": bool(spatial_enabled and has_xy),
            "target_crs": target_crs,
            "cal_x": cal_x.tolist() if cal_x is not None else None,
            "cal_y": cal_y.tolist() if cal_y is not None else None,
            "residuals": residual.tolist(),
            "spatial_config": report["spatial_config"],
        }

    def _compute_spatial_gcp_half_widths(self, grid_df: Any, valid_pred: Any, width: int, height: int) -> tuple[Any, dict[str, Any]]:
        """Return spatially varying conformal half-width array for the prediction grid."""
        if np is None or pd is None:
            raise RuntimeError("缺少 numpy/pandas，不能计算空间GCP宽度。")
        cal = self._gee_gcp_calibration or {}
        q_global = float(cal.get("q", 0.0))
        half = np.full((height, width), q_global, dtype="float32")
        if not cal.get("spatial_enabled"):
            half[~valid_pred] = -9999.0
            return half, {"spatial_applied": False, "reason": "spatial_disabled_or_no_calibration_xy", "global_q": q_global}
        cal_x = np.asarray(cal.get("cal_x") or [], dtype="float64")
        cal_y = np.asarray(cal.get("cal_y") or [], dtype="float64")
        residuals = np.asarray(cal.get("residuals") or [], dtype="float64")
        ok = np.isfinite(cal_x) & np.isfinite(cal_y) & np.isfinite(residuals)
        cal_x, cal_y, residuals = cal_x[ok], cal_y[ok], residuals[ok]
        if len(residuals) < 8:
            half[~valid_pred] = -9999.0
            return half, {"spatial_applied": False, "reason": "too_few_valid_calibration_points", "global_q": q_global}

        cfg = cal.get("spatial_config") or {}
        k = min(int(cfg.get("k_neighbors") or os.getenv("PRO_GCP_SPATIAL_K", "48")), len(residuals))
        k = max(1, k)
        power = float(cfg.get("power") or os.getenv("PRO_GCP_SPATIAL_POWER", "1.5"))
        shrink = float(cfg.get("shrinkage_to_global") or os.getenv("PRO_GCP_SPATIAL_SHRINKAGE", "0.35"))
        min_scale = float(cfg.get("min_scale_vs_global_q") or os.getenv("PRO_GCP_SPATIAL_MIN_SCALE", "0.65"))
        max_scale = float(cfg.get("max_scale_vs_global_q") or os.getenv("PRO_GCP_SPATIAL_MAX_SCALE", "1.80"))
        alpha = float(cal.get("alpha", os.getenv("PRO_GCP_ALPHA", "0.10")))
        q_prob = 1.0 - alpha

        # Derive prediction-cell coordinates in the same projected CRS as calibration points.
        coords = []
        index_rc = []
        if "row" in grid_df.columns and "col" in grid_df.columns and "x" in grid_df.columns and "y" in grid_df.columns:
            rows = pd.to_numeric(grid_df["row"], errors="coerce").astype("Int64")
            cols = pd.to_numeric(grid_df["col"], errors="coerce").astype("Int64")
            xs = pd.to_numeric(grid_df["x"], errors="coerce").to_numpy(dtype="float64")
            ys = pd.to_numeric(grid_df["y"], errors="coerce").to_numpy(dtype="float64")
            for rr, cc, x, y in zip(rows, cols, xs, ys):
                if pd.isna(rr) or pd.isna(cc) or not np.isfinite(x) or not np.isfinite(y):
                    continue
                r, c = int(rr), int(cc)
                if 0 <= r < height and 0 <= c < width and bool(valid_pred[r, c]):
                    coords.append((float(x), float(y)))
                    index_rc.append((r, c))
        else:
            # Fallback: only compute for all valid cells by row-major order if x/y are unavailable.
            rr, cc = np.where(valid_pred)
            return half, {"spatial_applied": False, "reason": "grid_xy_missing", "global_q": q_global, "valid_cells": int(len(rr))}

        if not coords:
            half[~valid_pred] = -9999.0
            return half, {"spatial_applied": False, "reason": "no_valid_prediction_cells", "global_q": q_global}

        coords_np = np.asarray(coords, dtype="float64")
        cal_xy = np.column_stack([cal_x, cal_y])
        local_qs = np.full(len(coords_np), q_global, dtype="float64")
        try:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
            nn.fit(cal_xy)
            batch = int(os.getenv("PRO_GCP_SPATIAL_BATCH", "10000"))
            for start in range(0, len(coords_np), batch):
                end = min(start + batch, len(coords_np))
                dist, ind = nn.kneighbors(coords_np[start:end], return_distance=True)
                for i in range(end - start):
                    d = dist[i]
                    rridx = ind[i]
                    w = 1.0 / np.power(np.maximum(d, 1.0), power)
                    q_loc = self._weighted_quantile_1d(residuals[rridx], w, q_prob)
                    if not np.isfinite(q_loc):
                        q_loc = q_global
                    # shrink local quantile toward global q to avoid unstable islands.
                    q_loc = (1.0 - shrink) * q_loc + shrink * q_global
                    q_loc = min(max(q_loc, q_global * min_scale), q_global * max_scale)
                    local_qs[start + i] = q_loc
        except Exception as exc:
            half[~valid_pred] = -9999.0
            return half, {"spatial_applied": False, "reason": f"nearest_neighbor_failed:{exc}", "global_q": q_global}

        for qv, (r, c) in zip(local_qs, index_rc):
            half[r, c] = float(qv)
        half[~valid_pred] = -9999.0
        q_valid = half[valid_pred]
        return half, {
            "spatial_applied": True,
            "method": "nearest_neighbor_idw_weighted_residual_quantile",
            "global_q": q_global,
            "local_q_min": float(np.nanmin(q_valid)) if q_valid.size else None,
            "local_q_mean": float(np.nanmean(q_valid)) if q_valid.size else None,
            "local_q_max": float(np.nanmax(q_valid)) if q_valid.size else None,
            "valid_prediction_cells": int(valid_pred.sum()),
            "calibration_points": int(len(residuals)),
            "k_neighbors": int(k),
            "power": float(power),
            "shrinkage_to_global": float(shrink),
        }

    def _make_spatial_groups(self, samples_df: Any) -> Any:
        """Create coarse spatial-block groups from lon/lat for formal validation."""
        if np is None or pd is None or samples_df is None:
            return None
        if "lon" not in samples_df.columns or "lat" not in samples_df.columns:
            return None
        bins = int(os.getenv("PRO_SPATIAL_CV_GRID_BINS", "5"))
        bins = max(2, min(bins, 12))
        lon = pd.to_numeric(samples_df["lon"], errors="coerce").to_numpy(dtype="float64")
        lat = pd.to_numeric(samples_df["lat"], errors="coerce").to_numpy(dtype="float64")
        if not np.isfinite(lon).any() or not np.isfinite(lat).any():
            return None
        lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
        lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
        if lon_max <= lon_min or lat_max <= lat_min:
            return None
        lon_edges = np.linspace(lon_min, lon_max, bins + 1)
        lat_edges = np.linspace(lat_min, lat_max, bins + 1)
        gx = np.clip(np.digitize(lon, lon_edges[1:-1], right=False), 0, bins - 1)
        gy = np.clip(np.digitize(lat, lat_edges[1:-1], right=False), 0, bins - 1)
        groups = (gy * bins + gx).astype("int64")
        return groups

    def _base_rf_params_from_env(self) -> dict[str, Any]:
        """RandomForest baseline parameters before user override / auto tuning."""
        return {
            "n_estimators": int(os.getenv("PRO_PLATFORM_RF_TREES", "500" if self.is_formal_run else "120")),
            "max_depth": None if os.getenv("PRO_RF_MAX_DEPTH", "").strip() == "" else int(os.getenv("PRO_RF_MAX_DEPTH")),
            "min_samples_leaf": int(os.getenv("PRO_RF_MIN_SAMPLES_LEAF", "1")),
            "min_samples_split": int(os.getenv("PRO_RF_MIN_SAMPLES_SPLIT", "2")),
            "max_features": float(os.getenv("PRO_RF_MAX_FEATURES", "1.0")),
            "random_state": 42,
            "n_jobs": int(os.getenv("PRO_RF_N_JOBS", "1")),
        }

    def _rf_params(self) -> dict[str, Any]:
        """RandomForest parameters used by both RF and RFK formal pipelines."""
        return dict(self._active_rf_params or self._base_rf_params_from_env())

    def _parse_user_model_params(self, text: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Parse explicit model parameters from user text.

        Supported examples:
        - n_estimators=800 / 树数800 / 随机森林树数为800
        - max_depth=16 / 最大深度16
        - min_samples_leaf=2 / 叶节点最小样本2
        - max_features=0.7 / 特征比例0.7
        - variogram_model=spherical / 变差函数 spherical / nlags=8
        """
        text = text or ""
        rf: dict[str, Any] = {}
        rfk: dict[str, Any] = {}

        def _int_patterns(keys: list[str]) -> int | None:
            for pat in keys:
                m = re.search(pat, text, flags=re.I)
                if m:
                    try:
                        return int(float(m.group(1)))
                    except Exception:
                        pass
            return None

        def _float_patterns(keys: list[str]) -> float | None:
            for pat in keys:
                m = re.search(pat, text, flags=re.I)
                if m:
                    try:
                        return float(m.group(1))
                    except Exception:
                        pass
            return None

        v = _int_patterns([r"n_estimators\s*[=:：为]?\s*(\d+)", r"(?:树数|树的数量|随机森林树数|RF树数)\s*[=:：为]?\s*(\d+)"])
        if v is not None: rf["n_estimators"] = max(50, min(v, 3000))
        v = _int_patterns([r"max_depth\s*[=:：为]?\s*(\d+)", r"(?:最大深度|树深度|深度)\s*[=:：为]?\s*(\d+)"])
        if v is not None: rf["max_depth"] = max(2, min(v, 80))
        if re.search(r"max_depth\s*[=:：为]?\s*(None|none|不限|不限制)", text):
            rf["max_depth"] = None
        v = _int_patterns([r"min_samples_leaf\s*[=:：为]?\s*(\d+)", r"(?:最小叶节点样本|叶节点最小样本|叶子最小样本|叶样本)\s*[=:：为]?\s*(\d+)"])
        if v is not None: rf["min_samples_leaf"] = max(1, min(v, 50))
        v = _int_patterns([r"min_samples_split\s*[=:：为]?\s*(\d+)", r"(?:最小分裂样本|分裂最小样本)\s*[=:：为]?\s*(\d+)"])
        if v is not None: rf["min_samples_split"] = max(2, min(v, 100))
        v = _float_patterns([r"max_features\s*[=:：为]?\s*([0-9]*\.?[0-9]+)", r"(?:最大特征比例|特征比例|特征抽样比例)\s*[=:：为]?\s*([0-9]*\.?[0-9]+)"])
        if v is not None: rf["max_features"] = max(0.1, min(float(v), 1.0))
        v = _int_patterns([r"nlags\s*[=:：为]?\s*(\d+)", r"(?:变差函数分组数|滞后组数|半方差分组数)\s*[=:：为]?\s*(\d+)"])
        if v is not None: rfk["nlags"] = max(4, min(v, 30))
        m = re.search(r"(?:variogram_model|变差函数|半方差模型)\s*[=:：为]?\s*(spherical|exponential|gaussian|linear|power)", text, flags=re.I)
        if m:
            rfk["variogram_model"] = m.group(1).lower()
        return rf, rfk

    def _sample_size_rf_defaults(self, n_region: int, n_total: int) -> dict[str, Any]:
        """Conservative defaults by target-AOI sample count."""
        scope = os.getenv("PRO_PARAM_SAMPLE_SCOPE", "training").strip().lower()
        n_ref = int(n_region if scope in {"aoi", "target", "region"} and n_region else n_total or n_region or 0)
        base = self._base_rf_params_from_env()
        if n_ref < 30:
            base.update({"n_estimators": 400, "max_depth": 6, "min_samples_leaf": 4, "min_samples_split": 6, "max_features": 0.6})
        elif n_ref < 80:
            base.update({"n_estimators": 600, "max_depth": 10, "min_samples_leaf": 3, "min_samples_split": 4, "max_features": 0.7})
        elif n_ref < 200:
            base.update({"n_estimators": 800, "max_depth": 16, "min_samples_leaf": 2, "min_samples_split": 2, "max_features": 0.8})
        else:
            base.update({"n_estimators": 1000, "max_depth": None, "min_samples_leaf": 1, "min_samples_split": 2, "max_features": 1.0})
        return base

    def _rf_tuning_candidates(self, n_region: int, n_total: int) -> list[dict[str, Any]]:
        base = self._sample_size_rf_defaults(n_region, n_total)
        scope = os.getenv("PRO_PARAM_SAMPLE_SCOPE", "training").strip().lower()
        n_ref = int(n_region if scope in {"aoi", "target", "region"} and n_region else n_total or n_region or 0)
        if n_ref < 30:
            grid = {"n_estimators": [300, 500, 700], "max_depth": [4, 6, 8, 10], "min_samples_leaf": [3, 5, 8], "max_features": ["sqrt", 0.5, 0.8]}
        elif n_ref < 80:
            grid = {"n_estimators": [500, 700, 900], "max_depth": [6, 10, 14, None], "min_samples_leaf": [2, 4, 6], "max_features": ["sqrt", 0.5, 0.8, 1.0]}
        elif n_ref < 200:
            grid = {"n_estimators": [600, 900, 1200], "max_depth": [8, 14, 20, None], "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.5, 0.7, 1.0]}
        else:
            grid = {"n_estimators": [600, 900, 1200], "max_depth": [8, 14, 24, None], "min_samples_leaf": [1, 2, 4, 6], "max_features": ["sqrt", 0.5, 0.7, 1.0]}
        cand = []
        for vals in itertools.product(grid["n_estimators"], grid["max_depth"], grid["min_samples_leaf"], grid["max_features"]):
            p = dict(base)
            p.update({"n_estimators": vals[0], "max_depth": vals[1], "min_samples_leaf": vals[2], "min_samples_split": max(2, vals[2] * 2), "max_features": vals[3]})
            cand.append(p)
        # Always include the deterministic sample-size default.
        cand.insert(0, base)
        # De-duplicate and cap budget deterministically.
        seen = set(); out = []
        for c in cand:
            key = json.dumps(c, ensure_ascii=False, sort_keys=True, default=str)
            if key not in seen:
                out.append(c); seen.add(key)
        max_c = int(os.getenv("PRO_RF_TUNING_MAX_CANDIDATES", "32"))
        return out[:max(1, max_c)]

    def _score_rf_candidate(self, params: dict[str, Any], X: Any, yv: Any, samples_df: Any) -> dict[str, Any]:
        if np is None or len(yv) < 8:
            return {"ok": False, "error": "insufficient_samples"}
        groups = self._make_spatial_groups(samples_df)
        splits = []
        if groups is not None:
            valid_group_count = int(len(set([int(g) for g in groups if np.isfinite(g)])))
            if valid_group_count >= 3 and GroupShuffleSplit is not None:
                n_splits = min(int(os.getenv("PRO_RF_TUNING_SPATIAL_SPLITS", "3")), max(2, valid_group_count - 1))
                gss = GroupShuffleSplit(n_splits=n_splits, test_size=float(os.getenv("PRO_RF_TUNING_TEST_SIZE", "0.30")), random_state=42)
                splits = list(gss.split(X, yv, groups=groups))
        if not splits and KFold is not None:
            k = min(5, max(3, len(yv) // 10)) if len(yv) >= 30 else min(3, len(yv))
            if k >= 2:
                splits = list(KFold(n_splits=k, shuffle=True, random_state=42).split(X, yv))
        if not splits:
            return {"ok": False, "error": "no_cv_splits"}
        pooled_y, pooled_pred = [], []
        for train_idx, test_idx in splits:
            model = RandomForestRegressor(**params)
            w = self._aoi_sample_weight(samples_df.iloc[train_idx].copy() if hasattr(samples_df, "iloc") else None)
            if w is not None and len(w) == len(train_idx):
                model.fit(X[train_idx], yv[train_idx], sample_weight=w)
            else:
                model.fit(X[train_idx], yv[train_idx])
            pred = np.asarray(model.predict(X[test_idx]), dtype="float64")
            pooled_y.extend([float(v) for v in yv[test_idx]])
            pooled_pred.extend([float(v) for v in pred])
        rmse = float(math.sqrt(mean_squared_error(pooled_y, pooled_pred))) if mean_squared_error else None
        mae = float(mean_absolute_error(pooled_y, pooled_pred)) if mean_absolute_error else None
        r2 = float(r2_score(pooled_y, pooled_pred)) if r2_score and len(set(pooled_y)) > 1 else None
        return {"ok": True, "rmse": rmse, "mae": mae, "r2": r2, "split_count": len(splits)}

    def _hash_array_for_audit(self, arr: Any, max_items: int = 20000) -> str | None:
        """Small deterministic hash for model-input auditing; does not store raw data."""
        if np is None:
            return None
        try:
            a = np.asarray(arr)
            if a.size > max_items:
                flat = a.reshape(-1)[:max_items]
            else:
                flat = a.reshape(-1)
            return hashlib.md5(np.asarray(flat).tobytes()).hexdigest()[:12]
        except Exception:
            try:
                return hashlib.md5(str(arr).encode('utf-8', errors='ignore')).hexdigest()[:12]
            except Exception:
                return None

    def _feature_matrix_deep_audit(self, df: Any, feature_cols: list[str], y_col: str = "som") -> dict[str, Any]:
        """Audit the exact CSV-to-model matrix before RFK training.

        This catches the class of bugs where sample/covariate fusion succeeds on
        paper but the matrix used by sklearn is constant, over-imputed, missing
        key fields, or silently different from the model-ready CSV.
        """
        audit: dict[str, Any] = {"ok": False}
        if pd is None or np is None or df is None:
            audit["error"] = "pandas/numpy unavailable"
            return audit
        try:
            cols = [c for c in (feature_cols or []) if c in df.columns]
            audit.update({
                "ok": True,
                "row_count": int(len(df)),
                "feature_count": int(len(cols)),
                "feature_cols": [str(c) for c in cols],
                "feature_cols_hash": self._hash_array_for_audit([str(c) for c in cols]),
            })
            if y_col in df.columns:
                y_ser = pd.to_numeric(df[y_col], errors="coerce")
                audit["target"] = {
                    "count": int(y_ser.notna().sum()),
                    "missing_count": int(y_ser.isna().sum()),
                    "min": float(y_ser.min()) if y_ser.notna().any() else None,
                    "max": float(y_ser.max()) if y_ser.notna().any() else None,
                    "mean": float(y_ser.mean()) if y_ser.notna().any() else None,
                    "std": float(y_ser.std()) if y_ser.notna().any() else None,
                    "hash": self._hash_array_for_audit(y_ser.fillna(-999999).to_numpy()),
                }
            rows = []
            for c in cols:
                ser = pd.to_numeric(df[c], errors="coerce")
                miss = int(ser.isna().sum())
                nunique = int(ser.nunique(dropna=True))
                zero_count = int((ser == 0).sum())
                rows.append({
                    "feature": str(c),
                    "valid_count": int(ser.notna().sum()),
                    "missing_count": miss,
                    "missing_rate": float(miss / len(ser)) if len(ser) else None,
                    "zero_count": zero_count,
                    "zero_rate": float(zero_count / len(ser)) if len(ser) else None,
                    "unique_count": nunique,
                    "min": float(ser.min()) if ser.notna().any() else None,
                    "max": float(ser.max()) if ser.notna().any() else None,
                    "mean": float(ser.mean()) if ser.notna().any() else None,
                    "std": float(ser.std()) if ser.notna().any() else None,
                    "near_constant": bool(nunique <= 1),
                })
            audit["features"] = rows
            audit["near_constant_features"] = [r["feature"] for r in rows if r.get("near_constant")]
            audit["high_missing_features"] = [r for r in rows if (r.get("missing_rate") is not None and r.get("missing_rate") > 0.30)]
            try:
                X_num = df[cols].apply(pd.to_numeric, errors="coerce")
                audit["X_hash_pre_impute"] = self._hash_array_for_audit(X_num.fillna(-999999).to_numpy(dtype="float64"))
                # Correlation is only a warning aid; do not let it fail the model.
                corr_pairs = []
                if len(cols) >= 2:
                    corr = X_num.corr(numeric_only=True).abs()
                    for i, a in enumerate(cols):
                        for b in cols[i+1:]:
                            try:
                                v = float(corr.loc[a, b])
                                if np.isfinite(v) and v >= float(os.getenv("PRO_MODEL_AUDIT_HIGH_CORR", "0.98")):
                                    corr_pairs.append({"a": str(a), "b": str(b), "abs_corr": v})
                            except Exception:
                                pass
                audit["high_corr_pairs"] = corr_pairs[:50]
            except Exception as exc:
                audit["matrix_hash_error"] = str(exc)
        except Exception as exc:
            audit["ok"] = False
            audit["error"] = str(exc)
        return audit

    def _write_training_split_audit(self, out_dir: Path, samples_df: Any, X: Any, yv: Any, splits: list[tuple[Any, Any]], split_meta: dict[str, Any], name: str) -> str | None:
        """Save fold membership for 7:3 spatial validation/tuning so split logic is auditable."""
        if pd is None or np is None:
            return None
        try:
            rows = []
            for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
                for idx in train_idx:
                    rows.append({"fold": int(fold_id), "row_index": int(idx), "role": "train"})
                for idx in test_idx:
                    rows.append({"fold": int(fold_id), "row_index": int(idx), "role": "validation"})
            df_split = pd.DataFrame(rows)
            out = Path(out_dir) / f"{name}_7比3划分明细.csv"
            df_split.to_csv(out, index=False, encoding="utf-8-sig")
            return str(out)
        except Exception:
            return None

    def _cv_splits_for_tuning(self, X: Any, yv: Any, samples_df: Any, n_splits_env: str = "PRO_RFK_TUNING_SPATIAL_SPLITS") -> tuple[list[tuple[Any, Any]], dict[str, Any]]:
        """Build reproducible spatial CV splits for RFK tuning/evaluation."""
        if np is None:
            return [], {"splitter": "none", "error": "numpy unavailable"}
        groups = self._make_spatial_groups(samples_df)
        if groups is not None and GroupShuffleSplit is not None:
            valid_group_count = int(len(set([int(g) for g in groups if np.isfinite(g)])))
            if valid_group_count >= 3:
                n_splits = min(int(os.getenv(n_splits_env, "3")), max(2, valid_group_count - 1))
                test_size = float(os.getenv("PRO_RFK_TUNING_TEST_SIZE", os.getenv("PRO_RF_TUNING_TEST_SIZE", "0.30")))
                gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=int(os.getenv("PRO_CV_RANDOM_STATE", "42")))
                return list(gss.split(X, yv, groups=groups)), {
                    "splitter": "GroupShuffleSplit-spatial-block",
                    "split_count": int(n_splits),
                    "test_size": float(test_size),
                    "spatial_group_count": int(valid_group_count),
                    "groups_hash": self._hash_array_for_audit(groups),
                }
        if KFold is not None:
            k = min(5, max(3, len(yv) // 10)) if len(yv) >= 30 else min(3, len(yv))
            if k >= 2:
                kf = KFold(n_splits=k, shuffle=True, random_state=int(os.getenv("PRO_CV_RANDOM_STATE", "42")))
                return list(kf.split(X, yv)), {"splitter": "KFold-random-fallback", "split_count": int(k)}
        return [], {"splitter": "none", "error": "no usable CV split"}

    def _rfk_kriging_candidates(self, user_rfk: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Dynamic Ordinary Kriging search space for RFK residuals.

        PyKrige RK/OK exposes variogram_model, nlags, moving-window closest points,
        weighting and exact_values. We search them as part of the *full RFK* CV score,
        not as a post-hoc variogram diagnostic.
        """
        user_rfk = dict(user_rfk or {})
        if user_rfk.get("variogram_model"):
            models = [str(user_rfk["variogram_model"]).lower()]
        else:
            models = [x.strip().lower() for x in os.getenv("PRO_RFK_VARIAGRAM_CANDIDATES", "spherical,exponential,gaussian,linear,power").split(",") if x.strip()]
        if user_rfk.get("nlags"):
            lags = [int(user_rfk["nlags"])]
        else:
            lags = []
            for x in os.getenv("PRO_RFK_NLAGS_CANDIDATES", "4,6,8,10,12").split(","):
                try:
                    lags.append(max(3, int(float(x.strip()))))
                except Exception:
                    pass
            lags = lags or [6]
        if user_rfk.get("n_closest_points"):
            closest = [int(user_rfk["n_closest_points"])]
        else:
            closest = []
            for x in os.getenv("PRO_RFK_CLOSEST_POINTS_CANDIDATES", "0,16,32,64").split(","):
                try:
                    closest.append(max(0, int(float(x.strip()))))
                except Exception:
                    pass
            closest = closest or [0]
        if "weight" in user_rfk:
            weights = [bool(user_rfk["weight"])]
        else:
            weights = [x.strip() in {"1", "true", "yes", "y", "是"} for x in os.getenv("PRO_RFK_WEIGHT_CANDIDATES", "0,1").split(",") if x.strip()]
            weights = weights or [False]
        if "exact_values" in user_rfk:
            exact_values = [bool(user_rfk["exact_values"])]
        else:
            exact_values = [True]
        out: list[dict[str, Any]] = []
        for m, nl, nc, wt, ev in itertools.product(models, lags, closest, weights, exact_values):
            row = {"variogram_model": m, "nlags": int(nl), "n_closest_points": int(nc), "weight": bool(wt), "exact_values": bool(ev)}
            # User supplied fields are locks and override generated values.
            row.update(user_rfk)
            if "nlags" in row:
                row["nlags"] = int(row["nlags"])
            if "n_closest_points" in row:
                row["n_closest_points"] = int(row["n_closest_points"] or 0)
            out.append(row)
        # De-duplicate while preserving order.
        seen = set(); clean = []
        for r in out:
            key = json.dumps(r, ensure_ascii=False, sort_keys=True, default=str)
            if key not in seen:
                clean.append(r); seen.add(key)
        return clean or [{"variogram_model": "spherical", "nlags": 6, "n_closest_points": 0, "weight": False, "exact_values": True}]

    def _rfk_joint_tuning_candidates(self, n_region: int, n_total: int, user_rf: dict[str, Any] | None = None, user_rfk: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Generate RF + OK candidate combinations dynamically for this dataset."""
        user_rf = dict(user_rf or {})
        user_rfk = dict(user_rfk or {})
        rf_candidates = self._rf_tuning_candidates(n_region, n_total)
        # Partial user parameters lock those dimensions, while missing dimensions remain searchable.
        if user_rf:
            locked = []
            seen = set()
            for rf in rf_candidates:
                rr = dict(rf)
                rr.update(user_rf)
                key = json.dumps(rr, ensure_ascii=False, sort_keys=True, default=str)
                if key not in seen:
                    locked.append(rr); seen.add(key)
            rf_candidates = locked
        krig_candidates = self._rfk_kriging_candidates(user_rfk)
        combos = [{"rf_params": rf, "rfk_params": kg} for rf in rf_candidates for kg in krig_candidates]
        max_c = int(os.getenv("PRO_RFK_TUNING_MAX_CANDIDATES", "64"))
        max_c = max(1, max_c)
        if len(combos) <= max_c:
            return combos
        # Deterministic coverage across the whole product rather than always taking the first block.
        idx = np.linspace(0, len(combos) - 1, max_c).round().astype(int).tolist() if np is not None else list(range(max_c))
        out = []
        seen = set()
        for i in idx:
            c = combos[int(i)]
            key = json.dumps(c, ensure_ascii=False, sort_keys=True, default=str)
            if key not in seen:
                out.append(c); seen.add(key)
        return out

    def _fit_rf_with_params(self, X: Any, yv: Any, samples_df: Any | None, rf_params: dict[str, Any]) -> Any:
        if RandomForestRegressor is None:
            raise RuntimeError("缺少 scikit-learn，无法训练随机森林。")
        params = dict(rf_params or self._rf_params())
        params.setdefault("random_state", int(os.getenv("PRO_MODEL_RANDOM_STATE", "42")))
        params.setdefault("n_jobs", int(os.getenv("PRO_RF_N_JOBS", "1")))
        rf = RandomForestRegressor(**params)
        w = self._aoi_sample_weight(samples_df)
        if w is not None and len(w) == len(yv):
            rf.fit(X, yv, sample_weight=w)
        else:
            rf.fit(X, yv)
        return rf

    def _fit_rfk_state_with_params(self, X: Any, yv: Any, samples_df: Any, target: dict[str, Any] | None, rf_params: dict[str, Any], rfk_params: dict[str, Any]) -> dict[str, Any]:
        """Reference-style RFK: p=covariates, x=coordinates, y=target; RF residuals are kriged."""
        if OrdinaryKriging is None:
            raise RuntimeError("正式 RFK 需要 pykrige。请在 geo_env 中执行：pip install pykrige")
        rf = self._fit_rf_with_params(X, yv, samples_df, rf_params)
        rf_fit = np.asarray(rf.predict(X), dtype="float64")
        residual = np.asarray(yv, dtype="float64") - rf_fit
        x, y, coord_crs = self._coords_xy(samples_df, target)
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(residual)
        state: dict[str, Any] = {
            "algorithm": "RFK",
            "rf": rf,
            "ok": None,
            "kriging_enabled": False,
            "coord_crs": coord_crs,
            "train_n": int(len(yv)),
            "rf_params": dict(rf_params),
            "rfk_params": dict(rfk_params or {}),
            "residual_summary": {
                "mean": float(np.nanmean(residual)),
                "std": float(np.nanstd(residual)),
                "min": float(np.nanmin(residual)),
                "max": float(np.nanmax(residual)),
            },
        }
        if int(finite.sum()) < 8:
            raise RuntimeError(f"RFK残差克里金有效训练点不足：{int(finite.sum())}")
        if float(np.nanstd(residual[finite])) < 1e-9:
            state["kriging_warning"] = "RF残差近似常数，已跳过残差克里金。"
            return state
        kg = dict(rfk_params or {})
        kg.setdefault("variogram_model", "spherical")
        kg.setdefault("nlags", 6)
        kg.setdefault("weight", False)
        kg.setdefault("exact_values", True)
        try:
            ok_kwargs = {
                "variogram_model": str(kg.get("variogram_model", "spherical")).lower(),
                "nlags": int(kg.get("nlags", 6)),
                "coordinates_type": "euclidean",
                "enable_plotting": False,
                "verbose": False,
            }
            # PyKrige versions differ slightly; try richer kwargs then fallback.
            try:
                ok = OrdinaryKriging(
                    x[finite], y[finite], residual[finite],
                    weight=bool(kg.get("weight", False)),
                    exact_values=bool(kg.get("exact_values", True)),
                    **ok_kwargs,
                )
            except TypeError:
                ok = OrdinaryKriging(x[finite], y[finite], residual[finite], **ok_kwargs)
            state.update({
                "ok": ok,
                "kriging_enabled": True,
                "variogram_model": str(kg.get("variogram_model", "spherical")).lower(),
                "nlags": int(kg.get("nlags", 6)),
                "n_closest_points": int(kg.get("n_closest_points", 0) or 0),
                "weight": bool(kg.get("weight", False)),
                "exact_values": bool(kg.get("exact_values", True)),
                "kriging_train_n": int(finite.sum()),
                "variogram_quality_score": float(self._ordinary_kriging_quality_score(ok)),
                "variogram_search_mode": "full_rfk_cv_selected",
            })
        except Exception as exc:
            raise RuntimeError(f"RFK残差克里金拟合失败：{exc}")
        return state

    def _score_rfk_candidate(self, candidate: dict[str, Any], X: Any, yv: Any, samples_df: Any, target: dict[str, Any]) -> dict[str, Any]:
        """Score a complete RFK candidate by spatial CV: RF + OK residuals in every fold."""
        if np is None or len(yv) < 8:
            return {"ok": False, "error": "insufficient_samples"}
        splits, split_meta = self._cv_splits_for_tuning(X, yv, samples_df)
        if not splits:
            return {"ok": False, "error": "no_cv_splits", "split_meta": split_meta}
        pooled_y: list[float] = []
        pooled_pred: list[float] = []
        fold_rows: list[dict[str, Any]] = []
        rf_params = dict(candidate.get("rf_params") or {})
        rfk_params = dict(candidate.get("rfk_params") or {})
        max_eval_splits = int(os.getenv("PRO_RFK_TUNING_MAX_SPLITS_PER_CANDIDATE", "3"))
        for fold_id, (train_idx, test_idx) in enumerate(splits[:max(1, max_eval_splits)], start=1):
            if len(train_idx) < 8 or len(test_idx) < 2:
                continue
            train_df = samples_df.iloc[train_idx].copy()
            test_df = samples_df.iloc[test_idx].copy()
            try:
                state = self._fit_rfk_state_with_params(X[train_idx], yv[train_idx], train_df, target, rf_params, rfk_params)
                pred = self._predict_model(state, X[test_idx], test_df, target)
                used_kriging = bool(state.get("kriging_enabled"))
                warning = state.get("kriging_warning")
            except Exception as exc:
                if os.getenv("PRO_RFK_TUNING_ALLOW_RF_FALLBACK", "0") != "1":
                    return {"ok": False, "error": str(exc), "split_meta": split_meta}
                rf = self._fit_rf_with_params(X[train_idx], yv[train_idx], train_df, rf_params)
                pred = rf.predict(X[test_idx])
                used_kriging = False
                warning = str(exc)
            pooled_y.extend([float(v) for v in yv[test_idx]])
            pooled_pred.extend([float(v) for v in np.asarray(pred, dtype="float64")])
            fold_rows.append({
                "fold": int(fold_id),
                "train_n": int(len(train_idx)),
                "test_n": int(len(test_idx)),
                "kriging_enabled": bool(used_kriging),
                "kriging_warning": warning,
                "rmse": float(math.sqrt(mean_squared_error(yv[test_idx], pred))) if mean_squared_error else None,
                "mae": float(mean_absolute_error(yv[test_idx], pred)) if mean_absolute_error else None,
                "r2": float(r2_score(yv[test_idx], pred)) if r2_score and len(set(yv[test_idx].tolist())) > 1 else None,
            })
        if not pooled_y or len(set(pooled_y)) <= 1:
            return {"ok": False, "error": "no_valid_fold_predictions", "split_meta": split_meta}
        return {
            "ok": True,
            "rmse": float(math.sqrt(mean_squared_error(pooled_y, pooled_pred))) if mean_squared_error else None,
            "mae": float(mean_absolute_error(pooled_y, pooled_pred)) if mean_absolute_error else None,
            "r2": float(r2_score(pooled_y, pooled_pred)) if r2_score else None,
            "split_count": int(len(fold_rows)),
            "split_meta": split_meta,
            "fold_scores": fold_rows,
        }

    def _resolve_model_parameters(self, X: Any, yv: Any, samples_df: Any, target: dict[str, Any]) -> dict[str, Any]:
        """Resolve model parameters by complete RFK joint CV unless explicitly disabled.

        User-supplied parameters are locks, not a reason to skip tuning: unspecified RF/OK
        dimensions remain searchable. This prevents fixed RFK parameters from silently being
        reused across regions.
        """
        user_rf, user_rfk = self._parse_user_model_params(self._current_request_text)
        overlap = (target or {}).get("sample_aoi_overlap") or {}
        n_region = int(overlap.get("inside_count") or 0)
        n_total = int(len(yv))
        algorithm = str(getattr(self, "model_algorithm", "RFK") or "RFK").upper()
        if algorithm in {"RANDOMFOREST", "RANDOM_FOREST"}:
            algorithm = "RF"

        auto_enable = os.getenv("PRO_RF_AUTO_TUNE", "1") == "1"
        sample_default = self._sample_size_rf_defaults(n_region, n_total)
        if not auto_enable or n_total < int(os.getenv("PRO_RF_TUNING_MIN_TOTAL_SAMPLES", "20")):
            sample_default.update(user_rf)
            self._active_rf_params = sample_default
            self._active_rfk_params = user_rfk
            plan = {
                "mode": "sample_size_default_user_locked" if (user_rf or user_rfk) else "sample_size_default",
                "auto_tune_enabled": auto_enable,
                "region_sample_count": n_region,
                "total_sample_count": n_total,
                "selected_rf_params": sample_default,
                "selected_rfk_params": user_rfk,
                "reason": "样点过少或自动调参关闭，采用按样点数生成的默认参数；用户显式参数已锁定。",
            }
            self._hyperparameter_plan = plan
            _emit("MODEL", "模型参数已按样点数/用户锁定参数设置", plan, task_id=self.task_id)
            return plan

        # RFK: tune the complete RF + residual-OK model, not only the RF trend model.
        if algorithm != "RF" and OrdinaryKriging is not None and os.getenv("PRO_RFK_JOINT_AUTO_TUNE", "1") == "1":
            candidates = self._rfk_joint_tuning_candidates(n_region, n_total, user_rf=user_rf, user_rfk=user_rfk)
            scored: list[dict[str, Any]] = []
            best = None
            for idx, cand in enumerate(candidates, start=1):
                score = self._score_rfk_candidate(cand, X, yv, samples_df, target)
                row = {"candidate_id": idx, "rf_params": cand.get("rf_params"), "rfk_params": cand.get("rfk_params"), **score}
                scored.append(row)
                _emit("MODEL", f"完整RFK联合候选 {idx}/{len(candidates)} 已评估", row, task_id=self.task_id)
                if score.get("ok"):
                    key = (float(score.get("rmse") if score.get("rmse") is not None else 1e18), -float(score.get("r2") if score.get("r2") is not None else -1e18))
                    if best is None or key < best[0]:
                        best = (key, cand, score)
            if best:
                selected_rf = dict(best[1].get("rf_params") or sample_default)
                selected_rfk = dict(best[1].get("rfk_params") or {})
                self._active_rf_params = selected_rf
                self._active_rfk_params = selected_rfk
                plan = {
                    "mode": "full_rfk_joint_auto_tuned_user_locked" if (user_rf or user_rfk) else "full_rfk_joint_auto_tuned",
                    "region_sample_count": n_region,
                    "total_sample_count": n_total,
                    "candidate_count": len(candidates),
                    "selected_rf_params": selected_rf,
                    "selected_rfk_params": selected_rfk,
                    "best_score": best[2],
                    "candidate_scores": scored,
                    "input_hash": {"X": self._hash_array_for_audit(X), "y": self._hash_array_for_audit(yv)},
                    "note": "本轮按完整RFK进行联合交叉验证寻参：每个候选在每个fold内重新训练RF、计算训练残差、拟合OrdinaryKriging，并以RF趋势项+OK残差的验证精度选最优参数。",
                }
                self._hyperparameter_plan = plan
                _emit("MODEL", "完整RFK联合超参数搜索完成", plan, task_id=self.task_id)
                return plan
            # If every kriging candidate fails, fall through to RF tuning only but report it.
            self.warnings.append("完整RFK联合寻参全部失败，已临时退回RF基学习器寻参；请检查pykrige/坐标/变异函数配置。")

        # RF-only fallback or explicit RF algorithm.
        candidates = self._rf_tuning_candidates(n_region, n_total)
        if user_rf:
            candidates = [{**c, **user_rf} for c in candidates]
        scored = []
        best = None
        for idx, params in enumerate(candidates, start=1):
            score = self._score_rf_candidate(params, X, yv, samples_df)
            row = {"candidate_id": idx, "params": params, **score}
            scored.append(row)
            _emit("MODEL", f"RF超参数候选 {idx}/{len(candidates)} 已评估", row, task_id=self.task_id)
            if score.get("ok"):
                key = (float(score.get("rmse") if score.get("rmse") is not None else 1e18), -float(score.get("r2") if score.get("r2") is not None else -1e18))
                if best is None or key < best[0]:
                    best = (key, params, score)
        selected = best[1] if best else {**sample_default, **user_rf}
        self._active_rf_params = selected
        self._active_rfk_params = user_rfk
        plan = {
            "mode": "rf_auto_tuned" if best else "sample_size_default_after_tuning_failed",
            "region_sample_count": n_region,
            "total_sample_count": n_total,
            "candidate_count": len(candidates),
            "selected_rf_params": selected,
            "selected_rfk_params": user_rfk,
            "best_score": best[2] if best else None,
            "candidate_scores": scored,
            "input_hash": {"X": self._hash_array_for_audit(X), "y": self._hash_array_for_audit(yv)},
            "note": "仅在RF算法或完整RFK联合寻参失败时使用RF基学习器寻参。",
        }
        self._hyperparameter_plan = plan
        _emit("MODEL", "RF基学习器超参数搜索完成", plan, task_id=self.task_id)
        return plan

    def _coords_xy(self, samples_df: Any, target: dict[str, Any] | None = None) -> tuple[Any, Any, str]:
        """Return metric/euclidean x/y coordinates for kriging.

        Preference:
        1. already projected grid columns x/y;
        2. transform lon/lat to target CRS, default EPSG:32648;
        3. lon/lat degrees as a last-resort diagnostic fallback.
        """
        if np is None or pd is None:
            raise RuntimeError("缺少 numpy/pandas，无法构建 RFK 坐标。")
        if "x" in samples_df.columns and "y" in samples_df.columns:
            x = pd.to_numeric(samples_df["x"], errors="coerce").to_numpy(dtype="float64")
            y = pd.to_numeric(samples_df["y"], errors="coerce").to_numpy(dtype="float64")
            if np.isfinite(x).all() and np.isfinite(y).all():
                return x, y, "existing_projected_xy"
        lon = pd.to_numeric(samples_df["lon"], errors="coerce").to_numpy(dtype="float64")
        lat = pd.to_numeric(samples_df["lat"], errors="coerce").to_numpy(dtype="float64")
        target_crs = str((target or {}).get("target_crs") or os.getenv("PRO_GEE_OUTPUT_CRS") or os.getenv("PRO_TARGET_CRS_CHINA") or "EPSG:32648").strip()
        if Transformer is not None:
            try:
                tr = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
                x, y = tr.transform(lon, lat)
                x = np.asarray(x, dtype="float64")
                y = np.asarray(y, dtype="float64")
                if np.isfinite(x).all() and np.isfinite(y).all():
                    return x, y, target_crs
            except Exception as exc:
                self.warnings.append(f"RFK坐标投影到 {target_crs} 失败，退回经纬度克里金：{exc}")
        return lon, lat, "EPSG:4326_degree_fallback"

    def _aoi_sample_weight(self, samples_df: Any | None) -> Any | None:
        """Return AOI-aware sample weights for county output.

        When the requested map is a county/district but the available training set is
        city-level, using all city samples is usually more stable than fitting RFK on
        30--50 local points only. However, the target AOI should have more influence
        on RF splits. This weighting is auditable and can be disabled with
        PRO_AOI_SAMPLE_WEIGHT_ENABLE=0.
        """
        if os.getenv("PRO_AOI_SAMPLE_WEIGHT_ENABLE", "1") != "1" or samples_df is None or np is None:
            return None
        try:
            if "是否在目标区内" not in samples_df.columns:
                return None
            inside = pd.to_numeric(samples_df["是否在目标区内"], errors="coerce").fillna(0).to_numpy(dtype="float64")
            if inside.sum() <= 0:
                return None
            factor = float(os.getenv("PRO_AOI_SAMPLE_WEIGHT_FACTOR", "3.0"))
            factor = max(1.0, factor)
            w = np.ones(len(samples_df), dtype="float64")
            w[inside > 0.5] = factor
            return w
        except Exception:
            return None

    def _fit_rf(self, X: Any, yv: Any, samples_df: Any | None = None) -> Any:
        if RandomForestRegressor is None:
            raise RuntimeError("缺少 scikit-learn，无法训练随机森林。")
        rf = RandomForestRegressor(**self._rf_params())
        w = self._aoi_sample_weight(samples_df)
        if w is not None and len(w) == len(yv):
            rf.fit(X, yv, sample_weight=w)
        else:
            rf.fit(X, yv)
        return rf

    def _rfk_variogram_candidates(self) -> list[dict[str, Any]]:
        models = [x.strip().lower() for x in os.getenv("PRO_RFK_VARIAGRAM_CANDIDATES", "spherical,exponential,gaussian").split(",") if x.strip()]
        lags = []
        for x in os.getenv("PRO_RFK_NLAGS_CANDIDATES", "4,6,8,10,12").split(","):
            try:
                lags.append(max(3, int(float(x.strip()))))
            except Exception:
                pass
        out = []
        for m in models:
            for n in (lags or [8]):
                out.append({"variogram_model": m, "nlags": n})
        return out or [{"variogram_model": "spherical", "nlags": 8}]

    def _ordinary_kriging_quality_score(self, ok: Any) -> float:
        """Small score is better. Prefer stable variograms with Q1≈0, Q2≈1 and low cR."""
        try:
            stats = ok.get_statistics()
            if stats and len(stats) >= 3:
                q1, q2, cr = [float(x) for x in stats[:3]]
                return abs(q1) + abs(q2 - 1.0) + 0.01 * abs(cr)
        except Exception:
            pass
        return 1.0

    def _fit_rfk_state(self, X: Any, yv: Any, samples_df: Any, target: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fit the final RFK model with parameters selected by full RFK CV.

        This delegates to the reference-style RFK implementation so final training and
        CV candidates use the same p/x/y separation and RF+OK residual construction.
        """
        rf_params = dict(self._active_rf_params or self._base_rf_params_from_env())
        rfk_params = dict(self._active_rfk_params or {})
        if not rfk_params:
            # Last-resort only: if parameter resolution was bypassed, use the first
            # dynamic kriging candidate rather than hard-coded spherical/nlags=8.
            kcs = self._rfk_kriging_candidates({})
            rfk_params = dict(kcs[0] if kcs else {"variogram_model": "spherical", "nlags": 6, "n_closest_points": 0})
        return self._fit_rfk_state_with_params(X, yv, samples_df, target, rf_params, rfk_params)

    def _krige_residuals(self, state: dict[str, Any], pred_df: Any, target: dict[str, Any] | None = None) -> Any:
        if np is None:
            return None
        n = len(pred_df)
        if not state or not state.get("kriging_enabled") or state.get("ok") is None:
            return np.zeros(n, dtype="float64")
        x, y, _ = self._coords_xy(pred_df, target)
        out = np.zeros(n, dtype="float64")
        chunk = int(os.getenv("PRO_RFK_KRIGING_PRED_CHUNK", "10000"))
        chunk = max(1000, chunk)
        ok = state.get("ok")
        n_closest = int(state.get("n_closest_points") or 0)
        try:
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                xx = np.asarray(x[start:end], dtype="float64")
                yy = np.asarray(y[start:end], dtype="float64")
                try:
                    if n_closest > 0:
                        kk, _ = ok.execute("points", xx, yy, backend="loop", n_closest_points=n_closest)
                    else:
                        kk, _ = ok.execute("points", xx, yy)
                except TypeError:
                    kk, _ = ok.execute("points", xx, yy)
                out[start:end] = np.asarray(kk, dtype="float64")
        except Exception as exc:
            if self.is_formal_run and os.getenv("PRO_RFK_REQUIRE_KRIGING", "1") == "1":
                raise RuntimeError(f"RFK预测阶段残差克里金失败：{exc}")
            self.warnings.append(f"RFK预测阶段残差克里金失败，残差项置零：{exc}")
            return np.zeros(n, dtype="float64")
        return out

    def _predict_model(self, model: Any, X: Any, pred_df: Any | None = None, target: dict[str, Any] | None = None) -> Any:
        """Predict with RF or RFK state."""
        if isinstance(model, dict) and model.get("algorithm") == "RFK":
            rf = model["rf"]
            rf_pred = np.asarray(rf.predict(X), dtype="float64")
            if pred_df is None:
                return rf_pred
            krig = self._krige_residuals(model, pred_df, target)
            return rf_pred + np.asarray(krig, dtype="float64")
        return model.predict(X)

    def _aoi_mask_for_samples(self, samples_df: Any | None) -> Any | None:
        """Return boolean mask for samples inside requested output AOI, if available.

        Global/city-level validation can remain almost identical when the user changes
        only the output county. To make the report genuinely target-sensitive, every
        validation also carries an AOI-subset metric. This lets the UI and txt report
        show whether the model is actually reliable inside the requested county.
        """
        if samples_df is None or pd is None or np is None:
            return None
        if "是否在目标区内" not in getattr(samples_df, "columns", []):
            return None
        try:
            arr = pd.to_numeric(samples_df["是否在目标区内"], errors="coerce").fillna(0).to_numpy(dtype="float64")
            return arr > 0.5
        except Exception:
            return None

    def _aoi_metric_summary(self, y_true: list[float], y_pred: list[float], aoi_flags: list[int], prefix: str = "target_aoi") -> dict[str, Any]:
        if np is None or not y_true or not y_pred or not aoi_flags:
            return {}
        try:
            y_arr = np.asarray(y_true, dtype="float64")
            p_arr = np.asarray(y_pred, dtype="float64")
            m = np.asarray(aoi_flags, dtype="float64") > 0.5
            out = {
                f"{prefix}_test_n": int(m.sum()),
                f"{prefix}_test_ratio": float(m.mean()) if len(m) else 0.0,
            }
            if int(m.sum()) >= int(os.getenv("PRO_TARGET_AOI_METRIC_MIN_TEST_N", "8")) and len(set(y_arr[m].tolist())) > 1:
                out.update({
                    f"{prefix}_r2": float(r2_score(y_arr[m], p_arr[m])) if r2_score else None,
                    f"{prefix}_rmse": float(math.sqrt(mean_squared_error(y_arr[m], p_arr[m]))) if mean_squared_error else None,
                    f"{prefix}_mae": float(mean_absolute_error(y_arr[m], p_arr[m])) if mean_absolute_error else None,
                    f"{prefix}_bias": float(np.nanmean(p_arr[m] - y_arr[m])),
                })
            else:
                out[f"{prefix}_warning"] = "目标AOI验证样点过少，AOI子集指标仅作提示或不计算R²。"
            return out
        except Exception as exc:
            return {f"{prefix}_error": str(exc)}

    def _spatial_group_validation_rfk(self, X: Any, yv: Any, samples_df: Any, target: dict[str, Any] | None = None) -> dict[str, Any]:
        """Formal spatial validation for true RFK."""
        if GroupShuffleSplit is None or np is None:
            return {"cv": "not_available", "cv_error": "sklearn GroupShuffleSplit unavailable"}
        groups = self._make_spatial_groups(samples_df)
        if groups is None:
            return {"cv": "not_available", "cv_error": "lon/lat unavailable for spatial grouping"}
        valid_group_count = int(len(set([int(g) for g in groups if np.isfinite(g)])))
        if valid_group_count < 3 or len(yv) < 12:
            return {"cv": "not_available", "cv_error": f"insufficient samples/groups for spatial RFK CV: n={len(yv)}, groups={valid_group_count}"}
        n_splits = min(int(os.getenv("PRO_SPATIAL_CV_SPLITS", "5")), max(2, valid_group_count - 1))
        test_size = float(os.getenv("PRO_SPATIAL_CV_TEST_SIZE", "0.30"))
        gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
        split_rows: list[dict[str, Any]] = []
        pooled_y: list[float] = []
        pooled_pred: list[float] = []
        pooled_aoi: list[int] = []
        aoi_mask_all = self._aoi_mask_for_samples(samples_df)
        oof_rows: list[dict[str, Any]] = []
        try:
            x_all, y_all, coord_crs = self._coords_xy(samples_df, target)
        except Exception:
            x_all = y_all = None
            coord_crs = "unknown"
        for i, (train_idx, test_idx) in enumerate(gss.split(X, yv, groups=groups), start=1):
            if len(train_idx) < 8 or len(test_idx) < 2:
                continue
            train_df = samples_df.iloc[train_idx].copy()
            test_df = samples_df.iloc[test_idx].copy()
            try:
                state = self._fit_rfk_state(X[train_idx], yv[train_idx], train_df, target)
                pred = self._predict_model(state, X[test_idx], test_df, target)
                kriging_enabled = bool(state.get("kriging_enabled"))
                kriging_warning = state.get("kriging_warning")
            except Exception as exc:
                if os.getenv("PRO_RFK_CV_STRICT", "0") == "1":
                    raise
                # Keep the run going, but label the split as RF fallback. This prevents one ill-conditioned
                # variogram from killing the entire UI while still exposing the weakness in the report.
                rf = self._fit_rf(X[train_idx], yv[train_idx], train_df)
                pred = rf.predict(X[test_idx])
                kriging_enabled = False
                kriging_warning = str(exc)
            row = {
                "split": i,
                "train_n": int(len(train_idx)),
                "test_n": int(len(test_idx)),
                "test_group_count": int(len(set(groups[test_idx].tolist()))),
                "target_aoi_test_n": int(aoi_mask_all[test_idx].sum()) if aoi_mask_all is not None else None,
                "kriging_enabled": kriging_enabled,
                "kriging_warning": kriging_warning,
                "r2": float(r2_score(yv[test_idx], pred)) if r2_score and len(set(yv[test_idx].tolist())) > 1 else None,
                "rmse": float(math.sqrt(mean_squared_error(yv[test_idx], pred))) if mean_squared_error else None,
                "mae": float(mean_absolute_error(yv[test_idx], pred)) if mean_absolute_error else None,
            }
            split_rows.append(row)
            pooled_y.extend([float(v) for v in yv[test_idx]])
            pooled_pred.extend([float(v) for v in pred])
            if aoi_mask_all is not None:
                pooled_aoi.extend([int(v) for v in aoi_mask_all[test_idx]])
            # GCP + AOA requires an auditable calibration/OOF table.  Keep the
            # actual validation holdout predictions instead of fabricating a
            # training-fit residual table.  Repeated GroupShuffleSplit may test a
            # sample more than once; the writer averages repeated predictions by
            # sample_index before saving strict_groupkfold_oof.csv.
            for local_pos, sample_i in enumerate(test_idx):
                try:
                    rec = samples_df.iloc[int(sample_i)].to_dict()
                except Exception:
                    rec = {}
                rec.update({
                    "sample_index": int(sample_i),
                    "fold": int(i),
                    "som_value": float(yv[int(sample_i)]),
                    "y_true": float(yv[int(sample_i)]),
                    "strict_pred_final": float(pred[int(local_pos)]),
                    "pred": float(pred[int(local_pos)]),
                    "validation_kind": "GroupShuffleSplit-spatial-block-RFK",
                    "coord_crs": str(coord_crs),
                })
                try:
                    if x_all is not None and y_all is not None:
                        rec["x_32648"] = float(x_all[int(sample_i)])
                        rec["y_32648"] = float(y_all[int(sample_i)])
                        rec["x"] = float(x_all[int(sample_i)])
                        rec["y"] = float(y_all[int(sample_i)])
                except Exception:
                    pass
                oof_rows.append(rec)
        if not split_rows:
            return {"cv": "failed", "cv_error": "no valid RFK spatial CV split"}
        r2_vals = [r["r2"] for r in split_rows if r.get("r2") is not None and np.isfinite(r.get("r2"))]
        rmse_vals = [r["rmse"] for r in split_rows if r.get("rmse") is not None and np.isfinite(r.get("rmse"))]
        mae_vals = [r["mae"] for r in split_rows if r.get("mae") is not None and np.isfinite(r.get("mae"))]
        metrics = {
            "cv": f"GroupShuffleSplit-spatial-block-RFK-7:3({len(split_rows)} splits)",
            "model_algorithm": "RFK",
            "spatial_group_grid_bins": int(os.getenv("PRO_SPATIAL_CV_GRID_BINS", "5")),
            "spatial_group_count": valid_group_count,
            "test_size": test_size,
            "r2_mean": float(np.mean(r2_vals)) if r2_vals else None,
            "r2_std": float(np.std(r2_vals)) if r2_vals else None,
            "rmse_mean": float(np.mean(rmse_vals)) if rmse_vals else None,
            "mae_mean": float(np.mean(mae_vals)) if mae_vals else None,
            "splits": split_rows,
        }
        if pooled_y and len(set(pooled_y)) > 1:
            metrics["pooled_r2"] = float(r2_score(pooled_y, pooled_pred)) if r2_score else None
            metrics["pooled_rmse"] = float(math.sqrt(mean_squared_error(pooled_y, pooled_pred))) if mean_squared_error else None
            metrics["pooled_mae"] = float(mean_absolute_error(pooled_y, pooled_pred)) if mean_absolute_error else None
            metrics.update(self._aoi_metric_summary(pooled_y, pooled_pred, pooled_aoi, prefix="target_aoi"))
        if oof_rows:
            metrics["_gcp_oof_rows"] = oof_rows
            metrics["strict_oof_raw_row_count"] = int(len(oof_rows))
            try:
                metrics["strict_oof_unique_sample_count"] = int(len({int(r.get("sample_index")) for r in oof_rows if r.get("sample_index") is not None}))
            except Exception:
                pass
        return metrics

    def _spatial_group_validation(self, base_model: Any, X: Any, yv: Any, samples_df: Any) -> dict[str, Any]:
        """Formal default evaluation: repeated GroupShuffleSplit over spatial grid blocks."""
        metrics: dict[str, Any] = {}
        if GroupShuffleSplit is None or clone is None or np is None:
            return {"cv": "not_available", "cv_error": "sklearn GroupShuffleSplit/clone unavailable"}
        groups = self._make_spatial_groups(samples_df)
        if groups is None:
            return {"cv": "not_available", "cv_error": "lon/lat unavailable for spatial grouping"}
        valid_group_count = int(len(set([int(g) for g in groups if np.isfinite(g)])))
        if valid_group_count < 3 or len(yv) < 12:
            return {"cv": "not_available", "cv_error": f"insufficient samples/groups for spatial CV: n={len(yv)}, groups={valid_group_count}"}
        n_splits = min(int(os.getenv("PRO_SPATIAL_CV_SPLITS", "5")), max(2, valid_group_count - 1))
        test_size = float(os.getenv("PRO_SPATIAL_CV_TEST_SIZE", "0.30"))
        gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
        split_rows: list[dict[str, Any]] = []
        pooled_y: list[float] = []
        pooled_pred: list[float] = []
        pooled_aoi: list[int] = []
        aoi_mask_all = self._aoi_mask_for_samples(samples_df)
        for i, (train_idx, test_idx) in enumerate(gss.split(X, yv, groups=groups), start=1):
            if len(train_idx) < 5 or len(test_idx) < 2:
                continue
            m = clone(base_model)
            m.fit(X[train_idx], yv[train_idx])
            pred = m.predict(X[test_idx])
            row = {
                "split": i,
                "train_n": int(len(train_idx)),
                "test_n": int(len(test_idx)),
                "test_group_count": int(len(set(groups[test_idx].tolist()))),
                "r2": float(r2_score(yv[test_idx], pred)) if r2_score and len(set(yv[test_idx].tolist())) > 1 else None,
                "rmse": float(math.sqrt(mean_squared_error(yv[test_idx], pred))) if mean_squared_error else None,
                "mae": float(mean_absolute_error(yv[test_idx], pred)) if mean_absolute_error else None,
            }
            split_rows.append(row)
            pooled_y.extend([float(v) for v in yv[test_idx]])
            pooled_pred.extend([float(v) for v in pred])
            if aoi_mask_all is not None:
                pooled_aoi.extend([int(v) for v in aoi_mask_all[test_idx]])
        if not split_rows:
            return {"cv": "failed", "cv_error": "no valid spatial CV split"}
        r2_vals = [r["r2"] for r in split_rows if r.get("r2") is not None and np.isfinite(r.get("r2"))]
        rmse_vals = [r["rmse"] for r in split_rows if r.get("rmse") is not None and np.isfinite(r.get("rmse"))]
        mae_vals = [r["mae"] for r in split_rows if r.get("mae") is not None and np.isfinite(r.get("mae"))]
        metrics.update({
            "cv": f"GroupShuffleSplit-spatial-block-7:3({len(split_rows)} splits)",
            "spatial_group_grid_bins": int(os.getenv("PRO_SPATIAL_CV_GRID_BINS", "5")),
            "spatial_group_count": valid_group_count,
            "test_size": test_size,
            "r2_mean": float(np.mean(r2_vals)) if r2_vals else None,
            "r2_std": float(np.std(r2_vals)) if r2_vals else None,
            "rmse_mean": float(np.mean(rmse_vals)) if rmse_vals else None,
            "mae_mean": float(np.mean(mae_vals)) if mae_vals else None,
            "splits": split_rows,
        })
        if pooled_y and len(set(pooled_y)) > 1:
            metrics["pooled_r2"] = float(r2_score(pooled_y, pooled_pred)) if r2_score else None
            metrics["pooled_rmse"] = float(math.sqrt(mean_squared_error(pooled_y, pooled_pred))) if mean_squared_error else None
            metrics["pooled_mae"] = float(mean_absolute_error(pooled_y, pooled_pred)) if mean_absolute_error else None
        return metrics

    def _random_kfold_validation_rfk(self, X: Any, yv: Any, samples_df: Any, target: dict[str, Any] | None = None) -> dict[str, Any]:
        """Relaxed/random validation for RFK. It is not the primary scientific metric.

        This runs a shuffled KFold RFK validation and is reported as a loose reference
        beside the stricter spatial validation. It helps users see whether the model is
        merely fitting near-sample variation while still being weak in spatial transfer.
        """
        if KFold is None or np is None or len(yv) < 8:
            return {"cv": "not_available", "cv_error": "insufficient samples or sklearn KFold unavailable"}
        k = min(int(os.getenv("PRO_RELAXED_CV_SPLITS", "5")), len(yv))
        if k < 2:
            return {"cv": "not_available", "cv_error": "not enough folds"}
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        split_rows: list[dict[str, Any]] = []
        pooled_y: list[float] = []
        pooled_pred: list[float] = []
        pooled_aoi: list[int] = []
        aoi_mask_all = self._aoi_mask_for_samples(samples_df)
        for i, (train_idx, test_idx) in enumerate(kf.split(X, yv), start=1):
            if len(train_idx) < 8 or len(test_idx) < 2:
                continue
            train_df = samples_df.iloc[train_idx].copy()
            test_df = samples_df.iloc[test_idx].copy()
            try:
                state = self._fit_rfk_state(X[train_idx], yv[train_idx], train_df, target)
                pred = self._predict_model(state, X[test_idx], test_df, target)
                kriging_enabled = bool(state.get("kriging_enabled"))
                kriging_warning = state.get("kriging_warning")
            except Exception as exc:
                # Keep relaxed validation available; label the fallback explicitly.
                rf = self._fit_rf(X[train_idx], yv[train_idx], train_df)
                pred = rf.predict(X[test_idx])
                kriging_enabled = False
                kriging_warning = str(exc)
            row = {
                "split": i,
                "train_n": int(len(train_idx)),
                "test_n": int(len(test_idx)),
                "target_aoi_test_n": int(aoi_mask_all[test_idx].sum()) if aoi_mask_all is not None else None,
                "kriging_enabled": kriging_enabled,
                "kriging_warning": kriging_warning,
                "r2": float(r2_score(yv[test_idx], pred)) if r2_score and len(set(yv[test_idx].tolist())) > 1 else None,
                "rmse": float(math.sqrt(mean_squared_error(yv[test_idx], pred))) if mean_squared_error else None,
                "mae": float(mean_absolute_error(yv[test_idx], pred)) if mean_absolute_error else None,
                "bias": float(np.nanmean(np.asarray(pred, dtype="float64") - np.asarray(yv[test_idx], dtype="float64"))),
            }
            split_rows.append(row)
            pooled_y.extend([float(v) for v in yv[test_idx]])
            pooled_pred.extend([float(v) for v in pred])
            if aoi_mask_all is not None:
                pooled_aoi.extend([int(v) for v in aoi_mask_all[test_idx]])
        if not split_rows:
            return {"cv": "failed", "cv_error": "no valid random RFK CV split"}
        r2_vals = [r["r2"] for r in split_rows if r.get("r2") is not None and np.isfinite(r.get("r2"))]
        rmse_vals = [r["rmse"] for r in split_rows if r.get("rmse") is not None and np.isfinite(r.get("rmse"))]
        mae_vals = [r["mae"] for r in split_rows if r.get("mae") is not None and np.isfinite(r.get("mae"))]
        metrics = {
            "cv": f"KFold-random-RFK({len(split_rows)} splits)",
            "model_algorithm": "RFK",
            "r2_mean": float(np.mean(r2_vals)) if r2_vals else None,
            "r2_std": float(np.std(r2_vals)) if r2_vals else None,
            "rmse_mean": float(np.mean(rmse_vals)) if rmse_vals else None,
            "mae_mean": float(np.mean(mae_vals)) if mae_vals else None,
            "splits": split_rows,
        }
        if pooled_y and len(set(pooled_y)) > 1:
            metrics["pooled_r2"] = float(r2_score(pooled_y, pooled_pred)) if r2_score else None
            metrics["pooled_rmse"] = float(math.sqrt(mean_squared_error(pooled_y, pooled_pred))) if mean_squared_error else None
            metrics["pooled_mae"] = float(mean_absolute_error(pooled_y, pooled_pred)) if mean_absolute_error else None
            metrics["pooled_bias"] = float(np.nanmean(np.asarray(pooled_pred, dtype="float64") - np.asarray(pooled_y, dtype="float64")))
            metrics.update(self._aoi_metric_summary(pooled_y, pooled_pred, pooled_aoi, prefix="target_aoi"))
        return metrics

    def _write_gcp_strict_oof_csv(self, out_dir: Path, metrics: dict[str, Any], feature_cols: list[str]) -> str:
        """Persist the calibration table required by formal GCP + AOA.

        The table is built from real validation holdout predictions emitted by the
        formal spatial RFK validation.  It contains observed SOM, OOF prediction,
        projected coordinates and available covariates so the uncertainty module can
        compute GCP intervals and AOA DI from the same run context.
        """
        if pd is None:
            return ""
        rows = list((metrics or {}).pop("_gcp_oof_rows", []) or [])
        if not rows:
            return ""
        try:
            gcp_dir = out_dir / "03_GCP输入"
            gcp_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(rows)
            # Remove columns that cannot be represented cleanly in CSV.
            for col in list(df.columns):
                if str(col).lower() in {"geometry", "geom"}:
                    df = df.drop(columns=[col])
            # Ensure mandatory column aliases expected by GCP_AOA_formal_uncertainty.py.
            if "som_value" not in df.columns and "有机质" in df.columns:
                df["som_value"] = pd.to_numeric(df["有机质"], errors="coerce")
            if "y_true" not in df.columns and "som_value" in df.columns:
                df["y_true"] = pd.to_numeric(df["som_value"], errors="coerce")
            if "strict_pred_final" not in df.columns and "pred" in df.columns:
                df["strict_pred_final"] = pd.to_numeric(df["pred"], errors="coerce")
            if "pred" not in df.columns and "strict_pred_final" in df.columns:
                df["pred"] = pd.to_numeric(df["strict_pred_final"], errors="coerce")
            if "x" not in df.columns and "x_32648" in df.columns:
                df["x"] = pd.to_numeric(df["x_32648"], errors="coerce")
            if "y" not in df.columns and "y_32648" in df.columns:
                df["y"] = pd.to_numeric(df["y_32648"], errors="coerce")
            if "lon" not in df.columns and "经度" in df.columns:
                df["lon"] = pd.to_numeric(df["经度"], errors="coerce")
            if "lat" not in df.columns and "纬度" in df.columns:
                df["lat"] = pd.to_numeric(df["纬度"], errors="coerce")

            # Average repeated holdout predictions for the same sample while keeping
            # covariate values and identifiers from the first occurrence.
            if "sample_index" in df.columns:
                avg_cols = [c for c in ["som_value", "y_true", "strict_pred_final", "pred", "x_32648", "y_32648", "x", "y", "lon", "lat"] if c in df.columns]
                agg = {c: "first" for c in df.columns if c != "sample_index"}
                for c in avg_cols:
                    agg[c] = "mean"
                df = df.groupby("sample_index", as_index=False).agg(agg)

            for c in ["som_value", "y_true", "strict_pred_final", "pred", "x_32648", "y_32648", "x", "y", "lon", "lat"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            if "abs_residual" not in df.columns and {"som_value", "strict_pred_final"}.issubset(df.columns):
                df["abs_residual"] = (df["som_value"] - df["strict_pred_final"]).abs()

            preferred = [
                "sample_index", "fold", "validation_kind", "coord_crs",
                "lon", "lat", "x_32648", "y_32648", "x", "y",
                "som_value", "y_true", "strict_pred_final", "pred", "abs_residual",
            ]
            covs = [c for c in feature_cols if c in df.columns and c not in preferred]
            rest = [c for c in df.columns if c not in preferred and c not in covs]
            df = df[[c for c in preferred if c in df.columns] + covs + rest]
            out = gcp_dir / "strict_groupkfold_oof.csv"
            df.to_csv(out, index=False, encoding="utf-8-sig")
            metrics["strict_oof_csv"] = str(out)
            metrics["strict_oof_row_count"] = int(len(df))
            _emit("MODEL", "GCP输入校准样点表已生成", {"strict_oof_csv": str(out), "rows": int(len(df))}, task_id=self.task_id)
            return str(out)
        except Exception as exc:
            self.warnings.append(f"GCP输入校准样点表生成失败：{exc}")
            return ""

    def _random_kfold_validation(self, base_model: Any, X: Any, yv: Any) -> tuple[dict[str, Any], Any | None]:
        """Fallback/easy diagnostic only. Formal reports label it as random CV, not spatial validation."""
        if len(yv) < 8 or KFold is None or cross_val_predict is None:
            return ({"cv": "not_available", "cv_error": "insufficient samples or sklearn unavailable"}, None)
        k = min(5, len(yv))
        oof_pred = cross_val_predict(base_model, X, yv, cv=KFold(n_splits=k, shuffle=True, random_state=42), n_jobs=1)
        return ({
            "cv": f"KFold({k}) random-diagnostic",
            "r2": float(r2_score(yv, oof_pred)) if r2_score else None,
            "rmse": float(math.sqrt(mean_squared_error(yv, oof_pred))) if mean_squared_error else None,
            "mae": float(mean_absolute_error(yv, oof_pred)) if mean_absolute_error else None,
        }, oof_pred)


    def _prepare_model_ready_table(self, model_df: Any, feature_cols: list[str], out_dir: Path, target: dict[str, Any]) -> tuple[Any, list[str], dict[str, Any]]:
        """Build the single auditable modeling table used by RFK.

        This is the central DSM table required by the thesis workflow: sample id,
        lon/lat, SOM, and all environmental covariates extracted at sample points.
        RFK training must read this table back from disk; no hidden session/raster
        state should be used as a training source.
        """
        if pd is None:
            return model_df, feature_cols, {"ok": False, "message": "pandas unavailable"}
        df0 = model_df.copy()
        # Normalize mandatory columns and keep them first.
        for col in ["sample_id", "lon", "lat", "som"]:
            if col not in df0.columns:
                if col == "sample_id":
                    df0[col] = range(len(df0))
                else:
                    df0[col] = np.nan if np is not None else None
        df0["lon"] = pd.to_numeric(df0["lon"], errors="coerce")
        df0["lat"] = pd.to_numeric(df0["lat"], errors="coerce")
        df0["som"] = pd.to_numeric(df0["som"], errors="coerce")

        # Mark whether the sample falls inside the requested AOI; do not force this
        # as the training filter by default because city-level samples can be valid
        # training data for a county output. The report makes this explicit.
        aoi_meta = {"enabled": False}
        try:
            geom, meta = self._load_local_admin_geometry(target, target_crs="EPSG:4326")
            aoi_meta = dict(meta or {})
            if geom is not None and meta.get("ok"):
                try:
                    import geopandas as gpd
                    from shapely.geometry import Point
                    pts = gpd.GeoSeries([Point(float(x), float(y)) if np.isfinite(float(x)) and np.isfinite(float(y)) else None for x, y in zip(df0["lon"], df0["lat"])], crs="EPSG:4326")
                    inside = pts.within(geom) | pts.touches(geom)
                    df0["是否在目标区内"] = inside.fillna(False).astype(int).to_numpy()
                    aoi_meta.update({"enabled": True, "inside_count": int(df0["是否在目标区内"].sum()), "sample_count": int(len(df0))})
                except Exception as exc:
                    df0["是否在目标区内"] = ""
                    aoi_meta.update({"enabled": True, "inside_mark_error": str(exc)})
        except Exception as exc:
            df0["是否在目标区内"] = ""
            aoi_meta = {"enabled": False, "error": str(exc)}

        # Drop duplicate feature names while preserving order and remove mask/pseudo columns.
        raw_features = []
        seen = set()
        for c in feature_cols or []:
            c = str(c)
            if c in seen or c in {"sample_id", "som", "lon", "lat", "x", "y"}:
                continue
            if c in GEE_OUTPUT_MASK_COLUMNS or c.startswith("src_") or c.startswith("coord_"):
                continue
            if c in df0.columns:
                raw_features.append(c); seen.add(c)

        # V161: disable LULCcd and use CLCD as the land-cover source.
        disabled_lulccd_features = []
        if self._lulccd_disabled():
            kept_raw = []
            for c in raw_features:
                if self._is_lulccd_name(c):
                    disabled_lulccd_features.append(c)
                else:
                    kept_raw.append(c)
            raw_features = kept_raw

        # Numeric conversion for all candidate covariates.
        for c in raw_features:
            df0[c] = pd.to_numeric(df0[c], errors="coerce")

        # V173: CLCD is not a SOM predictor for cropland-only mapping.
        # CLCD valid classes remain fixed: 1农田, 2森林, 3灌木, 4草地,
        # 5水体, 6冰雪, 7裸地, 8不透水面. Class 1 is used only to build
        # the cropland prediction mask; classes 2-8 are non-cropland and do
        # not enter the feature matrix. This avoids the previous error where
        # CLCD one-hot variables became the main/only deployable model features.
        clcd_encoding = []
        model_candidate_features = []
        clcd_base_cols = []
        clcd_mask_only = self._clcd_mask_only_policy()
        for c in raw_features:
            if self._is_clcd_name(c):
                clcd_base_cols.append(c)
                ser = pd.to_numeric(df0[c], errors="coerce")
                rounded = ser.round()
                valid = ser.notna() & rounded.isin(list(self._clcd_class_map().keys())) & ((ser - rounded).abs() <= 1e-6)
                invalid_count = int((ser.notna() & ~valid).sum())
                df0[c] = ser.where(valid, np.nan)
                if clcd_mask_only:
                    clcd_encoding.append({
                        "base_col": c,
                        "dummy_cols": [],
                        "class_map": self._clcd_class_map(),
                        "invalid_non_1_8_count": invalid_count,
                        "encoding": "mask_only",
                        "resampling": "nearest",
                        "note": "V173：CLCD仅用于耕地掩膜；CLCD=1参与预测网格筛选/显示，CLCD=2-8不入模、不预测。",
                    })
                    continue
                dummy_cols = []
                for code, label in self._clcd_class_map().items():
                    dc = self._clcd_dummy_col(c, code, label)
                    vals = np.where(df0[c].isna(), 0.0, (df0[c].round().astype("Int64") == int(code)).astype(float))
                    df0[dc] = vals
                    dummy_cols.append(dc)
                    model_candidate_features.append(dc)
                clcd_encoding.append({
                    "base_col": c,
                    "dummy_cols": dummy_cols,
                    "class_map": self._clcd_class_map(),
                    "invalid_non_1_8_count": invalid_count,
                    "encoding": "one_hot",
                    "resampling": "nearest",
                    "note": "CLCD按1-8分类变量处理；原始编码列保留审计，不直接作为连续特征入模。",
                })
            else:
                model_candidate_features.append(c)

        missing_rate = {c: float(df0[c].isna().mean()) for c in model_candidate_features}
        # Zero audit: 0 is a valid environmental value by default and is not counted
        # as missing here. This table lets us verify whether a high missing rate is
        # true NoData/NaN, not valid zeros being discarded.
        zero_audit = {}
        for c in model_candidate_features:
            ser = pd.to_numeric(df0[c], errors="coerce")
            zero_count = int((ser == 0).sum())
            zero_audit[c] = {
                "zero_count": zero_count,
                "zero_rate": float(zero_count / len(ser)) if len(ser) else 0.0,
                "nan_or_nodata_count": int(ser.isna().sum()),
                "missing_rate": float(ser.isna().mean()) if len(ser) else 0.0,
                "zero_counted_as_missing_in_table": False,
                "raster_sampling_audit": (getattr(self, "_covariate_value_audit", {}) or {}).get(c),
                "min": float(ser.min()) if ser.notna().any() else None,
                "max": float(ser.max()) if ser.notna().any() else None,
            }
        max_missing = float(os.getenv("PRO_MODEL_FEATURE_MAX_MISSING_RATE", "0.25"))
        min_valid = int(os.getenv("PRO_MODEL_FEATURE_MIN_VALID_SAMPLES", "50"))
        keep_all_first_run = (os.getenv("PRO_FIRST_RUN_KEEP_ALL_COVARIATES", "1") or "1").strip().lower() not in {"0", "false", "no", "off"}
        filtered_features = []
        dropped = []
        quality_flags = []
        for c in model_candidate_features:
            valid_n = int(df0[c].notna().sum())
            mr = float(missing_rate.get(c, 1.0))
            # In first-run DSM, keep almost all usable user covariates and let RFK/importance
            # judge them. Hard-drop only features with no usable values or explicitly disabled sources.
            if valid_n <= 0 or mr >= 1.0:
                dropped.append({"feature": c, "missing_rate": mr, "valid_count": valid_n, "reason": "all_missing_or_no_valid_samples"})
            elif (not keep_all_first_run) and (mr > max_missing or valid_n < min_valid):
                dropped.append({"feature": c, "missing_rate": mr, "valid_count": valid_n, "reason": "missing_or_too_few_valid_samples"})
            else:
                filtered_features.append(c)
                if mr > max_missing or valid_n < min_valid:
                    quality_flags.append({"feature": c, "missing_rate": mr, "valid_count": valid_n, "flag": "kept_for_first_run_but_quality_risk"})
        for c in disabled_lulccd_features:
            dropped.append({"feature": c, "missing_rate": None, "valid_count": None, "reason": "disabled_by_policy_use_CLCD_instead_of_LULCcd"})
        if not filtered_features and model_candidate_features:
            ordered = sorted(model_candidate_features, key=lambda c: (missing_rate.get(c, 1.0), str(c)))
            filtered_features = ordered[:max(1, min(5, len(ordered)))]
            dropped = [d for d in dropped if d.get("feature") not in filtered_features]
            self.warnings.append("可用协变量经过硬过滤后为空，已临时保留缺失率最低的少量变量。")

        df0["协变量缺失数"] = df0[model_candidate_features].isna().sum(axis=1) if model_candidate_features else 0
        df0["协变量缺失率"] = df0[model_candidate_features].isna().mean(axis=1) if model_candidate_features else 0.0

        # 样点质量闸门：RFK 的唯一训练入口就是这个 CSV，因此这里必须把
        # 目标值/坐标无效和协变量缺失过多的样点先剔除，并记录数量。
        row_max_missing = float(os.getenv("PRO_MODEL_ROW_MAX_MISSING_RATE", "0.50"))
        before_rows = int(len(df0))
        valid_core = df0["lon"].notna() & df0["lat"].notna() & df0["som"].notna()
        if filtered_features:
            row_missing_selected = df0[filtered_features].isna().mean(axis=1)
            valid_core = valid_core & (row_missing_selected <= row_max_missing)
        df0 = df0.loc[valid_core].copy().reset_index(drop=True)
        dropped_sample_count = before_rows - int(len(df0))

        # 对保留变量的少量缺失值进行中位数填补；填补发生在建模CSV中，避免模型训练阶段隐式处理。
        imputed_features = []
        for c in filtered_features:
            if c in df0.columns and df0[c].isna().any():
                if self._is_clcd_dummy_col(c):
                    fill_value = 0.0
                    impute_name = "zero_for_missing_clcd_class_indicator"
                else:
                    fill_value = pd.to_numeric(df0[c], errors="coerce").median()
                    if pd.isna(fill_value):
                        fill_value = 0.0
                    impute_name = "median"
                df0[c] = pd.to_numeric(df0[c], errors="coerce").fillna(float(fill_value))
                imputed_features.append({"feature": c, "impute": impute_name, "value": float(fill_value)})

        mandatory = ["sample_id", "lon", "lat", "som", "是否在目标区内", "协变量缺失数", "协变量缺失率"]
        ordered_cols = [c for c in mandatory if c in df0.columns] + [c for c in raw_features if c in df0.columns] + [c for c in model_candidate_features if c in df0.columns]
        # Keep any original diagnostic columns after the auditable modeling columns.
        ordered_cols += [c for c in df0.columns if c not in ordered_cols]
        df0 = df0.loc[:, ordered_cols]
        audit = {
            "ok": True,
            "raw_feature_count": len(raw_features) + len(disabled_lulccd_features),
            "active_raw_feature_count": len(raw_features),
            "model_candidate_feature_count": len(model_candidate_features),
            "model_feature_count": len(filtered_features),
            "dropped_features": dropped,
            "quality_flags": quality_flags,
            "first_run_keep_all_covariates": keep_all_first_run,
            "disabled_lulccd_features": disabled_lulccd_features,
            "clcd_encoding": clcd_encoding,
            "clcd_base_cols": clcd_base_cols,
            "missing_rate": missing_rate,
            "zero_value_audit": zero_audit,
            "raster_value_audit": getattr(self, "_covariate_value_audit", {}) or {},
            "missing_policy": {
                "zero_is_missing_by_default": False,
                "zero_nodata_policy": os.getenv("PRO_ZERO_NODATA_POLICY", "never"),
                "rule": "0 is valid by default and is never filtered as missing unless PRO_ZERO_NODATA_POLICY is manually changed. Missing is limited to raster masks outside valid coverage, explicit non-zero NoData, NaN/non-finite values, and extreme fill-value artifacts.",
            },
            "sample_count_before_gate": before_rows,
            "sample_count_after_gate": int(len(df0)),
            "dropped_sample_count": int(dropped_sample_count),
            "row_max_missing_rate": row_max_missing,
            "imputed_features": imputed_features,
            "aoi": aoi_meta,
            "resolution": target.get("resolution_audit") or {"resolution_m": target.get("resolution_m"), "source": target.get("resolution_source")},
            "training_scope": os.getenv("PRO_RFK_TRAINING_SCOPE", "all_compatible_samples_weighted_aoi"),
            "aoi_sample_weight_enabled": os.getenv("PRO_AOI_SAMPLE_WEIGHT_ENABLE", "1") == "1",
            "aoi_sample_weight_factor": float(os.getenv("PRO_AOI_SAMPLE_WEIGHT_FACTOR", "3.0")),
            "note": "RFK训练只读取建模样点表.csv；V173默认保留全部可用非CLCD协变量；LULCcd默认禁用，CLCD仅作为耕地掩膜，不作为SOM模型特征。",
        }
        target["clcd_encoding"] = clcd_encoding
        target["disabled_lulccd_features"] = disabled_lulccd_features
        audit_path = out_dir / "建模样点表_审计.json"
        try:
            audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            audit["audit_path"] = str(audit_path)
        except Exception:
            pass
        return df0, filtered_features, audit

    def _load_model_ready_table_for_training(self, df: Any, feature_cols: list[str]) -> tuple[Any, list[str]]:
        """Reload RFK inputs from the saved model-ready CSV to enforce one source of truth."""
        csv_path = getattr(self, "_model_ready_csv", None)
        if csv_path and pd is not None:
            try:
                loaded = pd.read_csv(csv_path, encoding="utf-8-sig")
                available = [c for c in feature_cols if c in loaded.columns]
                missing = [c for c in feature_cols if c not in loaded.columns]
                if missing:
                    self.warnings.append("建模样点表中缺少部分特征列，已剔除：" + ", ".join(missing[:12]))
                if available:
                    _emit("MODEL", "已从建模样点表.csv重新读取RFK训练数据", {"path": str(csv_path), "rows": int(len(loaded)), "feature_count": len(available)}, task_id=self.task_id)
                    return loaded, available
            except Exception as exc:
                self.warnings.append(f"重新读取建模样点表失败，临时使用内存表：{exc}")
        return df, feature_cols

    def _train_formal_model(self, df: Any, feature_cols: list[str], out_dir: Path, target: dict[str, Any], real_cov_cols: list[str]) -> tuple[Path | None, Path | None]:
        # V165: model-first RFK core with dynamic fused-CSV features and separate
        # deployable/mappable feature set. The old implementation remains below
        # as an explicit fallback only.
        if (os.getenv("PRO_CLEAN_RFK_V166", "1") or "1").strip().lower() not in {"0", "false", "no", "off"}:
            from services.rfk_clean_v166 import run_clean_rfk_model
            return run_clean_rfk_model(self, df, feature_cols, out_dir, target, real_cov_cols)
        if (os.getenv("PRO_CLEAN_RFK_V165", "0") or "0").strip().lower() not in {"0", "false", "no", "off"}:
            from services.rfk_clean_v165 import run_clean_rfk_model
            return run_clean_rfk_model(self, df, feature_cols, out_dir, target, real_cov_cols)
        if (os.getenv("PRO_CLEAN_RFK_V164", "0") or "0").strip().lower() not in {"0", "false", "no", "off"}:
            from services.rfk_clean_v164 import run_clean_rfk_model
            return run_clean_rfk_model(self, df, feature_cols, out_dir, target, real_cov_cols)
        if RandomForestRegressor is None or np is None:
            self.warnings.append("缺少 sklearn/numpy，跳过模型训练。")
            return None, None
        if self.is_formal_run and not feature_cols:
            raise RuntimeError("正式模式未获得可用于建模的真实环境协变量，已停止输出。")
        df, feature_cols = self._load_model_ready_table_for_training(df, feature_cols)
        tmp = df.dropna(subset=["som"]).copy()
        Xdf = tmp[feature_cols].apply(pd.to_numeric, errors="coerce")
        missing_rate = Xdf.isna().mean().to_dict()
        Xdf = Xdf.fillna(Xdf.median(numeric_only=True)).fillna(0)
        y = pd.to_numeric(tmp["som"], errors="coerce")
        valid = y.notna()
        X = Xdf.loc[valid].to_numpy(dtype="float64")
        yv = y.loc[valid].to_numpy(dtype="float64")
        samples_for_eval = tmp.loc[valid].copy().reset_index(drop=True)

        # V162: deep audit of the exact CSV-to-model matrix. This is intentionally
        # written before any RFK fitting so low precision can be traced to either
        # sample/covariate fusion or to the model itself.
        training_matrix_audit = {
            "rule": "RFK训练只读取建模样点表.csv；本审计记录进入模型前后的特征矩阵。",
            "primary_validation_split": "spatial_group_shuffle_7_3",
            "primary_validation_test_size": float(os.getenv("PRO_SPATIAL_CV_TEST_SIZE", "0.30")),
            "tuning_validation_split": "spatial_group_shuffle_7_3",
            "tuning_validation_test_size": float(os.getenv("PRO_RFK_TUNING_TEST_SIZE", os.getenv("PRO_RF_TUNING_TEST_SIZE", "0.30"))),
            "raw_from_model_ready_csv": self._feature_matrix_deep_audit(samples_for_eval, feature_cols, y_col="som"),
        }
        try:
            imputed_df = pd.DataFrame(X, columns=feature_cols)
            imputed_df["som"] = yv
            training_matrix_audit["post_imputation_matrix"] = self._feature_matrix_deep_audit(imputed_df, feature_cols, y_col="som")
            training_matrix_audit["input_hash"] = {"X": self._hash_array_for_audit(X), "y": self._hash_array_for_audit(yv)}
            audit_p = out_dir / "训练矩阵_审计.json"
            audit_p.write_text(json.dumps(training_matrix_audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            training_matrix_audit["audit_path"] = str(audit_p)
        except Exception as exc:
            training_matrix_audit["write_error"] = str(exc)
        target["training_matrix_audit"] = training_matrix_audit

        if len(yv) < 8:
            raise RuntimeError(f"有效样点数不足，无法训练正式模型：{len(yv)}")
        gee_feature_cols = [c for c in feature_cols if str(c).startswith("cov_gee_") and c not in GEE_OUTPUT_MASK_COLUMNS]
        domestic_feature_cols = [c for c in feature_cols if str(c).startswith("cov_") and not str(c).startswith("cov_gee_")]
        formal_covariate_model = bool(real_cov_cols and (gee_feature_cols or domestic_feature_cols))
        formal_gee_model = bool(gee_feature_cols)
        if self.is_formal_run and not formal_covariate_model:
            raise RuntimeError("正式模式要求使用真实栅格环境协变量（本地/国内栅格）；当前未满足，停止输出。")

        algorithm = self.model_algorithm
        if algorithm in {"RANDOMFOREST", "RANDOM_FOREST", "RF"}:
            algorithm = "RF"
        elif algorithm not in {"RFK", "RF"}:
            self.warnings.append(f"未知 PRO_MODEL_ALGORITHM={self.model_algorithm}，已回退 RFK。")
            algorithm = "RFK"
        if self.is_formal_run and algorithm == "RF" and os.getenv("PRO_ALLOW_FORMAL_RF_ONLY", "0") != "1":
            # User explicitly requested real RFK formal mapping. Keep RF-only only as an explicit escape hatch.
            algorithm = "RFK"
        if algorithm == "RFK" and OrdinaryKriging is None:
            raise RuntimeError("当前正式模型为 RFK，但 Python 环境缺少 pykrige。请执行：pip install pykrige")

        hyperparameter_plan = self._resolve_model_parameters(X, yv, samples_for_eval, target)
        target["model_parameter_plan"] = hyperparameter_plan
        base_rf = RandomForestRegressor(**self._rf_params())
        metrics: dict[str, Any] = {
            "mode": ("gee_plus_domestic_formal_mapping" if (gee_feature_cols and domestic_feature_cols) else ("gee_formal_mapping" if gee_feature_cols else "local_domestic_formal_mapping")),
            "run_mode": self.run_mode,
            "model_algorithm": algorithm,
            "map_scope": target.get("map_scope") or target.get("output_scope") or "full_domain",
            "output_scope": target.get("output_scope") or target.get("map_scope") or "full_domain",
            "mask_non_cropland": bool(target.get("map_scope") == "cropland"),
            "real_covariate_count": len(real_cov_cols),
            "evaluation_primary": "spatial_group_shuffle_rfk" if (self.is_formal_run and algorithm == "RFK") else ("spatial_group_shuffle" if self.is_formal_run else "random_kfold_diagnostic"),
        }
        self._gee_gcp_calibration = None
        strict_oof_csv = ""
        if self.is_formal_run:
            if algorithm == "RFK":
                metrics.update(self._spatial_group_validation_rfk(X, yv, samples_for_eval, target))
                # Always provide a relaxed/random RFK validation as an auxiliary reference.
                # The UI and report label it as looser than spatial CV, not as the primary metric.
                metrics["random_validation"] = self._random_kfold_validation_rfk(X, yv, samples_for_eval, target)
            else:
                metrics.update(self._spatial_group_validation(base_rf, X, yv, samples_for_eval))
                diag, _ = self._random_kfold_validation(base_rf, X, yv)
                metrics["random_validation"] = diag
            if metrics.get("cv") in {"not_available", "failed"}:
                diag, _ = self._random_kfold_validation(base_rf, X, yv)
                metrics["random_diagnostic_when_spatial_unavailable"] = diag
            strict_oof_csv = self._write_gcp_strict_oof_csv(out_dir, metrics, feature_cols)
        else:
            diag, oof_pred = self._random_kfold_validation(base_rf, X, yv)
            metrics.update(diag)
            metrics["random_validation"] = diag
            if formal_gee_model and oof_pred is not None and os.getenv("PRO_GCP_AUTO_FROM_GEE", "0") == "1":
                self._gee_gcp_calibration = self._build_kfold_conformal_calibration(yv, oof_pred, out_dir, samples_df=samples_for_eval, target=target)
                if self._gee_gcp_calibration:
                    metrics["gcp_auto"] = self._gee_gcp_calibration.get("metrics", {})

        if algorithm == "RFK":
            model = self._fit_rfk_state(X, yv, samples_for_eval, target)
            self._rfk_state = model
        else:
            model = base_rf
            model.fit(X, yv)
            self._rfk_state = None

        report = {
            "formal_mapping_pipeline": True,
            "gee_formal_flow": bool(formal_gee_model),
            "formal_covariate_model": bool(formal_covariate_model),
            "run_mode": self.run_mode,
            "sample_source": "user_uploaded_samples",
            "covariate_source": ("Local/domestic raster covariates" if (gee_feature_cols and domestic_feature_cols) else ("Local/domestic raster covariates" if gee_feature_cols else "Local/domestic raster covariates")),
            "model_algorithm": algorithm,
            "map_scope": target.get("map_scope") or target.get("output_scope") or "full_domain",
            "output_scope": target.get("output_scope") or target.get("map_scope") or "full_domain",
            "mask_non_cropland": bool(target.get("map_scope") == "cropland"),
            "not_for_scientific_use": False if formal_covariate_model else True,
            "message": "正式流程：用户上传样点，系统自动接入本地/国内真实栅格协变量；正式模型为 RFK（随机森林 + 残差普通克里金），输出GeoTIFF预测图。" if algorithm == "RFK" else "正式流程：使用真实环境协变量完成建模与预测图输出。",
            "target": target,
            "rows": int(len(yv)),
            "feature_count": int(X.shape[1]),
            "feature_cols": feature_cols,
            "real_covariate_cols": real_cov_cols,
            # Prefer pre-imputation/model-ready missing rates. The table itself may
            # be median-imputed before training, so recomputing missing rate from Xdf
            # here would misleadingly report 0 for every feature.
            "feature_missing_rate": {str(k): float(v) for k, v in ((target.get("model_ready_table") or {}).get("missing_rate") or missing_rate).items()},
            "feature_zero_value_audit": (target.get("model_ready_table") or {}).get("zero_value_audit") or {},
            "raster_value_audit": (target.get("model_ready_table") or {}).get("raster_value_audit") or {},
            "missing_policy": (target.get("model_ready_table") or {}).get("missing_policy") or {},
            "rf_params": self._rf_params(),
            "hyperparameter_plan": self._hyperparameter_plan,
            "sample_aoi_overlap": target.get("sample_aoi_overlap"),
            "sample_region_mismatch": target.get("sample_region_mismatch"),
            "rfk": {
                "enabled": bool(algorithm == "RFK"),
                "variogram_model": (model.get("variogram_model") if isinstance(model, dict) else None),
                "nlags": (model.get("nlags") if isinstance(model, dict) else None),
                "n_closest_points": (model.get("n_closest_points") if isinstance(model, dict) else None),
                "weight": (model.get("weight") if isinstance(model, dict) else None),
                "exact_values": (model.get("exact_values") if isinstance(model, dict) else None),
                "kriging_enabled": bool(model.get("kriging_enabled")) if isinstance(model, dict) else False,
                "coord_crs": (model.get("coord_crs") if isinstance(model, dict) else None),
                "residual_summary": (model.get("residual_summary") if isinstance(model, dict) else None),
                "warning": (model.get("kriging_warning") if isinstance(model, dict) else None),
            },
            "metrics": metrics,
        }
        report_path = out_dir / "formal_model_report.json"
        # First create prediction, then analyze the actual result together with validation and parameters.
        pred_tif = self._make_prediction_tif(model, feature_cols, df, out_dir, target)
        report["model_ready_csv"] = getattr(self, "_model_ready_csv", None)
        report["prediction_grid_csv"] = getattr(self, "_prediction_grid_csv", None)
        report["strict_oof_csv"] = strict_oof_csv
        report["features"] = feature_cols
        report["outputs"] = {
            "pred_tif": str(pred_tif) if pred_tif else "",
            "prediction_tif": str(pred_tif) if pred_tif else "",
            "strict_oof_csv": strict_oof_csv,
            "oof_csv": strict_oof_csv,
            "calibration_csv": strict_oof_csv,
            "model_ready_csv": getattr(self, "_model_ready_csv", None),
            "prediction_grid_csv": getattr(self, "_prediction_grid_csv", None),
            "prediction_grid_covariates_csv": getattr(self, "_prediction_grid_csv", None),
            "grid_covariates_csv": getattr(self, "_prediction_grid_csv", None),
            "report_json": str(report_path),
        }
        report["model_ready_table"] = target.get("model_ready_table")
        report["training_matrix_audit"] = target.get("training_matrix_audit")
        report["prediction_grid_audit"] = target.get("prediction_grid_audit")
        # V171 user-facing CLCD mask products and display policy.
        report["clcd_binary_mask_outputs"] = target.get("clcd_binary_mask_outputs") or {}
        _scope = str(target.get("map_scope") or target.get("output_scope") or "full_domain")
        _lc_label = str(target.get("landcover_class_label") or "目标地类")
        _lc_code = target.get("landcover_class_code")
        _is_lc_mask = bool(target.get("mask_by_landcover") or _scope in {"cropland", "landcover_class"})
        report["display_policy"] = {
            "map_scope": _scope,
            "landcover_class_code": _lc_code,
            "landcover_class_label": _lc_label,
            "som_geotiff": (f"地类制图：仅在CLCD={_lc_code}（{_lc_label}）像元上保留连续SOM预测；其他地类写为NoData。" if _is_lc_mask else "全域制图：在行政区有效预测网格内输出连续SOM预测，不按CLCD地类过滤。"),
            "som_png_preview": (f"目标地类（{_lc_label}）显示连续色带；非目标地类默认透明或按用户设定显示；AOI外透明。" if _is_lc_mask else "行政区有效预测网格显示连续色带；AOI外透明。"),
            "clcd_mask_preview": "如存在CLCD，仅作为制图范围审计/目标地类掩膜来源记录。",
            "clcd_model_policy": "CLCD默认不作为SOM模型特征；是否作为输出掩膜取决于本轮map_scope和landcover_class_code。",
            "target_landcover_color_default": os.getenv("PRO_CLCD_CROPLAND_COLOR", "#2EA043"),
            "non_target_landcover_color_default": os.getenv("PRO_CLCD_NON_CROPLAND_COLOR", "#E53935"),
        }
        try:
            rf_for_importance = model.get("rf") if isinstance(model, dict) else model
            importances = getattr(rf_for_importance, "feature_importances_", None)
            if importances is not None:
                rows = []
                for col, imp in zip(feature_cols, list(importances)):
                    rows.append({"feature": str(col), "label": str(col).replace("cov_", ""), "importance": float(imp)})
                rows.sort(key=lambda r: r.get("importance", 0.0), reverse=True)
                report["feature_importance"] = rows
        except Exception as exc:
            report["feature_importance_error"] = str(exc)
        try:
            # A short fingerprint makes it obvious whether a run reused the same
            # model-ready table/report. If the user changes region/data and this
            # fingerprint does not change, the pipeline is reusing stale inputs.
            fp_src = {
                "target_region": (target or {}).get("region"),
                "target_year": (target or {}).get("year"),
                "map_scope": (target or {}).get("map_scope") or (target or {}).get("output_scope"),
                "mask_non_cropland": bool((target or {}).get("map_scope") == "cropland"),
                "rows": int(len(yv)),
                "feature_cols": list(feature_cols),
                "metrics": metrics,
                "model_ready_csv": getattr(self, "_model_ready_csv", None),
                "prediction_grid_csv": getattr(self, "_prediction_grid_csv", None),
            }
            report["run_fingerprint"] = _public_six_digit_run_code()
            report["public_run_code"] = report.get("run_fingerprint")
            report["public_run_code"] = report["run_fingerprint"]
            report["created_at"] = int(time.time())
        except Exception:
            report["run_fingerprint"] = _public_six_digit_run_code()
            report["public_run_code"] = report.get("run_fingerprint")
        try:
            from services.rfk_result_report_service import generate_ai_rfk_analysis, write_model_record_txt, build_analysis_payload
            report["result_analysis_payload"] = build_analysis_payload(report, pred_tif)
            report["ai_result_analysis"] = generate_ai_rfk_analysis(report, pred_tif)
            model_record = write_model_record_txt(out_dir, report, pred_tif=pred_tif, ai_analysis=report.get("ai_result_analysis"), round_name="第1轮")
            report["model_record_txt"] = str(model_record)
        except Exception as exc:
            report["ai_result_analysis_error"] = str(exc)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        _emit("MODEL", f"建模CSV已完成正式{algorithm}训练", {"report": str(report_path), "pred_tif": str(pred_tif) if pred_tif else None, "model_record_txt": report.get("model_record_txt"), "metrics": metrics}, task_id=self.task_id)
        return report_path, pred_tif

    def _make_prediction_tif(self, model: Any, feature_cols: list[str], df: Any, out_dir: Path, target: dict[str, Any]) -> Path | None:
        """Create RFK prediction raster from a real AOI prediction grid table.

        The prediction workflow mirrors the modeling workflow:
        1) build a regular 250 m grid inside the requested AOI;
        2) extract every raster covariate to that grid;
        3) save 预测网格表.csv as the auditable prediction input;
        4) run RFK on that grid and rasterize only AOI cells.
        """
        if rasterio is None or np is None or pd is None:
            return None
        if Transformer is None:
            raise RuntimeError("缺少 pyproj Transformer，无法构建投影坐标预测网格。")

        # If GEE is explicitly enabled and feature columns are GEE columns, keep the old cloud path.
        gee_feature_cols = [c for c in feature_cols if str(c).startswith("cov_gee_") and c not in GEE_OUTPUT_MASK_COLUMNS]
        formal_gee_model = bool(os.getenv("PRO_GEE_FORMAL_FLOW", "0") == "1" and gee_feature_cols)
        if formal_gee_model:
            gee_out = self._make_gee_formal_prediction_tif(model, feature_cols, df, out_dir, target)
            if gee_out:
                return gee_out

        output_crs = str(target.get("target_crs") or os.getenv("PRO_OUTPUT_CRS") or os.getenv("PRO_TARGET_CRS_CHINA") or "EPSG:32648").strip()
        res = float(target.get("resolution_m") or os.getenv("PRO_OUTPUT_RESOLUTION_M", "250"))
        res = max(res, 1.0)

        # Load AOI polygon in the output projected CRS. This is mandatory for formal county/city output.
        geom, admin_meta = self._load_local_admin_geometry(target, target_crs=output_crs)
        if geom is None or not admin_meta.get("ok"):
            msg = "正式制图未能读取目标行政区边界，无法保证输出范围正确。"
            if os.getenv("PRO_LOCAL_ADMIN_REQUIRE", "1") == "1":
                raise RuntimeError(msg + " " + str((admin_meta or {}).get("message") or ""))
            self.warnings.append(msg + " 已临时退回样点包络范围。")
            # Fallback sample bounds transformed to output CRS.
            tr_fwd = Transformer.from_crs("EPSG:4326", output_crs, always_xy=True)
            xs, ys = tr_fwd.transform(pd.to_numeric(df["lon"], errors="coerce"), pd.to_numeric(df["lat"], errors="coerce"))
            xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
            ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
        else:
            xmin, ymin, xmax, ymax = [float(v) for v in geom.bounds]

        # Snap bounds to the requested resolution, so all rasters and output cells have a stable grid.
        xmin = math.floor(xmin / res) * res
        ymin = math.floor(ymin / res) * res
        xmax = math.ceil(xmax / res) * res
        ymax = math.ceil(ymax / res) * res
        width = int(math.ceil((xmax - xmin) / res))
        height = int(math.ceil((ymax - ymin) / res))
        width = max(width, 1); height = max(height, 1)
        max_cells = int(os.getenv("PRO_MAX_PRED_GRID_CELLS", "500000"))
        cells = width * height
        _emit("MODEL", "预测网格规模预检", {
            "width": int(width),
            "height": int(height),
            "cells": int(cells),
            "max_cells": int(max_cells),
            "resolution_m": float(res),
            "output_crs": output_crs,
            "aoi_bounds_projected": {"xmin": float(xmin), "ymin": float(ymin), "xmax": float(xmax), "ymax": float(ymax)}
        }, task_id=self.task_id)
        if cells > max_cells:
            raise RuntimeError(f"预测网格像元数 {cells} 超过 PRO_MAX_PRED_GRID_CELLS={max_cells}。请提高分辨率或缩小AOI。")

        from rasterio.transform import from_origin
        transform = from_origin(xmin, ymax, res, res)
        # Rasterize AOI mask in output CRS before sampling/prediction.
        _emit("MODEL", "开始栅格化行政区AOI掩膜", {"aoi": target.get("region"), "width": int(width), "height": int(height), "resolution_m": float(res)}, task_id=self.task_id)
        admin_mask, admin_mask_meta = self._local_admin_mask_array(target, width, height, transform, output_crs)
        _emit("MODEL", "行政区AOI掩膜栅格化完成", {"ok": bool(admin_mask is not None), "meta": admin_mask_meta}, task_id=self.task_id)
        if admin_mask is None or int(admin_mask.sum()) <= 0:
            if os.getenv("PRO_LOCAL_ADMIN_REQUIRE", "1") == "1":
                raise RuntimeError("目标行政区mask为空，已停止输出。" + str((admin_mask_meta or {}).get("message") or ""))
            admin_mask = np.ones((height, width), dtype="uint8")
            self.warnings.append("目标行政区mask为空，已临时输出矩形范围。")
        valid_flat = admin_mask.ravel().astype(bool)

        cols_idx = np.arange(width, dtype="float64")
        rows_idx = np.arange(height, dtype="float64")
        xx = xmin + (cols_idx + 0.5) * res
        yy = ymax - (rows_idx + 0.5) * res
        grid_x, grid_y = np.meshgrid(xx, yy)
        flat_x = grid_x.ravel()[valid_flat]
        flat_y = grid_y.ravel()[valid_flat]
        flat_row, flat_col = np.indices((height, width))
        flat_row = flat_row.ravel()[valid_flat]
        flat_col = flat_col.ravel()[valid_flat]
        tr_inv = Transformer.from_crs(output_crs, "EPSG:4326", always_xy=True)
        lon, lat = tr_inv.transform(flat_x, flat_y)
        grid = pd.DataFrame({
            "row": flat_row.astype(int),
            "col": flat_col.astype(int),
            "x": flat_x.astype(float),
            "y": flat_y.astype(float),
            "lon": np.asarray(lon, dtype="float64"),
            "lat": np.asarray(lat, dtype="float64"),
        })

        # Extract every local/domestic raster covariate to the AOI grid. Features with no raster
        # map are not silently invented; they are filled with training medians and recorded.
        grid_sampling_warnings = []
        raster_map = getattr(self, "_covariate_raster_map", {}) or {}
        # V165: merged alias map from fused-CSV feature names to actual standardized rasters.
        # This allows CSV columns such as DEM/pH/LSWI to match pipeline raster keys such as
        # cov_domestic_local_DEM without hard-coding feature names.
        try:
            for _k, _v in (target.get("v165_feature_raster_map") or {}).items():
                if _k and _v:
                    raster_map[str(_k)] = str(_v)
            self._covariate_raster_map = raster_map
        except Exception:
            pass

        # V161: if CLCD was one-hot encoded in the modeling table, sample the base
        # CLCD raster once and recreate the exact same one-hot columns on the prediction grid.
        for enc in (target.get("clcd_encoding") or []):
            if str((enc or {}).get("encoding") or "") != "one_hot":
                continue
            try:
                base_col = enc.get("base_col")
                rp = raster_map.get(base_col)
                if base_col and rp:
                    grid[base_col] = self._sample_raster(Path(rp), grid, str(base_col))
                    base = pd.to_numeric(grid[base_col], errors="coerce")
                    rounded = base.round()
                    valid = base.notna() & rounded.isin(list(self._clcd_class_map().keys())) & ((base - rounded).abs() <= 1e-6)
                    base_clean = rounded.where(valid, np.nan)
                    for code, label in self._clcd_class_map().items():
                        dc = self._clcd_dummy_col(str(base_col), int(code), str(label))
                        grid[dc] = np.where(base_clean.isna(), 0.0, (base_clean.astype("Int64") == int(code)).astype(float))
                else:
                    grid_sampling_warnings.append(f"CLCD base raster missing for one-hot encoding: {base_col}")
            except Exception as exc:
                grid_sampling_warnings.append(f"CLCD one-hot grid encoding failed: {exc}")

        for col in feature_cols:
            rp = raster_map.get(col)
            if rp:
                try:
                    grid[col] = self._sample_raster(Path(rp), grid, col)
                except Exception as exc:
                    grid_sampling_warnings.append(f"{col}: {exc}")
                    grid[col] = np.nan
            elif col in grid.columns:
                continue
            else:
                grid[col] = np.nan

        # V165: strict grid deployability check. A formal map cell is valid only if
        # every selected mappable feature has a real raster value at that cell. Values
        # may be temporarily imputed only for the sklearn predict call; invalid cells
        # are written back to NODATA in the final raster.
        grid_feature_valid_mask = np.ones(len(grid), dtype=bool)
        for _c in feature_cols:
            if _c in grid.columns:
                grid_feature_valid_mask &= pd.to_numeric(grid[_c], errors="coerce").notna().to_numpy(dtype=bool)
            else:
                grid_feature_valid_mask &= False

        # V221: CLCD target land-cover mask.  If the user asks for cropland,
        # forest, grassland, etc., keep only that class.  If the user asks for
        # full-domain mapping, CLCD must not restrict prediction.
        clcd_mask_info = {"enabled": False}
        clcd_target_mask = np.ones(len(grid), dtype=bool)
        def _is_clcd_key_or_path(_x):
            _lo = str(_x or "").lower().replace("-", "_")
            return ("clcd" in _lo) or ("china_land_cover" in _lo) or ("china land cover" in _lo)
        requested_map_scope = str((target or {}).get("map_scope") or (target or {}).get("output_scope") or "full_domain").strip().lower()
        target_class_code = target.get("landcover_class_code")
        try:
            target_class_code = int(target_class_code) if target_class_code is not None and str(target_class_code).strip() != "" else None
        except Exception:
            target_class_code = None
        target_class_label = str(target.get("landcover_class_label") or CLCD_CLASS_LABELS.get(target_class_code, "目标地类"))
        clcd_mask_enable = bool(target.get("mask_by_landcover") or target.get("mask_non_target_landcover") or requested_map_scope in {"cropland", "landcover_class"})
        clcd_mask_enable = clcd_mask_enable and ((os.getenv("PRO_V165_CLCD_CROPLAND_MASK_ENABLE", "1") or "1").strip().lower() not in {"0", "false", "no", "off"})
        if requested_map_scope in {"full_domain", "full", "all", "全域"}:
            clcd_mask_enable = False
            target_class_code = None
        if clcd_mask_enable and target_class_code is None:
            target_class_code = 1
            target_class_label = "耕地/农田"
        clcd_raster = target.get("v165_clcd_mask_raster") or target.get("clcd_mask_raster") or target.get("v166_clcd_mask_raster")
        if not clcd_raster:
            for _k, _v in raster_map.items():
                if _is_clcd_key_or_path(_k) or _is_clcd_key_or_path(_v):
                    clcd_raster = _v; break
        if clcd_mask_enable and not clcd_raster:
            clcd_raster = self._find_builtin_landcover_raster(target)
            if clcd_raster:
                target["clcd_mask_raster"] = str(clcd_raster)
                target["v165_clcd_mask_raster"] = str(clcd_raster)
                try:
                    self._covariate_raster_map["builtin_landcover_mask"] = str(clcd_raster)
                except Exception:
                    pass
                _emit("MASK", "已使用内置全国土地覆盖数据作为目标地类掩膜", {"path": str(clcd_raster), "class_code": target_class_code, "class_label": target_class_label, "builtin_root": r"E:/Agent_DSM/土地覆盖数据"}, task_id=self.task_id)
        if clcd_mask_enable and not clcd_raster:
            self.warnings.append("用户请求目标地类制图，但未找到土地覆盖/CLCD掩膜数据；已停止目标地类掩膜输出，请检查 E:/Agent_DSM/土地覆盖数据。")
        if not clcd_mask_enable and requested_map_scope in {"full_domain", "full", "all", "全域"}:
            clcd_mask_info = {"enabled": False, "reason": "full_domain_mapping_requested", "keep_rule": "用户请求全域制图，预测像元不按CLCD地类过滤。"}
        if clcd_mask_enable and clcd_raster:
            try:
                grid["CLCD_class_for_target_landcover_mask"] = self._sample_raster(Path(clcd_raster), grid, "CLCD_target_landcover_mask")
                _cl = pd.to_numeric(grid["CLCD_class_for_target_landcover_mask"], errors="coerce")
                _rounded = _cl.round()
                clcd_target_mask = _cl.notna().to_numpy(dtype=bool) & (_rounded.to_numpy(dtype="float64") == float(target_class_code)) & ((_cl - _rounded).abs().to_numpy(dtype="float64") <= 1e-6)
                clcd_mask_info = {
                    "enabled": True,
                    "raster": str(clcd_raster),
                    "class_map": CLCD_CLASS_LABELS,
                    "target_class_code": int(target_class_code),
                    "target_class_label": target_class_label,
                    "keep_rule": f"V221：仅 CLCD == {int(target_class_code)}（{target_class_label}）像元进入/保留连续SOM预测；其他地类写为 NoData。",
                    "display_rule": f"SOM预览中仅目标地类（{target_class_label}）使用连续色带；非目标地类默认透明，除非用户明确指定非目标地类颜色。",
                    "target_landcover_cell_count": int(np.sum(clcd_target_mask)),
                    "non_target_or_invalid_cell_count": int(len(clcd_target_mask) - np.sum(clcd_target_mask)),
                    "cropland_cell_count": int(np.sum(clcd_target_mask)) if int(target_class_code) == 1 else None,
                    "non_cropland_or_invalid_cell_count": int(len(clcd_target_mask) - np.sum(clcd_target_mask)) if int(target_class_code) == 1 else None,
                }
            except Exception as exc:
                clcd_mask_info = {"enabled": False, "error": str(exc), "raster": str(clcd_raster), "target_class_code": target_class_code, "target_class_label": target_class_label}
                self.warnings.append(f"目标地类掩膜生成失败，当前结果不能作为目标地类图交付：{exc}")

        if clcd_mask_enable and not bool((clcd_mask_info or {}).get("enabled")):
            raise RuntimeError(
                "用户请求目标地类制图（如耕地/林地/草地），但未能生成有效土地覆盖掩膜；"
                "系统不会退回全域图冒充目标地类图。请确认 E:/Agent_DSM/土地覆盖数据 中存在1-9分类土地覆盖栅格。"
                + (" 详细原因：" + str((clcd_mask_info or {}).get("error")) if (clcd_mask_info or {}).get("error") else "")
            )

        # Compatibility alias: subsequent code historically uses the variable
        # name clcd_cropland_mask.  In V221 it means target land-cover mask.
        clcd_cropland_mask = clcd_target_mask
        grid_final_valid_mask = grid_feature_valid_mask & clcd_target_mask

        grid_missing_rate = {str(c): float(pd.to_numeric(grid[c], errors="coerce").isna().mean()) for c in feature_cols if c in grid.columns}
        high_grid_missing = [
            {"feature": c, "prediction_grid_missing_rate": r}
            for c, r in grid_missing_rate.items()
            if r > float(os.getenv("PRO_PRED_GRID_FEATURE_MAX_MISSING_RATE_WARN", "0.40"))
        ]
        if high_grid_missing:
            self.warnings.append("预测网格中部分协变量缺失率较高：" + ", ".join([x["feature"] for x in high_grid_missing[:8]]))
        # V164: if using a fused CSV without matching raster layers, do not fabricate
        # a full prediction map from training medians. Training/evaluation remains valid,
        # but GeoTIFF output requires matching feature rasters or a prediction grid.
        full_missing_features = [c for c, r in grid_missing_rate.items() if r >= 0.999]
        if full_missing_features and (os.getenv("PRO_V165_ALLOW_FULL_MISSING_GRID_FEATURE", "0") or "0").strip().lower() not in {"1", "true", "yes", "on"}:
            self.warnings.append("V165已停止预测图输出：正式出图特征存在整幅预测栅格缺失，不能用训练中位数伪造整幅图：" + ", ".join(full_missing_features[:12]))
            target["prediction_grid_audit"] = {"stopped": True, "reason": "full_missing_prediction_grid_feature", "full_missing_features": full_missing_features, "grid_missing_rate": grid_missing_rate, "clcd_cropland_mask": clcd_mask_info, "v165_feature_deployment_plan": target.get("v165_feature_deployment_plan")}
            return None

        # Fill temporary NaNs from training medians only for sklearn prediction.
        # Cells that had any missing mappable raster value are set back to NODATA.
        for col in feature_cols:
            if col not in grid.columns:
                grid[col] = np.nan
            train_med = pd.to_numeric(df[col], errors="coerce").median() if col in df.columns else np.nan
            if pd.isna(train_med):
                train_med = 0.0
            grid[col] = pd.to_numeric(grid[col], errors="coerce").fillna(float(train_med))

        # Save the actual prediction table before modeling. This is the second auditable table.
        grid_csv = out_dir / "预测网格表.csv"
        grid.to_csv(grid_csv, index=False, encoding="utf-8-sig")
        self._prediction_grid_csv = str(grid_csv)
        _emit("MODEL", "预测网格表.csv已生成，开始RFK逐像元预测", {
            "path": str(grid_csv),
            "rows": int(len(grid)),
            "width": int(width),
            "height": int(height),
            "aoi_grid_rows": int(len(grid)),
            "cropland_candidate_cells": int(np.sum(clcd_cropland_mask)) if 'clcd_cropland_mask' in locals() else None,
            "output_crs": output_crs,
            "resolution_m": res,
            "aoi": admin_mask_meta,
            "sampling_warnings": grid_sampling_warnings[:12],
            "grid_missing_rate": grid_missing_rate,
            "high_grid_missing_features": high_grid_missing[:20],
        }, task_id=self.task_id)
        target["prediction_grid_audit"] = {
            "path": str(grid_csv),
            "rows": int(len(grid)),
            "grid_missing_rate": grid_missing_rate,
            "high_grid_missing_features": high_grid_missing[:50],
            "strict_feature_valid_cell_count": int(np.sum(grid_feature_valid_mask)),
            "strict_final_valid_cell_count": int(np.sum(grid_final_valid_mask)),
            "clcd_cropland_mask": clcd_mask_info,
            "v165_feature_deployment_plan": target.get("v165_feature_deployment_plan"),
        }

        # At this point grid values have been filled from training medians above.
        # Any remaining NaN is an exceptional fallback; fill with 0 only after audit,
        # never for missing-rate calculation.
        # V173: cropland-only SOM prediction. Non-cropland rows remain in the
        # audit grid so that a red non-cropland preview can be built, but they
        # are not passed to RF/RFK predict at all. The continuous prediction is
        # computed only where CLCD=1 and all non-CLCD mappable features are valid.
        Xg_df = grid[feature_cols].apply(pd.to_numeric, errors="coerce")
        if Xg_df.isna().any().any():
            self.warnings.append("预测网格仍存在未填补缺失值，已在最终矩阵阶段兜底填0；请检查预测网格表.csv。")
        _pv = np.full(len(grid), -9999.0, dtype="float32")
        predict_mask = np.asarray(grid_final_valid_mask, dtype=bool)
        if predict_mask.any():
            Xg = Xg_df.loc[predict_mask, :].fillna(0).to_numpy(dtype="float64")
            pred_part = self._predict_model(model, Xg, grid.loc[predict_mask].copy().reset_index(drop=True), target).astype("float32")
            _pv[predict_mask] = pred_part
        else:
            self.warnings.append("V173耕地SOM预测没有任何可预测像元：请检查CLCD=1掩膜和非CLCD预测栅格是否覆盖AOI。")
        grid["RFK预测有机质"] = _pv
        grid["V173_参与SOM连续预测"] = predict_mask.astype(int)
        grid["V165_最终有效像元"] = predict_mask.astype(int)
        try:
            grid.to_csv(grid_csv, index=False, encoding="utf-8-sig")
        except Exception:
            pass

        pred = np.full((height, width), -9999.0, dtype="float32")
        _rows = grid["row"].to_numpy(dtype=int)
        _cols = grid["col"].to_numpy(dtype=int)
        pred[_rows, _cols] = _pv
        pred[admin_mask == 0] = -9999.0

        # V171: create an explicit two-color CLCD cropland/non-cropland mask product.
        # The SOM GeoTIFF still writes non-cropland as NoData, but the preview/UI can
        # show non-cropland with a visible contrast color instead of white/transparent.
        clcd_mask_outputs = {}
        if bool((clcd_mask_info or {}).get("enabled")):
            clcd_mask_outputs = self._write_clcd_binary_mask_outputs(
                out_dir=out_dir, admin_mask=admin_mask, grid_df=grid,
                cropland_mask=clcd_cropland_mask, rows=_rows, cols=_cols,
                transform=transform, crs=output_crs,
                target_label=str((clcd_mask_info or {}).get("target_class_label") or "目标地类"),
                target_code=(clcd_mask_info or {}).get("target_class_code"),
            )
            clcd_mask_info["binary_mask_outputs"] = clcd_mask_outputs
            target["clcd_binary_mask_outputs"] = clcd_mask_outputs

        out = out_dir / "有机质图_内部.tif"
        with rasterio.open(out, "w", driver="GTiff", height=height, width=width, count=1, dtype="float32", crs=output_crs, transform=transform, nodata=-9999.0, compress="lzw") as dst:
            dst.write(pred, 1)
        try:
            self._write_mask_tif(out_dir / "目标区mask.tif", admin_mask.astype("uint8"), transform, output_crs)
        except Exception:
            pass
        _emit("MODEL", "RFK预测图已按目标行政区裁剪生成", {
            "pred_tif": str(out),
            "prediction_grid_csv": str(grid_csv),
            "output_crs": output_crs,
            "resolution_m": res,
            "width": int(width),
            "height": int(height),
            "valid_cells": int(np.sum(pred != -9999.0)),
            "clcd_cropland_mask": clcd_mask_info,
            "clcd_binary_mask_outputs": clcd_mask_outputs,
            "strict_final_valid_cells": int(np.sum(grid_final_valid_mask)),
        }, task_id=self.task_id)
        return out


    def _default_covariate_catalog(self) -> dict[str, list[str]]:
        """Canonical names and aliases for the default local covariate set.

        The default list mirrors the rasters usually stored in E:/Agent_DSM/data:
        BD, CEC, DEM, gravel, pH, porosity, sand, silt, clay, LULCcd, CLCD,
        bare-soil composite, ChinaCP, management proxy, seasonal climate-water,
        and vegetation phenology.
        """
        return {
            "BD": ["bd", "bulk_density", "bulk density", "容重", "土壤容重", "cov_gee_bulk_density"],
            "CEC": ["cec", "阳离子交换量", "cation", "cov_gee_cec"],
            "DEM": ["dem", "elevation", "高程", "海拔", "地形", "cov_gee_elevation", "cov_gee_slope", "slope", "aspect", "tpi", "roughness", "relief", "curvature", "twi"],
            "gravel": ["gravel", "砾石", "石砾"],
            "pH": ["ph", "pH", "酸碱度", "cov_gee_ph"],
            "porosity": ["porosity", "孔隙度", "孔隙"],
            "sand": ["sand", "砂粒", "砂", "cov_gee_sand"],
            "silt": ["silt", "粉粒", "粉砂", "cov_gee_silt"],
            "clay": ["clay", "黏粒", "粘粒", "cov_gee_clay"],
            "LULCcd": ["lulccd", "lulc", "landuse", "land_use", "landcover", "land_cover", "土地利用", "土地覆盖"],
            "CLCD": ["clcd", "china land cover", "china_land_cover", "中国土地覆盖", "土地覆盖"],
            "BareSoil": ["baresoil", "bare_soil", "bare soil", "裸土", "s2", "sentinel", "sentinel-2"],
            "ChinaCP": ["chinacp", "china_cp", "crop", "cropping", "作物", "种植模式", "复种", "轮作"],
            "Management": ["management", "management_proxy", "管理", "灌溉", "irrig", "irrigation", "耕作"],
            "SeasonalClimateWater": ["seasonalclimatewater", "seasonalclimate", "seasonal_climate", "climatewater", "climatewate", "climate", "water", "气候", "水热", "降水", "温度", "precip", "temp", "pet", "lst", "cov_gee_precip", "cov_gee_temp", "cov_gee_lst", "climate_water_balance"],
            "VegPhenology": ["vegphenology", "veg_phenology", "phenology", "物候", "植被物候", "ndvi", "evi", "npp", "modis", "cov_gee_ndvi", "cov_gee_evi", "cov_gee_npp"],
        }

    def _default_covariate_names(self) -> list[str]:
        raw = os.getenv(
            "PRO_DEFAULT_LOCAL_COVARIATES",
            "BD,CEC,DEM,gravel,pH,porosity,sand,silt,clay,CLCD,BareSoil,ChinaCP,Management,SeasonalClimateWater,VegPhenology",
        )
        names = [x.strip() for x in re.split(r"[;,，、\s]+", raw) if x.strip()]
        catalog = self._default_covariate_catalog()
        clean: list[str] = []
        for n in names:
            hit = None
            for k in catalog:
                if n.lower() == k.lower():
                    hit = k
                    break
            clean.append(hit or n)
        return list(dict.fromkeys(clean))

    def _parse_covariate_selection_from_text(self, text: str) -> dict[str, Any]:
        text = text or ""
        text_l = text.lower()
        catalog = self._default_covariate_catalog()
        default_names = self._default_covariate_names()
        if any(k in text for k in ["全部协变量", "所有协变量", "全部环境协变量", "所有环境协变量", "全量协变量"]):
            selected_all = [k for k in catalog.keys() if not (self._lulccd_disabled() and k == "LULCcd")]
            return {"mode": "all", "source": "request_text", "selected_names": selected_all, "default_names": default_names}
        matched: list[str] = []
        excluded: list[str] = []
        for name, aliases in catalog.items():
            alias_hit = False
            for a in [name] + aliases:
                if not a:
                    continue
                a_l = str(a).lower()
                if a in text or a_l in text_l:
                    alias_hit = True
                    break
            if alias_hit:
                if any(x in text for x in [f"不要{name}", f"不用{name}", f"排除{name}"]):
                    excluded.append(name)
                else:
                    matched.append(name)
        if matched:
            selected = [x for x in list(dict.fromkeys(matched)) if x not in set(excluded)]
            return {"mode": "explicit", "source": "request_text", "selected_names": selected, "excluded_names": excluded, "default_names": default_names}
        if any(k in text for k in ["默认协变量", "默认环境协变量", "默认环境变量", "默认变量", "图中的环境协变量", "内置环境协变量"]):
            return {"mode": "default", "source": "request_text_default", "selected_names": default_names, "default_names": default_names}
        return {"mode": "default", "source": "unspecified_default", "selected_names": default_names, "default_names": default_names, "message": "用户未指定协变量，已采用默认环境协变量清单。"}

    def _covariate_name_matches_text(self, canonical: str, text: str) -> bool:
        text_l = str(text or "").lower()
        catalog = self._default_covariate_catalog()
        aliases = [canonical] + catalog.get(canonical, [])
        for a in aliases:
            a_l = str(a).lower()
            if a_l and (a_l in text_l):
                return True
        return False

    def _local_covariate_file_allowed(self, path: Path) -> bool:
        if self._lulccd_disabled() and self._is_lulccd_name(path.name):
            return False
        # V164: first-run model priority. Use all user/uploaded/local raster covariates
        # by default; only explicit user selection should narrow them. This prevents
        # variables such as prec/rhum/LSWI/guangai/NIGHT from being dropped merely
        # because they are not in the older default catalog.
        use_all_uploaded = (os.getenv("PRO_USE_ALL_UPLOADED_COVARIATES", "1") or "1").strip().lower() not in {"0", "false", "no", "off"}
        plan = getattr(self, "_covariate_selection_plan", None) or {"mode": "default", "selected_names": self._default_covariate_names()}
        mode = str(plan.get("mode") or "default").lower()
        if mode == "none":
            return False
        if mode == "all" or (use_all_uploaded and mode in {"default", "unspecified_default"}):
            return True
        selected = plan.get("selected_names") or self._default_covariate_names()
        stem = str(path.stem)
        if not selected:
            return True
        return any(self._covariate_name_matches_text(str(name), stem) for name in selected)

    def _feature_col_allowed_by_covariate_selection(self, col: str) -> bool:
        if col in GEE_OUTPUT_MASK_COLUMNS:
            return False
        if self._lulccd_disabled() and self._is_lulccd_name(col):
            return False
        use_all_uploaded = (os.getenv("PRO_USE_ALL_UPLOADED_COVARIATES", "1") or "1").strip().lower() not in {"0", "false", "no", "off"}
        plan = getattr(self, "_covariate_selection_plan", None) or {"mode": "default", "selected_names": self._default_covariate_names()}
        mode = str(plan.get("mode") or "default").lower()
        if mode == "all" or (use_all_uploaded and mode in {"default", "unspecified_default"}):
            return True
        if mode == "none":
            return False
        selected = plan.get("selected_names") or self._default_covariate_names()
        col_s = str(col)
        if not selected:
            return True
        return any(self._covariate_name_matches_text(str(name), col_s) for name in selected)

    def _filter_feature_cols_by_covariate_selection(self, cols: list[str]) -> list[str]:
        cols = [c for c in cols if c not in GEE_OUTPUT_MASK_COLUMNS]
        filtered = [c for c in cols if self._feature_col_allowed_by_covariate_selection(c)]
        if not filtered and cols:
            policy = (os.getenv("PRO_COVARIATE_SELECTION_EMPTY_POLICY", "warn_keep_all") or "warn_keep_all").strip().lower()
            msg = "当前协变量选择方案没有匹配到任何可用特征列。"
            if policy in {"stop", "strict_stop"}:
                raise RuntimeError(msg + " 请改用默认协变量，或检查本地协变量文件名/字段名。")
            self.warnings.append(msg + " 已按策略保留全部真实栅格协变量，避免模型无特征。")
            return cols
        return filtered

    def _target_matches_sample_region(self, target: dict[str, Any], inferred_region: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        """Return whether requested AOI is compatible with uploaded sample distribution.

        Important correction: a county/district AOI such as 温江区 is compatible with samples
        inferred as 成都市 when the local admin table has parent 市名=成都市. It is not compatible
        with samples inferred as 成都市 when the requested AOI parent city is 鄂州市 or another city.
        """
        target = target or {}
        inferred_region = inferred_region or {}
        req_region = str(target.get("region") or "").strip()
        req_city = str(target.get("city") or "").strip()
        req_prov = str(target.get("province") or "").strip()
        inf_region = str(inferred_region.get("region") or "").strip()
        inf_city = str(inferred_region.get("city") or "").strip()
        inf_prov = str(inferred_region.get("province") or "").strip()
        if not req_region or not inf_region:
            return True, {}

        def _same(a: str, b: str) -> bool:
            if not a or not b:
                return False
            aa = {a, self._strip_admin_suffix_local(a)}
            bb = {b, self._strip_admin_suffix_local(b)}
            aa = {x for x in aa if x}; bb = {x for x in bb if x}
            return bool(aa & bb) or a in b or b in a

        if _same(req_region, inf_region) or _same(req_region, inf_city) or _same(req_region, inf_prov):
            return True, {}
        # County/district AOI under the same city/province is compatible. Some county
        # shapefiles only contain 县名 and no parent city. In that case, use the
        # region_text_match detected from the same user request, e.g. "成都市温江区".
        ctx = target.get("region_text_match") if isinstance(target.get("region_text_match"), dict) else {}
        ctx_city = str(ctx.get("city") or (ctx.get("region") if ctx.get("level") == "city" else "") or "").strip()
        ctx_prov = str(ctx.get("province") or (ctx.get("region") if ctx.get("level") == "province" else "") or "").strip()
        if req_city and ( _same(req_city, inf_city) or _same(req_city, inf_region) ):
            return True, {}
        if ctx_city and (_same(ctx_city, inf_city) or _same(ctx_city, inf_region)):
            return True, {}
        if req_prov and inf_prov and _same(req_prov, inf_prov) and not req_city and not inf_city:
            return True, {}
        if ctx_prov and inf_prov and _same(ctx_prov, inf_prov):
            return True, {}
        # Strong mismatch: both city or province are known and different.
        if req_city and (inf_city or inf_region):
            inf_city_like = inf_city or inf_region
            if not _same(req_city, inf_city_like):
                return False, {
                    "target_region": req_region,
                    "target_city": req_city,
                    "target_province": req_prov,
                    "sample_region": inf_region,
                    "sample_city": inf_city,
                    "sample_province": inf_prov,
                    "message": f"样点数据主要落在 {inf_city_like}，但制图区域属于 {req_city}（{req_region}）。样点与制图区域不匹配，结果属于跨区外推。",
                }
        if req_prov and inf_prov and not _same(req_prov, inf_prov):
            return False, {
                "target_region": req_region,
                "target_city": req_city,
                "target_province": req_prov,
                "sample_region": inf_region,
                "sample_city": inf_city,
                "sample_province": inf_prov,
                "message": f"样点数据主要落在 {inf_prov}，但制图区域属于 {req_prov}（{req_region}）。样点与制图区域不匹配，结果不能作为正式交付图。",
            }
        # Fallback to old conservative rule only when no parent context exists.
        old_conflict = region_conflict(req_region, inferred_region)
        if old_conflict and not (req_city or req_prov):
            return False, {
                "target_region": req_region,
                "sample_region": inf_region,
                "sample_city": inf_city,
                "sample_province": inf_prov,
                "message": f"样点数据主要落在 {inf_region}，但请求制图区域为 {req_region}，二者没有可确认的父级行政兼容关系。",
            }
        return True, {}

    def _sample_aoi_overlap(self, samples: Any, target: dict[str, Any]) -> dict[str, Any]:
        """Count uploaded sample points inside the requested local admin AOI."""
        out: dict[str, Any] = {"ok": False, "method": "local_admin_point_overlay"}
        if pd is None or samples is None or len(samples) == 0:
            out["message"] = "样点为空，无法统计目标区域内样点数。"
            return out
        if not {"lon", "lat"}.issubset(set(samples.columns)):
            out["message"] = "样点缺少 lon/lat 字段，无法统计目标区域内样点数。"
            return out
        try:
            import geopandas as gpd
        except Exception as exc:
            out.update({"message": f"geopandas不可用，无法统计目标区域内样点数：{exc}", "error": str(exc)})
            return out
        try:
            geom, meta = self._load_local_admin_geometry(target, target_crs="EPSG:4326")
            out["admin_meta"] = meta
            if geom is None or not meta.get("ok"):
                out["message"] = meta.get("message") or "未匹配到本地行政区几何，无法统计目标区域内样点数。"
                return out
            work = samples[["lon", "lat"]].copy()
            work["lon"] = pd.to_numeric(work["lon"], errors="coerce")
            work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
            work = work.dropna(subset=["lon", "lat"])
            if work.empty:
                out["message"] = "样点坐标无有效数值，无法统计目标区域内样点数。"
                return out
            pts = gpd.GeoDataFrame(work, geometry=gpd.points_from_xy(work["lon"], work["lat"]), crs="EPSG:4326")
            mask = pts.geometry.within(geom) | pts.geometry.touches(geom)
            inside = int(mask.sum())
            total = int(len(pts))
            ratio = inside / max(total, 1)
            out.update({
                "ok": True,
                "region": target.get("region"),
                "target_city": target.get("city"),
                "target_province": target.get("province"),
                "inside_count": inside,
                "total_count": total,
                "inside_ratio": float(ratio),
                "outside_count": int(total - inside),
                "message": f"目标制图区 {target.get('region')} 内样点数：{inside}/{total}，占比 {ratio:.3f}。",
            })
            return out
        except Exception as exc:
            out.update({"message": f"目标区域内样点数统计失败：{exc}", "error": str(exc)})
            return out

    def _refine_target_by_local_admin_text(self, request_text: str, target: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Use local admin attribute names to detect smaller requested AOI from text.

        This solves the case where sample points cover a large city/province, while the user asks
        to draw only a district/county, e.g. "用成都市样点绘制温江区土壤有机质图".
        """
        text = request_text or ""
        if os.getenv("PRO_USE_LOCAL_ADMIN_VECTOR", "1") != "1" or not text.strip():
            return target, None
        try:
            import geopandas as gpd
        except Exception as exc:
            return target, {"ok": False, "method": "local_admin_text_match", "message": f"geopandas不可用，跳过本地行政区文本识别：{exc}"}
        candidates: list[dict[str, Any]] = []
        level_priority = {"county": 3, "city": 2, "province": 1}
        for level in ["county", "city", "province"]:
            for vec_path in self._candidate_local_admin_paths(level):
                try:
                    if not vec_path.exists():
                        continue
                    gdf = gpd.read_file(vec_path)
                    if gdf.empty:
                        continue
                    fields = self._local_admin_name_fields(list(gdf.columns), level)
                    if not fields:
                        continue
                    parent_city_fields = self._local_admin_name_fields(list(gdf.columns), "city")
                    parent_prov_fields = self._local_admin_name_fields(list(gdf.columns), "province")
                    for field in fields:
                        vals = gdf[field].dropna().astype(str).str.strip().drop_duplicates().tolist()
                        for raw in vals:
                            if not raw or raw in {"全国", "中国"}:
                                continue
                            short = self._strip_admin_suffix_local(raw)
                            aliases = [raw]
                            if short and short != raw and len(short) >= 2:
                                aliases.append(short)
                            hit_alias = None
                            for a in aliases:
                                if a and a in text:
                                    hit_alias = a
                                    break
                            if not hit_alias:
                                continue
                            row = gdf[gdf[field].astype(str).str.strip() == raw].head(1)
                            city_val = None
                            prov_val = None
                            if not row.empty:
                                for cf in parent_city_fields:
                                    v = str(row.iloc[0].get(cf, "") or "").strip()
                                    if v:
                                        city_val = v; break
                                for pf in parent_prov_fields:
                                    v = str(row.iloc[0].get(pf, "") or "").strip()
                                    if v:
                                        prov_val = v; break
                            candidates.append({
                                "level": level,
                                "region": raw,
                                "matched_alias": hit_alias,
                                "field": field,
                                "path": str(vec_path),
                                "city": city_val,
                                "province": prov_val,
                                "score": level_priority.get(level, 0) * 1000 + len(str(hit_alias)),
                            })
                except Exception:
                    continue
        if not candidates:
            return target, None
        candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
        best = candidates[0]
        new_target = dict(target or {})
        new_target = apply_region_to_target(new_target, str(best["region"]), level=str(best["level"]), source="local_admin_vector_text_match", inference={"method": "local_admin_text_match", **best})
        if best.get("level") == "county":
            new_target["county"] = best.get("region")
            new_target["district"] = best.get("region")
        if best.get("city"):
            new_target["city"] = best.get("city")
        if best.get("province"):
            new_target["province"] = best.get("province")
        note = {
            "ok": True,
            "method": "local_admin_text_match",
            "message": f"已从本地行政区划属性表识别到更具体制图区域：{best.get('region')}（{best.get('level')}）。",
            "selected": best,
            "candidate_count": len(candidates),
        }
        return new_target, note

    def _strip_admin_suffix_local(self, name: str) -> str:
        s = str(name or "").strip()
        for suf in ["特别行政区", "壮族自治区", "维吾尔自治区", "回族自治区", "自治区", "地区", "自治州", "盟", "省", "市", "县", "区", "旗"]:
            if s.endswith(suf) and len(s) > len(suf):
                s = s[: -len(suf)]
                break
        return s

    def _local_admin_vector_root(self) -> str:
        """Return the local admin vector root/file path.

        V93: supports the user's built-in directory, e.g.
        E:/Agent_DSM/xingzhengquhua_shp. The directory may contain city/county/province
        shapefiles. GEE cannot read a local shapefile directly, so we use it on the Python
        side for prediction bounds, raster mask, and WebGIS display control.
        """
        return (
            os.getenv("PRO_LOCAL_ADMIN_VECTOR_DIR")
            or os.getenv("PRO_ADMIN_VECTOR_DIR")
            or os.getenv("PRO_XZQH_VECTOR_DIR")
            or os.getenv("PRO_ADMIN_CITY_VECTOR")
            or os.getenv("PRO_ADMIN_COUNTY_VECTOR")
            or os.getenv("PRO_ADMIN_PROVINCE_VECTOR")
            or ""
        ).strip()

    def _candidate_local_admin_paths(self, level: str) -> list[Path]:
        roots: list[str] = []
        if level == "county":
            roots += [os.getenv("PRO_LOCAL_ADMIN_COUNTY_VECTOR", ""), os.getenv("PRO_ADMIN_COUNTY_VECTOR", "")]
        elif level == "city":
            roots += [os.getenv("PRO_LOCAL_ADMIN_CITY_VECTOR", ""), os.getenv("PRO_ADMIN_CITY_VECTOR", "")]
        else:
            roots += [os.getenv("PRO_LOCAL_ADMIN_PROVINCE_VECTOR", ""), os.getenv("PRO_ADMIN_PROVINCE_VECTOR", "")]
        roots.append(self._local_admin_vector_root())
        out: list[Path] = []
        seen: set[str] = set()
        level_keywords = {
            "county": ["县", "区", "county", "xian", "district"],
            "city": ["市", "city", "shi", "地级"],
            "province": ["省", "province", "sheng"],
        }.get(level, [])
        for raw in roots:
            if not raw:
                continue
            # allow users to separate several files/folders by ; or ,
            for part in re.split(r"[;,\n]+", raw):
                s = part.strip().strip('"').strip("'")
                if not s:
                    continue
                p = Path(s)
                if p.is_file() and p.suffix.lower() in {".shp", ".geojson", ".gpkg", ".json"}:
                    key = str(p.resolve()) if p.exists() else str(p)
                    if key not in seen:
                        out.append(p); seen.add(key)
                elif p.is_dir():
                    # V95: the user's directory normally contains four shapefile groups:
                    # 县.shp, 市.shp, 省.shp and 十段线.shp. For AOI matching we must use
                    # the polygon administrative layers, not the line layer. Prefer exact
                    # Chinese filenames before fuzzy keyword matching.
                    exclude_default = {"十段线", "九段线", "南海诸岛", "界线", "boundary_line"}
                    exclude_extra = {x.strip() for x in re.split(r"[;,\n]+", os.getenv("PRO_LOCAL_ADMIN_EXCLUDE_STEMS", "")) if x.strip()}
                    exclude_stems = exclude_default | exclude_extra
                    shp_files = [x for x in sorted(p.rglob("*.shp")) if x.stem not in exclude_stems]
                    exact_names = {
                        "county": ["县.shp", "区县.shp", "县级.shp", "county.shp", "district.shp"],
                        "city": ["市.shp", "市级.shp", "地级市.shp", "city.shp"],
                        "province": ["省.shp", "省级.shp", "province.shp"],
                    }.get(level, [])
                    exact = [x for x in shp_files if x.name in exact_names]
                    preferred = [x for x in shp_files if x not in exact and any(k.lower() in x.stem.lower() for k in level_keywords)]
                    ordered = exact + preferred + [x for x in shp_files if x not in exact and x not in preferred]
                    for x in ordered:
                        key = str(x.resolve()) if x.exists() else str(x)
                        if key not in seen:
                            out.append(x); seen.add(key)
        return out

    def _local_admin_match_values(self, target: dict[str, Any], level: str) -> list[str]:
        target = target or {}
        vals: list[str] = []
        if level == "county":
            vals += [target.get("county"), target.get("district"), target.get("region"), target.get("city")]
        elif level == "city":
            vals += [target.get("city"), target.get("region")]
        else:
            vals += [target.get("province"), target.get("region")]
        # User override, useful when the shapefile stores aliases or non-standard names.
        vals += re.split(r"[;,\n]+", os.getenv("PRO_LOCAL_ADMIN_NAME_VALUE", ""))
        clean = []
        for v in vals:
            if v is None:
                continue
            s = str(v).strip()
            if not s or s in {"全国", "中国", "样点包络范围"}:
                continue
            clean.append(s)
            short = self._strip_admin_suffix_local(s)
            if short and short != s:
                clean.append(short)
        return list(dict.fromkeys(clean))

    def _local_admin_name_fields(self, columns: list[str], level: str) -> list[str]:
        env_key = {
            "county": "PRO_LOCAL_ADMIN_COUNTY_NAME_FIELD",
            "city": "PRO_LOCAL_ADMIN_CITY_NAME_FIELD",
            "province": "PRO_LOCAL_ADMIN_PROVINCE_NAME_FIELD",
        }.get(level, "PRO_LOCAL_ADMIN_NAME_FIELD")
        env_fields = [x.strip() for x in re.split(r"[;,\n]+", os.getenv(env_key, "") + ";" + os.getenv("PRO_LOCAL_ADMIN_NAME_FIELD", "")) if x.strip()]
        defaults = {
            "county": ["县名", "区名", "县", "区", "NAME", "name", "Name", "county", "district", "县名称", "区县名"],
            "city": ["市名", "市", "地市", "地级市", "NAME", "name", "Name", "city", "市名称"],
            "province": ["省名", "省", "NAME", "name", "Name", "province", "省名称"],
        }.get(level, [])
        fields = env_fields + defaults
        return [f for f in fields if f in columns]

    def _load_local_admin_geometry(self, target: dict[str, Any], prefer_level: str | None = None, target_crs: str = "EPSG:4326") -> tuple[Any | None, dict[str, Any]]:
        """Load and match local admin boundary geometry.

        Returns a unary geometry in target_crs plus JSON-safe metadata. Does not raise unless
        PRO_LOCAL_ADMIN_REQUIRE=1, because formal mapping should still be able to fall back to
        GEE/bbox when the local directory is absent on another machine.
        """
        meta: dict[str, Any] = {
            "enabled": os.getenv("PRO_USE_LOCAL_ADMIN_VECTOR", "1") == "1",
            "root": self._local_admin_vector_root(),
            "target_crs": target_crs,
            "ok": False,
        }
        if meta["enabled"] is False:
            meta["message"] = "PRO_USE_LOCAL_ADMIN_VECTOR=0，未启用本地行政区划矢量。"
            return None, meta
        if not meta["root"]:
            meta["message"] = "未配置 PRO_LOCAL_ADMIN_VECTOR_DIR / PRO_ADMIN_VECTOR_DIR。"
            return None, meta
        try:
            import geopandas as gpd
        except Exception as exc:  # pragma: no cover
            meta.update({"message": f"geopandas不可用，无法读取本地行政区划：{exc}", "error": str(exc)})
            if os.getenv("PRO_LOCAL_ADMIN_REQUIRE", "0") == "1":
                raise RuntimeError(meta["message"])
            return None, meta

        target = target or {}
        region = str(target.get("region") or "")
        levels: list[str] = []
        if prefer_level:
            levels.append(prefer_level)
        if str(target.get("county") or target.get("district") or "") or any(region.endswith(x) for x in ["县", "区", "旗"]):
            levels += ["county", "city", "province"]
        elif bool(target.get("is_city")) or str(target.get("city") or "") or region.endswith("市"):
            levels += ["city", "county", "province"]
        else:
            levels += ["province", "city", "county"]
        levels = list(dict.fromkeys(levels))
        attempts: list[dict[str, Any]] = []

        for level in levels:
            values = self._local_admin_match_values(target, level)
            paths = self._candidate_local_admin_paths(level)
            for vec_path in paths:
                att: dict[str, Any] = {"level": level, "path": str(vec_path), "values": values}
                try:
                    if not vec_path.exists():
                        att["error"] = "path_not_exists"; attempts.append(att); continue
                    gdf = gpd.read_file(vec_path)
                    if gdf.empty:
                        att["error"] = "empty_vector"; attempts.append(att); continue
                    if gdf.crs is None:
                        assumed = os.getenv("PRO_LOCAL_ADMIN_VECTOR_CRS", "EPSG:4326")
                        gdf = gdf.set_crs(assumed, allow_override=True)
                        att["assumed_crs"] = assumed
                    name_fields = self._local_admin_name_fields(list(gdf.columns), level)
                    att["columns"] = [str(c) for c in list(gdf.columns)]
                    att["name_fields"] = name_fields
                    if not name_fields:
                        att["error"] = "no_name_field"; attempts.append(att); continue
                    sub = None
                    matched_field = None
                    matched_value = None
                    value_norm = {self._strip_admin_suffix_local(v): v for v in values}
                    value_set = set(values) | set(value_norm.keys())
                    for field in name_fields:
                        ser = gdf[field].astype(str).str.strip()
                        exact = ser.isin(value_set)
                        stripped = ser.map(self._strip_admin_suffix_local).isin(value_set)
                        mask = exact | stripped
                        if mask.any():
                            sub = gdf.loc[mask].copy()
                            matched_field = field
                            # Store a small sample of matched raw names.
                            matched_value = ",".join(list(dict.fromkeys(ser[mask].astype(str).head(5).tolist())))
                            break
                    if sub is None or sub.empty:
                        att["error"] = "no_matched_feature"; attempts.append(att); continue

                    # V95: when the target carries parent administrative context, use the
                    # parent fields in the user's official county/city layers to avoid merging
                    # same-name districts/counties from other cities or provinces.
                    parent_filters: dict[str, Any] = {}
                    parent_mismatch = False
                    def _filter_parent_level(_sub, _parent_level: str, _target_value: str):
                        _target_value = str(_target_value or "").strip()
                        if not _target_value:
                            return _sub, None, False
                        _fields = self._local_admin_name_fields(list(_sub.columns), _parent_level)
                        _value_set = {_target_value, self._strip_admin_suffix_local(_target_value)}
                        _value_set = {v for v in _value_set if v}
                        for _field in _fields:
                            _ser = _sub[_field].astype(str).str.strip()
                            _mask = _ser.isin(_value_set) | _ser.map(self._strip_admin_suffix_local).isin(_value_set)
                            if _mask.any():
                                return _sub.loc[_mask].copy(), {"field": _field, "value": _target_value, "matched_count": int(_mask.sum())}, False
                            return _sub.iloc[0:0].copy(), {"field": _field, "value": _target_value, "matched_count": 0}, True
                        return _sub, None, False

                    if level == "county":
                        sub, info, bad = _filter_parent_level(sub, "city", target.get("city"))
                        if info: parent_filters["city"] = info
                        parent_mismatch = parent_mismatch or bad
                    if level in {"county", "city"}:
                        sub, info, bad = _filter_parent_level(sub, "province", target.get("province"))
                        if info: parent_filters["province"] = info
                        parent_mismatch = parent_mismatch or bad
                    if parent_filters:
                        att["parent_filters"] = parent_filters
                    if parent_mismatch or sub.empty:
                        att["error"] = "parent_admin_mismatch"; attempts.append(att); continue

                    sub = sub.to_crs(target_crs)
                    geom = sub.geometry.unary_union
                    if geom is None or geom.is_empty:
                        att["error"] = "empty_geometry_after_match"; attempts.append(att); continue
                    bounds = tuple(float(x) for x in geom.bounds)
                    meta.update({
                        "ok": True,
                        "source": "local_admin_vector",
                        "level": level,
                        "path": str(vec_path),
                        "matched_field": matched_field,
                        "matched_value": matched_value,
                        "matched_count": int(len(sub)),
                        "parent_filters": parent_filters,
                        "bounds": {"minx": bounds[0], "miny": bounds[1], "maxx": bounds[2], "maxy": bounds[3]},
                        "message": f"已使用本地行政区划矢量匹配 {len(sub)} 个要素：{matched_value or ''}",
                        "attempts_tail": attempts[-8:],
                    })
                    return geom, meta
                except Exception as exc:
                    att["error"] = str(exc)
                    attempts.append(att)
                    continue
        meta.update({"attempts_tail": attempts[-12:], "message": "本地行政区划矢量未匹配到目标区域。"})
        if os.getenv("PRO_LOCAL_ADMIN_REQUIRE", "0") == "1":
            raise RuntimeError(meta["message"] + " 请检查字段名、市名/县名、省名和目标区域解析。")
        return None, meta

    def _local_admin_bounds(self, target: dict[str, Any]) -> tuple[dict[str, float] | None, dict[str, Any]]:
        geom, meta = self._load_local_admin_geometry(target, target_crs="EPSG:4326")
        if geom is None or not meta.get("ok"):
            return None, meta
        minx, miny, maxx, maxy = [float(v) for v in geom.bounds]
        pad = float(os.getenv("PRO_LOCAL_ADMIN_BOUNDS_PAD_DEG", "0.002"))
        return {
            "lon_min": minx - pad,
            "lon_max": maxx + pad,
            "lat_min": miny - pad,
            "lat_max": maxy + pad,
        }, meta

    def _local_admin_mask_array(self, target: dict[str, Any], width: int, height: int, transform: Any, crs: str):
        """Rasterize matched local admin boundary on the output grid."""
        if rasterize is None:
            return None, {"ok": False, "message": "rasterio.features.rasterize不可用，无法栅格化本地行政边界。"}
        geom, meta = self._load_local_admin_geometry(target, target_crs=str(crs or "EPSG:4326"))
        if geom is None or not meta.get("ok"):
            return None, meta
        try:
            all_touched = os.getenv("PRO_LOCAL_ADMIN_MASK_ALL_TOUCHED", "1") == "1"
            mask = rasterize(
                [(geom, 1)],
                out_shape=(int(height), int(width)),
                transform=transform,
                fill=0,
                default_value=1,
                dtype="uint8",
                all_touched=all_touched,
            )
            meta.update({"rasterized": True, "all_touched": all_touched, "valid_cells": int(mask.sum())})
            return mask.astype("uint8"), meta
        except Exception as exc:
            meta.update({"rasterized": False, "error": str(exc), "message": f"本地行政边界栅格化失败：{exc}"})
            if os.getenv("PRO_LOCAL_ADMIN_REQUIRE", "0") == "1":
                raise RuntimeError(meta["message"])
            return None, meta

    def _fallback_gee_prediction_bounds(self, df: Any, target: dict[str, Any]) -> tuple[dict[str, float], str]:
        """Resolve a rectangular fallback prediction extent in EPSG:4326.

        This is only an extent. V77 formal clipping is controlled by cov_gee_admin_mask and
        cov_gee_cropland_mask returned from GEE.
        """
        mode = (os.getenv("PRO_GEE_PRED_BOUNDS_MODE", "auto") or "auto").strip().lower()
        region = str(target.get("region") or "")
        is_city = bool(target.get("is_city"))
        if mode in {"auto", "region_bbox", "region"}:
            bbox = None
            if region in CITY_BBOXES and (mode != "auto" or is_city):
                bbox = CITY_BBOXES[region]["bbox"]
            elif region in PROVINCE_BBOXES and mode in {"region_bbox", "region"}:
                bbox = PROVINCE_BBOXES[region]
            if bbox:
                lon_min, lat_min, lon_max, lat_max = map(float, bbox)
                pad = float(os.getenv("PRO_GEE_REGION_BBOX_PAD_DEG", "0.005"))
                return {
                    "lon_min": lon_min - pad,
                    "lon_max": lon_max + pad,
                    "lat_min": lat_min - pad,
                    "lat_max": lat_max + pad,
                }, f"builtin_region_bbox:{region}"

        lon_min, lon_max = float(df["lon"].min()), float(df["lon"].max())
        lat_min, lat_max = float(df["lat"].min()), float(df["lat"].max())
        lon_pad = max((lon_max - lon_min) * float(os.getenv("PRO_GEE_PRED_GRID_PAD_RATIO", "0.12")), float(os.getenv("PRO_GEE_PRED_GRID_MIN_PAD_DEG", "0.02")))
        lat_pad = max((lat_max - lat_min) * float(os.getenv("PRO_GEE_PRED_GRID_PAD_RATIO", "0.12")), float(os.getenv("PRO_GEE_PRED_GRID_MIN_PAD_DEG", "0.02")))
        return {
            "lon_min": lon_min - lon_pad,
            "lon_max": lon_max + lon_pad,
            "lat_min": lat_min - lat_pad,
            "lat_max": lat_max + lat_pad,
        }, "sample_bbox_with_buffer"

    def _resolve_gee_prediction_bounds(self, df: Any, target: dict[str, Any]) -> tuple[dict[str, float], str, dict[str, Any]]:
        """Resolve prediction bounds for GEE formal grid.

        V77 priority:
        1. If PRO_GEE_USE_ADMIN_BOUNDARY=1, try to resolve an administrative geometry in GEE and
           use its bounding rectangle as the prediction grid extent.
        2. If admin resolution fails and PRO_GEE_ADMIN_REQUIRE=0, fall back to built-in bbox or
           sample envelope; the output report will explicitly mark admin_boundary_applied=false.
        """
        fallback_bounds, fallback_source = self._fallback_gee_prediction_bounds(df, target)
        admin_meta: dict[str, Any] = {"enabled": os.getenv("PRO_GEE_USE_ADMIN_BOUNDARY", "1") == "1", "ok": False}
        # V93: prefer the user's local official administrative division vectors for the
        # prediction grid extent. GEE cannot directly read E:\... shapefiles, so this
        # Python-side bound resolution is intentionally placed before GAUL/custom GEE asset.
        if os.getenv("PRO_USE_LOCAL_ADMIN_VECTOR", "1") == "1" and os.getenv("PRO_LOCAL_ADMIN_BOUNDS_PRIORITY", "1") == "1":
            try:
                local_bounds, local_meta = self._local_admin_bounds(target)
                if local_bounds and local_meta.get("ok"):
                    _emit("GEE", "已使用本地行政区划矢量bounds构建预测网格", {"bounds": local_bounds, "admin_meta": local_meta}, task_id=self.task_id)
                    return local_bounds, "local_admin_vector:" + str(local_meta.get("level") or "unknown"), local_meta
                if local_meta.get("message"):
                    self.warnings.append("本地行政区划bounds未启用：" + str(local_meta.get("message")))
            except Exception as exc:
                msg = f"本地行政区划bounds解析失败，转入GEE/备用范围：{exc}"
                if os.getenv("PRO_LOCAL_ADMIN_REQUIRE", "0") == "1":
                    raise RuntimeError(msg)
                self.warnings.append(msg)
        if os.getenv("PRO_GEE_USE_ADMIN_BOUNDARY", "1") == "1":
            try:
                from services.cloud_sampler.gee_sampler import GeeSampler
                sampler = GeeSampler(
                    project_id=self.gee_project,
                    log_fn=lambda stage, msg, payload=None: _emit(stage, msg, payload, task_id=self.task_id),
                )
                sampler.initialize()
                bounds, source, admin_meta = sampler.resolve_target_bounds(target, fallback_bounds)
                if str(source).startswith("admin_boundary"):
                    _emit("GEE", "已解析行政边界并使用其bounds构建预测网格", {"bounds": bounds, "source": source, "admin_meta": admin_meta}, task_id=self.task_id)
                    return bounds, source, admin_meta
                self.warnings.append(f"行政边界bounds未启用，使用备用范围：{fallback_source}；原因：{admin_meta.get('message')}")
            except Exception as exc:
                msg = f"行政边界bounds解析失败，使用备用范围：{fallback_source}；错误：{exc}"
                if os.getenv("PRO_GEE_ADMIN_REQUIRE", "0") == "1":
                    raise RuntimeError(msg)
                self.warnings.append(msg)
                admin_meta = {"enabled": True, "ok": False, "error": str(exc), "message": msg}
        return fallback_bounds, fallback_source, admin_meta

    def _grid_mask_array(self, grid_df: Any, mask_col: str, width: int, height: int, threshold: float = 0.5):
        if np is None or pd is None:
            raise RuntimeError("缺少 numpy/pandas，不能构建mask数组。")
        if mask_col not in grid_df.columns:
            raise KeyError(mask_col)
        mask_arr = np.zeros((height, width), dtype="uint8")
        if "row" in grid_df.columns and "col" in grid_df.columns:
            rows = pd.to_numeric(grid_df["row"], errors="coerce").astype("Int64")
            cols = pd.to_numeric(grid_df["col"], errors="coerce").astype("Int64")
            vals = pd.to_numeric(grid_df[mask_col], errors="coerce").fillna(0).to_numpy()
            for mv, rr, cc in zip(vals, rows, cols):
                if pd.isna(rr) or pd.isna(cc):
                    continue
                r, c = int(rr), int(cc)
                if 0 <= r < height and 0 <= c < width:
                    mask_arr[r, c] = 1 if float(mv) >= threshold else 0
        else:
            vals = pd.to_numeric(grid_df[mask_col], errors="coerce").fillna(0).to_numpy()
            if len(vals) != width * height:
                raise RuntimeError(f"mask列 {mask_col} 行数 {len(vals)} 无法匹配 {height}x{width} 网格。")
            mask_arr = (vals.reshape((height, width)) >= threshold).astype("uint8")
        return mask_arr

    def _write_mask_tif(self, out_path: Path, mask_arr: Any, transform: Any, crs: str = "EPSG:4326") -> str | None:
        if rasterio is None:
            return None
        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=mask_arr.shape[0],
            width=mask_arr.shape[1],
            count=1,
            dtype="uint8",
            crs=crs,
            transform=transform,
            nodata=0,
            compress="lzw",
        ) as dst:
            dst.write(mask_arr.astype("uint8"), 1)
        return str(out_path)


    def _write_float_tif(self, out_path: Path, arr: Any, transform: Any, crs: str = "EPSG:4326", nodata: float = -9999.0) -> str | None:
        if rasterio is None:
            return None
        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=arr.shape[0],
            width=arr.shape[1],
            count=1,
            dtype="float32",
            crs=crs,
            transform=transform,
            nodata=nodata,
            compress="lzw",
        ) as dst:
            dst.write(arr.astype("float32"), 1)
        return str(out_path)


    def _build_projected_meter_grid(self, bounds: dict[str, float], target: dict[str, Any]) -> dict[str, Any]:
        """Build a metric prediction grid for formal GeoTIFF output.

        V82 correction:
        Earlier GEE preview grids were built directly in EPSG:4326 by width/height.
        That made the report say target_crs=EPSG:32648 while the GeoTIFF itself was
        EPSG:4326, and the real pixel size was not the configured meter resolution.
        This helper builds the grid in target_crs meters, then transforms cell centers
        back to lon/lat only for GEE sampling.
        """
        if Transformer is None:
            raise RuntimeError("缺少 pyproj，不能构建投影坐标预测网格。")
        try:
            from rasterio.transform import from_origin
        except Exception as exc:
            raise RuntimeError(f"缺少 rasterio.transform.from_origin，不能写出投影网格：{exc}")

        target_crs = str(target.get("target_crs") or os.getenv("PRO_GEE_OUTPUT_CRS") or os.getenv("PRO_TARGET_CRS") or "EPSG:32648").strip()
        requested_res = float(os.getenv("PRO_GEE_FORMAL_OUTPUT_RESOLUTION_M", str(target.get("resolution_m") or os.getenv("PRO_GEE_SAMPLE_SCALE", "250"))))
        requested_res = max(10.0, requested_res)
        max_points = int(os.getenv("PRO_GEE_GETINFO_MAX_SAMPLES", "120000"))
        coarsen = os.getenv("PRO_GEE_ALLOW_RESOLUTION_COARSEN", "1").strip() != "0"

        tr_to_proj = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        tr_to_ll = Transformer.from_crs(target_crs, "EPSG:4326", always_xy=True)

        corners_ll = [
            (float(bounds["lon_min"]), float(bounds["lat_min"])),
            (float(bounds["lon_min"]), float(bounds["lat_max"])),
            (float(bounds["lon_max"]), float(bounds["lat_min"])),
            (float(bounds["lon_max"]), float(bounds["lat_max"])),
        ]
        xs, ys = zip(*[tr_to_proj.transform(lon, lat) for lon, lat in corners_ll])
        x_min, x_max = float(min(xs)), float(max(xs))
        y_min, y_max = float(min(ys)), float(max(ys))

        # Snap outward to the resolution grid so repeated runs are stable.
        res = requested_res
        x_min = math.floor(x_min / res) * res
        x_max = math.ceil(x_max / res) * res
        y_min = math.floor(y_min / res) * res
        y_max = math.ceil(y_max / res) * res

        def dims_for(r: float) -> tuple[int, int, int]:
            w = int(math.ceil((x_max - x_min) / r))
            h = int(math.ceil((y_max - y_min) / r))
            return max(1, w), max(1, h), max(1, w) * max(1, h)

        width, height, points = dims_for(res)
        if points > max_points:
            if not coarsen:
                raise RuntimeError(
                    f"按 {requested_res:g} m 构建预测网格需要 {points} 个点，超过 PRO_GEE_GETINFO_MAX_SAMPLES={max_points}。"
                    "请增大 PRO_GEE_FORMAL_OUTPUT_RESOLUTION_M，或开启 PRO_GEE_ALLOW_RESOLUTION_COARSEN=1。"
                )
            # Coarsen only as much as needed. Use a square-root scaling so the total points
            # falls below the getInfo safety cap while preserving the AOI extent.
            scale = math.sqrt(points / max(max_points, 1))
            res = math.ceil((requested_res * scale) / 10.0) * 10.0
            width, height, points = dims_for(res)
            while points > max_points:
                res = math.ceil((res * 1.08) / 10.0) * 10.0
                width, height, points = dims_for(res)

        rows = []
        for row in range(height):
            y = y_max - (row + 0.5) * res
            for col in range(width):
                x = x_min + (col + 0.5) * res
                lon, lat = tr_to_ll.transform(x, y)
                pid = row * width + col
                rows.append({
                    "grid_id": int(pid),
                    "row": int(row),
                    "col": int(col),
                    "x": float(x),
                    "y": float(y),
                    "lon": float(lon),
                    "lat": float(lat),
                })

        grid_df = pd.DataFrame(rows)
        return {
            "grid_df": grid_df,
            "width": int(width),
            "height": int(height),
            "point_count": int(points),
            "transform": from_origin(x_min, y_max, res, res),
            "target_crs": target_crs,
            "requested_resolution_m": float(requested_res),
            "output_resolution_m": float(res),
            "resolution_coarsened": bool(res != requested_res),
            "projected_bounds": {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
            "source_bounds_epsg4326": bounds,
        }

    def _make_gee_formal_prediction_tif(self, model: Any, feature_cols: list[str], df: Any, out_dir: Path, target: dict[str, Any]) -> Path | None:
        """Use GEE to sample cov_gee_* features on a prediction grid and write a formal GeoTIFF.

        V77 upgrades over V70/V75:
        - prediction grid extent can come from a resolved administrative boundary;
        - output is clipped to cov_gee_admin_mask when available;
        - output is clipped to cov_gee_cropland_mask when enabled;
        - mask GeoTIFFs and detailed JSON audit are written beside the prediction GeoTIFF.
        """
        if rasterio is None or np is None or pd is None:
            self.warnings.append("缺少 rasterio/numpy/pandas，跳过GEE正式预测图输出。")
            return None
        year = target.get("year")
        if not year:
            self.warnings.append("未识别到年份，跳过GEE正式预测图输出。")
            return None
        try:
            from services.cloud_sampler.gee_sampler import GeeSampler
            from rasterio.transform import from_origin
        except Exception as exc:
            self.warnings.append(f"导入GeeSampler或rasterio.transform失败，跳过GEE正式预测图输出：{exc}")
            return None

        try:
            bounds, bounds_source, admin_bounds_meta = self._resolve_gee_prediction_bounds(df, target)

            grid_mode = os.getenv("PRO_GEE_OUTPUT_GRID_MODE", "projected_meter").strip().lower()
            projected_grid_info = None
            if grid_mode in {"projected", "projected_meter", "meter", "metric"}:
                projected_grid_info = self._build_projected_meter_grid(bounds, target)
                width = int(projected_grid_info["width"])
                height = int(projected_grid_info["height"])
                transform = projected_grid_info["transform"]
                output_crs = str(projected_grid_info["target_crs"])
                sampler_grid_df = projected_grid_info["grid_df"]
                _emit("GEE", "V82 已构建投影坐标预测网格", {
                    "grid_mode": "projected_meter",
                    "output_crs": output_crs,
                    "requested_resolution_m": projected_grid_info["requested_resolution_m"],
                    "output_resolution_m": projected_grid_info["output_resolution_m"],
                    "resolution_coarsened": projected_grid_info["resolution_coarsened"],
                    "width": width,
                    "height": height,
                    "points": int(projected_grid_info["point_count"]),
                    "projected_bounds": projected_grid_info["projected_bounds"],
                }, task_id=self.task_id)
            else:
                width = int(os.getenv("PRO_GEE_PRED_GRID_WIDTH", os.getenv("PRO_PLATFORM_PRED_GRID_SIZE", "70")))
                width = max(20, min(width, int(os.getenv("PRO_GEE_PRED_GRID_MAX_WIDTH", "140"))))
                aspect = max((bounds["lon_max"] - bounds["lon_min"]) / max(bounds["lat_max"] - bounds["lat_min"], 1e-9), 0.2)
                height = max(20, int(round(width / aspect)))
                height = min(height, int(os.getenv("PRO_GEE_PRED_GRID_MAX_HEIGHT", "140")))

                max_points = int(os.getenv("PRO_GEE_GETINFO_MAX_SAMPLES", "5000"))
                while width * height > max_points and width > 20 and height > 20:
                    width = max(20, int(width * 0.9))
                    height = max(20, int(height * 0.9))

                xres = (bounds["lon_max"] - bounds["lon_min"]) / max(width - 1, 1)
                yres = (bounds["lat_max"] - bounds["lat_min"]) / max(height - 1, 1)
                transform = from_origin(bounds["lon_min"] - xres / 2, bounds["lat_max"] + yres / 2, xres, yres)
                output_crs = "EPSG:4326"
                sampler_grid_df = None

            sampler = GeeSampler(
                project_id=self.gee_project,
                log_fn=lambda stage, msg, payload=None: _emit(stage, msg, payload, task_id=self.task_id),
            )
            if sampler_grid_df is not None:
                result = sampler.sample_points_grid_to_dataframe(grid_df=sampler_grid_df, year=int(year), width=width, height=height, target=target)
            else:
                result = sampler.sample_grid_to_dataframe(bounds=bounds, year=int(year), width=width, height=height, target=target)
            if not result.ok or result.dataframe is None or result.dataframe.empty:
                self.warnings.append(f"GEE预测网格协变量抽取失败：{result.message}")
                return None

            grid_df = result.dataframe.copy()
            grid_csv = out_dir / "gee_prediction_grid_covariates.csv"
            grid_df.to_csv(grid_csv, index=False, encoding="utf-8-sig")

            # V91: sample domestic/local standardized rasters on the formal prediction grid.
            # Without this, local covariates would be filled by training medians and lose spatial signal.
            for col, raster_path in getattr(self, "_covariate_raster_map", {}).items():
                if col in feature_cols and col not in grid_df.columns:
                    try:
                        grid_df[col] = self._sample_raster(Path(raster_path), grid_df, col)
                    except Exception as exc:
                        self.warnings.append(f"预测网格抽取本地协变量失败 {col}: {exc}")

            for col in feature_cols:
                if col not in grid_df.columns:
                    med = pd.to_numeric(df[col], errors="coerce").median() if col in df.columns else 0
                    grid_df[col] = 0 if pd.isna(med) else med
                grid_df[col] = pd.to_numeric(grid_df[col], errors="coerce")
                med = pd.to_numeric(df[col], errors="coerce").median() if col in df.columns else grid_df[col].median()
                grid_df[col] = grid_df[col].fillna(0 if pd.isna(med) else med)

            Xg = grid_df[feature_cols].to_numpy(dtype="float64")
            preds = self._predict_model(model, Xg, grid_df, target).astype("float32")

            arr = np.full((height, width), -9999.0, dtype="float32")
            if "row" in grid_df.columns and "col" in grid_df.columns:
                rows = pd.to_numeric(grid_df["row"], errors="coerce").astype("Int64")
                cols = pd.to_numeric(grid_df["col"], errors="coerce").astype("Int64")
                for val, rr, cc in zip(preds, rows, cols):
                    if pd.isna(rr) or pd.isna(cc):
                        continue
                    r, c = int(rr), int(cc)
                    if 0 <= r < height and 0 <= c < width:
                        arr[r, c] = float(val)
            else:
                if len(preds) == width * height:
                    arr = preds.reshape((height, width)).astype("float32")
                else:
                    raise RuntimeError(f"GEE预测网格返回 {len(preds)} 行，无法匹配 {height}x{width} 网格。")

            admin_mask_applied = False
            cropland_mask_applied = False
            mask_outputs: dict[str, str | None] = {}
            mask_stats: dict[str, Any] = {}

            if os.getenv("PRO_GEE_CLIP_TO_ADMIN", "1") == "1":
                local_admin_mask = None
                local_admin_mask_meta: dict[str, Any] = {}
                if os.getenv("PRO_USE_LOCAL_ADMIN_VECTOR", "1") == "1" and os.getenv("PRO_LOCAL_ADMIN_MASK_PRIORITY", "1") == "1":
                    local_admin_mask, local_admin_mask_meta = self._local_admin_mask_array(target, width, height, transform, output_crs)
                if local_admin_mask is not None and int(local_admin_mask.sum()) > 0:
                    arr[local_admin_mask == 0] = -9999.0
                    admin_mask_applied = True
                    mask_outputs["admin_mask_tif"] = self._write_mask_tif(out_dir / "gee_admin_boundary_mask.tif", local_admin_mask, transform, output_crs)
                    mask_outputs["local_admin_mask_tif"] = mask_outputs["admin_mask_tif"]
                    mask_stats["admin_valid_cells"] = int(local_admin_mask.sum())
                    mask_stats["admin_mask_source"] = "local_admin_vector"
                    mask_stats["local_admin_mask_meta"] = local_admin_mask_meta
                elif "cov_gee_admin_mask" in grid_df.columns:
                    admin_mask = self._grid_mask_array(grid_df, "cov_gee_admin_mask", width, height, threshold=0.5)
                    arr[admin_mask == 0] = -9999.0
                    admin_mask_applied = True
                    mask_outputs["admin_mask_tif"] = self._write_mask_tif(out_dir / "gee_admin_boundary_mask.tif", admin_mask, transform, output_crs)
                    mask_stats["admin_valid_cells"] = int(admin_mask.sum())
                    mask_stats["admin_mask_source"] = "gee_admin_mask"
                    if local_admin_mask_meta.get("message"):
                        self.warnings.append("本地行政区划mask未启用，已回退GEE行政mask：" + str(local_admin_mask_meta.get("message")))
                else:
                    msg = "已启用PRO_GEE_CLIP_TO_ADMIN=1，但本地行政区划未匹配且GEE返回表没有cov_gee_admin_mask，预测图未执行行政边界裁剪。"
                    if os.getenv("PRO_GEE_ADMIN_REQUIRE", "0") == "1" or os.getenv("PRO_LOCAL_ADMIN_REQUIRE", "0") == "1":
                        raise RuntimeError(msg)
                    self.warnings.append(msg)

            admin_continuous_tif = None
            if admin_mask_applied and os.getenv("PRO_WRITE_ADMIN_CONTINUOUS_TIF", "1") == "1":
                try:
                    admin_continuous_tif = out_dir / "gee_formal_prediction_admin_continuous.tif"
                    with rasterio.open(
                        admin_continuous_tif,
                        "w",
                        driver="GTiff",
                        height=height,
                        width=width,
                        count=1,
                        dtype="float32",
                        crs=output_crs,
                        transform=transform,
                        nodata=-9999.0,
                        compress="lzw",
                    ) as dst:
                        dst.write(arr.astype("float32"), 1)
                    mask_outputs["admin_continuous_prediction_tif"] = str(admin_continuous_tif)
                except Exception as exc:
                    self.warnings.append(f"行政区连续预测图写出失败：{exc}")

            if os.getenv("PRO_GEE_MASK_TO_CROPLAND", "1") == "1":
                if "cov_gee_cropland_mask" in grid_df.columns:
                    cropland_threshold = float(os.getenv("PRO_GEE_CROPLAND_FRACTION_THRESHOLD", os.getenv("PRO_GEE_CROPLAND_MASK_THRESHOLD", "0.05")))
                    cropland_mask = self._grid_mask_array(grid_df, "cov_gee_cropland_mask", width, height, threshold=cropland_threshold)
                    arr[cropland_mask == 0] = -9999.0
                    cropland_mask_applied = True
                    mask_outputs["cropland_mask_tif"] = self._write_mask_tif(out_dir / "gee_cropland_mask.tif", cropland_mask, transform, output_crs)
                    mask_stats["cropland_valid_cells"] = int(cropland_mask.sum())
                    mask_stats["cropland_fraction_threshold"] = float(cropland_threshold)
                else:
                    msg = "已启用PRO_GEE_MASK_TO_CROPLAND=1，但GEE返回表没有cov_gee_cropland_mask，预测图未执行耕地掩膜。"
                    if os.getenv("PRO_GEE_CROPLAND_REQUIRE", "0") == "1":
                        raise RuntimeError(msg)
                    self.warnings.append(msg)

            final_valid = (arr != -9999.0).astype("uint8")
            mask_outputs["final_valid_mask_tif"] = self._write_mask_tif(out_dir / "gee_final_valid_mask.tif", final_valid, transform, output_crs)
            mask_stats["final_valid_cells"] = int(final_valid.sum())
            mask_stats["final_nodata_cells"] = int((final_valid == 0).sum())

            out_tif = out_dir / "gee_formal_prediction.tif"
            with rasterio.open(
                out_tif,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype="float32",
                crs=output_crs,
                transform=transform,
                nodata=-9999.0,
                compress="lzw",
            ) as dst:
                dst.write(arr, 1)

            # automatically output spatially varying conformal/GCP-style uncertainty
            # on the exact same grid/mask as the SOM prediction. V83 used a constant global
            # half-width; The formal workflow computes a local residual quantile from nearby OOF calibration points.
            gcp_uncertainty: dict[str, Any] | None = None
            if os.getenv("PRO_GCP_AUTO_FROM_GEE", "0") == "1" and self._gee_gcp_calibration:
                try:
                    valid_pred = arr != -9999.0
                    half_width_arr, spatial_meta = self._compute_spatial_gcp_half_widths(grid_df, valid_pred, width, height)
                    lower_arr = np.full_like(arr, -9999.0, dtype="float32")
                    upper_arr = np.full_like(arr, -9999.0, dtype="float32")
                    width_arr = np.full_like(arr, -9999.0, dtype="float32")
                    q_valid = half_width_arr[valid_pred]
                    lower_arr[valid_pred] = arr[valid_pred] - q_valid
                    upper_arr[valid_pred] = arr[valid_pred] + q_valid
                    width_arr[valid_pred] = 2.0 * q_valid
                    lower_tif = self._write_float_tif(out_dir / "gee_gcp_lower.tif", lower_arr, transform, output_crs)
                    upper_tif = self._write_float_tif(out_dir / "gee_gcp_upper.tif", upper_arr, transform, output_crs)
                    width_tif = self._write_float_tif(out_dir / "gee_gcp_width.tif", width_arr, transform, output_crs)
                    q_tif = self._write_float_tif(out_dir / "gee_gcp_half_width_q.tif", half_width_arr, transform, output_crs)
                    gcp_report = dict(self._gee_gcp_calibration.get("metrics", {}))
                    local_width_valid = width_arr[valid_pred]
                    gcp_report.update({
                        "method": "spatial_weighted_kfold_conformal_residual" if spatial_meta.get("spatial_applied") else gcp_report.get("method", "global_kfold_conformal_residual"),
                        "lower_tif": lower_tif,
                        "upper_tif": upper_tif,
                        "width_tif": width_tif,
                        "half_width_q_tif": q_tif,
                        "prediction_tif": str(out_tif),
                        "valid_prediction_cells": int(valid_pred.sum()),
                        "output_crs": output_crs,
                        "grid_mode": ("projected_meter" if projected_grid_info is not None else "epsg4326_width_height"),
                        "admin_boundary_applied": bool(admin_mask_applied),
                        "cropland_mask_applied": bool(cropland_mask_applied),
                        "mask_stats": mask_stats,
                        "spatial_uncertainty": spatial_meta,
                        "local_width_summary": {
                            "min": float(np.nanmin(local_width_valid)) if local_width_valid.size else None,
                            "mean": float(np.nanmean(local_width_valid)) if local_width_valid.size else None,
                            "max": float(np.nanmax(local_width_valid)) if local_width_valid.size else None,
                        },
                    })
                    gcp_report_path = out_dir / "gee_gcp_report.json"
                    gcp_report_path.write_text(json.dumps({"metrics": gcp_report}, ensure_ascii=False, indent=2), encoding="utf-8")
                    gcp_uncertainty = {
                        "enabled": True,
                        "method": gcp_report.get("method"),
                        "alpha": gcp_report.get("alpha"),
                        "nominal_coverage": gcp_report.get("nominal_coverage"),
                        "residual_quantile": gcp_report.get("residual_quantile"),
                        "global_residual_quantile": gcp_report.get("global_residual_quantile"),
                        "PICP": gcp_report.get("PICP"),
                        "NMPIW": gcp_report.get("NMPIW"),
                        "CCB": gcp_report.get("CCB"),
                        "IntervalScore": gcp_report.get("IntervalScore"),
                        "spatial_applied": spatial_meta.get("spatial_applied"),
                        "local_width_summary": gcp_report.get("local_width_summary"),
                        "spatial_uncertainty": spatial_meta,
                        "report_json": str(gcp_report_path),
                        "lower_tif": lower_tif,
                        "upper_tif": upper_tif,
                        "width_tif": width_tif,
                        "half_width_q_tif": q_tif,
                    }
                    _emit("MODEL", "已自动生成空间加权 GCP 不确定性图", gcp_uncertainty, task_id=self.task_id)
                except Exception as exc:
                    self.warnings.append(f"空间 GCP 不确定性输出失败：{exc}")
                    gcp_uncertainty = {"enabled": False, "error": str(exc)}

            default_result_mode = (os.getenv("PRO_DEFAULT_RESULT_TIF", "admin_continuous") or "admin_continuous").strip().lower()
            display_tif = admin_continuous_tif if (default_result_mode in {"admin", "admin_continuous", "continuous", "citywide"} and admin_continuous_tif is not None) else out_tif

            report = {
                "mode": "gee_formal_prediction_grid_v94_small_aoi_covariate_selection",
                "year": int(year),
                "target": target,
                "bounds_epsg4326": bounds,
                "bounds_source": bounds_source,
                "admin_bounds_meta": admin_bounds_meta,
                "width": int(width),
                "height": int(height),
                "point_count": int(width * height),
                "grid_mode": ("projected_meter" if projected_grid_info is not None else "epsg4326_width_height"),
                "output_crs": output_crs,
                "projected_grid": {
                    k: v for k, v in (projected_grid_info or {}).items()
                    if k not in {"grid_df", "transform"}
                } if projected_grid_info is not None else None,
                "feature_cols": feature_cols,
                "excluded_mask_cols": sorted([c for c in GEE_OUTPUT_MASK_COLUMNS if c in grid_df.columns]),
                "admin_boundary_applied": bool(admin_mask_applied),
                "cropland_mask_applied": bool(cropland_mask_applied),
                "mask_outputs": mask_outputs,
                "mask_stats": mask_stats,
                "gcp_uncertainty": gcp_uncertainty,
                "grid_covariates_csv": str(grid_csv),
                "gee_grid_meta": result.meta,
                "prediction_tif": str(out_tif),
                "display_prediction_tif": str(display_tif),
                "default_result_mode": default_result_mode,
                "note": "V94：支持上传样点覆盖大区域但按用户请求的小行政区制图；优先使用本地行政区划属性表识别县/区/市并裁剪输出；支持默认/用户指定环境协变量选择；默认WebGIS显示行政区连续预测图，耕地掩膜图保留为审计/专题输出。若resolution_coarsened=true，说明为了避免getInfo超限已自动放粗输出分辨率。",
            }
            report_path = out_dir / "gee_formal_prediction_grid_report.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            _emit("MODEL", "V82 GEE正式预测图已生成，并完成投影网格/行政边界/耕地mask审计", {
                "pred_tif": str(display_tif),
                "cropland_masked_tif": str(out_tif),
                "grid_csv": str(grid_csv),
                "report": str(report_path),
                "width": width,
                "height": height,
                "bounds_source": bounds_source,
                "grid_mode": ("projected_meter" if projected_grid_info is not None else "epsg4326_width_height"),
                "output_crs": output_crs,
                "output_resolution_m": (projected_grid_info or {}).get("output_resolution_m"),
                "resolution_coarsened": (projected_grid_info or {}).get("resolution_coarsened"),
                "feature_cols": feature_cols,
                "admin_boundary_applied": bool(admin_mask_applied),
                "cropland_mask_applied": bool(cropland_mask_applied),
                "mask_outputs": mask_outputs,
                "mask_stats": mask_stats,
                "gcp_uncertainty": gcp_uncertainty,
            }, task_id=self.task_id)
            return Path(display_tif)
        except Exception as exc:
            self.warnings.append(f"GEE正式预测图输出失败：{exc}")
            _emit("MODEL", "GEE正式预测图输出失败，已回退普通预览图逻辑", {"error": str(exc)}, task_id=self.task_id)
            return None

    def _classify_downloaded_file(self, path: Path, content_type: str = "", content_disposition: str = "") -> dict[str, Any]:
        """判断下载结果是否是真实原始数据，而不是 HTML 登录页/搜索页。

        返回字段：
        - is_true_data: 是否可计入 true_raw_downloaded_count
        - kind: geotiff/zip/hdf/netcdf/table/json/html/unknown
        - reason: 判定依据
        """
        suffix = path.suffix.lower()
        try:
            head = path.read_bytes()[:4096]
        except Exception as exc:
            return {"is_true_data": False, "kind": "unreadable", "reason": f"文件头读取失败: {exc}"}
        text_head = ""
        try:
            text_head = head[:1024].decode("utf-8", errors="ignore").lower().strip()
        except Exception:
            text_head = ""
        ct = (content_type or "").lower()
        cd = (content_disposition or "").lower()
        html_magic = text_head.startswith("<!doctype html") or text_head.startswith("<html") or "<html" in text_head[:300]
        if "text/html" in ct or html_magic:
            return {"is_true_data": False, "kind": "html", "reason": "Content-Type 或文件头表明这是 HTML 页面，不是原始数据文件。"}
        if head.startswith(b"PK\x03\x04") or suffix == ".zip":
            return {"is_true_data": True, "kind": "zip", "reason": "ZIP 文件头或 .zip 扩展名。"}
        if head.startswith(b"II*\x00") or head.startswith(b"MM\x00*") or suffix in {".tif", ".tiff"}:
            return {"is_true_data": True, "kind": "geotiff", "reason": "TIFF 文件头或 GeoTIFF 扩展名。"}
        if head.startswith(b"\x89HDF") or suffix in {".hdf", ".h5", ".hdf5"}:
            return {"is_true_data": True, "kind": "hdf", "reason": "HDF 文件头或 HDF 扩展名。"}
        if head.startswith(b"CDF") or suffix == ".nc":
            return {"is_true_data": True, "kind": "netcdf", "reason": "NetCDF 文件头或 .nc 扩展名。"}
        if suffix in {".csv", ".xls", ".xlsx"} or "spreadsheet" in ct or "excel" in ct or "csv" in ct:
            return {"is_true_data": True, "kind": "table", "reason": "表格扩展名或表格 Content-Type。"}
        if suffix in {".json", ".geojson"} or "application/json" in ct:
            # 避免把 CensorWords.json、前端配置、语言包等站点资源误判为真实数据。
            # 只有 GeoJSON/STAC/含明确空间数据资产字段的 JSON 才算作真实数据。
            try:
                payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
                text_probe = json.dumps(payload, ensure_ascii=False)[:8000].lower()
                if isinstance(payload, dict):
                    typ = str(payload.get("type", "")).lower()
                    if typ in {"featurecollection", "feature"}:
                        return {"is_true_data": True, "kind": "geojson", "reason": "GeoJSON Feature/FeatureCollection。"}
                    if any(k in payload for k in ["assets", "bbox", "geometry", "links", "features"]):
                        return {"is_true_data": True, "kind": "json", "reason": "JSON含空间/资产/链接字段，可作为数据或STAC/GeoJSON元数据。"}
                if any(k in text_probe for k in ["geotiff", ".tif", ".tiff", ".hdf", ".nc", "downloadurl", "download_url", "spatial", "extent", "bbox", "geometry"]):
                    return {"is_true_data": True, "kind": "json", "reason": "JSON文本含空间范围或真实文件下载线索。"}
            except Exception:
                pass
            return {"is_true_data": False, "kind": "json_site_resource", "reason": "普通JSON站点资源/配置文件，未识别为空间数据或真实数据元数据。"}
        if "application/octet-stream" in ct and len(head) > 1024:
            return {"is_true_data": True, "kind": "binary", "reason": "二进制流且不是 HTML，暂按原始二进制数据处理。"}
        return {"is_true_data": False, "kind": "unknown", "reason": f"未识别为真实数据文件；suffix={suffix}, content_type={content_type}"}

    def _download_file(self, url: str, out_dir: Path, stem: str, session: requests.Session | None = None) -> Path | None:
        """下载候选文件，带 HEAD 大小检查、文件大小阈值、复用缓存和短超时。

        正式数据准备可以把阈值调大；默认跳过大文件，避免一次任务被 100MB+ 文件拖住。
        """
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            dl_session = session or self.session
            suffix = Path(urlparse(url).path).suffix or ".dat"
            url_key = hashlib.sha1(url.encode("utf-8", errors="ignore")).hexdigest()
            cache_file = self.download_cache_dir / f"{url_key}{suffix}"
            out = out_dir / (_safe_name(stem, 80) + suffix)
            if self.download_reuse and cache_file.exists() and cache_file.stat().st_size > 0:
                shutil.copy2(cache_file, out)
                _emit("DOWNLOAD", "命中下载缓存，已复用本地文件", {"url": url, "cache": str(cache_file), "out": str(out), "bytes": out.stat().st_size}, task_id=self.task_id)
                return out

            head_ok = True
            size = 0
            try:
                h = dl_session.head(url, allow_redirects=True, timeout=min(self.timeout, int(os.getenv("PRO_PLATFORM_HEAD_TIMEOUT", "10"))))
                size = int(h.headers.get("content-length") or 0)
                limit_mb = min(self.max_download_mb, self.diagnostic_max_file_mb) if self.skip_large_file else self.max_download_mb
                head_ct = (h.headers.get("content-type") or "").lower()
                head_cd = (h.headers.get("content-disposition") or "").lower()
                if self.domestic_validate_download_links and "text/html" in head_ct and "attachment" not in head_cd:
                    self.warnings.append(f"候选下载地址返回HTML页面，判定为网页界面/登录页而非真实数据直链：{url}")
                    _emit("DOWNLOAD", "候选下载链接被判定为网页界面，未计入真实数据下载", {"url": url, "content_type": head_ct, "content_disposition": head_cd, "assessment": self._static_download_url_assessment(url)}, task_id=self.task_id)
                    return None
                if size and size > limit_mb * 1024 * 1024:
                    self.warnings.append(f"跳过过大文件：{url} size={size} bytes，limit={limit_mb} MB")
                    _emit("DOWNLOAD", "下载前HEAD检查：文件超过阈值，已跳过", {"url": url, "bytes": size, "limit_mb": limit_mb}, task_id=self.task_id)
                    return None
            except Exception as exc:
                head_ok = False
                self.warnings.append(f"HEAD检查失败，继续按下载超时策略尝试：{url} | {exc}")
            _emit("DOWNLOAD", "开始下载候选数据文件", {"url": url, "head_checked": head_ok, "content_length": size, "auth_used": bool(session)}, task_id=self.task_id)
            download_timeout = int(os.getenv("PRO_PLATFORM_DOWNLOAD_TIMEOUT", str(self.timeout)))
            r = dl_session.get(url, stream=True, timeout=download_timeout)
            if not r.ok:
                self.warnings.append(f"下载失败 HTTP {r.status_code}: {url}")
                return None
            total = 0
            limit_mb = min(self.max_download_mb, self.diagnostic_max_file_mb) if self.skip_large_file else self.max_download_mb
            limit = limit_mb * 1024 * 1024
            tmp = out.with_suffix(out.suffix + ".part")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > limit:
                        try:
                            tmp.unlink(missing_ok=True)
                        except Exception:
                            pass
                        raise RuntimeError(f"下载超过大小限制 {limit_mb} MB")
                    f.write(chunk)
            tmp.replace(out)
            classification = self._classify_downloaded_file(
                out,
                content_type=r.headers.get("content-type", ""),
                content_disposition=r.headers.get("content-disposition", ""),
            )
            if self.strict_true_data_download and not classification.get("is_true_data"):
                probe_path = out
                if classification.get("kind") == "html" and self.save_html_probe:
                    probe_path = out.with_suffix(out.suffix + ".html_probe")
                    try:
                        out.replace(probe_path)
                    except Exception:
                        probe_path = out
                _emit("DOWNLOAD", "下载结果不是真实原始数据，已按探测页面/无效下载处理", {
                    "url": url,
                    "path": str(probe_path),
                    "bytes": total,
                    "classification": classification,
                }, task_id=self.task_id)
                return None
            if self.download_reuse:
                try:
                    shutil.copy2(out, cache_file)
                except Exception as exc:
                    self.warnings.append(f"写入下载缓存失败：{exc}")
            _emit("DOWNLOAD", "真实原始数据文件下载成功", {"url": url, "path": str(out), "bytes": total, "cached": self.download_reuse, "classification": classification}, task_id=self.task_id)
            return out
        except Exception as exc:
            self.warnings.append(f"下载异常：{url} | {exc}")
            _emit("DOWNLOAD", "候选数据文件下载异常，已跳过", {"url": url, "error": str(exc)}, task_id=self.task_id)
            return None

    def _extract_links(self, html: str, base_url: str) -> list[tuple[str, str]]:
        if not html:
            return []
        links = []
        # href="..." 或 href='...'
        pattern = re.compile(r"<a[^>]+href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.I | re.S)
        for href, text in pattern.findall(html):
            if not href or href.startswith(("javascript:", "#", "mailto:")):
                continue
            txt = re.sub(r"<[^>]+>", " ", text)
            txt = re.sub(r"\s+", " ", txt).strip()
            links.append((txt, urljoin(base_url, href)))
        return links

    def _is_download_url(self, url: str) -> bool:
        if not url:
            return False
        parsed = urlparse(url)
        path = parsed.path.lower()
        suffix = Path(path).suffix
        if suffix in DOWNLOAD_EXTS:
            return True
        # 有些国内平台下载接口没有文件后缀，依靠 query/path 中的 download/export/file 标识。
        full = (path + "?" + parsed.query).lower()
        if any(k in full for k in ["download", "downfile", "export", "fileid", "file_id", "attach", "attachment", "getfile", "datafile", "下载"]):
            return True
        # URL 查询参数中出现文件名后缀。
        if any(ext in full for ext in DOWNLOAD_EXTS):
            return True
        return False

    def _build_report(self, target: dict[str, Any], sample_meta: dict[str, Any], statuses: list[PlatformStatus], model_ready_csv: Path, model_report_json: Path | None, pred_tif: Path | None, real_cov_cols: list[str], feature_cols: list[str] | None = None, gee_mask_cols: list[str] | None = None) -> str:
        lines = [
            "# V108 正式制图运行报告：用户样点 + 国内平台数据下载",
            "",
            "## 目标",
            f"- 区域：{target.get('region')}",
            f"- 年份：{target.get('year')}",
            f"- 目标CRS：{target.get('target_crs')}",
            f"- 目标分辨率：{target.get('resolution_m')} m",
            "",
            "## 样点读取",
            f"- 有效样点数：{sample_meta.get('valid_rows')}",
            f"- 编码/读取信息：{sample_meta.get('encoding')}",
            f"- 字段映射：{json.dumps(sample_meta.get('matched_fields'), ensure_ascii=False)}",
            "",
            "## 数据源审计状态",
            "| 平台 | 可访问 | 下载链接 | 下载文件 | 标准化栅格 | 抽样字段 | 状态 |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for s in statuses:
            lines.append(f"| {s.platform_name} | {int(s.reachable)} | {s.download_link_count} | {s.downloaded_count} | {s.processed_raster_count} | {s.sample_aligned_count} | {s.status} |")
        mb_audit = target.get("v175_multiband_stack_audit") or []
        if mb_audit:
            lines += ["", "## V175 多波段协变量栈审计"]
            for i, item in enumerate(mb_audit, start=1):
                lines.append(f"- 栈{i}：`{item.get('stack_path')}`；band数：{item.get('band_count')}；命名来源：{item.get('source')}；映射表：`{item.get('mapping_file')}`")
                added = item.get("added_columns") or []
                mask_only = item.get("mask_only_columns") or []
                lines.append(f"  - 已作为SOM协变量接入：{len(added)} 个")
                if added:
                    lines.append("  - 前10个协变量：" + ", ".join([str((x or {}).get("column")) for x in added[:10]]))
                if mask_only:
                    lines.append("  - 仅作掩膜、不入模：" + ", ".join([str(x) for x in mask_only[:10]]))
                if item.get("warnings"):
                    lines.append("  - 警告：" + "; ".join([str(x) for x in (item.get("warnings") or [])[:5]]))
        lines += [
            "",
            "## 建模CSV",
            f"- 路径：`{model_ready_csv}`",
            f"- GEE回传 cov_* 列总数：{len(real_cov_cols)}",
            f"- 实际参与建模的环境协变量数：{len(feature_cols or [])}",
            f"- 实际参与建模的环境协变量列：{', '.join(feature_cols or []) if feature_cols else '无'}",
            f"- 仅用于输出裁剪/审计的mask列：{', '.join(gee_mask_cols or []) if gee_mask_cols else '无'}",
            "- 行政边界/耕地掩膜列不会作为模型特征；它们只用于最终 GeoTIFF 的 NoData 裁剪。",
            "",
            "## 模型训练与制图输出",
            f"- 模型报告：`{model_report_json}`" if model_report_json else "- 模型报告：未生成",
            f"- 预测图：`{pred_tif}`" if pred_tif else "- 预测图：未生成",
            "",
            "## 严格说明",
            "当前 已固化为“用户上传样点 + GEE 自动环境协变量 + RFK（随机森林 + 残差普通克里金）正式制图”流程。该版本可作为正式运行版主线；后续研究版仍建议继续接入多模型比较、GCP/AOA 与更严格空间验证。",
        ]
        return "\n".join(lines)


def run_platform_model_csv_pipeline(out_root: str | Path, request_text: str, sample_path: str | Path, task_id: str | None = None) -> PipelineResult:
    pipeline = PlatformModelCsvPipeline(out_root=out_root, task_id=task_id)
    return pipeline.run(request_text=request_text, sample_path=sample_path)
