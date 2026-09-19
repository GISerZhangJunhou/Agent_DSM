from __future__ import annotations

import atexit
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

try:
    import rasterio  # type: ignore
except Exception:  # pragma: no cover
    rasterio = None

DOMESTIC_DOWNLOAD_EXTS = {
    ".tif", ".tiff", ".img", ".vrt", ".zip", ".rar", ".7z",
    ".hdf", ".h5", ".hdf5", ".nc", ".csv", ".xls", ".xlsx", ".geojson", ".json"
}
BLOCKED_WEB_EXTS = {".html", ".htm", ".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".txt"}
DEFAULT_SAVE_ROOT = Path(os.getenv("PRO_DOWNLOAD_TASK_ROOT", os.getenv("TPDC_DOWNLOAD_DIR", r"E:\Agent_DSM\data_download")))
DEFAULT_MANUAL_ROOT = Path(os.getenv("PRO_MANUAL_DOWNLOAD_DIR", str(DEFAULT_SAVE_ROOT / "manual")))
ALLOW_CUSTOM_SAVE_DIR = os.getenv("PRO_DOWNLOAD_ALLOW_CUSTOM_SAVE_DIR", "1") != "0"

# Manual FTP downloads are child processes of the Dash app.
# On a normal app exit, terminate tracked children so stale progress does not
# continue after the user has closed the whole system.
_MANUAL_FTP_CHILDREN: list[subprocess.Popen] = []

def _terminate_manual_ftp_children_on_exit() -> None:
    if os.getenv("DOMESTIC_STOP_FTP_ON_APP_EXIT", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    for proc in list(_MANUAL_FTP_CHILDREN):
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass

atexit.register(_terminate_manual_ftp_children_on_exit)



# V110: platform credentials are separated from the app login account.
# Keep these values server-side only. Do not expose passwords to the frontend.
PLATFORM_CREDENTIAL_ENV = {
    "gscloud": ("GSCLOUD_USERNAME", "GSCLOUD_PASSWORD"),
    "geodata": ("GEODATA_USERNAME", "GEODATA_PASSWORD"),
    "resdc": ("RESDC_USERNAME", "RESDC_PASSWORD"),
    "noda": ("NODA_USERNAME", "NODA_PASSWORD"),
    "cma_nmic": ("NMIC_USERNAME", "NMIC_PASSWORD"),
    "nsmc": ("NSMC_USERNAME", "NSMC_PASSWORD"),
    "tpdc": ("TPDC_USERNAME", "TPDC_PASSWORD"),
    "nesdc": ("NESDC_USERNAME", "NESDC_PASSWORD"),
    "ngac": ("NGAC_USERNAME", "NGAC_PASSWORD"),
}


def _env_truth(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _clean_secret(value: str | None) -> str:
    v = str(value or "").strip()
    bad = {"", "your_username", "your_password", "填写账号", "填写密码", "请填写", "your-key-here"}
    if v in bad or v.lower().startswith("your_") or "请" in v or "填写" in v:
        return ""
    return v


def _mask_username(username: str) -> str:
    u = str(username or "").strip()
    if not u:
        return "未配置"
    if "@" in u:
        head, tail = u.split("@", 1)
        if len(head) <= 2:
            return head[:1] + "***@" + tail
        return head[:2] + "***@" + tail
    if len(u) <= 3:
        return u[:1] + "***"
    return u[:2] + "***" + u[-1:]


def get_platform_credentials(platform_id: str | None) -> dict[str, Any]:
    pid = str(platform_id or "").strip()
    user_env, pass_env = PLATFORM_CREDENTIAL_ENV.get(pid, ("", ""))
    username = _clean_secret(os.getenv(user_env)) if user_env else ""
    password = _clean_secret(os.getenv(pass_env)) if pass_env else ""
    return {
        "platform_id": pid,
        "username_env": user_env,
        "password_env": pass_env,
        "has_username": bool(username),
        "has_password": bool(password),
        "username_masked": _mask_username(username),
        "credential_source": "env" if (username or password) else "not_configured",
        # Never include raw password in manifest/payload.
    }


def build_platform_account_config(platform_ids: list[str] | None = None) -> dict[str, Any]:
    pids = platform_ids or list(PLATFORM_CREDENTIAL_ENV.keys())
    credentials = [get_platform_credentials(pid) for pid in pids]
    configured = [c for c in credentials if c.get("has_username") and c.get("has_password")]
    return {
        "use_env_credentials": _env_truth("DOMESTIC_PLATFORM_USE_ENV_CREDENTIALS", "1"),
        "auto_login": _env_truth("DOMESTIC_PLATFORM_AUTO_LOGIN", "1"),
        "auto_register": _env_truth("DOMESTIC_PLATFORM_AUTO_REGISTER", "0"),
        "human_verify_mode": os.getenv("DOMESTIC_PLATFORM_HUMAN_VERIFY_MODE", "manual"),
        "save_session": _env_truth("DOMESTIC_PLATFORM_SAVE_SESSION", "1"),
        "session_dir": os.getenv("DOMESTIC_PLATFORM_SESSION_DIR", r"E:\Agent_DSM\runs\sessions"),
        "configured_platform_count": len(configured),
        "credentials": credentials,
        "security_note": "平台账号密码仅在后端读取；前端不显示密码。验证码、人机验证、实名确认由用户在平台页面手动完成。",
    }


@dataclass
class DownloadIntent:
    task_type: str
    request_text: str
    data_name: str
    region: str
    year: int | None
    temporal_type: str
    platform_scope: str = "domestic_only"
    free_only: bool = True
    save_dir: str | None = None
    custom_save_dir: bool = False
    requested_platform_id: str | None = None
    requested_platform_name: str | None = None
    requested_resolution_meters: float | None = None


DOMESTIC_FREE_PLATFORMS: list[dict[str, Any]] = [
    {
        "id": "gscloud",
        "name": "地理空间数据云",
        "url": "https://www.gscloud.cn/",
        "priority": 2,
        "best_for": ["DEM", "Landsat", "Sentinel", "MODIS", "LUCC", "NDVI"],
        "free_status": "free_after_login",
        "access_type": "login_required_or_direct",
        "note": "优先用于 DEM、遥感影像和 LUCC 等公开免费数据；如需登录/验证码，由用户接管。",
    },
    {
        "id": "geodata",
        "name": "国家地球系统科学数据中心",
        "url": "https://www.geodata.cn/data/",
        "priority": 3,
        "best_for": ["DEM", "地形", "气候", "陆表", "NPP"],
        "free_status": "free_after_login_or_order",
        "access_type": "login_or_order_required",
        "note": "用于地形、气候、陆表等数据候选；是否免费以具体数据页面为准。",
    },
    {
        "id": "resdc",
        "name": "资源环境科学与数据中心",
        "url": "https://www.resdc.cn/",
        "priority": 4,
        "best_for": ["DEM", "土地利用", "土地覆盖", "LUCC", "LULC", "资源环境"],
        "free_status": "free_after_login_or_unknown",
        "access_type": "login_required_or_manual",
        "note": "用于 DEM、土地利用、资源环境类数据候选；是否免费以具体数据页面为准。",
    },
    {
        "id": "noda",
        "name": "国家综合地球观测数据共享平台",
        "url": "https://www.noda.ac.cn/",
        "priority": 5,
        "best_for": ["遥感影像", "高分", "资源", "环境减灾", "Landsat", "Sentinel"],
        "free_status": "free_application_or_order",
        "access_type": "application_or_order_required",
        "note": "用于遥感影像和对地观测数据候选；通常需要申请/订单/人工接管。",
    },
    {
        "id": "cma_nmic",
        "name": "国家气象信息中心 / 中国气象数据网",
        "url": "https://data.cma.cn/",
        "priority": 6,
        "best_for": ["气象", "降水", "气温", "站点"],
        "free_status": "free_after_login_or_order",
        "access_type": "login_or_order_required",
        "note": "用于气象数据候选；通常需要登录、检索、订单或人工下载。",
    },
    {
        "id": "nsmc",
        "name": "国家卫星气象中心 / 风云卫星数据服务网",
        "url": "https://satellite.nsmc.org.cn/portalsite/default.aspx",
        "priority": 7,
        "best_for": ["风云", "卫星", "气象遥感", "FTP"],
        "free_status": "free_after_realname_or_order",
        "access_type": "realname_order_or_ftp",
        "note": "用于风云卫星数据；可能需要实名、订单或 FTP 下载。",
    },
    {
        "id": "tpdc",
        "name": "国家青藏高原科学数据中心",
        "url": "https://data.tpdc.ac.cn/",
        "priority": 0,
        "best_for": ["高原", "气候", "水文", "生态", "DEM", "数字高程", "地形", "FTP", "NPP", "土壤"],
        "free_status": "free_after_login_or_ftp",
        "access_type": "login_required_or_ftp",
        "note": "V151 全局默认优先平台；优先查找免费 FTP 数据入口。若页面提供 FTP 主机、端口、账号、密码，系统会自动识别并尝试启动 FTP 下载。",
    },
    {
        "id": "ngac",
        "name": "全国地质资料馆",
        "url": "https://www.ngac.cn/125cms/c/qggnew/index.htm",
        "priority": 8,
        "best_for": ["地质", "地形", "DEM", "基础地理"],
        "free_status": "free_after_login_or_unknown",
        "access_type": "login_required_or_manual",
        "note": "用于地质、地形、基础地理相关数据候选；是否免费以具体数据页面为准。",
    },
    {
        "id": "nesdc",
        "name": "国家生态数据中心资源共享服务平台",
        "url": "https://www.nesdc.org.cn/",
        "priority": 9,
        "best_for": ["生态", "土壤", "植被", "NPP", "LULC"],
        "free_status": "free_after_login_or_unknown",
        "access_type": "login_required_or_manual",
        "note": "用于生态、植被、土壤相关数据候选；是否免费以具体数据页面为准。",
    },
]


PLATFORM_NAME_ALIASES: dict[str, str] = {
    "地理空间数据云": "gscloud",
    "gscloud": "gscloud",
    "国家地球系统科学数据中心": "geodata",
    "地球系统科学数据中心": "geodata",
    "geodata": "geodata",
    "资源环境科学与数据中心": "resdc",
    "资源环境科学数据中心": "resdc",
    "resdc": "resdc",
    "国家综合地球观测数据共享平台": "noda",
    "综合地球观测": "noda",
    "noda": "noda",
    "中国气象数据网": "cma_nmic",
    "国家气象信息中心": "cma_nmic",
    "气象数据网": "cma_nmic",
    "风云卫星数据服务网": "nsmc",
    "国家卫星气象中心": "nsmc",
    "风云": "nsmc",
    "青藏高原科学数据中心": "tpdc",
    "国家青藏高原科学数据中心": "tpdc",
    "tpdc": "tpdc",
    "TPDC": "tpdc",
    "全国地质资料馆": "ngac",
    "国家地质资料馆": "ngac",
    "地质资料馆": "ngac",
    "ngac": "ngac",
    "国家生态数据中心": "nesdc",
    "国家生态数据中心资源共享服务平台": "nesdc",
    "生态数据中心": "nesdc",
    "nesdc": "nesdc",
}

DATA_ALIASES = {
    "DEM": ["dem", "数字高程", "高程模型", "高程", "海拔", "地形", "srtm", "aster gdem", "gdem"],
    "NDVI": ["ndvi", "植被指数"],
    "EVI": ["evi", "增强植被指数"],
    "LAI": ["lai", "叶面积指数"],
    "FVC": ["fvc", "植被覆盖度", "植被覆盖率"],
    "LULC": ["lulc", "土地利用", "土地覆盖", "lucc", "clcd", "地表覆盖"],
    "NPP": ["npp", "净初级生产力", "生产力"],
    "降水": ["降水", "降雨", "precipitation", "rainfall"],
    "气温": ["气温", "温度", "temperature", "地表温度", "lst"],
    # 专业气候/水文变量必须优先作为独立数据名识别，不能退化为“气象”。
    # 例如用户说“潜在蒸散发量”，TPDC 搜索框就应填“潜在蒸散发量”，
    # 而不是填过宽的“气象”。
    "潜在蒸散发量": [
        "潜在蒸散发量", "潜在蒸散发", "潜在蒸散量", "参考蒸散发", "参考作物蒸散发",
        "potential evapotranspiration", "PET", "pet", "ET0", "ET₀", "eto",
    ],
    "实际蒸散发量": [
        "实际蒸散发量", "实际蒸散发", "蒸散发量", "蒸散发",
        "actual evapotranspiration", "evapotranspiration", "AET", "aet", "ET", "et",
    ],
    "风速": ["风速", "风场", "wind speed", "wind"],
    "相对湿度": ["相对湿度", "湿度", "relative humidity", "humidity", "RH", "rh"],
    "气象": ["气象", "水热"],
    "土壤水分": ["土壤水分", "土壤湿度", "soil moisture", "sm"],
    "土壤": ["土壤", "ph", "有机质", "质地", "砂粒", "黏粒", "容重"],
    "水文": ["水文", "径流", "河流", "湖泊", "水系", "冰川", "积雪"],
    "遥感影像": ["遥感影像", "landsat", "sentinel", "modis", "高分", "gf", "影像"],
    "地表太阳辐射": ["地表太阳辐射", "太阳辐射", "短波辐射", "地表下行短波辐射", "surface solar radiation", "solar radiation", "ssrd"],
}

GENERIC_DATA_WORDS = ["数据", "数据集", "栅格", "影像", "产品", "协变量", "变量", "图层"]

STATIC_DATA = {"DEM", "土壤", "土壤水分"}
REGION_SUFFIXES = ["省", "市", "区", "县", "州", "盟", "旗"]


AUTO_PRODUCT_RULES: dict[str, dict[str, Any]] = {
    "DEM": {
        "preferred_platform_id": "tpdc",
        "product_name": "DEM / 数字高程模型（全局优先国家青藏高原科学数据中心 TPDC，FTP 优先；按分辨率约束选择可重采样数据）",
        "search_keywords": ["DEM", "数字高程模型", "数字高程", "高程", "SRTM", "ASTER GDEM", "GDEM"],
        "required_dataset_terms": ["DEM", "数字高程", "数字高程模型", "高程模型", "SRTM", "ASTER GDEM", "GDEM", "地形"],
        "reject_dataset_terms": ["语义分割", "图像分割", "semantic segmentation", "segmentation", "深度学习", "标注样本", "训练样本", "遥感影像分割"],
        "entry_url_by_platform": {"tpdc": "https://data.tpdc.ac.cn/"},
        "temporal_note": "DEM 属于静态地形协变量；用户给出的年份只作为任务年份记录，不强制寻找年度 DEM。",
    },
    "NDVI": {
        "preferred_platform_id": "tpdc",
        "product_name": "NDVI / 植被指数（V151 固定只使用 TPDC，FTP 优先）",
        "search_keywords": ["NDVI", "植被指数", "MODIS NDVI", "归一化植被指数"],
        "required_dataset_terms": ["NDVI", "植被指数", "归一化植被指数"],
        "reject_dataset_terms": ["语义分割", "图像分割", "标注样本", "训练样本"],
        "entry_url_by_platform": {"tpdc": "https://data.tpdc.ac.cn/"},
        "temporal_note": "NDVI 属于动态变量，必须优先匹配目标年份或覆盖目标年份的时间范围。",
    },
    "EVI": {
        "preferred_platform_id": "tpdc",
        "product_name": "EVI / 增强植被指数（全局优先 TPDC，FTP 优先）",
        "search_keywords": ["EVI", "增强植被指数", "MODIS EVI"],
        "required_dataset_terms": ["EVI", "增强植被指数"],
        "reject_dataset_terms": ["语义分割", "图像分割", "标注样本", "训练样本"],
        "entry_url_by_platform": {"tpdc": "https://data.tpdc.ac.cn/"},
        "temporal_note": "EVI 属于动态变量，必须优先匹配目标年份或覆盖目标年份的时间范围。",
    },
    "LAI": {
        "preferred_platform_id": "tpdc",
        "product_name": "LAI / 叶面积指数（全局优先 TPDC，FTP 优先）",
        "search_keywords": ["LAI", "叶面积指数"],
        "required_dataset_terms": ["LAI", "叶面积指数"],
        "reject_dataset_terms": ["语义分割", "图像分割", "标注样本", "训练样本"],
        "entry_url_by_platform": {"tpdc": "https://data.tpdc.ac.cn/"},
        "temporal_note": "LAI 属于动态变量，必须优先匹配目标年份或覆盖目标年份的时间范围。",
    },
    "FVC": {
        "preferred_platform_id": "tpdc",
        "product_name": "FVC / 植被覆盖度（全局优先 TPDC，FTP 优先）",
        "search_keywords": ["FVC", "植被覆盖度", "植被覆盖率"],
        "required_dataset_terms": ["FVC", "植被覆盖度", "植被覆盖率"],
        "reject_dataset_terms": ["语义分割", "图像分割", "标注样本", "训练样本"],
        "entry_url_by_platform": {"tpdc": "https://data.tpdc.ac.cn/"},
        "temporal_note": "FVC 属于动态变量，必须优先匹配目标年份或覆盖目标年份的时间范围。",
    },
    "LULC": {
        "preferred_platform_id": "tpdc",
        "product_name": "土地利用 / 土地覆盖 / LUCC / CLCD（全局优先 TPDC，FTP 优先）",
        "search_keywords": ["土地利用", "土地覆盖", "LUCC", "CLCD", "地表覆盖"],
        "required_dataset_terms": ["土地利用", "土地覆盖", "LUCC", "CLCD", "地表覆盖", "LULC"],
        "reject_dataset_terms": ["语义分割", "图像分割", "标注样本", "训练样本"],
        "entry_url_by_platform": {"tpdc": "https://data.tpdc.ac.cn/"},
        "temporal_note": "土地利用属于动态或准动态变量，应优先匹配目标年份。",
    },
    "NPP": {
        "preferred_platform_id": "tpdc",
        "product_name": "NPP / 净初级生产力（全局优先 TPDC，FTP 优先）",
        "search_keywords": ["NPP", "净初级生产力"],
        "required_dataset_terms": ["NPP", "净初级生产力", "生产力"],
        "reject_dataset_terms": ["语义分割", "图像分割", "标注样本", "训练样本"],
        "entry_url_by_platform": {"tpdc": "https://data.tpdc.ac.cn/"},
        "temporal_note": "NPP 属于动态变量，必须匹配目标年份或覆盖目标年份的时间范围。",
    },
    "降水": {
        "preferred_platform_id": "tpdc",
        "product_name": "降水 / 降雨数据（全局优先 TPDC，FTP 优先）",
        "search_keywords": ["降水", "降雨", "precipitation", "rainfall"],
        "required_dataset_terms": ["降水", "降雨", "precipitation", "rainfall"],
        "reject_dataset_terms": ["语义分割", "图像分割", "标注样本", "训练样本"],
        "entry_url_by_platform": {"tpdc": "https://data.tpdc.ac.cn/"},
        "temporal_note": "降水属于动态变量，必须匹配目标年份、月份或日期范围。",
    },
    "气温": {
        "preferred_platform_id": "tpdc",
        "product_name": "气温 / 温度 / LST 数据（全局优先 TPDC，FTP 优先）",
        "search_keywords": ["气温", "温度", "temperature", "LST", "地表温度"],
        "required_dataset_terms": ["气温", "温度", "temperature", "LST", "地表温度"],
        "reject_dataset_terms": ["语义分割", "图像分割", "标注样本", "训练样本"],
        "entry_url_by_platform": {"tpdc": "https://data.tpdc.ac.cn/"},
        "temporal_note": "气温/温度属于动态变量，必须匹配目标年份、月份或日期范围。",
    },
    "潜在蒸散发量": {
        "preferred_platform_id": "tpdc",
        "product_name": "潜在蒸散发量 / 参考蒸散发数据（固定使用 TPDC，FTP 优先）",
        "search_keywords": ["潜在蒸散发量", "潜在蒸散发", "参考蒸散发"],
        "required_dataset_terms": ["潜在蒸散发", "潜在蒸散发量", "参考蒸散发", "potential evapotranspiration", "PET", "ET0", "eto"],
        "reject_dataset_terms": ["语义分割", "图像分割", "标注样本", "训练样本"],
        "entry_url_by_platform": {"tpdc": "https://data.tpdc.ac.cn/"},
        "temporal_note": "潜在蒸散发量属于动态水热变量，若数据集时间范围覆盖目标年份即可下载，后续只保留目标年份并裁剪到研究区。",
    },
    "实际蒸散发量": {
        "preferred_platform_id": "tpdc",
        "product_name": "实际蒸散发量 / 蒸散发数据（固定使用 TPDC，FTP 优先）",
        "search_keywords": ["蒸散发", "实际蒸散发", "蒸散发量"],
        "required_dataset_terms": ["蒸散发", "实际蒸散发", "蒸散发量", "evapotranspiration", "ET", "AET"],
        "reject_dataset_terms": ["语义分割", "图像分割", "标注样本", "训练样本"],
        "entry_url_by_platform": {"tpdc": "https://data.tpdc.ac.cn/"},
        "temporal_note": "蒸散发属于动态水热变量，必须优先匹配目标年份或覆盖目标年份的时间范围。",
    },
    "风速": {
        "preferred_platform_id": "tpdc",
        "product_name": "风速数据（固定使用 TPDC，FTP 优先）",
        "search_keywords": ["风速"],
        "required_dataset_terms": ["风速", "风场", "wind speed", "wind"],
        "reject_dataset_terms": ["语义分割", "图像分割", "标注样本", "训练样本"],
        "entry_url_by_platform": {"tpdc": "https://data.tpdc.ac.cn/"},
        "temporal_note": "风速属于动态气象变量，应优先匹配目标年份或覆盖目标年份的时间范围。",
    },
    "相对湿度": {
        "preferred_platform_id": "tpdc",
        "product_name": "相对湿度数据（固定使用 TPDC，FTP 优先）",
        "search_keywords": ["相对湿度"],
        "required_dataset_terms": ["相对湿度", "湿度", "relative humidity", "humidity", "RH"],
        "reject_dataset_terms": ["语义分割", "图像分割", "标注样本", "训练样本"],
        "entry_url_by_platform": {"tpdc": "https://data.tpdc.ac.cn/"},
        "temporal_note": "相对湿度属于动态气象变量，应优先匹配目标年份或覆盖目标年份的时间范围。",
    },
    "地表太阳辐射": {
        "preferred_platform_id": "tpdc",
        "product_name": "地表太阳辐射 / 太阳辐射数据（固定使用 TPDC，FTP 优先）",
        "search_keywords": ["地表太阳辐射"],
        "required_dataset_terms": ["地表太阳辐射", "太阳辐射", "短波辐射", "surface solar radiation", "solar radiation", "SSRD"],
        "reject_dataset_terms": ["语义分割", "图像分割", "标注样本", "训练样本"],
        "entry_url_by_platform": {"tpdc": "https://data.tpdc.ac.cn/"},
        "temporal_note": "地表太阳辐射属于动态气候/辐射变量，若数据集时间范围覆盖目标年份即可下载，后续只保留目标年份。",
    },
    "气象": {
        "preferred_platform_id": "tpdc",
        "product_name": "气象协变量（V151 固定只使用 TPDC，FTP 优先）",
        "search_keywords": ["气象", "气温", "降水", "风速", "湿度", "蒸散"],
        "required_dataset_terms": ["气象", "气温", "降水", "风速", "湿度", "蒸散"],
        "reject_dataset_terms": ["语义分割", "图像分割", "标注样本", "训练样本"],
        "entry_url_by_platform": {"tpdc": "https://data.tpdc.ac.cn/"},
        "temporal_note": "气象属于动态变量，必须匹配目标年份或日期范围。",
    },
    "土壤水分": {
        "preferred_platform_id": "tpdc",
        "product_name": "土壤水分 / 土壤湿度数据（全局优先 TPDC，FTP 优先）",
        "search_keywords": ["土壤水分", "土壤湿度", "soil moisture"],
        "required_dataset_terms": ["土壤水分", "土壤湿度", "soil moisture"],
        "reject_dataset_terms": ["语义分割", "图像分割", "标注样本", "训练样本"],
        "entry_url_by_platform": {"tpdc": "https://data.tpdc.ac.cn/"},
        "temporal_note": "土壤水分可能是动态变量；若用户给出年份或日期，应优先匹配对应时间范围。",
    },
    "土壤": {
        "preferred_platform_id": "tpdc",
        "product_name": "土壤属性协变量（全局优先 TPDC，FTP 优先）",
        "search_keywords": ["土壤", "土壤属性", "土壤质地", "土壤有机质", "pH"],
        "required_dataset_terms": ["土壤", "土壤属性", "土壤质地", "有机质", "pH"],
        "reject_dataset_terms": ["语义分割", "图像分割", "标注样本", "训练样本"],
        "entry_url_by_platform": {"tpdc": "https://data.tpdc.ac.cn/"},
        "temporal_note": "土壤属性多为静态或准静态变量；若用户指定年份，应作为元数据筛选条件。",
    },
    "水文": {
        "preferred_platform_id": "tpdc",
        "product_name": "水文/径流/水系相关数据（全局优先 TPDC，FTP 优先）",
        "search_keywords": ["水文", "径流", "河流", "水系", "湖泊", "冰川", "积雪"],
        "required_dataset_terms": ["水文", "径流", "河流", "水系", "湖泊", "冰川", "积雪"],
        "reject_dataset_terms": ["语义分割", "图像分割", "标注样本", "训练样本"],
        "entry_url_by_platform": {"tpdc": "https://data.tpdc.ac.cn/"},
        "temporal_note": "水文数据通常具有时间范围，应优先匹配用户指定年份或日期。",
    },
    "遥感影像": {
        "preferred_platform_id": "tpdc",
        "product_name": "遥感影像 / 遥感栅格产品（V151 固定只使用 TPDC，FTP 优先）",
        "search_keywords": ["遥感影像", "Landsat", "Sentinel", "MODIS", "高分", "影像"],
        "required_dataset_terms": ["遥感影像", "Landsat", "Sentinel", "MODIS", "高分", "影像"],
        "reject_dataset_terms": ["语义分割", "图像分割", "标注样本", "训练样本"],
        "entry_url_by_platform": {"tpdc": "https://data.tpdc.ac.cn/"},
        "temporal_note": "遥感影像属于动态数据，应匹配目标年份、月份或日期范围。",
    },
}


def is_download_only_request(text: str | None) -> bool:
    t = str(text or "").strip().lower()
    raw = str(text or "")
    if not t:
        return False
    if "下载" not in raw and "获取" not in raw and "准备" not in raw:
        return False
    mapping_words = ["绘制", "制图", "出图", "预测图", "土壤有机质图", "做图"]
    if any(w in raw for w in mapping_words) and "只" not in raw and "仅" not in raw:
        return False
    alias_hit = any(alias.lower() in t for aliases in DATA_ALIASES.values() for alias in aliases)
    generic_hit = any(w in raw for w in GENERIC_DATA_WORDS)
    # V151: 不再把下载任务限定在 DEM。只要用户明确说“下载/获取/准备 + 数据/影像/栅格/产品”，就进入 download_only。
    return bool(alias_hit or generic_hit)


def _parse_year(text: str) -> int | None:
    m = re.search(r"(19\d{2}|20\d{2})", text)
    return int(m.group(1)) if m else None


def _parse_requested_resolution_meters(text: str) -> float | None:
    """Parse user-requested raster resolution. Smaller meter values are finer.

    Policy used later: if the user requests 1 km, 500 m / 250 m / 30 m are
    acceptable because they can be resampled coarser; 2 km is not acceptable
    because it is already coarser than the requested resolution.
    """
    t = str(text or "")
    patterns = [
        (r"(\d+(?:\.\d+)?)\s*(?:km|KM|Km|公里|千米)", 1000.0),
        (r"(\d+(?:\.\d+)?)\s*(?:m|M|米)", 1.0),
    ]
    found: list[float] = []
    for pat, scale in patterns:
        for m in re.finditer(pat, t):
            try:
                val = float(m.group(1)) * scale
                if 0 < val <= 100000:
                    found.append(val)
            except Exception:
                continue
    if not found:
        return None
    return min(found)


def _resolution_terms(res_m: float | None) -> list[str]:
    if not res_m:
        return []
    terms: list[str] = []
    if abs(res_m % 1000) < 1e-6:
        km = res_m / 1000.0
        terms.extend([f"{km:g}km", f"{km:g} km", f"{int(res_m)}m", f"{int(res_m)}米"])
    else:
        terms.extend([f"{res_m:g}m", f"{int(res_m)}米"])
    # Acceptable finer alternatives for resampling. Keep this bounded so a 1 km
    # request can discover 500 m / 250 m data, but not coarser 2 km products.
    for cand in [500, 250, 100, 90, 30]:
        if cand <= res_m and f"{cand}m" not in terms:
            terms.extend([f"{cand}m", f"{cand}米"])
    return terms


def _extract_requested_data_label(text: str) -> str:
    raw = str(text or "")
    cleaned = re.sub(r"(19\d{2}|20\d{2})年?", " ", raw)
    cleaned = re.sub(r"\d+(?:\.\d+)?\s*(?:km|KM|Km|公里|千米|m|M|米)", " ", cleaned)
    cleaned = re.sub(r"保存到\s*[A-Za-z]:\\[^，。；;\n]+", " ", cleaned)
    cleaned = re.sub(r"(?:成都市|四川省|全国|中国|[\u4e00-\u9fa5]{2,12}(?:省|市|区|县|州|盟|旗))", " ", cleaned)
    patterns = [
        r"(?:下载|获取|准备|寻找|查找)([^，。；;\n]{1,32}?)(?:数据集|数据|栅格|影像|产品|协变量|图层)",
        r"(?:下载|获取|准备|寻找|查找)([^，。；;\n]{1,32})",
    ]
    bad_words = ["为我", "帮我", "一个", "一些", "国内", "免费", "平台", "FTP", "使用", "优先", "国家青藏高原科学数据中心"]
    for pat in patterns:
        m = re.search(pat, cleaned)
        if not m:
            continue
        cand = m.group(1).strip()
        for b in bad_words:
            cand = cand.replace(b, "")
        cand = cand.strip(" 的年月日到从和、 ，,。；;：:（）()[]【】")
        if 1 <= len(cand) <= 24:
            return cand
    return "用户指定数据"


def _parse_data_name(text: str) -> str:
    low = str(text or "").lower()
    raw = str(text or "")

    # V176: 专业变量优先。这里不能让“蒸散/湿度”等词被宽泛的“气象”吞掉。
    # 规则：先匹配长专业短语，再匹配通用大类。
    explicit_priority = [
        "潜在蒸散发量", "实际蒸散发量", "地表太阳辐射", "土壤水分",
        "相对湿度", "风速", "气温", "降水", "NDVI", "EVI", "LAI", "FVC", "LULC", "DEM",
    ]
    for name in explicit_priority:
        for alias in DATA_ALIASES.get(name, []):
            a = str(alias or "")
            if a and a.lower() in low:
                return name

    # 如果用户在“下载/获取/准备”后直接给了一个专有数据名，优先保留这个专有名，
    # 例如“潜在蒸散发量数据”“O3数据”“臭氧数据”。
    extracted = _extract_requested_data_label(raw)
    if extracted and extracted != "用户指定数据":
        # 除非明确只是“气象数据/遥感数据/国内数据”等泛称，否则保留用户原始专业词。
        generic = {"气象", "气象数据", "遥感", "遥感数据", "数据", "国内数据", "免费数据"}
        if extracted not in generic and len(extracted) >= 2:
            return extracted

    # Longer aliases first prevents “土壤” eating “土壤水分”. 通用兜底放最后。
    alias_pairs: list[tuple[str, str]] = []
    for name, aliases in DATA_ALIASES.items():
        for a in aliases:
            alias_pairs.append((name, str(a)))
    for name, alias in sorted(alias_pairs, key=lambda x: len(x[1]), reverse=True):
        if alias and alias.lower() in low:
            return name
    return extracted or "用户指定数据"


def _parse_requested_platform(text: str) -> tuple[str | None, str | None]:
    t = str(text or "")
    # Highest confidence: explicit phrases such as “用地理空间数据云下载” or “从全国地质资料馆下载”.
    explicit_patterns = [
        r"(?:使用|用|从|通过|在|指定|选择)([^，。；;\n]{2,40}?)(?:平台)?(?:下载|获取|检索)",
        r"(?:平台|数据平台)(?:为|是|选择|指定)?([^，。；;\n]{2,40})",
    ]
    candidates: list[str] = []
    for pat in explicit_patterns:
        for m in re.finditer(pat, t):
            candidates.append(m.group(1).strip())
    # Fallback: any known platform name appearing in text.
    candidates.extend([name for name in PLATFORM_NAME_ALIASES if name and name in t])
    for cand in candidates:
        for name, pid in sorted(PLATFORM_NAME_ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True):
            if name and name in cand:
                platform = next((p for p in DOMESTIC_FREE_PLATFORMS if p.get("id") == pid), None)
                return pid, (platform or {}).get("name") or name
    return None, None


def _parse_region(text: str) -> str:
    after_year = re.search(r"(?:19\d{2}|20\d{2})年?([\u4e00-\u9fa5]{2,12}(?:省|市|区|县|州|盟|旗))", text)
    if after_year:
        cand = after_year.group(1).lstrip("年在于为给下载获取准备我")
        if cand:
            return cand
    after_verb = re.search(r"(?:下载|获取|准备|寻找|查找)([\u4e00-\u9fa5]{2,12}(?:省|市|区|县|州|盟|旗))", text)
    if after_verb:
        cand = after_verb.group(1).lstrip("年在于为给下载获取准备我")
        if cand:
            return cand
    candidates = []
    for suf in REGION_SUFFIXES:
        candidates += re.findall(r"([\u4e00-\u9fa5]{2,12}" + re.escape(suf) + r")", text)
    bad = {"土壤有机质", "国内平台", "免费数据", "气象数据", "遥感数据"}
    cleaned = []
    for c in candidates:
        c = c.lstrip("年在于为给下载获取准备我")
        if c and c not in bad and not c.endswith("数据"):
            cleaned.append(c)
    if cleaned:
        return sorted(set(cleaned), key=len, reverse=True)[0]
    return "未指定区域"


def _strip_quoted_path(value: str) -> str:
    value = (value or "").strip().strip("。；;,，")
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1].strip()
    return value


def _parse_save_dir(text: str) -> str | None:
    if not ALLOW_CUSTOM_SAVE_DIR:
        return None
    # 用户可在自然语言中指定保存位置。
    # 支持：保存到/下载到/数据保存位置为/保存目录：E:\xxx。
    path_prefix = r"(?:数据保存位置|保存位置|保存路径|保存目录|下载目录|下载保存位置|下载保存目录|保存到|存到|下载到|保存至|存放到|输出到)"
    sep = r"\s*(?:为|是|到|:|：)?\s*"
    patterns = [
        path_prefix + sep + r"([A-Za-z]:\\[^，。；;\n]+)",
        path_prefix + sep + r"([/\\][^，。；;\n]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return _strip_quoted_path(m.group(1))
    return None


def build_account_hint(auth_user: dict[str, Any] | None = None, platform_ids: list[str] | None = None) -> dict[str, Any]:
    """V110 account policy.

    App login credentials are not reused for external platforms. Domestic-platform
    credentials must be configured separately in .env. This keeps the app account
    boundary clean and prevents accidental password leakage.
    """
    user = auth_user or {}
    app_username = str(user.get("username") or "").strip()
    app_contact = str(user.get("contact") or "").strip()
    account_cfg = build_platform_account_config(platform_ids)
    return {
        "account_policy": "env_platform_credentials_only",
        "use_app_account_as_hint": False,
        "app_username_hint": app_username,
        "app_contact_hint": app_contact,
        "login_name_hint": "使用 .env 中配置的平台账号；不复用智能体登录密码。",
        "password_reuse_policy": "forbidden",
        "password_note": "V151 起下载阶段只使用 TPDC。请在项目根目录 .env 中填写 TPDC_USERNAME/TPDC_PASSWORD；验证码、人机验证、实名信息仍由用户完成。",
        "auto_register": account_cfg.get("auto_register", False),
        "platform_account_config": account_cfg,
    }


def parse_download_intent(text: str) -> DownloadIntent:
    data_name = _parse_data_name(text)
    year = _parse_year(text)
    region = _parse_region(text)
    temporal_type = "static" if data_name in STATIC_DATA else "dynamic"
    save_dir = _parse_save_dir(text)
    platform_id, platform_name = _parse_requested_platform(text)
    requested_resolution_meters = _parse_requested_resolution_meters(text)
    return DownloadIntent(
        task_type="download_only",
        request_text=text,
        data_name=data_name,
        region=region,
        year=year,
        temporal_type=temporal_type,
        save_dir=save_dir,
        custom_save_dir=bool(save_dir),
        requested_platform_id=platform_id,
        requested_platform_name=platform_name,
        requested_resolution_meters=requested_resolution_meters,
    )




REGION_PARENT_HINTS: dict[str, list[str]] = {
    "成都市": ["四川省", "全国"],
    "温江区": ["成都市", "四川省", "全国"],
    "简阳市": ["成都市", "四川省", "全国"],
    "邛崃市": ["成都市", "四川省", "全国"],
    "浙江省": ["全国"],
    "四川省": ["全国"],
}


def _build_spatial_query_scales(region: str) -> list[str]:
    """Return acceptable spatial scales for data download.

    Downloading a city-level dataset is not always possible on domestic portals.
    For covariates such as DEM/LULC/NDVI, using a province- or national-scale
    file and clipping later is valid. This list is for searching/downloading,
    not for final AOI clipping.
    """
    r = str(region or "").strip() or "未指定区域"
    out: list[str] = []
    if r and r != "未指定区域":
        out.append(r)
    for x in REGION_PARENT_HINTS.get(r, []):
        if x not in out:
            out.append(x)
    if r.endswith(("市", "区", "县", "州", "盟", "旗")) and "全国" not in out:
        out.append("全国")
    if r.endswith("省") and "全国" not in out:
        out.append("全国")
    if "中国" not in out:
        out.append("中国")
    return out or ["全国", "中国"]

def _platform_score(platform: dict[str, Any], data_name: str, region: str) -> int:
    score = 100 - int(platform.get("priority", 99))
    best = " ".join(platform.get("best_for") or [])
    if data_name and data_name in best:
        score += 50
    rule = AUTO_PRODUCT_RULES.get(data_name) or {}
    if rule.get("preferred_platform_id") == platform.get("id"):
        score += 100
    if platform.get("id") == "tpdc":
        # V151: 全数据类型默认优先 TPDC，而不是只对 DEM 加权。
        score += 1000
    if data_name == "气象" and platform.get("id") in {"cma_nmic", "nsmc"}:
        score += 50
    if data_name == "遥感影像" and platform.get("id") in {"noda", "gscloud"}:
        score += 50
    if any(k in region for k in ["西藏", "青海", "青藏", "高原"]) and platform.get("id") == "tpdc":
        score += 50
    return score


def select_auto_product(intent: DownloadIntent) -> dict[str, Any]:
    rule = AUTO_PRODUCT_RULES.get(intent.data_name, {})
    # V151: TPDC 搜索框很死板。实际搜索关键词只能是数据名称，
    # 例如“地表太阳辐射”，不能把年份、区域、分辨率一起塞进去。
    # 区域/年份/分辨率保留给搜索结果评分与筛选使用。
    keywords = list(rule.get("search_keywords") or [intent.data_name])
    spatial_scales = _build_spatial_query_scales(intent.region)
    # Static data such as DEM do not have to be annual. Do not push the year
    # into the search keyword, otherwise portals may return irrelevant yearly
    # products instead of the static DEM product.
    # V151: user may mention another portal, but this build deliberately supports only TPDC.
    preferred_pid = "tpdc"
    if not rule:
        rule = {
            "preferred_platform_id": "tpdc",
            "product_name": f"{intent.data_name}（用户指定数据；全局优先 TPDC，FTP 优先）",
            "search_keywords": [intent.data_name],
            "required_dataset_terms": [intent.data_name],
            "reject_dataset_terms": ["语义分割", "图像分割", "semantic segmentation", "segmentation", "标注样本", "训练样本"],
            "entry_url_by_platform": {"tpdc": "https://data.tpdc.ac.cn/"},
        }
    # V151: TPDC 检索词必须短而准。不要把“成都市/四川省/全国/1km/500m/2010”
    # 全塞进搜索框，否则平台会返回空结果或跑偏。区域、年份、分辨率只用于后续评分。
    primary_search_keyword = str((rule.get("search_keywords") or [intent.data_name])[0] or intent.data_name).strip()
    tpdc_direct_search_keyword = str(intent.data_name or primary_search_keyword).strip() or primary_search_keyword
    return {
        "auto_selected": True,
        "data_name": intent.data_name,
        "product_name": rule.get("product_name") or f"{intent.data_name} 国内免费数据产品",
        "preferred_platform_id": preferred_pid,
        "platform_selection_mode": "tpdc_only",
        "requested_platform_name": "国家青藏高原科学数据中心",
        "search_keywords": [tpdc_direct_search_keyword],
        "keyword_string": tpdc_direct_search_keyword,
        "tpdc_direct_search_keyword": tpdc_direct_search_keyword,
        "primary_search_keyword": primary_search_keyword,
        "spatial_query_scales": spatial_scales,
        "spatial_strategy": "target_then_parent_or_national",
        "spatial_note": "下载阶段允许在 TPDC 中使用省级或全国尺度数据，后续再按目标 AOI 裁剪；不要强求平台一定存在市/区县级成品。",
        "temporal_note": rule.get("temporal_note") or "按用户任务年份与区域优先匹配 TPDC 免费数据；若数据集时间范围覆盖目标年份，例如 1985–2024 覆盖 2020，则允许下载，后续只保留目标年份。",
        "requested_resolution_meters": intent.requested_resolution_meters,
        "resolution_terms": _resolution_terms(intent.requested_resolution_meters),
        "resolution_policy": "若用户指定分辨率，只允许选择等于或更精细的数据；例如 1 km 可用 500 m/250 m 后重采样，禁止选择 2 km 等更粗数据。",
        "strict_dataset_match": {
            "required_terms": rule.get("required_dataset_terms") or [intent.data_name],
            "reject_terms": rule.get("reject_dataset_terms") or [],
            "whole_word_required_for_ascii_terms": True,
        },
        "download_priority": "V151 只使用国家青藏高原科学数据中心 https://data.tpdc.ac.cn/；强制本机 Google Chrome；不再跳转到地理空间数据云或其他平台。若页面提供 FTP，则优先 FTP 下载。",
        "entry_url_by_platform": rule.get("entry_url_by_platform") or {},
        "user_selection_required": False,
    }


def select_platform_candidates(intent: DownloadIntent) -> list[dict[str, Any]]:
    product = select_auto_product(intent)
    # V151 hard constraint: download-only mode uses TPDC only.
    # Previous versions still kept GSCloud/Geodata/RESDC in the internal fallback
    # queue, which is why the browser could jump to 地理空间数据云. That behavior is
    # now removed: no hidden fallback platform, no alternate URL, no platform loop.
    tpdc = next((p for p in DOMESTIC_FREE_PLATFORMS if p.get("id") == "tpdc"), None)
    if not tpdc:
        return []
    q = dict(tpdc)
    q["score"] = 9999
    q["selection_reason"] = "tpdc_only_hard_policy"
    q["domestic"] = True
    q["free_only_allowed"] = True
    q["auto_selected_product"] = product.get("product_name")
    q["search_keywords"] = product.get("search_keywords")
    q["forced_single_platform"] = True
    q["disabled_fallback_platforms"] = [p.get("id") for p in DOMESTIC_FREE_PLATFORMS if p.get("id") != "tpdc"]
    return [q]


def _safe_name(s: Any) -> str:
    txt = str(s or "unknown")
    txt = re.sub(r"[\\/:*?\"<>|\s]+", "_", txt)
    return txt.strip("_")[:60] or "unknown"


def default_download_work_dir(intent: DownloadIntent) -> Path:
    year = str(intent.year or "unknown_year")
    name = f"{year}_{_safe_name(intent.region)}_{_safe_name(intent.data_name)}"
    # Each download instruction receives a fresh shallow job directory to avoid
    # showing stale FTP errors/progress from a previous attempt with the same request.
    job = time.strftime("%Y%m%d_%H%M%S") + "_" + f"{int(time.time_ns()) % 1000000:06d}"
    return DEFAULT_SAVE_ROOT / name / job


def raw_save_dir(intent: DownloadIntent) -> Path:
    if intent.custom_save_dir and intent.save_dir:
        return Path(intent.save_dir)
    return default_download_work_dir(intent) / "raw"


def manifest_path_for(intent: DownloadIntent) -> Path:
    if intent.custom_save_dir and intent.save_dir:
        return Path(intent.save_dir) / "download_manifest.json"
    return default_download_work_dir(intent) / "meta" / "download_manifest.json"


def _path_warning(path: Path) -> str | None:
    s = str(path)
    if os.name == "nt" and len(s) > 180:
        return "保存路径较长，Windows 可能触发路径长度限制；建议改用 E:\\DSM_DL 或 E:\\Agent_DSM\\dl 等短路径。"
    if len(s) > 180:
        return "保存路径较长；建议使用更短路径。"
    return None


def make_manual_handoff_payload(intent: DownloadIntent, manifest_path: str | None = None, auth_user: dict[str, Any] | None = None) -> dict[str, Any]:
    platforms = select_platform_candidates(intent)
    product = select_auto_product(intent)
    raw_dir = raw_save_dir(intent)
    manual_dir = DEFAULT_MANUAL_ROOT
    raw_dir.mkdir(parents=True, exist_ok=True)
    manual_dir.mkdir(parents=True, exist_ok=True)
    preferred = platforms[0] if platforms else {}
    # V151: candidate platforms are an internal fallback queue. The user-facing
    # handoff should expose only the currently selected/recommended platform,
    # not the whole ranked queue. The full queue remains in download_manifest.json
    # for the backend browser orchestrator.
    visible_platforms = [preferred] if preferred else []
    return {
        "kind": "tpdc_direct_browser_launch",
        "suppress_modal": True,
        "title": "TPDC 数据检索已启动",
        "message": (
            f"已识别下载任务：{intent.year or '未指定年份'} 年 {intent.region} {intent.data_name}。"
            + "平台已固定为国家青藏高原科学数据中心 TPDC，不再尝试地理空间数据云或其他平台。"
            + "系统会直接启动 TPDC 登录与检索流程：优先使用 Microsoft Edge；系统自动填写 .env 中配置的账号密码，验证码、人机验证、协议确认和具体数据集选择由用户在浏览器页面完成。"
        ),
        "capture_dir": str(raw_dir),
        "manual_dir": str(manual_dir),
        "manifest_path": manifest_path or "",
        "request": asdict(intent),
        "custom_save_dir": bool(intent.custom_save_dir),
        "path_warning": _path_warning(raw_dir),
        "auto_selected_product": product,
        "recommended_platform": preferred,
        "spatial_query_scales": product.get("spatial_query_scales") or _build_spatial_query_scales(intent.region),
        "platform_try_policy": "tpdc_only_no_fallback",
        "platforms": visible_platforms,
        "internal_platform_candidates_count": len(platforms),
        "platform_candidates_hidden": True,
        "account_hint": build_account_hint(auth_user, [str(p.get("id")) for p in visible_platforms]),
        "true_link_rules": [
            "浏览器出现真实下载任务，才算下载已启动。",
            "FTP 工具开始传输，才算下载已启动。",
            "打开搜索页、详情页、登录页、验证码页，不算下载已启动。",
            "下载到几 KB 的 HTML/JS/CSS，不算真实数据。",
            "明确收费、积分购买、定制报价的数据必须排除。",
        ],
        "next_steps": [
            "无需在智能体弹窗中操作；AI 接收到下载指令后会直接打开 TPDC 登录页并填写账号密码。",
            "若页面出现验证码、人机验证、实名确认或协议确认，只需要你在弹出的浏览器窗口中完成该验证。",
            "若页面弹出 FTP 主机、端口、用户名、密码，智能体后台捕捉器会自动识别并启动 FTP 下载；无需在智能体界面手动填写。",
            "若平台最终要求人工点击下载按钮，你只需要点击该下载按钮。",
        ],
        "buttons": {
            "launch_browser_capture": True,
            "confirm_started": True,
            "confirm_completed": True,
        },
    }


def write_manifest(intent: DownloadIntent, status: str, download_mode: str, extra: dict[str, Any] | None = None, auth_user: dict[str, Any] | None = None) -> dict[str, Any]:
    raw_dir = raw_save_dir(intent)
    meta_path = manifest_path_for(intent)
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    product = select_auto_product(intent)
    platforms = select_platform_candidates(intent)
    manifest = {
        "task_type": "download_only",
        "request_text": intent.request_text,
        "region": intent.region,
        "year": intent.year,
        "data_name": intent.data_name,
        "requested_resolution_meters": intent.requested_resolution_meters,
        "resolution_policy": "requested_resolution_meters 是最大允许像元大小；实际数据必须等于或更精细，后续可重采样到目标分辨率。",
        "temporal_type": intent.temporal_type,
        "temporal_policy": "用户只要某一年时，允许下载覆盖该年份的多年数据集；后处理阶段必须只保留目标年份。例如用户要 2020 年，1985-2024 数据集可下载，最终筛选 2020。",
        "platform_scope": "domestic_only",
        "free_only": True,
        "free_status": "free_or_free_after_login_required",
        "payment_required": False,
        "status": status,
        "download_mode": download_mode,
        "ready_for_modeling": False,
        "custom_save_dir": intent.custom_save_dir,
        "local_save_dir": str(raw_dir),
        "manual_download_dir": str(DEFAULT_MANUAL_ROOT),
        "manifest_path": str(meta_path),
        "auto_selected_product": product,
        "spatial_query_scales": product.get("spatial_query_scales") or _build_spatial_query_scales(intent.region),
        "spatial_download_strategy": "target_region_parent_province_or_national_then_clip",
        "platform_try_policy": "tpdc_only_no_fallback",
        "requested_platform_id": "tpdc",
        "requested_platform_name": "国家青藏高原科学数据中心",
        "user_requested_platform_ignored": intent.requested_platform_id not in {None, "tpdc"},
        "user_requested_platform_original": intent.requested_platform_name,
        "selected_platform": platforms[0] if platforms else {},
        "visible_selected_platform": platforms[0] if platforms else {},
        "platform_candidates": platforms,
        "platform_candidates_hidden_from_user": True,
        "account_hint": build_account_hint(auth_user, [str(p.get("id")) for p in platforms]),
        "path_warning": _path_warning(raw_dir),
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
    }
    if extra:
        manifest.update(extra)
    meta_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def prepare_download_only_task(text: str, auth_user: dict[str, Any] | None = None) -> dict[str, Any]:
    intent = parse_download_intent(text)
    manifest = write_manifest(intent, status="manual_handoff_needed", download_mode="manual_handoff", auth_user=auth_user)
    handoff = make_manual_handoff_payload(intent, manifest_path=manifest.get("manifest_path"), auth_user=auth_user)
    manifest["handoff_payload"] = handoff
    Path(manifest["manifest_path"]).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        state_path = Path(manifest.get("local_save_dir") or raw_save_dir(intent)) / "ftp_watcher_state.json"
        state_path.write_text(json.dumps({
            "status": "ftp_prepare_pending",
            "ftp_watcher_status": "not_started",
            "message": "已创建新的下载任务，等待打开TPDC并捕捉FTP参数。",
            "updated_at": int(time.time()),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return {"intent": asdict(intent), "manifest": manifest, "handoff": handoff}


def _load_manifest(manifest_path: str | None) -> dict[str, Any] | None:
    if not manifest_path:
        return None
    p = Path(manifest_path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    path = manifest.get("manifest_path")
    if not path:
        local = manifest.get("local_save_dir") or str(DEFAULT_SAVE_ROOT / "unknown")
        path = str(Path(local) / "download_manifest.json")
        manifest["manifest_path"] = path
    manifest["updated_at"] = int(time.time())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def mark_download_started(manifest_path: str | None, mode: str = "manual_handoff") -> dict[str, Any]:
    manifest = _load_manifest(manifest_path) or {}
    if not manifest:
        manifest = {
            "task_type": "download_only",
            "platform_scope": "domestic_only",
            "free_only": True,
            "ready_for_modeling": False,
        }
    manifest.update({
        "status": "download_started",
        "download_mode": mode,
        "confirmed_by_user": True,
        "ready_for_modeling": False,
    })
    return _save_manifest(manifest)


def _looks_like_real_data_file(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "not_file"
    ext = path.suffix.lower()
    try:
        size = path.stat().st_size
    except Exception:
        size = 0
    if ext in BLOCKED_WEB_EXTS:
        return False, "web_or_preview_file"
    if size < int(os.getenv("PRO_DOWNLOAD_MIN_REAL_FILE_BYTES", "10240")):
        return False, "too_small"
    if ext not in DOMESTIC_DOWNLOAD_EXTS:
        return False, "unsupported_extension"
    try:
        head = path.read_bytes()[:512].lower()
        if b"<html" in head or b"<!doctype html" in head or b"login" in head or b"javascript" in head:
            return False, "html_or_web_content"
    except Exception:
        pass
    if ext in {".tif", ".tiff", ".img", ".vrt"} and rasterio is not None:
        try:
            with rasterio.open(path) as ds:
                if ds.width > 0 and ds.height > 0:
                    return True, "rasterio_open_check"
        except Exception as exc:
            return False, f"rasterio_failed:{exc}"
    return True, "extension_size_signature_check"


def scan_downloaded_files(manifest_path: str | None) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path) or {}
    dirs = []
    if manifest.get("local_save_dir"):
        dirs.append(Path(str(manifest.get("local_save_dir"))))
    if manifest.get("manual_download_dir"):
        dirs.append(Path(str(manifest.get("manual_download_dir"))))
    dirs.append(DEFAULT_MANUAL_ROOT)
    seen = set()
    validated = []
    rejected = []
    for d in dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            for p in d.glob("**/*"):
                key = str(p.resolve())
                if key in seen or not p.is_file():
                    continue
                seen.add(key)
                if p.name.lower().endswith("download_manifest.json"):
                    continue
                ok, method = _looks_like_real_data_file(p)
                rec = {"path": str(p), "size": p.stat().st_size if p.exists() else 0, "validation": method}
                if ok:
                    validated.append(rec)
                else:
                    rejected.append(rec)
        except Exception as exc:
            rejected.append({"dir": str(d), "validation": f"scan_failed:{exc}"})
    if validated:
        manifest.update({
            "status": "download_verified",
            "ready_for_modeling": True,
            "validated_files": validated,
            "rejected_files": rejected[:50],
            "validation_method": "local_file_scan",
        })
    else:
        manifest.update({
            "status": "download_pending",
            "ready_for_modeling": False,
            "validated_files": [],
            "rejected_files": rejected[:50],
            "validation_method": "local_file_scan_no_valid_file_yet",
        })
    return _save_manifest(manifest)


def launch_capture_browser(manifest_path: str | None) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    if not manifest:
        return {"ok": False, "error": "未找到 download_manifest.json，无法启动TPDC 浏览器。"}
    local_save_dir = Path(str(manifest.get("local_save_dir") or DEFAULT_MANUAL_ROOT))
    local_save_dir.mkdir(parents=True, exist_ok=True)
    product = manifest.get("auto_selected_product") or {}
    candidates = manifest.get("platform_candidates") or []
    platform = candidates[0] if candidates else {}
    platform_ids = ["tpdc"]
    platform_id = "tpdc"
    entry_by_platform = product.get("entry_url_by_platform") or {}
    url = entry_by_platform.get(str(platform_id)) or platform.get("url") or ""
    keyword = product.get("tpdc_direct_search_keyword") or product.get("primary_search_keyword") or manifest.get("data_name") or product.get("keyword_string") or " ".join(product.get("search_keywords") or []) or "DEM"
    live_dir = local_save_dir / "_browser_live"
    live_dir.mkdir(parents=True, exist_ok=True)
    live_screenshot = live_dir / "live.png"
    live_state = live_dir / "state.json"
    tool_path = Path(__file__).resolve().parents[1] / "tools" / "playwright_domestic_download_capture.py"
    if not tool_path.exists():
        return {"ok": False, "error": f"TPDC 浏览器脚本不存在：{tool_path}"}
    account_cfg = build_platform_account_config([str(platform_id)])
    cred_summary = get_platform_credentials(str(platform_id))
    cmd = [
        sys.executable,
        str(tool_path),
        "--platform", str(platform_id),
        "--platform-sequence", ",".join(platform_ids),
        "--download-root", str(local_save_dir),
        "--keyword", str(keyword),
        "--timeout", str(int(os.getenv("PRO_DOMESTIC_BROWSER_CAPTURE_TIMEOUT", "86400"))),
        "--session-dir", str(account_cfg.get("session_dir") or r"E:\Agent_DSM\runs\sessions"),
        "--browser-channel", str(os.getenv("DOMESTIC_BROWSER_CHANNEL", "msedge")),
        "--data-name", str(manifest.get("data_name") or product.get("data_name") or ""),
        "--requested-resolution-meters", str(manifest.get("requested_resolution_meters") or ""),
        "--requested-year", str(manifest.get("year") or ""),
        "--live-screenshot", str(live_screenshot),
        "--state-json", str(live_state),
    ]
    if account_cfg.get("use_env_credentials") and account_cfg.get("auto_login"):
        cmd += ["--auto-login"]
    browser_exe = os.getenv("DOMESTIC_BROWSER_EXECUTABLE_PATH", "").strip() or os.getenv("DOMESTIC_CHROME_EXECUTABLE_PATH", "").strip() or os.getenv("DOMESTIC_EDGE_EXECUTABLE_PATH", "").strip()
    if browser_exe:
        cmd += ["--browser-executable", browser_exe]
    if url and len(platform_ids) == 1:
        cmd += ["--url", str(url)]
    env = os.environ.copy()
    env.setdefault("DOMESTIC_BROWSER_CHANNEL", "msedge")
    env.setdefault("DOMESTIC_BROWSER_STRICT_EDGE", "1")
    env.setdefault("DOMESTIC_EDGE_ONLY", "1")
    env.setdefault("DOMESTIC_CHROME_ONLY", "0")
    env["DOMESTIC_CAPTURE_PLATFORM_ID"] = str(platform_id)
    env["DOMESTIC_CAPTURE_AUTO_LOGIN"] = "1" if (account_cfg.get("use_env_credentials") and account_cfg.get("auto_login")) else "0"
    env.setdefault("DOMESTIC_FTP_AUTO_DOWNLOAD", "0")
    try:
        # Do not wait for the long-running browser process.
        proc = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parents[1]), env=env)
        manifest.update({
            "status": "download_started",
            "download_mode": "edge_browser_auto_ftp_capture",
            "platform_sequence": platform_ids,
            "platform_selection_visibility": "tpdc_only_no_fallback",
            "browser_capture_pid": proc.pid,
            "browser_capture_command": cmd,
            "browser_live_screenshot": str(live_screenshot),
            "browser_live_state": str(live_state),
            "platform_account_config": account_cfg,
            "selected_platform_credentials": cred_summary,
            "ready_for_modeling": False,
            "ftp_auto_capture_enabled": True,
            "ftp_auto_download_enabled": os.getenv("DOMESTIC_FTP_AUTO_DOWNLOAD", "0"),
        })
        _save_manifest(manifest)
        return {
            "ok": True,
            "pid": proc.pid,
            "platform": platform.get("name") or platform_id,
            "platform_sequence": platform_ids,
            "platform_selection_visibility": "tpdc_only_no_fallback",
            "save_dir": str(local_save_dir),
            "manifest_path": manifest.get("manifest_path"),
            "live_screenshot": str(live_screenshot),
            "live_state": str(live_state),
            "auto_login": bool(account_cfg.get("use_env_credentials") and account_cfg.get("auto_login")),
            "browser": os.getenv("DOMESTIC_BROWSER_CHANNEL", "chrome"),
            "credential_status": cred_summary,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "command": cmd}




def _safe_ftp_value(value: str | None) -> str:
    return str(value or "").strip().replace("\r", "").replace("\n", "")


def _mask_ftp_secret(value: str | None) -> str:
    v = str(value or "")
    if not v:
        return ""
    if len(v) <= 2:
        return "*" * len(v)
    return v[:1] + "*" * max(2, len(v) - 2) + v[-1:]


def _ftp_ticket_id(manifest: dict[str, Any] | None, primary: dict[str, Any] | None) -> str:
    """Create a stable-ish dataset-level ticket id for TPDC FTP downloads."""
    m = manifest or {}
    primary = primary or {}
    bits = [
        str(m.get("data_name") or "data"),
        str(m.get("year") or "unknown_year"),
        str(m.get("region") or "unknown_region"),
        str(primary.get("host") or "ftp"),
        str(primary.get("username") or "user"),
    ]
    base = _safe_name("_".join(bits))[:96]
    return f"tpdc_{base}_{int(time.time())}"


def _redact_ftp_accounts(accounts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for a in accounts or []:
        b = dict(a)
        if "password" in b:
            b["password_masked"] = _mask_ftp_secret(str(b.get("password") or ""))
            b.pop("password", None)
        out.append(b)
    return out


def _write_ftp_ticket_files(
    download_dir: Path,
    accounts: list[dict[str, Any]],
    manifest: dict[str, Any] | None = None,
    source: str = "manual_or_captured",
) -> dict[str, str]:
    """Write dataset-level TPDC FTP ticket files.

    TPDC often generates a different FTP host/port/user/password for each
    dataset/order. Treat each account block as a dataset-level download ticket,
    not as a reusable platform account. The public ticket masks passwords; the
    local secret ticket is for this machine only and should not be shared.
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    accounts = list(accounts or [])
    primary = accounts[0] if accounts else {}
    ticket_id = _ftp_ticket_id(manifest, primary)
    ticket_dir = download_dir / "ftp_tickets"
    ticket_dir.mkdir(parents=True, exist_ok=True)
    m = manifest or {}
    public_ticket = {
        "ticket_id": ticket_id,
        "platform": "国家青藏高原科学数据中心（TPDC）",
        "ticket_scope": "dataset_level_ftp_ticket",
        "source": source,
        "dataset_name": ((m.get("auto_selected_product") or {}).get("product_name") if isinstance(m.get("auto_selected_product"), dict) else None) or m.get("data_name") or "unknown_dataset",
        "variable": m.get("data_name"),
        "target_region": m.get("region"),
        "target_year": m.get("year"),
        "local_save_dir": str(download_dir),
        "status": "ticket_created",
        "ready_for_modeling": False,
        "accounts_count": len(accounts),
        "accounts": _redact_ftp_accounts(accounts),
        "password_policy": "public ticket masks FTP password; local secret file is machine-local and must not be shared",
        "created_at": int(time.time()),
    }
    secret_ticket = dict(public_ticket)
    secret_ticket["accounts"] = accounts
    secret_ticket["password_policy"] = "contains full TPDC dataset FTP password; keep local only"
    public_path = ticket_dir / f"{ticket_id}.json"
    secret_path = ticket_dir / f"{ticket_id}.secret.json"
    latest_path = ticket_dir / "ftp_ticket_latest.json"
    public_path.write_text(json.dumps(public_ticket, ensure_ascii=False, indent=2), encoding="utf-8")
    secret_path.write_text(json.dumps(secret_ticket, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(public_ticket, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "ftp_ticket_id": ticket_id,
        "ftp_ticket_path": str(public_path),
        "ftp_ticket_latest_path": str(latest_path),
        "ftp_ticket_secret_path": str(secret_path),
        "ftp_ticket_dir": str(ticket_dir),
    }


def _write_manual_ftp_handoff_files(
    download_dir: Path,
    accounts: list[dict[str, Any]],
    manifest: dict[str, Any] | None = None,
    source: str = "manual_user_input",
) -> dict[str, str]:
    """Write FTP account JSON, dataset-level ticket, and fallback client scripts."""
    download_dir.mkdir(parents=True, exist_ok=True)
    primary = accounts[0] if accounts else {}
    accounts_path = download_dir / "ftp_accounts_detected.json"
    accounts_path.write_text(json.dumps({"source": source, "primary": primary, "accounts": accounts}, ensure_ascii=False, indent=2), encoding="utf-8")
    ticket_files = _write_ftp_ticket_files(download_dir, accounts, manifest=manifest, source=source)

    note_path = download_dir / "ftp_download_note.txt"
    safe_lines = [
        "TPDC FTP 下载账号（由用户在智能体弹窗中手动填写）",
        "说明：若 Python 自动下载中断，可用 WinSCP/FileZilla/FTP Rush 接管。",
        "",
    ]
    for i, a in enumerate(accounts, start=1):
        safe_lines += [
            f"[{i}] 主机: {a.get('host')}",
            f"    端口: {a.get('port')}",
            f"    用户名: {a.get('username')}",
            f"    密码: {a.get('password')}",
            "",
        ]
    note_path.write_text("\n".join(safe_lines), encoding="utf-8")

    try:
        from urllib.parse import quote
        ftp_url = f"ftp://{quote(str(primary.get('username') or ''), safe='')}:{quote(str(primary.get('password') or ''), safe='')}@{primary.get('host')}:{primary.get('port')}/" if primary else ""
    except Exception:
        ftp_url = ""

    open_ftp_bat = download_dir / "open_ftp_client.bat"
    open_ftp_bat.write_text(
        "@echo off\r\n"
        "chcp 65001 >nul\r\n"
        "echo 正在打开系统 FTP URL。如浏览器不支持大文件续传，请改用 WinSCP/FileZilla。\r\n"
        f"start \"\" \"{ftp_url}\"\r\n"
        "pause\r\n",
        encoding="utf-8",
    )

    winscp_script = download_dir / "winscp_download_script.txt"
    local_dir = download_dir / "files"
    local_dir.mkdir(parents=True, exist_ok=True)
    if ftp_url:
        winscp_script.write_text(
            "option batch continue\n"
            "option confirm off\n"
            f"open {ftp_url}\n"
            f"lcd \"{local_dir}\"\n"
            "get -resume *\n"
            "exit\n",
            encoding="utf-8",
        )
    open_winscp_bat = download_dir / "open_winscp_download.bat"
    open_winscp_bat.write_text(
        "@echo off\r\n"
        "chcp 65001 >nul\r\n"
        "where winscp.com >nul 2>nul\r\n"
        "if errorlevel 1 (\r\n"
        "  echo 未找到 winscp.com。请安装 WinSCP，或用 FileZilla 手动填写 ftp_accounts_detected.json 中的信息。\r\n"
        "  pause\r\n"
        "  exit /b 1\r\n"
        ")\r\n"
        f"winscp.com /script=\"{winscp_script}\"\r\n"
        "pause\r\n",
        encoding="utf-8",
    )
    out = {
        "accounts_path": str(accounts_path),
        "note_path": str(note_path),
        "open_ftp_client_bat": str(open_ftp_bat),
        "winscp_script": str(winscp_script),
        "open_winscp_download_bat": str(open_winscp_bat),
    }
    out.update(ticket_files)
    return out


def _normalize_manual_save_dir(save_dir_override: str | None, fallback: Path) -> Path:
    """Normalize user-supplied FTP save directory without silently discarding Windows paths."""
    raw = str(save_dir_override or "").strip().strip('"').strip("'")
    if not raw:
        return fallback
    # Keep Windows drive paths as provided on Windows; Path also accepts them there.
    try:
        p = Path(raw).expanduser()
        return p
    except Exception:
        return fallback


def start_manual_ftp_download(
    manifest_path: str | None,
    host: str | None,
    port: str | int | None,
    username: str | None,
    password: str | None,
    backup_host: str | None = "",
    save_dir_override: str | None = None,
) -> dict[str, Any]:
    """Start Python FTP download from user-supplied TPDC FTP parameters.

    V151: keep automatic capture as optional, but make manual input a reliable
    primary path. This function now writes immediate status/log files before
    validation so the UI and console can prove the click reached the backend.
    """
    print(f"[MANUAL_FTP_SERVICE] request manifest={manifest_path} host={host} backup_ignored={backup_host} port={port} username={username}", flush=True)
    manifest = _load_manifest(manifest_path)
    if not manifest:
        return {"ok": False, "error": "未找到 download_manifest.json，无法启动 FTP 下载。"}
    h = _safe_ftp_value(host)
    bh = ""  # V232: backup host disabled; use one fixed/primary host only.
    u = _safe_ftp_value(username)
    pw = _safe_ftp_value(password)
    raw_port = _safe_ftp_value(str(port or ""))
    default_save_dir = Path(str(manifest.get("local_save_dir") or DEFAULT_MANUAL_ROOT))
    local_save_dir = _normalize_manual_save_dir(save_dir_override, default_save_dir)
    live_state = Path(str(manifest.get("browser_live_state") or (local_save_dir / "_browser_live" / "state.json")))
    live_state.parent.mkdir(parents=True, exist_ok=True)
    try:
        request_probe = {
            "status": "manual_ftp_button_clicked",
            "message": "已收到用户点击：开始 FTP 下载，正在校验主机/端口/用户名/密码。",
            "host": h,
            "backup_host": bh,
            "port": raw_port,
            "username": u,
            "password_masked": _mask_ftp_secret(pw),
            "save_dir": str(local_save_dir),
            "updated_at": int(time.time()),
        }
        live_state.write_text(json.dumps(request_probe, ensure_ascii=False, indent=2), encoding="utf-8")
        (local_save_dir / "manual_ftp_start_request.json").write_text(json.dumps(request_probe, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[MANUAL_FTP_SERVICE] failed to write initial status: {exc}", flush=True)
    if not h:
        return {"ok": False, "error": "请填写 FTP 主机，例如 ftp2.tpdc.ac.cn。"}
    if not raw_port:
        return {"ok": False, "error": "请填写 FTP 端口，例如 6201。"}
    if not re.fullmatch(r"\d{1,5}", raw_port):
        return {"ok": False, "error": "FTP 端口必须是数字，例如 6201。"}
    port_i = int(raw_port)
    if port_i <= 0 or port_i > 65535:
        return {"ok": False, "error": "FTP 端口超出合法范围。"}
    if not u:
        return {"ok": False, "error": "请填写 FTP 用户名，例如 download_81747713。"}
    if not pw:
        return {"ok": False, "error": "请填写 FTP 密码。"}

    accounts: list[dict[str, Any]] = [{"host": h, "port": port_i, "username": u, "password": pw, "source": "manual_user_input_single_primary", "priority": 1}]

    local_save_dir.mkdir(parents=True, exist_ok=True)
    files = _write_manual_ftp_handoff_files(local_save_dir, accounts, manifest=manifest, source="manual_user_input")
    live_state.parent.mkdir(parents=True, exist_ok=True)
    try:
        state = json.loads(live_state.read_text(encoding="utf-8") or "{}") if live_state.exists() else {}
    except Exception:
        state = {}
    state.update({
        "status": "manual_ftp_credentials_received",
        "ftp_watcher_status": "manual_credentials_received",
        "ftp_accounts_count": len(accounts),
        "message": "已收到用户手动填写的 FTP 主机/端口/用户名/密码，正在启动单主机 Python FTP 下载进程。",
        "manual_ftp_primary": {k: v for k, v in accounts[0].items() if k != "password"},
        "manual_ftp_password_masked": _mask_ftp_secret(pw),
        "updated_at": int(time.time()),
    })
    live_state.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    watcher = Path(__file__).resolve().parents[1] / "tools" / "ftp_clipboard_watcher.py"
    if not watcher.exists():
        return {"ok": False, "error": f"FTP 下载脚本不存在：{watcher}"}
    cmd = [
        sys.executable,
        str(watcher),
        "--run-download-from-json", str(files["accounts_path"]),
        "--download-dir", str(local_save_dir),
        "--requested-year", str(manifest.get("year") or ""),
        "--state-json", str(live_state),
    ]
    try:
        status_seed = {
            "status": "ftp_download_process_starting",
            "message": "后台 Python FTP 下载进程正在启动。若文件很大，连接和列目录可能需要较长时间。",
            "primary_host": h,
            "backup_host": bh,
            "download_dir": str(local_save_dir),
            "accounts_path": files.get("accounts_path"),
            "updated_at": int(time.time()),
        }
        try:
            (local_save_dir / "ftp_download_status.json").write_text(json.dumps(status_seed, ensure_ascii=False, indent=2), encoding="utf-8")
            with (local_save_dir / "ftp_python_downloader.log").open("a", encoding="utf-8") as logf:
                logf.write(json.dumps({"event": "manual_ftp_process_starting", **status_seed}, ensure_ascii=False) + "\n")
        except Exception:
            pass
        # Do not deliberately detach the downloader. The user expects a full
        # app exit to stop the background FTP transfer. Track the child process
        # and terminate it on normal interpreter shutdown.
        proc = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parents[1]))
        _MANUAL_FTP_CHILDREN.append(proc)
        print(f"[MANUAL_FTP_SERVICE] started downloader pid={proc.pid} host={h} dir={local_save_dir}", flush=True)
        manifest.update({
            "status": "manual_ftp_download_started",
            "download_mode": "manual_ftp_python",
            "local_save_dir": str(local_save_dir),
            "manual_ftp_download_pid": proc.pid,
            "manual_ftp_download_command": cmd,
            "ftp_accounts_count": len(accounts),
            "ftp_handoff_files": files,
            "ftp_ticket_manager": {
                "enabled": True,
                "ticket_scope": "dataset_level",
                "ticket_id": files.get("ftp_ticket_id"),
                "ticket_path": files.get("ftp_ticket_path"),
                "ticket_latest_path": files.get("ftp_ticket_latest_path"),
                "ticket_dir": files.get("ftp_ticket_dir"),
                "status": "ticket_created_and_download_started",
            },
            "ftp_primary_account": {k: v for k, v in accounts[0].items() if k != "password"},
            "browser_live_state": str(live_state),
            "ready_for_modeling": False,
        })
        _save_manifest(manifest)
        return {"ok": True, "pid": proc.pid, "download_dir": str(local_save_dir), "accounts_count": len(accounts), "primary_host": h, "backup_host": bh, "files": files, "state_json": str(live_state), "command": cmd}
    except Exception as exc:
        print(f"[MANUAL_FTP_SERVICE][ERROR] failed to start downloader: {exc}", flush=True)
        try:
            (local_save_dir / "ftp_download_status.json").write_text(json.dumps({"status": "ftp_download_process_start_failed", "error": str(exc), "updated_at": int(time.time())}, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return {"ok": False, "error": str(exc), "command": cmd, "files": files}


def launch_ftp_capture_now(manifest_path: str | None) -> dict[str, Any]:
    """Start an immediate FTP watcher for the current TPDC Chrome session.

    V151: this is a manual fallback button. It does not touch TPDC navigation;
    it only reads the existing CDP endpoint and clipboard, then starts the
    passive watcher so the user gets visible feedback after the FTP dialog pops.
    """
    manifest = _load_manifest(manifest_path)
    if not manifest:
        return {"ok": False, "error": "未找到 download_manifest.json，无法启动 FTP 捕捉器。"}
    local_save_dir = Path(str(manifest.get("local_save_dir") or DEFAULT_MANUAL_ROOT))
    live_state = Path(str(manifest.get("browser_live_state") or (local_save_dir / "_browser_live" / "state.json")))
    endpoint = ""
    browser_pid = ""
    live_screenshot = Path(str(manifest.get("browser_live_screenshot") or (local_save_dir / "_browser_live" / "live.png")))
    try:
        if live_state.exists():
            state = json.loads(live_state.read_text(encoding="utf-8") or "{}")
            launch_result = state.get("launch_result") or {}
            endpoint = str(launch_result.get("endpoint") or state.get("cdp_endpoint") or "")
            browser_pid = str(launch_result.get("pid") or state.get("browser_pid") or "")
            live_screenshot = Path(str(state.get("live_screenshot") or live_screenshot))
    except Exception:
        endpoint = ""
    if not endpoint:
        # Last-resort: read the stable TPDC CDP profile port file used by V137+.
        try:
            session_dir = Path(os.getenv("DOMESTIC_PLATFORM_SESSION_DIR", r"E:\Agent_DSM\runs\sessions")) / "tpdc_cdp_external"
            port_file = session_dir / "cdp_port.txt"
            if port_file.exists():
                port = (port_file.read_text(encoding="utf-8") or "").strip()
                if port:
                    endpoint = f"http://127.0.0.1:{port}"
        except Exception:
            endpoint = ""
    watcher = Path(__file__).resolve().parents[1] / "tools" / "ftp_clipboard_watcher.py"
    if not watcher.exists():
        return {"ok": False, "error": f"FTP 捕捉脚本不存在：{watcher}"}
    cmd = [
        sys.executable,
        str(watcher),
        "--download-dir", str(local_save_dir),
        "--timeout", str(int(os.getenv("DOMESTIC_FTP_ACCOUNT_WAIT_TIMEOUT_SECONDS", "86400"))),
        "--interval", str(os.getenv("DOMESTIC_FTP_CLIPBOARD_INTERVAL_SECONDS", "2")),
        "--auto-download", str(os.getenv("DOMESTIC_FTP_AUTO_DOWNLOAD", "1")),
        "--requested-year", str(manifest.get("year") or ""),
        "--data-name", str(manifest.get("data_name") or ""),
    ]
    if endpoint:
        cmd += ["--cdp-endpoint", endpoint]
    if live_screenshot:
        cmd += ["--live-screenshot", str(live_screenshot)]
    if browser_pid:
        cmd += ["--browser-pid", str(browser_pid)]
    if live_state:
        cmd += ["--state-json", str(live_state)]
    try:
        proc = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parents[1]))
        manifest["ftp_capture_now_pid"] = proc.pid
        manifest["ftp_capture_now_command"] = cmd
        manifest["status"] = "ftp_account_waiting"
        _save_manifest(manifest)
        return {"ok": True, "pid": proc.pid, "endpoint": endpoint, "browser_pid": browser_pid, "live_screenshot": str(live_screenshot), "download_dir": str(local_save_dir), "state_json": str(live_state), "command": cmd}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "command": cmd}


def cancel_download_task(manifest_path: str | None, reason: str = "用户已停止当前下载任务") -> dict[str, Any]:
    """Stop the current TPDC/manual FTP download when the user says to stop.

    This updates download_manifest.json and ftp_download_status.json, and
    terminates the tracked Python FTP downloader child process when present.
    """
    manifest = _load_manifest(manifest_path) or {}
    if not manifest:
        return {"ok": False, "stopped": False, "message": "未找到当前下载任务记录。"}
    stopped_pids: list[int] = []
    raw_pids = [
        manifest.get("manual_ftp_download_pid"),
        manifest.get("ftp_download_pid"),
        manifest.get("ftp_capture_now_pid"),
    ]
    pid_values: list[int] = []
    for pid in raw_pids:
        try:
            if pid:
                pid_values.append(int(pid))
        except Exception:
            pass
    for pid_i in list(dict.fromkeys(pid_values)):
        for proc in list(_MANUAL_FTP_CHILDREN):
            try:
                if proc.pid == pid_i and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except Exception:
                        proc.kill()
                    stopped_pids.append(pid_i)
                if proc.poll() is not None:
                    try:
                        _MANUAL_FTP_CHILDREN.remove(proc)
                    except Exception:
                        pass
            except Exception:
                pass
        # Best-effort fallback when the process is not in this interpreter list.
        try:
            import signal
            if pid_i not in stopped_pids and os.name != "nt":
                os.kill(pid_i, signal.SIGTERM)
                stopped_pids.append(pid_i)
        except Exception:
            pass
    manifest.update({
        "status": "cancelled",
        "download_mode": manifest.get("download_mode") or "manual_handoff",
        "cancelled_by_user": True,
        "cancel_reason": str(reason),
        "ready_for_modeling": False,
    })
    _save_manifest(manifest)
    local_dir = Path(str(manifest.get("local_save_dir") or manifest.get("manual_download_dir") or DEFAULT_MANUAL_ROOT))
    status_payload = {
        "status": "cancelled",
        "message": str(reason),
        "cancelled_by_user": True,
        "stopped_pid": stopped_pids[0] if stopped_pids else None,
        "stopped_pids": stopped_pids,
        "updated_at": int(time.time()),
    }
    try:
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "ftp_download_status.json").write_text(json.dumps(status_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with (local_dir / "ftp_python_downloader.log").open("a", encoding="utf-8") as logf:
            logf.write(json.dumps({"event": "download_cancelled_by_user", **status_payload}, ensure_ascii=False) + "\n")
    except Exception:
        pass
    live = manifest.get("browser_live_state")
    if live:
        try:
            Path(str(live)).write_text(json.dumps(status_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    return {"ok": True, "stopped": True, "pid": stopped_pids[0] if stopped_pids else None, "stopped_pids": stopped_pids, "manifest_path": manifest.get("manifest_path"), "message": str(reason)}
