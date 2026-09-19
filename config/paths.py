from __future__ import annotations

import os
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

RFK_SCRIPT = MODEL_DIR / "RFK_formal_minimal.py"
GCP_SCRIPT = MODEL_DIR / "GCP_AOA_formal_uncertainty.py"
PIPELINE_SCRIPT = MODEL_DIR / "run_rfk_gcp_pipeline.py"

DOWNLOAD_RESULT_ROOT = Path(os.getenv("AGENT_DOWNLOAD_RESULT_ROOT", r"E:\Agent_DSM\数据下载结果"))
MAPPING_RESULT_ROOT = Path(os.getenv("AGENT_MAPPING_RESULT_ROOT", r"E:\Agent_DSM\制图结果"))
UNCERTAINTY_RESULT_ROOT = Path(os.getenv("AGENT_UNCERTAINTY_RESULT_ROOT", r"E:\Agent_DSM\不确定性分析结果"))

for _p in [DOWNLOAD_RESULT_ROOT, MAPPING_RESULT_ROOT, UNCERTAINTY_RESULT_ROOT]:
    try:
        _p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def timestamp_name(suffix: str = "") -> str:
    base = time.strftime("%Y%m%d_%H%M")
    suffix = str(suffix or "").strip("_ ")
    return f"{base}_{suffix}" if suffix else base


def new_download_dir() -> Path:
    p = DOWNLOAD_RESULT_ROOT / timestamp_name("download")
    p.mkdir(parents=True, exist_ok=True)
    return p


def new_mapping_dir() -> Path:
    p = MAPPING_RESULT_ROOT / timestamp_name("mapping")
    p.mkdir(parents=True, exist_ok=True)
    return p


def new_uncertainty_dir() -> Path:
    p = UNCERTAINTY_RESULT_ROOT / timestamp_name("GCP_AOA")
    p.mkdir(parents=True, exist_ok=True)
    return p

# RFK/GCP 运行过程使用正式输出根目录下的标准运行子目录。
DEFAULT_RFK_OUT_ROOT = MAPPING_RESULT_ROOT / "_rfk_runtime"
DEFAULT_GCP_OUT_ROOT = UNCERTAINTY_RESULT_ROOT / "_gcp_runtime"
DEFAULT_RFK_OUT_ROOT.mkdir(parents=True, exist_ok=True)
DEFAULT_GCP_OUT_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_RFK_OUTPUTS = {
    "pred_tif": DEFAULT_RFK_OUT_ROOT / "02_预测图" / "RFK最终预测图.tif",
    "report_json": DEFAULT_RFK_OUT_ROOT / "01_精度报告" / "RFK精度报告.json",
    "report_txt": DEFAULT_RFK_OUT_ROOT / "01_精度报告" / "RFK精度报告.txt",
    "strict_oof_csv": DEFAULT_RFK_OUT_ROOT / "03_GCP输入" / "strict_groupkfold_oof.csv",
    "manifest": DEFAULT_RFK_OUT_ROOT / "_state" / "rfk_manifest.json",
}

DEFAULT_GCP_OUTPUTS = {
    "center_tif": DEFAULT_GCP_OUT_ROOT / "02_不确定性与适用域图" / "GCP中心预测图.tif",
    "width_tif": DEFAULT_GCP_OUT_ROOT / "02_不确定性与适用域图" / "GCP预测区间宽度.tif",
    "lower_tif": DEFAULT_GCP_OUT_ROOT / "02_不确定性与适用域图" / "GCP预测区间下界.tif",
    "upper_tif": DEFAULT_GCP_OUT_ROOT / "02_不确定性与适用域图" / "GCP预测区间上界.tif",
    "qhat_tif": DEFAULT_GCP_OUT_ROOT / "02_不确定性与适用域图" / "GCP局地非一致性阈值.tif",
    "aoa_di_tif": DEFAULT_GCP_OUT_ROOT / "02_不确定性与适用域图" / "AOA不相似性指数_DI.tif",
    "aoa_inside_tif": DEFAULT_GCP_OUT_ROOT / "02_不确定性与适用域图" / "AOA适用域二值图.tif",
    "risk_class_tif": DEFAULT_GCP_OUT_ROOT / "02_不确定性与适用域图" / "GCP_AOA综合风险分区.tif",
    "report_json": DEFAULT_GCP_OUT_ROOT / "01_指标报告" / "GCP_AOA指标报告.json",
    "report_txt": DEFAULT_GCP_OUT_ROOT / "01_指标报告" / "GCP_AOA分析报告.txt",
    "manifest": DEFAULT_GCP_OUT_ROOT / "_state" / "gcp_aoa_manifest.json",
}
