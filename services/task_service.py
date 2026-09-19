import importlib.util
import json
import os
import re
import subprocess
import shutil
import threading
import time
import uuid
import secrets
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from config.settings import (
    DATA_DIR,
    ENABLE_CUSTOM_USER_RUN,
    PYTHON_EXECUTABLE,
    PERSIST_TASKS_TO_JSON,
    TASK_LOG_LIMIT,
    TASK_STATE_DIR,
    USER_GCP_COMMAND_TEMPLATE,
    USER_RFK_COMMAND_TEMPLATE,
)
from config.paths import RFK_SCRIPT, GCP_SCRIPT, DEFAULT_RFK_OUTPUTS, DEFAULT_GCP_OUTPUTS, new_uncertainty_dir
from services.pro_data_service import PRO_DATA_STEPS, prepare_online_data_manifest
from services.pro_mainline_diagnostic_model import run_pro_mainline_diagnostic_model
from services.pro_platform_model_csv_pipeline import run_platform_model_csv_pipeline
from services.aoi_preflight_service import preflight_mapping_region
from services.mapping_source_choice_service import choice_from_data_source
from utils.pro_console import pro_console_log
from services.pipeline_step_logger import append_step_record, attach_audit_paths, audit_paths
from services.landcover_scope_service import infer_landcover_scope, landcover_scope_title_prefix

try:
    from services.formal_map_png_service import create_formal_continuous_png
except Exception:  # pragma: no cover
    create_formal_continuous_png = None


@dataclass
class TaskState:
    task_id: str
    kind: str
    status: str = "queued"
    progress: int = 0
    stage: str = "等待开始"
    logs: list[str] = field(default_factory=list)
    error: str | None = None
    result_paths: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    pid: int | None = None
    session_id: str | None = None
    step_records: list[dict] = field(default_factory=list)
    step_audit_paths: dict = field(default_factory=dict)
    cancel_requested: bool = False

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        data = dict(data or {})
        data.setdefault("step_records", [])
        data.setdefault("step_audit_paths", {})
        data.setdefault("cancel_requested", False)
        return cls(**data)



def _formal_rfk_progress_from_log_line(line: str, current: int = 0):
    """Map PRO formal RFK log messages to a live UI progress estimate.

    The progress is not a fake timer: it advances only when actual pipeline
    milestones are emitted. Repeated raster logs move the bar gradually but
    remain capped until the next major stage is reached.
    """
    s = str(line or "")
    low = s.lower()
    if "收到 rfk" in low or "收到 rfk/pro 制图任务" in s:
        return 2, "已接收制图任务"
    if "已进入 pro 自动数据准备流程" in s:
        return max(current, 5), "读取数据与任务信息"
    if "样点csv读取成功" in s or "样点 CSV读取成功" in s or "样点CSV读取成功" in s:
        return max(current, 10), "读取样点数据"
    if "已从本地行政区划" in s or "已使用本地行政区划" in s:
        return max(current, 13), "识别制图区域"
    if "正式制图流程启动" in s:
        return max(current, 16), "启动正式制图流程"
    if "环境协变量选择方案已确定" in s:
        return max(current, 19), "确定协变量方案"
    if "已按v168规则自动确定目标分辨率" in low or "已按v170规则自动确定目标分辨率" in low:
        return max(current, 21), "确定输出分辨率"
    if "栅格标准化完成" in s:
        return min(max(current + 2, 22), 45), "统一协变量坐标系和分辨率"
    if "建模样点表.csv已生成" in low or "最终建模csv已生成" in low:
        return max(current, 48), "生成建模样本表"
    if "即将启动rfk训练" in low or "从建模样点表重新开始训练" in low:
        return max(current, 50), "启动模型训练"
    if "已识别融合csv训练诊断特征" in low or "开始构建rfk训练矩阵" in low:
        return max(current, 52), "构建模型训练矩阵"
    if "clean rfk 开始联合寻参" in low:
        return max(current, 55), "开始模型自动寻参"
    m = re.search(r"Clean RFK 正在评分候选参数\s*(\d+)\s*/\s*(\d+)", s, flags=re.I) or re.search(r"Clean RFK 候选参数\s*(\d+)\s*/\s*(\d+)\s*完成", s, flags=re.I)
    if m:
        i, n = int(m.group(1)), max(1, int(m.group(2)))
        return max(current, 55 + int(17 * i / n)), f"模型自动寻参 {i}/{n}"
    m = re.search(r"RF超参数候选\s*(\d+)\s*/\s*(\d+)", s)
    if m:
        i, n = int(m.group(1)), max(1, int(m.group(2)))
        return max(current, 55 + int(17 * i / n)), f"自动寻参 {i}/{n}"
    if "rf基学习器超参数搜索完成" in low or "完成联合寻参" in low:
        return max(current, 72), "完成自动寻参"
    if "可出图特征rfk" in low and ("候选参数" in low or "联合寻参" in low):
        mm = (re.search(r"可出图特征RFK(?:正在评分)?候选参数\s*(\d+)\s*/\s*(\d+)", s, flags=re.I)
              or re.search(r"可出图特征RFK联合寻参\s*(\d+)\s*/\s*(\d+)", s, flags=re.I))
        if mm:
            i, n = int(mm.group(1)), max(1, int(mm.group(2)))
            return max(current, 72 + int(8 * i / n)), f"可出图特征寻参 {i}/{n}"
        return max(current, 74), "可出图特征寻参"
    if "空间7:3验证" in s:
        if "完成" in s:
            return max(current, 75), "完成空间7:3验证"
        return max(current, 73), "空间7:3验证"
    if "随机7:3验证" in s:
        if "完成" in s:
            return max(current, 77), "完成随机7:3验证"
        return max(current, 76), "随机7:3验证"
    if "重复验证完成" in s:
        return max(current, 79), "完成重复验证"
    if "已训练正式出图rfk模型" in low:
        return max(current, 81), "训练正式出图模型"
    if "预测网格规模预检" in s or "预测网格" in s or "构建投影坐标预测网格" in s:
        return max(current, 84), "构建预测网格"
    if "栅格值已抽取到样点" in s:
        return min(max(current + 1, 78), 88), "抽取预测网格协变量"
    if "开始生成正式geotiff" in low or "正在写出预测geotiff" in low or "make_prediction_tif" in low:
        return max(current, 88), "生成预测图"
    if "clean rfk 建模完成" in low or "建模csv已完成正式" in low or "建模CSV已完成正式" in s:
        return max(current, 94), "生成预测图和模型记录"
    if "正式制图流程结束" in s:
        return max(current, 98), "整理输出文件"
    if "任务成功完成" in s:
        return 99, "等待前端加载结果"
    return None

class TaskRegistry:
    def __init__(self):
        self._tasks: dict[str, TaskState] = {}
        self._lock = threading.Lock()
        self._state_file = Path(TASK_STATE_DIR) / "tasks.json"
        self._load()

    def _load(self):
        if not PERSIST_TASKS_TO_JSON or not self._state_file.exists():
            return
        try:
            raw = json.loads(self._state_file.read_text(encoding="utf-8"))
            for item in raw.get("tasks", []):
                t = TaskState.from_dict(item)
                if t.status == "running":
                    t.status = "interrupted"
                    t.stage = "服务重启前任务中断"
                    t.error = "后台服务重启，原运行中任务未继续追踪。"
                self._tasks[t.task_id] = t
        except Exception:
            return

    def _persist(self):
        if not PERSIST_TASKS_TO_JSON:
            return
        payload = {"tasks": [t.to_dict() for t in sorted(self._tasks.values(), key=lambda x: x.created_at)]}
        self._state_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def create(self, kind: str, session_id: str | None = None):
        t = TaskState(task_id=str(uuid.uuid4()), kind=kind, session_id=session_id)
        t.step_audit_paths = audit_paths(t.task_id)
        try:
            rec = append_step_record(t.task_id, kind=t.kind, status=t.status, progress=t.progress, stage=t.stage, message="任务已创建", event="create")
            t.step_records.append(rec)
        except Exception:
            pass
        with self._lock:
            self._tasks[t.task_id] = t
            self._persist()
        return t

    def get(self, task_id: str):
        with self._lock:
            return self._tasks.get(task_id)

    def update(self, task_id: str, **kwargs):
        with self._lock:
            t = self._tasks.get(task_id)
            if not t:
                return
            for k, v in kwargs.items():
                setattr(t, k, v)
            self._persist()

    def append_log(self, task_id: str, line: str):
        # 同步打印到 PyCharm 控制台，方便检查任务进度；同时写入 UI 任务日志。
        pro_console_log("TASK", str(line), task_id=task_id)
        with self._lock:
            t = self._tasks.get(task_id)
            if not t:
                return
            t.logs.append(line)
            if len(t.logs) > TASK_LOG_LIMIT:
                t.logs = t.logs[-TASK_LOG_LIMIT:]
            # PRO/正式制图流程在同一 Python 线程内执行，不经过外部脚本 stdout parser。
            # 因此这里直接根据真实任务日志更新进度，避免前端一直停在 4%。
            try:
                if t.kind == "rfk" and t.status in {"queued", "running"}:
                    parsed = _formal_rfk_progress_from_log_line(str(line), int(t.progress or 0))
                    if parsed:
                        p, stage = parsed
                        if p >= int(t.progress or 0):
                            old_stage = t.stage
                            old_progress = int(t.progress or 0)
                            t.progress = max(0, min(99, int(p)))
                            t.stage = stage
                            t.status = "running"
                            if int(t.progress or 0) != old_progress or str(stage) != str(old_stage):
                                try:
                                    rec = append_step_record(task_id, kind=t.kind, status=t.status, progress=t.progress, stage=t.stage, message=str(line), event="progress")
                                    t.step_records.append(rec)
                                    t.step_records = t.step_records[-300:]
                                except Exception:
                                    pass
            except Exception:
                pass
            self._persist()

    def reset(self):
        with self._lock:
            self._tasks = {}
            if self._state_file.exists():
                try:
                    self._state_file.unlink()
                except Exception:
                    pass


TASKS = TaskRegistry()
RUNNING_PROCS: dict[str, subprocess.Popen] = {}
RUNNING_LOCK = threading.Lock()



def _find_active_task(kind: str, session_id: str | None):
    """Return an existing queued/running task for the same session and kind.

    Dash can fire duplicate callbacks when the user double-clicks Send or when
    the browser submits twice before session-store updates. A server-side guard
    is required; UI checks alone are insufficient.
    """
    if not session_id:
        return None
    try:
        with TASKS._lock:
            active = [
                t for t in TASKS._tasks.values()
                if t.kind == kind and t.session_id == session_id and t.status in {"queued", "running"}
            ]
        if not active:
            return None
        active.sort(key=lambda x: x.created_at, reverse=True)
        return active[0]
    except Exception:
        return None

# V232: no fixed legacy training paths. Formal mapping inputs are resolved
# from the current conversation/session, imported local paths, uploaded files,
# or task manifests.  Preflight only checks the runnable script/module.
RFK_REQUIRED_INPUTS: list[Path] = []

GCP_REQUIRED_INPUTS = [
    Path(r"E:\Agent_DSM\RFK_OUT\03_模型与中间数据\rfk_manifest.json"),
    Path(r"E:\Agent_DSM\RFK_OUT\03_模型与中间数据\strict_groupkfold_oof.csv"),
]

RFK_REQUIRED_MODULES = ["numpy", "pandas", "rasterio", "sklearn", "pykrige", "joblib"]
GCP_REQUIRED_MODULES = ["numpy", "pandas", "rasterio"]


def _missing_modules(modules: list[str]) -> list[str]:
    missing = []
    for m in modules:
        try:
            if importlib.util.find_spec(m) is None:
                missing.append(m)
        except Exception:
            missing.append(m)
    return missing


def _preflight_script(kind: str, skip_input_paths: bool = False) -> str | None:
    if kind == "rfk":
        miss_mod = _missing_modules(RFK_REQUIRED_MODULES)
        if miss_mod:
            return "模型启动前检查失败，当前 Python 环境缺少模块：" + "、".join(miss_mod)
        miss_path = _missing_paths(([Path(RFK_SCRIPT)] if skip_input_paths else (RFK_REQUIRED_INPUTS + [Path(RFK_SCRIPT)])))
        if miss_path:
            return "模型启动前检查失败，缺少以下输入或脚本：" + "；".join(miss_path)
        return None
    if kind == "gcp":
        miss_mod = _missing_modules(GCP_REQUIRED_MODULES)
        if miss_mod:
            return "GCP + AOA 启动前检查失败，当前 Python 环境缺少模块：" + "、".join(miss_mod)
        if not Path(GCP_SCRIPT).exists():
            return "GCP + AOA 启动前检查失败，缺少脚本：" + str(GCP_SCRIPT)
        return None
    return None


def _resolve_gcp_input_paths(rfk_result_paths: dict | None) -> dict[str, str]:
    """Resolve formal GCP + AOA inputs from the latest mapping result memory.

    PRO mapping publishes clean user-facing outputs but keeps modeling metadata in
    report_json/model_report_json.  This resolver treats that JSON as the manifest
    when a classic rfk_manifest key is absent, then pulls pred_tif, strict_oof_csv
    and prediction-grid covariates from both top-level fields and outputs.
    """
    paths = dict(rfk_result_paths or {})
    manifest_path = (
        paths.get("manifest") or paths.get("manifest_path") or paths.get("rfk_manifest")
        or paths.get("report_json") or paths.get("model_report_json") or paths.get("formal_model_report_json")
    )
    pred_tif = (
        paths.get("pred_tif") or paths.get("prediction_tif")
        or paths.get("gee_cropland_masked_prediction_tif") or paths.get("gee_admin_continuous_prediction_tif")
    )
    strict_oof_csv = paths.get("strict_oof_csv") or paths.get("oof_csv") or paths.get("calibration_csv")
    grid_cov_csv = paths.get("prediction_grid_covariates_csv") or paths.get("grid_covariates_csv") or paths.get("prediction_grid_csv")

    try:
        if manifest_path and Path(str(manifest_path)).exists():
            m = json.loads(Path(str(manifest_path)).read_text(encoding="utf-8"))
            outputs = m.get("outputs") or {}
            pred_tif = pred_tif or outputs.get("pred_tif") or outputs.get("prediction_tif")
            strict_oof_csv = strict_oof_csv or outputs.get("strict_oof_csv") or outputs.get("oof_csv") or outputs.get("calibration_csv")
            grid_cov_csv = grid_cov_csv or outputs.get("prediction_grid_covariates_csv") or outputs.get("grid_covariates_csv") or outputs.get("prediction_grid_csv")
            # Older PRO model reports stored these as top-level fields.
            pred_tif = pred_tif or m.get("pred_tif") or m.get("prediction_tif")
            strict_oof_csv = strict_oof_csv or m.get("strict_oof_csv") or m.get("oof_csv") or m.get("calibration_csv")
            grid_cov_csv = grid_cov_csv or m.get("prediction_grid_covariates_csv") or m.get("grid_covariates_csv") or m.get("prediction_grid_csv")
            if not manifest_path:
                manifest_path = str(paths.get("report_json") or paths.get("model_report_json") or "")
    except Exception:
        pass

    # Last-resort search around the model report / internal work directory.  This
    # does not fabricate uncertainty inputs; it only finds audited files already
    # written by the RFK pipeline.
    search_dirs = []
    for v in [manifest_path, paths.get("report_json"), paths.get("model_report_json"), paths.get("formal_mapping_work_dir"), paths.get("final_result_dir")]:
        try:
            if v:
                q = Path(str(v))
                search_dirs.append(q if q.is_dir() else q.parent)
        except Exception:
            pass
    for d in list(dict.fromkeys(search_dirs)):
        if not strict_oof_csv:
            for name in ["03_GCP输入/strict_groupkfold_oof.csv", "strict_groupkfold_oof.csv", "strict_oof.csv", "calibration_points.csv"]:
                cand = d / name
                if cand.exists():
                    strict_oof_csv = str(cand)
                    break
        if not grid_cov_csv:
            for name in ["out/预测网格表.csv", "预测网格表.csv", "prediction_grid_covariates.csv", "grid_covariates.csv"]:
                cand = d / name
                if cand.exists():
                    grid_cov_csv = str(cand)
                    break

    out = {
        "manifest": str(manifest_path or ""),
        "pred_tif": str(pred_tif or ""),
        "strict_oof_csv": str(strict_oof_csv or ""),
    }
    if grid_cov_csv:
        out["prediction_grid_covariates_csv"] = str(grid_cov_csv)
    return out


def _preflight_gcp_from_rfk_result(rfk_result_paths: dict | None) -> str | None:
    if not rfk_result_paths:
        return "请先完成一次制图，再运行 GCP + AOA 不确定性与适用域分析。"

    resolved = _resolve_gcp_input_paths(rfk_result_paths)
    missing_fields = []
    if not resolved.get("manifest"):
        missing_fields.append("manifest/report_json")
    if not resolved.get("pred_tif"):
        missing_fields.append("pred_tif")
    if not resolved.get("strict_oof_csv"):
        missing_fields.append("strict_oof_csv")
    if missing_fields:
        return "本次制图结果缺少 GCP + AOA 所需字段：" + "、".join(missing_fields)

    miss = _missing_paths([Path(resolved["manifest"]), Path(resolved["pred_tif"]), Path(resolved["strict_oof_csv"]), Path(GCP_SCRIPT)])
    if miss:
        return "GCP + AOA 启动前检查失败，缺少以下制图结果、中间数据或脚本：" + "；".join(miss)
    return None

def _missing_paths(paths: list[Path]) -> list[str]:
    return [str(p) for p in paths if not Path(p).exists()]


def _read_manifest_outputs(manifest_path: Path) -> dict:
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        outputs = raw.get("outputs") or {}
        return {k: str(v) for k, v in outputs.items() if v}
    except Exception:
        return {}


def _resolve_result_paths(kind: str, fallback: dict) -> dict:
    if kind == "rfk":
        manifest = Path(fallback.get("manifest", DEFAULT_RFK_OUTPUTS.get("manifest")))
        outputs = _read_manifest_outputs(manifest)
        if outputs:
            outputs.setdefault("manifest", str(manifest))
            return outputs
        return {k: str(v) for k, v in fallback.items()}
    if kind == "gcp":
        manifest = Path(fallback.get("manifest", DEFAULT_GCP_OUTPUTS.get("manifest")))
        resolved = {k: str(v) for k, v in fallback.items()}
        outputs = _read_manifest_outputs(manifest)
        if outputs:
            resolved.update(outputs)
            resolved.setdefault("manifest", str(manifest))
            return resolved
        return resolved
    return {k: str(v) for k, v in fallback.items()}



def _set_stage(task_id: str, progress: int, stage: str):
    TASKS.update(task_id, progress=progress, stage=stage, status="running")
    try:
        t = TASKS.get(task_id)
        if t is not None:
            rec = append_step_record(task_id, kind=t.kind, status="running", progress=progress, stage=stage, message=stage, event="stage")
            t.step_records.append(rec)
            t.step_records = t.step_records[-300:]
            TASKS.update(task_id, step_records=t.step_records, step_audit_paths=audit_paths(task_id))
    except Exception:
        pass


@contextmanager
def _task_heartbeat(task_id: str, label: str = "后台任务", interval_sec: float | None = None):
    """Print a periodic backend heartbeat and mirror it into the UI task log.

    Long in-process RFK stages can run without producing stdout for minutes.
    The heartbeat does not pretend that a computation milestone has completed;
    it simply proves that the worker thread is alive and reports elapsed time,
    current progress, current stage, and the latest task log.
    """
    try:
        if interval_sec is None:
            interval_sec = float(os.getenv("PRO_TASK_HEARTBEAT_SEC", os.getenv("PRO_PROGRESS_HEARTBEAT_SEC", "30")))
    except Exception:
        interval_sec = 30.0
    interval_sec = max(10.0, float(interval_sec or 30.0))
    stop = threading.Event()
    started = time.monotonic()

    def _fmt_elapsed(sec: float) -> str:
        sec = max(0, int(sec))
        h, rem = divmod(sec, 3600)
        m, ss = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{ss:02d}" if h else f"{m:02d}:{ss:02d}"

    def _loop():
        tick = 0
        while not stop.wait(interval_sec):
            tick += 1
            try:
                t = TASKS.get(task_id)
                if not t or t.status not in {"queued", "running"}:
                    break
                latest = ""
                try:
                    latest = str((t.logs or [""])[-1])
                    if len(latest) > 220:
                        latest = latest[:220] + "..."
                except Exception:
                    latest = ""
                TASKS.append_log(
                    task_id,
                    f"[HEARTBEAT] {label}仍在运行 | tick={tick} | elapsed={_fmt_elapsed(time.monotonic()-started)} | progress={int(t.progress or 0)}% | stage={t.stage or ''} | last_log={latest}"
                )
            except Exception:
                pass

    thread = threading.Thread(target=_loop, name=f"task-heartbeat-{str(task_id)[:8]}", daemon=True)
    try:
        TASKS.append_log(task_id, f"[HEARTBEAT] {label}心跳已启动，每{int(interval_sec)}秒记录一次运行状态。")
    except Exception:
        pass
    thread.start()
    try:
        yield
    finally:
        stop.set()
        try:
            thread.join(timeout=1.0)
        except Exception:
            pass


def _collect_gee_formal_outputs(pred_tif: str | None) -> dict:
    """Collect V77/V78 GEE formal-flow artifacts beside gee_formal_prediction.tif.

    These paths are stored in task.result_paths so the Dash UI can display the
    full GEE workflow instead of only knowing the final prediction GeoTIFF.
    """
    out: dict[str, str] = {}
    if not pred_tif:
        return out
    try:
        p = Path(pred_tif)
        out_dir = p.parent
        candidates = {
            "gee_formal_report_json": out_dir / "gee_formal_prediction_grid_report.json",
            "gee_grid_covariates_csv": out_dir / "gee_prediction_grid_covariates.csv",
            "gee_admin_mask_tif": out_dir / "gee_admin_boundary_mask.tif",
            "gee_cropland_mask_tif": out_dir / "gee_cropland_mask.tif",
            "gee_final_valid_mask_tif": out_dir / "gee_final_valid_mask.tif",
            "gee_admin_continuous_prediction_tif": out_dir / "gee_formal_prediction_admin_continuous.tif",
            "gee_cropland_masked_prediction_tif": out_dir / "gee_formal_prediction.tif",
            "gee_covariate_plan_json": out_dir / "gee_sampler" / "gee_covariate_plan.json",
            "gee_sample_covariates_csv": out_dir / "gee_sampler" / "gee_model_ready_covariates.csv",
            "gee_gcp_report_json": out_dir / "gee_gcp_report.json",
            "gee_gcp_width_tif": out_dir / "gee_gcp_width.tif",
            "gee_gcp_lower_tif": out_dir / "gee_gcp_lower.tif",
            "gee_gcp_upper_tif": out_dir / "gee_gcp_upper.tif",
            "gee_gcp_half_width_q_tif": out_dir / "gee_gcp_half_width_q.tif",
        }
        for key, path in candidates.items():
            if path.exists():
                out[key] = str(path)
    except Exception:
        return out
    return out



def _safe_short_name(text: str | None, default: str = "制图结果") -> str:
    """Return a short Windows-safe Chinese filename stem."""
    t = str(text or "").strip() or default
    t = re.sub(r"[\\/:*?\"<>|\r\n\t]+", "_", t)
    t = re.sub(r"\s+", "", t)
    return t[:36] or default


def _parse_result_region_year(request_text: str | None, result_paths: dict | None = None) -> tuple[str, str]:
    text = str(request_text or "")
    year = ""
    m = re.search(r"(19\d{2}|20\d{2})", text)
    if m:
        year = m.group(1)

    # Prefer explicit county/district names over long command fragments such as
    # “为我绘制成都市温江区”.  The regex is intentionally conservative so a
    # typo like “20220年成都市” cannot produce “年成都市” as the region.
    region = ""
    county_patterns = [
        r"(?:[\u4e00-\u9fa5]{2,10}省)?\s*[\u4e00-\u9fa5]{2,10}市\s*([\u4e00-\u9fa5]{2,8}(?:区|县|旗))",
        r"([\u4e00-\u9fa5]{2,8}(?:区|县|旗))",
    ]
    bad_prefixes = ("为我", "绘制", "制作", "进行", "根据", "我的", "现在", "土壤", "有机质", "环境", "年")
    for pat in county_patterns:
        for mm in re.finditer(pat, text):
            cand = str(mm.group(1)).strip()
            for bp in bad_prefixes:
                while cand.startswith(bp) and len(cand) > len(bp) + 1:
                    cand = cand[len(bp):]
            if cand and cand not in {"有机质图", "土壤有机质", "环境协变量"} and len(cand) <= 8:
                region = cand
                break
        if region:
            break

    if not region:
        for mm in re.finditer(r"([\u4e00-\u9fa5]{2,8}市)", text):
            cand = str(mm.group(1)).strip()
            for bp in bad_prefixes:
                while cand.startswith(bp) and len(cand) > len(bp) + 1:
                    cand = cand[len(bp):]
            # Avoid accepting leading junk in strings such as “20220年成都市”.
            if cand and len(cand) <= 6 and not cand.startswith("年"):
                region = cand
                break
    if not region:
        region = "目标区"
    if not year:
        year = time.strftime("%Y%m%d")
    return _safe_short_name(region, "目标区"), year


def _derive_formal_map_title(request_text: str | None, result_paths: dict | None = None) -> str:
    """Derive the user-facing map name from the user's mapping request.

    The interactive layout tab and final PNG title should reflect what the user
    asked for, e.g. “成都市耕地土壤有机质图”.  This is a display title, not a
    filename parser; formal cartographic elements are still managed separately.
    """
    text = str(request_text or "")
    # Directly reuse a concise phrase ending with 有机质图 when present.
    direct_patterns = [
        r"([\u4e00-\u9fa5]{2,10}(?:市|区|县|旗)(?:耕地|农田|林地|森林|草地|灌木|水体|裸地|建设用地)?(?:土壤)?有机质图)",
        r"([\u4e00-\u9fa5]{2,10}(?:市|区|县|旗)(?:耕地|农田|林地|森林|草地|灌木|水体|裸地|建设用地)?(?:土壤)?有机质(?:空间分布)?图)",
    ]
    for pat in direct_patterns:
        mm = re.search(pat, text)
        if mm:
            title = re.sub(r"\s+", "", mm.group(1))
            for prefix in ("请", "为我", "帮我", "现在", "根据我的数据", "绘制", "制作", "进行"):
                while title.startswith(prefix) and len(title) > len(prefix) + 2:
                    title = title[len(prefix):]
            title = re.sub(r"^(?:\d{4,5})?年+", "", title)
            # If a district/county appears inside a longer command fragment, keep the
            # meaningful administrative suffix rather than command words.
            inner = re.search(r"([\u4e00-\u9fa5]{2,8}(?:区|县|旗)(?:耕地|农田|林地|森林|草地|灌木|水体|裸地|建设用地)?(?:土壤)?有机质(?:空间分布)?图)", title)
            if inner:
                title = inner.group(1)
            if 4 <= len(title) <= 24:
                if "土壤" not in title and "有机质" in title:
                    title = title.replace("有机质", "土壤有机质", 1)
                return title
    region, _year = _parse_result_region_year(text, result_paths)
    scope_payload = infer_landcover_scope(text)
    scope = landcover_scope_title_prefix(scope_payload)
    return f"{region}{scope}土壤有机质图"



def _new_public_run_code() -> str:
    """Return a non-sensitive six-digit run code for backend/result tracking.

    This is not a cryptographic checksum and must not be used to prove file
    integrity.  The full SHA256 remains in backend metadata only.  The code is
    regenerated for every mapping publication so repeated runs do not share the
    same visible identifier.
    """
    try:
        return f"{secrets.randbelow(900000) + 100000:06d}"
    except Exception:
        return f"{int(time.time() * 1000) % 900000 + 100000:06d}"

def _copy_file(src: str | Path | None, dst: Path) -> str:
    if not src:
        return ""
    try:
        sp = Path(src)
        if not sp.exists():
            return ""
        dst.parent.mkdir(parents=True, exist_ok=True)
        if sp.resolve() != dst.resolve():
            shutil.copy2(sp, dst)
        return str(dst)
    except Exception:
        return ""


def _parse_hex_color_rgba(value: str | None, default: str) -> tuple[int, int, int, int]:
    raw = str(value or default or "").strip()
    if raw.startswith("#"):
        raw = raw[1:]
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    try:
        return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16), 255
    except Exception:
        return (216, 27, 96, 255)



def _find_clcd_mask_near_prediction(tif_path: str | Path | None) -> str:
    """V176: find a binary cropland mask for SOM PNG compositing.

    Final SOM preview must show CLCD=1 cropland with green continuous values and
    AOI-internal non-cropland in red. The mask can be passed through result_paths,
    stored in formal_model_report.json, or simply sit beside the prediction TIF.
    """
    try:
        if not tif_path:
            return ""
        p = Path(tif_path)
        dirs = [p.parent]
        try:
            dirs.append(p.parent.parent)
        except Exception:
            pass
        names = [
            "目标地类掩膜.tif",
            "CLCD目标地类掩膜_1目标0其他.tif",
            "耕地掩膜.tif",
            "CLCD耕地掩膜_1耕地0非耕地.tif",
            "clcd_cropland_mask.tif",
            "cropland_mask.tif",
        ]
        for d in dirs:
            for name in names:
                cand = d / name
                if cand.exists():
                    return str(cand)
            for pat in ["*目标地类*掩膜*.tif", "*耕地*掩膜*.tif", "*crop*mask*.tif", "*cropland*mask*.tif", "*landcover*mask*.tif"]:
                for cand in d.glob(pat):
                    return str(cand)
    except Exception:
        return ""
    return ""


def _read_mask_for_png(mask_tif_path: str | Path | None, ref_ds):
    try:
        import numpy as np
        import rasterio
        if not mask_tif_path:
            return None
        mp = Path(mask_tif_path)
        if not mp.exists():
            return None
        with rasterio.open(mp) as ms:
            if ms.width == ref_ds.width and ms.height == ref_ds.height and ms.crs == ref_ds.crs and ms.transform == ref_ds.transform:
                return ms.read(1)
            from rasterio.warp import reproject, Resampling
            dst = np.full((ref_ds.height, ref_ds.width), 255, dtype="uint8")
            reproject(
                source=ms.read(1), destination=dst,
                src_transform=ms.transform, src_crs=ms.crs,
                dst_transform=ref_ds.transform, dst_crs=ref_ds.crs,
                resampling=Resampling.nearest,
                src_nodata=ms.nodata if ms.nodata is not None else 255,
                dst_nodata=255,
            )
            return dst
    except Exception:
        return None

def _make_prediction_png(tif_path: str | Path | None, png_path: Path, mask_tif_path: str | Path | None = None, title: str | None = None, noncrop_color: tuple[int, int, int, int] | None | str = "default") -> str:
    """Create the formal SOM PNG layout from a GeoTIFF. Failure is non-fatal.

    V191: the exported PNG is no longer a bare transparent raster preview. It is
    a complete cartographic product with map name, legend, scale bar, north
    arrow and neatline. The scientific GeoTIFF remains unchanged and contains no
    cartographic decoration.
    """
    if not tif_path:
        return ""
    try:
        tif = Path(tif_path)
        if not tif.exists():
            return ""
        if create_formal_continuous_png is not None:
            non_color = None if noncrop_color is None else (
                _parse_hex_color_rgba(os.getenv("PRO_CLCD_NON_CROPLAND_COLOR", "#E53935"), "#E53935")
                if noncrop_color == "default" else noncrop_color
            )
            out = create_formal_continuous_png(
                tif,
                png_path,
                title=str(title or "土壤有机质图"),
                legend_title="有机质含量（g/kg）",
                ramp_name="green",
                mask_tif_path=(mask_tif_path or _find_clcd_mask_near_prediction(tif)),
                noncrop_color=non_color,
            )
            if out:
                return out
    except Exception:
        pass

    # Fallback kept for environments without PIL/rasterio cartographic support.
    try:
        import numpy as np
        import rasterio
        from PIL import Image
        tif = Path(tif_path)
        if not tif.exists():
            return ""
        with rasterio.open(tif) as ds:
            arr = ds.read(1, masked=True).astype("float32")
            data = arr.filled(np.nan)
            mask_for_display = _read_mask_for_png(mask_tif_path or _find_clcd_mask_near_prediction(tif), ds)
        finite = np.isfinite(data)
        if not finite.any():
            return ""
        lo, hi = np.nanpercentile(data[finite], [2, 98])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.nanmin(data[finite])), float(np.nanmax(data[finite]))
        scaled = np.clip((data - lo) / max(hi - lo, 1e-6), 0, 1)
        r = (230 - 210 * scaled).astype("uint8")
        g = (245 - 105 * scaled).astype("uint8")
        b = (224 - 150 * scaled).astype("uint8")
        a = np.where(finite, 255, 0).astype("uint8")
        rgba = np.dstack([r, g, b, a])
        if mask_for_display is not None:
            if noncrop_color is None:
                rgba[mask_for_display == 0] = (0, 0, 0, 0)
            else:
                non_color = (
                    _parse_hex_color_rgba(os.getenv("PRO_CLCD_NON_CROPLAND_COLOR", "#E53935"), "#E53935")
                    if noncrop_color == "default" else noncrop_color
                )
                rgba[mask_for_display == 0] = non_color
            rgba[mask_for_display == 255] = (0, 0, 0, 0)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba, mode="RGBA").save(png_path)
        return str(png_path)
    except Exception:
        return ""


def _make_binary_mask_png(mask_tif_path: str | Path | None, png_path: Path) -> str:
    """Create a two-color cropland/non-cropland PNG from CLCD binary mask."""
    if not mask_tif_path:
        return ""
    try:
        import numpy as np
        import rasterio
        from PIL import Image
        mp = Path(mask_tif_path)
        if not mp.exists():
            return ""
        with rasterio.open(mp) as ds:
            m = ds.read(1)
        crop = _parse_hex_color_rgba(os.getenv("PRO_CLCD_CROPLAND_COLOR", "#2EA043"), "#2EA043")
        non = _parse_hex_color_rgba(os.getenv("PRO_CLCD_NON_CROPLAND_COLOR", "#E53935"), "#E53935")
        rgba = np.zeros((m.shape[0], m.shape[1], 4), dtype="uint8")
        rgba[m == 0] = non
        rgba[m == 1] = crop
        rgba[m == 255] = (0, 0, 0, 0)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba, mode="RGBA").save(png_path)
        return str(png_path)
    except Exception:
        return ""

def _publish_final_mapping_outputs(result_paths: dict, request_text: str = "") -> dict:
    """Publish a clean, short, user-facing mapping result folder.

    Internal pipeline directories may contain JSON/CSV/cache/std/raw files. The user-facing
    folder is intentionally simple and contains only: 有机质图.tif、 有机质图.png、模型记录.txt.
    This function must never break the modeling task; on partial failure it returns the
    original paths plus a warning.
    """
    result_paths = dict(result_paths or {})
    warnings = []
    try:
        root = Path(os.getenv("PRO_FINAL_RESULT_ROOT", os.getenv("PRO_OUTPUT_ROOT", r"E:\Agent_DSM\制图结果")))
        if not root.is_absolute():
            root = Path.cwd() / root
        region, year = _parse_result_region_year(request_text, result_paths)
        map_title = _derive_formal_map_title(request_text, result_paths)
        result_paths["map_title"] = map_title
        result_paths["map_title_base"] = map_title
        result_paths["request_text"] = str(request_text or "")
        # Backend-only six-digit run code.  This replaces any user-facing long
        # checksum display; full SHA256 is still preserved in report metadata and logs.
        result_paths["public_run_code"] = _new_public_run_code()
        pro_console_log("RESULT", "本轮制图运行编号已生成", {"public_run_code": result_paths["public_run_code"]})
        final_dir = root / _safe_short_name(f"{region}_{year}_结果")
        if final_dir.exists() and any(final_dir.iterdir()):
            final_dir = root / _safe_short_name(f"{region}_{year}_{time.strftime('%H%M')}")
        final_dir.mkdir(parents=True, exist_ok=True)

        pred_src = result_paths.get("pred_tif") or result_paths.get("prediction_tif")
        txt_src = result_paths.get("model_record_txt")
        clcd_mask_src = result_paths.get("clcd_mask_tif") or ""
        clcd_mask_png_src = result_paths.get("clcd_mask_png") or ""
        try:
            report_json = result_paths.get("report_json") or result_paths.get("model_report_json") or ""
            if report_json and Path(report_json).exists():
                _rj = json.loads(Path(report_json).read_text(encoding="utf-8"))
                _mo = _rj.get("clcd_binary_mask_outputs") or {}
                clcd_mask_src = clcd_mask_src or str(_mo.get("clcd_mask_tif") or "")
                clcd_mask_png_src = clcd_mask_png_src or str(_mo.get("clcd_mask_png") or "")
        except Exception:
            pass
        tif_dst = final_dir / "有机质图.tif"
        png_dst = final_dir / "有机质图.png"
        txt_dst = final_dir / "模型记录.txt"
        mask_tif_dst = final_dir / "目标地类掩膜.tif"
        mask_png_dst = final_dir / "目标地类掩膜.png"

        pred_final = _copy_file(pred_src, tif_dst)
        if pred_final:
            result_paths["internal_pred_tif"] = str(pred_src)
            result_paths["pred_tif"] = pred_final
        else:
            warnings.append("未找到预测栅格，无法复制有机质图.tif。")

        if not clcd_mask_src:
            clcd_mask_src = _find_clcd_mask_near_prediction(pred_src)
        mask_final = _copy_file(clcd_mask_src, mask_tif_dst) if clcd_mask_src else ""
        if mask_final:
            result_paths["clcd_mask_tif"] = mask_final
        mask_png_final = _copy_file(clcd_mask_png_src, mask_png_dst) if clcd_mask_png_src else ""
        if not mask_png_final and mask_final:
            mask_png_final = _make_binary_mask_png(mask_final, mask_png_dst)
        if mask_png_final:
            result_paths["clcd_mask_png"] = mask_png_final

        scope_payload = infer_landcover_scope(str(request_text or ""))
        target_landcover_only = bool(scope_payload.get("mask_by_landcover")) or any(k in str(map_title) for k in ["耕地", "林地", "森林", "草地", "灌木", "水体", "裸地", "建设用地"])
        result_paths["map_scope"] = scope_payload.get("map_scope") or result_paths.get("map_scope") or "full_domain"
        result_paths["mask_by_landcover"] = bool(scope_payload.get("mask_by_landcover"))
        result_paths["landcover_class_code"] = scope_payload.get("landcover_class_code")
        result_paths["landcover_class_label"] = scope_payload.get("landcover_class_label")
        png_final = _make_prediction_png(
            result_paths.get("pred_tif") or pred_src,
            png_dst,
            mask_tif_path=(mask_final or None),
            title=map_title,
            noncrop_color=(None if target_landcover_only else "default"),
        )
        if png_final:
            result_paths["preview_png"] = png_final
        else:
            warnings.append("未生成PNG预览图。")

        txt_final = _copy_file(txt_src, txt_dst)
        if txt_final:
            result_paths["internal_model_record_txt"] = str(txt_src)
            result_paths["model_record_txt"] = txt_final
        else:
            # Create a minimal txt if model record is unavailable.
            try:
                txt_dst.write_text("土壤有机质制图模型记录\n\n模型已运行，但未找到完整模型记录源文件。请查看内部报告。\n", encoding="utf-8")
                result_paths["model_record_txt"] = str(txt_dst)
            except Exception:
                warnings.append("未找到模型记录txt。")

        result_paths["final_result_dir"] = str(final_dir)
        if warnings:
            result_paths["publish_warnings"] = warnings
    except Exception as exc:
        result_paths.setdefault("publish_warnings", []).append(f"结果整理失败：{exc}")
    return result_paths


def _finalize_success(task_id: str, result_paths: dict):
    if _task_is_cancelled(task_id):
        return
    result_paths = attach_audit_paths(result_paths or {}, task_id)
    pro_console_log("MODEL", "任务成功完成", {"result_paths": result_paths}, task_id=task_id)
    existing = {}
    task = TASKS.get(task_id)
    if task and isinstance(task.result_paths, dict):
        existing.update(task.result_paths)
    existing.update(result_paths or {})
    try:
        rec = append_step_record(task_id, kind=(task.kind if task else ""), status="done", progress=100, stage="已完成", message="任务成功完成", event="finish")
        if task:
            task.step_records.append(rec)
            task.step_records = task.step_records[-300:]
            existing.update(audit_paths(task_id))
    except Exception:
        pass
    TASKS.update(task_id, status="done", progress=100, stage="已完成", result_paths=existing, step_audit_paths=audit_paths(task_id), finished_at=time.time())


def _finalize_error(task_id: str, error: str):
    if _task_is_cancelled(task_id):
        return
    pro_console_log("MODEL", "任务运行失败", {"error": error}, task_id=task_id)
    task = TASKS.get(task_id)
    try:
        rec = append_step_record(task_id, kind=(task.kind if task else ""), status="error", progress=100, stage="运行失败", message=str(error), event="error")
        if task:
            task.step_records.append(rec)
            task.step_records = task.step_records[-300:]
    except Exception:
        pass
    # Mark progress as finished so the UI does not stay at an early percentage such as 4%.
    # The status/stage communicates failure; the completed bar communicates that the backend stopped.
    TASKS.update(task_id, status="error", progress=100, stage="运行失败", error=error, step_audit_paths=audit_paths(task_id), finished_at=time.time())
    try:
        TASKS.append_log(task_id, "[ERROR] " + str(error))
    except Exception:
        pass


def _parse_trial_progress(line: str, prefix: str, start: int, span: int):
    m = re.search(rf"\[{prefix}\]\s*trial\s*(\d+)\s*/\s*(\d+)", line, flags=re.I)
    if not m:
        return None
    i, n = int(m.group(1)), int(m.group(2))
    if n <= 0:
        return None
    p = start + int(span * i / n)
    return min(max(p, start), start + span)


def _rfk_progress_from_line(line: str):
    s = line.lower()
    if "开始 rfk 正式精简建模" in s:
        return 5, "任务已创建"
    if "样点 csv" in s:
        return 16, "读取样点与栅格"
    if "固定特征数" in s or "固定 rf 参数" in s:
        return 28, "检查变量与路径"
    if "重复空间 7:3 | fold=" in s or "strict groupkfold | fold=" in s:
        return 48, "模型计算"
    if "开始输出全区 rfk 预测图" in s:
        return 78, "生成预测图"
    if "gcp输入" in s or "strict_groupkfold_oof" in s or "manifest" in s:
        return 90, "整理输出"
    if "全部完成" in s:
        return 100, "完成"
    p = _parse_trial_progress(line, "RFK", 20, 50)
    if p is not None:
        return p, "模型计算"
    return None


def _gcp_progress_from_line(line: str):
    s = line.lower()
    if "启动 gcp + aoa" in s or "开始 gcp" in s:
        return 5, "任务已创建"
    if "预测栅格" in s or "校准样点表" in s or "读取本次 rfk 结果" in s:
        return 18, "读取制图结果与校准样点"
    if "gcp 区间" in s or "预测区间" in s:
        return 42, "构建 GCP 预测区间"
    if "读取预测栅格" in s:
        return 56, "读取预测栅格"
    if "计算 aoa" in s or "适用域" in s:
        return 72, "计算 AOA 适用域"
    if "综合风险" in s or "风险分区" in s:
        return 86, "生成综合风险分区"
    if "分析完成" in s or "全部完成" in s:
        return 100, "完成"
    p = _parse_trial_progress(line, "GCP", 18, 50)
    if p is not None:
        return p, "构建 GCP 预测区间"
    return None



def _task_is_cancelled(task_id: str) -> bool:
    try:
        t = TASKS.get(task_id)
        return bool(t and (t.cancel_requested or t.status in {"cancelled", "canceled"}))
    except Exception:
        return False


def _finalize_cancelled(task_id: str, reason: str = "当前任务已停止"):
    pro_console_log("MODEL", "任务已按用户指令停止", {"reason": reason}, task_id=task_id)
    task = TASKS.get(task_id)
    try:
        rec = append_step_record(task_id, kind=(task.kind if task else ""), status="cancelled", progress=100, stage="已停止", message=str(reason), event="cancel")
        if task:
            task.step_records.append(rec)
            task.step_records = task.step_records[-300:]
    except Exception:
        pass
    TASKS.update(task_id, status="cancelled", progress=100, stage="已停止", error=None, cancel_requested=True, step_audit_paths=audit_paths(task_id), finished_at=time.time())
    try:
        TASKS.append_log(task_id, "[CANCEL] " + str(reason))
    except Exception:
        pass

def terminate_task(task_id: str, reason: str = "任务已被终止"):
    """Cancel a running task.

    External RFK/GCP subprocesses are terminated. In-process formal mapping tasks
    cannot be safely killed by Python, so they are marked cancelled immediately;
    their eventual success/error finalization is ignored and the UI is released.
    """
    proc = None
    TASKS.update(task_id, cancel_requested=True)
    with RUNNING_LOCK:
        proc = RUNNING_PROCS.get(task_id)
    if proc is not None:
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()
        except Exception:
            pass
        finally:
            with RUNNING_LOCK:
                RUNNING_PROCS.pop(task_id, None)
    _finalize_cancelled(task_id, reason)



def terminate_session_tasks(session_id: str, reason: str = "会话已关闭，后台任务已终止"):
    if not session_id:
        return
    with RUNNING_LOCK:
        task_ids = [task_id for task_id, proc in RUNNING_PROCS.items() if (TASKS.get(task_id) and TASKS.get(task_id).session_id == session_id)]
    for task_id in task_ids:
        terminate_task(task_id, reason=reason)


def cancel_session_tasks(session_id: str, kinds: set[str] | None = None, reason: str = "用户已停止当前任务") -> list[str]:
    """Cancel queued/running tasks for this session and return cancelled ids."""
    if not session_id:
        return []
    kinds = set(kinds or [])
    cancelled: list[str] = []
    try:
        with TASKS._lock:
            active = [
                t for t in TASKS._tasks.values()
                if t.session_id == session_id and t.status in {"queued", "running"} and (not kinds or t.kind in kinds)
            ]
        for t in active:
            terminate_task(t.task_id, reason=reason)
            cancelled.append(t.task_id)
    except Exception:
        pass
    return cancelled

def terminate_all_running_tasks(reason: str = "浏览器已关闭，后台任务已终止"):
    with RUNNING_LOCK:
        task_ids = list(RUNNING_PROCS.keys())
    for task_id in task_ids:
        terminate_task(task_id, reason=reason)


def reset_runtime_state():
    terminate_all_running_tasks(reason="服务重置，后台任务已终止")
    TASKS.reset()


@contextmanager
def _temporary_mapping_source_env(choice: str, local_covariate_dir: str | None = None, uploaded_covariate_files: list[str] | None = None):
    """Per-task env switch for full mapping source selection.

    This is a local service. Only one long mapping task should run at a time;
    the context restores previous env values after the task finishes.
    """
    c = str(choice or "").strip().lower()
    keys = [
        "PRO_DOWNLOAD_ONLY_MODE", "PRO_GEE_ENABLE", "PRO_GEE_PRIORITY_MODE", "PRO_GEE_REQUIRE_SUCCESS",
        "PRO_SOURCE_MODE", "PRO_DOMESTIC_ONLY", "PRO_FORCE_DOMESTIC_PLATFORM_DOWNLOAD",
        "PRO_DOMESTIC_TRUE_DATA_MIN_DOWNLOADS", "PRO_DOMESTIC_FALLBACK_FOR_MISSING_GEE",
        "PRO_DOMESTIC_FALLBACK_REQUIRED", "PRO_SKIP_DOMESTIC_WEB_AFTER_LOCAL_SUCCESS",
        "PRO_PLATFORM_PROBE_AFTER_GEE_SUCCESS", "PRO_DOMESTIC_RAW_TARGETS",
        "PRO_DOMESTIC_FALLBACK_TARGETS", "PRO_DOMESTIC_MAX_PLATFORM_CANDIDATES",
        "PRO_DOMESTIC_TRY_MULTIPLE_PLATFORMS", "DOMESTIC_TPDC_ONLY", "DOMESTIC_PLATFORM_ONLY",
        "PRO_DOMESTIC_DISABLED_PLATFORMS", "PRO_DOMESTIC_FALLBACK_LOCAL_DIR",
        "PRO_LOCAL_COVARIATES_ALWAYS_LOAD", "PRO_INCLUDE_DOMESTIC_COVARIATES_IN_FORMAL_MODEL",
        "PRO_DOMESTIC_RAW_CRAWL", "PRO_FOREIGN_DOWNLOAD_MODE",
        "PRO_USER_UPLOADED_COVARIATE_FILES_JSON",
    ]
    old = {k: os.environ.get(k) for k in keys}
    try:
        os.environ["PRO_DOWNLOAD_ONLY_MODE"] = "0"
        # V166: persist explicit raster paths imported in this session so the model
        # can match them to fused CSV feature columns even when they are not in the
        # CSV parent folder.
        try:
            import json as _json
            _ufs = [str(x) for x in (uploaded_covariate_files or []) if str(x).strip()]
            if _ufs:
                os.environ["PRO_USER_UPLOADED_COVARIATE_FILES_JSON"] = _json.dumps(_ufs, ensure_ascii=False)
        except Exception:
            pass
        if c == "user":
            # User-data-only baseline: use uploaded/local covariate rasters only, no TPDC/GEE download.
            os.environ.update({
                "PRO_GEE_ENABLE": "0",
                "PRO_GEE_PRIORITY_MODE": "0",
                "PRO_GEE_REQUIRE_SUCCESS": "0",
                "PRO_SOURCE_MODE": "user_uploaded_only",
                "PRO_DOMESTIC_ONLY": "0",
                "PRO_FORCE_DOMESTIC_PLATFORM_DOWNLOAD": "0",
                "PRO_DOMESTIC_TRUE_DATA_MIN_DOWNLOADS": "0",
                "PRO_DOMESTIC_FALLBACK_FOR_MISSING_GEE": "0",
                "PRO_DOMESTIC_FALLBACK_REQUIRED": "0",
                "PRO_SKIP_DOMESTIC_WEB_AFTER_LOCAL_SUCCESS": "1",
                "PRO_PLATFORM_PROBE_AFTER_GEE_SUCCESS": "0",
                "PRO_LOCAL_COVARIATES_ALWAYS_LOAD": "1",
                "PRO_INCLUDE_DOMESTIC_COVARIATES_IN_FORMAL_MODEL": "1",
                "PRO_DOMESTIC_RAW_CRAWL": "0",
                "PRO_FOREIGN_DOWNLOAD_MODE": "metadata_only",
            })
            if local_covariate_dir:
                os.environ["PRO_DOMESTIC_FALLBACK_LOCAL_DIR"] = str(local_covariate_dir)
        elif c == "gee":
            # GEE 已不作为本系统默认/推荐数据源；为兼容旧指令，降级为用户数据流程。
            os.environ.update({
                "PRO_GEE_ENABLE": "0",
                "PRO_GEE_PRIORITY_MODE": "0",
                "PRO_GEE_REQUIRE_SUCCESS": "0",
                "PRO_SOURCE_MODE": "user_uploaded_only",
                "PRO_DOMESTIC_ONLY": "0",
                "PRO_FORCE_DOMESTIC_PLATFORM_DOWNLOAD": "0",
                "PRO_DOMESTIC_TRUE_DATA_MIN_DOWNLOADS": "0",
                "PRO_DOMESTIC_FALLBACK_FOR_MISSING_GEE": "0",
                "PRO_DOMESTIC_FALLBACK_REQUIRED": "0",
                "PRO_SKIP_DOMESTIC_WEB_AFTER_LOCAL_SUCCESS": "1",
                "PRO_PLATFORM_PROBE_AFTER_GEE_SUCCESS": "0",
                "PRO_LOCAL_COVARIATES_ALWAYS_LOAD": "1",
                "PRO_INCLUDE_DOMESTIC_COVARIATES_IN_FORMAL_MODEL": "1",
                "PRO_DOMESTIC_RAW_CRAWL": "0",
                "PRO_FOREIGN_DOWNLOAD_MODE": "metadata_only",
            })
            if local_covariate_dir:
                os.environ["PRO_DOMESTIC_FALLBACK_LOCAL_DIR"] = str(local_covariate_dir)
        else:
            # 国内/TPDC模式：本地/用户数据优先；如需下载，只走国内平台，不启用GEE。
            os.environ.update({
                "PRO_GEE_ENABLE": "0",
                "PRO_GEE_PRIORITY_MODE": "0",
                "PRO_SOURCE_MODE": "domestic_first",
                "PRO_DOMESTIC_ONLY": "0",
                "PRO_FORCE_DOMESTIC_PLATFORM_DOWNLOAD": "0",
                "PRO_DOMESTIC_TRUE_DATA_MIN_DOWNLOADS": "0",
                "PRO_DOMESTIC_FALLBACK_FOR_MISSING_GEE": "0",
                "PRO_DOMESTIC_FALLBACK_REQUIRED": "0",
                "PRO_SKIP_DOMESTIC_WEB_AFTER_LOCAL_SUCCESS": "1",
                "PRO_PLATFORM_PROBE_AFTER_GEE_SUCCESS": "1",
                "PRO_LOCAL_COVARIATES_ALWAYS_LOAD": "1",
                "PRO_INCLUDE_DOMESTIC_COVARIATES_IN_FORMAL_MODEL": "1",
                "PRO_DOMESTIC_RAW_TARGETS": "tpdc",
                "PRO_DOMESTIC_FALLBACK_TARGETS": "tpdc",
                "PRO_DOMESTIC_MAX_PLATFORM_CANDIDATES": "1",
                "PRO_DOMESTIC_TRY_MULTIPLE_PLATFORMS": "0",
                "DOMESTIC_TPDC_ONLY": "1",
                "DOMESTIC_PLATFORM_ONLY": "tpdc",
                "PRO_DOMESTIC_DISABLED_PLATFORMS": "gscloud,geodata,resdc,noda,nesdc,ngac,worldclim,isric",
                "PRO_DOMESTIC_RAW_CRAWL": "0",
                "PRO_FOREIGN_DOWNLOAD_MODE": "metadata_only",
            })
            if local_covariate_dir and os.getenv("PRO_DOMESTIC_FALLBACK_LOCAL_DIR", "").strip() == "":
                os.environ["PRO_DOMESTIC_FALLBACK_LOCAL_DIR"] = str(local_covariate_dir)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v




def _uploaded_covariate_file_paths(uploaded_files: list[dict] | None) -> list[str]:
    """V166: collect rasters/NetCDF/HDF files explicitly uploaded/imported in the session."""
    out: list[str] = []
    seen: set[str] = set()
    for f in uploaded_files or []:
        p = str((f or {}).get("path") or "").strip()
        if not p:
            continue
        low = p.lower()
        if not low.endswith((".tif", ".tiff", ".nc", ".hdf", ".h5", ".hdf5", ".img")):
            continue
        if p not in seen:
            seen.add(p); out.append(p)
    return out


def _run_pro_rfk_with_download(task_id: str, cmd: list[str], result_paths: dict, cwd: str | None, request_text: str, session_id: str | None, user_id: str | None = None, uploaded_files: list[dict] | None = None, data_source: str = "pro_web"):
    """Pro 版：先自动数据准备某年某地的制图数据，再复用制图流程。"""
    try:
        if _task_is_cancelled(task_id):
            return
        _set_stage(task_id, 4, "Pro 自动数据准备")
        TASKS.append_log(task_id, "[PRO] 已进入 Pro 自动数据准备流程。")
        TASKS.append_log(task_id, "[PRO] 开始检查用户上传样点文件：CSV/Excel/TIF 均会打印读取状态。")
        if uploaded_files:
            TASKS.append_log(task_id, "[PRO] 当前会话检测到上传文件：" + "；".join([f"{x.get('name')}({x.get('size')} bytes)" for x in uploaded_files]))
        else:
            TASKS.append_log(task_id, "[PRO] 当前会话未检测到上传文件；正式制图需要用户上传样点CSV/Excel。")

        # V85 正式主线：用户上传样点，系统自动从 GEE 获取环境协变量并完成建模制图。
        # 如需回到旧调试流程，可设置 PRO_RUN_MODE=test。
        run_mode = (os.getenv("PRO_RUN_MODE", "formal") or "formal").strip().lower()
        platform_flag = os.getenv("PRO_PLATFORM_PIPELINE_TEST")
        use_platform_pipeline = (platform_flag == "1") or (platform_flag is None and bool(uploaded_files)) or run_mode == "formal"
        if use_platform_pipeline:
            sample_path = None
            for f in (uploaded_files or []):
                p = f.get("path") or ""
                if str(p).lower().endswith((".csv", ".txt", ".xls", ".xlsx")):
                    sample_path = p
                    break
            sample_path = sample_path or os.getenv("PRO_SAMPLE_PATH") or os.getenv("PRO_TEST_SAMPLE_PATH")
            if not sample_path:
                raise RuntimeError("正式制图需要CSV/Excel样点文件。请先上传包含 lon/lat/有机质 字段的样点，或配置 PRO_SAMPLE_PATH。")
            mapping_choice = choice_from_data_source(data_source, request_text)
            if mapping_choice == "user":
                TASKS.append_log(task_id, "[PRO-FORMAL] 已启用用户数据优先基线制图：用户上传样点 + 用户上传环境协变量 -> 模型训练 -> GeoTIFF输出；不主动下载缺失变量。")
            elif mapping_choice == "domestic":
                TASKS.append_log(task_id, "[PRO-FORMAL] 已启用用户数据优先 + 国内数据平台补齐：用户上传样点/协变量 -> 国家青藏高原科学数据中心（TPDC）补缺失项 -> 必要时国内/本地协变量补齐 -> 模型训练。")
            else:
                TASKS.append_log(task_id, "[PRO-FORMAL] 已启用用户数据优先 + 国内/本地协变量补齐：用户上传样点/协变量 -> GEE自动补缺失项 -> 模型训练 -> GeoTIFF输出。")
            # V182: keep internal RFK working files away from the user-facing result root.
            # The clean result root remains E:\Agent_DSM\制图结果 and is populated only by
            # _publish_final_mapping_outputs with 有机质图.tif/png and 模型记录.txt.
            output_root = Path(os.getenv("PRO_INTERNAL_RUN_ROOT", str(Path.cwd() / "data" / "formal_mapping_runs")))
            if not output_root.is_absolute():
                output_root = Path.cwd() / output_root
            upload_cov_dir = str(Path(sample_path).parent) if sample_path else None
            uploaded_cov_files = _uploaded_covariate_file_paths(uploaded_files)
            if uploaded_cov_files:
                TASKS.append_log(task_id, "[PRO-FORMAL][V166] 已接入本会话显式上传/导入的预测栅格：" + "；".join(uploaded_cov_files[:30]))
            with _task_heartbeat(task_id, label="正式制图", interval_sec=None):
                with _temporary_mapping_source_env(mapping_choice, local_covariate_dir=upload_cov_dir, uploaded_covariate_files=uploaded_cov_files):
                    result = run_platform_model_csv_pipeline(output_root, request_text, sample_path, task_id=task_id)
            if _task_is_cancelled(task_id):
                return
            model_record_txt = ""
            model_report_data: dict[str, Any] = {}
            try:
                if result.model_report_json:
                    model_report_data = json.loads(Path(result.model_report_json).read_text(encoding="utf-8"))
                    model_record_txt = str(model_report_data.get("model_record_txt") or "")
            except Exception:
                model_record_txt = ""
                model_report_data = {}
            model_outputs = model_report_data.get("outputs") or {}
            result_paths_done = {
                "formal_mapping_work_dir": result.work_dir,
                "model_ready_csv": result.model_ready_csv,
                "platform_report_csv": result.platform_report_csv,
                "platform_report_json": result.platform_report_json,
                "pro_data_report_md": result.report_md,
                "manifest": result.model_report_json or "",
                "report_json": result.model_report_json or "",
                "pred_tif": result.pred_tif or "",
                "strict_oof_csv": model_outputs.get("strict_oof_csv") or model_report_data.get("strict_oof_csv") or "",
                "oof_csv": model_outputs.get("oof_csv") or model_outputs.get("strict_oof_csv") or model_report_data.get("strict_oof_csv") or "",
                "calibration_csv": model_outputs.get("calibration_csv") or model_outputs.get("strict_oof_csv") or model_report_data.get("strict_oof_csv") or "",
                "prediction_grid_covariates_csv": model_outputs.get("prediction_grid_covariates_csv") or model_outputs.get("prediction_grid_csv") or model_report_data.get("prediction_grid_csv") or "",
                "grid_covariates_csv": model_outputs.get("grid_covariates_csv") or model_outputs.get("prediction_grid_csv") or model_report_data.get("prediction_grid_csv") or "",
                "model_record_txt": model_record_txt,
            }
            # Publish a clean, short, user-facing result folder.
            result_paths_done = _publish_final_mapping_outputs(result_paths_done, request_text=request_text)
            TASKS.append_log(task_id, "[PRO-FORMAL] 最终建模CSV：" + result.model_ready_csv)
            if result_paths_done.get("final_result_dir"):
                TASKS.append_log(task_id, "[PRO-FORMAL] 用户结果目录：" + str(result_paths_done.get("final_result_dir")))
            if result_paths_done.get("pred_tif"):
                TASKS.append_log(task_id, "[PRO-FORMAL] 有机质图：" + str(result_paths_done.get("pred_tif")))
            if result_paths_done.get("model_record_txt"):
                TASKS.append_log(task_id, "[PRO-FORMAL] 模型记录：" + str(result_paths_done.get("model_record_txt")))
            if result_paths_done.get("strict_oof_csv"):
                TASKS.append_log(task_id, "[PRO-FORMAL] GCP + AOA 校准样点表：" + str(result_paths_done.get("strict_oof_csv")))
            _finalize_success(task_id, result_paths_done)
            return

        manifest = prepare_online_data_manifest(session_id=session_id, user_text=request_text, user_id=user_id, uploaded_files=uploaded_files)
        target = manifest.get("target") or {}
        TASKS.append_log(task_id, f"[PRO] 目标识别结果：year={target.get('year')}，location={target.get('location')}，raw={target}。")
        sample_check = manifest.get("sample_check") or {}
        if sample_check:
            TASKS.append_log(task_id, "[PRO] 样点读取/检查结果：" + json.dumps(sample_check, ensure_ascii=False)[:1200])
        data_sources = manifest.get("data_sources") or []
        if data_sources:
            TASKS.append_log(task_id, f"[PRO] 数据源记录数量：{len(data_sources)}。")
            for i, rec in enumerate(data_sources[:30], start=1):
                TASKS.append_log(task_id, "[PRO][DATA_SOURCE_%02d] %s | %s | %s | %s" % (i, rec.get("source"), rec.get("name"), rec.get("status"), rec.get("url") or rec.get("local_path") or rec.get("message")))
        total = max(len(PRO_DATA_STEPS), 1)
        for idx, step in enumerate(PRO_DATA_STEPS, start=1):
            progress = 4 + int(14 * idx / total)
            _set_stage(task_id, progress, "Pro 自动数据准备")
            TASKS.append_log(task_id, f"[PRO] {step}")
            time.sleep(0.12)
        TASKS.append_log(task_id, "[PRO] 数据准备 manifest：" + str((manifest.get("outputs") or {}).get("manifest", "")))
        TASKS.update(task_id, result_paths={
            "pro_data_manifest": str((manifest.get("outputs") or {}).get("manifest", "")),
            "pro_data_work_dir": str((manifest.get("outputs") or {}).get("work_dir", "")),
            "pro_target_year": str(target.get("year") or ""),
            "pro_target_location": str(target.get("location") or ""),
        })
        report_path = str((manifest.get("outputs") or {}).get("report_md", ""))
        if report_path:
            TASKS.append_log(task_id, "[PRO] 数据可用性报告：" + report_path)
        online_mode = (manifest.get("online_mode") or "real").lower()
        run_rfk_after_data = os.getenv("PRO_RUN_RFK_AFTER_DATA", "0") == "1"
        if online_mode in {"public_test", "test", "real"} and not run_rfk_after_data:
            TASKS.append_log(task_id, "[PRO] 诊断模式：已完成公开数据下载/解析验证，未继续启动制图。若要继续制图，设置 PRO_RUN_RFK_AFTER_DATA=1。")
            _finalize_success(task_id, {
                "pro_data_manifest": str((manifest.get("outputs") or {}).get("manifest", "")),
                "pro_data_report_md": report_path,
                "pro_data_work_dir": str((manifest.get("outputs") or {}).get("work_dir", "")),
                "pro_target_year": str(target.get("year") or ""),
                "pro_target_location": str(target.get("location") or ""),
            })
            return

        mainline_diagnostic = os.getenv("PRO_MAINLINE_DIAGNOSTIC_MODEL", "0") == "1" and run_mode != "formal"
        if online_mode in {"public_test", "test", "real"} and run_rfk_after_data and mainline_diagnostic:
            TASKS.append_log(task_id, "[PRO-DIAG] 已启用旧版连通性诊断模型：仅用于调试，不作为正式制图成果。")
            diagnostic_paths = run_pro_mainline_diagnostic_model(
                (manifest.get("outputs") or {}).get("manifest", ""),
                target=target,
                task_id=task_id,
            )
            merged_paths = {
                "pro_data_manifest": str((manifest.get("outputs") or {}).get("manifest", "")),
                "pro_data_report_md": report_path,
                "pro_data_work_dir": str((manifest.get("outputs") or {}).get("work_dir", "")),
                "pro_target_year": str(target.get("year") or ""),
                "pro_target_location": str(target.get("location") or ""),
            }
            merged_paths.update(diagnostic_paths)
            TASKS.append_log(task_id, "[PRO-DIAG] 旧版诊断图已输出：" + str(diagnostic_paths.get("pred_tif", "")))
            _finalize_success(task_id, merged_paths)
            return

        TASKS.append_log(task_id, "[PRO] 正式运行：已生成数据准备清单；正式建模前需用户上传样点并完成协变量栅格化。")
        TASKS.append_log(task_id, "[PRO] 已允许继续启动 制图脚本。")
    except Exception as exc:
        _finalize_error(task_id, "Pro 自动数据准备失败：" + str(exc))
        return
    if _task_is_cancelled(task_id):
        return
    _run_command_stream(task_id, cmd, "rfk", result_paths, cwd)

def _run_command_stream(task_id: str, cmd: list[str], kind: str, result_paths: dict, cwd: str | None = None):
    proc = None
    try:
        pro_console_log("MODEL", "开始启动外部制图/分析脚本", {"kind": kind, "cwd": cwd, "cmd": cmd}, task_id=task_id)
        try:
            t = TASKS.get(task_id)
            rec = append_step_record(task_id, kind=kind, status=(t.status if t else "running"), progress=(t.progress if t else 0), stage="启动外部脚本", message="启动外部制图/分析脚本", event="process_start", extra={"cwd": cwd, "cmd": cmd})
            if t:
                t.step_records.append(rec); t.step_records = t.step_records[-300:]; TASKS.update(task_id, step_records=t.step_records, step_audit_paths=audit_paths(task_id))
        except Exception:
            pass
        TASKS.append_log(task_id, f"[CMD] cwd={cwd or ''}")
        TASKS.append_log(task_id, "[CMD] " + " ".join(map(str, cmd)))
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            bufsize=1,
        )
        with RUNNING_LOCK:
            RUNNING_PROCS[task_id] = proc
        TASKS.update(task_id, pid=proc.pid)
        pro_console_log("MODEL", "外部脚本进程已启动", {"pid": proc.pid, "kind": kind}, task_id=task_id)
        parser = _rfk_progress_from_line if kind == "rfk" else _gcp_progress_from_line
        if proc.stdout is None:
            raise RuntimeError("无法获取脚本标准输出。")
        with _task_heartbeat(task_id, label=f"{kind.upper()}外部脚本", interval_sec=None):
            for line in proc.stdout:
                if _task_is_cancelled(task_id):
                    try:
                        if proc.poll() is None:
                            proc.terminate()
                    except Exception:
                        pass
                    return
                line = line.rstrip()
                if line:
                    TASKS.append_log(task_id, line)
                    parsed = parser(line)
                    if parsed:
                        _set_stage(task_id, *parsed)
            proc.wait()

        if _task_is_cancelled(task_id):
            return
        if proc.returncode != 0:
            tail = "\n".join((TASKS.get(task_id).logs or [])[-20:]) if TASKS.get(task_id) else ""
            msg = f"{kind.upper()} 脚本返回非零状态码：{proc.returncode}"
            if tail:
                msg += "\n最近日志：\n" + tail
            _finalize_error(task_id, msg)
            return

        resolved_paths = _resolve_result_paths(kind, result_paths)
        try:
            t = TASKS.get(task_id)
            rec = append_step_record(task_id, kind=kind, status=(t.status if t else "running"), progress=(t.progress if t else 95), stage="校验输出文件", message="脚本执行结束，正在校验输出文件", event="validate_outputs", extra={"outputs": resolved_paths})
            if t:
                t.step_records.append(rec); t.step_records = t.step_records[-300:]; TASKS.update(task_id, step_records=t.step_records, step_audit_paths=audit_paths(task_id))
        except Exception:
            pass
        missing = [str(p) for p in resolved_paths.values() if not Path(p).exists()]
        if missing:
            _finalize_error(task_id, "脚本运行完成，但未找到输出文件：" + "；".join(missing))
            return

        _finalize_success(task_id, resolved_paths)
    except Exception as e:
        _finalize_error(task_id, str(e))
    finally:
        with RUNNING_LOCK:
            RUNNING_PROCS.pop(task_id, None)
        if proc is not None and proc.stdout is not None:
            try:
                proc.stdout.close()
            except Exception:
                pass


def start_rfk_task(data_source: str = "default", session_id: str | None = None, request_text: str = "", user_id: str | None = None, uploaded_files: list[dict] | None = None) -> str:
    existing = _find_active_task("rfk", session_id)
    if existing is not None:
        pro_console_log("MODEL", "检测到同一会话已有制图任务在运行，已复用现有任务，避免重复制图。", {"session_id": session_id, "existing_task_id": existing.task_id, "status": existing.status}, task_id=existing.task_id)
        TASKS.append_log(existing.task_id, "[GUARD] 检测到重复提交，已复用当前制图任务，未启动第二个后台任务。")
        return existing.task_id
    task = TASKS.create("rfk", session_id=session_id)
    pro_console_log("MODEL", "收到土壤有机质制图任务", {"data_source": data_source, "session_id": session_id, "request_text": request_text, "uploaded_files": uploaded_files}, task_id=task.task_id)
    _set_stage(task.task_id, 2, "准备启动土壤有机质制图")

    # V97 server-side guard: UI should block cross-region mismatch before this point,
    # but keep a backend stop to prevent hidden/duplicate routes from starting mapping.
    if os.getenv("PRO_STRICT_AOI_PREFLIGHT", "1") == "1" and uploaded_files:
        try:
            preflight = preflight_mapping_region(request_text, uploaded_files)
            if preflight.get("blocked"):
                _finalize_error(task.task_id, preflight.get("message") or "样点数据与制图区域不匹配，已停止制图。")
                return task.task_id
            TASKS.update(task.task_id, result_paths={"aoi_preflight": preflight})
        except Exception as exc:
            _finalize_error(task.task_id, f"AOI-样点一致性预检失败，已停止制图：{exc}")
            return task.task_id

    # 防御性兜底：V85 PRO 目标是“用户上传 SOM 样点 + 系统自动准备协变量”。
    # 如果 UI/LLM 仍把任务路由成 user，而用户数据重跑模板又没有配置，优先切到 pro_web，
    # 避免直接报“未配置 USER_RFK_COMMAND_TEMPLATE”。
    online_mode = os.getenv("PRO_ONLINE_DATA_MODE", "real").strip().lower()
    force_uploaded_sample_to_pro = os.getenv("PRO_UPLOADED_SAMPLE_TO_PRO_WEB", "1") == "1"
    if (
        data_source == "user"
        and uploaded_files
        and force_uploaded_sample_to_pro
        and (not ENABLE_CUSTOM_USER_RUN or not USER_RFK_COMMAND_TEMPLATE)
        and online_mode in {"public_test", "test", "real", "real"}
    ):
        pro_console_log(
            "ROUTE",
            "用户上传样点但未配置旧版制图命令模板，已切换到 PRO 用户数据制图流程：仅使用用户上传/本地协变量，不启用 GEE。",
            {
                "old_data_source": "user",
                "new_data_source": "pro_user",
                "online_mode": online_mode,
                "uploaded_count": len(uploaded_files or []),
            },
            task_id=task.task_id,
        )
        data_source = "pro_user"

    is_pro_data_source = str(data_source or "").startswith("pro_") or data_source == "pro_web"
    if data_source == "user" and (not ENABLE_CUSTOM_USER_RUN or not USER_RFK_COMMAND_TEMPLATE):
        _finalize_error(task.task_id, "当前未配置用户数据重跑 制图命令模板。")
        return task.task_id
    preflight_error = _preflight_script("rfk", skip_input_paths=is_pro_data_source)
    if preflight_error:
        _finalize_error(task.task_id, preflight_error)
        return task.task_id
    cmd = [PYTHON_EXECUTABLE, str(RFK_SCRIPT)]
    if is_pro_data_source:
        thread = threading.Thread(
            target=_run_pro_rfk_with_download,
            args=(task.task_id, cmd, DEFAULT_RFK_OUTPUTS, str(RFK_SCRIPT.parent), request_text, session_id, user_id, uploaded_files, data_source),
            daemon=False,
        )
    else:
        thread = threading.Thread(
            target=_run_command_stream,
            args=(task.task_id, cmd, "rfk", DEFAULT_RFK_OUTPUTS, str(RFK_SCRIPT.parent)),
            daemon=False,
        )
    thread.start()
    return task.task_id


def _resolve_auto_gee_gcp_outputs(rfk_result_paths: dict | None) -> dict:
    """Return GCP-compatible result paths if automatic uncertainty artifacts exist."""
    paths = rfk_result_paths or {}
    out: dict[str, str] = {}
    # Direct keys collected from the RFK/PRO task.
    direct = {
        "report_json": paths.get("gee_gcp_report_json"),
        "width_tif": paths.get("gee_gcp_width_tif"),
        "lower_tif": paths.get("gee_gcp_lower_tif"),
        "upper_tif": paths.get("gee_gcp_upper_tif"),
        "half_width_q_tif": paths.get("gee_gcp_half_width_q_tif"),
    }
    # Fallback: infer from pred_tif directory.
    if not direct.get("width_tif") and paths.get("pred_tif"):
        try:
            d = Path(paths["pred_tif"]).parent
            direct = {
                "report_json": str(d / "gee_gcp_report.json"),
                "width_tif": str(d / "gee_gcp_width.tif"),
                "lower_tif": str(d / "gee_gcp_lower.tif"),
                "upper_tif": str(d / "gee_gcp_upper.tif"),
                "half_width_q_tif": str(d / "gee_gcp_half_width_q.tif"),
            }
        except Exception:
            pass
    for k, v in direct.items():
        if v and Path(str(v)).exists():
            out[k] = str(v)
    if out.get("width_tif") and out.get("report_json"):
        out.setdefault("source", "v84_auto_gee_gcp")
    return out


def start_gcp_task(rfk_result_paths: dict | None = None, session_id: str | None = None) -> str:
    existing = _find_active_task("gcp", session_id)
    if existing is not None:
        pro_console_log("MODEL", "检测到同一会话已有GCP任务在运行，已复用现有任务。", {"session_id": session_id, "existing_task_id": existing.task_id, "status": existing.status}, task_id=existing.task_id)
        TASKS.append_log(existing.task_id, "[GUARD] 检测到重复提交，已复用当前GCP任务，未启动第二个后台任务。")
        return existing.task_id
    task = TASKS.create("gcp", session_id=session_id)
    _set_stage(task.task_id, 2, "准备启动 GCP + AOA 不确定性分析")

    # V188：不确定性分析只在用户明确提出 GCP/AOA/不确定性分析指令后执行。
    # 制图任务本身只负责输出预测栅格、PNG、模型与精度报告；即便制图目录中存在
    # 历史 GCP 文件，也不会在这里自动绑定为不确定性结果，避免“开始制图”被误执行为
    # “制图 + 不确定性分析”。

    preflight_error = _preflight_script("gcp")
    if preflight_error:
        _finalize_error(task.task_id, preflight_error)
        return task.task_id
    preflight_error = _preflight_gcp_from_rfk_result(rfk_result_paths)
    if preflight_error:
        _finalize_error(task.task_id, preflight_error)
        return task.task_id

    resolved_inputs = _resolve_gcp_input_paths(rfk_result_paths)
    manifest_path = str(resolved_inputs.get("manifest") or "")
    pred_tif = str(resolved_inputs.get("pred_tif") or "")
    strict_oof_csv = str(resolved_inputs.get("strict_oof_csv") or "")
    pro_console_log("MODEL", "GCP + AOA 输入链路已解析", resolved_inputs, task_id=task.task_id)
    TASKS.append_log(task.task_id, "[GCP-AOA] 输入链路：manifest=" + manifest_path + "；pred_tif=" + pred_tif + "；strict_oof_csv=" + strict_oof_csv)

    out_dir = new_uncertainty_dir()
    cmd = [
        PYTHON_EXECUTABLE,
        str(GCP_SCRIPT),
        "--manifest", manifest_path,
        "--pred_tif", pred_tif,
        "--strict_oof_csv", strict_oof_csv,
        "--out_dir", str(out_dir),
    ]
    expected_outputs = {
        "manifest": str(out_dir / "_state" / "gcp_aoa_manifest.json"),
        "report_txt": str(out_dir / "01_指标报告" / "GCP_AOA分析报告.txt"),
        "report_json": str(out_dir / "01_指标报告" / "GCP_AOA指标报告.json"),
        "center_tif": str(out_dir / "02_不确定性与适用域图" / "GCP中心预测图.tif"),
        "width_tif": str(out_dir / "02_不确定性与适用域图" / "GCP预测区间宽度.tif"),
        "lower_tif": str(out_dir / "02_不确定性与适用域图" / "GCP预测区间下界.tif"),
        "upper_tif": str(out_dir / "02_不确定性与适用域图" / "GCP预测区间上界.tif"),
        "aoa_di_tif": str(out_dir / "02_不确定性与适用域图" / "AOA不相似性指数_DI.tif"),
        "aoa_inside_tif": str(out_dir / "02_不确定性与适用域图" / "AOA适用域二值图.tif"),
        "qhat_tif": str(out_dir / "02_不确定性与适用域图" / "GCP局地非一致性阈值.tif"),
        "risk_class_tif": str(out_dir / "02_不确定性与适用域图" / "GCP_AOA综合风险分区.tif"),
        "sample_audit_csv": str(out_dir / "03_样点与统计表" / "GCP_AOA_样点区间与适用域审计.csv"),
        "width_png": str(out_dir / "02_不确定性与适用域图" / "GCP预测区间宽度.png"),
        "aoa_di_png": str(out_dir / "02_不确定性与适用域图" / "AOA不相似性指数_DI.png"),
        "risk_class_png": str(out_dir / "02_不确定性与适用域图" / "GCP_AOA综合风险分区.png"),
    }
    thread = threading.Thread(
        target=_run_command_stream,
        args=(task.task_id, cmd, "gcp", expected_outputs, str(GCP_SCRIPT.parent)),
        daemon=False,
    )
    thread.start()
    return task.task_id
