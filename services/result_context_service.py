from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json

from services.model_label_service import model_label


def _safe_read_text(path: str, max_chars: int = 5000) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    try:
        return p.read_text(encoding="utf-8")[:max_chars]
    except Exception:
        return ""


def _safe_read_json(path: str) -> Dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_result_context_text(rfk_paths: Dict[str, str] | None = None, gcp_paths: Dict[str, str] | None = None) -> str:
    rfk_paths = rfk_paths or {}
    gcp_paths = gcp_paths or {}
    chunks: list[str] = []

    rfk_json = _safe_read_json(rfk_paths.get("report_json", ""))
    if rfk_json:
        main = rfk_json.get("main_result") or rfk_json.get("formal_result") or rfk_json.get("metrics") or {}
        strict = rfk_json.get("strict_diagnosis", {})
        feat = rfk_json.get("features") or rfk_json.get("feature_cols") or []
        chunks.append(f"本次制图结果摘要（模型：{model_label(rfk_paths, rfk_json)}）：")
        if main:
            chunks.append(
                f"- 主结果：R²={main.get('r2', main.get('R2', 'NA'))}, RMSE={main.get('rmse', main.get('RMSE', 'NA'))}, MAE={main.get('mae', main.get('MAE', 'NA'))}"
            )
        if strict:
            chunks.append(
                f"- 严格空间诊断：R²={strict.get('r2', 'NA')}, RMSE={strict.get('rmse', 'NA')}, MAE={strict.get('mae', 'NA')}"
            )
        if feat:
            chunks.append(f"- 当前特征数：{len(feat)}；特征：{', '.join(map(str, feat[:20]))}")

    gee_json = _safe_read_json(rfk_paths.get("gee_formal_report_json", ""))
    if not gee_json and rfk_paths.get("pred_tif"):
        try:
            gee_json = _safe_read_json(str(Path(rfk_paths["pred_tif"]).parent / "gee_formal_prediction_grid_report.json"))
        except Exception:
            gee_json = None
    if gee_json:
        stats = gee_json.get("mask_stats") or {}
        chunks.append("本次GEE完整流程审计：")
        chunks.append(f"- 行政边界裁剪：{gee_json.get('admin_boundary_applied')}；耕地掩膜：{gee_json.get('cropland_mask_applied')}；边界来源：{gee_json.get('bounds_source', 'NA')}")
        if stats:
            chunks.append(f"- 格网：行政区内={stats.get('admin_valid_cells', 'NA')}，耕地={stats.get('cropland_valid_cells', 'NA')}，最终有效={stats.get('final_valid_cells', 'NA')}，NoData={stats.get('final_nodata_cells', 'NA')}")

    auto_gcp_json = _safe_read_json(rfk_paths.get("gee_gcp_report_json", ""))
    if auto_gcp_json:
        mm = auto_gcp_json.get("metrics", auto_gcp_json)
        chunks.append("本次空间加权 GCP 不确定性摘要：")
        chunks.append(
            "- 方法={method}, 空间加权={spatial_applied}, PICP={PICP}, MPIW={MPIW}, NMPIW={NMPIW}, QCP={CCB}, IntervalScore={IntervalScore}, 宽度图={width_tif}".format(
                method=mm.get("method", "NA"),
                spatial_applied=(mm.get("spatial_uncertainty") or {}).get("spatial_applied", mm.get("spatial_applied", "NA")),
                PICP=mm.get("PICP", "NA"),
                MPIW=mm.get("MPIW", mm.get("mean_width", "NA")),
                NMPIW=mm.get("NMPIW", "NA"),
                CCB=mm.get("CCB", "NA"),
                IntervalScore=mm.get("IntervalScore", "NA"),
                width_tif=rfk_paths.get("gee_gcp_width_tif", mm.get("width_tif", "NA")),
            )
        )

    rfk_txt = _safe_read_text(rfk_paths.get("report_txt", ""), max_chars=2500)
    if rfk_txt:
        chunks.append(f"本次制图报告原文摘录（模型：{model_label(rfk_paths)}）：")
        chunks.append(rfk_txt)

    gcp_json = _safe_read_json(gcp_paths.get("report_json", ""))
    if gcp_json:
        mm = gcp_json.get("sample_metrics") or gcp_json.get("metrics") or gcp_json
        grid = gcp_json.get("grid_statistics") or {}
        chunks.append("本次 GCP + AOA 结果摘要：")
        chunks.append(
            "- PICP={PICP}, MPIW={MPIW}, NMPIW={NMPIW}, QCP={QCP}, IntervalScore={IntervalScore}".format(
                PICP=mm.get("PICP", "NA"),
                MPIW=mm.get("MPIW", "NA"),
                NMPIW=mm.get("NMPIW", "NA"),
                QCP=mm.get("QCP", mm.get("CCB", "NA")),
                IntervalScore=mm.get("IntervalScore", "NA"),
            )
        )
        if grid:
            chunks.append(f"- AOA 内像元比例={grid.get('AOA_inside_ratio', 'NA')}；风险分区统计={grid.get('risk_class_counts', 'NA')}")

    gcp_txt = _safe_read_text(gcp_paths.get("report_txt", ""), max_chars=2500)
    if gcp_txt:
        chunks.append("本次 GCP + AOA 报告原文摘录：")
        chunks.append(gcp_txt)

    if rfk_paths.get("pred_tif"):
        chunks.append(f"本次制图预测图路径（模型：{model_label(rfk_paths)}）：{rfk_paths['pred_tif']}")
    if gcp_paths.get("width_tif"):
        chunks.append(f"本次 GCP + AOA 不确定性图路径：{gcp_paths['width_tif']}")

    return "\n".join(chunks).strip()
