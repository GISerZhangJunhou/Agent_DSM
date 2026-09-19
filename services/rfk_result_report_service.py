from __future__ import annotations

import json
import math
import os
import hashlib
from pathlib import Path
from typing import Any

from services.model_label_service import model_label, extract_model_name_from_report, UNKNOWN_MODEL_LABEL


def _sha256_file(path: str | Path | None) -> str:
    try:
        if not path:
            return ""
        p = Path(path)
        if not p.exists():
            return ""
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _read_report_json(path: str | Path | None) -> dict[str, Any]:
    try:
        if not path:
            return {}
        p = Path(path)
        if not p.exists():
            return {}
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            obj.setdefault("report_json", str(p))
            obj.setdefault("report_sha256", _sha256_file(p))
            return obj
    except Exception:
        return {}
    return {}

def _num(v: Any, nd: int = 4) -> str:
    try:
        if v is None:
            return "NA"
        f = float(v)
        if not math.isfinite(f):
            return "NA"
        return f"{f:.{nd}f}"
    except Exception:
        return "NA"


def _metric_first(d: dict[str, Any] | None, keys: list[str]) -> Any:
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d and d.get(k) is not None:
            return d.get(k)
    return None


def _compact_metrics(m: dict[str, Any] | None) -> dict[str, Any]:
    m = m or {}
    return {
        "cv": m.get("cv"),
        "r2": _metric_first(m, ["pooled_r2", "r2_mean", "r2", "R2", "final_r2"]),
        "rmse": _metric_first(m, ["pooled_rmse", "rmse_mean", "rmse", "RMSE"]),
        "mae": _metric_first(m, ["pooled_mae", "mae_mean", "mae", "MAE"]),
        "bias": _metric_first(m, ["pooled_bias", "bias", "Bias"]),
        "split_count": len(m.get("splits") or []) if isinstance(m.get("splits"), list) else m.get("split_count"),
    }


def _format_metrics_block(title: str, m: dict[str, Any] | None) -> str:
    c = _compact_metrics(m)
    return "\n".join([
        f"{title}",
        f"验证方式：{c.get('cv') or 'NA'}",
        f"R²：{_num(c.get('r2'))}",
        f"RMSE：{_num(c.get('rmse'))}",
        f"MAE：{_num(c.get('mae'))}",
        f"Bias：{_num(c.get('bias'))}",
    ])


def _feature_importance_rows(report: dict[str, Any], top_n: int = 12) -> list[dict[str, Any]]:
    rows = report.get("feature_importance") or []
    if not isinstance(rows, list):
        return []
    clean = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            imp = float(r.get("importance") or 0)
        except Exception:
            imp = 0.0
        clean.append({"feature": str(r.get("feature") or ""), "importance": imp, "label": str(r.get("label") or r.get("feature") or "")})
    clean.sort(key=lambda x: x["importance"], reverse=True)
    return clean[:top_n]


def _main_params(report: dict[str, Any]) -> dict[str, Any]:
    rf = report.get("rf_params") or {}
    rfk = report.get("rfk") or {}
    return {
        "树数量": rf.get("n_estimators"),
        "最大树深": rf.get("max_depth"),
        "叶节点最小样本数": rf.get("min_samples_leaf"),
        "变量抽样比例": rf.get("max_features"),
        "变异函数": rfk.get("variogram_model"),
        "滞后分组数": rfk.get("nlags"),
        "邻近点数": rfk.get("n_closest_points"),
        "半变异函数加权": rfk.get("weight"),
        "克里金是否启用": rfk.get("kriging_enabled"),
    }


def _prediction_stats(pred_tif: str | Path | None) -> dict[str, Any]:
    if not pred_tif:
        return {}
    try:
        import numpy as np
        import rasterio
        p = Path(pred_tif)
        if not p.exists():
            return {}
        with rasterio.open(p) as ds:
            arr = ds.read(1, masked=True).astype("float64")
            vals = arr.compressed()
            if vals.size == 0:
                return {"path": str(p), "valid_count": 0}
            return {
                "path": str(p),
                "crs": str(ds.crs),
                "width": int(ds.width),
                "height": int(ds.height),
                "valid_count": int(vals.size),
                "min": float(np.nanmin(vals)),
                "max": float(np.nanmax(vals)),
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals)),
            }
    except Exception as exc:
        return {"path": str(pred_tif), "error": str(exc)}


def _target_aoi_metrics(m: dict[str, Any]) -> dict[str, Any]:
    return {
        "test_n": m.get("target_aoi_test_n"),
        "test_ratio": m.get("target_aoi_test_ratio"),
        "r2": m.get("target_aoi_r2"),
        "rmse": m.get("target_aoi_rmse"),
        "mae": m.get("target_aoi_mae"),
        "bias": m.get("target_aoi_bias"),
        "warning": m.get("target_aoi_warning"),
    }


def _model_ready_aoi_summary(report: dict[str, Any]) -> dict[str, Any]:
    mrt = report.get("model_ready_table") or {}
    if isinstance(mrt, dict):
        aoi = mrt.get("aoi") or {}
    else:
        aoi = {}
    sample_overlap = report.get("sample_aoi_overlap") or (report.get("target") or {}).get("sample_aoi_overlap") or {}
    return {
        "inside_count": sample_overlap.get("inside_count") or aoi.get("inside_count"),
        "total_count": sample_overlap.get("total_count") or aoi.get("sample_count"),
        "inside_ratio": sample_overlap.get("inside_ratio"),
        "message": sample_overlap.get("message"),
    }


def build_analysis_payload(report: dict[str, Any], pred_tif: str | Path | None = None) -> dict[str, Any]:
    metrics = report.get("metrics") or {}
    payload = {
        "task": {
            "model": extract_model_name_from_report(report) or UNKNOWN_MODEL_LABEL,
            "run_mode": report.get("run_mode"),
            "rows": report.get("rows"),
            "feature_count": report.get("feature_count"),
            "covariate_source": report.get("covariate_source"),
            "model_ready_csv": report.get("model_ready_csv"),
            "prediction_grid_csv": report.get("prediction_grid_csv"),
            "report_json": report.get("report_json") or report.get("model_report_json"),
            "report_sha256": report.get("report_sha256") or _sha256_file(report.get("report_json") or report.get("model_report_json")),
        },
        "strict_validation": _compact_metrics(metrics),
        "strict_target_aoi_validation": _target_aoi_metrics(metrics),
        "relaxed_validation": _compact_metrics(metrics.get("random_validation") or metrics.get("aux_validation") or metrics.get("single_random_validation") or metrics.get("relaxed_validation") or {}),
        "relaxed_target_aoi_validation": _target_aoi_metrics(metrics.get("random_validation") or metrics.get("aux_validation") or metrics.get("single_random_validation") or metrics.get("relaxed_validation") or {}),
        "single_spatial_validation": _compact_metrics(metrics.get("single_spatial_validation") or {}),
        "single_random_validation": _compact_metrics(metrics.get("single_random_validation") or {}),
        "repeated_spatial_validation": _compact_metrics(metrics.get("repeated_spatial_validation") or {}),
        "target_aoi_sample_summary": _model_ready_aoi_summary(report),
        "run_fingerprint": report.get("run_fingerprint"),
        "created_at": report.get("created_at"),
        "main_params": _main_params(report),
        "all_rf_params": report.get("rf_params") or {},
        "rfk": report.get("rfk") or {},
        "actual_model_name": extract_model_name_from_report(report) or UNKNOWN_MODEL_LABEL,
        "hyperparameter_plan": report.get("hyperparameter_plan") or {},
        "top_feature_importance": _feature_importance_rows(report, top_n=12),
        "feature_missing_rate_top": sorted((report.get("feature_missing_rate") or {}).items(), key=lambda kv: float(kv[1] or 0), reverse=True)[:12],
        "feature_zero_value_audit": report.get("feature_zero_value_audit") or {},
        "raster_value_audit": report.get("raster_value_audit") or {},
        "missing_policy": report.get("missing_policy") or {},
        "model_ready_table": report.get("model_ready_table") or {},
        "training_matrix_audit": report.get("training_matrix_audit") or {},
        "prediction_grid_audit": report.get("prediction_grid_audit") or {},
        "prediction_stats": _prediction_stats(pred_tif),
        "warnings": report.get("warnings") or [],
        "target": report.get("target") or {},
    }
    return payload



def generate_data_driven_rfk_analysis(report: dict[str, Any], pred_tif: str | Path | None = None) -> str:
    """Create a report-specific analysis from actual metrics when the remote LLM returns empty.

    This is not a canned fallback: every sentence is assembled from the current report's
    metrics, feature importance, prediction statistics and warnings.  It never fabricates
    missing fields and never reuses previous analysis text.
    """
    payload = build_analysis_payload(report, pred_tif)
    strict = payload.get("strict_validation") or {}
    relaxed = payload.get("relaxed_validation") or {}
    pred_stats = payload.get("prediction_stats") or {}
    fi = payload.get("top_feature_importance") or []
    pga = payload.get("prediction_grid_audit") or {}
    warnings = payload.get("warnings") or []
    target = payload.get("target") or {}
    model_name = payload.get("actual_model_name") or (payload.get("task") or {}).get("model") or UNKNOWN_MODEL_LABEL
    scope = target.get("map_scope") or target.get("output_scope") or report.get("map_scope") or "未读取到制图范围"
    lines = []
    lines.append("1. 本轮结果判断")
    r2 = strict.get("r2") or strict.get("R2")
    rmse = strict.get("rmse") or strict.get("RMSE")
    mae = strict.get("mae") or strict.get("MAE")
    metric_bits = []
    if r2 is not None: metric_bits.append(f"R²={_num(r2, 4)}")
    if rmse is not None: metric_bits.append(f"RMSE={_num(rmse, 4)}")
    if mae is not None: metric_bits.append(f"MAE={_num(mae, 4)}")
    if metric_bits:
        lines.append(f"本轮模型为{model_name}，制图范围为{scope}；主验证读取到" + "，".join(metric_bits) + "。这些数值来自当前 formal_model_report.json 的 metrics 字段。")
    else:
        lines.append(f"本轮模型为{model_name}，但主验证指标字段不完整，因此不评价精度等级。")
    if pred_stats and pred_stats.get("valid_count") is not None:
        lines.append(f"预测图有效像元为{pred_stats.get('valid_count')}，预测范围为{_num(pred_stats.get('min'), 3)}–{_num(pred_stats.get('max'), 3)}，均值为{_num(pred_stats.get('mean'), 3)}，标准差为{_num(pred_stats.get('std'), 3)}。")
    lines.append("\n2. 验证方式差异")
    rr2 = relaxed.get("r2") or relaxed.get("R2")
    rrmse = relaxed.get("rmse") or relaxed.get("RMSE")
    if rr2 is not None or rrmse is not None:
        lines.append(f"对照验证读取到R²={_num(rr2,4)}、RMSE={_num(rrmse,4)}。若主验证与对照验证差异较大，应优先相信空间约束更强的一组；若差异较小，说明当前划分方式对总体精度影响有限。")
    else:
        lines.append("本轮报告没有读取到完整对照验证指标，因此不比较随机/空间验证差异。")
    lines.append("\n3. 变量贡献与协变量建议")
    if fi:
        top_names = [str(x.get("label") or x.get("feature")) + "=" + _num(x.get("importance"), 5) for x in fi[:8]]
        lines.append("变量重要性靠前的是：" + "、".join(top_names) + "。下一轮优先保留这些贡献较高且空间覆盖稳定的变量。")
    else:
        lines.append("本轮报告未读取到变量重要性，暂不提出基于重要性的删减建议。")
    if isinstance(pga, dict):
        high_missing = pga.get("high_grid_missing_features") or []
        if high_missing:
            lines.append("预测网格存在高缺失协变量：" + "、".join([str(x.get('feature')) for x in high_missing[:8] if isinstance(x, dict)]) + "；这些变量需要优先检查栅格覆盖、NoData和对齐。")
        if pga.get("strict_final_valid_cell_count") is not None:
            lines.append(f"严格最终有效预测像元为{pga.get('strict_final_valid_cell_count')}，应结合输出范围判断是否存在大面积无预测区域。")
    lines.append("\n4. 下一轮优化方向")
    if warnings:
        lines.append("本轮报告记录的主要警告包括：" + "；".join([str(x) for x in warnings[:5]]) + "。下一轮应先处理这些数据/范围问题，再调整模型参数。")
    else:
        lines.append("当前报告未记录显著警告。下一轮优化可重点比较不同协变量组合、空间验证方案和目标范围设置，而不是只调单个模型参数。")
    return "\n".join(lines).strip()

def generate_ai_rfk_analysis(report: dict[str, Any], pred_tif: str | Path | None = None) -> str:
    """Ask the configured LLM to analyze this actual model result.

    The prompt deliberately supplies metrics, parameters, feature importance, prediction stats and warnings.
    It asks for result-specific analysis rather than a fixed canned explanation.
    """
    payload = build_analysis_payload(report, pred_tif)
    prompt = (
        "请你扮演资深数字土壤制图专家。下面是一次土壤有机质制图的真实结果数据，模型名称必须以真实结果JSON里的 actual_model_name/task.model 为准，"
        "包括主验证（随机7:3或空间7:3，取决于PRO_V164_PRIMARY_VALIDATION）、另一个7:3对照验证、目标AOI子集验证、模型参数、变量重要性、预测图统计、训练矩阵审计和警告。\n"
        "请必须结合这些具体数值进行分析，不要套用固定模板，不要编造不存在的指标。\n"
        "如果更换区域后全局验证几乎不变，必须解释：全局验证是在兼容训练样点上做的，只有目标AOI子集验证/预测范围会随区县明显变化。\n"
        "必须引用本轮 run_fingerprint；如果目标AOI样点数和总样点数有差异，不能写成100%。\n"
        "输出要求：\n"
        "1）先给出本轮结果是否可接受的判断，并说明依据；\n"
        "2）比较随机7:3与空间7:3差异；\n"
        "3）评价主要参数是否偏复杂、偏保守或较均衡，并说明这些参数是否来自本轮模型搜索/调参过程；\n"
        "4）结合变量重要性和缺失率，提出保留、建议减少、建议补充的环境协变量；\n"
        "5）提出下一轮优化方向。\n"
        "重要约束：如果目标AOI内样点数少于80，不要建议只用目标AOI内样点单独训练模型；应建议兼容区域训练+AOI加权、或补充样点后再做区县独立模型。\n"
        "重要约束：必须区分训练范围、验证范围和输出范围；不能把输出到区县误判为必须仅用区县样点训练。\n"
        "重要约束：先诊断数据质量和协变量覆盖，再诊断参数；不要反复给出固定阈值模板。\n"
        "语言简洁，分段编号，每段都要基于输入数据。\n\n"
        f"真实结果JSON：\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    try:
        from services.llm_service import answer_knowledge_question
        out = answer_knowledge_question(prompt, extra_context="土壤有机质制图结果分析", chat_history=[])
        if out and str(out).strip():
            return str(out).strip()
        try:
            report["ai_result_analysis_error"] = "LLM返回空内容；报告已读取，但未得到可展示的AI分析文本。"
        except Exception:
            pass
    except Exception as exc:
        try:
            report["ai_result_analysis_error"] = str(exc)
        except Exception:
            pass
    # If the remote LLM returned empty, still analyze the current report values locally.
    # This uses actual metrics/importance/prediction statistics and never fabricates missing fields.
    try:
        return generate_data_driven_rfk_analysis(report, pred_tif)
    except Exception as exc:
        try:
            report["ai_result_analysis_error"] = "本地报告驱动分析失败：" + str(exc)
        except Exception:
            pass
        return ""



def write_model_record_txt(out_dir: str | Path, report: dict[str, Any], pred_tif: str | Path | None = None, ai_analysis: str | None = None, round_name: str = "第1轮") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"模型记录_{round_name}.txt"
    metrics = report.get("metrics") or {}
    strict_block = _format_metrics_block("主验证（V164 7:3，优先参考）", metrics)
    relaxed_block = _format_metrics_block("对照验证（另一个7:3划分）", metrics.get("random_validation") or metrics.get("aux_validation") or metrics.get("single_random_validation") or metrics.get("relaxed_validation") or {})
    main_params = _main_params(report)
    fi = _feature_importance_rows(report, 20)
    pred_stats = _prediction_stats(pred_tif)
    plan = report.get("hyperparameter_plan") or {}
    all_params = {
        "rf_params": report.get("rf_params") or {},
        "rfk": report.get("rfk") or {},
        "hyperparameter_plan": plan,
    }
    lines = []
    actual_model_name = extract_model_name_from_report(report) or UNKNOWN_MODEL_LABEL
    lines.append("土壤有机质制图模型记录")
    lines.append(f"本轮模型名称：{actual_model_name}")
    lines.append("")
    lines.append("一、核心结果")
    # 长运行指纹只写后台，不在用户界面显示。
    aoi_sum = _model_ready_aoi_summary(report)
    if aoi_sum.get("inside_count") is not None:
        lines.append(f"目标AOI样点：{aoi_sum.get('inside_count')} / {aoi_sum.get('total_count')}（用于判断区县结果代表性）")
    lines.append(strict_block)
    lines.append("")
    lines.append(relaxed_block)
    if metrics.get("target_aoi_test_n") is not None:
        lines.append("")
        lines.append("目标AOI子集验证（从严格空间验证测试集中提取）")
        lines.append(f"测试样点数：{metrics.get('target_aoi_test_n')}")
        lines.append(f"R²：{_num(metrics.get('target_aoi_r2'), 4)}")
        lines.append(f"RMSE：{_num(metrics.get('target_aoi_rmse'), 4)}")
        lines.append(f"MAE：{_num(metrics.get('target_aoi_mae'), 4)}")
        if metrics.get('target_aoi_warning'):
            lines.append(f"提示：{metrics.get('target_aoi_warning')}")
    lines.append("")
    lines.append("二、主要超参数")
    for k, v in main_params.items():
        lines.append(f"{k}：{v}")
    lines.append("")
    lines.append("三、数据质量与分辨率审计")
    mrt = report.get("model_ready_table") or {}
    res = (mrt.get("resolution") or (report.get("target") or {}).get("resolution_audit") or {}) if isinstance(mrt, dict) else {}
    lines.append(f"目标分辨率：{(res or {}).get('chosen_resolution_m') or (report.get('target') or {}).get('resolution_m')} m")
    lines.append(f"分辨率来源：{(report.get('target') or {}).get('resolution_source') or (res or {}).get('rule') or ''}")
    if isinstance(mrt, dict):
        lines.append(f"建模样点数：{mrt.get('sample_count_after_gate', report.get('rows'))} / 原始 {mrt.get('sample_count_before_gate', '')}")
        lines.append(f"实际入模协变量数：{mrt.get('model_feature_count', report.get('feature_count'))} / 原始 {mrt.get('raw_feature_count', '')}")
        v165_plan = report.get("v165_feature_deployment_plan") or (report.get("target") or {}).get("v165_feature_deployment_plan") or {}
        if isinstance(v165_plan, dict) and v165_plan:
            lines.append(f"V165 CSV诊断协变量数：{v165_plan.get('csv_feature_count')}；正式出图协变量数：{v165_plan.get('mappable_feature_count')}；缺少预测栅格协变量数：{v165_plan.get('missing_raster_feature_count')}")
            if v165_plan.get('mappable_features'):
                lines.append("正式出图协变量：" + "、".join([str(x) for x in v165_plan.get('mappable_features', [])[:30]]))
            if v165_plan.get('missing_raster_features'):
                lines.append("CSV中有但未参与正式出图的协变量（缺少预测栅格）：" + "、".join([str(x) for x in v165_plan.get('missing_raster_features', [])[:30]]))
            lines.append("V165特征规则：训练诊断以融合CSV为准；正式GeoTIFF只使用CSV列与可用预测栅格的交集。")
        tma = report.get("training_matrix_audit") or {}
        if isinstance(tma, dict):
            if tma.get("primary_validation_split"):
                lines.append(f"主验证划分：{tma.get('primary_validation_split')}，验证比例={tma.get('primary_validation_test_size')}（即7:3）")
            if tma.get("audit_path"):
                lines.append(f"训练矩阵审计：{tma.get('audit_path')}")
            raw_a = tma.get("raw_from_model_ready_csv") or {}
            if isinstance(raw_a, dict):
                if raw_a.get("X_hash_pre_impute"):
                    lines.append(f"建模矩阵Hash：X={raw_a.get('X_hash_pre_impute')}；y={(raw_a.get('target') or {}).get('hash')}")
                if raw_a.get("near_constant_features"):
                    lines.append("近常数协变量警告：" + "、".join([str(x) for x in raw_a.get("near_constant_features", [])[:20]]))
                if raw_a.get("high_missing_features"):
                    lines.append("高缺失协变量警告：" + "、".join([str((x or {}).get('feature')) for x in raw_a.get("high_missing_features", [])[:20] if isinstance(x, dict)]))
        dropped = mrt.get('dropped_features') or []
        if dropped:
            lines.append("被质量闸门剔除的协变量：" + "、".join([str(x.get('feature')) + ("(" + str(x.get('reason')) + ")" if x.get('reason') else "") for x in dropped[:20] if isinstance(x, dict)]))
        qflags = mrt.get('quality_flags') or []
        if qflags:
            lines.append("首轮保留但需关注的协变量：" + "、".join([str(x.get('feature')) for x in qflags[:20] if isinstance(x, dict)]))
        if mrt.get('disabled_lulccd_features'):
            lines.append("土地覆盖策略：已禁用 LULCcd，使用 CLCD。")
        if mrt.get('clcd_encoding'):
            _encs = mrt.get('clcd_encoding') or []
            _mask_only = any(str((x or {}).get('encoding')) == 'mask_only' for x in _encs if isinstance(x, dict))
            if _mask_only:
                lines.append("CLCD处理：1=农田，2=森林，3=灌木，4=草地，5=水体，6=冰雪，7=裸地，8=不透水面；最近邻重采样；CLCD仅作用户指定地类范围掩膜，不作为SOM模型特征。")
            else:
                lines.append("CLCD处理：1=农田，2=森林，3=灌木，4=草地，5=水体，6=冰雪，7=裸地，8=不透水面；最近邻重采样，one-hot入模。")
        mp = report.get("missing_policy") or mrt.get("missing_policy") or {}
        if mp:
            lines.append("缺失判定策略：0值默认有效；仅 raster mask / 显式NoData / NaN / 非有限值计为缺失。zero_nodata_policy=" + str(mp.get("zero_nodata_policy")))
        zra = report.get("raster_value_audit") or mrt.get("raster_value_audit") or {}
        if isinstance(zra, dict) and zra:
            suspicious = []
            for _c, _a in zra.items():
                if isinstance(_a, dict) and int(_a.get("zero_missing_count") or 0) > 0:
                    suspicious.append(f"{_c}(0缺失={_a.get('zero_missing_count')},0有效={_a.get('zero_valid_count')})")
            if suspicious:
                lines.append("0值缺失审计：" + "；".join(suspicious[:12]))
            else:
                lines.append("0值缺失审计：未发现0值被通用规则计为缺失。")
    pga = report.get("prediction_grid_audit") or {}
    if isinstance(pga, dict) and pga.get("high_grid_missing_features"):
        lines.append("预测网格高缺失协变量：" + "、".join([str(x.get('feature')) for x in pga.get('high_grid_missing_features', [])[:20] if isinstance(x, dict)]))
    if isinstance(pga, dict) and pga.get("clcd_cropland_mask"):
        cmi = pga.get("clcd_cropland_mask") or {}
        if cmi.get("enabled"):
            _lbl = cmi.get('target_class_label') or '目标地类'
            _code = cmi.get('target_class_code') or '目标'
            lines.append(f"CLCD目标地类掩膜：已启用；仅对CLCD={_code}（{_lbl}）像元进行连续SOM预测，其他地类写为NODATA。")
            lines.append(f"CLCD目标地类像元数：{cmi.get('target_landcover_cell_count', cmi.get('cropland_cell_count'))}；非目标地类/无效像元数：{cmi.get('non_target_or_invalid_cell_count', cmi.get('non_cropland_or_invalid_cell_count'))}")
        elif cmi.get("error"):
            lines.append("CLCD目标地类掩膜：未启用/失败，原因：" + str(cmi.get("error")))
    lines.append("")
    if ai_analysis:
        lines.append("四、AI结合本轮结果的分析与建议")
        lines.append(ai_analysis)
    else:
        lines.append("四、结果分析状态")
        lines.append("未生成可写入报告的AI分析文本；后台已保留真实指标、参数和错误日志。")
    lines.append("")
    lines.append("五、变量重要性（前20项）")
    if fi:
        for i, r in enumerate(fi, 1):
            lines.append(f"{i}. {r.get('label') or r.get('feature')}：{_num(r.get('importance'), 6)}")
    else:
        lines.append("未生成变量重要性。")
    lines.append("")
    lines.append("六、建模与预测表")
    lines.append(f"建模样点表：{report.get('model_ready_csv') or ''}")
    lines.append(f"预测网格表：{report.get('prediction_grid_csv') or ''}")
    lines.append(f"说明：本轮模型名称以报告字段为准：{actual_model_name}；预测图只对有效目标像元进行连续预测，非目标/NoData 区域不参与预测。")
    lines.append("")
    lines.append("七、预测图统计")
    if pred_stats:
        for k, v in pred_stats.items():
            lines.append(f"{k}：{v}")
    else:
        lines.append("未读取到预测图统计。")
    lines.append("")
    lines.append("八、完整参数与寻参记录")
    lines.append(json.dumps(all_params, ensure_ascii=False, indent=2, default=str))
    lines.append("")
    lines.append("九、结果文件")
    lines.append(f"预测图：{pred_tif or ''}")
    lines.append(f"模型报告JSON：{out_dir / 'formal_model_report.json'}")
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def build_chat_text_from_report(report: dict[str, Any], model_record_txt: str | Path | None = None) -> str:
    # If the caller only passed result paths, load the current run report first.
    if isinstance(report, dict) and not report.get("metrics"):
        loaded = _read_report_json(report.get("report_json") or report.get("model_report_json"))
        if loaded:
            loaded.update({k: v for k, v in report.items() if k not in loaded})
            report = loaded
    metrics = report.get("metrics") or {}
    pred_tif = report.get("pred_tif") or report.get("internal_pred_tif") or report.get("prediction_tif")
    report_json_path = report.get("report_json") or report.get("model_report_json")
    if not metrics:
        missing = []
        if not report_json_path:
            missing.append("report_json/model_report_json")
        missing.append("metrics")
        return (
            "✅ 土壤有机质制图完成\n\n"
            "模型名称：" + (extract_model_name_from_report(report) or UNKNOWN_MODEL_LABEL) + "\n"
            "⚠️ 未生成AI结果分析：未读取到本轮精度报告的必要字段。\n"
            "缺失项：" + "、".join(missing) + "\n"
            "系统不会使用模板、固定精度或旧分析文本代替真实分析。"
            + ("\n预测图：" + str(pred_tif) if pred_tif else "")
        )
    payload = build_analysis_payload(report, pred_tif)
    # Do not call the LLM from Dash polling/render callbacks.  If the modeling
    # task already wrote an AI analysis into the current report, display it; if
    # not, state that no analysis was generated.  This prevents post-completion
    # UI callbacks from blocking the chat input while still enforcing the rule
    # that no template or stale analysis may be substituted.
    ai = str(report.get("ai_result_analysis") or "").strip()
    main_params = _main_params(report)
    params_line = "；".join([f"{k}：{v}" for k, v in main_params.items() if v is not None])
    pred_stats = payload.get("prediction_stats") or {}
    top_fi = payload.get("top_feature_importance") or []
    pga = payload.get("prediction_grid_audit") or {}
    lines = []
    actual_model_name = extract_model_name_from_report(report) or UNKNOWN_MODEL_LABEL
    lines.append("✅ 土壤有机质制图完成")
    lines.append(f"模型名称：{actual_model_name}")
    lines.append("")
    lines.append("📊 1. 本轮真实精度与运行标识")
    report_json = report.get("report_json") or report.get("model_report_json")
    if report_json:
        lines.append(f"精度报告来源：{report_json}")
    # 完整 SHA256 只写后台/报告元数据，不在用户界面显示。
    # 长运行指纹只写后台，不在用户界面显示。
    if report.get("created_at"):
        lines.append(f"报告生成时间：{report.get('created_at')}")
    aoi_sum = _model_ready_aoi_summary(report)
    if aoi_sum.get("inside_count") is not None:
        lines.append(f"目标AOI样点：{aoi_sum.get('inside_count')} / {aoi_sum.get('total_count')}")
    if metrics:
        metric_source = str(metrics.get("metric_source") or report.get("metric_source") or "").strip()
        if metric_source:
            if metric_source == "realtime_deployed_mapping_model_mappable_features":
                lines.append("精度来源：当前正式出图模型实时重算（可映射协变量特征，重复空间验证）。")
            elif metric_source == "deployed_mapping_model_mappable_features":
                lines.append("精度来源：正式出图模型（可映射协变量特征）。")
            elif metric_source in {"diagnostic_csv_model_no_deployed_mapping_model", "unavailable_no_deployed_mapping_model"}:
                lines.append("精度来源：未生成正式出图模型；诊断模型指标不作为最终制图精度展示。")
            else:
                lines.append("精度来源：" + metric_source)
        if metrics.get("metric_warning") or report.get("metric_warning"):
            lines.append("精度警告：" + str(metrics.get("metric_warning") or report.get("metric_warning")))
        lines.append(_format_metrics_block("主验证（按本轮配置执行）", metrics))
        aux = metrics.get("realtime_random_validation") or metrics.get("random_validation") or metrics.get("aux_validation") or metrics.get("single_random_validation") or metrics.get("relaxed_validation") or {}
        if aux:
            lines.append("")
            lines.append(_format_metrics_block("对照验证（本轮辅助参考）", aux))
        if metrics.get("target_aoi_test_n") is not None:
            lines.append("")
            lines.append("目标AOI子集验证（严格验证测试集中提取）")
            lines.append(f"测试样点数：{metrics.get('target_aoi_test_n')}；R²：{_num(metrics.get('target_aoi_r2'), 4)}；RMSE：{_num(metrics.get('target_aoi_rmse'), 4)}；MAE：{_num(metrics.get('target_aoi_mae'), 4)}")
    else:
        lines.append("未读取到本轮 formal_model_report.json 中的 metrics 字段，因此不输出 R²/RMSE/MAE 固定值。")
    lines.append("")
    lines.append("🗺️ 2. 预测图统计与图层状态")
    if pred_stats:
        lines.append(f"有效像元：{pred_stats.get('valid_count', 'NA')}；范围：{_num(pred_stats.get('min'))}–{_num(pred_stats.get('max'))}；均值：{_num(pred_stats.get('mean'))}；标准差：{_num(pred_stats.get('std'))}；尺寸：{pred_stats.get('width', 'NA')}×{pred_stats.get('height', 'NA')}；CRS：{pred_stats.get('crs', 'NA')}。")
    else:
        lines.append("未从预测 GeoTIFF 读取到像元统计；不生成固定空间分布判断。")
    if isinstance(pga, dict) and pga.get("valid_cells") is not None:
        lines.append(f"预测网格有效像元：{pga.get('valid_cells')}；严格最终有效像元：{pga.get('strict_final_valid_cells', 'NA')}。")
    lines.append("")
    lines.append("⚙️ 3. 主要超参数与协变量")
    lines.append(params_line or "本轮报告未读取到关键超参数。")
    if top_fi:
        lines.append("本轮变量重要性靠前：" + "、".join([str(x.get("label") or x.get("feature")) + "=" + _num(x.get("importance"), 5) for x in top_fi[:8]]))
    else:
        lines.append("本轮报告未读取到变量重要性。")
    lines.append("")
    if ai:
        lines.append("🤖 4. AI读取本轮结果后的分析")
        lines.append(ai)
    else:
        # V216：AI 分析未生成时不在用户界面显示诊断段落；只在后台日志/审计中保留。
        try:
            from utils.pro_console import pro_console_log
            pro_console_log("AI_ANALYSIS", "本轮报告已读取但AI分析未生成，前端不展示失败说明", {"report_json": str(report.get("report_json") or ""), "error": str(report.get("ai_result_analysis_error") or "")})
        except Exception:
            pass
    if model_record_txt:
        lines.append("")
        lines.append("📁 5. 完整记录")
        lines.append(f"完整精度、全部参数、变量审计和每轮记录已保存：{model_record_txt}")
    return "\n".join(lines)
