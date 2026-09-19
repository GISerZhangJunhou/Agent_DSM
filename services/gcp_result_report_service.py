from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from services.model_label_service import model_label
from services.gcp_terminology_service import sanitize_gcp_terminology, GCP_DISPLAY_NAME


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


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        p = Path(path)
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv_summary(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        import pandas as pd
        p = Path(path)
        if not p.exists():
            return {}
        df = pd.read_csv(p, encoding="utf-8")
        out: dict[str, Any] = {"rows": int(len(df)), "columns": list(df.columns)}
        for col in ["residual", "abs_residual", "interval_width", "gcp_width", "DI", "aoa_di", "inside_aoa", "aoa_inside"]:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(vals):
                    out[col] = {
                        "count": int(len(vals)),
                        "min": float(vals.min()),
                        "max": float(vals.max()),
                        "mean": float(vals.mean()),
                        "median": float(vals.median()),
                    }
        return out
    except Exception as exc:
        return {"error": str(exc)}


def _risk_share(counts: dict[str, Any], key: str, total: int) -> str:
    try:
        n = int(counts.get(str(key)) or counts.get(int(key)) or 0)
        return f"{n}（{(n / total * 100.0):.2f}%）" if total else str(n)
    except Exception:
        return "NA"


def build_gcp_analysis_payload(metrics: dict[str, Any] | None, result_paths: dict[str, Any] | None = None) -> dict[str, Any]:
    report = dict(metrics or {})
    paths = dict(result_paths or {})
    if not report and paths.get("report_json"):
        report = _read_json(paths.get("report_json"))
    outputs = report.get("outputs") or {}
    sample_metrics = report.get("sample_metrics") or report.get("metrics") or report
    grid = report.get("grid_statistics") or {}
    sample_csv = paths.get("sample_audit_csv") or outputs.get("sample_csv") or paths.get("sample_csv")
    payload = {
        "method": report.get("method"),
        "alpha": report.get("alpha"),
        "nominal_coverage": report.get("nominal_coverage"),
        "sample_metrics": sample_metrics,
        "aoa_params": report.get("aoa_params") or {},
        "gcp_params": report.get("gcp_params") or {},
        "grid_statistics": grid,
        "inputs": report.get("inputs") or {},
        "outputs": {**outputs, **paths},
        "sample_audit_summary": _read_csv_summary(sample_csv),
    }
    return payload


def generate_ai_gcp_analysis(metrics: dict[str, Any] | None, result_paths: dict[str, Any] | None = None) -> str:
    payload = build_gcp_analysis_payload(metrics, result_paths)
    prompt = (
        f"请你扮演资深数字土壤制图不确定性分析专家。这里的 GCP 必须解释为{GCP_DISPLAY_NAME}，不是广义克里金精度、不是广义克里金预测、也不是任何克里金方法。下面是一次 GCP + AOA 不确定性分析的真实输出数据，"
        "包括样点覆盖率指标、区间宽度、AOA 像元比例、风险分区像元数、参数和样点审计摘要。"
        "请必须基于这些具体数值进行结果解读，不要套用固定模板，不要编造不存在的指标；"
        "如果某个指标缺失，请说明未读取到。\n"
        "输出要求：1）判断不确定性结果是否可交付；2）解释 PICP、MPIW/NMPIW 与区间宽度；"
        "3）解释 AOA 内外比例和四类风险区；4）指出重点核查/补充采样区域；5）说明这些图件应如何和 SOM 图一起使用。\n\n"
        f"真实 GCP+AOA 结果 JSON：\n{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}"
    )
    try:
        from services.llm_service import answer_knowledge_question
        out = answer_knowledge_question(prompt, extra_context="GCP+AOA不确定性结果分析", chat_history=[])
        if out and str(out).strip():
            return sanitize_gcp_terminology(str(out).strip())
    except Exception:
        pass
    # Hard delivery rule: do not synthesize GCP/AOA interpretation when AI
    # analysis fails.  The caller will state that no AI analysis was generated.
    return ""



def build_chat_text_from_gcp_report(metrics: dict[str, Any] | None, result_paths: dict[str, Any] | None = None) -> str:
    payload = build_gcp_analysis_payload(metrics, result_paths)
    sm = payload.get("sample_metrics") or {}
    grid = payload.get("grid_statistics") or {}
    outputs = payload.get("outputs") or {}
    if not sm or not any(k in sm for k in ["PICP", "MPIW", "NMPIW", "QCP", "IntervalScore"]):
        return (
            "✅ GCP + AOA 不确定性分析完成\n\n"
            "⚠️ 未生成AI不确定性分析：未读取到本轮 GCP_AOA 指标报告中的 sample_metrics。\n"
            "系统不会使用模板、固定指标解释或旧分析文本替代真实分析。"
            + ("\n报告路径：" + str((result_paths or {}).get("report_json") or outputs.get("report_json")) if (result_paths or outputs) else "")
        )
    # Do not call the LLM from Dash polling/render callbacks.  The GCP/AOA
    # backend may write ai_result_analysis into the current report; if it is not
    # present, the UI must say so and must not synthesize a template analysis.
    ai = str((metrics or {}).get("ai_result_analysis") or "").strip()
    base_model = model_label(result_paths or {}, metrics or {})
    lines = [
        "✅ 地理共形预测（GCP）+ AOA 不确定性分析完成",
        f"上一轮制图模型：{base_model}",
        "方法说明：GCP 指地理共形预测（Geographic Conformal Prediction），用于基于校准残差构建预测区间；不是广义克里金精度或克里金模型。",
        "",
        "📊 1. 样点区间指标",
        f"名义覆盖率：{_num(payload.get('nominal_coverage'))}；PICP：{_num(sm.get('PICP'))}；MPIW：{_num(sm.get('MPIW'))}；NMPIW：{_num(sm.get('NMPIW'))}；QCP：{_num(sm.get('QCP'))}；Interval Score：{_num(sm.get('IntervalScore'))}。",
        "",
        "🧭 2. AOA 与风险区",
        f"有效像元：{grid.get('valid_pixel_count', 'NA')}；AOA 内像元比例：{_num(grid.get('AOA_inside_ratio'))}；AOA 内：{grid.get('AOA_inside_pixel_count', 'NA')}；AOA 外：{grid.get('AOA_outside_pixel_count', 'NA')}。",
        "区间宽度单位与目标变量一致；当前土壤有机质目标按 g/kg 解释，因此宽度图单位也按 g/kg 展示。",
    ]
    if ai:
        lines.extend(["", "🤖 3. AI结合本轮结果的分析", ai])
    else:
        try:
            from utils.pro_console import pro_console_log
            pro_console_log("AI_ANALYSIS", "本轮GCP/AOA报告已读取但AI分析未生成，前端不展示失败说明", {"report_json": str((result_paths or {}).get("report_json") or "")})
        except Exception:
            pass
    if outputs.get("report_txt") or outputs.get("sample_csv") or outputs.get("risk_class_tif"):
        lines.extend(["", "📁 4. 结果文件"])
        if outputs.get("report_txt"):
            lines.append("分析报告：" + str(outputs.get("report_txt")))
        if outputs.get("risk_class_tif"):
            lines.append("综合风险分区：" + str(outputs.get("risk_class_tif")))
        if outputs.get("sample_csv"):
            lines.append("样点审计表：" + str(outputs.get("sample_csv")))
    return "\n".join(lines)
