from services.gcp_terminology_service import sanitize_gcp_terminology
def _fmt(v, nd=3):
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return "NA"


def _pick_metric(d, *keys):
    if not isinstance(d, dict):
        return "NA"
    for k in keys:
        if d.get(k) is not None:
            return d.get(k)
    return "NA"


def build_map_completion_text(metrics=None) -> str:
    """Build completion text through the unified current-run result reader.

    Hard rule: read actual current-run evidence first.  If required files/fields
    are missing, do not synthesize accuracy or analysis.
    """
    try:
        from services.result_reader_service import read_mapping_result
        evidence = read_mapping_result(metrics if isinstance(metrics, dict) else {}, metrics if isinstance(metrics, dict) else {})
    except Exception as exc:
        return (
            "✅ 土壤有机质制图完成\n\n"
            "⚠️ 未生成结果分析：统一结果读取中心运行失败。\n"
            f"错误：{exc}\n"
            "系统不会使用模板、固定精度或旧分析文本代替真实分析。"
        )
    if not evidence.get("ok"):
        return (
            "✅ 土壤有机质制图完成\n\n"
            f"模型名称：{evidence.get('model_name') or '未读取到本轮模型名称'}\n"
            "⚠️ 未生成结果分析：未读取到本轮制图结果的必要报告或字段。\n"
            "缺失项：" + "、".join(evidence.get("missing") or ["未知"]) + "\n"
            + ("报告路径：" + str(evidence.get("report_json")) + "\n" if evidence.get("report_json") else "")
            + ("预测图：" + str(evidence.get("pred_tif")) + "\n" if evidence.get("pred_tif") else "")
            + "系统不会使用模板、固定精度或旧分析文本代替真实分析。"
        )
    report = dict(evidence.get("report") or {})
    report.setdefault("report_json", evidence.get("report_json"))
    report.setdefault("report_sha256", evidence.get("report_sha256"))
    report.setdefault("pred_tif", evidence.get("pred_tif"))
    report.setdefault("display_model_name", evidence.get("model_name"))
    try:
        from services.rfk_result_report_service import build_chat_text_from_report
        txt = build_chat_text_from_report(report, model_record_txt=(metrics or {}).get("model_record_txt") if isinstance(metrics, dict) else None)
        if txt and txt.strip():
            return sanitize_gcp_terminology(txt)
    except Exception as exc:
        return (
            "✅ 土壤有机质制图完成\n\n"
            f"模型名称：{evidence.get('model_name')}\n"
            f"精度报告来源：{evidence.get('report_json')}\n"
            "⚠️ 未生成结果分析：报告已读取，但分析文本构造失败。\n"
            f"错误：{exc}\n"
            "系统不会使用模板或旧分析文本替代。"
        )
    return (
        "✅ 土壤有机质制图完成\n\n"
        f"模型名称：{evidence.get('model_name')}\n"
        f"精度报告来源：{evidence.get('report_json')}\n"
        "报告已读取；AI分析服务未返回有效内容，前端不展示分析段落。"
    )


def build_gcp_completion_text(metrics=None, result_paths=None) -> str:
    """Build GCP/AOA completion text through the unified current-run result reader."""
    try:
        from services.result_reader_service import read_gcp_result
        evidence = read_gcp_result(result_paths if isinstance(result_paths, dict) else {}, metrics if isinstance(metrics, dict) else {})
    except Exception as exc:
        return (
            "✅ 地理共形预测（GCP）+ AOA 不确定性分析完成\n\n"
            "⚠️ 未生成不确定性分析解读：统一结果读取中心运行失败。\n"
            f"错误：{exc}\n"
            "系统不会使用模板解释替代真实分析。"
        )
    if not evidence.get("ok"):
        return (
            "✅ 地理共形预测（GCP）+ AOA 不确定性分析完成\n\n"
            "⚠️ 未生成不确定性分析解读：未读取到本轮 GCP/AOA 结果的必要报告或字段。\n"
            "缺失项：" + "、".join(evidence.get("missing") or ["未知"]) + "\n"
            + ("报告路径：" + str(evidence.get("report_json")) + "\n" if evidence.get("report_json") else "")
            + ("区间宽度图：" + str(evidence.get("width_tif")) + "\n" if evidence.get("width_tif") else "")
            + "系统不会使用模板解释替代真实分析。"
        )
    report = dict(evidence.get("report") or {})
    report.setdefault("report_json", evidence.get("report_json"))
    report.setdefault("report_sha256", evidence.get("report_sha256"))
    try:
        from services.gcp_result_report_service import build_chat_text_from_gcp_report
        txt = build_chat_text_from_gcp_report(report, result_paths=result_paths)
        if txt and txt.strip():
            return sanitize_gcp_terminology(txt)
    except Exception as exc:
        return (
            "✅ 地理共形预测（GCP）+ AOA 不确定性分析完成\n\n"
            f"指标报告来源：{evidence.get('report_json')}\n"
            "⚠️ 未生成不确定性分析解读：报告已读取，但分析文本构造失败。\n"
            f"错误：{exc}\n"
            "系统不会使用模板解释替代。"
        )
    return (
        "✅ 地理共形预测（GCP）+ AOA 不确定性分析完成\n\n"
        f"指标报告来源：{evidence.get('report_json')}\n"
        "报告已读取；AI分析服务未返回有效内容，前端不展示分析段落。"
    )


def build_model_text() -> str:
    return "模型名称以本轮正式结果报告为准：读到什么模型就显示什么模型；未读取到模型字段时显示“未读取到本轮模型名称”，不会默认写成 RFK。"


def build_accuracy_text(metrics=None) -> str:
    if not isinstance(metrics, dict) or not metrics:
        return "当前没有读取到本轮精度结果；不会输出固定精度或模板解释。"
    main = metrics.get("main_result") or metrics.get("formal_result") or metrics.get("metrics") or {}
    if not isinstance(main, dict) or not main:
        return "当前没有从本轮报告中读取到 metrics 字段；不会输出固定精度或模板解释。"
    chunks = ["本轮报告中读取到的精度字段如下："]
    chunks.append(
        "- R²={r2}，RMSE={rmse}，MAE={mae}".format(
            r2=_pick_metric(main, "r2", "R2", "final_r2", "pooled_r2"),
            rmse=_pick_metric(main, "rmse", "RMSE", "pooled_rmse"),
            mae=_pick_metric(main, "mae", "MAE", "pooled_mae"),
        )
    )
    aux = main.get("realtime_random_validation") or main.get("random_validation") or main.get("aux_validation") or main.get("single_random_validation") or {}
    if isinstance(aux, dict) and aux:
        chunks.append(
            "- 对照验证：R²={r2}，RMSE={rmse}，MAE={mae}".format(
                r2=_pick_metric(aux, "r2", "R2", "pooled_r2"),
                rmse=_pick_metric(aux, "rmse", "RMSE", "pooled_rmse"),
                mae=_pick_metric(aux, "mae", "MAE", "pooled_mae"),
            )
        )
    return "\n".join(chunks)


def build_guidance_text() -> str:
    return "你可以直接告诉我需求：数据下载、有机质制图、不确定性分析（须先完成有机质制图）、数据检查、结果解释，或 DSM 知识问答。"


def build_followup_required_text(reason: str) -> str:
    return f"当前还不能直接执行这个请求。{reason}"


def build_user_data_unavailable_text() -> str:
    return "当前还没有识别到可显示的上传栅格。采样点通常上传 CSV/Excel；环境协变量通常上传 TIF/NC/HDF。采样点会持续显示，环境协变量会按上传顺序进入地图预览。"
