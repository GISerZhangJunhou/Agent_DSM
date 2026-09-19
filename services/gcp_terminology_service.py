# -*- coding: utf-8 -*-
"""Central terminology guard for GCP/AOA uncertainty analysis.

In this project GCP means Geographic Conformal Prediction (地理共形预测).
It must never be expanded as generalized kriging precision or any kriging-related term.
"""
from __future__ import annotations

from typing import Any

GCP_FULL_CN = "地理共形预测"
GCP_FULL_EN = "Geographic Conformal Prediction"
GCP_DISPLAY_NAME = f"{GCP_FULL_CN}（{GCP_FULL_EN}, GCP）"

_BAD_REPLACEMENTS = {
    "广义克里金精度（GCP）": GCP_DISPLAY_NAME,
    "GCP（广义克里金精度）": GCP_DISPLAY_NAME,
    "广义克里金精度(GCP)": GCP_DISPLAY_NAME,
    "GCP(广义克里金精度)": GCP_DISPLAY_NAME,
    "广义克里金预测（GCP）": GCP_DISPLAY_NAME,
    "GCP（广义克里金预测）": GCP_DISPLAY_NAME,
    "广义克里金预测(GCP)": GCP_DISPLAY_NAME,
    "GCP(广义克里金预测)": GCP_DISPLAY_NAME,
    # Bare terms are not replaced because they may appear in the explicit
    # negative warning "不是广义克里金精度". Only acronym-expanded wrong forms
    # are normalized.
    "地理共性预测": GCP_FULL_CN,
    "地理共性预估": GCP_FULL_CN,
    "地理共形预估": GCP_FULL_CN,
}


def sanitize_gcp_terminology(text: Any) -> str:
    """Normalize GCP terminology in user-facing text.

    This is a safety guard for LLM-generated content. The computational method is
    implemented in ``models/GCP_AOA_formal_uncertainty.py`` as local conformal
    prediction calibrated with spatially weighted residual quantiles plus AOA.
    """
    s = str(text or "")
    for bad, good in _BAD_REPLACEMENTS.items():
        s = s.replace(bad, good)
    return s


def gcp_method_summary() -> str:
    return (
        "GCP 指地理共形预测（Geographic Conformal Prediction），"
        "本系统用校准/验证残差的空间加权分位数构建预测区间；"
        "AOA 用于识别模型适用域和外推风险。GCP 不是广义克里金精度，也不是克里金模型。"
    )
