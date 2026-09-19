from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def export_with_qgis(layout_state: Dict[str, Any], output_dir: str | Path) -> Dict[str, str]:
    """可选：若本机已配置 PyQGIS，则在这里按 layout_state 重建专题图布局并导出。

    当前版本只提供接口占位，便于后续把网页端布局状态对接到 QGIS Print Layout。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "status": "not_implemented",
        "message": "当前版本未启用 PyQGIS 自动导出；网页端已支持 ArcGIS Pro 风格的自由拖拽与缩放。",
        "output_dir": str(out),
    }
