from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from config.settings import (
    ENABLE_QGIS_STYLE_BRIDGE,
    QGIS_BRIDGE_MODE,
    QGIS_PREFIX_PATH,
    QGIS_BIN_PATH,
    QGIS_PYTHON_BAT,
    QGIS_EXECUTABLE,
    QGIS_SVG_PATHS,
    QGIS_STYLE_CACHE_DIR,
    BASE_DIR,
)
from services.layout_map_service import COLOR_RAMPS

STYLE_JSON = QGIS_STYLE_CACHE_DIR / "qgis_style_catalog.json"
BRIDGE_SCRIPT = QGIS_STYLE_CACHE_DIR / "export_qgis_style_catalog.py"


def _safe_copy_to_cache(src: Path) -> str:
    src = src.resolve()
    key = hashlib.md5(str(src).encode("utf-8")).hexdigest()[:12]
    dst = QGIS_STYLE_CACHE_DIR / f"{key}_{src.name}"
    if not dst.exists():
        shutil.copyfile(src, dst)
    return f"/__local_file?path={dst.as_posix()}"


def _default_catalog(message: str = "fallback") -> dict[str, Any]:
    return {
        "backend": "fallback",
        "message": message,
        "palettes": [
            {"label": name, "value": name, "colors": colors}
            for name, colors in COLOR_RAMPS.items()
        ],
        "north_arrows": [
            {"key": "arcgis", "label": "ArcGIS", "src": "/assets/north_arrow_arcgis.svg"},
            {"key": "qgis", "label": "QGIS", "src": "/assets/north_arrow_qgis.svg"},
            {"key": "minimal", "label": "简洁", "src": "/assets/north_arrow_minimal.svg"},
            {"key": "classic", "label": "经典", "src": "/assets/north_arrow_classic.svg"},
        ],
    }


def _ensure_bridge_script() -> None:
    if BRIDGE_SCRIPT.exists():
        return
    BRIDGE_SCRIPT.write_text(
        """from __future__ import annotations

import json
import os
from pathlib import Path


def collect_svg_files(svg_root: Path):
    items = []
    if not svg_root.exists():
        return items
    for p in svg_root.rglob('*.svg'):
        low = p.name.lower()
        if any(k in low for k in ['north', 'arrow', 'compass', 'rose']):
            items.append({'label': p.stem.replace('_', ' '), 'path': str(p)})
    return items


def main():
    out_json = Path(os.environ['QGIS_STYLE_OUT_JSON'])
    svg_dirs = [Path(p) for p in os.environ.get('QGIS_SVG_PATHS', '').split(os.pathsep) if p.strip()]
    ramps = [
        {'label': 'Green Continuous', 'value': 'Green Continuous', 'colors': ['#f3f7f1', '#b7ddb1', '#0b8f3a']},
        {'label': 'Yellow Green Blue', 'value': 'Yellow Green Blue', 'colors': ['#ffffcc', '#78c679', '#225ea8']},
        {'label': 'Yellow Orange Red', 'value': 'Yellow Orange Red', 'colors': ['#ffffb2', '#fd8d3c', '#e31a1c']},
        {'label': 'RdYlGn', 'value': 'RdYlGn', 'colors': ['#d73027', '#fee08b', '#1a9850']},
    ]
    arrows = []
    for root in svg_dirs:
        arrows.extend(collect_svg_files(root))
    payload = {'backend': 'qgis-external', 'message': 'external bridge', 'palettes': ramps, 'north_arrows': arrows[:48]}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
""",
        encoding="utf-8",
    )


def _external_catalog() -> dict[str, Any]:
    if not QGIS_PYTHON_BAT or not Path(QGIS_PYTHON_BAT).exists():
        return _default_catalog(f"QGIS external bridge unavailable: {QGIS_PYTHON_BAT}")
    _ensure_bridge_script()
    env = os.environ.copy()
    env["QGIS_STYLE_OUT_JSON"] = str(STYLE_JSON)
    env["QGIS_SVG_PATHS"] = os.pathsep.join(QGIS_SVG_PATHS)
    if QGIS_PREFIX_PATH:
        env["QGIS_PREFIX_PATH"] = QGIS_PREFIX_PATH
    if QGIS_BIN_PATH:
        env["QGIS_BIN_PATH"] = QGIS_BIN_PATH
    if QGIS_EXECUTABLE:
        env["QGIS_EXECUTABLE"] = QGIS_EXECUTABLE
    try:
        subprocess.run(
            [str(QGIS_PYTHON_BAT), str(BRIDGE_SCRIPT)],
            check=True,
            cwd=str(BASE_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if STYLE_JSON.exists():
            data = json.loads(STYLE_JSON.read_text(encoding="utf-8"))
            arrows = []
            for idx, item in enumerate(data.get("north_arrows", [])):
                p = Path(item.get("path", ""))
                if p.exists():
                    arrows.append({
                        "key": f"qgis_{idx}",
                        "label": item.get("label") or p.stem,
                        "src": f"/__qgis_svg?path={str(p)}",
                    })
            data["north_arrows"] = arrows or _default_catalog()["north_arrows"]
            return data
    except Exception as exc:
        return _default_catalog(f"QGIS external bridge failed: {exc}")
    return _default_catalog("QGIS external bridge returned no catalog")


def get_style_catalog() -> dict[str, Any]:
    if not ENABLE_QGIS_STYLE_BRIDGE:
        return _default_catalog("bridge disabled")
    if QGIS_BRIDGE_MODE == "external":
        return _external_catalog()
    return _default_catalog("unsupported bridge mode for python 3.10; use external")
