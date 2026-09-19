from pathlib import Path
import math
import numpy as np
import plotly.graph_objects as go


MONOCHROME_PALETTES = {
    "green": [
        [0.0, "#f7fcf5"], [0.2, "#d9f0d3"], [0.4, "#a6dba0"], [0.6, "#5aae61"], [0.8, "#1b7837"], [1.0, "#00441b"],
    ],
    "blue": [
        [0.0, "#f7fbff"], [0.2, "#deebf7"], [0.4, "#9ecae1"], [0.6, "#6baed6"], [0.8, "#2171b5"], [1.0, "#08306b"],
    ],
    "purple": [
        [0.0, "#fcfbfd"], [0.2, "#efedf5"], [0.4, "#bcbddc"], [0.6, "#9e9ac8"], [0.8, "#756bb1"], [1.0, "#3f007d"],
    ],
    "orange": [
        [0.0, "#fff5eb"], [0.2, "#fee6ce"], [0.4, "#fdd0a2"], [0.6, "#fdae6b"], [0.8, "#e6550d"], [1.0, "#7f2704"],
    ],
    "red": [
        [0.0, "#fff5f0"], [0.2, "#fee0d2"], [0.4, "#fcbba1"], [0.6, "#fc9272"], [0.8, "#de2d26"], [1.0, "#67000d"],
    ],
    "gray": [
        [0.0, "#ffffff"], [0.2, "#f0f0f0"], [0.4, "#d9d9d9"], [0.6, "#969696"], [0.8, "#636363"], [1.0, "#252525"],
    ],
}

DEFAULT_LAYOUT_CFG = {
    "graph_height": 760,
    "map_x": 0.08,
    "map_y": 0.08,
    "map_w": 0.78,
    "map_h": 0.82,
    "legend_x": 1.03,
    "legend_y": 0.50,
    "legend_len": 0.72,
    "legend_thickness": 18,
    "north_x": 0.90,
    "north_y": 0.81,
    "north_size": 0.060,
    "scale_x": 0.12,
    "scale_y": 0.095,
    "scale_width": 0.18,
    "scale_height": 0.022,
    "map_left_pct": 0.05,
    "map_top_pct": 0.06,
    "map_width_pct": 0.76,
    "map_height_pct": 0.80,
    "legend_left_pct": 0.84,
    "legend_top_pct": 0.16,
    "legend_width_pct": 0.12,
    "legend_height_pct": 0.48,
    "north_left_pct": 0.82,
    "north_top_pct": 0.03,
    "north_width_pct": 0.13,
    "north_height_pct": 0.17,
    "scale_left_pct": 0.09,
    "scale_top_pct": 0.87,
    "scale_width_pct": 0.28,
    "scale_height_pct": 0.10,
}


def palette_dropdown_options():
    return [
        {"label": "绿色", "value": "green"},
        {"label": "蓝色", "value": "blue"},
        {"label": "紫色", "value": "purple"},
        {"label": "橙色", "value": "orange"},
        {"label": "红色", "value": "red"},
        {"label": "灰色", "value": "gray"},
    ]


def default_layout_cfg():
    return dict(DEFAULT_LAYOUT_CFG)


def _merge_cfg(layout_cfg: dict | None) -> dict:
    cfg = default_layout_cfg()
    if layout_cfg:
        cfg.update(layout_cfg)
    cfg["map_x"] = float(min(max(cfg["map_x"], 0.02), 0.45))
    cfg["map_y"] = float(min(max(cfg["map_y"], 0.02), 0.35))
    cfg["map_w"] = float(min(max(cfg["map_w"], 0.30), 0.94 - cfg["map_x"]))
    cfg["map_h"] = float(min(max(cfg["map_h"], 0.30), 0.95 - cfg["map_y"]))
    cfg["legend_x"] = float(min(max(cfg["legend_x"], 0.70), 1.20))
    cfg["legend_y"] = float(min(max(cfg["legend_y"], 0.08), 0.92))
    cfg["legend_len"] = float(min(max(cfg["legend_len"], 0.20), 0.95))
    cfg["legend_thickness"] = int(min(max(int(cfg["legend_thickness"]), 8), 40))
    cfg["north_x"] = float(min(max(cfg["north_x"], 0.03), 0.97))
    cfg["north_y"] = float(min(max(cfg["north_y"], 0.08), 0.95))
    cfg["north_size"] = float(min(max(cfg["north_size"], 0.020), 0.12))
    cfg["scale_x"] = float(min(max(cfg["scale_x"], 0.03), 0.90))
    cfg["scale_y"] = float(min(max(cfg["scale_y"], 0.02), 0.28))
    cfg["scale_width"] = float(min(max(cfg["scale_width"], 0.04), 0.45))
    cfg["scale_height"] = float(min(max(cfg["scale_height"], 0.008), 0.06))
    cfg["graph_height"] = int(min(max(int(cfg["graph_height"]), 480), 1200))
    return cfg


def palette_css_gradient(palette: str, direction: str = "to top") -> str:
    colors = [c for _, c in MONOCHROME_PALETTES.get(palette, MONOCHROME_PALETTES["green"])]
    return f"linear-gradient({direction}, {', '.join(colors)})"


def _safe_transform_bounds(src):
    try:
        from rasterio.warp import transform_bounds
        if src.crs:
            return transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
    except Exception:
        pass
    return (0.0, float(src.width), 0.0, float(src.height))


def _nice_scale_km(map_width_km: float) -> float:
    candidates = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    target = max(map_width_km * 0.18, 0.5)
    eligible = [c for c in candidates if c <= target]
    return eligible[-1] if eligible else candidates[0]


def _display_bounds(x_min, x_max, y_min, y_max):
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    x_pad = x_span * 0.12
    y_pad = y_span * 0.12
    return (x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)


def _read_raster_preview(tif_path, max_dim: int = 500):
    import rasterio

    p = Path(tif_path)
    with rasterio.open(p) as src:
        arr = src.read(1).astype("float32")
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        h, w = arr.shape
        row_step = max(1, h // max_dim)
        col_step = max(1, w // max_dim)
        arr2 = arr[::row_step, ::col_step]
        x_min, y_min, x_max, y_max = _safe_transform_bounds(src)

    geo_mode = x_max > x_min and y_max > y_min
    if geo_mode:
        x_vals = np.linspace(x_min, x_max, arr2.shape[1])
        y_vals = np.linspace(y_max, y_min, arr2.shape[0])
        disp_x_min, disp_x_max, disp_y_min, disp_y_max = _display_bounds(x_min, x_max, y_min, y_max)
        mean_lat = (y_min + y_max) / 2.0
        width_km = max((x_max - x_min) * 111.32 * max(math.cos(math.radians(mean_lat)), 0.2), 0.5)
        scale_km = _nice_scale_km(width_km)
    else:
        x_vals = np.arange(arr2.shape[1])
        y_vals = np.arange(arr2.shape[0])
        disp_x_min, disp_x_max, disp_y_min, disp_y_max = _display_bounds(float(x_vals.min()), float(x_vals.max()), float(y_vals.min()), float(y_vals.max()))
        scale_km = None

    finite = np.isfinite(arr2)
    zmin = float(np.nanpercentile(arr2, 2)) if finite.any() else None
    zmax = float(np.nanpercentile(arr2, 98)) if finite.any() else None
    return {
        "array": arr2,
        "x_vals": x_vals,
        "y_vals": y_vals,
        "disp_bounds": (disp_x_min, disp_x_max, disp_y_min, disp_y_max),
        "raw_bounds": (x_min, x_max, y_min, y_max),
        "zmin": zmin,
        "zmax": zmax,
        "geo_mode": geo_mode,
        "scale_km": scale_km,
        "path": str(p),
        "name": p.name,
    }


def build_raster_canvas_bundle(tif_path, title: str = "", palette: str = "green", legend_title: str = "值"):
    preview = _read_raster_preview(tif_path)
    colorscale = MONOCHROME_PALETTES.get(palette, MONOCHROME_PALETTES["green"])
    disp_x_min, disp_x_max, disp_y_min, disp_y_max = preview["disp_bounds"]

    fig = go.Figure(data=go.Heatmap(
        z=preview["array"],
        x=preview["x_vals"],
        y=preview["y_vals"],
        zmin=preview["zmin"],
        zmax=preview["zmax"],
        colorscale=colorscale,
        showscale=False,
        hovertemplate=f"X=%{{x:.4f}}<br>Y=%{{y:.4f}}<br>{legend_title}=%{{z:.3f}}<extra></extra>",
    ))
    fig.update_layout(
        title=None,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial", size=12, color="#111827"),
        dragmode=False,
    )
    fig.update_xaxes(
        range=[disp_x_min, disp_x_max],
        showline=True,
        linecolor="black",
        linewidth=1.1,
        mirror=True,
        ticks="",
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        fixedrange=True,
    )
    fig.update_yaxes(
        range=[disp_y_min, disp_y_max],
        showline=True,
        linecolor="black",
        linewidth=1.1,
        mirror=True,
        ticks="",
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        fixedrange=True,
        scaleanchor="x" if preview["geo_mode"] else None,
        scaleratio=1,
    )
    return {
        "figure": fig,
        "gradient_css": palette_css_gradient(palette),
        "legend_title": legend_title,
        "zmin": preview["zmin"],
        "zmax": preview["zmax"],
        "scale_label": f"{preview['scale_km']:g} km" if preview["scale_km"] is not None else "比例尺",
        "title": title or preview["name"],
        "path": preview["path"],
    }


def _add_compass_rose(fig, cfg: dict):
    cx = cfg["north_x"]
    cy = cfg["north_y"]
    r = cfg["north_size"] * 0.42
    needle = cfg["north_size"] * 0.82

    fig.add_shape(type="circle", xref="paper", yref="paper", x0=cx - r, x1=cx + r, y0=cy - r, y1=cy + r, line=dict(color="black", width=1.4), fillcolor="rgba(255,255,255,0.90)")
    fig.add_shape(type="line", xref="paper", yref="paper", x0=cx, y0=cy - needle, x1=cx, y1=cy + needle, line=dict(color="black", width=1.6))
    fig.add_shape(type="line", xref="paper", yref="paper", x0=cx - needle, y0=cy, x1=cx + needle, y1=cy, line=dict(color="black", width=1.2))
    fig.add_shape(type="path", xref="paper", yref="paper", path=(f"M {cx} {cy + needle} L {cx - r * 0.35} {cy + r * 0.18} L {cx + r * 0.35} {cy + r * 0.18} Z"), line=dict(color="black", width=1.2), fillcolor="black")
    fig.add_shape(type="path", xref="paper", yref="paper", path=(f"M {cx} {cy - needle} L {cx - r * 0.23} {cy - r * 0.12} L {cx + r * 0.23} {cy - r * 0.12} Z"), line=dict(color="black", width=1.0), fillcolor="white")
    for label, dx, dy, size in [("N", 0, needle + r * 0.48, 15), ("E", needle + r * 0.52, 0, 10), ("S", 0, -(needle + r * 0.48), 10), ("W", -(needle + r * 0.52), 0, 10)]:
        fig.add_annotation(x=cx + dx, y=cy + dy, xref="paper", yref="paper", text=label, showarrow=False, font=dict(size=size, color="black", family="Arial Black" if label == "N" else "Arial"))


def _add_scale_bar(fig, lon_min, lon_max, lat_min, lat_max, cfg: dict):
    mean_lat = (lat_min + lat_max) / 2.0
    width_km = max((lon_max - lon_min) * 111.32 * max(math.cos(math.radians(mean_lat)), 0.2), 0.5)
    map_domain_width = cfg["map_w"]
    visual_fraction = min(max(cfg["scale_width"] * map_domain_width, 0.02), 0.60)
    bar_km = _nice_scale_km(width_km * min(max(cfg["scale_width"], 0.03), 0.50))
    x0 = cfg["scale_x"]
    y0 = cfg["scale_y"]
    w = visual_fraction
    h = cfg["scale_height"]
    seg = w / 4.0
    fills = ["black", "white", "black", "white"]

    fig.add_shape(type="rect", xref="paper", yref="paper", x0=x0 - 0.015, x1=x0 + w + 0.015, y0=y0 - 0.018, y1=y0 + h + 0.022, line=dict(color="rgba(0,0,0,0)", width=0), fillcolor="rgba(255,255,255,0.84)", layer="below")
    for i in range(4):
        fig.add_shape(type="rect", xref="paper", yref="paper", x0=x0 + i * seg, x1=x0 + (i + 1) * seg, y0=y0, y1=y0 + h, line=dict(color="black", width=1), fillcolor=fills[i])
    fig.add_annotation(x=x0, y=y0 - 0.012, xref="paper", yref="paper", text="0", showarrow=False, xanchor="center", yanchor="top", font=dict(size=11, color="black"))
    fig.add_annotation(x=x0 + w, y=y0 - 0.012, xref="paper", yref="paper", text=f"{bar_km:g} km", showarrow=False, xanchor="center", yanchor="top", font=dict(size=11, color="black"))


def build_raster_figure(tif_path, title: str = "", palette: str = "green", legend_title: str = "值", layout_cfg: dict | None = None):
    preview = _read_raster_preview(tif_path)
    cfg = _merge_cfg(layout_cfg)
    colorscale = MONOCHROME_PALETTES.get(palette, MONOCHROME_PALETTES["green"])
    x_min, x_max, y_min, y_max = preview["raw_bounds"]
    disp_x_min, disp_x_max, disp_y_min, disp_y_max = preview["disp_bounds"]

    fig = go.Figure(data=go.Heatmap(
        z=preview["array"],
        x=preview["x_vals"],
        y=preview["y_vals"],
        zmin=preview["zmin"],
        zmax=preview["zmax"],
        colorscale=colorscale,
        colorbar=dict(title=legend_title, len=cfg["legend_len"], thickness=cfg["legend_thickness"], x=cfg["legend_x"], y=cfg["legend_y"], ticks="outside"),
        hovertemplate=f"X=%{{x:.4f}}<br>Y=%{{y:.4f}}<br>{legend_title}=%{{z:.3f}}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title or Path(tif_path).name, x=0.5, xanchor="center", font=dict(size=20)),
        margin=dict(l=42, r=110, t=68, b=36),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial", size=12, color="#111827"),
        shapes=[dict(type="rect", xref="paper", yref="paper", x0=0.0, y0=0.0, x1=1.0, y1=1.0, line=dict(color="black", width=1.2), fillcolor="rgba(0,0,0,0)")],
    )
    fig.update_xaxes(title="", range=[disp_x_min, disp_x_max], domain=[cfg["map_x"], cfg["map_x"] + cfg["map_w"]], showline=True, linecolor="black", linewidth=1.2, mirror=True, ticks="", showticklabels=False, showgrid=False, zeroline=False, fixedrange=True)
    fig.update_yaxes(title="", range=[disp_y_min, disp_y_max], domain=[cfg["map_y"], cfg["map_y"] + cfg["map_h"]], showline=True, linecolor="black", linewidth=1.2, mirror=True, ticks="", showticklabels=False, showgrid=False, zeroline=False, fixedrange=True, scaleanchor="x" if preview["geo_mode"] else None, scaleratio=1)

    _add_compass_rose(fig, cfg)
    if preview["geo_mode"]:
        _add_scale_bar(fig, x_min, x_max, y_min, y_max, cfg)

    return fig
