from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import rasterio
    from rasterio.warp import transform_bounds, reproject, Resampling
except Exception:  # pragma: no cover
    rasterio = None
    transform_bounds = None
    reproject = None
    Resampling = None

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None


_GREEN_RAMP = [
    (247, 252, 245), (229, 245, 224), (199, 233, 192), (161, 217, 155),
    (116, 196, 118), (65, 171, 93), (35, 139, 69), (0, 90, 50),
]
_BLUE_RAMP = [
    (247, 251, 255), (222, 235, 247), (198, 219, 239), (158, 202, 225),
    (107, 174, 214), (66, 146, 198), (33, 113, 181), (8, 69, 148),
]
_MAGMA_RAMP = [
    (0, 0, 4), (39, 11, 84), (85, 15, 109), (136, 34, 106),
    (186, 54, 85), (227, 89, 51), (249, 140, 10), (252, 195, 40), (252, 253, 191),
]
_VIRIDIS_RAMP = [
    (68, 1, 84), (71, 44, 122), (59, 81, 139), (44, 113, 142),
    (33, 144, 141), (39, 173, 129), (92, 200, 99), (170, 220, 50), (253, 231, 37),
]

RAMP_TABLE = {
    "green": _GREEN_RAMP,
    "blue": _BLUE_RAMP,
    "magma": _MAGMA_RAMP,
    "viridis": _VIRIDIS_RAMP,
}

DEFAULT_RISK_CLASSES = [
    (1, "高可信", (46, 160, 67, 255)),
    (2, "域内高不确定", (247, 183, 49, 255)),
    (3, "外推高风险", (220, 53, 69, 255)),
    (4, "虚假自信", (138, 43, 226, 255)),
]


def _font(size: int, bold: bool = False):
    if ImageFont is None:
        return None
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
    ]
    for p in candidates:
        try:
            if Path(p).exists():
                return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _text_size(draw, text: str, font) -> tuple[int, int]:
    try:
        box = draw.textbbox((0, 0), str(text), font=font)
        return int(box[2] - box[0]), int(box[3] - box[1])
    except Exception:
        return (len(str(text)) * 10, 14)


def _draw_centered_text(draw, xy: tuple[int, int], text: str, font, fill=(20, 20, 20)):
    w, h = _text_size(draw, text, font)
    draw.text((int(xy[0] - w / 2), int(xy[1] - h / 2)), text, font=font, fill=fill)


def _make_lut(ramp: Sequence[tuple[int, int, int]], n: int = 256) -> np.ndarray:
    arr = np.asarray(ramp, dtype="float32")
    xp = np.linspace(0.0, 1.0, len(arr))
    x = np.linspace(0.0, 1.0, n)
    lut = np.zeros((n, 3), dtype="uint8")
    for i in range(3):
        lut[:, i] = np.interp(x, xp, arr[:, i]).astype("uint8")
    return lut


def _colorize_continuous(data: np.ndarray, valid: np.ndarray, ramp_name: str = "green") -> tuple[np.ndarray, tuple[float, float]]:
    vals = data[valid & np.isfinite(data)]
    if vals.size == 0:
        rgba = np.zeros((*data.shape, 4), dtype="uint8")
        return rgba, (float("nan"), float("nan"))
    lo, hi = np.nanpercentile(vals, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        if hi <= lo:
            hi = lo + 1.0
    scaled = np.clip((data - lo) / max(hi - lo, 1e-9), 0, 1)
    idx = np.clip((scaled * 255).astype("int16"), 0, 255)
    lut = _make_lut(RAMP_TABLE.get(str(ramp_name).lower(), _GREEN_RAMP))
    rgb = lut[idx]
    alpha = np.where(valid & np.isfinite(data), 255, 0).astype("uint8")
    rgba = np.dstack([rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2], alpha])
    return rgba, (float(lo), float(hi))


def _class_rgba(data: np.ndarray, class_items: Sequence[tuple[int, str, tuple[int, int, int, int]]]) -> np.ndarray:
    rgba = np.zeros((*data.shape, 4), dtype="uint8")
    for value, _label, color in class_items:
        rgba[np.isclose(data, float(value), rtol=0, atol=1e-6)] = np.asarray(color, dtype="uint8")
    return rgba


def _read_mask_on_ref(mask_tif_path: str | Path | None, ref_ds):
    if not mask_tif_path or rasterio is None:
        return None
    try:
        mp = Path(mask_tif_path)
        if not mp.exists():
            return None
        with rasterio.open(mp) as ms:
            if ms.width == ref_ds.width and ms.height == ref_ds.height and ms.crs == ref_ds.crs and ms.transform == ref_ds.transform:
                return ms.read(1)
            if reproject is None or Resampling is None:
                return None
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


def _bounds_lonlat(ds) -> tuple[float, float, float, float] | None:
    try:
        if ds.crs and transform_bounds is not None:
            west, south, east, north = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
            return float(west), float(south), float(east), float(north)
    except Exception:
        pass
    try:
        return float(ds.bounds.left), float(ds.bounds.bottom), float(ds.bounds.right), float(ds.bounds.top)
    except Exception:
        return None


def _map_width_km(ds) -> float:
    try:
        if ds.crs and getattr(ds.crs, "is_projected", False):
            return max(abs(float(ds.bounds.right - ds.bounds.left)) / 1000.0, 0.1)
    except Exception:
        pass
    b = _bounds_lonlat(ds)
    if b:
        west, south, east, north = b
        mean_lat = (south + north) / 2.0
        return max(abs(east - west) * 111.32 * max(math.cos(math.radians(mean_lat)), 0.2), 0.1)
    return 10.0


def _nice_scale_km(width_km: float) -> float:
    candidates = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    target = max(float(width_km) * 0.20, 0.1)
    eligible = [x for x in candidates if x <= target]
    return eligible[-1] if eligible else candidates[0]


def _paste_map(canvas, map_rgba: np.ndarray, map_box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = map_box
    box_w = x1 - x0
    box_h = y1 - y0
    img = Image.fromarray(map_rgba, mode="RGBA")
    w, h = img.size
    if w <= 0 or h <= 0:
        return map_box
    scale = min(box_w / w, box_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((nw, nh), Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR)
    px = x0 + (box_w - nw) // 2
    py = y0 + (box_h - nh) // 2
    canvas.alpha_composite(img, (px, py))
    return px, py, px + nw, py + nh


def _draw_north_arrow(draw, box: tuple[int, int, int, int]):
    x0, y0, x1, y1 = box
    cx = (x0 + x1) // 2
    top = y0 + 20
    bottom = y1 - 26
    mid = (top + bottom) // 2
    draw.polygon([(cx, top), (cx - 18, mid + 8), (cx, mid - 2), (cx + 18, mid + 8)], fill=(20, 20, 20), outline=(20, 20, 20))
    draw.polygon([(cx, bottom), (cx - 13, mid + 12), (cx, mid + 5), (cx + 13, mid + 12)], fill=(255, 255, 255), outline=(20, 20, 20))
    f = _font(28, bold=True)
    _draw_centered_text(draw, (cx, y0 + 8), "N", f, fill=(20, 20, 20))
    # No opaque background around the north arrow; keep it as a transparent map component.


def _draw_scale_bar(draw, map_drawn_box: tuple[int, int, int, int], width_km: float):
    x0, y0, x1, y1 = map_drawn_box
    bar_km = _nice_scale_km(width_km)
    bar_px = int(max(70, min((x1 - x0) * 0.38, (x1 - x0) * (bar_km / max(width_km, 1e-9)))))
    bx = x0 + 48
    by = y1 - 56
    h = 18
    f = _font(22)
    # Transparent scale component: only the scale bar and labels are drawn.
    seg = bar_px / 4.0
    for i in range(4):
        fill = (30, 30, 30) if i % 2 == 0 else (255, 255, 255)
        draw.rectangle((int(bx + i * seg), by, int(bx + (i + 1) * seg), by + h), fill=fill, outline=(30, 30, 30), width=1)
    draw.text((bx - 5, by + h + 4), "0", font=f, fill=(20, 20, 20), anchor="ma")
    draw.text((bx + bar_px, by + h + 4), f"{bar_km:g} km", font=f, fill=(20, 20, 20), anchor="ma")


def _draw_continuous_legend(draw, canvas, legend_box: tuple[int, int, int, int], title: str, value_range: tuple[float, float], ramp_name: str):
    x0, y0, x1, y1 = legend_box
    # Transparent legend background: the legend does not cover the map with a white card.
    title_f = _font(25, bold=True)
    label_f = _font(21)
    draw.text((x0 + 18, y0 + 18), title or "图例", font=title_f, fill=(20, 20, 20))
    gx0, gy0 = x0 + 34, y0 + 78
    gx1, gy1 = x0 + 70, y1 - 52
    ramp = RAMP_TABLE.get(str(ramp_name).lower(), _GREEN_RAMP)
    lut = _make_lut(ramp, max(2, gy1 - gy0))
    grad = np.zeros((gy1 - gy0, gx1 - gx0, 4), dtype="uint8")
    grad[:, :, 3] = 255
    for row in range(gy1 - gy0):
        grad[row, :, :3] = lut[(gy1 - gy0 - 1) - row]
    canvas.alpha_composite(Image.fromarray(grad, mode="RGBA"), (gx0, gy0))
    draw.rectangle((gx0, gy0, gx1, gy1), outline=(40, 40, 40), width=1)
    lo, hi = value_range
    def fmt(v):
        if not np.isfinite(v):
            return "N/A"
        if abs(v) >= 100:
            return f"{v:.0f}"
        if abs(v) >= 10:
            return f"{v:.1f}"
        return f"{v:.2f}"
    draw.text((gx1 + 16, gy0 - 10), fmt(hi), font=label_f, fill=(20, 20, 20))
    draw.text((gx1 + 16, gy1 - 16), fmt(lo), font=label_f, fill=(20, 20, 20))


def _draw_class_legend(draw, legend_box: tuple[int, int, int, int], title: str, class_items: Sequence[tuple[int, str, tuple[int, int, int, int]]]):
    x0, y0, x1, y1 = legend_box
    # Transparent legend background: the legend does not cover the map with a white card.
    title_f = _font(25, bold=True)
    label_f = _font(21)
    draw.text((x0 + 18, y0 + 18), title or "图例", font=title_f, fill=(20, 20, 20))
    y = y0 + 75
    for _value, label, color in class_items:
        draw.rectangle((x0 + 24, y, x0 + 58, y + 26), fill=tuple(color), outline=(40, 40, 40), width=1)
        draw.text((x0 + 72, y - 2), str(label), font=label_f, fill=(20, 20, 20))
        y += 48


def create_formal_continuous_png(
    tif_path: str | Path,
    out_path: str | Path,
    title: str,
    legend_title: str = "值",
    ramp_name: str = "green",
    mask_tif_path: str | Path | None = None,
    noncrop_color: tuple[int, int, int, int] | None = None,
) -> str:
    if rasterio is None or Image is None:
        return ""
    try:
        tif = Path(tif_path)
        out = Path(out_path)
        with rasterio.open(tif) as ds:
            arr = ds.read(1, masked=True).astype("float32")
            data = arr.filled(np.nan)
            valid = np.isfinite(data) & ~np.ma.getmaskarray(arr)
            if ds.nodata is not None:
                valid &= ~np.isclose(data, float(ds.nodata), rtol=0, atol=max(1e-6, abs(float(ds.nodata)) * 1e-6))
            rgba, value_range = _colorize_continuous(data, valid, ramp_name=ramp_name)
            mask = _read_mask_on_ref(mask_tif_path, ds)
            if mask is not None:
                if noncrop_color is not None:
                    rgba[mask == 0] = np.asarray(noncrop_color, dtype="uint8")
                rgba[mask == 255] = (0, 0, 0, 0)
            width_km = _map_width_km(ds)
        return _compose_formal_map(rgba, out, title, legend_title, value_range, ramp_name, width_km, class_items=None)
    except Exception:
        return ""


def create_formal_class_png(
    tif_path: str | Path,
    out_path: str | Path,
    title: str,
    legend_title: str = "风险分区",
    class_items: Sequence[tuple[int, str, tuple[int, int, int, int]]] | None = None,
) -> str:
    if rasterio is None or Image is None:
        return ""
    try:
        class_items = list(class_items or DEFAULT_RISK_CLASSES)
        tif = Path(tif_path)
        out = Path(out_path)
        with rasterio.open(tif) as ds:
            arr = ds.read(1, masked=True).astype("float32")
            data = arr.filled(np.nan)
            rgba = _class_rgba(data, class_items)
            if ds.nodata is not None:
                rgba[np.isclose(data, float(ds.nodata), rtol=0, atol=max(1e-6, abs(float(ds.nodata)) * 1e-6))] = (0, 0, 0, 0)
            rgba[~np.isfinite(data)] = (0, 0, 0, 0)
            width_km = _map_width_km(ds)
        return _compose_formal_map(rgba, out, title, legend_title, (float("nan"), float("nan")), "green", width_km, class_items=class_items)
    except Exception:
        return ""


def _compose_formal_map(
    map_rgba: np.ndarray,
    out_path: Path,
    title: str,
    legend_title: str,
    value_range: tuple[float, float],
    ramp_name: str,
    width_km: float,
    class_items: Sequence[tuple[int, str, tuple[int, int, int, int]]] | None = None,
) -> str:
    canvas_w, canvas_h = 2400, 1600
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")
    title_f = _font(46, bold=True)
    # Title is a cartographic element and must stay inside the neatline/map layout frame.
    _draw_centered_text(draw, (canvas_w // 2, 165), title or "专题图", title_f, fill=(15, 23, 42))
    map_box = (115, 220, 1785, 1450)
    drawn = _paste_map(canvas, map_rgba, map_box)
    draw.rectangle(drawn, outline=(20, 20, 20), width=3)
    draw.rectangle((70, 130, 2330, 1525), outline=(20, 20, 20), width=3)
    legend_box = (1855, 300, 2255, 1110)
    if class_items:
        _draw_class_legend(draw, legend_box, legend_title or "图例", class_items)
    else:
        _draw_continuous_legend(draw, canvas, legend_box, legend_title or "图例", value_range, ramp_name)
    _draw_north_arrow(draw, (1970, 150, 2140, 280))
    _draw_scale_bar(draw, drawn, width_km)
    note_f = _font(18)
    # No note outside the map frame; keep all cartographic elements within the neatline.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_path, quality=95)
    return str(out_path)
