from __future__ import annotations

import re
from typing import Any
from services.landcover_scope_service import infer_landcover_scope, landcover_scope_title_prefix

PALETTE_ALIASES = {
    # 连续单色/常用色系
    "蓝色": "ArcGIS Pro｜Blue Continuous", "蓝": "ArcGIS Pro｜Blue Continuous", "蓝色色带": "ArcGIS Pro｜Blue Continuous", "冷色": "ArcGIS Pro｜Blue Continuous",
    "绿色": "ArcGIS Pro｜Green Continuous", "绿": "ArcGIS Pro｜Green Continuous", "绿色色带": "ArcGIS Pro｜Green Continuous",
    "黄色": "ArcGIS Pro｜Brown Orange", "黄": "ArcGIS Pro｜Brown Orange", "黄色色带": "ArcGIS Pro｜Brown Orange", "棕黄": "ArcGIS Pro｜Brown Orange", "棕橙": "ArcGIS Pro｜Brown Orange",
    "红色": "ArcGIS Pro｜Heat", "红": "ArcGIS Pro｜Heat", "热力": "ArcGIS Pro｜Heat", "暖色": "ArcGIS Pro｜Heat",
    "紫色": "ArcGIS Pro｜Cyan Purple", "紫": "ArcGIS Pro｜Cyan Purple", "青紫": "ArcGIS Pro｜Cyan Purple", "青紫色带": "ArcGIS Pro｜Cyan Purple",
    "灰色": "ArcGIS Pro｜Gray Continuous", "灰": "ArcGIS Pro｜Gray Continuous", "黑白": "ArcGIS Pro｜Gray Continuous", "灰度": "ArcGIS Pro｜Gray Continuous",
    # 多色带/ArcGIS Pro 风格色带
    "黄绿蓝": "ArcGIS Pro｜Yellow Green Blue", "黄绿色": "ArcGIS Pro｜Yellow Green Blue", "黄绿": "ArcGIS Pro｜Yellow Green Blue", "YlGnBu": "ArcGIS Pro｜Yellow Green Blue",
    "黄橙红": "ArcGIS Pro｜Yellow Orange Red", "橙红": "ArcGIS Pro｜Yellow Orange Red", "YlOrRd": "ArcGIS Pro｜Yellow Orange Red",
    "红黄绿": "ArcGIS Pro｜RdYlGn", "RdYlGn": "ArcGIS Pro｜RdYlGn", "高低对比": "ArcGIS Pro｜RdYlGn",
    "光谱": "ArcGIS Pro｜Spectral", "Spectral": "ArcGIS Pro｜Spectral", "彩虹": "ArcGIS Pro｜Spectral",
    "地形": "ArcGIS Pro｜Terrain", "Terrain": "ArcGIS Pro｜Terrain",
    "Viridis": "ArcGIS Pro｜Viridis", "viridis": "ArcGIS Pro｜Viridis", "科学色带": "ArcGIS Pro｜Viridis",
}



def resolve_palette_from_text(text: str | None) -> tuple[str | None, str | None]:
    """Resolve a user-requested palette from natural language.

    Supports commands such as “换成蓝色”, “改为黄绿蓝色带”, “使用 Viridis”,
    “把当前图换成地形色带”.  Returns (palette_name, matched_alias).
    """
    t = str(text or "")
    if not t:
        return None, None
    has_style_verb = bool(re.search(r"(色带|颜色|配色|调色板|palette|colormap|cmap|换|改|改成|改为|换成|设置|设为|使用|用|选用|选择)", t, flags=re.I))
    if not has_style_verb:
        return None, None
    # Prefer aliases that are near explicit style words, then fall back to any alias in the sentence.
    for alias, ramp in sorted(PALETTE_ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True):
        pat1 = rf"(?:颜色|配色|色带|调色板|palette|colormap|cmap).{{0,10}}(?:设置为|设为|选择|选用|指定为|改成|改为|换成|变成|为|用)?\s*{re.escape(alias)}"
        pat2 = rf"(?:改成|改为|换成|换为|设置为|设为|选择|选用|使用|用)\s*{re.escape(alias)}(?:色带|配色|颜色|调色板)?"
        pat3 = rf"{re.escape(alias)}(?:色带|配色|颜色|调色板)"
        if re.search(pat1, t, flags=re.I) or re.search(pat2, t, flags=re.I) or re.search(pat3, t, flags=re.I):
            return ramp, alias
    for alias, ramp in sorted(PALETTE_ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True):
        if alias in t:
            return ramp, alias
    return None, None

def infer_map_scope(text: str | None) -> str:
    """Infer formal mapping scope from user wording.

    Supports arbitrary CLCD land-cover target classes.  The presence of these
    words does not trigger mapping; it only defines the mask once AI decides the
    user is asking to execute a map.
    """
    return str(infer_landcover_scope(text).get("map_scope") or "full_domain")



def _is_consultation_or_preparation_question(t: str) -> bool:
    """Return True for questions that mention a map/scope but are asking for advice.

    These must stay in the AI dialog trunk.  Words like “耕地有机质图” are
    semantic context here, not a request to change the current displayed layer
    or cartographic layout.
    """
    if not t:
        return False
    has_question = bool(re.search(r"(\?|？|哪些|什么|怎么|如何|为何|为什么|是否|能否|能不能|可不可以|需要|应该|准备|提供|收集|建议|推荐|解释|说明|介绍|流程|条件|要求)", t))
    has_consult = bool(re.search(r"(需要准备|准备哪些|需要哪些|要准备什么|需要什么|哪些.*(协变量|数据|资料|条件)|什么.*(协变量|数据|资料|条件)|怎么准备|如何准备|能不能用|是否可以|可不可以|适不适合|用于制图|用来制图|支撑制图|分析.*能不能|审查.*能不能|检查.*能不能)", t))
    explicit_style = bool(resolve_palette_from_text(t)[0]) or bool(re.search(r"(隐藏|显示|打开|关闭|保留|去掉|取消|改成|改为|换成|设置为|设为|调整|移动|放到|放在|居中|透明度|图例|比例尺|指北针|标题|图名|边框|图框|底图|样点)", t))
    explicit_layer_switch = bool(re.search(r"(切换|显示|打开|查看|隐藏|不显示).{0,12}(有机质图|不确定性|GCP|AOA|风险图|样点|采样点|底图|图层)", t, flags=re.I))
    if (has_question or has_consult) and not explicit_style and not explicit_layer_switch:
        return True
    return False


def _has_explicit_cartography_or_scope_instruction(t: str) -> bool:
    """Return True only when the user is modifying map layout/style/scope.

    General questions such as “需要准备哪些环境协变量” may contain “耕地土壤有机质制图”,
    but they are not cartographic layout instructions and must be answered by the AI dialog trunk.
    """
    if not t:
        return False
    if _is_consultation_or_preparation_question(t):
        return False
    # Style / layout components.  Palette changes may be phrased as
    # “换成蓝色 / 用 Viridis / 改为黄绿蓝色带”, where the color word appears
    # after the verb, so use the palette resolver rather than a single regex.
    if resolve_palette_from_text(t)[0]:
        return True
    if re.search(r"(色带|颜色|配色|调色板|palette|colormap|cmap|透明|图例|比例尺|指北针|北箭头|标题|图名|名称|边框|图框|主图|图面|地图).{0,20}(位置|居中|放|占|大小|宽度|面积|显示|隐藏|保留|添加|去掉|关闭|不要|改|换)", t, flags=re.I):
        return True
    if re.search(r"(位置)?居中|放在中间|居中显示|左上|右上|左下|右下", t):
        return True
    if re.search(r"(有机质图|主图|地图|图面).{0,12}(占|大小|宽度|面积).{0,8}\d{2,3}\s*%", t):
        return True
    # Explicit scope/mask changes.
    scope_payload = infer_landcover_scope(t)
    if scope_payload.get("mask_by_landcover"):
        return True
    if re.search(r"(耕地范围|耕地部分|林地范围|森林范围|草地范围|只(绘制|显示|输出).{0,8}(耕地|农田|林地|森林|草地|灌木|水体|裸地|建设用地)|非(耕地|林地|草地|目标地类).{0,12}(白色|掩膜|不显示|透明)|掩膜非(耕地|林地|草地|目标地类))", t):
        return True
    if re.search(r"(全域|全成都|全市|整个区域|所有区域|不掩膜|不要掩膜|取消掩膜).{0,12}(制图|显示|输出|绘制|预测|有机质图)?", t):
        return True
    return False


def parse_cartography_instruction(text: str | None) -> dict[str, Any]:
    t = str(text or "")
    out: dict[str, Any] = {}
    if not _has_explicit_cartography_or_scope_instruction(t):
        return out

    # Scope is written only when the user explicitly changes mapping scope.
    # A pure style command such as “换成黄色色带” must not reset a previous
    # cropland-only map back to full-domain display.
    scope_payload = infer_landcover_scope(t)
    if scope_payload.get("mask_by_landcover"):
        out.update(scope_payload)
        out["mask_non_cropland"] = True
        out["non_cropland_color"] = "#ffffff"
    elif re.search(r"(全域|全成都|全市|整个区域|所有区域|不掩膜|不要掩膜|取消掩膜)", t):
        out.update(scope_payload)
        out["map_scope"] = "full_domain"
        out["mask_non_cropland"] = False
        out["non_cropland_color"] = "#ffffff"

    # Map frame size: e.g. “有机质图占整个页面的80%”.
    m = re.search(r"(?:有机质图|主图|地图|图面).{0,12}(?:占|大小|宽度|面积).{0,8}(\d{2,3})\s*%", t)
    if m:
        pct = max(20, min(96, int(m.group(1))))
        out["map_frame_percent"] = pct
    elif re.search(r"(放大|大一点|更大)", t):
        out["map_frame_percent"] = 84
    elif re.search(r"(缩小|小一点|更小)", t):
        out["map_frame_percent"] = 65

    # Position.
    if re.search(r"(位置)?居中|放在中间|居中显示", t):
        out["map_position"] = "center"
    elif re.search(r"左上", t):
        out["map_position"] = "top_left"
    elif re.search(r"右上", t):
        out["map_position"] = "top_right"
    elif re.search(r"左下", t):
        out["map_position"] = "bottom_left"
    elif re.search(r"右下", t):
        out["map_position"] = "bottom_right"

    # Palette.  When cropland-only, this applies to cropland cells only.
    palette, matched_alias = resolve_palette_from_text(t)
    if palette:
        out["palette"] = palette
        out["palette_target"] = "target_landcover_only" if out.get("mask_by_landcover") or out.get("map_scope") in {"cropland", "landcover_class"} else "prediction"
        out["palette_alias"] = matched_alias

    # Cartographic elements.
    elements = {}
    for name, pats in {
        "title": ["标题", "图名", "名称"],
        "legend": ["图例", "色带"],
        "scale_bar": ["比例尺"],
        "north_arrow": ["指北针", "北箭头"],
        "border": ["边框", "图框"],
    }.items():
        joined = "|".join(pats)
        if re.search(rf"(不要|隐藏|去掉|关闭).{{0,8}}({joined})", t):
            elements[name] = False
        elif re.search(rf"(显示|保留|添加|加上|需要).{{0,8}}({joined})|({joined}).{{0,8}}(显示|保留|添加|加上|需要)", t):
            elements[name] = True
    if elements:
        out["layout_elements"] = elements

    return out

def merge_layout_instruction(current: dict | None, parsed: dict | None) -> dict:
    current = dict(current or {})
    parsed = dict(parsed or {})
    if not parsed:
        return current
    # Keep earlier manual settings unless the new instruction explicitly overrides them.
    for k, v in parsed.items():
        if k == "layout_elements":
            old = dict(current.get("layout_elements") or {})
            old.update(v or {})
            current[k] = old
        else:
            current[k] = v
    return current
