from __future__ import annotations

import re
from typing import Any

CLCD_CLASS_LABELS = {
    1: "耕地/农田",
    2: "森林/林地",
    3: "灌木",
    4: "草地",
    5: "水体",
    6: "冰雪",
    7: "裸地",
    8: "不透水面/建设用地",
    9: "湿地",
}

# User wording -> CLCD code.  These are semantic aliases; they do not trigger a
# task by themselves.  They only define the target land-cover mask once the AI
# planner has decided that a mapping task should run.
LANDCOVER_ALIASES: list[tuple[int, tuple[str, ...]]] = [
    (1, ("耕地", "农田", "农用地", "耕作地", "种植地")),
    (2, ("林地", "森林", "树林", "乔木林", "有林地")),
    (3, ("灌木", "灌丛", "灌木林")),
    (4, ("草地", "草原", "牧草地")),
    (5, ("水体", "水域", "湖泊", "河流", "湿地")),
    (6, ("冰雪", "冰川", "积雪")),
    (7, ("裸地", "裸土", "荒地", "未利用地")),
    (8, ("建设用地", "不透水面", "城镇", "城市建设区", "居民地")),
    (9, ("湿地", "沼泽", "湿地区")),
]

FULL_DOMAIN_PAT = re.compile(r"(全域|全市|全成都|全成都市|整个成都市|全范围|全部区域|所有区域|不掩膜|不要掩膜|取消掩膜|不按地类)")


def infer_landcover_scope(text: str | None) -> dict[str, Any]:
    """Infer target output mask semantics from the whole user request.

    Returns a stable payload used by mapping and display code.  If the user asks
    for a land-cover class map, non-target classes are masked out by default.
    """
    t = str(text or "")
    if FULL_DOMAIN_PAT.search(t):
        return {
            "map_scope": "full_domain",
            "output_scope": "full_domain",
            "mask_by_landcover": False,
            "mask_non_target_landcover": False,
            "landcover_class_code": None,
            "landcover_class_label": "全域",
            "scope_source": "request_text_semantic_full_domain",
        }
    # Prefer phrases near organic matter / mapping, but fall back to any explicit
    # land-cover class if the sentence is a mapping command already selected by AI.
    for code, aliases in LANDCOVER_ALIASES:
        for alias in aliases:
            pat_near_1 = rf"{re.escape(alias)}.{{0,18}}(?:土壤)?有机质|(?:土壤)?有机质.{{0,18}}{re.escape(alias)}"
            pat_near_2 = rf"(?:做|绘制|制作|生成|输出|显示|预测|制图).{{0,18}}{re.escape(alias)}|{re.escape(alias)}.{{0,18}}(?:制图|图|预测)"
            if re.search(pat_near_1, t) or re.search(pat_near_2, t):
                scope = "cropland" if code == 1 else "landcover_class"
                return {
                    "map_scope": scope,
                    "output_scope": scope,
                    "mask_by_landcover": True,
                    "mask_non_target_landcover": True,
                    "landcover_class_code": int(code),
                    "landcover_class_label": CLCD_CLASS_LABELS.get(code, alias),
                    "scope_source": "request_text_semantic_landcover_class",
                }
    return {
        "map_scope": "full_domain",
        "output_scope": "full_domain",
        "mask_by_landcover": False,
        "mask_non_target_landcover": False,
        "landcover_class_code": None,
        "landcover_class_label": "全域",
        "scope_source": "default_full_domain",
    }


def landcover_scope_title_prefix(scope: dict[str, Any] | None) -> str:
    s = scope or {}
    code = s.get("landcover_class_code")
    label = str(s.get("landcover_class_label") or "")
    if code == 1:
        return "耕地"
    if code == 2:
        return "林地"
    if code == 3:
        return "灌木"
    if code == 4:
        return "草地"
    if code == 5:
        return "水体"
    if code == 6:
        return "冰雪"
    if code == 7:
        return "裸地"
    if code == 8:
        return "建设用地"
    if code == 9:
        return "湿地"
    return ""
