from __future__ import annotations

"""Small smoke test for V168 auto-resolution policy.
Run from work_progress:
    python tools/test_v168_resolution_policy.py
"""

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.pro_platform_model_csv_pipeline import PlatformModelCsvPipeline


def main() -> None:
    p = PlatformModelCsvPipeline(Path("data") / "_v168_resolution_test")

    # Fused CSV declares 250 m while uploaded CLCD may be 30 m: output should be 250 m.
    t = {"region": "成都市", "region_level": "city", "resolution_m": None, "is_city": True}
    df = pd.DataFrame({"target_scale_m": [250, 250, 250]})
    out = p._infer_target_resolution_from_user_rasters([], t, sample_df=df)
    assert float(out["resolution_m"]) == 250.0, out

    # All user data are 30 m and AOI is city: output should be 250 m.
    t = {"region": "成都市", "region_level": "city", "resolution_m": None, "is_city": True}
    df = pd.DataFrame({"target_scale_m": [30, 30, 30]})
    out = p._infer_target_resolution_from_user_rasters([], t, sample_df=df)
    assert float(out["resolution_m"]) == 250.0, out

    # All user data are 30 m and AOI is county/district: output should remain 30 m.
    t = {"region": "温江区", "region_level": "county", "resolution_m": None, "is_city": True}
    out = p._infer_target_resolution_from_user_rasters([], t, sample_df=df)
    assert float(out["resolution_m"]) == 30.0, out

    # Mixed 30/250/500 m rasters: choose coarsest = 500 m.
    p._scan_files = lambda roots: [Path("a_30.tif"), Path("b_250.tif"), Path("c_500.tif")]
    p._raster_resolution_meters = lambda f: float(str(f).split("_")[1].split(".")[0])
    t = {"region": "成都市", "region_level": "city", "resolution_m": None, "is_city": True}
    out = p._infer_target_resolution_from_user_rasters([Path(".")], t, sample_df=pd.DataFrame({"x": [1]}))
    assert float(out["resolution_m"]) == 500.0, out

    print("V168 resolution policy smoke test passed.")


if __name__ == "__main__":
    main()
