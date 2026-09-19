"""Minimal V172 zero-safe missing-value policy smoke test."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.pro_platform_model_csv_pipeline import PlatformModelCsvPipeline

os.environ.setdefault("PRO_ZERO_NODATA_POLICY", "never")
p = PlatformModelCsvPipeline(out_root=ROOT / "tmp_v172_zero_test", task_id="test_v172_zero_policy")

assert p._zero_should_be_nodata("slope", 0) is False
assert p._is_missing_raster_value(0, None, "slope", masked=False)[0] is False
assert p._is_missing_raster_value(0, 0, "slope", masked=False)[0] is False
assert p._is_missing_raster_value(1e31, None, "any", masked=False)[0] is True
assert p._is_missing_raster_value(float("nan"), None, "any", masked=False)[0] is True
assert p._is_missing_raster_value(-9999, -9999, "any", masked=False)[0] is True

print("V172 zero-safe policy OK: ordinary 0 is valid; only explicit non-zero NoData, NaN/Inf, mask/out-of-bounds and extreme fill values are missing.")
