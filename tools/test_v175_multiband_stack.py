from pathlib import Path
import json
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from services.pro_platform_model_csv_pipeline import PlatformModelCsvPipeline


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        stack = root / "cov_stack_250m.tif"
        transform = from_origin(500000, 3400000, 250, 250)
        data = np.stack([
            np.ones((10, 10), dtype="float32") * 100,
            np.arange(100, dtype="float32").reshape(10, 10),
            np.zeros((10, 10), dtype="float32"),  # zero must be valid
        ])
        with rasterio.open(stack, "w", driver="GTiff", height=10, width=10, count=3,
                           dtype="float32", crs="EPSG:32648", transform=transform,
                           nodata=-9999.0) as dst:
            dst.write(data)
            dst.set_band_description(1, "DEM_point_m")
            dst.set_band_description(2, "slope_250m_deg")
            dst.set_band_description(3, "zero_valid_test")
        mapping = root / "cov_stack_250m_band_mapping.csv"
        pd.DataFrame({
            "band": [1, 2, 3],
            "feature_name": ["DEM_point_m", "slope_250m_deg", "zero_valid_test"],
            "resampling": ["bilinear", "bilinear", "bilinear"],
        }).to_csv(mapping, index=False, encoding="utf-8-sig")
        pipe = PlatformModelCsvPipeline(out_root=root / "out", task_id="test_v175")
        files = [stack, mapping]
        names, audit = pipe._multiband_feature_names(stack, files)
        assert names[1] == "DEM_point_m", names
        assert names[2] == "slope_250m_deg", names
        assert names[3] == "zero_valid_test", names
        assert audit["source"] == "companion_mapping_file", audit
        assert pipe._is_multiband_stack_raster(stack) is True
        print(json.dumps({"ok": True, "band_names": names, "audit_source": audit["source"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
