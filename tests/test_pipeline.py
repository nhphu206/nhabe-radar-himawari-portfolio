"""Kiểm thử nhanh cho các hợp đồng quan trọng, không đọc dữ liệu raw lớn."""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhabe_preprocessing.common_grid import build_common_grid
from nhabe_preprocessing.config import load_config
from nhabe_preprocessing.io_utils import atomic_write_json, atomic_write_npz
from nhabe_preprocessing.radar import classify_signature
from nhabe_preprocessing.time_matching import match_nearest_cycle
from nhabe_preprocessing.validation import validate_arrays, validate_sample_files


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()

    def test_common_grid_geometry(self) -> None:
        grid = build_common_grid(self.config)
        self.assertEqual(grid.radius.shape, (256, 256))
        self.assertAlmostEqual(float(np.diff(grid.x)[0]), 1562.5)
        self.assertAlmostEqual(float(np.diff(grid.y)[0]), -1562.5)
        self.assertFalse(np.any(grid.x == 0.0))
        self.assertFalse(np.any(grid.y == 0.0))

    def test_signature_classifier(self) -> None:
        for label, signature in self.config["radar"]["signatures"].items():
            metadata = {
                **signature,
                "gate_spacing_constant": True,
                "fields": sorted(signature["fields"]),
            }
            self.assertEqual(classify_signature(metadata, self.config), label)
        metadata["gate_spacing_m"] = 999.0
        self.assertEqual(classify_signature(metadata, self.config), "UNKNOWN")

    def test_time_matching_tolerance_and_tie_break(self) -> None:
        radar = {
            "date": "20250801",
            "radar_representative_time_utc": "2025-08-01T00:05:00Z",
        }
        satellites = [
            {"date": "20250801", "cycle": "0010",
             "nominal_start_utc": "2025-08-01T00:10:00Z",
             "local_nhabe_line_time_utc": "2025-08-01T00:06:00Z"},
            {"date": "20250801", "cycle": "0000",
             "nominal_start_utc": "2025-08-01T00:00:00Z",
             "local_nhabe_line_time_utc": "2025-08-01T00:04:00Z"},
        ]
        result = match_nearest_cycle(radar, satellites, 180.0)
        self.assertEqual(result["matched_himawari_cycle"], "0000")
        self.assertEqual(result["time_match_status"], "MATCHED")
        self.assertEqual(result["signed_delta_seconds"], -60.0)

    def test_array_contract_and_sample_files(self) -> None:
        radar = np.full((256, 256), np.nan, dtype=np.float32)
        radar[0, 0] = 1.0
        radar_valid = np.isfinite(radar)
        satellite = np.ones((5, 256, 256), dtype=np.float32)
        arrays = {
            "radar_reflectivity_dbz": radar,
            "radar_data_valid_mask": radar_valid,
            "radar_range_support_mask": np.ones((256, 256), dtype=bool),
            "himawari_bt": satellite,
            "himawari_band_valid_mask": np.isfinite(satellite),
            "himawari_all_bands_valid_mask": np.ones((256, 256), dtype=bool),
        }
        validate_arrays(arrays)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            npz = root / "sample.npz"
            metadata = root / "sample.json"
            atomic_write_npz(npz, arrays)
            atomic_write_json(metadata, {
                "radar_filename": "NHB250801000007.RAWLTHU",
                "time_match_status": "MATCHED",
                "absolute_delta_seconds": 10.0,
            })
            value = validate_sample_files(npz, metadata, "NHB250801000007.RAWLTHU")
            self.assertEqual(value["time_match_status"], "MATCHED")

    def test_parseable_current_utc(self) -> None:
        self.assertIsNotNone(datetime.now(timezone.utc).tzinfo)


if __name__ == "__main__":
    unittest.main()
