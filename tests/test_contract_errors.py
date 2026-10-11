"""Boundary and invalid-input checks for configuration, timing and sample storage."""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhabe_preprocessing.config import load_config
from nhabe_preprocessing.io_utils import atomic_write_json, atomic_write_npz
from nhabe_preprocessing.radar import classify_signature
from nhabe_preprocessing.sample_pipeline import build_mock_arrays
from nhabe_preprocessing.time_matching import match_nearest_cycle, parse_utc
from nhabe_preprocessing.validation import validate_arrays, validate_sample_files


class ContractErrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "configs" / "pipeline.yaml")

    def test_explicit_config_path_preserves_settings(self) -> None:
        self.assertEqual(self.config["radar_processing"]["grid_z_m_above_grid_origin"], 1000.0)
        self.assertEqual(self.config["radar_processing"]["min_radius_m"], 500.0)
        self.assertEqual(self.config["grid"]["shape_yx"], [256, 256])

    def test_missing_config_group_is_rejected(self) -> None:
        invalid = {key: value for key, value in self.config.items() if key != "grid"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "incomplete.yaml"
            path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "grid"):
                load_config(path)

    def test_unknown_scan_geometry_is_not_forced_into_a_group(self) -> None:
        expected = self.config["radar"]["signatures"]["A"]
        metadata = {**expected, "gate_spacing_constant": True,
                    "fields": sorted(expected["fields"])}
        for changes in (
            {"fixed_angles_deg": [30.0] * 4},
            {"nsweeps": 8},
            {"fields": ["reflectivity"]},
            {"gate_spacing_constant": False},
        ):
            with self.subTest(changes=changes):
                self.assertEqual(classify_signature({**metadata, **changes}, self.config), "UNKNOWN")

    def test_time_tolerance_boundary_is_inclusive(self) -> None:
        instant = datetime(2025, 8, 1, tzinfo=timezone.utc)
        radar = {"date": "20250801", "radar_representative_time_utc": instant.isoformat()}
        for offset in (-180.0, 180.0, -180.001, 180.001):
            with self.subTest(offset=offset):
                satellite = {"cycle": "0000", "nominal_start_utc": instant.isoformat(),
                             "local_nhabe_line_time_utc":
                                 (instant + timedelta(seconds=offset)).isoformat()}
                result = match_nearest_cycle(radar, [satellite], 180.0)
                expected = "MATCHED" if abs(offset) <= 180 else "UNMATCHED_OUTSIDE_TOLERANCE"
                self.assertEqual(result["time_match_status"], expected)
                self.assertAlmostEqual(result["signed_delta_seconds"], offset)

    def test_no_satellite_cycle_returns_unmatched(self) -> None:
        result = match_nearest_cycle(
            {"radar_representative_time_utc": "2025-08-01T00:00:00Z"}, [], 180.0
        )
        self.assertEqual(result["time_match_status"], "UNMATCHED_NO_HIMAWARI_CYCLE")

    def test_naive_time_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_utc("2025-08-01T00:00:00")

    def test_array_shape_and_dtype_errors_are_rejected(self) -> None:
        arrays = build_mock_arrays(self.config)
        mutations = (
            ("radar_reflectivity_dbz", arrays["radar_reflectivity_dbz"][:1]),
            ("himawari_bt", arrays["himawari_bt"][:4]),
            ("radar_reflectivity_dbz", arrays["radar_reflectivity_dbz"].astype(np.float64)),
            ("himawari_band_valid_mask", arrays["himawari_band_valid_mask"].astype(np.uint8)),
        )
        for key, value in mutations:
            with self.subTest(key=key, shape=value.shape, dtype=str(value.dtype)):
                with self.assertRaises(ValueError):
                    validate_arrays({**arrays, key: value})

    def test_nan_mask_mismatch_is_rejected(self) -> None:
        arrays = build_mock_arrays(self.config)
        for key in ("radar_data_valid_mask", "himawari_band_valid_mask",
                    "himawari_all_bands_valid_mask"):
            with self.subTest(key=key):
                mask = arrays[key].copy()
                mask.flat[0] = not mask.flat[0]
                with self.assertRaises(ValueError):
                    validate_arrays({**arrays, key: mask})

    def test_missing_array_is_rejected(self) -> None:
        arrays = build_mock_arrays(self.config)
        arrays.pop("himawari_bt")
        with self.assertRaisesRegex(ValueError, "himawari_bt"):
            validate_arrays(arrays)

    def test_missing_json_pair_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atomic_write_npz(root / "sample.npz", build_mock_arrays(self.config))
            with self.assertRaises(FileNotFoundError):
                validate_sample_files(root / "sample.npz", root / "sample.json")

    def test_roundtrip_preserves_negative_zero_and_missing(self) -> None:
        arrays = build_mock_arrays(self.config)
        radar = arrays["radar_reflectivity_dbz"]
        radar[128, 128:130] = [-5.0, 0.0]
        arrays["radar_data_valid_mask"] = np.isfinite(radar)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atomic_write_npz(root / "sample.npz", arrays)
            atomic_write_json(root / "sample.json", {
                "radar_filename": "MOCK", "time_match_status": "MATCHED",
                "absolute_delta_seconds": 0.0,
            })
            validate_sample_files(root / "sample.npz", root / "sample.json", "MOCK")
            with np.load(root / "sample.npz", allow_pickle=False) as stored:
                for name, values in arrays.items():
                    self.assertTrue(np.array_equal(stored[name], values, equal_nan=True), name)
            self.assertEqual(list(root.glob("*.tmp")), [])

    def test_bad_sample_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atomic_write_npz(root / "sample.npz", build_mock_arrays(self.config))
            good = {"radar_filename": "MOCK", "time_match_status": "MATCHED",
                    "absolute_delta_seconds": 0.0}
            for changes in ({"radar_filename": "OTHER"},
                            {"time_match_status": "UNMATCHED"},
                            {"absolute_delta_seconds": 180.001}):
                with self.subTest(changes=changes):
                    atomic_write_json(root / "sample.json", {**good, **changes})
                    with self.assertRaises(ValueError):
                        validate_sample_files(root / "sample.npz", root / "sample.json", "MOCK")


if __name__ == "__main__":
    unittest.main()
