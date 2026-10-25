"""Train-only normalization tests never read research data or future tensors."""

import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhabe_forecasting.normalization import (
    BANDS, POLICY, RunningMoments, file_sha256, fit_train_statistics, unique_sources,
)

SPLITS = yaml.safe_load((ROOT / "configs/forecasting.yaml").read_text())["splits"]


class MomentTests(unittest.TestCase):
    def test_chunked_matches_numpy_population_statistics(self) -> None:
        values = np.array([-5, 0, 5, 10, 15], dtype=np.float64)
        moment = RunningMoments()
        for chunk in [values[:1], values[1:3], values[3:]]:
            moment.update(chunk)
        result = moment.result("dBZ")
        self.assertAlmostEqual(result["mean"], float(values.mean()))
        self.assertAlmostEqual(result["std"], float(values.std(ddof=0)))
        self.assertEqual(result["count"], 5)
        self.assertEqual(result["min"], -5)

    def test_large_offset_is_numerically_stable(self) -> None:
        values = 1e9 + np.arange(100, dtype=np.float64)
        moment = RunningMoments()
        for chunk in np.array_split(values, 10):
            moment.update(chunk)
        self.assertAlmostEqual(moment.result("K")["std"], float(values.std()), places=8)

    def test_empty_constant_and_nonfinite_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RunningMoments().result("K")
        moment = RunningMoments()
        moment.update(np.array([2, 2]))
        with self.assertRaises(ValueError):
            moment.result("K")
        for bad in [np.nan, np.inf]:
            with self.assertRaises(ValueError):
                RunningMoments().update(np.array([bad]))

    def test_pixel_weighted_not_mean_of_images(self) -> None:
        moment = RunningMoments()
        moment.update(np.array([0.0]))
        moment.update(np.array([10.0, 10.0, 10.0]))
        self.assertEqual(moment.result("dBZ")["mean"], 7.5)


class NormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.radar = np.full((256, 256), np.nan, dtype=np.float32)
        self.radar[0, :3] = [-5, 0, 5]
        self.satellite = np.full((5, 256, 256), np.nan, dtype=np.float32)
        for i in range(5):
            self.satellite[i, 0, :3] = [249 + 10 * i, 250 + 10 * i, 251 + 10 * i]
        np.savez_compressed(self.root / "radar.npz", radar_reflectivity_dbz=self.radar,
                            radar_data_valid_mask=np.isfinite(self.radar))
        self.write_satellite()
        for name in ["radar.json", "satellite.json"]:
            (self.root / name).write_text("{}")
        self.row = {
            "pair_id": "p", "input_sample_id": "r", "target_sample_id": "future",
            "satellite_sample_id": "s", "input_time_utc": "2025-08-01T00:11:30Z",
            "target_time_utc": "2025-08-01T00:21:30Z", "satellite_time_utc": "2025-08-01T00:04:05Z",
            "input_npz_path": "radar.npz", "satellite_npz_path": "satellite.npz",
            "target_npz_path": "TARGET_MUST_NOT_BE_OPENED.npz", "input_json_path": "radar.json",
            "satellite_json_path": "satellite.json", "input_json_sha256": file_sha256(self.root / "radar.json"),
            "satellite_json_sha256": file_sha256(self.root / "satellite.json"),
            "split": "train", "radar_signature": "A", "qc_status": "PASS", "timing_header_count": "10",
        }

    def write_satellite(self) -> None:
        mask = np.isfinite(self.satellite)
        np.savez_compressed(self.root / "satellite.npz", himawari_bt=self.satellite,
                            himawari_band_valid_mask=mask,
                            himawari_all_bands_valid_mask=np.all(mask, axis=0))

    def fit(self, rows: list[dict] | None = None) -> tuple:
        return fit_train_statistics(rows or [self.row], self.root, SPLITS, POLICY)

    def test_known_statistics_keep_negative_zero_and_nan(self) -> None:
        before = {name: file_sha256(self.root / name) for name in ["radar.npz", "satellite.npz"]}
        result, sources = self.fit()
        self.assertEqual(result["fields"]["radar_reflectivity_dbz"]["count"], 3)
        self.assertEqual(result["fields"]["radar_reflectivity_dbz"]["mean"], 0)
        self.assertAlmostEqual(result["fields"]["radar_reflectivity_dbz"]["std"], np.std([-5, 0, 5]))
        self.assertEqual(result["source_counts"], {"input": 1, "satellite": 1})
        for i, band in enumerate(BANDS):
            self.assertEqual(result["fields"][band]["mean"], 250 + 10 * i)
        self.assertEqual(len(sources), 2)
        self.assertEqual(before, {name: file_sha256(self.root / name) for name in before})

    def test_validation_test_and_target_arrays_are_not_opened(self) -> None:
        rows = [self.row]
        for split, day in [("validation", "22"), ("test", "27")]:
            row = {**self.row, "pair_id": split, "split": split}
            for role in ["input", "target", "satellite"]:
                row[f"{role}_sample_id"] = role + split
                row[f"{role}_time_utc"] = self.row[f"{role}_time_utc"].replace("08-01", "08-" + day)
                row[f"{role}_npz_path"] = f"DO_NOT_READ_{role}_{split}.npz"
                if role != "target":
                    row[f"{role}_json_path"] = f"DO_NOT_READ_{role}_{split}.json"
            rows.append(row)
        original = np.load
        opened = []
        def read(path: Path, **kwargs):
            opened.append(Path(path).name)
            return original(path, **kwargs)
        with patch("nhabe_forecasting.normalization.np.load", side_effect=read):
            result, _ = self.fit(rows)
        self.assertEqual(opened, ["radar.npz", "satellite.npz"])
        self.assertEqual(result["train_pairs"], 1)

    def test_per_band_mask_not_all_bands_intersection(self) -> None:
        self.satellite[2, 0, 0] = np.nan
        self.write_satellite()
        result, _ = self.fit()
        self.assertEqual(result["fields"]["B08"]["count"], 3)
        self.assertEqual(result["fields"]["B14"]["count"], 2)
        self.assertEqual(result["fields"]["B14"]["mean"], 270.5)

    def test_unique_source_dedup_and_alias_rejection(self) -> None:
        self.assertEqual(len(unique_sources([self.row, self.row], "input", self.root)), 1)
        with self.assertRaises(ValueError):
            unique_sources([self.row, {**self.row, "input_sample_id": "alias"}], "input", self.root)
        with self.assertRaises(ValueError):
            unique_sources([self.row, {**self.row, "input_npz_path": "other.npz"}], "input", self.root)

    def test_invalid_mask_dtype_shape_or_infinity_rejected(self) -> None:
        bads = [
            {"radar_reflectivity_dbz": self.radar.astype(np.float64), "radar_data_valid_mask": np.isfinite(self.radar)},
            {"radar_reflectivity_dbz": self.radar, "radar_data_valid_mask": np.zeros((256, 256), dtype=bool)},
            {"radar_reflectivity_dbz": self.radar[:2], "radar_data_valid_mask": np.isfinite(self.radar[:2])},
        ]
        inf = self.radar.copy(); inf[1, 1] = np.inf
        bads.append({"radar_reflectivity_dbz": inf, "radar_data_valid_mask": np.isfinite(inf)})
        for bad in bads:
            np.savez_compressed(self.root / "radar.npz", **bad)
            with self.assertRaises(ValueError):
                self.fit()

    def test_bad_all_band_mask_rejected(self) -> None:
        np.savez_compressed(self.root / "satellite.npz", himawari_bt=self.satellite,
                            himawari_band_valid_mask=np.isfinite(self.satellite),
                            himawari_all_bands_valid_mask=np.ones((256, 256), dtype=bool))
        with self.assertRaises(ValueError):
            self.fit()

    def test_empty_or_constant_training_field_rejected(self) -> None:
        for values in [np.full((256, 256), np.nan, dtype=np.float32), np.ones((256, 256), dtype=np.float32)]:
            np.savez_compressed(self.root / "radar.npz", radar_reflectivity_dbz=values,
                                radar_data_valid_mask=np.isfinite(values))
            with self.assertRaises(ValueError):
                self.fit()

    def test_changed_metadata_rejected(self) -> None:
        (self.root / "radar.json").write_text('{"changed": true}')
        with self.assertRaises(ValueError):
            self.fit()

    def test_leakage_rejected_before_arrays_are_read(self) -> None:
        leaking = {**self.row, "pair_id": "other", "split": "validation",
                   "input_time_utc": "2025-08-22T00:11:30Z", "target_time_utc": "2025-08-22T00:21:30Z",
                   "satellite_time_utc": "2025-08-22T00:04:05Z"}
        with patch("nhabe_forecasting.normalization.np.load") as read:
            with self.assertRaises(ValueError):
                self.fit([self.row, leaking])
        read.assert_not_called()

    def test_policy_change_or_no_train_rejected(self) -> None:
        bad = copy.deepcopy(POLICY); bad["include_target_only_samples"] = True
        with self.assertRaises(ValueError):
            fit_train_statistics([self.row], self.root, SPLITS, bad)
        with self.assertRaises(ValueError):
            fit_train_statistics([], self.root, SPLITS, POLICY)

    def test_cli_roundtrip_and_no_overwrite(self) -> None:
        folder = self.root / "audited"
        folder.mkdir()
        manifest = folder / "forecast_pairs.csv"
        with manifest.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.row.keys()); writer.writeheader(); writer.writerow(self.row)
        summary = {"schema_version": "nhabe-forecasting-observation-timing-v1",
                   "status": "OBSERVATION_TIMING_AUDITED", "scope": "full_exact_candidate_set", "pairs": 1,
                   "artifact_sha256": {"forecast_pairs.csv": file_sha256(manifest)},
                   "provenance": {"config_sha256": file_sha256(ROOT / "configs/forecasting.yaml")}}
        (folder / "pair_summary.json").write_text(json.dumps(summary))
        output = self.root / "stats"
        args = [sys.executable, str(ROOT / "scripts/10_compute_train_statistics.py"),
                "--data-root", str(self.root), "--manifest-dir", str(folder), "--output", str(output)]
        result = subprocess.run(args, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        stats = json.loads((output / "normalization.json").read_text())
        self.assertEqual(stats["fields"]["radar_reflectivity_dbz"]["mean"], 0)
        self.assertEqual(stats["provenance"]["source_manifest_sha256"], file_sha256(output / "normalization_sources.csv"))
        again = subprocess.run(args, cwd=self.root, capture_output=True, text=True)
        self.assertNotEqual(again.returncode, 0)
        self.assertIn("existing artifacts are protected", again.stderr)
