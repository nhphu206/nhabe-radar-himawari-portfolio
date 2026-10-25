"""Exact forecast pairing tests use temporary placeholders, never real NPZ data."""

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhabe_forecasting.pairs import build_forecast_pairs


CONFIG = {
    "radar_signature": "A", "lead_time_minutes": 10,
    "time_column": "radar_representative_time_utc", "matching": "exact_timestamp",
    "allow_cross_day": False,
    "splits": {"train": ["20250801", "20250821"],
               "validation": ["20250822", "20250826"], "test": ["20250827", "20250831"]},
}


class ForecastPairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def row(self, time: str, signature: str = "A") -> dict[str, str]:
        dt = datetime.fromisoformat(time.replace("Z", "+00:00"))
        name = f"{dt.strftime('%Y%m%dT%H%M%S%f')}_{signature}"
        (self.root / f"{name}.npz").touch()
        return {
            "sample_id": name, "date": dt.strftime("%Y%m%d"),
            "radar_signature": signature, "radar_representative_time_utc": time,
            "npz_path": f"{name}.npz", "qc_status": "PASS",
            "leakage_group_id": dt.strftime("%Y%m%d_%H%M"),
            "himawari_local_line_time_utc": (dt + timedelta(seconds=150)).isoformat(),
        }

    def test_exact_ten_minutes_and_unique_target(self) -> None:
        rows = [self.row(f"2025-08-01T00:{minute}:00Z") for minute in ["00", "10", "20"]]
        pairs, summary = build_forecast_pairs(rows, self.root, CONFIG)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["target_sample_id"], rows[1]["sample_id"])
        self.assertNotEqual(pairs[0]["input_sample_id"], pairs[0]["target_sample_id"])
        self.assertEqual(summary["split_counts"], {"train": 2, "validation": 0, "test": 0})
        self.assertEqual(summary["later_satellite_inputs"], 2)

    def test_no_approximate_or_nominal_fallback(self) -> None:
        rows = [self.row("2025-08-01T00:00:00Z"), self.row("2025-08-01T00:09:59.500000Z")]
        self.assertEqual(build_forecast_pairs(rows, self.root, CONFIG)[0], [])

    def test_no_cross_day(self) -> None:
        rows = [self.row("2025-08-01T23:55:00Z"), self.row("2025-08-02T00:05:00Z")]
        self.assertEqual(build_forecast_pairs(rows, self.root, CONFIG)[0], [])

    def test_signature_b_is_not_a_target(self) -> None:
        rows = [self.row("2025-08-01T00:00:00Z"), self.row("2025-08-01T00:10:00Z", "B")]
        self.assertEqual(build_forecast_pairs(rows, self.root, CONFIG)[0], [])

    def test_duplicate_rows_rejected(self) -> None:
        row = self.row("2025-08-01T00:00:00Z")
        with self.assertRaises(ValueError):
            build_forecast_pairs([row, row], self.root, CONFIG)

    def test_missing_input_or_target_path_rejected(self) -> None:
        for missing in [0, 1]:
            rows = [self.row("2025-08-01T00:00:00Z"), self.row("2025-08-01T00:10:00Z")]
            (self.root / rows[missing]["npz_path"]).unlink()
            with self.assertRaises(FileNotFoundError):
                build_forecast_pairs(rows, self.root, CONFIG)

    def test_same_path_is_rejected(self) -> None:
        rows = [self.row("2025-08-01T00:00:00Z"), self.row("2025-08-01T00:10:00Z")]
        rows[1]["npz_path"] = rows[0]["npz_path"]
        with self.assertRaises(ValueError):
            build_forecast_pairs(rows, self.root, CONFIG)

    def test_naive_timestamp_is_rejected(self) -> None:
        row = self.row("2025-08-01T00:00:00Z")
        row["radar_representative_time_utc"] = "2025-08-01T00:00:00"
        with self.assertRaises(ValueError):
            build_forecast_pairs([row], self.root, CONFIG)

    def test_bad_date_or_qc_rejected(self) -> None:
        for field, value in [("date", "20250802"), ("qc_status", "FAIL")]:
            row = self.row("2025-08-01T00:00:00Z")
            row[field] = value
            with self.assertRaises(ValueError):
                build_forecast_pairs([row], self.root, CONFIG)

    def test_experiment_configuration_cannot_silently_change(self) -> None:
        config = copy.deepcopy(CONFIG)
        config["matching"] = "nearest"
        with self.assertRaises(ValueError):
            build_forecast_pairs([], self.root, config)
