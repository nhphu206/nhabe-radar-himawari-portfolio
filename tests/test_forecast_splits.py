"""Temporal partition checks, including deliberately leaking manifests."""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhabe_forecasting.splits import split_for_time, utc_time, validate_split_isolation
from test_forecast_pairs import CONFIG


class ForecastSplitTests(unittest.TestCase):
    def test_date_boundaries(self) -> None:
        for day, split in [(1, "train"), (21, "train"), (22, "validation"),
                           (26, "validation"), (27, "test"), (31, "test")]:
            time = utc_time(f"2025-08-{day:02d}T23:59:59Z")
            self.assertEqual(split_for_time(time, CONFIG["splits"]), split)

    def test_pair_crossing_split_is_rejected(self) -> None:
        pair = {"pair_id": "p", "split": "train", "input_sample_id": "i",
                "target_sample_id": "t", "input_time_utc": "2025-08-21T23:55:00Z",
                "target_time_utc": "2025-08-22T00:05:00Z"}
        with self.assertRaises(ValueError):
            validate_split_isolation([pair], CONFIG["splits"])

    def test_shared_sample_time_or_group_cannot_cross_splits(self) -> None:
        p = {"pair_id": "p", "split": "train", "input_sample_id": "i",
             "target_sample_id": "t", "input_time_utc": "2025-08-01T00:00:00Z",
             "target_time_utc": "2025-08-01T00:10:00Z", "input_leakage_group_id": "g"}
        q = {"pair_id": "q", "split": "validation", "input_sample_id": "j",
             "target_sample_id": "u", "input_time_utc": "2025-08-22T00:00:00Z",
             "target_time_utc": "2025-08-22T00:10:00Z"}
        for key, value in [("input_sample_id", "i"), ("input_leakage_group_id", "g")]:
            with self.assertRaises(ValueError):
                validate_split_isolation([p, {**q, key: value}], CONFIG["splits"])

    def test_duplicate_pair_rejected(self) -> None:
        pair = {"pair_id": "p", "split": "train", "input_sample_id": "i",
                "target_sample_id": "t", "input_time_utc": "2025-08-01T00:00:00Z",
                "target_time_utc": "2025-08-01T00:10:00Z"}
        with self.assertRaises(ValueError):
            validate_split_isolation([pair, pair], CONFIG["splits"])

    def test_same_sample_within_split_is_allowed(self) -> None:
        pair = {"pair_id": "p", "split": "train", "input_sample_id": "i",
                "target_sample_id": "t", "input_time_utc": "2025-08-01T00:00:00Z",
                "target_time_utc": "2025-08-01T00:10:00Z"}
        following = {**pair, "pair_id": "q", "input_sample_id": "t",
                     "target_sample_id": "u", "input_time_utc": pair["target_time_utc"],
                     "target_time_utc": "2025-08-01T00:20:00Z"}
        validate_split_isolation([pair, following], CONFIG["splits"])

    def test_same_edge_with_different_identifier_is_rejected(self) -> None:
        pair = {"pair_id": "p", "split": "train", "input_sample_id": "i",
                "target_sample_id": "t", "input_time_utc": "2025-08-01T00:00:00Z",
                "target_time_utc": "2025-08-01T00:10:00Z"}
        with self.assertRaises(ValueError):
            validate_split_isolation([pair, {**pair, "pair_id": "q"}], CONFIG["splits"])

    def test_overlapping_split_dates_are_rejected(self) -> None:
        splits = {"train": ["20250801", "20250822"], "validation": ["20250822", "20250826"]}
        with self.assertRaises(ValueError):
            split_for_time(utc_time("2025-08-22T00:00:00Z"), splits)
