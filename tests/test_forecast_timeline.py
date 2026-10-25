"""Source-timing tests use generated HSD headers and temporary data only."""

import bz2
import copy
import csv
from datetime import datetime, timedelta, timezone
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

from nhabe_forecasting.satellite_timing import parse_header_timing, read_header_timing
from nhabe_forecasting.splits import validate_split_isolation
from nhabe_forecasting.timeline import build_audited_pairs


def header_bytes(band: int = 13, segment: int = 4, cycle: str = "0000") -> bytes:
    """Construct small valid blocks using SatPy dtypes, not real observations."""
    from satpy.readers import ahi_hsd as hsd

    nominal = datetime.strptime("20250801" + cycle, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    epoch = datetime(1858, 11, 17, tzinfo=timezone.utc)
    def mjd(seconds: float) -> float:
        return ((nominal + timedelta(seconds=seconds)) - epoch).total_seconds() / 86400

    blocks = []
    for number in range(1, 10):
        dtype = {1: hsd._BASIC_INFO_TYPE, 5: hsd._CAL_INFO_TYPE,
                 7: hsd._SEGMENT_INFO_TYPE, 9: hsd._OBSERVATION_TIME_INFO_TYPE}.get(number)
        if dtype is None:
            blocks.append(bytes([number, 3, 0]))
            continue
        block = np.zeros(1, dtype=dtype)
        block["hblock_number"] = number
        block["blocklength"] = dtype.itemsize
        extra = b""
        if number == 1:
            block["satellite"] = b"Himawari-9"
            block["observation_area"] = b"FLDK"
            block["observation_timeline"] = int(cycle)
            block["observation_start_time"] = mjd(60)
            block["observation_end_time"] = mjd(180)
            block["file_creation_time"] = mjd(480)
        elif number == 5:
            block["band_number"] = band
        elif number == 7:
            block["total_number_of_segments"] = 10
            block["segment_sequence_number"] = segment
        elif number == 9:
            records = np.zeros(2, dtype=hsd._OBSERVATION_LINE_TIME_INFO_TYPE)
            records["line_number"] = [1651, 2200]
            records["observation_time"] = [mjd(60), mjd(180)]
            block["number_of_observation_times"] = 2
            extra = records.tobytes()
            block["blocklength"] = dtype.itemsize + len(extra)
        blocks.append(block.tobytes() + extra)
    return b"".join(blocks)


class HeaderTimingTests(unittest.TestCase):
    def test_valid_header_and_fingerprint(self) -> None:
        raw = header_bytes()
        value = parse_header_timing(raw)
        self.assertEqual((value["band"], value["segment"]), ("B13", "04"))
        self.assertEqual(value["header_through_block9_sha256"], hashlib.sha256(raw).hexdigest())

    def test_truncated_and_misnumbered_header_rejected(self) -> None:
        raw = header_bytes()
        for bad in [raw[:2], raw[:-1], bytes([2]) + raw[1:]]:
            with self.assertRaises(ValueError):
                parse_header_timing(bad)

    def test_nonfinite_and_reversed_times_rejected(self) -> None:
        from satpy.readers import ahi_hsd as hsd
        for value in [float("nan"), 0.0]:
            raw = bytearray(header_bytes())
            first = np.frombuffer(raw, dtype=hsd._BASIC_INFO_TYPE, count=1)
            first["file_creation_time"] = value
            with self.assertRaises(ValueError):
                parse_header_timing(bytes(raw))

    def test_big_endian_not_silently_accepted(self) -> None:
        from satpy.readers import ahi_hsd as hsd
        raw = bytearray(header_bytes())
        np.frombuffer(raw, dtype=hsd._BASIC_INFO_TYPE, count=1)["byte_order"] = 1
        with self.assertRaises(ValueError):
            parse_header_timing(bytes(raw))

    def test_compressed_roundtrip_does_not_modify_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mock.DAT.bz2"
            source = bz2.compress(header_bytes())
            path.write_bytes(source)
            self.assertEqual(read_header_timing(path)["band"], "B13")
            self.assertEqual(path.read_bytes(), source)


class ForecastTimelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = yaml.safe_load((ROOT / "configs/forecasting.yaml").read_text())
        self.pipeline = {
            "paths": {"himawari_day01_root": "data/raw/himawari/20250801",
                      "himawari_staging_root": "data/staging/himawari_month"},
            "grid": {"grid_id": "MOCK_GRID"},
            "himawari": {"bands": ["B08", "B13", "B14", "B15", "B16"], "segments": ["04", "05"]},
        }
        self.rows = []
        for minute in [1, 11, 21]:
            time = datetime(2025, 8, 1, 0, minute, 30, tzinfo=timezone.utc)
            nominal = time.replace(minute=minute - 1, second=0)
            sid = time.strftime("mock_%H%M%S")
            row = {"sample_id": sid, "date": "20250801", "radar_filename": sid + ".RAW",
                   "radar_signature": "A", "radar_representative_time_utc": time.isoformat(),
                   "himawari_cycle": nominal.strftime("%H%M"), "leakage_group_id": sid,
                   "himawari_local_line_time_utc": (nominal + timedelta(seconds=150)).isoformat(),
                   "npz_path": sid + ".npz", "json_path": sid + ".json", "qc_status": "PASS"}
            (self.root / row["npz_path"]).touch()
            metadata = {"radar_filename": row["radar_filename"], "radar_signature": "A",
                        "grid_id": "MOCK_GRID", "band_order": self.pipeline["himawari"]["bands"],
                        "radar_representative_time_utc": time.isoformat(),
                        "radar_observation_start_utc": (time - timedelta(seconds=60)).isoformat(),
                        "radar_observation_end_utc": (time + timedelta(seconds=60)).isoformat(),
                        "himawari_nominal_time_utc": nominal.isoformat(),
                        "himawari_local_nhabe_line_time_utc": row["himawari_local_line_time_utc"]}
            (self.root / row["json_path"]).write_text(json.dumps(metadata))
            self.rows.append(row)

    def fake_header(self, path: Path) -> dict:
        tokens = path.name.split("_")
        value = parse_header_timing(header_bytes(int(tokens[4][1:]), int(tokens[7][1:3]), tokens[3]))
        return {**value, "file_size_bytes": 1234}

    def build(self) -> tuple:
        return build_audited_pairs(self.rows, self.root, self.config, self.pipeline)

    def test_previous_cycle_ten_headers_and_separate_paths(self) -> None:
        with patch("nhabe_forecasting.timeline.read_header_timing", side_effect=self.fake_header) as read:
            pairs, summary, headers, excluded = self.build()
        self.assertEqual(len(pairs), 1)
        self.assertEqual(read.call_count, 10)
        self.assertEqual(len(headers), 10)
        self.assertEqual(pairs[0]["satellite_sample_id"], self.rows[0]["sample_id"])
        self.assertNotEqual(pairs[0]["satellite_npz_path"], pairs[0]["input_npz_path"])
        self.assertEqual(pairs[0]["target_lead_from_cutoff_seconds"], 540)
        self.assertEqual(summary["timeline_exclusions"], {"satellite_cross_day": 1})
        self.assertFalse(summary["operational_availability_verified"])

    def test_missing_header_fails_closed(self) -> None:
        with patch("nhabe_forecasting.timeline.read_header_timing", side_effect=FileNotFoundError("mock missing")):
            pairs, summary, *_ = self.build()
        self.assertEqual(pairs, [])
        self.assertEqual(summary["timeline_exclusions"]["missing_or_invalid_source_header"], 1)

    def test_observation_or_creation_after_cutoff_rejected(self) -> None:
        for field, reason in [("observation_end_utc", "satellite_observation_after_cutoff"),
                              ("file_creation_utc", "satellite_creation_after_cutoff")]:
            def late(path: Path) -> dict:
                value = self.fake_header(path)
                value[field] = "2025-08-01T00:20:00Z"
                if field == "observation_end_utc":
                    value["file_creation_utc"] = "2025-08-01T00:21:00Z"
                return value
            with patch("nhabe_forecasting.timeline.read_header_timing", side_effect=late):
                pairs, summary, *_ = self.build()
            self.assertEqual(pairs, [])
            self.assertEqual(summary["timeline_exclusions"][reason], 1)

    def test_wrong_band_or_segment_rejected(self) -> None:
        for field, bad in [("band", "B01"), ("segment", "09")]:
            with patch("nhabe_forecasting.timeline.read_header_timing",
                       side_effect=lambda p: {**self.fake_header(p), field: bad}):
                pairs, summary, *_ = self.build()
            self.assertEqual(pairs, [])
            self.assertIn("missing_or_invalid_source_header", summary["timeline_exclusions"])

    def test_no_fallback_when_previous_cycle_absent(self) -> None:
        with patch("nhabe_forecasting.timeline.read_header_timing") as read:
            pairs, summary, *_ = build_audited_pairs(self.rows[1:], self.root, self.config, self.pipeline)
        read.assert_not_called()
        self.assertEqual(pairs, [])
        self.assertIn("previous_satellite_cycle_unavailable_in_index", summary["timeline_exclusions"])

    def test_json_identity_mismatch_rejected(self) -> None:
        path = self.root / self.rows[0]["json_path"]
        m = json.loads(path.read_text()); m["grid_id"] = "WRONG"
        path.write_text(json.dumps(m))
        with self.assertRaises(ValueError):
            self.build()

    def test_configuration_change_rejected(self) -> None:
        config = copy.deepcopy(self.config)
        config["satellite_nominal_lag_minutes"] = 20
        with self.assertRaises(ValueError):
            build_audited_pairs(self.rows, self.root, config, self.pipeline)

    def test_satellite_cross_split_detected(self) -> None:
        with patch("nhabe_forecasting.timeline.read_header_timing", side_effect=self.fake_header):
            pairs, *_ = self.build()
        pairs[0]["satellite_time_utc"] = "2025-08-22T00:00:00Z"
        with self.assertRaises(ValueError):
            validate_split_isolation(pairs, self.config["splits"])

    def test_cli_roundtrip_from_other_directory_and_no_overwrite(self) -> None:
        folder = self.root / "data/raw/himawari/20250801/0000"
        folder.mkdir(parents=True)
        for band in self.pipeline["himawari"]["bands"]:
            for segment in ["04", "05"]:
                name = f"HS_H09_20250801_0000_{band}_FLDK_R20_S{segment}10.DAT.bz2"
                (folder / name).write_bytes(bz2.compress(header_bytes(int(band[1:]), int(segment))))
        index = self.root / "index.csv"
        with index.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.rows[0].keys())
            writer.writeheader(); writer.writerows(self.rows)
        pipeline = {**self.pipeline, **{k: {} for k in ["radar", "radar_processing", "time_alignment",
                                                       "sample_contract", "expected_results"]}}
        config_path = self.root / "pipeline.yaml"
        config_path.write_text(yaml.safe_dump(pipeline))
        output = self.root / "result"
        args = [sys.executable, str(ROOT / "scripts/09_build_forecast_pairs.py"),
                "--data-root", str(self.root), "--index", str(index),
                "--pipeline-config", str(config_path), "--output", str(output)]
        result = subprocess.run(args, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with (output / "forecast_pairs.csv").open(newline="") as f:
            self.assertEqual(len(list(csv.DictReader(f))), 1)
        self.assertEqual(json.loads((output / "pair_summary.json").read_text())["headers_recorded"], 10)
        again = subprocess.run(args, cwd=self.root, capture_output=True, text=True)
        self.assertNotEqual(again.returncode, 0)
        self.assertIn("existing artifacts are protected", again.stderr)
