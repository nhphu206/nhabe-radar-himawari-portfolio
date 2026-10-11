"""Exercise the real CLI in subprocesses, without any observation files."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhabe_preprocessing.validation import validate_sample_files


class CLITests(unittest.TestCase):
    def run_cli(self, cwd: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        """Run the repository wrapper from a temporary unrelated directory."""
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "nhabe_pipeline.py"), *arguments],
            cwd=cwd, capture_output=True, text=True, timeout=60, check=False,
        )

    def test_help_from_another_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_cli(Path(temporary), "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("sample", result.stdout)
        self.assertIn("step", result.stdout)

    def test_unknown_command_returns_error_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_cli(root, "not-a-command")
            self.assertEqual(result.returncode, 2)
            self.assertIn("invalid choice", result.stderr)
            self.assertEqual(list(root.iterdir()), [])

    def test_step_forwards_help_without_reading_raw(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_cli(Path(temporary), "step", "inventory", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--limit", result.stdout)
        self.assertIn("--date", result.stdout)

    def test_sample_from_another_directory_roundtrips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "generated"
            result = self.run_cli(root, "sample", "--output", str(destination))
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)
            self.assertEqual(summary["status"], "PASS")
            npz = Path(summary["npz_path"])
            metadata_path = Path(summary["json_path"])
            # macOS may spell the same temporary directory as /var or /private/var.
            self.assertTrue(npz.parent.samefile(destination))
            self.assertTrue(metadata_path.parent.samefile(destination))
            metadata = validate_sample_files(npz, metadata_path, "MOCK_NOT_RAW_DATA")
            self.assertEqual(metadata["sample_kind"], "SYNTHETIC_CONTRACT_SMOKE_TEST")
            self.assertEqual(metadata["band_order"], ["B08", "B13", "B14", "B15", "B16"])
            with np.load(npz, allow_pickle=False) as arrays:
                self.assertEqual(arrays["radar_reflectivity_dbz"].shape, (256, 256))
                self.assertEqual(arrays["himawari_bt"].shape, (5, 256, 256))
                self.assertTrue(np.isnan(arrays["radar_reflectivity_dbz"]).any())
                self.assertTrue(np.isnan(arrays["himawari_bt"]).any())
            self.assertEqual(list(destination.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
