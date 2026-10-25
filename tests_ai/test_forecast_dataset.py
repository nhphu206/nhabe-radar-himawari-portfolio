"""Dataset contract tests run only in the dedicated PyTorch environment."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhabe_forecasting.dataset import BANDS, ForecastDataset
from nhabe_forecasting.normalization import file_sha256


class ForecastDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shape = (256, 256)
        radar_in = np.full(shape, 12, dtype=np.float32)
        radar_target = np.full(shape, 14, dtype=np.float32)
        radar_in[0, 0] = np.nan
        radar_target[0, 1] = np.nan
        self.write_radar("input.npz", radar_in)
        self.write_radar("target.npz", radar_target)
        satellite = np.stack([
            np.full(shape, 101 + 10 * index, dtype=np.float32) for index in range(5)
        ])
        satellite[2, 0, 2] = np.nan
        self.write_satellite(satellite)
        self.row = {
            "pair_id": "pair-1", "input_sample_id": "input", "target_sample_id": "target",
            "satellite_sample_id": "satellite", "input_time_utc": "2025-08-01T00:00:00Z",
            "target_time_utc": "2025-08-01T00:10:00Z", "lead_time_minutes": "10",
            "radar_signature": "A", "input_npz_path": "input.npz",
            "target_npz_path": "target.npz", "satellite_npz_path": "satellite.npz",
            "split": "train", "qc_status": "PASS", "timing_header_count": "10",
        }
        self.manifest = self.root / "pairs.csv"
        with self.manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.row)
            writer.writeheader(); writer.writerow(self.row)
        fields = {"radar_reflectivity_dbz": {"mean": 10, "std": 2}}
        fields.update({band: {"mean": 100 + 10 * index, "std": 1}
                       for index, band in enumerate(BANDS)})
        self.statistics = self.root / "normalization.json"
        self.statistics.write_text(json.dumps({
            "schema_version": "nhabe-train-normalization-v1",
            "status": "TRAIN_STATISTICS_FITTED", "band_order": list(BANDS), "fields": fields,
            "provenance": {"manifest_sha256": file_sha256(self.manifest)},
        }), encoding="utf-8")

    def write_radar(self, name: str, values: np.ndarray) -> None:
        np.savez_compressed(self.root / name, radar_reflectivity_dbz=values,
                            radar_data_valid_mask=np.isfinite(values))

    def write_satellite(self, values: np.ndarray) -> None:
        mask = np.isfinite(values)
        np.savez_compressed(self.root / "satellite.npz", himawari_bt=values,
                            himawari_band_valid_mask=mask,
                            himawari_all_bands_valid_mask=np.all(mask, axis=0))

    def dataset(self, mode: str = "multimodal") -> ForecastDataset:
        return ForecastDataset(self.manifest, self.statistics, self.root,
                               split="train", input_mode=mode)

    def test_multimodal_shapes_normalization_and_missing_padding(self) -> None:
        sample = self.dataset()[0]
        self.assertEqual(sample["inputs"].shape, (8, 256, 256))
        self.assertEqual(sample["target"].shape, (1, 256, 256))
        self.assertEqual(sample["input_radar_mask"].shape, (1, 256, 256))
        self.assertEqual(sample["target_mask"].shape, (1, 256, 256))
        self.assertEqual(sample["inputs"].dtype, torch.float32)
        self.assertEqual(sample["target_mask"].dtype, torch.bool)
        self.assertEqual(sample["input_radar_mask"].dtype, torch.bool)
        self.assertTrue(torch.isfinite(sample["inputs"]).all())
        self.assertTrue(torch.isfinite(sample["target"]).all())
        self.assertEqual(sample["inputs"][0, 1, 1].item(), 1.0)
        self.assertEqual(sample["inputs"][0, 0, 0].item(), 0.0)
        self.assertEqual(sample["inputs"][6, 0, 0].item(), 0.0)
        self.assertEqual(sample["inputs"][3, 0, 2].item(), 0.0)
        self.assertEqual(sample["inputs"][7, 0, 2].item(), 0.0)
        self.assertEqual(sample["target"][0, 1, 1].item(), 2.0)
        self.assertFalse(sample["target_mask"][0, 0, 1])

    def test_radar_mode_and_dataloader_batch(self) -> None:
        dataset = self.dataset("radar")
        batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
        self.assertEqual(batch["inputs"].shape, (1, 2, 256, 256))
        self.assertEqual(batch["target"].shape, (1, 1, 256, 256))
        self.assertEqual(dataset.channel_names, ("radar_reflectivity", "radar_valid_mask"))

    def test_manifest_change_is_rejected_by_normalization_provenance(self) -> None:
        with self.manifest.open("a", encoding="utf-8") as handle:
            handle.write("changed")
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.dataset()

    def test_bad_source_mask_is_rejected(self) -> None:
        with np.load(self.root / "input.npz") as arrays:
            values = arrays["radar_reflectivity_dbz"]
        np.savez_compressed(self.root / "input.npz", radar_reflectivity_dbz=values,
                            radar_data_valid_mask=np.ones((256, 256), dtype=bool))
        with self.assertRaisesRegex(ValueError, "mask"):
            self.dataset()[0]

    def test_wrong_lead_or_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "input_mode"):
            self.dataset("unknown")
        self.row["target_time_utc"] = "2025-08-01T00:11:00Z"
        with self.manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.row)
            writer.writeheader(); writer.writerow(self.row)
        document = json.loads(self.statistics.read_text())
        document["provenance"]["manifest_sha256"] = file_sha256(self.manifest)
        self.statistics.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "600-second"):
            self.dataset()


if __name__ == "__main__":
    unittest.main()
