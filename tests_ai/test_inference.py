"""Single-sample inference contract tests with synthetic observation arrays."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhabe_forecasting.api import APISettings, create_app
from nhabe_forecasting.inference import run_multimodal_inference, validate_prediction_files
from nhabe_forecasting.model import MODEL_CONFIG, SmallUNet2D
from nhabe_forecasting.normalization import BANDS, file_sha256
from nhabe_forecasting.training import save_model_checkpoint


class InferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shape = (256, 256)
        radar = np.full(shape, 12, dtype=np.float32)
        radar[0, 0] = np.nan
        self._write_radar(self.root / "input.npz", radar)
        satellite = np.stack([
            np.full(shape, 101 + 10 * index, dtype=np.float32) for index in range(5)
        ])
        satellite[2, 0, 2] = np.nan
        band_mask = np.isfinite(satellite)
        np.savez_compressed(
            self.root / "satellite.npz", himawari_bt=satellite,
            himawari_band_valid_mask=band_mask,
            himawari_all_bands_valid_mask=np.all(band_mask, axis=0),
        )
        self.row = {
            "pair_id": "pair-1", "input_sample_id": "input", "target_sample_id": "target",
            "satellite_sample_id": "satellite", "input_time_utc": "2025-08-22T00:00:00Z",
            "target_time_utc": "2025-08-22T00:10:00Z", "lead_time_minutes": "10",
            "radar_signature": "A", "input_npz_path": "input.npz",
            "target_npz_path": "future-target-does-not-exist.npz",
            "satellite_npz_path": "satellite.npz", "split": "validation",
            "qc_status": "PASS", "timing_header_count": "10",
        }
        self.manifest = self.root / "pairs.csv"
        with self.manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.row)
            writer.writeheader(); writer.writerow(self.row)
        fields = {"radar_reflectivity_dbz": {"mean": 10, "std": 2}}
        fields.update({band: {"mean": 100 + 10 * index, "std": 1}
                       for index, band in enumerate(BANDS)})
        self.normalization = self.root / "normalization.json"
        self.normalization.write_text(json.dumps({
            "schema_version": "nhabe-train-normalization-v1",
            "status": "TRAIN_STATISTICS_FITTED", "band_order": list(BANDS), "fields": fields,
            "provenance": {"manifest_sha256": file_sha256(self.manifest)},
        }), encoding="utf-8")
        self.model_config = ROOT / "configs/model.yaml"
        self.checkpoint = self.root / "model.pt"
        self._write_checkpoint("multimodal")

    @staticmethod
    def _write_radar(path: Path, values: np.ndarray) -> None:
        np.savez_compressed(
            path, radar_reflectivity_dbz=values,
            radar_data_valid_mask=np.isfinite(values),
        )

    def _write_checkpoint(self, mode: str) -> None:
        channels = MODEL_CONFIG["input_channels"][mode]
        model = SmallUNet2D(channels, MODEL_CONFIG["base_channels"], 1)
        save_model_checkpoint(
            self.checkpoint, model, mode, 1, {"rmse_dbz_common_mask": 1.0}, {
                "manifest_sha256": file_sha256(self.manifest),
                "normalization_sha256": file_sha256(self.normalization),
                "model_config_sha256": file_sha256(self.model_config),
            },
        )

    def test_prediction_roundtrip_preserves_nan_mask_and_provenance(self) -> None:
        output = self.root / "prediction"
        result = run_multimodal_inference(
            data_root=self.root, manifest_path=self.manifest,
            normalization_path=self.normalization, model_config_path=self.model_config,
            checkpoint_path=self.checkpoint, pair_id="pair-1", output_dir=output,
            requested_device="cpu",
        )
        self.assertFalse(result["future_target_array_read"])
        self.assertEqual(result["valid_pixels"], 256 * 256 - 1)
        metadata = validate_prediction_files(
            Path(result["npz_path"]), Path(result["json_path"]), "pair-1",
        )
        self.assertEqual(metadata["valid_mask_rule"], "input_radar_valid_mask")
        with np.load(result["npz_path"], allow_pickle=False) as arrays:
            self.assertTrue(np.isnan(arrays["predicted_reflectivity_dbz"][0, 0]))
            self.assertFalse(arrays["prediction_valid_mask"][0, 0])

    def test_radar_checkpoint_is_rejected_for_multimodal_inference(self) -> None:
        self._write_checkpoint("radar")
        with self.assertRaisesRegex(ValueError, "multimodal checkpoint"):
            run_multimodal_inference(
                data_root=self.root, manifest_path=self.manifest,
                normalization_path=self.normalization, model_config_path=self.model_config,
                checkpoint_path=self.checkpoint, pair_id="pair-1",
                output_dir=self.root / "rejected", requested_device="cpu",
            )

    def test_cli_help_works_outside_repository(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/18_predict_one_sample.py"), "--help"],
            cwd=self.root, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--checkpoint", result.stdout)

    def test_synthetic_dashboard_fixture_has_explicit_target(self) -> None:
        fixture = self.root / "interactive-fixture"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/20_create_synthetic_api_fixture.py"),
                "--output", str(fixture), "--include-historical-target",
            ],
            cwd=self.root, capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((fixture / "fixture_summary.json").read_text())
        self.assertEqual(summary["sample_kind"], "SYNTHETIC_INTERACTIVE_DEMO_FIXTURE")
        self.assertTrue(summary["future_target_file_exists"])
        self.assertFalse(summary["meteorological_observation"])
        self.assertFalse(summary["trained_checkpoint"])
        with np.load(fixture / "future_target.npz", allow_pickle=False) as arrays:
            self.assertEqual(arrays["radar_reflectivity_dbz"].shape, (256, 256))
            self.assertEqual(arrays["radar_reflectivity_dbz"].dtype, np.float32)
            self.assertEqual(arrays["radar_data_valid_mask"].dtype, np.bool_)
            self.assertTrue(np.array_equal(
                np.isfinite(arrays["radar_reflectivity_dbz"]),
                arrays["radar_data_valid_mask"],
            ))

    def test_unsafe_pair_identifier_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsafe"):
            run_multimodal_inference(
                data_root=self.root, manifest_path=self.manifest,
                normalization_path=self.normalization, model_config_path=self.model_config,
                checkpoint_path=self.checkpoint, pair_id="../pair-1",
                output_dir=self.root / "unsafe", requested_device="cpu",
            )

    def test_api_health_model_info_prediction_and_errors(self) -> None:
        settings = APISettings(
            data_root=self.root, manifest_path=self.manifest,
            normalization_path=self.normalization, model_config_path=self.model_config,
            checkpoint_path=self.checkpoint, output_root=self.root / "api-output", device="cpu",
        )
        with TestClient(create_app(settings)) as client:
            demo = client.get("/demo")
            self.assertEqual(demo.status_code, 200)
            self.assertIn("Nha Be Radar–Himawari Forecast Demo", demo.text)
            self.assertNotIn(str(self.root), demo.text)
            root = client.get("/")
            self.assertEqual(root.status_code, 200)
            self.assertIn("Run forecast", root.text)

            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["status"], "READY")
            info = client.get("/model-info")
            self.assertEqual(info.status_code, 200)
            self.assertEqual(info.json()["available_pairs"], 1)
            self.assertTrue(info.json()["trained_checkpoint"])
            self.assertEqual(info.json()["artifact_kind"], "AUDITED_RESEARCH_ARTIFACT")
            self.assertNotIn(str(self.root), info.text)

            pairs = client.get("/pairs?split=validation&limit=50")
            self.assertEqual(pairs.status_code, 200)
            self.assertEqual(pairs.json()["count"], 1)
            self.assertEqual(pairs.json()["pairs"][0]["pair_id"], "pair-1")
            self.assertNotIn("npz_path", pairs.text)
            current = client.get("/visualization/pair-1/current.png")
            self._assert_png(current)
            self.assertEqual(current.headers["x-nhabe-historical-target-read"], "false")
            missing_prediction = client.get("/visualization/pair-1/prediction.png")
            self.assertEqual(missing_prediction.status_code, 404)

            prediction = client.post("/predict", json={"pair_id": "pair-1"})
            self.assertEqual(prediction.status_code, 200, prediction.text)
            body = prediction.json()
            self.assertEqual(body["artifact_id"], "pair-1")
            self.assertFalse(body["future_target_array_read"])
            self.assertFalse(body["npz_artifact"].startswith("/"))
            self.assertTrue((settings.output_root / body["npz_artifact"]).is_file())

            existing = client.get("/predictions/pair-1")
            self.assertEqual(existing.status_code, 200)
            self.assertFalse(existing.json()["future_target_array_read"])
            self.assertNotIn(str(self.root), existing.text)
            predicted_image = client.get("/visualization/pair-1/prediction.png")
            self._assert_png(predicted_image)
            self.assertEqual(
                predicted_image.headers["x-nhabe-historical-target-read"], "false",
            )

            missing_target = client.get("/visualization/pair-1/target.png")
            self.assertEqual(missing_target.status_code, 404)
            target = np.full((256, 256), 15, dtype=np.float32)
            target[0, 0] = np.nan
            self._write_radar(self.root / "future-target-does-not-exist.npz", target)
            for layer in ("target", "error"):
                image = client.get(f"/visualization/pair-1/{layer}.png")
                self._assert_png(image)
                self.assertEqual(image.headers["x-nhabe-historical-target-read"], "true")

            duplicate = client.post("/predict", json={"pair_id": "pair-1"})
            self.assertEqual(duplicate.status_code, 409)
            unknown = client.post("/predict", json={"pair_id": "missing"})
            self.assertEqual(unknown.status_code, 404)
            extra = client.post("/predict", json={"pair_id": "pair-1", "path": "/tmp"})
            self.assertEqual(extra.status_code, 422)
            unknown_image = client.get("/visualization/missing/current.png")
            self.assertEqual(unknown_image.status_code, 404)
            invalid_layer = client.get("/visualization/pair-1/not-a-layer.png")
            self.assertEqual(invalid_layer.status_code, 422)

    def _assert_png(self, response) -> None:
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertTrue(response.content.startswith(b"\x89PNG\r\n\x1a\n"))
        width, height = struct.unpack(">II", response.content[16:24])
        self.assertEqual((width, height), (256, 256))


if __name__ == "__main__":
    unittest.main()
