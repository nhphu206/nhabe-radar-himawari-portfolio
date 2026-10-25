"""Audited single-sample inference for the frozen multimodal Small U-Net."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import torch
import yaml

from nhabe_preprocessing.io_utils import atomic_write_json, atomic_write_npz

from .dataset import ForecastDataset
from .model import MODEL_CONFIG, SmallUNet2D
from .normalization import file_sha256
from .training import choose_device, load_model_checkpoint


INFERENCE_SCHEMA = "nhabe-multimodal-inference-v1"
PREDICTION_ARRAY = "predicted_reflectivity_dbz"
PREDICTION_MASK = "prediction_valid_mask"


def _load_model(
    checkpoint_path: Path,
    manifest_path: Path,
    normalization_path: Path,
    model_config_path: Path,
    device: torch.device,
) -> tuple[SmallUNet2D, dict[str, Any]]:
    if yaml.safe_load(model_config_path.read_text(encoding="utf-8")) != MODEL_CONFIG:
        raise ValueError("Model configuration differs from the implemented contract")
    checkpoint = load_model_checkpoint(checkpoint_path)
    if checkpoint.get("input_mode") != "multimodal":
        raise ValueError("Inference requires the frozen multimodal checkpoint")
    expected = {
        "manifest_sha256": file_sha256(manifest_path),
        "normalization_sha256": file_sha256(normalization_path),
        "model_config_sha256": file_sha256(model_config_path),
    }
    provenance = checkpoint.get("provenance", {})
    for name, digest in expected.items():
        if provenance.get(name) != digest:
            raise ValueError(f"Checkpoint provenance mismatch: {name}")
    model = SmallUNet2D(
        MODEL_CONFIG["input_channels"]["multimodal"],
        MODEL_CONFIG["base_channels"],
        MODEL_CONFIG["output_channels"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def validate_prediction_files(npz_path: Path, json_path: Path, pair_id: str) -> dict[str, Any]:
    """Reopen an inference product and enforce shape, dtype and NaN-mask semantics."""
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if (metadata.get("schema_version") != INFERENCE_SCHEMA
            or metadata.get("status") != "PASS"
            or metadata.get("pair_id") != pair_id):
        raise ValueError("Prediction metadata identity or schema is invalid")
    with np.load(npz_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {PREDICTION_ARRAY, PREDICTION_MASK}:
            raise ValueError("Prediction NPZ keys differ from the inference contract")
        values = arrays[PREDICTION_ARRAY]
        mask = arrays[PREDICTION_MASK]
        if values.shape != (256, 256) or values.dtype != np.float32:
            raise ValueError("Prediction array shape or dtype is invalid")
        if mask.shape != (256, 256) or mask.dtype != np.bool_:
            raise ValueError("Prediction mask shape or dtype is invalid")
        if not np.array_equal(np.isfinite(values), mask):
            raise ValueError("Prediction NaN pattern differs from its boolean mask")
        if int(mask.sum()) != int(metadata.get("valid_pixels", -1)):
            raise ValueError("Prediction valid-pixel count differs from metadata")
    if file_sha256(npz_path) != metadata.get("npz_sha256"):
        raise ValueError("Prediction NPZ hash differs from metadata")
    return metadata


class MultimodalInferenceEngine:
    """Load the frozen model once and create audited products for manifest pair IDs."""

    def __init__(
        self, *, data_root: Path, manifest_path: Path, normalization_path: Path,
        model_config_path: Path, checkpoint_path: Path, requested_device: str = "auto",
    ) -> None:
        paths = [manifest_path, normalization_path, model_config_path, checkpoint_path]
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError("An inference manifest/config/checkpoint file is missing")
        self.data_root = data_root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.normalization_path = normalization_path.resolve()
        self.model_config_path = model_config_path.resolve()
        self.checkpoint_path = checkpoint_path.resolve()
        self.device = choose_device(requested_device)
        self.model, self.checkpoint = _load_model(
            self.checkpoint_path, self.manifest_path, self.normalization_path,
            self.model_config_path, self.device,
        )
        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            manifest_rows = list(csv.DictReader(handle))
        available_splits = sorted({row.get("split", "") for row in manifest_rows})
        if not available_splits or any(
                split not in {"train", "validation", "test"} for split in available_splits):
            raise ValueError("Inference manifest contains an unsupported or missing split")
        self.datasets = {
            split: ForecastDataset(
                self.manifest_path, self.normalization_path, self.data_root,
                split=split, input_mode="multimodal",
            )
            for split in available_splits
        }
        self.pair_locations = {
            row["pair_id"]: (split, index)
            for split, dataset in self.datasets.items()
            for index, row in enumerate(dataset.rows)
        }
        if len(self.pair_locations) != sum(len(dataset) for dataset in self.datasets.values()):
            raise ValueError("Forecast manifest contains duplicate pair identifiers")

    def model_info(self) -> dict[str, Any]:
        """Return public immutable model identity without local filesystem paths."""
        return {
            "input_mode": "multimodal",
            "checkpoint_epoch": int(self.checkpoint["epoch"]),
            "checkpoint_sha256": file_sha256(self.checkpoint_path),
            "manifest_sha256": file_sha256(self.manifest_path),
            "normalization_sha256": file_sha256(self.normalization_path),
            "model_config_sha256": file_sha256(self.model_config_path),
            "device": str(self.device),
            "available_pairs": len(self.pair_locations),
            "output_schema": INFERENCE_SCHEMA,
        }

    @torch.no_grad()
    def predict(self, pair_id: str, output_dir: Path) -> dict[str, Any]:
        """Predict one future field without reading its future target array."""
        if re.fullmatch(r"[A-Za-z0-9_.-]+", pair_id) is None:
            raise ValueError("pair_id contains characters unsafe for an output filename")
        if pair_id not in self.pair_locations:
            raise KeyError(pair_id)
        output_dir = output_dir.resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError("Inference output directory must be empty")
        output_dir.mkdir(parents=True, exist_ok=True)
        split, index = self.pair_locations[pair_id]
        dataset = self.datasets[split]
        sample = dataset.input_item(index)
        started = time.perf_counter()
        prediction = self.model(sample["inputs"].unsqueeze(0).to(self.device))[0, 0]
        if self.device.type == "mps":
            torch.mps.synchronize()
        runtime = time.perf_counter() - started
        prediction = prediction.detach().cpu().numpy().astype(np.float32, copy=False)
        radar_stats = dataset.normalization["fields"]["radar_reflectivity_dbz"]
        prediction = prediction * np.float32(radar_stats["std"]) + np.float32(
            radar_stats["mean"]
        )
        mask = sample["input_radar_mask"][0].numpy().astype(np.bool_, copy=False)
        serialized = np.full((256, 256), np.nan, dtype=np.float32)
        serialized[mask] = prediction[mask]
        if not np.all(np.isfinite(serialized[mask])):
            raise RuntimeError("Inference produced a non-finite value on a valid pixel")

        npz_path = output_dir / f"{pair_id}_prediction.npz"
        json_path = output_dir / f"{pair_id}_prediction.json"
        atomic_write_npz(npz_path, {PREDICTION_ARRAY: serialized, PREDICTION_MASK: mask})
        valid = serialized[mask]
        model_info = self.model_info()
        metadata = {
            "schema_version": INFERENCE_SCHEMA,
            "status": "PASS",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "pair_id": pair_id,
            "split": split,
            "input_sample_id": sample["input_sample_id"],
            "satellite_sample_id": sample["satellite_sample_id"],
            "input_time_utc": sample["input_time_utc"],
            "forecast_valid_time_utc": sample["target_time_utc"],
            "lead_time_minutes": 10,
            **{key: model_info[key] for key in (
                "input_mode", "checkpoint_epoch", "checkpoint_sha256", "manifest_sha256",
                "normalization_sha256", "model_config_sha256", "device",
            )},
            "runtime_seconds": runtime,
            "array_shape": [256, 256],
            "array_dtype": "float32",
            "unit": "dBZ",
            "valid_mask_rule": "input_radar_valid_mask",
            "valid_pixels": int(mask.sum()),
            "valid_percent": 100.0 * float(mask.mean()),
            "valid_value_min_dbz": float(valid.min()),
            "valid_value_mean_dbz": float(valid.mean(dtype=np.float64)),
            "valid_value_max_dbz": float(valid.max()),
            "missing_value_policy": "NaN where prediction_valid_mask is false; never zero-filled",
            "future_target_array_read": False,
            "npz_path": str(npz_path),
            "npz_sha256": file_sha256(npz_path),
        }
        atomic_write_json(json_path, metadata)
        reopened = validate_prediction_files(npz_path, json_path, pair_id)
        return {**reopened, "npz_path": str(npz_path), "json_path": str(json_path)}


def run_multimodal_inference(
    *,
    data_root: Path,
    manifest_path: Path,
    normalization_path: Path,
    model_config_path: Path,
    checkpoint_path: Path,
    pair_id: str,
    output_dir: Path,
    requested_device: str = "auto",
) -> dict[str, Any]:
    """Load a frozen engine and predict one audited manifest pair."""
    engine = MultimodalInferenceEngine(
        data_root=data_root, manifest_path=manifest_path,
        normalization_path=normalization_path, model_config_path=model_config_path,
        checkpoint_path=checkpoint_path, requested_device=requested_device,
    )
    return engine.predict(pair_id, output_dir)
