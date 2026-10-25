"""PyTorch dataset for audited radar--Himawari forecast pairs.

Source NPZ files remain read-only. Missing physical values are normalized only at
valid pixels and replaced by zero in the returned in-memory tensors; masks retain
the distinction between a missing value and a real normalized zero.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from .normalization import BANDS, file_sha256, valid_values
from .splits import utc_time


InputMode = Literal["radar", "multimodal"]
RADAR_CHANNELS = ("radar_reflectivity", "radar_valid_mask")
MULTIMODAL_CHANNELS = (
    "radar_reflectivity", *BANDS, "radar_valid_mask", "himawari_all_bands_valid_mask",
)


def _source_path(data_root: Path, value: str) -> Path:
    """Resolve a manifest path while preventing escape from the data root."""
    root = data_root.resolve()
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Source path escapes data root: {value}")
    return resolved


def _normalized(values: np.ndarray, mask: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Return finite float32 model values without changing the source array."""
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Normalization mean/std must be finite and std must be positive")
    result = np.zeros(values.shape, dtype=np.float32)
    result[mask] = ((values[mask] - np.float32(mean)) / np.float32(std)).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("Normalization produced a non-finite model value")
    return result


def _read_radar(path: Path, mean: float, std: float) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        values = arrays["radar_reflectivity_dbz"]
        mask = arrays["radar_data_valid_mask"]
        valid_values(values, mask, (256, 256))
        return _normalized(values, mask, mean, std), mask.copy()


def _read_satellite(
    path: Path, fields: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        values = arrays["himawari_bt"]
        band_mask = arrays["himawari_band_valid_mask"]
        valid_values(values, band_mask, (5, 256, 256))
        all_mask = arrays["himawari_all_bands_valid_mask"]
        if all_mask.shape != (256, 256) or all_mask.dtype != np.bool_:
            raise ValueError("Unexpected Himawari all-band mask shape or dtype")
        if not np.array_equal(all_mask, np.all(band_mask, axis=0)):
            raise ValueError("Himawari all-band mask disagrees with per-band masks")
        normalized = np.zeros(values.shape, dtype=np.float32)
        for index, band in enumerate(BANDS):
            stats = fields[band]
            normalized[index] = _normalized(
                values[index], band_mask[index], float(stats["mean"]), float(stats["std"]),
            )
        return normalized, band_mask.copy(), all_mask.copy()


class ForecastDataset(Dataset[dict[str, Any]]):
    """Read one audited forecast pair at a time and build model-ready tensors."""

    def __init__(
        self,
        manifest_path: str | Path,
        normalization_path: str | Path,
        data_root: str | Path,
        *,
        split: Literal["train", "validation", "test"],
        input_mode: InputMode = "multimodal",
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.normalization_path = Path(normalization_path).resolve()
        self.data_root = Path(data_root).resolve()
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation or test")
        if input_mode not in {"radar", "multimodal"}:
            raise ValueError("input_mode must be radar or multimodal")
        self.split = split
        self.input_mode = input_mode
        self.channel_names = RADAR_CHANNELS if input_mode == "radar" else MULTIMODAL_CHANNELS

        with self.normalization_path.open(encoding="utf-8") as handle:
            self.normalization = json.load(handle)
        self._validate_normalization()
        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            all_rows = list(csv.DictReader(handle))
        self.rows = [row for row in all_rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No forecast pairs for split: {split}")
        for row in self.rows:
            self._validate_row(row)

    def _validate_normalization(self) -> None:
        stats = self.normalization
        if (stats.get("schema_version") != "nhabe-train-normalization-v1"
                or stats.get("status") != "TRAIN_STATISTICS_FITTED"):
            raise ValueError("Normalization artifact has an unsupported schema or status")
        if stats.get("band_order") != list(BANDS):
            raise ValueError("Normalization band order differs from the dataset contract")
        fields = stats.get("fields", {})
        for name in ("radar_reflectivity_dbz", *BANDS):
            if name not in fields:
                raise ValueError(f"Normalization statistics missing field: {name}")
            mean, std = float(fields[name]["mean"]), float(fields[name]["std"])
            if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
                raise ValueError(f"Invalid normalization statistics for: {name}")
        expected = stats.get("provenance", {}).get("manifest_sha256")
        if not expected or file_sha256(self.manifest_path) != expected:
            raise ValueError("Forecast manifest differs from the normalization provenance")

    @staticmethod
    def _validate_row(row: Mapping[str, str]) -> None:
        required = (
            "pair_id", "input_sample_id", "target_sample_id", "satellite_sample_id",
            "input_npz_path", "target_npz_path", "satellite_npz_path", "input_time_utc",
            "target_time_utc", "lead_time_minutes", "split", "qc_status", "radar_signature",
            "timing_header_count",
        )
        if any(not row.get(name) for name in required):
            raise ValueError("Forecast manifest row is missing a required value")
        if row["qc_status"] != "PASS" or row["radar_signature"] != "A":
            raise ValueError("Dataset accepts only audited Signature A pairs")
        if int(row["timing_header_count"]) != 10:
            raise ValueError("Forecast pair lacks ten audited Himawari source headers")
        lead = (utc_time(row["target_time_utc"]) - utc_time(row["input_time_utc"])).total_seconds()
        if lead != 600 or float(row["lead_time_minutes"]) != 10:
            raise ValueError("Forecast pair must have an exact 600-second radar midpoint lead")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        fields = self.normalization["fields"]
        radar_stats = fields["radar_reflectivity_dbz"]
        input_radar, input_mask = _read_radar(
            _source_path(self.data_root, row["input_npz_path"]),
            float(radar_stats["mean"]), float(radar_stats["std"]),
        )
        target, target_mask = _read_radar(
            _source_path(self.data_root, row["target_npz_path"]),
            float(radar_stats["mean"]), float(radar_stats["std"]),
        )

        channels = [input_radar]
        if self.input_mode == "multimodal":
            satellite, _band_mask, all_band_mask = _read_satellite(
                _source_path(self.data_root, row["satellite_npz_path"]), fields,
            )
            channels.extend(satellite)
            channels.extend((input_mask.astype(np.float32), all_band_mask.astype(np.float32)))
        else:
            channels.append(input_mask.astype(np.float32))

        inputs = np.stack(channels).astype(np.float32, copy=False)
        target = target[np.newaxis, ...]
        target_mask = target_mask[np.newaxis, ...]
        if not np.all(np.isfinite(inputs)) or not np.all(np.isfinite(target)):
            raise ValueError("Model tensors must contain no NaN or infinity")
        return {
            "inputs": torch.from_numpy(inputs),
            "target": torch.from_numpy(target),
            "input_radar_mask": torch.from_numpy(input_mask[np.newaxis, ...]),
            "target_mask": torch.from_numpy(target_mask),
            "pair_id": row["pair_id"],
            "input_time_utc": row["input_time_utc"],
            "target_time_utc": row["target_time_utc"],
        }
