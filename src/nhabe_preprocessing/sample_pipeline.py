"""Deterministic mock workflow for validating the public sample contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .common_grid import build_common_grid
from .io_utils import atomic_write_json, atomic_write_npz
from .validation import validate_arrays, validate_sample_files


MOCK_SAMPLE_ID = "MOCK_NHB_20250801T000000Z_A"


def build_mock_arrays(config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Create small, deterministic fields with the production array shapes and masks."""
    grid = build_common_grid(config)
    height, width = map(int, config["grid"]["shape_yx"])
    yy, xx = np.meshgrid(grid.y, grid.x, indexing="ij")

    radar = (18.0 * np.exp(-((xx / 55_000.0) ** 2 + (yy / 45_000.0) ** 2)) - 8.0)
    radar = radar.astype(np.float32)
    radar_valid = (grid.radius <= 90_000.0) & ~((xx > 20_000.0) & (yy > 20_000.0))
    radar[~radar_valid] = np.nan

    satellite_layers = []
    for index in range(len(config["himawari"]["bands"])):
        layer = 255.0 + 4.0 * index + 7.0 * np.sin(xx / 80_000.0) * np.cos(yy / 70_000.0)
        satellite_layers.append(layer.astype(np.float32))
    satellite = np.stack(satellite_layers)
    satellite[:, :2, :] = np.nan
    satellite[:, :, -2:] = np.nan
    band_valid = np.isfinite(satellite)

    arrays = {
        "radar_reflectivity_dbz": radar,
        "radar_data_valid_mask": np.isfinite(radar),
        "radar_range_support_mask": (
            grid.radius <= float(config["radar"]["signatures"]["A"]["max_range_m"])
        ),
        "himawari_bt": satellite,
        "himawari_band_valid_mask": band_valid,
        "himawari_all_bands_valid_mask": np.all(band_valid, axis=0),
    }
    if arrays["radar_reflectivity_dbz"].shape != (height, width):
        raise ValueError("Mock radar shape does not match the configured grid")
    validate_arrays(arrays)
    return arrays


def run_mock_pipeline(output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Write, reopen, and validate one shareable synthetic NPZ/JSON sample."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = build_mock_arrays(config)
    npz_path = output_dir / f"{MOCK_SAMPLE_ID}.npz"
    json_path = output_dir / f"{MOCK_SAMPLE_ID}.json"
    metadata = {
        "schema_version": config["sample_contract"]["schema_version"],
        "sample_kind": "SYNTHETIC_CONTRACT_SMOKE_TEST",
        "radar_filename": "MOCK_NOT_RAW_DATA",
        "radar_signature": "A",
        "time_match_status": "MATCHED",
        "absolute_delta_seconds": 0.0,
        "signed_delta_seconds_himawari_minus_radar": 0.0,
        "leakage_group_id": "MOCK_20250801_0000",
        "grid_id": config["grid"]["grid_id"],
        "grid_shape_yx": list(config["grid"]["shape_yx"]),
        "band_order": list(config["sample_contract"]["band_order"]),
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        },
        "notice": "Generated data; not a meteorological observation.",
    }
    atomic_write_npz(npz_path, arrays)
    atomic_write_json(json_path, metadata)
    reopened = validate_sample_files(npz_path, json_path, "MOCK_NOT_RAW_DATA")
    summary = {
        "status": "PASS",
        "sample_id": MOCK_SAMPLE_ID,
        "npz_path": str(npz_path),
        "json_path": str(json_path),
        "radar_shape": list(arrays["radar_reflectivity_dbz"].shape),
        "himawari_shape": list(arrays["himawari_bt"].shape),
        "radar_dtype": str(arrays["radar_reflectivity_dbz"].dtype),
        "himawari_dtype": str(arrays["himawari_bt"].dtype),
        "nan_mask_contract_valid": True,
        "metadata_schema_version": reopened["schema_version"],
    }
    atomic_write_json(output_dir / "sample_run_summary.json", summary)
    return summary
