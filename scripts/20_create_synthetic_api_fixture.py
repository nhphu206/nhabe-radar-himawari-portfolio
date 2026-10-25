#!/usr/bin/env python3
"""Create a shareable synthetic bundle for API and Docker CI smoke checks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhabe_forecasting.model import MODEL_CONFIG, SmallUNet2D
from nhabe_forecasting.normalization import BANDS, file_sha256
from nhabe_forecasting.training import save_model_checkpoint, seed_everything
from nhabe_preprocessing.io_utils import atomic_write_json, atomic_write_npz


PAIR_ID = "CI_SYNTHETIC_PAIR"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs/model.yaml")
    parser.add_argument(
        "--include-historical-target", action="store_true",
        help="Add a synthetic future target for the optional dashboard evaluation view",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Synthetic API fixture output must be empty")
    output.mkdir(parents=True, exist_ok=True)

    shape = (256, 256)
    yy, xx = np.mgrid[:256, :256]
    radar = (12.0 + 4.0 * np.sin(xx / 25.0) + 2.0 * np.cos(yy / 31.0)).astype(np.float32)
    radar_mask = ((xx - 128) ** 2 + (yy - 128) ** 2 <= 105 ** 2)
    radar[~radar_mask] = np.nan
    radar_path = output / "input_radar.npz"
    atomic_write_npz(radar_path, {
        "radar_reflectivity_dbz": radar,
        "radar_data_valid_mask": radar_mask.astype(np.bool_),
    })

    satellite = np.stack([
        (230.0 + 6.0 * index + np.sin(xx / (18.0 + index))).astype(np.float32)
        for index in range(5)
    ])
    band_mask = np.broadcast_to(radar_mask, satellite.shape).copy()
    satellite[~band_mask] = np.nan
    satellite_path = output / "previous_himawari.npz"
    atomic_write_npz(satellite_path, {
        "himawari_bt": satellite,
        "himawari_band_valid_mask": band_mask.astype(np.bool_),
        "himawari_all_bands_valid_mask": np.all(band_mask, axis=0),
    })

    target_name = (
        "future_target.npz" if args.include_historical_target
        else "future_target_intentionally_absent.npz"
    )
    if args.include_historical_target:
        target = (
            13.0 + 4.0 * np.sin((xx - 8.0) / 25.0)
            + 2.0 * np.cos((yy + 3.0) / 31.0)
        ).astype(np.float32)
        target[~radar_mask] = np.nan
        atomic_write_npz(output / target_name, {
            "radar_reflectivity_dbz": target,
            "radar_data_valid_mask": radar_mask.astype(np.bool_),
        })

    sample_kind = (
        "SYNTHETIC_INTERACTIVE_DEMO_FIXTURE" if args.include_historical_target
        else "SYNTHETIC_API_CONTRACT_FIXTURE"
    )
    row = {
        "pair_id": PAIR_ID,
        "input_sample_id": "CI_INPUT_RADAR",
        "target_sample_id": (
            "CI_SYNTHETIC_FUTURE_TARGET" if args.include_historical_target
            else "CI_FUTURE_TARGET_NOT_PRESENT"
        ),
        "satellite_sample_id": "CI_PREVIOUS_HIMAWARI",
        "input_time_utc": "2025-08-22T00:00:00Z",
        "target_time_utc": "2025-08-22T00:10:00Z",
        "lead_time_minutes": "10",
        "radar_signature": "A",
        "input_npz_path": radar_path.name,
        "target_npz_path": target_name,
        "satellite_npz_path": satellite_path.name,
        "split": "validation",
        "qc_status": "PASS",
        "timing_header_count": "10",
    }
    manifest_path = output / "forecast_pairs.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)

    fields = {"radar_reflectivity_dbz": {"mean": 10.0, "std": 5.0}}
    fields.update({
        band: {"mean": 230.0 + 6.0 * index, "std": 8.0}
        for index, band in enumerate(BANDS)
    })
    normalization_path = output / "normalization.json"
    atomic_write_json(normalization_path, {
        "schema_version": "nhabe-train-normalization-v1",
        "status": "TRAIN_STATISTICS_FITTED",
        "sample_kind": sample_kind,
        "band_order": list(BANDS),
        "fields": fields,
        "provenance": {"manifest_sha256": file_sha256(manifest_path)},
    })

    seed_everything(2025)
    model = SmallUNet2D(
        MODEL_CONFIG["input_channels"]["multimodal"],
        MODEL_CONFIG["base_channels"], MODEL_CONFIG["output_channels"],
    )
    checkpoint_path = output / "best_model.pt"
    save_model_checkpoint(
        checkpoint_path, model, "multimodal", 0,
        {"rmse_dbz_common_mask": 0.0, "synthetic_weights_not_trained": True},
        {
            "manifest_sha256": file_sha256(manifest_path),
            "normalization_sha256": file_sha256(normalization_path),
            "model_config_sha256": file_sha256(args.model_config.resolve()),
            "sample_kind": sample_kind,
        },
    )
    summary = {
        "schema_version": "nhabe-synthetic-api-fixture-v1",
        "status": "PASS",
        "sample_kind": sample_kind,
        "meteorological_observation": False,
        "trained_checkpoint": False,
        "pair_id": PAIR_ID,
        "future_target_file_exists": (output / row["target_npz_path"]).exists(),
        "manifest_sha256": file_sha256(manifest_path),
        "normalization_sha256": file_sha256(normalization_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    }
    if summary["future_target_file_exists"] != args.include_historical_target:
        raise RuntimeError("Synthetic future-target policy differs from the requested mode")
    atomic_write_json(output / "fixture_summary.json", summary)
    reopened = json.loads((output / "fixture_summary.json").read_text(encoding="utf-8"))
    if reopened != summary:
        raise RuntimeError("Synthetic API fixture summary failed roundtrip")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
