#!/usr/bin/env python3
"""Evaluate the persistence baseline on validation data; no model training."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhabe_forecasting.dataset import ForecastDataset
from nhabe_forecasting.evaluation import EVALUATION_POLICY, RegressionAccumulator
from nhabe_forecasting.normalization import file_sha256


def write_artifacts(output: Path, summary: dict, rows: list[dict]) -> None:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory contains existing artifacts")
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "persistence_validation_samples.csv"
    json_path = output / "persistence_validation.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary["artifact_sha256"] = {csv_path.name: file_sha256(csv_path)}
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    reopened = json.loads(json_path.read_text(encoding="utf-8"))
    if reopened["status"] != "PASS" or file_sha256(csv_path) != reopened["artifact_sha256"][csv_path.name]:
        raise RuntimeError("Persistence artifacts failed reopen validation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/evaluation.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    policy = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if policy != EVALUATION_POLICY:
        raise ValueError("Evaluation configuration differs from the approved policy")

    dataset = ForecastDataset(
        args.manifest, args.normalization, args.data_root,
        split="validation", input_mode="radar",
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    radar_stats = dataset.normalization["fields"]["radar_reflectivity_dbz"]
    mean, std = float(radar_stats["mean"]), float(radar_stats["std"])
    overall = RegressionAccumulator()
    sample_rows: list[dict] = []
    total_grid_pixels = 0
    input_valid_pixels = 0
    target_valid_pixels = 0

    for batch_number, batch in enumerate(loader, 1):
        current_dbz = batch["inputs"][:, 0:1] * std + mean
        target_dbz = batch["target"] * std + mean
        input_mask = batch["input_radar_mask"]
        target_mask = batch["target_mask"]
        common_mask = input_mask & target_mask
        overall.update(current_dbz, target_dbz, common_mask)
        total_grid_pixels += int(common_mask.numel())
        input_valid_pixels += int(input_mask.sum())
        target_valid_pixels += int(target_mask.sum())

        for index, pair_id in enumerate(batch["pair_id"]):
            accumulator = RegressionAccumulator()
            accumulator.update(current_dbz[index], target_dbz[index], common_mask[index])
            metric = accumulator.result()
            sample_rows.append({
                "pair_id": pair_id,
                "input_time_utc": batch["input_time_utc"][index],
                "target_time_utc": batch["target_time_utc"][index],
                "evaluable_pixels": metric["valid_pixels"],
                "total_pixels": int(common_mask[index].numel()),
                "evaluable_percent": 100.0 * metric["valid_pixels"] / common_mask[index].numel(),
                "mae_dbz": metric["mae_dbz"],
                "rmse_dbz": metric["rmse_dbz"],
            })
        if batch_number % 10 == 0 or batch_number == len(loader):
            print(f"Validation batches read: {batch_number}/{len(loader)}", file=sys.stderr)

    metric = overall.result()
    summary = {
        "schema_version": "nhabe-persistence-baseline-v1",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "split": "validation",
        "test_split_opened": False,
        "baseline": "prediction at target time equals current radar reflectivity",
        "evaluation_policy": policy,
        "pairs": len(dataset),
        "total_grid_pixels": total_grid_pixels,
        "input_valid_pixels": input_valid_pixels,
        "target_valid_pixels": target_valid_pixels,
        "evaluable_pixels": metric["valid_pixels"],
        "input_valid_percent": 100.0 * input_valid_pixels / total_grid_pixels,
        "target_valid_percent": 100.0 * target_valid_pixels / total_grid_pixels,
        "evaluable_percent": 100.0 * metric["valid_pixels"] / total_grid_pixels,
        "mae_dbz": metric["mae_dbz"],
        "rmse_dbz": metric["rmse_dbz"],
        "aggregation": "global sums divided by all common valid pixels; not mean of per-image metrics",
        "provenance": {
            "manifest_sha256": file_sha256(args.manifest),
            "normalization_sha256": file_sha256(args.normalization),
            "evaluation_config_sha256": file_sha256(args.config),
            "torch_version": torch.__version__,
        },
    }
    write_artifacts(args.output, summary, sample_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
