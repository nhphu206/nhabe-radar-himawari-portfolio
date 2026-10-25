#!/usr/bin/env python3
"""Train one official Small U-Net mode with validation-only model selection."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhabe_forecasting.dataset import ForecastDataset
from nhabe_forecasting.model import MODEL_CONFIG, SmallUNet2D, trainable_parameter_count
from nhabe_forecasting.normalization import file_sha256
from nhabe_forecasting.training import (
    TRAINING_CONFIG, choose_device, evaluate_forecast, load_model_checkpoint, make_optimizer,
    save_model_checkpoint, seed_everything, train_one_epoch,
)


def write_history(path: Path, history: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs/model.yaml")
    parser.add_argument("--training-config", type=Path, default=ROOT / "configs/training.yaml")
    parser.add_argument("--input-mode", choices=("radar", "multimodal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    model_config = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    training_config = yaml.safe_load(args.training_config.read_text(encoding="utf-8"))
    if model_config != MODEL_CONFIG or training_config != TRAINING_CONFIG:
        raise ValueError("Model or training configuration differs from the implemented contract")
    settings = training_config["official"]
    if args.input_mode not in settings["input_modes"]:
        raise ValueError("Input mode is not approved for official training")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory contains existing artifacts")
    output.mkdir(parents=True, exist_ok=True)

    seed = int(training_config["seed"])
    seed_everything(seed)
    train_dataset = ForecastDataset(
        args.manifest, args.normalization, args.data_root,
        split="train", input_mode=args.input_mode,
    )
    validation_dataset = ForecastDataset(
        args.manifest, args.normalization, args.data_root,
        split="validation", input_mode=args.input_mode,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=int(settings["batch_size"]), shuffle=True,
        num_workers=int(training_config["num_workers"]), generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=int(settings["batch_size"]), shuffle=False,
        num_workers=int(training_config["num_workers"]),
    )
    device = choose_device(args.device)
    model = SmallUNet2D(
        int(model_config["input_channels"][args.input_mode]),
        int(model_config["base_channels"]), int(model_config["output_channels"]),
    ).to(device)
    optimizer = make_optimizer(model, training_config)
    radar_std = float(train_dataset.normalization["fields"]["radar_reflectivity_dbz"]["std"])
    provenance = {
        "manifest_sha256": file_sha256(args.manifest),
        "normalization_sha256": file_sha256(args.normalization),
        "model_config_sha256": file_sha256(args.model_config),
        "training_config_sha256": file_sha256(args.training_config),
        "torch_version": str(torch.__version__),
        "seed": seed,
    }
    checkpoint_path = output / "best_model.pt"
    history_path = output / "training_history.csv"
    history: list[dict] = []
    started = time.perf_counter()
    initial_validation = evaluate_forecast(model, validation_loader, device, radar_std)
    best_metric = float(initial_validation["rmse_dbz_common_mask"])
    best_epoch = 0
    epochs_without_improvement = 0
    save_model_checkpoint(checkpoint_path, model, args.input_mode, 0, initial_validation, provenance)
    history.append({"epoch": 0, "train_mse": "", **initial_validation, "is_best": True})
    write_history(history_path, history)
    print(f"Epoch 0 validation_rmse={best_metric:.6f} dBZ", file=sys.stderr)

    for epoch in range(1, int(settings["max_epochs"]) + 1):
        train_mse = train_one_epoch(
            model, train_loader, optimizer, device, float(training_config["gradient_clip_norm"]),
        )
        validation = evaluate_forecast(model, validation_loader, device, radar_std)
        metric = float(validation["rmse_dbz_common_mask"])
        improved = metric < best_metric - float(settings["minimum_improvement_rmse_dbz"])
        if improved:
            best_metric = metric
            best_epoch = epoch
            epochs_without_improvement = 0
            save_model_checkpoint(checkpoint_path, model, args.input_mode, epoch, validation, provenance)
        else:
            epochs_without_improvement += 1
        history.append({"epoch": epoch, "train_mse": train_mse, **validation,
                        "is_best": improved})
        write_history(history_path, history)
        print(f"Epoch {epoch} train_mse={train_mse:.6f} validation_rmse={metric:.6f} dBZ"
              f" best={best_metric:.6f} patience={epochs_without_improvement}", file=sys.stderr)
        if epochs_without_improvement >= int(settings["early_stopping_patience"]):
            break
    if device.type == "mps":
        torch.mps.synchronize()
    runtime = time.perf_counter() - started

    checkpoint = load_model_checkpoint(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    best_validation = evaluate_forecast(model, validation_loader, device, radar_std)
    if abs(float(best_validation["rmse_dbz_common_mask"]) - best_metric) > 1e-5:
        raise RuntimeError("Reopened best checkpoint does not reproduce validation RMSE")

    summary = {
        "schema_version": "nhabe-official-training-v1",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "official_training": True,
        "input_mode": args.input_mode,
        "device": str(device),
        "train_pairs": len(train_dataset),
        "validation_pairs": len(validation_dataset),
        "test_split_opened": False,
        "batch_size": int(settings["batch_size"]),
        "epochs_completed": len(history) - 1,
        "stopped_early": len(history) - 1 < int(settings["max_epochs"]),
        "best_epoch": best_epoch,
        "trainable_parameters": trainable_parameter_count(model),
        "best_validation": best_validation,
        "runtime_seconds": runtime,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "provenance": {**provenance, "history_sha256": file_sha256(history_path)},
    }
    summary_path = output / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    reopened = json.loads(summary_path.read_text(encoding="utf-8"))
    if (file_sha256(checkpoint_path) != reopened["checkpoint_sha256"]
            or file_sha256(history_path) != reopened["provenance"]["history_sha256"]):
        raise RuntimeError("Official training artifacts failed reopen verification")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
