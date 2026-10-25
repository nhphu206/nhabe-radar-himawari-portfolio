#!/usr/bin/env python3
"""Overfit a fixed tiny train subset to validate the learning pipeline."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from torch.utils.data import DataLoader
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhabe_forecasting.dataset import ForecastDataset
from nhabe_forecasting.model import MODEL_CONFIG, SmallUNet2D, trainable_parameter_count
from nhabe_forecasting.normalization import file_sha256
from nhabe_forecasting.training import (
    TRAINING_CONFIG, choose_device, evaluate_normalized_mse, make_optimizer,
    overfit_gate, seed_everything, train_one_epoch,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs/model.yaml")
    parser.add_argument("--training-config", type=Path, default=ROOT / "configs/training.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    model_config = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    training_config = yaml.safe_load(args.training_config.read_text(encoding="utf-8"))
    if model_config != MODEL_CONFIG or training_config != TRAINING_CONFIG:
        raise ValueError("Model or training configuration differs from the implemented contract")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory contains existing artifacts")
    output.mkdir(parents=True, exist_ok=True)

    settings = training_config["overfit"]
    mode = settings["input_mode"]
    seed = int(training_config["seed"])
    seed_everything(seed)
    dataset = ForecastDataset(
        args.manifest, args.normalization, args.data_root, split="train", input_mode=mode,
    )
    pair_count = int(settings["pairs"])
    if len(dataset) < pair_count:
        raise ValueError("Training split is smaller than the overfit subset")
    cached_samples = [dataset[index] for index in range(pair_count)]
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        cached_samples, batch_size=int(settings["batch_size"]), shuffle=True,
        num_workers=int(training_config["num_workers"]), generator=generator,
    )
    evaluation_loader = DataLoader(
        cached_samples, batch_size=int(settings["batch_size"]), shuffle=False,
        num_workers=int(training_config["num_workers"]),
    )
    device = choose_device(args.device)
    model = SmallUNet2D(
        in_channels=int(model_config["input_channels"][mode]),
        base_channels=int(model_config["base_channels"]),
        out_channels=int(model_config["output_channels"]),
    ).to(device)
    optimizer = make_optimizer(model, training_config)
    history = []
    started = time.perf_counter()
    initial = evaluate_normalized_mse(model, evaluation_loader, device)
    history.append({"epoch": 0, "train_update_mse": "", "evaluation_mse": initial})
    print(f"Epoch 0/{settings['epochs']} evaluation_mse={initial:.6f}", file=sys.stderr)
    for epoch in range(1, int(settings["epochs"]) + 1):
        update_mse = train_one_epoch(
            model, train_loader, optimizer, device, float(training_config["gradient_clip_norm"]),
        )
        evaluation_mse = evaluate_normalized_mse(model, evaluation_loader, device)
        history.append({"epoch": epoch, "train_update_mse": update_mse,
                        "evaluation_mse": evaluation_mse})
        print(f"Epoch {epoch}/{settings['epochs']} evaluation_mse={evaluation_mse:.6f}",
              file=sys.stderr)
    if device.type == "mps":
        torch.mps.synchronize()
    runtime = time.perf_counter() - started
    final = float(history[-1]["evaluation_mse"])
    threshold = float(settings["maximum_final_to_initial_loss_ratio"])
    passed, ratio = overfit_gate(initial, final, threshold)
    best_row = min(history, key=lambda row: float(row["evaluation_mse"]))

    history_path = output / "overfit_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)
    summary = {
        "schema_version": "nhabe-small-subset-overfit-v1",
        "status": "PASS" if passed else "FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "learning-pipeline diagnostic, not a forecasting result",
        "official_training": False,
        "checkpoint_written": False,
        "device": str(device),
        "input_mode": mode,
        "pairs": pair_count,
        "batch_size": int(settings["batch_size"]),
        "epochs": int(settings["epochs"]),
        "trainable_parameters": trainable_parameter_count(model),
        "initial_evaluation_mse": initial,
        "final_evaluation_mse": final,
        "best_evaluation_mse": float(best_row["evaluation_mse"]),
        "best_epoch": int(best_row["epoch"]),
        "final_to_initial_ratio": ratio,
        "acceptance_threshold": threshold,
        "runtime_seconds": runtime,
        "pair_ids": [sample["pair_id"] for sample in cached_samples],
        "provenance": {
            "manifest_sha256": file_sha256(args.manifest),
            "normalization_sha256": file_sha256(args.normalization),
            "model_config_sha256": file_sha256(args.model_config),
            "training_config_sha256": file_sha256(args.training_config),
            "history_sha256": file_sha256(history_path),
            "torch_version": str(torch.__version__),
        },
    }
    summary_path = output / "overfit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    reopened = json.loads(summary_path.read_text(encoding="utf-8"))
    if file_sha256(history_path) != reopened["provenance"]["history_sha256"]:
        raise RuntimeError("Overfit history failed reopen verification")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise RuntimeError("Small-subset overfit acceptance criterion was not met")


if __name__ == "__main__":
    main()
