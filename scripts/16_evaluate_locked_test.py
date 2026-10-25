#!/usr/bin/env python3
"""Open the locked test split once and evaluate the three frozen methods."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhabe_forecasting.dataset import ForecastDataset
from nhabe_forecasting.evaluation import EVALUATION_POLICY
from nhabe_forecasting.final_evaluation import evaluate_method, relative_reduction
from nhabe_forecasting.model import MODEL_CONFIG, SmallUNet2D
from nhabe_forecasting.normalization import file_sha256
from nhabe_forecasting.training import (
    TRAINING_CONFIG, choose_device, load_model_checkpoint,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_frozen_repository() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout
    if status:
        raise RuntimeError("Repository must be clean before opening the locked test split")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def audit_checkpoint(
    run_dir: Path, expected_mode: str, paths: dict[str, Path], source_commit: str,
) -> tuple[dict, dict]:
    summary_path = run_dir / "training_summary.json"
    checkpoint_path = run_dir / "best_model.pt"
    history_path = run_dir / "training_history.csv"
    summary = read_json(summary_path)
    if (summary.get("status") != "PASS" or not summary.get("official_training")
            or summary.get("input_mode") != expected_mode
            or summary.get("test_split_opened") is not False):
        raise ValueError(f"Invalid frozen training summary for {expected_mode}")
    expected_hashes = {
        "manifest_sha256": file_sha256(paths["manifest"]),
        "normalization_sha256": file_sha256(paths["normalization"]),
        "model_config_sha256": file_sha256(paths["model_config"]),
        "training_config_sha256": file_sha256(paths["training_config"]),
        "history_sha256": file_sha256(history_path),
    }
    for name, digest in expected_hashes.items():
        if summary["provenance"].get(name) != digest:
            raise ValueError(f"Frozen {expected_mode} provenance mismatch: {name}")
    if summary["checkpoint_sha256"] != file_sha256(checkpoint_path):
        raise ValueError(f"Frozen {expected_mode} checkpoint hash mismatch")
    checkpoint = load_model_checkpoint(checkpoint_path)
    if (checkpoint.get("input_mode") != expected_mode
            or checkpoint.get("epoch") != summary.get("best_epoch")):
        raise ValueError(f"Frozen {expected_mode} checkpoint identity mismatch")
    audit = {
        "mode": expected_mode,
        "training_summary_path": str(summary_path),
        "training_summary_sha256": file_sha256(summary_path),
        "history_sha256": expected_hashes["history_sha256"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "best_epoch": summary["best_epoch"],
        "frozen_validation": summary["best_validation"],
        "training_test_split_opened": False,
        "evaluation_source_commit": source_commit,
    }
    return checkpoint, audit


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, default=ROOT / "configs/evaluation.yaml")
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs/model.yaml")
    parser.add_argument("--training-config", type=Path, default=ROOT / "configs/training.yaml")
    parser.add_argument("--radar-run-dir", type=Path, required=True)
    parser.add_argument("--multimodal-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    source_commit = require_frozen_repository()
    if yaml.safe_load(args.evaluation_config.read_text()) != EVALUATION_POLICY:
        raise ValueError("Evaluation policy differs from the implemented contract")
    if yaml.safe_load(args.model_config.read_text()) != MODEL_CONFIG:
        raise ValueError("Model configuration differs from the implemented contract")
    if yaml.safe_load(args.training_config.read_text()) != TRAINING_CONFIG:
        raise ValueError("Training configuration differs from the implemented contract")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Final evaluation output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": args.manifest.resolve(),
        "normalization": args.normalization.resolve(),
        "model_config": args.model_config.resolve(),
        "training_config": args.training_config.resolve(),
    }
    radar_checkpoint, radar_audit = audit_checkpoint(
        args.radar_run_dir.resolve(), "radar", paths, source_commit,
    )
    multimodal_checkpoint, multimodal_audit = audit_checkpoint(
        args.multimodal_run_dir.resolve(), "multimodal", paths, source_commit,
    )
    checkpoint_audit = {
        "schema_version": "nhabe-locked-test-checkpoint-audit-v1",
        "status": "PASS",
        "source_commit": source_commit,
        "manifest_sha256": file_sha256(paths["manifest"]),
        "normalization_sha256": file_sha256(paths["normalization"]),
        "evaluation_config_sha256": file_sha256(args.evaluation_config),
        "model_config_sha256": file_sha256(paths["model_config"]),
        "training_config_sha256": file_sha256(paths["training_config"]),
        "checkpoints": [radar_audit, multimodal_audit],
    }
    audit_path = output / "checkpoint_audit.json"
    audit_path.write_text(json.dumps(checkpoint_audit, indent=2) + "\n")

    radar_dataset = ForecastDataset(
        args.manifest, args.normalization, args.data_root,
        split="test", input_mode="radar",
    )
    multimodal_dataset = ForecastDataset(
        args.manifest, args.normalization, args.data_root,
        split="test", input_mode="multimodal",
    )
    if [row["pair_id"] for row in radar_dataset.rows] != [row["pair_id"] for row in multimodal_dataset.rows]:
        raise RuntimeError("Radar and multimodal test pair order differs")
    radar_loader = DataLoader(radar_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    multimodal_loader = DataLoader(multimodal_dataset, batch_size=args.batch_size,
                                   shuffle=False, num_workers=0)
    device = choose_device(args.device)
    radar_std = float(radar_dataset.normalization["fields"]["radar_reflectivity_dbz"]["std"])
    radar_model = SmallUNet2D(2, MODEL_CONFIG["base_channels"], 1).to(device)
    radar_model.load_state_dict(radar_checkpoint["model_state_dict"])
    multimodal_model = SmallUNet2D(8, MODEL_CONFIG["base_channels"], 1).to(device)
    multimodal_model.load_state_dict(multimodal_checkpoint["model_state_dict"])

    started = time.perf_counter()
    evaluations = []
    sample_rows = []
    for method, loader, model in (
        ("persistence", radar_loader, None),
        ("radar_unet", radar_loader, radar_model),
        ("multimodal_unet", multimodal_loader, multimodal_model),
    ):
        result, rows = evaluate_method(method, loader, device, radar_std, model)
        evaluations.append(result); sample_rows.extend(rows)
        print(f"{method}: MAE={result['mae_dbz']:.6f} RMSE={result['rmse_dbz']:.6f} dBZ",
              file=sys.stderr)
    if device.type == "mps":
        torch.mps.synchronize()
    runtime = time.perf_counter() - started
    reference = evaluations[0]
    for result in evaluations:
        if (result["pairs"], result["total_grid_pixels"], result["evaluable_pixels"]) != (
                reference["pairs"], reference["total_grid_pixels"], reference["evaluable_pixels"]):
            raise RuntimeError("Methods were not evaluated on the same domain")
        result["mae_reduction_vs_persistence_percent"] = relative_reduction(
            reference["mae_dbz"], result["mae_dbz"])
        result["rmse_reduction_vs_persistence_percent"] = relative_reduction(
            reference["rmse_dbz"], result["rmse_dbz"])
    radar_result, multimodal_result = evaluations[1], evaluations[2]
    multimodal_result["mae_reduction_vs_radar_percent"] = relative_reduction(
        radar_result["mae_dbz"], multimodal_result["mae_dbz"])
    multimodal_result["rmse_reduction_vs_radar_percent"] = relative_reduction(
        radar_result["rmse_dbz"], multimodal_result["rmse_dbz"])

    samples_path = output / "final_test_samples.csv"
    comparison_path = output / "method_comparison.csv"
    write_csv(samples_path, sample_rows)
    write_csv(comparison_path, evaluations)
    summary = {
        "schema_version": "nhabe-locked-test-evaluation-v1",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": "test",
        "test_pairs": len(radar_dataset),
        "model_or_hyperparameter_changes_after_test": False,
        "device": str(device),
        "runtime_seconds": runtime,
        "evaluation_policy": EVALUATION_POLICY,
        "methods": evaluations,
        "source_commit": source_commit,
        "artifact_sha256": {
            "checkpoint_audit.json": file_sha256(audit_path),
            "final_test_samples.csv": file_sha256(samples_path),
            "method_comparison.csv": file_sha256(comparison_path),
        },
    }
    summary_path = output / "final_test_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    reopened = read_json(summary_path)
    for name, digest in reopened["artifact_sha256"].items():
        if file_sha256(output / name) != digest:
            raise RuntimeError(f"Final artifact failed reopen hash check: {name}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
