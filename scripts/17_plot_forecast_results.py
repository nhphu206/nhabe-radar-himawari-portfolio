#!/usr/bin/env python3
"""Create report-ready curves from verified training and evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_history(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radar-run-dir", type=Path, required=True)
    parser.add_argument("--multimodal-run-dir", type=Path, required=True)
    parser.add_argument("--persistence-validation", type=Path, required=True)
    parser.add_argument("--final-test-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Figure output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)

    radar_history = read_history(args.radar_run_dir / "training_history.csv")
    multimodal_history = read_history(args.multimodal_run_dir / "training_history.csv")
    radar_summary = read_json(args.radar_run_dir / "training_summary.json")
    multimodal_summary = read_json(args.multimodal_run_dir / "training_summary.json")
    persistence = read_json(args.persistence_validation)
    final_test = read_json(args.final_test_summary)
    if final_test.get("status") != "PASS" or final_test.get("split") != "test":
        raise ValueError("Final test summary is not a verified PASS artifact")

    series = (
        ("Radar-only U-Net", radar_history, "#1f77b4"),
        ("Radar + Himawari U-Net", multimodal_history, "#ff7f0e"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, first_epoch, title in (
        (axes[0], 0, "Full trajectory"),
        (axes[1], 1, "Epochs 1–30 (zoom)"),
    ):
        for label, history, color in series:
            selected = [row for row in history if int(row["epoch"]) >= first_epoch]
            axis.plot(
                [int(row["epoch"]) for row in selected],
                [float(row["rmse_dbz_common_mask"]) for row in selected],
                marker="o", markersize=3, linewidth=1.5, label=label, color=color,
            )
        axis.axhline(
            float(persistence["rmse_dbz"]), color="#555555", linestyle="--",
            label="Persistence validation",
        )
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation RMSE (dBZ)")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("Validation RMSE during official training")
    fig.savefig(output / "training_validation_rmse.png", dpi=200)
    plt.close(fig)

    test_by_method = {row["method"]: row for row in final_test["methods"]}
    labels = ["Persistence", "Radar-only", "Radar + Himawari"]
    validation_mae = [
        float(persistence["mae_dbz"]),
        float(radar_summary["best_validation"]["mae_dbz_common_mask"]),
        float(multimodal_summary["best_validation"]["mae_dbz_common_mask"]),
    ]
    validation_rmse = [
        float(persistence["rmse_dbz"]),
        float(radar_summary["best_validation"]["rmse_dbz_common_mask"]),
        float(multimodal_summary["best_validation"]["rmse_dbz_common_mask"]),
    ]
    test_mae = [float(test_by_method[name]["mae_dbz"])
                for name in ("persistence", "radar_unet", "multimodal_unet")]
    test_rmse = [float(test_by_method[name]["rmse_dbz"])
                 for name in ("persistence", "radar_unet", "multimodal_unet")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    positions = list(range(3)); width = 0.36
    for axis, validation, test, title in (
        (axes[0], validation_mae, test_mae, "MAE"),
        (axes[1], validation_rmse, test_rmse, "RMSE"),
    ):
        validation_bars = axis.bar(
            [position - width / 2 for position in positions], validation, width,
            label="Validation", color="#4c78a8",
        )
        test_bars = axis.bar(
            [position + width / 2 for position in positions], test, width,
            label="Locked test", color="#f58518",
        )
        axis.bar_label(validation_bars, fmt="%.2f", padding=2, fontsize=8)
        axis.bar_label(test_bars, fmt="%.2f", padding=2, fontsize=8)
        axis.set_xticks(positions, labels, rotation=15, ha="right")
        axis.set_ylabel(f"{title} (dBZ)")
        axis.set_title(f"{title}: validation vs locked test")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.savefig(output / "validation_test_comparison.png", dpi=200)
    plt.close(fig)
    print(output / "training_validation_rmse.png")
    print(output / "validation_test_comparison.png")


if __name__ == "__main__":
    main()
