#!/usr/bin/env python3
"""Plot deterministic validation examples from audited prediction products."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_path(data_root: Path, value: str) -> Path:
    root = data_root.resolve()
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Source path escapes data root: {value}")
    return resolved


def load_radar(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        values = arrays["radar_reflectivity_dbz"].copy()
        mask = arrays["radar_data_valid_mask"].copy()
    if values.shape != (256, 256) or values.dtype != np.float32:
        raise ValueError(f"Unexpected radar array contract: {path}")
    if mask.shape != values.shape or mask.dtype != np.bool_:
        raise ValueError(f"Unexpected radar mask contract: {path}")
    if not np.array_equal(np.isfinite(values), mask):
        raise ValueError(f"Radar NaN pattern differs from mask: {path}")
    return values, mask


def load_prediction(root: Path, pair_id: str) -> tuple[np.ndarray, np.ndarray, dict]:
    pair_root = root / pair_id
    npz_path = pair_root / f"{pair_id}_prediction.npz"
    json_path = pair_root / f"{pair_id}_prediction.json"
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if (metadata.get("status") != "PASS" or metadata.get("pair_id") != pair_id
            or metadata.get("future_target_array_read") is not False):
        raise ValueError(f"Prediction metadata is not an audited PASS product: {pair_id}")
    if metadata.get("npz_sha256") != sha256(npz_path):
        raise ValueError(f"Prediction hash mismatch: {pair_id}")
    with np.load(npz_path, allow_pickle=False) as arrays:
        values = arrays["predicted_reflectivity_dbz"].copy()
        mask = arrays["prediction_valid_mask"].copy()
    if values.shape != (256, 256) or values.dtype != np.float32:
        raise ValueError(f"Unexpected prediction array contract: {pair_id}")
    if mask.shape != values.shape or mask.dtype != np.bool_:
        raise ValueError(f"Unexpected prediction mask contract: {pair_id}")
    if not np.array_equal(np.isfinite(values), mask):
        raise ValueError(f"Prediction NaN pattern differs from mask: {pair_id}")
    return values, mask, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Qualitative output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        validation = [row for row in csv.DictReader(handle) if row["split"] == "validation"]
    if len(validation) != 141:
        raise ValueError("Expected the audited 141-pair validation split")
    selected_indices = (0, len(validation) // 2, len(validation) - 1)
    selected = [validation[index] for index in selected_indices]

    cases = []
    for position, row in zip(("first", "middle", "last"), selected, strict=True):
        pair_id = row["pair_id"]
        current, current_mask = load_radar(source_path(args.data_root, row["input_npz_path"]))
        target, target_mask = load_radar(source_path(args.data_root, row["target_npz_path"]))
        prediction, prediction_mask, metadata = load_prediction(args.predictions_root, pair_id)
        if not np.array_equal(prediction_mask, current_mask):
            raise ValueError(f"Prediction validity differs from input radar: {pair_id}")
        common = prediction_mask & target_mask
        if not np.any(common):
            raise ValueError(f"No common-valid pixels for qualitative case: {pair_id}")
        error = np.full((256, 256), np.nan, dtype=np.float32)
        error[common] = np.abs(prediction[common] - target[common])
        cases.append({
            "selection_position": position,
            "validation_index": selected_indices[len(cases)],
            "pair_id": pair_id,
            "input_time_utc": row["input_time_utc"],
            "target_time_utc": row["target_time_utc"],
            "current": current,
            "prediction": prediction,
            "target": target,
            "error": error,
            "common_valid_pixels": int(common.sum()),
            "mae_dbz": float(error[common].mean(dtype=np.float64)),
            "rmse_dbz": float(np.sqrt(np.square(
                prediction[common].astype(np.float64) - target[common].astype(np.float64)
            ).mean())),
            "prediction_npz_sha256": metadata["npz_sha256"],
        })

    figure, axes = plt.subplots(3, 4, figsize=(13.5, 11), constrained_layout=True)
    reflectivity_image = None
    error_image = None
    extent = (-200, 200, -200, 200)
    for row_index, case in enumerate(cases):
        for column, key in enumerate(("current", "prediction", "target")):
            reflectivity_image = axes[row_index, column].imshow(
                np.ma.masked_invalid(case[key]), origin="lower", extent=extent,
                cmap="turbo", vmin=-10, vmax=60, interpolation="nearest",
            )
        error_image = axes[row_index, 3].imshow(
            np.ma.masked_invalid(case["error"]), origin="lower", extent=extent,
            cmap="magma", vmin=0, vmax=30, interpolation="nearest",
        )
        axes[row_index, 0].set_ylabel(
            f"{case['selection_position'].title()} validation case\ny (km)"
        )
        axes[row_index, 3].text(
            0.02, 0.02,
            f"MAE {case['mae_dbz']:.2f} dBZ\nRMSE {case['rmse_dbz']:.2f} dBZ",
            transform=axes[row_index, 3].transAxes, color="white", fontsize=8,
            va="bottom", bbox={"facecolor": "black", "alpha": 0.55, "pad": 3},
        )
        for axis in axes[row_index]:
            axis.set_xlabel("x (km)")
            axis.set_aspect("equal")
    for axis, title in zip(
        axes[0], ("Current radar", "+10 min prediction", "+10 min target", "Absolute error"),
        strict=True,
    ):
        axis.set_title(title)
    if reflectivity_image is None or error_image is None:
        raise RuntimeError("No qualitative image was created")
    figure.colorbar(
        reflectivity_image, ax=axes[:, :3], label="Reflectivity display (dBZ)", shrink=0.75,
    )
    figure.colorbar(error_image, ax=axes[:, 3], label="Absolute error display (dBZ)", shrink=0.75)
    figure.suptitle(
        "Deterministic validation examples — display ranges are not processing thresholds",
        fontsize=14,
    )
    figure_path = output / "validation_qualitative_examples.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    summary_cases = [{key: value for key, value in case.items()
                      if not isinstance(value, np.ndarray)} for case in cases]
    summary = {
        "schema_version": "nhabe-qualitative-validation-figure-v1",
        "status": "PASS",
        "selection_policy": "chronological validation indices 0, floor(N/2), N-1",
        "validation_pairs": len(validation),
        "reflectivity_display_range_dbz": [-10, 60],
        "absolute_error_display_range_dbz": [0, 30],
        "display_ranges_are_processing_thresholds": False,
        "model_or_hyperparameter_changed": False,
        "cases": summary_cases,
        "figure_path": str(figure_path),
        "figure_sha256": sha256(figure_path),
    }
    summary_path = output / "validation_qualitative_examples.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    reopened = json.loads(summary_path.read_text(encoding="utf-8"))
    if reopened["figure_sha256"] != sha256(figure_path):
        raise RuntimeError("Qualitative figure failed hash revalidation")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
