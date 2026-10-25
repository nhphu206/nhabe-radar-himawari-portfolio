"""Locked-test evaluation helpers shared by persistence and trained models."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn

from .evaluation import cpu_float64


@torch.no_grad()
def evaluate_method(
    method: str,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    radar_std_dbz: float,
    model: nn.Module | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one frozen method on the common input-radar/target validity domain."""
    if method == "persistence":
        if model is not None:
            raise ValueError("Persistence must not receive a trained model")
    elif model is None:
        raise ValueError("A learned method requires a model")
    if not np.isfinite(radar_std_dbz) or radar_std_dbz <= 0:
        raise ValueError("Radar standard deviation must be positive and finite")
    if model is not None:
        model.eval()

    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    evaluable_pixels = 0
    input_valid_pixels = 0
    target_valid_pixels = 0
    total_pixels = 0
    sample_rows: list[dict[str, Any]] = []

    for batch in loader:
        inputs = batch["inputs"].to(device)
        target = batch["target"].to(device)
        input_mask = batch["input_radar_mask"].to(device)
        target_mask = batch["target_mask"].to(device)
        prediction = inputs[:, :1] if method == "persistence" else model(inputs)
        if prediction.shape != target.shape or not bool(torch.isfinite(prediction).all()):
            raise RuntimeError("Final prediction shape or values are invalid")
        common_mask = input_mask & target_mask
        total_pixels += int(common_mask.numel())
        input_valid_pixels += int(input_mask.sum().cpu())
        target_valid_pixels += int(target_mask.sum().cpu())

        for index, pair_id in enumerate(batch["pair_id"]):
            error_dbz = cpu_float64((prediction[index] - target[index])[common_mask[index]])
            error_dbz *= radar_std_dbz
            if error_dbz.numel() == 0 or not bool(torch.isfinite(error_dbz).all()):
                raise RuntimeError("A test pair has no finite common-valid error")
            count = int(error_dbz.numel())
            absolute = float(error_dbz.abs().sum())
            squared = float(error_dbz.square().sum())
            absolute_error_sum += absolute
            squared_error_sum += squared
            evaluable_pixels += count
            sample_rows.append({
                "pair_id": pair_id,
                "input_time_utc": batch["input_time_utc"][index],
                "target_time_utc": batch["target_time_utc"][index],
                "method": method,
                "evaluable_pixels": count,
                "total_pixels": int(common_mask[index].numel()),
                "evaluable_percent": 100.0 * count / common_mask[index].numel(),
                "mae_dbz": absolute / count,
                "rmse_dbz": (squared / count) ** 0.5,
            })
    if not sample_rows or evaluable_pixels == 0:
        raise ValueError("Final evaluation contains no samples or valid pixels")
    summary = {
        "method": method,
        "pairs": len(sample_rows),
        "total_grid_pixels": total_pixels,
        "input_valid_pixels": input_valid_pixels,
        "target_valid_pixels": target_valid_pixels,
        "evaluable_pixels": evaluable_pixels,
        "evaluable_percent": 100.0 * evaluable_pixels / total_pixels,
        "mae_dbz": absolute_error_sum / evaluable_pixels,
        "rmse_dbz": (squared_error_sum / evaluable_pixels) ** 0.5,
    }
    return summary, sample_rows


def relative_reduction(reference: float, candidate: float) -> float:
    if not np.isfinite(reference) or not np.isfinite(candidate) or reference <= 0:
        raise ValueError("Metric values must be finite and reference must be positive")
    return 100.0 * (reference - candidate) / reference
