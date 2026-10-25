"""Masked regression metrics for reflectivity forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


EVALUATION_POLICY = {
    "schema_version": "nhabe-forecast-evaluation-policy-v1",
    "primary_domain": "input_radar_valid_and_target_valid",
    "aggregation": "valid_pixel_weighted",
    "units": "dBZ",
    "metrics": ["MAE", "RMSE"],
    "baseline": "persistence_current_radar",
    "model_selection_split": "validation",
    "test_policy": "locked_until_model_selection_complete",
    "reflectivity_threshold": None,
}


def cpu_float64(tensor: torch.Tensor) -> torch.Tensor:
    """Transfer first, then cast; combined MPS transfer/cast can corrupt values."""
    return tensor.detach().cpu().to(torch.float64)


def _masked_error(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("Prediction, target and mask must have identical shapes")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise ValueError("Prediction and target must be floating-point tensors")
    if mask.dtype != torch.bool:
        raise ValueError("Metric mask must be boolean")
    if not bool(mask.any()):
        raise ValueError("Metric mask contains no valid pixels")
    selected_prediction = prediction[mask]
    selected_target = target[mask]
    if not bool(torch.isfinite(selected_prediction).all()) or not bool(torch.isfinite(selected_target).all()):
        raise ValueError("Valid metric pixels must be finite")
    return selected_prediction - selected_target


def masked_mae(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Mean absolute error over true mask pixels."""
    return _masked_error(prediction, target, mask).abs().mean()


def masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error over true mask pixels; suitable as a differentiable loss."""
    return _masked_error(prediction, target, mask).square().mean()


def masked_rmse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Root mean squared error over true mask pixels."""
    return masked_mse(prediction, target, mask).sqrt()


@dataclass
class RegressionAccumulator:
    """Accumulate pixel-weighted errors across batches using float64 sums."""

    valid_pixels: int = 0
    absolute_error_sum: float = 0.0
    squared_error_sum: float = 0.0

    def update(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    ) -> None:
        error = cpu_float64(_masked_error(prediction, target, mask))
        self.valid_pixels += int(error.numel())
        self.absolute_error_sum += float(error.abs().sum())
        self.squared_error_sum += float(error.square().sum())

    def result(self) -> dict[str, Any]:
        if self.valid_pixels == 0:
            raise ValueError("No valid pixels accumulated")
        mae = self.absolute_error_sum / self.valid_pixels
        rmse = (self.squared_error_sum / self.valid_pixels) ** 0.5
        return {
            "valid_pixels": self.valid_pixels,
            "mae_dbz": mae,
            "rmse_dbz": rmse,
            "absolute_error_sum": self.absolute_error_sum,
            "squared_error_sum": self.squared_error_sum,
        }
