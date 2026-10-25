"""Reusable training primitives for controlled Small U-Net experiments."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Mapping, Any

import numpy as np
import torch
from torch import nn

from .evaluation import cpu_float64, masked_mse


TRAINING_CONFIG = {
    "schema_version": "nhabe-small-unet-training-v1",
    "seed": 2025,
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "gradient_clip_norm": 1.0,
    "loss": "masked_mse_target_valid_normalized",
    "num_workers": 0,
    "official": {
        "input_modes": ["radar", "multimodal"],
        "batch_size": 4,
        "max_epochs": 40,
        "early_stopping_patience": 7,
        "minimum_improvement_rmse_dbz": 0.001,
        "model_selection_metric": "validation_rmse_dbz_common_mask",
        "checkpoint": "best_validation_epoch",
    },
    "overfit": {
        "input_mode": "multimodal",
        "pairs": 16,
        "batch_size": 2,
        "epochs": 25,
        "maximum_final_to_initial_loss_ratio": 0.5,
        "checkpoint": "none",
    },
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if requested not in {"cpu", "mps"}:
        raise ValueError("device must be auto, cpu or mps")
    return torch.device(requested)


def make_optimizer(model: nn.Module, config: Mapping[str, Any]) -> torch.optim.Optimizer:
    if config["optimizer"] != "AdamW":
        raise ValueError("Only the approved AdamW optimizer is supported")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )


def overfit_gate(initial_loss: float, final_loss: float, maximum_ratio: float) -> tuple[bool, float]:
    if not all(np.isfinite(value) for value in (initial_loss, final_loss, maximum_ratio)):
        raise ValueError("Overfit gate values must be finite")
    if initial_loss <= 0 or final_loss < 0 or not 0 < maximum_ratio < 1:
        raise ValueError("Invalid overfit gate values")
    ratio = final_loss / initial_loss
    return ratio <= maximum_ratio, ratio


def save_model_checkpoint(
    path: Path, model: nn.Module, mode: str, epoch: int,
    validation: Mapping[str, Any], provenance: Mapping[str, Any],
) -> None:
    """Atomically save a weights-only-compatible checkpoint."""
    document = {
        "schema_version": "nhabe-small-unet-checkpoint-v1",
        "input_mode": str(mode),
        "epoch": int(epoch),
        "validation": dict(validation),
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "provenance": dict(provenance),
    }
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(document, temporary)
    temporary.replace(path)


def load_model_checkpoint(path: Path) -> dict[str, Any]:
    document = torch.load(path, map_location="cpu", weights_only=True)
    if document.get("schema_version") != "nhabe-small-unet-checkpoint-v1":
        raise ValueError("Unsupported checkpoint schema")
    return document


def _batch_tensors(batch: Mapping[str, Any], device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        batch["inputs"].to(device),
        batch["target"].to(device),
        batch["target_mask"].to(device),
    )


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float,
) -> float:
    """Train one epoch and return global valid-pixel normalized MSE."""
    model.train()
    squared_error_sum = 0.0
    valid_pixels = 0
    for batch in loader:
        inputs, target, target_mask = _batch_tensors(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs)
        loss = masked_mse(prediction, target, target_mask)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError("Non-finite gradient norm")
        optimizer.step()
        errors = cpu_float64((prediction.detach() - target)[target_mask])
        squared_error_sum += float(errors.square().sum())
        valid_pixels += int(errors.numel())
    if valid_pixels == 0:
        raise ValueError("Training epoch contains no valid target pixels")
    return squared_error_sum / valid_pixels


@torch.no_grad()
def evaluate_normalized_mse(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
) -> float:
    """Evaluate global valid-target-pixel MSE without changing weights."""
    model.eval()
    squared_error_sum = 0.0
    valid_pixels = 0
    for batch in loader:
        inputs, target, target_mask = _batch_tensors(batch, device)
        prediction = model(inputs)
        errors = (prediction - target)[target_mask]
        if not bool(torch.isfinite(errors).all()):
            raise RuntimeError("Non-finite evaluation error")
        errors = cpu_float64(errors)
        squared_error_sum += float(errors.square().sum())
        valid_pixels += int(errors.numel())
    if valid_pixels == 0:
        raise ValueError("Evaluation contains no valid target pixels")
    return squared_error_sum / valid_pixels


@torch.no_grad()
def evaluate_forecast(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    radar_std_dbz: float,
) -> dict[str, float | int]:
    """Return target-mask loss and common-mask physical validation metrics."""
    if not np.isfinite(radar_std_dbz) or radar_std_dbz <= 0:
        raise ValueError("Radar standard deviation must be positive and finite")
    model.eval()
    target_squared_sum = 0.0
    target_valid_pixels = 0
    common_absolute_sum = 0.0
    common_squared_sum = 0.0
    common_valid_pixels = 0
    total_pixels = 0
    for batch in loader:
        inputs, target, target_mask = _batch_tensors(batch, device)
        input_mask = batch["input_radar_mask"].to(device)
        prediction = model(inputs)
        target_error = (prediction - target)[target_mask]
        if not bool(torch.isfinite(target_error).all()):
            raise RuntimeError("Non-finite validation error")
        target_error_cpu = cpu_float64(target_error)
        target_squared_sum += float(target_error_cpu.square().sum())
        target_valid_pixels += int(target_error_cpu.numel())

        common_mask = target_mask & input_mask
        common_error_dbz = cpu_float64((prediction - target)[common_mask]) * radar_std_dbz
        if common_error_dbz.numel() == 0:
            raise ValueError("Validation batch has no common valid radar pixels")
        common_absolute_sum += float(common_error_dbz.abs().sum())
        common_squared_sum += float(common_error_dbz.square().sum())
        common_valid_pixels += int(common_error_dbz.numel())
        total_pixels += int(common_mask.numel())
    if target_valid_pixels == 0 or common_valid_pixels == 0:
        raise ValueError("Validation loader contains no valid pixels")
    return {
        "normalized_mse_target_mask": target_squared_sum / target_valid_pixels,
        "target_valid_pixels": target_valid_pixels,
        "common_valid_pixels": common_valid_pixels,
        "common_valid_percent": 100.0 * common_valid_pixels / total_pixels,
        "mae_dbz_common_mask": common_absolute_sum / common_valid_pixels,
        "rmse_dbz_common_mask": (common_squared_sum / common_valid_pixels) ** 0.5,
    }
