#!/usr/bin/env python3
"""Run one real-data U-Net forward/backward check without an optimizer step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhabe_forecasting.dataset import ForecastDataset
from nhabe_forecasting.evaluation import masked_mse
from nhabe_forecasting.model import MODEL_CONFIG, SmallUNet2D, trainable_parameter_count


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/model.yaml")
    parser.add_argument("--input-mode", choices=("radar", "multimodal"), default="multimodal")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config != MODEL_CONFIG:
        raise ValueError("Unsupported model configuration")
    expected_channels = int(config["input_channels"][args.input_mode])
    dataset = ForecastDataset(
        args.manifest, args.normalization, args.data_root,
        split="train", input_mode=args.input_mode,
    )
    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)))
    if batch["inputs"].shape[1] != expected_channels:
        raise ValueError("Dataset channels differ from model configuration")

    torch.manual_seed(2025)
    device = choose_device(args.device)
    model = SmallUNet2D(
        in_channels=expected_channels,
        base_channels=int(config["base_channels"]),
        out_channels=int(config["output_channels"]),
    ).to(device)
    inputs = batch["inputs"].to(device)
    target = batch["target"].to(device)
    target_mask = batch["target_mask"].to(device)
    prediction = model(inputs)
    loss = masked_mse(prediction, target, target_mask)
    loss.backward()
    if device.type == "mps":
        torch.mps.synchronize()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    finite_gradients = all(gradient is not None and bool(torch.isfinite(gradient).all())
                           for gradient in gradients)
    gradient_norm = sum(float(gradient.detach().float().square().sum().cpu())
                        for gradient in gradients if gradient is not None) ** 0.5
    result = {
        "status": "PASS" if finite_gradients and bool(torch.isfinite(loss)) else "FAIL",
        "training_performed": False,
        "optimizer_created": False,
        "optimizer_step_performed": False,
        "device": str(device),
        "input_mode": args.input_mode,
        "channel_names": list(dataset.channel_names),
        "input_shape": list(inputs.shape),
        "prediction_shape": list(prediction.shape),
        "target_shape": list(target.shape),
        "target_mask_shape": list(target_mask.shape),
        "trainable_parameters": trainable_parameter_count(model),
        "masked_mse_normalized_units": float(loss.detach().cpu()),
        "parameters_with_gradients": len(gradients),
        "all_gradients_finite": finite_gradients,
        "global_gradient_l2_norm": gradient_norm,
        "pair_ids": list(batch["pair_id"]),
    }
    if result["status"] != "PASS":
        raise RuntimeError("U-Net smoke test produced a non-finite value")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
