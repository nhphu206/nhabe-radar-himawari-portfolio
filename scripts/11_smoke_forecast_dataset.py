#!/usr/bin/env python3
"""Open a few audited forecast pairs and report tensor contracts; no training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from torch.utils.data import DataLoader

from nhabe_forecasting.dataset import ForecastDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--input-mode", choices=("radar", "multimodal"), default="multimodal")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batches", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1 or args.batches < 1:
        parser.error("--batch-size and --batches must be positive")

    dataset = ForecastDataset(
        args.manifest, args.normalization, args.data_root,
        split=args.split, input_mode=args.input_mode,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    opened = []
    first_batch = None
    all_inputs_finite = True
    all_targets_finite = True
    for batch_number, batch in enumerate(loader, 1):
        first_batch = first_batch or batch
        opened.extend(batch["pair_id"])
        all_inputs_finite = all_inputs_finite and bool(batch["inputs"].isfinite().all())
        all_targets_finite = all_targets_finite and bool(batch["target"].isfinite().all())
        if batch_number == args.batches:
            break
    assert first_batch is not None
    result = {
        "status": "PASS",
        "training_performed": False,
        "split": args.split,
        "input_mode": args.input_mode,
        "pairs_in_split": len(dataset),
        "batches_read": min(args.batches, len(loader)),
        "pairs_read": len(opened),
        "channel_names": list(dataset.channel_names),
        "first_input_shape": list(first_batch["inputs"].shape),
        "first_target_shape": list(first_batch["target"].shape),
        "first_input_radar_mask_shape": list(first_batch["input_radar_mask"].shape),
        "first_target_mask_shape": list(first_batch["target_mask"].shape),
        "input_dtype": str(first_batch["inputs"].dtype),
        "target_dtype": str(first_batch["target"].dtype),
        "target_mask_dtype": str(first_batch["target_mask"].dtype),
        "all_inputs_finite": all_inputs_finite,
        "all_targets_finite": all_targets_finite,
        "pair_ids": opened,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
