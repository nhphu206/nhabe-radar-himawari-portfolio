#!/usr/bin/env python3
"""Run the frozen multimodal model on one audited preprocessed forecast input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhabe_forecasting.inference import run_multimodal_inference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs/model.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    result = run_multimodal_inference(
        data_root=args.data_root.resolve(),
        manifest_path=args.manifest.resolve(),
        normalization_path=args.normalization.resolve(),
        model_config_path=args.model_config.resolve(),
        checkpoint_path=args.checkpoint.resolve(),
        pair_id=args.pair_id,
        output_dir=args.output,
        requested_device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
