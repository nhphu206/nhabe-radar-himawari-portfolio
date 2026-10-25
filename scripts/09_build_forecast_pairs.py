#!/usr/bin/env python3
"""Create a new forecasting manifest; never modify the preprocessing dataset."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

import _bootstrap
from nhabe_forecasting.timeline import (
    AUDITED_PAIR_FIELDS, HEADER_FIELDS, EXCLUSION_FIELDS, build_audited_pairs,
)
from nhabe_preprocessing.config import load_config
from nhabe_preprocessing.io_utils import atomic_write_csv, atomic_write_json, read_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=_bootstrap.ROOT,
                        help="Root against which index NPZ paths are resolved")
    parser.add_argument("--index", type=Path, help="Defaults to DATA_ROOT/output/dataset/index.csv")
    parser.add_argument("--config", type=Path, default=_bootstrap.ROOT / "configs/forecasting.yaml")
    parser.add_argument("--pipeline-config", type=Path, default=_bootstrap.ROOT / "configs/pipeline.yaml")
    parser.add_argument("--limit", type=int, help="Audit only the first N exact candidates for a preview")
    parser.add_argument("--output", type=Path,
                        default=_bootstrap.ROOT / "output/forecasting/observation_timing")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    index = (args.index or data_root / "output/dataset/index.csv").resolve()
    config_path = args.config.resolve()
    output = args.output.resolve()
    for root in {data_root, _bootstrap.ROOT}:
        for protected in (root / "data", root / "output/preprocessed", root / "output/dataset",
                          root / "output/inventory", root / "output/audit"):
            if output == protected or protected in output.parents:
                raise ValueError("Forecast outputs must not enter protected input/preprocessing directories")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty forecasting output directory; existing artifacts are protected")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pipeline_path = args.pipeline_config.resolve()
    pipeline = load_config(pipeline_path)
    index_hash = hashlib.sha256(index.read_bytes()).hexdigest()
    pairs, summary, headers, excluded = build_audited_pairs(
        read_csv(index), data_root, config, pipeline, limit=args.limit,
        progress=lambda message: print(message, flush=True),
    )
    if hashlib.sha256(index.read_bytes()).hexdigest() != index_hash:
        raise ValueError("Source index changed during audit")
    summary["provenance"] = {
        "index_path": str(index),
        "index_sha256": index_hash,
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "data_root": str(data_root),
        "pipeline_config_path": str(pipeline_path),
        "pipeline_config_sha256": hashlib.sha256(pipeline_path.read_bytes()).hexdigest(),
    }
    atomic_write_csv(output / "forecast_pairs.csv", pairs, AUDITED_PAIR_FIELDS)
    reopened = read_csv(output / "forecast_pairs.csv")
    if len(reopened) != len(pairs) or any(
        saved != {k: str(pair[k]) for k in AUDITED_PAIR_FIELDS}
        for saved, pair in zip(reopened, pairs)
    ):
        raise ValueError("Manifest roundtrip failed")
    atomic_write_csv(output / "satellite_header_audit.csv", headers, HEADER_FIELDS)
    atomic_write_csv(output / "excluded_pairs.csv", excluded, EXCLUSION_FIELDS)
    for name in ("forecast_pairs.csv", "satellite_header_audit.csv", "excluded_pairs.csv"):
        summary.setdefault("artifact_sha256", {})[name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    atomic_write_json(output / "pair_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print("Three real pair examples:")
    for pair in pairs[:3]:
        print(pair["input_sample_id"], pair["input_time_utc"], "->",
              pair["target_sample_id"], pair["target_time_utc"])


if __name__ == "__main__":
    main()
