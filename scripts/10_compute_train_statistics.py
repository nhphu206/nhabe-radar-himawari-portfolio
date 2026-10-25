#!/usr/bin/env python3
"""Fit normalization from audited train-input sources without rewriting NPZ files."""

import argparse
import json
from pathlib import Path
import platform
import sys
from importlib.metadata import version

import yaml

import _bootstrap
from nhabe_forecasting.normalization import SOURCE_FIELDS, file_sha256, fit_train_statistics
from nhabe_preprocessing.io_utils import atomic_write_csv, atomic_write_json, read_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=_bootstrap.ROOT)
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--config", type=Path, default=_bootstrap.ROOT / "configs/normalization.yaml")
    parser.add_argument("--forecast-config", type=Path, default=_bootstrap.ROOT / "configs/forecasting.yaml")
    parser.add_argument("--output", type=Path,
                        default=_bootstrap.ROOT / "output/forecasting/normalization")
    args = parser.parse_args()
    root = args.data_root.resolve()
    output = args.output.resolve()
    for base in {root, _bootstrap.ROOT}:
        for protected in (base / "data", base / "output/preprocessed", base / "output/dataset",
                          base / "output/inventory", base / "output/audit"):
            if output == protected or protected in output.parents:
                raise ValueError("Normalization outputs must not enter protected source directories")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory; existing artifacts are protected")
    manifest_dir = (args.manifest_dir or root / "output/forecasting/observation_timing").resolve()
    manifest = manifest_dir / "forecast_pairs.csv"
    summary_path = manifest_dir / "pair_summary.json"
    summary = json.loads(summary_path.read_text())
    if (summary["schema_version"] != "nhabe-forecasting-observation-timing-v1"
            or summary["status"] != "OBSERVATION_TIMING_AUDITED"
            or summary["scope"] != "full_exact_candidate_set"):
        raise ValueError("A completed full-scope observation-timing manifest is required")
    manifest_hash = file_sha256(manifest)
    forecast_path = args.forecast_config.resolve()
    if (manifest_hash != summary["artifact_sha256"]["forecast_pairs.csv"]
            or file_sha256(forecast_path) != summary["provenance"]["config_sha256"]):
        raise ValueError("Manifest or forecasting config changed since the timing audit")
    rows = read_csv(manifest)
    if len(rows) != summary["pairs"]:
        raise ValueError("Pair count disagrees with the timing summary")
    policy_path = args.config.resolve()
    policy_hash = file_sha256(policy_path)
    forecast_hash = file_sha256(forecast_path)
    policy = yaml.safe_load(policy_path.read_text())
    forecast = yaml.safe_load(forecast_path.read_text())
    result, sources = fit_train_statistics(
        rows, root, forecast["splits"], policy,
        progress=lambda msg: print(msg, flush=True),
    )
    if file_sha256(manifest) != manifest_hash:
        raise ValueError("Manifest changed during normalization fitting")
    if file_sha256(policy_path) != policy_hash or file_sha256(forecast_path) != forecast_hash:
        raise ValueError("Configuration changed during normalization fitting")
    result["provenance"] = {
        "manifest_path": str(manifest), "manifest_sha256": manifest_hash,
        "timing_summary_sha256": file_sha256(summary_path),
        "policy_path": str(policy_path), "policy_sha256": policy_hash,
        "forecast_config_sha256": forecast_hash, "numpy_version": version("numpy"),
        "python_version": platform.python_version(), "python_executable": sys.executable,
        "code_sha256": {p.name: file_sha256(p) for p in (
            Path(__file__), _bootstrap.ROOT / "src/nhabe_forecasting/normalization.py",
            _bootstrap.ROOT / "src/nhabe_forecasting/splits.py")},
    }
    atomic_write_csv(output / "normalization_sources.csv", sources, SOURCE_FIELDS)
    result["provenance"]["source_manifest_sha256"] = file_sha256(output / "normalization_sources.csv")
    atomic_write_json(output / "normalization.json", result)
    reopened = json.loads((output / "normalization.json").read_text())
    if reopened != result or read_csv(output / "normalization_sources.csv") != sources:
        raise ValueError("Normalization artifacts failed roundtrip validation")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
