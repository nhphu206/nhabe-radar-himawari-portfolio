#!/usr/bin/env python3
"""Ghép trung điểm volume radar với thời gian dòng Himawari tại Nhà Bè."""

import argparse
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
from nhabe_preprocessing.config import load_config, project_path
from nhabe_preprocessing.inventory import (
    TIME_MANIFEST_FIELDS, adopt_verified_legacy_manifest, build_time_manifest,
    collect_himawari_timings,
)
from nhabe_preprocessing.io_utils import atomic_write_csv, read_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument(
        "--adopt-verified-manifest", type=Path,
        help="Di trú manifest cũ đã audit thay vì đọc lại toàn bộ headers",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.adopt_verified_manifest:
        rows = adopt_verified_legacy_manifest(args.adopt_verified_manifest)
    else:
        inventory = project_path(config["paths"]["inventory_root"]) / "radar_inventory.csv"
        rows = build_time_manifest(read_csv(inventory), collect_himawari_timings(config), config)
    destination = project_path(config["paths"]["inventory_root"]) / "time_manifest.csv"
    atomic_write_csv(destination, rows, TIME_MANIFEST_FIELDS)
    counts = Counter(row["time_match_status"] for row in rows)
    print(f"Đã ghi {len(rows)} dòng: {destination}")
    print("Trạng thái:", dict(counts))


if __name__ == "__main__":
    main()
