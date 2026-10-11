#!/usr/bin/env python3
"""Tải đúng 5 băng x 2 segment cho các chu kỳ đã ghép của một ngày."""

import argparse
import asyncio
from collections import Counter

import _bootstrap  # noqa: F401
from nhabe_preprocessing.config import load_config, project_path
from nhabe_preprocessing.downloader import download_all, planned_objects
from nhabe_preprocessing.io_utils import atomic_write_csv, atomic_write_json, read_csv


FIELDS = ("date", "cycle", "band", "segment", "name", "key", "url", "path",
          "status", "size_bytes", "attempts", "error")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("date", help="YYYYMMDD")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    inventory = project_path(config["paths"]["inventory_root"]) / "time_manifest.csv"
    rows = read_csv(inventory)
    cycles = {row["matched_himawari_cycle"] for row in rows
              if row["date"] == args.date and row["time_match_status"] == "MATCHED"}
    staging = project_path(config["paths"]["himawari_staging_root"])
    source_root = staging
    if args.date == "20250801":
        source_root = project_path(config["paths"]["himawari_day01_root"]).parent
    objects = planned_objects(args.date, cycles, source_root, config)
    if args.date == "20250801":
        missing = [str(item["path"]) for item in objects if not item["path"].is_file()]
        if missing:
            raise FileNotFoundError(
                "Ngày 01 nằm trong data/raw chỉ đọc; không tự tải bù vào raw. "
                f"Thiếu {len(missing)} tệp, ví dụ: {missing[0]}"
            )
    results = asyncio.run(download_all(objects, args.workers))
    day_root = staging / args.date
    atomic_write_csv(day_root / "download_manifest.csv", results, FIELDS)
    counts = Counter(row["status"] for row in results)
    summary = {"date": args.date, "objects": len(results), "status_counts": dict(counts),
               "status": "PASS" if not counts["FAILED"] else "FAIL"}
    atomic_write_json(day_root / "download_summary.json", summary)
    print(summary)
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
