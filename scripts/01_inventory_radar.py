#!/usr/bin/env python3
"""Đọc metadata của các volume radar và phân loại scan signature A/B."""

import argparse

import _bootstrap  # noqa: F401
from nhabe_preprocessing.config import load_config, project_path
from nhabe_preprocessing.inventory import RADAR_INVENTORY_FIELDS
from nhabe_preprocessing.io_utils import atomic_write_csv
from nhabe_preprocessing.radar import inventory_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--date", help="YYYYMMDD; bỏ trống để đọc toàn tháng")
    parser.add_argument("--limit", type=int, help="Giới hạn dùng cho kiểm thử")
    args = parser.parse_args()
    config = load_config(args.config)
    root = project_path(config["paths"]["radar_root"])
    days = [args.date[-2:]] if args.date else [f"{day:02d}" for day in range(1, 32)]
    paths = sorted(path for day in days for path in (root / day).glob("NHB*.RAW*"))
    if args.limit is not None:
        paths = paths[:args.limit]
    rows = []
    for index, path in enumerate(paths, 1):
        rows.append(inventory_row(path, config))
        if index % 100 == 0 or index == len(paths):
            print(f"Đã đọc {index}/{len(paths)} radar volumes", flush=True)
    destination = project_path(config["paths"]["inventory_root"]) / "radar_inventory.csv"
    atomic_write_csv(destination, rows, RADAR_INVENTORY_FIELDS)
    print(f"Đã ghi {len(rows)} dòng: {destination}")


if __name__ == "__main__":
    main()
