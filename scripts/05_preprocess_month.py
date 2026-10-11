#!/usr/bin/env python3
"""Điều phối tiền xử lý tuần tự toàn tháng, có khả năng tiếp tục an toàn."""

import argparse

import _bootstrap  # noqa: F401
from nhabe_preprocessing.config import load_config
from nhabe_preprocessing.dataset import audit_dataset, build_index
from nhabe_preprocessing.preprocess import process_day


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--start-day", type=int, default=1)
    parser.add_argument("--end-day", type=int, default=31)
    args = parser.parse_args()
    config = load_config(args.config)
    for day in range(args.start_day, args.end_day + 1):
        date = f"202508{day:02d}"
        summary = process_day(date, config)
        print(date, summary["status"], summary["valid_samples"], flush=True)
    if args.start_day == 1 and args.end_day == 31:
        print(build_index(config))
        print(audit_dataset(config))


if __name__ == "__main__":
    main()
