#!/usr/bin/env python3
"""Tạo hoặc tiếp tục các mẫu radar--Himawari của một ngày."""

import argparse

import _bootstrap  # noqa: F401
from nhabe_preprocessing.config import load_config
from nhabe_preprocessing.preprocess import process_day


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("date", help="YYYYMMDD")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    print(process_day(args.date, load_config(args.config)))


if __name__ == "__main__":
    main()
