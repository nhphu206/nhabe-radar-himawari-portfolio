#!/usr/bin/env python3
"""Mở lại toàn bộ mẫu và kiểm tra hợp đồng dữ liệu end-to-end."""

import argparse

import _bootstrap  # noqa: F401
from nhabe_preprocessing.config import load_config
from nhabe_preprocessing.dataset import audit_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    print(audit_dataset(load_config(args.config)))
