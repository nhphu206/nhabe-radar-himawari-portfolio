#!/usr/bin/env python3
"""Tạo index, daily summary và dataset card từ manifest từng ngày."""

import argparse

import _bootstrap  # noqa: F401
from nhabe_preprocessing.config import load_config
from nhabe_preprocessing.dataset import build_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    print(build_index(load_config(args.config)))
