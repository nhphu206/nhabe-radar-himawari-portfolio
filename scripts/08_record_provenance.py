#!/usr/bin/env python3
"""Ghi phiên bản Python/thư viện, Git commit và checksum cấu hình."""

import argparse

import _bootstrap  # noqa: F401
from nhabe_preprocessing.config import load_config
from nhabe_preprocessing.provenance import record_provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()
    print(record_provenance(load_config(args.config)))
