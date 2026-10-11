"""Đọc cấu hình và dựng đường dẫn độc lập với thư mục hiện hành."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(
    os.environ.get("NHABE_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Đọc cấu hình YAML và kiểm tra các nhóm bắt buộc."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    required = {
        "paths", "radar", "grid", "radar_processing", "himawari",
        "time_alignment", "sample_contract", "expected_results",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"Cấu hình thiếu nhóm bắt buộc: {missing}")
    value["_config_path"] = str(config_path)
    return value


def project_path(value: str | Path) -> Path:
    """Chuyển đường dẫn tương đối trong cấu hình thành đường dẫn tuyệt đối."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
