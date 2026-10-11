"""Đặc tả và kiểm tra các mảng của một mẫu radar--Himawari."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ARRAY_NAMES = (
    "radar_reflectivity_dbz",
    "radar_data_valid_mask",
    "radar_range_support_mask",
    "himawari_bt",
    "himawari_band_valid_mask",
    "himawari_all_bands_valid_mask",
)


def validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    missing = sorted(set(ARRAY_NAMES) - set(arrays))
    if missing:
        raise ValueError(f"Mẫu thiếu mảng: {missing}")
    radar = np.asarray(arrays["radar_reflectivity_dbz"])
    radar_valid = np.asarray(arrays["radar_data_valid_mask"])
    range_support = np.asarray(arrays["radar_range_support_mask"])
    satellite = np.asarray(arrays["himawari_bt"])
    band_valid = np.asarray(arrays["himawari_band_valid_mask"])
    all_valid = np.asarray(arrays["himawari_all_bands_valid_mask"])
    if radar.shape != (256, 256):
        raise ValueError(f"Sai shape radar: {radar.shape}")
    if radar_valid.shape != radar.shape or range_support.shape != radar.shape:
        raise ValueError("Mặt nạ radar không cùng shape với CAPPI")
    if satellite.shape != (5, 256, 256) or band_valid.shape != satellite.shape:
        raise ValueError("Sai shape Himawari hoặc mặt nạ từng băng")
    if all_valid.shape != (256, 256):
        raise ValueError("Sai shape mặt nạ năm băng")
    if radar.dtype != np.float32 or satellite.dtype != np.float32:
        raise ValueError("Các trường vật lý phải dùng float32")
    for name, value in (
        ("radar_data_valid_mask", radar_valid),
        ("radar_range_support_mask", range_support),
        ("himawari_band_valid_mask", band_valid),
        ("himawari_all_bands_valid_mask", all_valid),
    ):
        if value.dtype != np.bool_:
            raise ValueError(f"{name} phải dùng bool")
    if not np.array_equal(np.isfinite(radar), radar_valid):
        raise ValueError("NaN radar không khớp mặt nạ")
    if not np.array_equal(np.isfinite(satellite), band_valid):
        raise ValueError("NaN Himawari không khớp mặt nạ")
    if not np.array_equal(all_valid, np.all(band_valid, axis=0)):
        raise ValueError("Mặt nạ năm băng không phải phép AND")


def validate_sample_files(
    npz_path: Path,
    json_path: Path,
    expected_radar_filename: str | None = None,
) -> dict[str, Any]:
    if not npz_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"Thiếu cặp mẫu: {npz_path}, {json_path}")
    with np.load(npz_path, allow_pickle=False) as arrays:
        validate_arrays(arrays)
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if expected_radar_filename and metadata.get("radar_filename") != expected_radar_filename:
        raise ValueError("Tên radar trong JSON không khớp")
    if metadata.get("time_match_status") != "MATCHED":
        raise ValueError("Mẫu không có trạng thái MATCHED")
    if float(metadata.get("absolute_delta_seconds", float("inf"))) > 180.0:
        raise ValueError("Mẫu vượt dung sai thời gian")
    return metadata
