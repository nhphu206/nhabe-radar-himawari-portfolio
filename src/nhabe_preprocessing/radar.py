"""Đọc, mô tả, phân loại và nội suy radar Nhà Bè."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyart

from .common_grid import CommonGrid


UTC = timezone.utc


def parse_time_reference(units: str) -> datetime:
    prefix = "seconds since "
    if not units.startswith(prefix):
        raise ValueError(f"Đơn vị thời gian radar chưa được hỗ trợ: {units}")
    value = datetime.fromisoformat(units[len(prefix):].replace("Z", "+00:00"))
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def scan_metadata(radar: Any) -> dict[str, Any]:
    """Lấy metadata hình học và thời gian cần cho phân loại/chỉ mục."""
    ranges = np.asarray(radar.range["data"], dtype=np.float64)
    spacings = np.diff(ranges)
    spacing = float(np.median(spacings))
    offsets = np.asarray(radar.time["data"], dtype=np.float64)
    reference = parse_time_reference(radar.time["units"])
    start = reference + timedelta(seconds=float(np.nanmin(offsets)))
    end = reference + timedelta(seconds=float(np.nanmax(offsets)))
    midpoint = start + (end - start) / 2
    return {
        "scan_type": radar.scan_type,
        "nsweeps": int(radar.nsweeps),
        "nrays": int(radar.nrays),
        "ngates": int(radar.ngates),
        "gate_spacing_constant": bool(np.allclose(spacings, spacing)),
        "gate_spacing_m": spacing,
        "max_range_m": float(ranges[-1]),
        "fixed_angles_deg": [float(value) for value in radar.fixed_angle["data"]],
        "fields": sorted(radar.fields),
        "observation_start_utc": iso_utc(start),
        "observation_end_utc": iso_utc(end),
        "representative_time_utc": iso_utc(midpoint),
        "duration_seconds": float((end - start).total_seconds()),
    }


def classify_signature(metadata: dict[str, Any], config: dict[str, Any]) -> str:
    """Phân loại A/B chỉ từ scan metadata; trường không khớp trả UNKNOWN."""
    tolerances = config["radar"]["classifier_tolerances"]
    matches: list[str] = []
    for label, expected in config["radar"]["signatures"].items():
        angles = np.asarray(metadata["fixed_angles_deg"], dtype=np.float64)
        expected_angles = np.asarray(expected["fixed_angles_deg"], dtype=np.float64)
        same = (
            metadata["scan_type"] == expected["scan_type"]
            and metadata["nsweeps"] == int(expected["nsweeps"])
            and metadata["ngates"] == int(expected["ngates"])
            and metadata["gate_spacing_constant"]
            and np.isclose(
                metadata["gate_spacing_m"], expected["gate_spacing_m"],
                atol=tolerances["gate_spacing_absolute_m"], rtol=0.0,
            )
            and np.isclose(
                metadata["max_range_m"], expected["max_range_m"],
                atol=tolerances["max_range_absolute_m"], rtol=0.0,
            )
            and angles.shape == expected_angles.shape
            and np.allclose(
                angles, expected_angles,
                atol=tolerances["fixed_angle_absolute_deg"], rtol=0.0,
            )
            and metadata["fields"] == sorted(expected["fields"])
        )
        if same:
            matches.append(label)
    return matches[0] if len(matches) == 1 else "UNKNOWN"


def read_radar(path: Path) -> Any:
    """Đọc SIGMET RAW; hàm không thực hiện bất kỳ thao tác ghi nào lên nguồn."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return pyart.io.read_sigmet(path)


def inventory_row(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    radar = read_radar(path)
    metadata = scan_metadata(radar)
    signature = classify_signature(metadata, config)
    return {
        "date": "20" + path.name[3:9],
        "radar_filename": path.name,
        "radar_path": str(path),
        "signature": signature,
        **metadata,
        "fixed_angles_deg": ",".join(f"{value:.9f}" for value in metadata["fixed_angles_deg"]),
        "fields": ",".join(metadata["fields"]),
    }


def make_grid(radar: Any, common_grid: CommonGrid, config: dict[str, Any]) -> Any:
    radar_config = config["radar_processing"]
    grid_config = config["grid"]
    limits = (float(common_grid.x.min()), float(common_grid.x.max()))
    z = float(radar_config["grid_z_m_above_grid_origin"])
    return pyart.map.grid_from_radars(
        (radar,),
        grid_shape=tuple(radar_config["grid_shape"]),
        grid_limits=((z, z), limits, limits),
        fields=[config["radar"]["field"]],
        grid_origin=(grid_config["center"]["latitude"], grid_config["center"]["longitude"]),
        grid_origin_alt=float(radar.altitude["data"][0]),
        grid_projection=grid_config["projection"],
        gridding_algo=radar_config["gridding_algo"],
        copy_field_dtypes=True,
        gatefilters=False,
        map_roi=True,
        weighting_function=radar_config["weighting_function"],
        toa=radar_config["toa_m"],
        roi_func=radar_config["roi_func"],
        constant_roi=None,
        z_factor=radar_config["z_factor"],
        xy_factor=radar_config["xy_factor"],
        min_radius=radar_config["min_radius_m"],
        h_factor=tuple(radar_config["h_factor"]),
        nb=radar_config["nb"],
        bsp=radar_config["bsp"],
        dist_factor=tuple(radar_config["dist_factor"]),
    )


def extract_cappi(
    radar: Any,
    metadata: dict[str, Any],
    common_grid: CommonGrid,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tạo CAPPI, mặt nạ dữ liệu và mặt nạ miền khoảng cách."""
    grid = make_grid(radar, common_grid, config)
    raw = grid.fields[config["radar"]["field"]]["data"][0]
    aligned = np.ma.array(np.flipud(raw), copy=False)
    values = np.asarray(aligned.filled(np.nan), dtype=np.float32)
    valid = (~np.ma.getmaskarray(aligned) & np.isfinite(values)).astype(bool)
    values[~valid] = np.nan
    support = (common_grid.radius <= float(metadata["max_range_m"])).astype(bool)
    return values, valid, support
