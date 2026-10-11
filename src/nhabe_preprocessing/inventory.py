"""Dựng inventory radar và bảng thời gian Himawari từ nguồn cục bộ."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .config import project_path
from .himawari import hsd_filename, read_local_observation_time
from .io_utils import read_csv
from .time_matching import match_all


RADAR_INVENTORY_FIELDS = (
    "date", "radar_filename", "radar_path", "signature", "scan_type",
    "nsweeps", "nrays", "ngates", "gate_spacing_constant", "gate_spacing_m",
    "max_range_m", "fixed_angles_deg", "fields", "observation_start_utc",
    "observation_end_utc", "representative_time_utc", "duration_seconds",
)

TIME_MANIFEST_FIELDS = (
    "date", "radar_filename", "radar_signature", "radar_time_start_utc",
    "radar_time_end_utc", "radar_representative_time_utc",
    "matched_himawari_cycle", "himawari_nominal_time_utc",
    "himawari_local_line_time_utc", "signed_delta_seconds",
    "absolute_delta_seconds", "time_match_status",
)


def radar_rows_for_matching(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chuẩn hóa inventory radar thành đầu vào của thuật toán ghép thời gian."""
    return [{
        "date": row["date"],
        "radar_filename": row["radar_filename"],
        "radar_signature": row["signature"],
        "radar_time_start_utc": row["observation_start_utc"],
        "radar_time_end_utc": row["observation_end_utc"],
        "radar_representative_time_utc": row["representative_time_utc"],
    } for row in rows]


def _timing_from_cache(path: Path) -> dict[str, Any] | None:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("manifest", {}).get("validation_status") != "HEADER_VALID":
        return None
    return value.get("timing")


def collect_himawari_timings(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Đọc thời gian dòng Nhà Bè từ HSD đầy đủ hoặc header cache đã xác thực."""
    settings = config["himawari"]
    line = float(settings["nha_be_full_disk_line_one_based"])
    staging = project_path(config["paths"]["himawari_staging_root"])
    day01 = project_path(config["paths"]["himawari_day01_root"]).parent
    cache = project_path(config["paths"]["header_cache_root"])
    rows: list[dict[str, Any]] = []
    for day in range(1, 32):
        date = f"202508{day:02d}"
        cached = {path.stem: path for path in (cache / date).glob("*.json")}
        for minute in range(0, 24 * 60, 10):
            cycle = f"{minute // 60:02d}{minute % 60:02d}"
            root = day01 if date == "20250801" else staging
            file_path = root / date / cycle / hsd_filename(
                date, cycle, settings["timing_band"], settings["timing_segment"]
            )
            if file_path.is_file():
                rows.append(read_local_observation_time(file_path, date, cycle, line))
            elif cycle in cached:
                timing = _timing_from_cache(cached[cycle])
                if timing is not None:
                    rows.append(timing)
    return rows


def build_time_manifest(
    radar_rows: Iterable[dict[str, Any]],
    himawari_rows: Iterable[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    tolerance = float(config["time_alignment"]["maximum_absolute_delta_seconds"])
    return match_all(radar_rows_for_matching(radar_rows), himawari_rows, tolerance)


def adopt_verified_legacy_manifest(path: Path) -> list[dict[str, Any]]:
    """Di trú bảng đã audit; chỉ giữ các cột thuộc hợp đồng mới."""
    rows = read_csv(path)
    return [{field: row.get(field, "") for field in TIME_MANIFEST_FIELDS} for row in rows]
