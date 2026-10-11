"""Ghép trung điểm radar với thời gian quan trắc cục bộ Himawari."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Thời gian không có múi giờ: {value}")
    return parsed


def match_nearest_cycle(
    radar_row: dict[str, Any],
    himawari_rows: Iterable[dict[str, Any]],
    tolerance_seconds: float,
) -> dict[str, Any]:
    """Tìm chu kỳ gần nhất; khi hòa chọn chu kỳ danh nghĩa sớm hơn."""
    radar_time = parse_utc(radar_row["radar_representative_time_utc"])
    candidates = []
    for row in himawari_rows:
        local_time = parse_utc(row["local_nhabe_line_time_utc"])
        signed = float((local_time - radar_time).total_seconds())
        candidates.append((abs(signed), row["cycle"], signed, row))
    if not candidates:
        return {**radar_row, "time_match_status": "UNMATCHED_NO_HIMAWARI_CYCLE"}
    absolute, cycle, signed, satellite = min(candidates, key=lambda item: (item[0], item[1]))
    status = "MATCHED" if absolute <= tolerance_seconds else "UNMATCHED_OUTSIDE_TOLERANCE"
    return {
        **radar_row,
        "matched_himawari_cycle": cycle,
        "himawari_nominal_time_utc": satellite["nominal_start_utc"],
        "himawari_local_line_time_utc": satellite["local_nhabe_line_time_utc"],
        "signed_delta_seconds": signed,
        "absolute_delta_seconds": absolute,
        "time_match_status": status,
    }


def match_all(
    radar_rows: Iterable[dict[str, Any]],
    himawari_rows: Iterable[dict[str, Any]],
    tolerance_seconds: float,
) -> list[dict[str, Any]]:
    satellites_by_date: dict[str, list[dict[str, Any]]] = {}
    for row in himawari_rows:
        satellites_by_date.setdefault(row["date"], []).append(row)
    return [
        match_nearest_cycle(row, satellites_by_date.get(row["date"], []), tolerance_seconds)
        for row in radar_rows
    ]
