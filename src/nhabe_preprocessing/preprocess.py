"""Tạo mẫu ghép radar--Himawari cho một ngày."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .common_grid import build_common_grid
from .config import PROJECT_ROOT, project_path
from .himawari import hsd_filename, read_local_observation_time, resample_brightness_temperature
from .io_utils import atomic_write_csv, atomic_write_json, atomic_write_npz, read_csv
from .radar import classify_signature, extract_cappi, read_radar, scan_metadata
from .time_matching import parse_utc
from .validation import validate_arrays, validate_sample_files


MANIFEST_FIELDS = (
    "sample_id", "radar_filename", "signature", "himawari_cycle", "status",
    "npz_path", "json_path", "radar_valid_fraction",
    "radar_range_support_fraction", "himawari_all_bands_valid_fraction",
    "signed_delta_seconds", "runtime_seconds",
)


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def himawari_source_root(date: str, config: dict[str, Any]) -> Path:
    if date == "20250801":
        return project_path(config["paths"]["himawari_day01_root"]).parent
    return project_path(config["paths"]["himawari_staging_root"])


def exact_cycle_time(
    source_root: Path,
    date: str,
    cycle: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    settings = config["himawari"]
    filename = hsd_filename(
        date, cycle, settings["timing_band"], settings["timing_segment"]
    )
    path = source_root / date / cycle / filename
    result = read_local_observation_time(
        path, date, cycle, float(settings["nha_be_full_disk_line_one_based"])
    )
    result["source_file"] = relative(path)
    return result


def revalidate_matches(
    rows: list[dict[str, str]],
    timings: dict[str, dict[str, Any]],
    tolerance: float,
) -> list[dict[str, Any]]:
    exact = {
        cycle: parse_utc(value["local_nhabe_line_time_utc"])
        for cycle, value in timings.items()
    }
    validated: list[dict[str, Any]] = []
    for row in rows:
        radar_time = parse_utc(row["radar_representative_time_utc"])
        candidates = sorted(
            (
                abs((value - radar_time).total_seconds()),
                cycle,
                (value - radar_time).total_seconds(),
            )
            for cycle, value in exact.items()
        )
        absolute, cycle, signed = candidates[0]
        if cycle != row["matched_himawari_cycle"]:
            raise ValueError(
                f"Block 9 đổi chu kỳ gần nhất của {row['radar_filename']}: "
                f"{row['matched_himawari_cycle']} -> {cycle}"
            )
        if absolute > tolerance:
            raise ValueError(f"{row['radar_filename']} vượt dung sai: {absolute} s")
        validated.append({
            **row,
            "himawari_local_line_time_utc": timings[cycle]["local_nhabe_line_time_utc"],
            "signed_delta_seconds": signed,
            "absolute_delta_seconds": absolute,
        })
    return validated


def manifest_row(
    metadata: dict[str, Any],
    cycle: str,
    status: str,
    npz_path: Path,
    json_path: Path,
    runtime: float,
) -> dict[str, Any]:
    return {
        "sample_id": npz_path.stem,
        "radar_filename": metadata["radar_filename"],
        "signature": metadata["radar_signature"],
        "himawari_cycle": cycle,
        "status": status,
        "npz_path": relative(npz_path),
        "json_path": relative(json_path),
        "radar_valid_fraction": metadata["valid_fraction"]["radar_data"],
        "radar_range_support_fraction": metadata["valid_fraction"]["radar_range_support"],
        "himawari_all_bands_valid_fraction": metadata["valid_fraction"]["himawari_all_bands"],
        "signed_delta_seconds": metadata["signed_delta_seconds_himawari_minus_radar"],
        "runtime_seconds": runtime,
    }


def process_day(date: str, config: dict[str, Any]) -> dict[str, Any]:
    """Xử lý hoặc xác nhận lại toàn bộ mẫu đã ghép của một ngày."""
    if len(date) != 8 or not date.startswith("202508"):
        raise ValueError("Ngày phải có dạng YYYYMMDD trong tháng 08/2025")
    started = time.perf_counter()
    inventory_root = project_path(config["paths"]["inventory_root"])
    matching_path = inventory_root / "time_manifest.csv"
    if not matching_path.is_file():
        raise FileNotFoundError(f"Chưa có bảng ghép thời gian: {matching_path}")
    rows = [
        row for row in read_csv(matching_path)
        if row["date"] == date and row["time_match_status"] == "MATCHED"
    ]
    if not rows:
        raise RuntimeError(f"Không có mẫu MATCHED cho {date}")
    source_root = himawari_source_root(date, config)
    cycles = sorted({row["matched_himawari_cycle"] for row in rows})
    timings = {cycle: exact_cycle_time(source_root, date, cycle, config) for cycle in cycles}
    tolerance = float(config["time_alignment"]["maximum_absolute_delta_seconds"])
    rows = revalidate_matches(rows, timings, tolerance)
    by_cycle: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_cycle.setdefault(row["matched_himawari_cycle"], []).append(row)
    if not all(len(pair) == 2 for pair in by_cycle.values()):
        raise ValueError("Mỗi chu kỳ được dùng phải ghép đúng hai khối quét A/B")

    output_root = project_path(config["paths"]["preprocessed_root"]) / date
    samples_root = output_root / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)
    common_grid = build_common_grid(config)
    radar_root = project_path(config["paths"]["radar_root"]) / date[-2:]
    bands = tuple(config["himawari"]["bands"])
    manifest_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for cycle, pair in sorted(by_cycle.items()):
        pending: list[dict[str, Any]] = []
        for row in pair:
            sample_id = row["radar_filename"].replace(".", "_")
            npz_path = samples_root / f"{sample_id}.npz"
            json_path = samples_root / f"{sample_id}.json"
            try:
                metadata = validate_sample_files(npz_path, json_path, row["radar_filename"])
                manifest_rows.append(
                    manifest_row(metadata, cycle, "RESUMED_VALID", npz_path, json_path, 0.0)
                )
            except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
                pending.append(row)
        if not pending:
            continue
        try:
            satellite, band_valid, all_valid = resample_brightness_temperature(
                source_root, date, cycle, common_grid, config
            )
        except Exception as error:
            failures.extend({
                "radar_filename": row["radar_filename"],
                "cycle": cycle,
                "stage": "satellite_resample",
                "error": repr(error),
            } for row in pending)
            continue
        for row in pending:
            radar_started = time.perf_counter()
            filename = row["radar_filename"]
            sample_id = filename.replace(".", "_")
            npz_path = samples_root / f"{sample_id}.npz"
            json_path = samples_root / f"{sample_id}.json"
            try:
                radar = read_radar(radar_root / filename)
                radar_info = scan_metadata(radar)
                signature = classify_signature(radar_info, config)
                if signature not in ("A", "B") or signature != row["radar_signature"]:
                    raise ValueError(f"Sai chữ ký: tính={signature}, bảng={row['radar_signature']}")
                reflectivity, radar_valid, range_support = extract_cappi(
                    radar, radar_info, common_grid, config
                )
                arrays = {
                    "radar_reflectivity_dbz": reflectivity,
                    "radar_data_valid_mask": radar_valid,
                    "radar_range_support_mask": range_support,
                    "himawari_bt": satellite,
                    "himawari_band_valid_mask": band_valid,
                    "himawari_all_bands_valid_mask": all_valid,
                }
                validate_arrays(arrays)
                values = reflectivity[radar_valid]
                timing = timings[cycle]
                metadata = {
                    "schema_version": config["sample_contract"]["schema_version"],
                    "pipeline_config": relative(Path(config["_config_path"])),
                    "grid_id": config["grid"]["grid_id"],
                    "grid_shape_yx": list(config["grid"]["shape_yx"]),
                    "radar_filename": filename,
                    "radar_signature": signature,
                    "radar_scan_metadata": radar_info,
                    "radar_observation_start_utc": row["radar_time_start_utc"],
                    "radar_observation_end_utc": row["radar_time_end_utc"],
                    "radar_representative_time_utc": row["radar_representative_time_utc"],
                    "himawari_folder": cycle,
                    "himawari_nominal_time_utc": timing["nominal_start_utc"],
                    "himawari_observation_start_utc": timing["observation_start_utc"],
                    "himawari_observation_end_utc": timing["observation_end_utc"],
                    "himawari_local_nhabe_line_time_utc": timing["local_nhabe_line_time_utc"],
                    "signed_delta_seconds_himawari_minus_radar": float(row["signed_delta_seconds"]),
                    "absolute_delta_seconds": float(row["absolute_delta_seconds"]),
                    "time_match_status": "MATCHED",
                    "time_match_time_source": "EXACT_HSD_BLOCK9_REVALIDATED",
                    "tolerance_seconds": tolerance,
                    "band_order": list(bands),
                    "arrays": {
                        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                        for name, value in arrays.items()
                    },
                    "valid_fraction": {
                        "radar_data": float(np.mean(radar_valid)),
                        "radar_range_support": float(np.mean(range_support)),
                        "himawari_by_band": {
                            band: float(np.mean(band_valid[index]))
                            for index, band in enumerate(bands)
                        },
                        "himawari_all_bands": float(np.mean(all_valid)),
                    },
                    "radar_valid_statistics_dbz": {
                        "count": int(values.size),
                        "min": float(np.min(values)) if values.size else None,
                        "median": float(np.median(values)) if values.size else None,
                        "max": float(np.max(values)) if values.size else None,
                    },
                }
                atomic_write_npz(npz_path, arrays)
                atomic_write_json(json_path, metadata)
                validate_sample_files(npz_path, json_path, filename)
                manifest_rows.append(manifest_row(
                    metadata, cycle, "CREATED_VALID", npz_path, json_path,
                    time.perf_counter() - radar_started,
                ))
                del radar
                gc.collect()
            except Exception as error:
                failures.append({
                    "radar_filename": filename,
                    "cycle": cycle,
                    "stage": "radar_or_serialization",
                    "error": repr(error),
                })

    manifest_rows.sort(key=lambda row: row["radar_filename"])
    atomic_write_csv(output_root / "manifest.csv", manifest_rows, MANIFEST_FIELDS)
    atomic_write_csv(
        output_root / "failures.csv", failures,
        ("radar_filename", "cycle", "stage", "error"),
    )
    expected = len(rows)
    summary = {
        "schema_version": "nhabe-himawari-day-summary-2025-08",
        "status": "PASS" if len(manifest_rows) == expected and not failures else "FAIL",
        "date": date,
        "expected_matched_samples": expected,
        "valid_samples": len(manifest_rows),
        "failures": len(failures),
        "himawari_cycles": len(by_cycle),
        "exact_block9_time_revalidated": True,
        "max_absolute_delta_seconds": max(float(row["absolute_delta_seconds"]) for row in rows),
        "signatures": {
            label: sum(row["signature"] == label for row in manifest_rows)
            for label in ("A", "B")
        },
        "runtime_seconds": time.perf_counter() - started,
        "manifest": relative(output_root / "manifest.csv"),
        "failures_file": relative(output_root / "failures.csv"),
    }
    atomic_write_json(output_root / "summary.json", summary)
    if summary["status"] != "PASS":
        raise RuntimeError(f"Tiền xử lý {date} thất bại: {summary}")
    return summary
