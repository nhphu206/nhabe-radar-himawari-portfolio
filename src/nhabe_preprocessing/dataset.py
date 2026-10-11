"""Tạo chỉ mục và kiểm tra độc lập tập dữ liệu đã tiền xử lý."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import PROJECT_ROOT, project_path
from .io_utils import atomic_write_csv, atomic_write_json, read_csv
from .validation import validate_sample_files


INDEX_FIELDS = (
    "sample_id", "date", "radar_filename", "radar_signature", "himawari_cycle",
    "leakage_group_id", "radar_representative_time_utc",
    "himawari_local_line_time_utc", "signed_delta_seconds",
    "radar_valid_fraction", "radar_range_support_fraction",
    "himawari_all_bands_valid_fraction", "npz_path", "json_path", "qc_status",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def collect_manifests(config: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    root = project_path(config["paths"]["preprocessed_root"])
    manifests: dict[str, list[dict[str, str]]] = {}
    for day in range(1, 32):
        date = f"202508{day:02d}"
        path = root / date / "manifest.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        manifests[date] = read_csv(path)
    return manifests


def build_index(config: dict[str, Any]) -> dict[str, Any]:
    manifests = collect_manifests(config)
    dataset_root = project_path(config["paths"]["dataset_root"])
    dataset_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for date, manifest in sorted(manifests.items()):
        for row in manifest:
            sample_id = row["sample_id"]
            sample_root = project_path(config["paths"]["preprocessed_root"]) / date / "samples"
            npz_path = sample_root / f"{sample_id}.npz"
            json_path = sample_root / f"{sample_id}.json"
            metadata = validate_sample_files(npz_path, json_path, row["radar_filename"])
            cycle = row["himawari_cycle"]
            rows.append({
                "sample_id": sample_id,
                "date": date,
                "radar_filename": row["radar_filename"],
                "radar_signature": row["signature"],
                "himawari_cycle": cycle,
                "leakage_group_id": f"{date}_{cycle}",
                "radar_representative_time_utc": metadata["radar_representative_time_utc"],
                "himawari_local_line_time_utc": metadata["himawari_local_nhabe_line_time_utc"],
                "signed_delta_seconds": metadata["signed_delta_seconds_himawari_minus_radar"],
                "radar_valid_fraction": row["radar_valid_fraction"],
                "radar_range_support_fraction": row["radar_range_support_fraction"],
                "himawari_all_bands_valid_fraction": row["himawari_all_bands_valid_fraction"],
                "npz_path": _relative(npz_path),
                "json_path": _relative(json_path),
                "qc_status": "PASS",
            })
    rows.sort(key=lambda row: (row["radar_representative_time_utc"], row["radar_filename"]))
    expected = config["expected_results"]
    if len(rows) != int(expected["matched_samples"]):
        raise ValueError(f"Số mẫu không đúng: {len(rows)}")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Trùng sample_id")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["leakage_group_id"]].append(row)
    if len(groups) != int(expected["leakage_groups"]):
        raise ValueError(f"Số nhóm chống rò rỉ không đúng: {len(groups)}")
    if not all(len(group) == 2 and {row["radar_signature"] for row in group} == {"A", "B"}
               for group in groups.values()):
        raise ValueError("Có nhóm không gồm đúng một A và một B")

    index_path = dataset_root / "index.csv"
    atomic_write_csv(index_path, rows, INDEX_FIELDS)
    daily_rows: list[dict[str, Any]] = []
    for date in sorted(manifests):
        selected = [row for row in rows if row["date"] == date]
        daily_rows.append({
            "date": date,
            "samples": len(selected),
            "signature_A": sum(row["radar_signature"] == "A" for row in selected),
            "signature_B": sum(row["radar_signature"] == "B" for row in selected),
            "himawari_cycles": len({row["himawari_cycle"] for row in selected}),
            "radar_valid_fraction_median": float(np.median([
                float(row["radar_valid_fraction"]) for row in selected
            ])),
            "himawari_all_bands_valid_fraction_median": float(np.median([
                float(row["himawari_all_bands_valid_fraction"]) for row in selected
            ])),
            "absolute_delta_seconds_max": max(
                abs(float(row["signed_delta_seconds"])) for row in selected
            ),
        })
    daily_path = dataset_root / "daily_summary.csv"
    atomic_write_csv(daily_path, daily_rows, tuple(daily_rows[0]))
    card = {
        "schema_version": "nhabe-himawari-dataset-2025-08",
        "status": "AUDITED_PREPROCESSING_COMPLETE",
        "scope": "Radar Nhà Bè và Himawari-9 tháng 08/2025",
        "samples": len(rows),
        "leakage_groups": len(groups),
        "signature_counts": dict(Counter(row["radar_signature"] for row in rows)),
        "dates": sorted(manifests),
        "shape": {"radar_yx": [256, 256], "himawari_band_yx": [5, 256, 256]},
        "band_order": list(config["himawari"]["bands"]),
        "common_grid": config["grid"]["grid_id"],
        "pipeline_config": _relative(Path(config["_config_path"])),
        "index": _relative(index_path),
        "index_sha256": sha256(index_path),
        "daily_summary": _relative(daily_path),
        "split_policy": {
            "status": "NOT_ASSIGNED",
            "required_group_unit": "leakage_group_id",
            "recommendation": "Chia theo khối thời gian sau khi xác định nhãn dự báo",
            "forbidden": "independent random sample split",
        },
        "normalization": {
            "status": "NOT_COMPUTED",
            "rule": "Chỉ tính từ tập huấn luyện sau khi chốt cách chia tập",
        },
    }
    atomic_write_json(dataset_root / "dataset_card.json", card)
    return card


def audit_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Mở lại toàn bộ mẫu, kiểm tra dữ liệu và quan hệ A/B dùng chung vệ tinh."""
    dataset_root = project_path(config["paths"]["dataset_root"])
    index_path = dataset_root / "index.csv"
    rows = read_csv(index_path)
    expected = config["expected_results"]
    failures: list[dict[str, str]] = []
    signatures: Counter[str] = Counter()
    groups: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    seen: set[str] = set()
    total_bytes = 0
    for row in rows:
        try:
            if row["sample_id"] in seen:
                raise ValueError("Trùng sample_id")
            seen.add(row["sample_id"])
            npz_path = project_path(row["npz_path"])
            json_path = project_path(row["json_path"])
            metadata = validate_sample_files(npz_path, json_path, row["radar_filename"])
            if metadata["radar_signature"] != row["radar_signature"]:
                raise ValueError("Chữ ký trong JSON và chỉ mục không khớp")
            with np.load(npz_path, allow_pickle=False) as arrays:
                satellite = np.asarray(arrays["himawari_bt"]).copy()
                satellite_mask = np.asarray(arrays["himawari_band_valid_mask"]).copy()
                support = float(np.mean(arrays["radar_range_support_mask"]))
            expected_support = 1.0 if row["radar_signature"] == "A" else 0.28173828125
            if not np.isclose(support, expected_support, atol=0.0, rtol=0.0):
                raise ValueError("Tỷ lệ miền khoảng cách không đúng")
            groups[row["leakage_group_id"]].append((satellite, satellite_mask))
            signatures[row["radar_signature"]] += 1
            total_bytes += npz_path.stat().st_size
        except Exception as error:
            failures.append({"sample_id": row.get("sample_id", ""), "error": repr(error)})
    unequal_groups: list[str] = []
    for group_id, pair in groups.items():
        if len(pair) != 2:
            unequal_groups.append(group_id)
            continue
        same_data = np.array_equal(pair[0][0], pair[1][0], equal_nan=True)
        same_mask = np.array_equal(pair[0][1], pair[1][1])
        if not same_data or not same_mask:
            unequal_groups.append(group_id)
    passed = (
        not failures
        and not unequal_groups
        and len(rows) == int(expected["matched_samples"])
        and len(seen) == len(rows)
        and len(groups) == int(expected["leakage_groups"])
        and signatures["A"] == int(expected["signature_A_samples"])
        and signatures["B"] == int(expected["signature_B_samples"])
    )
    summary = {
        "schema_version": "nhabe-himawari-dataset-audit-2025-08",
        "status": "PASS" if passed else "FAIL",
        "expected_samples": int(expected["matched_samples"]),
        "audited_samples": len(rows) - len(failures),
        "failures": failures,
        "unique_sample_ids": len(seen),
        "signature_counts": dict(signatures),
        "unique_himawari_groups": len(groups),
        "groups_with_nonidentical_satellite_arrays": unequal_groups,
        "all_shapes_dtypes_masks_nan_semantics_valid": not failures,
        "all_time_deltas_within_180_seconds": not failures,
        "total_npz_bytes": total_bytes,
        "raw_data_modified": False,
    }
    audit_root = project_path(config["paths"]["audit_root"])
    atomic_write_json(audit_root / "dataset_audit.json", summary)
    if not passed:
        raise RuntimeError("Kiểm tra tập dữ liệu thất bại")
    return summary
