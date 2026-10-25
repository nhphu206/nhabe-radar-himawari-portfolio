"""Build exact radar-time forecast pairs without loading/copying source tensors."""

from __future__ import annotations

from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from .splits import split_for_time, utc_time, validate_split_isolation


PAIR_FIELDS = (
    "pair_id", "input_sample_id", "target_sample_id", "input_time_utc",
    "target_time_utc", "lead_time_minutes", "radar_signature", "input_npz_path",
    "target_npz_path", "split", "qc_status", "input_leakage_group_id",
    "target_leakage_group_id", "input_himawari_local_line_time_utc",
    "input_satellite_minus_radar_seconds",
)


def build_forecast_pairs(
    rows: Sequence[Mapping[str, str]], data_root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use exact UTC midpoint +600 s, same day, signature A and existing files."""
    if (config["radar_signature"] != "A" or config["lead_time_minutes"] != 10
            or config["time_column"] != "radar_representative_time_utc"
            or config["matching"] != "exact_timestamp" or config["allow_cross_day"]):
        raise ValueError("Configuration must preserve the approved exact A-only experiment")
    selected = [r for r in rows if r["radar_signature"] == "A"]
    lookup: dict[Any, Mapping[str, str]] = {}
    sample_ids: set[str] = set()
    for row in selected:
        time = utc_time(row["radar_representative_time_utc"])
        if row["date"] != time.strftime("%Y%m%d"):
            raise ValueError("Index date differs from UTC observation date")
        if time in lookup or row["sample_id"] in sample_ids:
            raise ValueError("Duplicate A timestamp or sample identifier")
        if row["qc_status"] != "PASS":
            raise ValueError("Input index contains an unvalidated A sample")
        if not (data_root / row["npz_path"]).is_file():
            raise FileNotFoundError(data_root / row["npz_path"])
        split_for_time(time, config["splits"])
        lookup[time] = row
        sample_ids.add(row["sample_id"])

    pairs: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    for time, row in sorted(lookup.items()):
        desired = time + timedelta(minutes=10)
        if desired.date() != time.date():
            exclusions["cross_day"] += 1
            continue
        target = lookup.get(desired)
        if target is None:
            exclusions["no_exact_target"] += 1
            continue
        split = split_for_time(time, config["splits"])
        if split_for_time(desired, config["splits"]) != split:
            exclusions["cross_split"] += 1
            continue
        if (row["sample_id"] == target["sample_id"] or
                (data_root / row["npz_path"]).resolve() ==
                (data_root / target["npz_path"]).resolve()):
            raise ValueError("Input and target must be distinct")
        satellite_time = utc_time(row["himawari_local_line_time_utc"])
        pairs.append({
            "pair_id": f"{row['sample_id']}__to__{target['sample_id']}",
            "input_sample_id": row["sample_id"],
            "target_sample_id": target["sample_id"],
            "input_time_utc": time.isoformat().replace("+00:00", "Z"),
            "target_time_utc": desired.isoformat().replace("+00:00", "Z"),
            "lead_time_minutes": 10,
            "radar_signature": "A",
            "input_npz_path": row["npz_path"],
            "target_npz_path": target["npz_path"],
            "split": split,
            "qc_status": "PASS",
            "input_leakage_group_id": row["leakage_group_id"],
            "target_leakage_group_id": target["leakage_group_id"],
            "input_himawari_local_line_time_utc": satellite_time.isoformat().replace("+00:00", "Z"),
            "input_satellite_minus_radar_seconds": (satellite_time - time).total_seconds(),
        })
    validate_split_isolation(pairs, config["splits"])
    summary = {
        "schema_version": "nhabe-forecasting-pairs-v1",
        "status": "PAIR_STRUCTURE_VALIDATED",
        "index_rows": len(rows), "signature_A_rows": len(selected),
        "pairs": len(pairs), "split_counts": {
            name: sum(p["split"] == name for p in pairs) for name in config["splits"]
        },
        "excluded_inputs": dict(exclusions),
        "time_basis": "radar_representative_time_utc",
        "lead_time_seconds": 600, "matching": "exact_without_rounding",
        "later_satellite_inputs": sum(p["input_satellite_minus_radar_seconds"] > 0 for p in pairs),
        "multimodal_causality": "REQUIRES_SCIENTIFIC_DECISION_BEFORE_TRAINING",
        "notice": "Structural QC does not establish availability of satellite data at forecast issue time.",
    }
    return pairs, summary
