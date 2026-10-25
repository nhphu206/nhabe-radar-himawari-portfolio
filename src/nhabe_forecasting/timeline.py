"""Select earlier satellite inputs and audit every required HSD header."""

from __future__ import annotations

from collections import Counter
from datetime import timedelta
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .pairs import PAIR_FIELDS, build_forecast_pairs
from .satellite_timing import read_header_timing
from .splits import split_for_time, utc_time, validate_split_isolation


AUDITED_PAIR_FIELDS = tuple(k for k in PAIR_FIELDS if k not in (
    "input_himawari_local_line_time_utc", "input_satellite_minus_radar_seconds",
)) + (
    "input_cutoff_utc", "radar_midpoint_lead_seconds", "target_lead_from_cutoff_seconds",
    "satellite_sample_id", "satellite_npz_path", "satellite_json_path",
    "satellite_time_utc", "satellite_leakage_group_id", "satellite_nominal_time_utc",
    "satellite_observation_end_utc", "satellite_file_creation_utc", "timing_header_count",
    "input_json_path", "target_json_path", "input_json_sha256", "target_json_sha256",
    "satellite_json_sha256",
)
HEADER_FIELDS = (
    "date", "cycle", "path", "band", "segment", "satellite", "observation_area",
    "observation_timeline", "observation_start_utc", "observation_end_utc",
    "file_creation_utc", "header_through_block9_sha256", "file_size_bytes",
)
EXCLUSION_FIELDS = ("pair_id", "reason", "detail")


def build_audited_pairs(
    rows: Sequence[Mapping[str, str]], data_root: Path,
    config: Mapping[str, Any], pipeline: Mapping[str, Any],
    *, limit: int | None = None, progress: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    """Preserve exact targets, then require the previous cycle's ten headers before cutoff."""
    approved = {
        "schema_version": "nhabe-forecasting-observation-timing-v1",
        "input_cutoff": "radar_observation_end_utc",
        "satellite_nominal_lag_minutes": 10,
        "satellite_selection": "exact_previous_nominal_cycle",
        "require_header_observation_end_before_cutoff": True,
        "require_header_creation_before_cutoff": True,
    }
    if any(config.get(k) != v for k, v in approved.items()):
        raise ValueError("Configuration differs from the approved timing protocol")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if (list(pipeline["himawari"]["bands"]) != ["B08", "B13", "B14", "B15", "B16"]
            or list(pipeline["himawari"]["segments"]) != ["04", "05"]):
        raise ValueError("Expected the fixed five bands and two segments")
    candidates, initial = build_forecast_pairs(rows, data_root, config)
    selected = {r["sample_id"]: r for r in rows if r["radar_signature"] == "A"}
    nominal: dict[Any, Mapping[str, str]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for sample_id, row in selected.items():
        path = data_root / row["json_path"]
        raw = path.read_bytes()
        value = json.loads(raw)
        if (value["radar_filename"] != row["radar_filename"]
                or value["radar_signature"] != "A"
                or value["grid_id"] != pipeline["grid"]["grid_id"]
                or list(value["band_order"]) != list(pipeline["himawari"]["bands"])
                or utc_time(value["radar_representative_time_utc"]) !=
                utc_time(row["radar_representative_time_utc"])):
            raise ValueError(f"JSON/index identity mismatch: {sample_id}")
        start = utc_time(value["radar_observation_start_utc"])
        end = utc_time(value["radar_observation_end_utc"])
        if end <= start or start + (end - start) / 2 != utc_time(row["radar_representative_time_utc"]):
            raise ValueError(f"Invalid radar midpoint: {sample_id}")
        time = utc_time(value["himawari_nominal_time_utc"])
        if time.strftime("%Y%m%d") != row["date"] or time.strftime("%H%M") != row["himawari_cycle"]:
            raise ValueError("Satellite nominal time/index mismatch")
        if time in nominal:
            raise ValueError("Duplicate nominal cycle among A samples")
        nominal[time] = row
        metadata[sample_id] = value
        hashes[sample_id] = hashlib.sha256(raw).hexdigest()

    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    headers: list[dict[str, Any]] = []
    examined = candidates if limit is None else candidates[:limit]
    for number, pair in enumerate(examined, 1):
        input_id = pair["input_sample_id"]
        m = metadata[input_id]
        t = metadata[pair["target_sample_id"]]
        cutoff = utc_time(m["radar_observation_end_utc"])
        previous = utc_time(m["himawari_nominal_time_utc"]) - timedelta(minutes=10)
        reason = ""
        detail = ""
        source = nominal.get(previous)
        if previous.date() != utc_time(pair["input_time_utc"]).date():
            reason = "satellite_cross_day"
        elif source is None:
            reason = "previous_satellite_cycle_unavailable_in_index"
        elif split_for_time(previous, config["splits"]) != pair["split"]:
            reason = "satellite_cross_split"
        elif cutoff.date() != previous.date() or utc_time(t["radar_observation_start_utc"]) <= cutoff:
            reason = "radar_target_not_strictly_after_cutoff_or_cutoff_cross_day"
        if reason:
            excluded.append({"pair_id": pair["pair_id"], "reason": reason, "detail": detail})
            continue
        assert source is not None
        date, cycle = previous.strftime("%Y%m%d"), previous.strftime("%H%M")
        source_root = (Path(pipeline["paths"]["himawari_day01_root"]).parent
                       if date == "20250801" else Path(pipeline["paths"]["himawari_staging_root"]))
        cycle_headers: list[dict[str, Any]] = []
        try:
            for band in pipeline["himawari"]["bands"]:
                for segment in pipeline["himawari"]["segments"]:
                    name = f"HS_H09_{date}_{cycle}_{band}_FLDK_R20_S{segment}10.DAT.bz2"
                    path = data_root / source_root / date / cycle / name
                    h = read_header_timing(path)
                    if (h["band"] != band or h["segment"] != segment
                            or h["satellite"] != "Himawari-9" or h["observation_area"] != "FLDK"
                            or h["observation_timeline"] != int(cycle)):
                        raise ValueError("HSD header identity differs from the requested source")
                    start = utc_time(h["observation_start_utc"])
                    end = utc_time(h["observation_end_utc"])
                    creation = utc_time(h["file_creation_utc"])
                    if not previous <= start <= end <= creation:
                        raise ValueError("HSD timestamps precede nominal cycle or are unordered")
                    if band == "B13" and segment == "04":
                        local_time = utc_time(metadata[source["sample_id"]]["himawari_local_nhabe_line_time_utc"])
                        if not start <= local_time <= end:
                            raise ValueError("Source JSON local time falls outside B13/S04 observations")
                    cycle_headers.append({"date": date, "cycle": cycle,
                                          "path": str(path.relative_to(data_root)), **h})
        except (OSError, ValueError, EOFError, OverflowError) as error:
            reason = "missing_or_invalid_source_header"
            detail = str(error)
        headers.extend(cycle_headers)
        if not reason:
            observation_end = max(utc_time(h["observation_end_utc"]) for h in cycle_headers)
            creation = max(utc_time(h["file_creation_utc"]) for h in cycle_headers)
            if observation_end > cutoff:
                reason = "satellite_observation_after_cutoff"
            elif creation > cutoff:
                reason = "satellite_creation_after_cutoff"
        if reason:
            excluded.append({"pair_id": pair["pair_id"], "reason": reason, "detail": detail})
        else:
            sid = source["sample_id"]
            sm = metadata[sid]
            result = {k: pair[k] for k in PAIR_FIELDS if k in AUDITED_PAIR_FIELDS}
            result.update({
                "input_cutoff_utc": cutoff.isoformat().replace("+00:00", "Z"),
                "radar_midpoint_lead_seconds": 600,
                "target_lead_from_cutoff_seconds": (utc_time(pair["target_time_utc"]) - cutoff).total_seconds(),
                "satellite_sample_id": sid, "satellite_npz_path": source["npz_path"],
                "satellite_json_path": source["json_path"],
                "satellite_time_utc": sm["himawari_local_nhabe_line_time_utc"],
                "satellite_leakage_group_id": source["leakage_group_id"],
                "satellite_nominal_time_utc": sm["himawari_nominal_time_utc"],
                "satellite_observation_end_utc": observation_end.isoformat().replace("+00:00", "Z"),
                "satellite_file_creation_utc": creation.isoformat().replace("+00:00", "Z"),
                "timing_header_count": len(cycle_headers),
                "input_json_path": selected[input_id]["json_path"],
                "target_json_path": selected[pair["target_sample_id"]]["json_path"],
                "input_json_sha256": hashes[input_id],
                "target_json_sha256": hashes[pair["target_sample_id"]],
                "satellite_json_sha256": hashes[sid],
            })
            accepted.append(result)
        if progress and (number % 25 == 0 or number == len(examined)):
            progress(f"Headers audited: {number}/{len(examined)} candidates; {len(accepted)} accepted")
    validate_split_isolation(accepted, config["splits"])
    assert len(accepted) + len(excluded) == len(examined)
    summary = {
        "schema_version": config["schema_version"], "status": "OBSERVATION_TIMING_AUDITED",
        "scope": "full_exact_candidate_set" if limit is None else "limited_preview",
        "index_rows": len(rows), "signature_A_rows": initial["signature_A_rows"],
        "exact_candidate_pairs": len(candidates), "examined_candidates": len(examined),
        "pairs": len(accepted), "headers_recorded": len(headers),
        "initial_excluded_inputs": initial["excluded_inputs"],
        "timeline_exclusions": dict(Counter(e["reason"] for e in excluded)),
        "split_counts": {name: sum(p["split"] == name for p in accepted) for name in config["splits"]},
        "lead_time_basis": "exact_600_seconds_between_radar_midpoints_not_operational_issue_time",
        "cutoff": config["input_cutoff"],
        "operational_availability_verified": False,
        "notice": "Header observation/creation checks do not establish download or processing latency.",
    }
    if accepted:
        leads = [p["target_lead_from_cutoff_seconds"] for p in accepted]
        summary["target_lead_from_cutoff_seconds"] = {"min": min(leads), "max": max(leads)}
    return accepted, summary, headers, excluded
