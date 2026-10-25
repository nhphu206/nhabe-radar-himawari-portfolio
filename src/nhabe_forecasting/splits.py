"""Temporal partitions and checks against cross-partition sample reuse."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


def utc_time(value: str) -> datetime:
    """Parse an aware timestamp and canonicalize it to UTC without rounding."""
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamp must include a timezone")
    return result.astimezone(timezone.utc)


def split_for_time(value: datetime, splits: Mapping[str, list[str]]) -> str:
    """Return the single inclusive UTC date partition containing a timestamp."""
    date = value.astimezone(timezone.utc).strftime("%Y%m%d")
    matches = [name for name, (start, end) in splits.items() if start <= date <= end]
    if len(matches) != 1:
        raise ValueError(f"Timestamp has no unique split: {value.isoformat()}")
    return matches[0]


def validate_split_isolation(
    pairs: Iterable[Mapping[str, Any]], splits: Mapping[str, list[str]]
) -> None:
    """Reject pair/sample/time/leakage-group reuse across different partitions."""
    pair_ids: set[str] = set()
    edges: set[tuple[str, str]] = set()
    owners: dict[tuple[str, str], str] = {}
    for pair in pairs:
        if pair["pair_id"] in pair_ids:
            raise ValueError("Duplicate pair")
        pair_ids.add(pair["pair_id"])
        edge = (pair["input_sample_id"], pair["target_sample_id"])
        if edge in edges:
            raise ValueError("Duplicate input-target pair")
        edges.add(edge)
        split = pair["split"]
        roles = ("input", "target", "satellite") if "satellite_sample_id" in pair else ("input", "target")
        for role in roles:
            time = utc_time(pair[f"{role}_time_utc"])
            if split_for_time(time, splits) != split:
                raise ValueError("Pair crosses temporal split boundary")
            keys = [("sample", pair[f"{role}_sample_id"]), ("time", time.isoformat())]
            group = pair.get(f"{role}_leakage_group_id")
            if group:
                keys.append(("group", group))
            for key in keys:
                if key in owners and owners[key] != split:
                    raise ValueError(f"Cross-split reuse: {key[0]}")
                owners[key] = split
