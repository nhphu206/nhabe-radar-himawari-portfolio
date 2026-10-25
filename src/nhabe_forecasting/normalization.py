"""Fit train-input-only statistics; never normalize or overwrite source NPZ files."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .splits import utc_time, validate_split_isolation


BANDS = ("B08", "B13", "B14", "B15", "B16")
POLICY = {
    "schema_version": "nhabe-train-normalization-policy-v1", "fit_split": "train",
    "radar_population": "unique_train_input_samples",
    "satellite_population": "unique_train_satellite_samples",
    "weighting": "valid_pixel_weighted", "satellite_validity": "per_band_mask",
    "accumulator_dtype": "float64", "standard_deviation_ddof": 0,
    "include_target_only_samples": False, "physical_arrays_zero_fill": False,
}
SOURCE_FIELDS = ("role", "sample_id", "npz_path", "npz_sha256", "json_path", "json_sha256")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class RunningMoments:
    """Merge float64 batch moments (population standard deviation, ddof=0)."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).ravel()
        if values.size == 0:
            return
        if not np.all(np.isfinite(values)):
            raise ValueError("Moments require finite valid values")
        n = int(values.size)
        mean = float(np.mean(values, dtype=np.float64))
        m2 = float(np.sum((values - mean) ** 2, dtype=np.float64))
        total = self.count + n
        delta = mean - self.mean
        self.m2 += m2 + delta * delta * self.count * n / total
        self.mean += delta * n / total
        self.count = total
        self.minimum = min(self.minimum, float(np.min(values)))
        self.maximum = max(self.maximum, float(np.max(values)))

    def result(self, units: str) -> dict[str, Any]:
        if self.count == 0:
            raise ValueError("No valid training pixels for a field")
        std = float(np.sqrt(self.m2 / self.count))
        if not np.isfinite(self.mean) or not np.isfinite(std) or std <= 0:
            raise ValueError("Training standard deviation must be positive and finite")
        return {"count": self.count, "mean": self.mean, "std": std,
                "min": self.minimum, "max": self.maximum, "units": units}


def unique_sources(
    rows: Sequence[Mapping[str, str]], role: str, data_root: Path,
) -> list[dict[str, str]]:
    """Deduplicate role sources; reject ambiguous IDs, paths or metadata fingerprints."""
    by_id: dict[str, dict[str, str]] = {}
    path_ids: dict[Path, str] = {}
    for row in rows:
        sid = row[f"{role}_sample_id"]
        source = {"role": role, "sample_id": sid, "npz_path": row[f"{role}_npz_path"],
                  "json_path": row[f"{role}_json_path"], "json_sha256": row[f"{role}_json_sha256"]}
        path = (data_root / source["npz_path"]).resolve()
        if sid in by_id and by_id[sid] != source:
            raise ValueError("Same sample ID has conflicting source references")
        if path in path_ids and path_ids[path] != sid:
            raise ValueError("One NPZ path has multiple sample identifiers")
        by_id[sid] = source
        path_ids[path] = sid
    return [by_id[key] for key in sorted(by_id)]


def valid_values(values: np.ndarray, mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Validate masks before selecting values; never include missing-value padding."""
    if values.shape != shape or mask.shape != shape:
        raise ValueError("Unexpected field/mask shape")
    if values.dtype != np.float32 or mask.dtype != np.bool_:
        raise ValueError("Expected float32 physical field and boolean mask")
    if not np.array_equal(np.isfinite(values), mask):
        raise ValueError("Finite values and validity mask disagree")
    if not np.all(np.isnan(values[~mask])):
        raise ValueError("Missing physical values must remain NaN")
    return values[mask]


def fit_train_statistics(
    rows: Sequence[Mapping[str, str]], data_root: Path,
    splits: Mapping[str, list[str]], policy: Mapping[str, Any],
    *, progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Read only unique input/satellite sources belonging to the train partition."""
    if dict(policy) != POLICY:
        raise ValueError("Normalization policy differs from the approved input-only protocol")
    validate_split_isolation(rows, splits)
    for row in rows:
        if row["qc_status"] != "PASS" or row["radar_signature"] != "A":
            raise ValueError("Only validated A forecast pairs are supported")
        if (utc_time(row["target_time_utc"]) - utc_time(row["input_time_utc"])).total_seconds() != 600:
            raise ValueError("Pair is not exactly ten radar-midpoint minutes apart")
        if int(row["timing_header_count"]) != 10:
            raise ValueError("Pair lacks the complete source header audit")
    train = [row for row in rows if row["split"] == "train"]
    if not train:
        raise ValueError("No training pairs")
    sources = (unique_sources(train, "input", data_root)
               + unique_sources(train, "satellite", data_root))
    moments = {key: RunningMoments() for key in ("radar_reflectivity_dbz", *BANDS)}
    audited_sources: list[dict[str, str]] = []
    for number, source in enumerate(sources, 1):
        path = data_root / source["npz_path"]
        if file_sha256(data_root / source["json_path"]) != source["json_sha256"]:
            raise ValueError("Source JSON changed since the timing audit")
        before = path.stat()
        digest = file_sha256(path)
        with np.load(path, allow_pickle=False) as arrays:
            if source["role"] == "input":
                values = valid_values(arrays["radar_reflectivity_dbz"],
                                      arrays["radar_data_valid_mask"], (256, 256))
                moments["radar_reflectivity_dbz"].update(values)
            else:
                values = arrays["himawari_bt"]
                mask = arrays["himawari_band_valid_mask"]
                valid_values(values, mask, (5, 256, 256))
                all_valid = arrays["himawari_all_bands_valid_mask"]
                if (all_valid.shape != (256, 256) or all_valid.dtype != np.bool_
                        or not np.array_equal(all_valid, np.all(mask, axis=0))):
                    raise ValueError("Satellite all-band mask disagrees with per-band masks")
                for index, band in enumerate(BANDS):
                    moments[band].update(values[index][mask[index]])
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Source NPZ changed while computing statistics")
        audited_sources.append({**source, "npz_sha256": digest})
        if progress and (number % 50 == 0 or number == len(sources)):
            progress(f"Train sources read: {number}/{len(sources)}")
    statistics = {name: moment.result("dBZ" if name == "radar_reflectivity_dbz" else "K")
                  for name, moment in moments.items()}
    result = {
        "schema_version": "nhabe-train-normalization-v1", "status": "TRAIN_STATISTICS_FITTED",
        "policy": dict(policy), "band_order": list(BANDS), "train_pairs": len(train),
        "source_counts": {role: sum(s["role"] == role for s in sources) for role in ("input", "satellite")},
        "fields": statistics,
        "transform": "(x - mean) / std; masks unchanged; missing padding only later in memory",
        "radar_target_transform": "reuse radar input train mean/std; do not fit target-only samples",
        "physical_source_arrays_modified": False,
    }
    return result, audited_sources
