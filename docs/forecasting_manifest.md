# Observation-timed forecast manifest

Schema: `nhabe-forecasting-observation-timing-v1`. Scientific preprocessing configuration
is unchanged; this protocol is defined separately in `configs/forecasting.yaml`.

## Protocol and limits

Use signature A only, with radar target midpoint exactly input midpoint +600 seconds.
Both volumes and the satellite source must be in the same UTC day and temporal partition.
Research input cutoff is radar observation end, not its midpoint. Require target
observation start to be strictly later than cutoff.

Satellite comes from the cycle exactly ten nominal minutes before the cycle originally
matched to the radar input. Read the satellite tensor from that earlier sample's existing
NPZ. No nearest-time fallback, cross-day context, temporal interpolation, or array rewrite.
All ten HSD sources must match band/segment/satellite/area/timeline identifiers. Every
header must have ordered, finite observation/creation times, ending no later than cutoff.
The source JSON's local time must lie inside the B13/S04 observation interval.

This is an offline observation/creation-time experiment. File creation is not receipt
time, and radar end is not measured CAPPI availability. Download and processing latencies
remain unverified. Do not label `input_cutoff_utc` a verified operational issue time.

## Artifacts

| File | Meaning |
| --- | --- |
| `forecast_pairs.csv` | Accepted radar/satellite/target source references and temporal partition |
| `satellite_header_audit.csv` | Header timings, identity, size and header fingerprint for inspected sources |
| `excluded_pairs.csv` | Rejected exact candidates, reason and diagnostic detail |
| `pair_summary.json` | Counts, scope, assumptions, source config/index checksums and artifact checksums |

The default CLI output is the repository's `output/forecasting/observation_timing/`.
Use `--data-root` for the workspace that holds the original index and observations.
Source paths in CSV are relative to that data root. All generated artifacts stay out of Git.
An existing nonempty output directory is protected from overwrite. A `--limit` preview
is explicitly marked `limited_preview`, not full audit.

## Pair fields

| Field(s) | Meaning |
| --- | --- |
| `pair_id`, `input_sample_id`, `target_sample_id` | Unique directed radar input → target identifiers |
| `input_time_utc`, `target_time_utc` | Radar midpoint timestamps, exactly 600 seconds apart |
| `lead_time_minutes`, `radar_midpoint_lead_seconds` | 10 / 600, measured between radar midpoints |
| `input_cutoff_utc` | Input radar's observation end |
| `target_lead_from_cutoff_seconds` | Target midpoint minus cutoff, not nominal ten-minute lead |
| `radar_signature`, `split`, `qc_status` | A; train/validation/test; PASS under this protocol, not meteorological QC |
| `input_npz_path`, `target_npz_path` | Files providing radar input and radar target |
| `satellite_sample_id`, `satellite_npz_path`, `satellite_json_path` | Earlier sample supplying satellite bands/masks, not radar input |
| `satellite_time_utc` | Earlier sample's B13 local line time at Nhà Bè; not the end of all ten observations |
| `satellite_nominal_time_utc` | Earlier satellite cycle's nominal time |
| `satellite_observation_end_utc`, `satellite_file_creation_utc` | Max observation end and max creation across all ten headers |
| `timing_header_count` | 10 for every accepted pair |
| `input_leakage_group_id`, `target_leakage_group_id`, `satellite_leakage_group_id` | Dependency groups, all checked for cross-partition reuse |
| `input_json_path`, `target_json_path` | Radar source metadata references |
| `input_json_sha256`, `target_json_sha256`, `satellite_json_sha256` | SHA256 fingerprints of the metadata bytes read |

Midpoint, cutoff, satellite observation and creation timestamps have distinct meanings.
Read date/cycle values as strings to preserve leading zeros. All timestamp fields are UTC.

## Header audit fields

`date`, `cycle`, `path`, `band`, `segment`, `satellite`, `observation_area`,
`observation_timeline`, `observation_start_utc`, `observation_end_utc`, `file_creation_utc`,
`header_through_block9_sha256`, `file_size_bytes`.

The fingerprint covers decoded bytes through Block 9, not the whole compressed file or
image payload. The reader consumes at most 1 MB compressed and exposes at most 64 KiB
decoded per file. It does not validate/reprocess full image payloads. Header records can
include partially inspected rejected cycles; accepted pairs require ten valid records.
Array validity is a separate contract checked by the preprocessing audit/data loader.

## Verified local run (October 20, 2026)

- 4,387 A samples; 842 exact same-day radar candidate pairs.
- 822 accepted pairs; 508 train, 141 validation, 173 test.
- 15 rejected candidates with no previous-cycle A source; 5 with cross-day source.
- 8,220 HSD headers recorded; all accepted sources passed observation/creation cutoff checks.
- Target midpoint minus cutoff: 514.5–527 seconds.
- CSVs were reopened; all pair paths, source JSON hashes, header sets/times, config/index
  checksums and split isolation were rechecked. NPZ shape/dtype/masks were reopened for
  the first three pairs' radar/satellite/target roles, not all 822 pairs in this new step.

No normalization, trained model, baseline metric, Docker verification of the new extension,
or operational-availability result is claimed here.
