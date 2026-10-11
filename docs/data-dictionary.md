# Output data dictionary

This document describes the current implementation, not a future model input format.
The scientific settings remain in [pipeline.yaml](../configs/pipeline.yaml).
Real observations and generated dataset files are excluded from Git.

## Files and relationships

```text
output/inventory/time_manifest.csv       Radar-to-satellite matching decisions
output/preprocessed/YYYYMMDD/
  samples/<sample_id>.npz                Six numerical arrays
  samples/<sample_id>.json               Metadata for the same sample
  manifest.csv                          Successfully validated samples for this day
  failures.csv                          Failed attempts, stage and error
  summary.json                          Day-level counts and processing status
output/dataset/index.csv                 One row per validated paired sample
```

A real `sample_id` is the radar filename with its dot replaced by an underscore.
An NPZ and JSON with the same stem form one sample. Two samples can share a satellite
cycle while containing different radar volumes. CSV files are bookkeeping, not additional
image tensors or training labels. Configured paths can override these default locations.

## NPZ arrays

Load with `np.load(path, allow_pickle=False)`. The container holds ordinary NumPy arrays,
not a serialized Py-ART Radar object or NumPy masked array.

| Key | Shape / dtype | Meaning |
| --- | --- | --- |
| `radar_reflectivity_dbz` | `(256,256)`, float32 | Gridded reflectivity in dBZ at 1,000 m above the grid origin |
| `radar_data_valid_mask` | `(256,256)`, bool | True exactly where reflectivity is finite |
| `radar_range_support_mask` | `(256,256)`, bool | True where distance from grid center is no greater than the volume's maximum stored range |
| `himawari_bt` | `(5,256,256)`, float32 | Resampled brightness temperatures in kelvin |
| `himawari_band_valid_mask` | `(5,256,256)`, bool | True exactly where each band value is finite |
| `himawari_all_bands_valid_mask` | `(256,256)`, bool | Logical AND of the five band-valid masks at each pixel |

`R[i,j]` denotes radar row `i`, column `j`. `H[b,i,j]` denotes satellite band index `b`
at that same grid location. Channel order is **B08, B13, B14, B15, B16**.
Rows run from positive to negative projected y, columns from negative to positive x.
Pixel centers span ±199,218.75 m; the domain edges are ±200,000 m and spacing is
1,562.5 m. The radar layer is flipped along y after Py-ART extraction to align with
the satellite grid. Coordinates are reconstructed from configuration, not stored in NPZ.

Missing physical values are **NaN**, accompanied by boolean validity masks. They are
not replaced by zero. Negative dBZ and finite zero dBZ remain valid measurements.
A range-support True value does not guarantee a finite observation, meteorological
quality, or vertical beam coverage. Satellite per-band validity may differ; the all-band
mask requires all five bands to be finite. These masks are not forecasting labels.

## Per-sample JSON

Real sample metadata is generated in `preprocess.py`.

| Field(s) | Meaning |
| --- | --- |
| `schema_version` | `nhabe-himawari-paired-sample-2025-08` |
| `pipeline_config`, `grid_id`, `grid_shape_yx` | Configuration reference and grid identity/shape |
| `radar_filename`, `radar_signature`, `radar_scan_metadata` | Source radar identity and measured scan geometry |
| `radar_observation_start_utc`, `radar_observation_end_utc` | First/last observation times derived from ray offsets |
| `radar_representative_time_utc` | Midpoint of the minimum/maximum ray times, not filename time |
| `himawari_folder`, `himawari_nominal_time_utc` | Cycle folder (HHMM) and nominal cycle start |
| `himawari_observation_start_utc`, `himawari_observation_end_utc` | Satellite observation interval |
| `himawari_local_nhabe_line_time_utc` | Local observation time at Nhà Bè from HSD Block 9 |
| `signed_delta_seconds_himawari_minus_radar` | Satellite local time minus radar midpoint, in seconds; positive means satellite is later |
| `absolute_delta_seconds`, `tolerance_seconds` | Absolute offset and allowed tolerance (180 s) |
| `time_match_status`, `time_match_time_source` | `MATCHED` and `EXACT_HSD_BLOCK9_REVALIDATED` for created real samples |
| `band_order`, `arrays` | Channel names and each array's shape/dtype |
| `valid_fraction` | Fractions in [0,1] for radar data, range support, each satellite band and all bands |
| `radar_valid_statistics_dbz` | Count, min, median and max over finite radar values; empty statistics are null |

Timestamp fields use timezone-aware ISO 8601 strings. Nominal satellite start and local
line observation time are different concepts; matching uses the latter. Historical files
may retain additional provenance fields from configuration-reference migration.

The mock command uses the same six-array contract but reduced JSON metadata, explicitly
marked `sample_kind: SYNTHETIC_CONTRACT_SMOKE_TEST` and `radar_filename: MOCK_NOT_RAW_DATA`.
It also writes `sample_run_summary.json`. It is not a real observation and does not create
the full daily manifest or monthly index. Its JSON includes a mock `leakage_group_id`;
for real samples the group identifier is constructed in the dataset index.

## CSV dictionaries

### Time matching manifest

`output/inventory/time_manifest.csv` retains radar inventory columns and adds:

| Column | Meaning |
| --- | --- |
| `matched_himawari_cycle` | Nearest candidate cycle (HHMM), when one exists |
| `himawari_nominal_time_utc`, `himawari_local_line_time_utc` | Nominal and local observation times |
| `signed_delta_seconds`, `absolute_delta_seconds` | Signed/absolute time difference, in seconds |
| `time_match_status` | `MATCHED`, `UNMATCHED_OUTSIDE_TOLERANCE`, or `UNMATCHED_NO_HIMAWARI_CYCLE` |

A nearest candidate outside tolerance is recorded but is not a matched sample. Missing
candidate fields can be empty. Ties select the earlier nominal cycle; no temporal
interpolation is performed. This table contains matching decisions, not proof that
preprocessing subsequently succeeded.

### Daily processing manifest

| Column(s) | Meaning |
| --- | --- |
| `sample_id`, `radar_filename`, `signature`, `himawari_cycle` | Sample/source identifiers; signature is A or B |
| `status` | `CREATED_VALID` for newly written/reopened output; `RESUMED_VALID` for existing output that passed revalidation |
| `npz_path`, `json_path` | Output paths, normally relative to project root |
| `radar_valid_fraction`, `radar_range_support_fraction`, `himawari_all_bands_valid_fraction` | Fractions in [0,1], not percentages |
| `signed_delta_seconds` | Satellite local time minus radar midpoint |
| `runtime_seconds` | Per-sample recorded processing duration; resumed rows use 0, not their original runtime |

Failed attempts are recorded separately in `failures.csv` with `radar_filename`, `cycle`,
`stage`, and `error`. A day's `summary.json` is PASS only when all expected matched samples
are validated and no failures remain. A daily manifest is not the time-matching manifest.

### Dataset index

`output/dataset/index.csv` contains all of the following columns:

| Column(s) | Meaning |
| --- | --- |
| `sample_id`, `date`, `radar_filename`, `radar_signature`, `himawari_cycle` | Identity; date is YYYYMMDD and cycle is HHMM |
| `leakage_group_id` | `YYYYMMDD_HHMM`: dependency group sharing a satellite cycle |
| `radar_representative_time_utc`, `himawari_local_line_time_utc` | Times used for alignment |
| `signed_delta_seconds` | Satellite-minus-radar difference in seconds |
| `radar_valid_fraction`, `radar_range_support_fraction`, `himawari_all_bands_valid_fraction` | Descriptive fractions in [0,1] |
| `npz_path`, `json_path` | Project-relative file paths |
| `qc_status` | `PASS` from sample contract validation; not meteorological quality control |

Treat date/cycle identifiers as strings when reading CSV to retain leading zeros.
The current index builder expects the complete August dataset, not a two-sample demo.

## Leakage groups and future learning

The audited dataset has 4,387 groups, each containing one A and one B sample sharing
the same satellite tensor. Splitting those samples independently could place the same
satellite input in training and testing. Keep each group in one partition and additionally
consider temporal blocks because adjacent observations are correlated. Grouping alone
does not guarantee a leakage-free forecasting evaluation. No split, normalization,
forecasting target or trained model is supplied yet.

## Validation scope

Current checks verify required arrays, shape, dtype, finite-value masks, the logical AND
of band masks, source filename, MATCHED status and time tolerance. They are not a complete
JSON-schema validator or an assessment of meteorological accuracy. The historical full
dataset audit and the Docker/CI synthetic smoke tests are distinct evidence.

Implementation references: [array validation](../src/nhabe_preprocessing/validation.py),
[sample generation](../src/nhabe_preprocessing/preprocess.py),
[index and audit](../src/nhabe_preprocessing/dataset.py),
[time matching](../src/nhabe_preprocessing/time_matching.py),
[mock generation](../src/nhabe_preprocessing/sample_pipeline.py).
