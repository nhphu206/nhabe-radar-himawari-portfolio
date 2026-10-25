# Persistence baseline

Persistence is the first forecasting reference, not a trained model. It assumes that the
future radar reflectivity field is unchanged from the current field:

\[
\hat{R}_{t+10}(i,j) = R_t(i,j).
\]

Here, `R` is reflectivity in dBZ, `(i,j)` identifies one common-grid cell and the two radar
midpoints are exactly 600 seconds apart. This baseline answers a necessary question: does a
learned model improve on simply carrying the latest radar observation forward?

## Evaluation mask and metrics

Persistence is defined only where the current radar has a valid observation. The target is
scored only where it is also valid. The primary mask is therefore

\[
M(i,j) = M_{input}(i,j) \land M_{target}(i,j).
\]

For `N` valid pixels across all validation pairs:

\[
MAE = \frac{1}{N}\sum M(i,j)|\hat{R}(i,j)-R(i,j)|,
\]

\[
RMSE = \sqrt{\frac{1}{N}\sum M(i,j)(\hat{R}(i,j)-R(i,j))^2}.
\]

The implementation accumulates absolute and squared error sums in float64 and divides once
by all valid pixels. It does not average per-image MAE/RMSE, because images contain different
numbers of evaluable pixels. Metrics are calculated in dBZ after reversing the train-only
normalization. No reflectivity threshold or weather-class filtering is used.

## Verified validation result

The baseline opened all 141 validation pairs (August 22–26, 2025) and no test NPZ.

| Quantity | Result |
| --- | ---: |
| Grid pixels | 9,240,576 |
| Input-radar valid pixels | 1,627,007 (17.607%) |
| Target-radar valid pixels | 1,595,558 (17.267%) |
| Common evaluable pixels | 1,139,645 (12.333%) |
| Global MAE | 5.539317 dBZ |
| Global RMSE | 8.036566 dBZ |

Per-pair MAE ranges from 2.378 to 9.519 dBZ with median 4.881 dBZ. Per-pair RMSE ranges
from 3.185 to 12.734 dBZ with median 6.910 dBZ. These per-pair summaries describe variation;
they are not substituted for the global primary metrics.

The low whole-grid valid fraction partly reflects the 400 km square domain and the source
masks. It is not by itself a statement about rainfall frequency or sensor quality.

## Reproducibility and scope

The policy is fixed in `configs/evaluation.yaml`. The script writes a JSON summary and a
per-pair CSV outside Git, refuses to overwrite a non-empty output directory, reopens the
artifacts and records manifest, normalization and policy hashes.

```bash
./.venv/bin/python scripts/12_evaluate_persistence.py \
  --data-root /path/to/research-workspace \
  --manifest /path/to/forecast_pairs.csv \
  --normalization /path/to/normalization.json \
  --config configs/evaluation.yaml \
  --output /path/to/output/forecasting/baselines/persistence_validation
```

This result is a single-month offline validation baseline. It does not establish real-time
availability, rainfall or lightning skill, or production performance. The test partition
remains locked until model selection is complete. No model training occurred in this step.
