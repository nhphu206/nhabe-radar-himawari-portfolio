# Qualitative validation examples

The qualitative figure uses three validation pairs selected before inspecting their images:
chronological index 0, `floor(N/2)` and `N-1` from the fixed 141-pair validation split. This
first/middle/last policy is deterministic and avoids manually choosing visually favorable
cases. It is not a claim that three cases represent the full month.

Each row displays current radar reflectivity, the frozen multimodal model's ten-minute
prediction, the ten-minute radar target and absolute error on their common-valid pixels. All
reflectivity panels use the same -10 to 60 dBZ display range; error panels use 0 to 30 dBZ.
These ranges control colors only. They do not threshold, clip or modify the stored arrays.

| Position | Validation index | Common-valid pixels | MAE (dBZ) | RMSE (dBZ) |
| --- | ---: | ---: | ---: | ---: |
| First | 0 | 4,878 | 8.339 | 10.214 |
| Middle | 70 | 13,024 | 3.950 | 5.095 |
| Last | 140 | 5,008 | 2.852 | 3.622 |

The first case changes substantially between the current and future radar fields and has the
largest error of the three. The middle case contains broader spatial echo coverage and has an
intermediate error. The last case has less common-valid coverage and the lowest error among
these examples. These are descriptive statements about displayed arrays, not diagnoses of
rain intensity, storm type, clutter or radar quality.

Predictions were produced first without reading their future target arrays. The visualization
stage then opened the targets solely for evaluation and plotting; it did not update weights,
choose a checkpoint or change a hyperparameter. Global validation and locked-test metrics in
the evaluation documents remain the quantitative evidence. The generated figure and JSON
audit are stored outside Git under `output/forecasting/qualitative_figures_v1/`.
