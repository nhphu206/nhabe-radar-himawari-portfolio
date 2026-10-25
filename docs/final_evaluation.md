# Locked-test evaluation

The final evaluator was committed before test data were opened. Commit
`b5b839918ed6f186f488f3c4e8d1b3c2434beac2` fixes the model code, source selection,
normalization, metrics, checkpoints and test procedure. The repository was clean when the
173-pair test split was constructed. No model or hyperparameter was changed afterward.

## Final results

All methods were evaluated on the same 1,746,163 common-valid pixels, representing 15.401%
of the 11,337,728 cells across 173 test pairs.

| Method | Test MAE (dBZ) | Test RMSE (dBZ) | RMSE reduction vs persistence |
| --- | ---: | ---: | ---: |
| Persistence | 5.011121 | 7.407763 | 0.00% |
| Radar-only Small U-Net | 4.234776 | 5.956063 | 19.60% |
| Radar + Himawari Small U-Net | **4.215341** | **5.903871** | **20.30%** |

Relative to radar-only, multimodal reduces test MAE by 0.46% and RMSE by 0.88%. The same
ordering appeared on validation, but the multimodal advantage is smaller on test than its
1.64% MAE and 2.77% RMSE validation advantage.

## Interpretation

Both learned models improve substantially over persistence on the locked test set. Adding
Himawari retains a small benefit over radar-only under the fixed protocol, so the multimodal
checkpoint remains the selected portfolio candidate. The difference is too small to support
a broad claim that satellite input is always superior. There is one seed, one month, one
radar site and no statistical uncertainty estimate or multi-month replication.

Test errors are lower than validation errors for all three methods. This can reflect different
weather/sample difficulty across date blocks and does not indicate leakage by itself. The
split remains temporal and the test set was not used for fitting, early stopping or model
choice. No threshold, rain-rate conversion or lightning label is involved.

## Audit artifacts

`output/forecasting/final_evaluation/` contains:

- `final_test_summary.json`: global metrics, policy, commit and artifact hashes;
- `final_test_samples.csv`: 519 rows (173 pairs × three methods);
- `method_comparison.csv`: the three global method rows;
- `checkpoint_audit.json`: frozen checkpoint/config/source fingerprints.

The audit confirms identical pair order and metric domain for all methods. Both checkpoints
were loaded with `weights_only=True`. The artifacts and model files remain outside Git.

## Report figures

`scripts/17_plot_forecast_results.py` reads only the frozen training/evaluation artifacts and
creates two report-ready figures: the complete and zoomed validation RMSE trajectories, and
the validation-versus-locked-test MAE/RMSE comparison. The canonical generated files are kept
outside Git in `output/forecasting/report_figures_v2/`. Plotting changes presentation only;
they do not load training data, update a checkpoint or alter any scientific configuration.
