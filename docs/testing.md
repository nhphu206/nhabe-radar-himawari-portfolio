# Test coverage

The preprocessing release suite contains 21 test methods: five original contract checks,
twelve boundary/error checks, and four CLI subprocess checks. The project owner confirmed all 21 pass with the
local research interpreter and again inside the rebuilt ARM64 Docker image. The expanded
suite also passed both hosted CI jobs in
[run #3](https://github.com/nhphu206/nhabe-radar-himawari-portfolio/actions/runs/38147133453),
as confirmed by the project owner's screenshot. Docker checks took 1 minute 40 seconds.

The working forecasting extension adds 46 methods (17 exact-pair/split checks, 14
header/timeline/CLI checks and 15 normalization checks), for **67 local tests**, all passing
on October 20, 2026. The 67-test suite has not yet been verified in Docker or hosted CI.
The image's explicit configuration copies include forecasting/normalization policies.

Twenty-one additional Dataset/DataLoader, masked-metric, U-Net and training tests pass in the dedicated PyTorch environment
on October 24, 2026. They are kept in `tests_ai/` so the preprocessing-only Docker image and
release suite do not silently acquire the much larger PyTorch dependency.

```bash
python -m unittest discover -s tests -v

# Dedicated AI environment
./.venv/bin/python -m unittest discover -s tests_ai -v
```

| Test file | Purpose |
| --- | --- |
| `test_pipeline.py` | Common-grid geometry, A/B identification, timing tie-break, basic sample I/O |
| `test_contract_errors.py` | Missing config groups/arrays, unknown scans, timing boundaries, invalid shapes/types/masks, sample metadata and serialization |
| `test_cli.py` | Real subprocess CLI help, invalid command exit code, step argument forwarding and mock generation from another directory |
| `test_forecast_pairs.py` | Exact 600-second A-only targets, unique pairs, same-day policy, missing paths and malformed index metadata |
| `test_forecast_splits.py` | Date boundaries and sample/group reuse across partitions |
| `test_forecast_timeline.py` | Synthetic HSD parsing, ten earlier-source headers, invalid/missing/future sources, satellite split isolation and CLI roundtrip/no-overwrite |
| `test_forecast_normalization.py` | Train-input-only reads, streaming population moments, unique sources, per-band masks, invalid/constant fields, source hashes and CLI artifacts |
| `tests_ai/test_forecast_dataset.py` | PyTorch shapes/dtypes, correct normalization, in-memory missing-value padding, masks, manifest provenance, exact lead time and DataLoader collation |
| `tests_ai/test_forecast_evaluation.py` | Known masked MAE/MSE/RMSE values, gradient flow, global pixel weighting and invalid metric inputs |
| `tests_ai/test_unet.py` | Repository model config, two input channel modes, output shape, unrestricted regression, masked backward gradients and invalid geometry |
| `tests_ai/test_training.py` | Optimizer updates, overfit gate, validation mask/units, safe checkpoint roundtrip and fixed training configuration |

Negative tests intentionally supply invalid data. A test passes when the code rejects that
data with the expected error, not when invalid data is accepted. The timing tests distinguish
the inclusive ±180-second boundary from ±180.001 seconds. Serialization tests preserve negative
dBZ, valid zero values, and NaNs, and check that temporary files do not remain after success.

CLI tests run using the current interpreter and the absolute wrapper path, with an unrelated
temporary working directory. Generated mock arrays and deliberately invalid configurations
remain in temporary directories. No raw radar/HSD data or audited monthly samples are used.

These checks cover the exercised data contract; they are not a complete schema validator,
weather-quality assessment, or end-to-end validation of real observation readers/gridding.
