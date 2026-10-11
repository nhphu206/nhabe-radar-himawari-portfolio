# Test coverage

The suite contains 21 test methods: five original contract checks, twelve boundary/error
checks, and four CLI subprocess checks. The project owner confirmed all 21 pass with the
local research interpreter and again inside the rebuilt ARM64 Docker image. The expanded
suite's hosted CI run remains to be verified after commit/push.

```bash
python -m unittest discover -s tests -v
```

| Test file | Purpose |
| --- | --- |
| `test_pipeline.py` | Common-grid geometry, A/B identification, timing tie-break, basic sample I/O |
| `test_contract_errors.py` | Missing config groups/arrays, unknown scans, timing boundaries, invalid shapes/types/masks, sample metadata and serialization |
| `test_cli.py` | Real subprocess CLI help, invalid command exit code, step argument forwarding and mock generation from another directory |

Negative tests intentionally supply invalid data. A test passes when the code rejects that
data with the expected error, not when invalid data is accepted. The timing tests distinguish
the inclusive ±180-second boundary from ±180.001 seconds. Serialization tests preserve negative
dBZ, valid zero values, and NaNs, and check that temporary files do not remain after success.

CLI tests run using the current interpreter and the absolute wrapper path, with an unrelated
temporary working directory. Generated mock arrays and deliberately invalid configurations
remain in temporary directories. No raw radar/HSD data or audited monthly samples are used.

These checks cover the exercised data contract; they are not a complete schema validator,
weather-quality assessment, or end-to-end validation of real observation readers/gridding.
