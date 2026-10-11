# Synthetic sample workflow

The sample implementation lives in `src/nhabe_preprocessing/sample_pipeline.py`; examples
do not duplicate production code or distribute real radar/Himawari observations.

The prepared CLI can generate a deterministic synthetic NPZ/JSON sample with the production
array shapes, float32 fields, NaN missing values, and boolean masks:

```bash
python /absolute/path/to/repository/scripts/nhabe_pipeline.py sample \
  --output /absolute/path/to/repository/output/sample
```

The workflow passed a local smoke run in the fresh portfolio checkout. It exercises array creation, serialization, and
validation; it does not decode SIGMET or HSD files and does not validate real-data gridding.
Generated artifacts remain under the Git-ignored `output/` directory.
