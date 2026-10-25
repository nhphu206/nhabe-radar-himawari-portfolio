# Single-sample inference

The selected portfolio checkpoint is the frozen multimodal Small U-Net from epoch 23. The
inference command predicts one reflectivity field ten minutes after the input radar midpoint.
It consumes an audited manifest row, the current Signature-A radar array and the separately
selected previous-cycle Himawari array. It does **not** open the future radar target array.

```bash
./.venv/bin/python scripts/18_predict_one_sample.py \
  --data-root /path/to/research-workspace \
  --manifest /path/to/forecast_pairs.csv \
  --normalization /path/to/normalization.json \
  --checkpoint /path/to/multimodal/best_model.pt \
  --pair-id PAIR_ID \
  --output /new/empty/output/directory \
  --device cpu
```

The program verifies the manifest, train-only normalization, model configuration and
checkpoint fingerprints before inference. The checkpoint is opened with PyTorch's
`weights_only=True` loader. The output directory must be empty so an earlier result cannot be
silently overwritten.

## Output contract

The NPZ contains exactly two arrays:

| Key | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `predicted_reflectivity_dbz` | 256 × 256 | float32 | Ten-minute reflectivity prediction in dBZ |
| `prediction_valid_mask` | 256 × 256 | bool | Pixels supported by valid input radar data |

Predictions outside the input-radar mask are serialized as NaN, never as zero. The model does
produce a finite numerical output at every grid location internally, but the stored mask makes
the conservative validity domain explicit. The companion JSON records pair/sample identity,
input and forecast-valid times, checkpoint/config hashes, device/runtime, valid-pixel count,
descriptive value statistics and the NPZ hash. The command reopens both files and validates
shape, dtype, hash, keys and exact NaN–mask agreement before returning `PASS`.

## Verified local example

The validation pair
`NHB250822004007_RAWNDW4__to__NHB250822005007_RAWNDWB` was run on CPU. Its input midpoint
is `2025-08-22T00:41:23Z`, and the forecast-valid time is exactly ten minutes later. The
product has 7,292 valid pixels (11.1267% of the grid), float32 dBZ values and a boolean mask;
the roundtrip audit passed. Runtime for the model forward pass was approximately 0.079 seconds
on the tested local machine. Runtime is hardware-specific and is not an operational latency
claim.

This remains offline research inference. The CLI starts from an already audited pair manifest;
it does not discover newly arriving observations, measure upstream availability, expose an API,
or provide production monitoring. Test targets may exist in the research manifest for later
evaluation, but the inference input path never reads them.
