# Train-only normalization

The working forecasting extension fits six scalers from the **accepted training inputs**,
not from the complete preprocessing dataset. The protocol is in `configs/normalization.yaml`;
the implementation is `src/nhabe_forecasting/normalization.py` and script 10.

## Population and masks

- Select rows with `split=train` from the completed observation-timing manifest.
- Radar: unique `input_sample_id` sources, using `radar_data_valid_mask`.
- Satellite: unique `satellite_sample_id` sources, using each band's validity mask.
- Do not append target-only sources or read validation/test arrays for fitting.
- Deduplicate each role independently; conflicting IDs, paths or fingerprints fail closed.
- Weight valid pixels equally, not each image equally. Keep negative and zero dBZ.
- Validate float32 fields, boolean masks, NaN consistency and satellite all-band mask.
- Reject empty fields, non-finite statistics, and zero standard deviation.

A source can be a target in another training pair and still legitimately be an input here.
Selection is based on the input role, not a prohibition on every file that ever appears
as a target. A single NPZ can supply radar for one pair and satellite for another: each
physical field is counted once per unique source within its role.

This counts valid pixels for scaling, not independent meteorological examples. Pixel
counts must not be presented as millions of independent training samples.

## Formula

For valid training values of each field, use population standard deviation (`ddof=0`):

\[
\mu=\frac{1}{N}\sum_{k=1}^{N}x_k,\qquad
\sigma=\sqrt{\frac{1}{N}\sum_{k=1}^{N}(x_k-\mu)^2}.
\]

Here N is that field's valid-pixel count, not its number of images. The implementation
merges float64 batch means and centered squared deviations without stacking all images.
Population standard deviation and precision choices follow the definitions described in
[NumPy's standard-deviation documentation](https://numpy.org/doc/stable/reference/generated/numpy.std.html).

Later, transform `z=(x-mean)/std` and invert `x=z*std+mean`. Reuse the radar input train
statistics for radar targets/predictions. Do not fit a different scaler to target-only
images or refit on validation/test/inference inputs. The mean of dBZ here is a feature
scaling statistic, not a rainfall estimate or a linear-reflectivity average.

Physical NPZ arrays remain unnormalized with NaNs. Padding missing values to zero is a
later in-memory Dataset operation after normalization, accompanied by masks. Masks are
not standardized. No normalized dataset files are written by this stage.

## Verified local results (October 20, 2026)

508 train pairs reference 508 distinct radar inputs and 508 distinct satellite inputs.
There are 1,016 source-role records across 909 physical NPZ paths.

| Field | Valid pixels | Mean | Std | Unit |
| --- | ---: | ---: | ---: | --- |
| Radar reflectivity | 5,607,919 | 8.382429 | 11.441322 | dBZ |
| B08 | 33,273,492 | 229.124144 | 8.139035 | K |
| B13 | 33,273,492 | 267.501815 | 25.988496 | K |
| B14 | 33,273,492 | 265.261867 | 26.045386 | K |
| B15 | 33,273,492 | 261.517266 | 24.909850 | K |
| B16 | 33,273,492 | 250.584941 | 19.192389 | K |

Rounded numbers above are for reading only; use full-precision values in the generated
JSON. Equal band counts do not establish that all band masks are spatially identical.
Train radar valid values span approximately -23.14 to 61.89 dBZ; no threshold was applied.

Independent revalidation confirmed source-role selection matches only train inputs,
all source NPZ/JSON checksums remain unchanged, and normalized valid train pixels have
absolute mean below 1e-10 and standard deviation within 1e-10 of one for all six fields.
This is a scaler check, not evidence of forecast skill or normalized validation/test behavior.

## Output schema and reproducibility

Default output: `output/forecasting/normalization/` (ignored by Git).

| Artifact / field | Meaning |
| --- | --- |
| `normalization.json` | Schema `nhabe-train-normalization-v1`, status, policy, band order, source counts and six field statistics |
| `fields.<name>` | Valid count, mean, std, min, max and original physical units |
| `radar_target_transform` | Reuse radar input train mean/std |
| `provenance` | Manifest/summary/config/source-manifest/code SHA256 and Python/NumPy versions |
| `normalization_sources.csv` | Role, sample ID, NPZ/JSON paths and full-file SHA256 for each selected source |

The CLI verifies the manifest checksum, completed full-scope timing audit and forecasting
config fingerprint before fitting. It verifies source JSON hashes, detects NPZ size/mtime
changes during reads, writes artifacts atomically and reopens them. Existing nonempty
output directories and protected observation/preprocessing locations are not overwritten.

```bash
python scripts/10_compute_train_statistics.py \
  --data-root /path/to/research-workspace \
  --output output/forecasting/normalization
```

Use the absolute script path when running from a different directory. No PyTorch, raw
decoding, satellite downloads, preprocessing rerun, or model training is required.

## Tests and remaining work

15 normalization tests cover streaming moments, pixel weighting, negative/zero/NaN values,
per-band masking, deduplication, invalid inputs, zero std, metadata mutation, leakage,
absence of validation/test/target-only reads, CLI roundtrip and overwrite protection.
The full local suite has 67 passing tests. Docker and hosted CI for these changes are
not yet verified. Dataset tensor transformation, model training and evaluation are future stages.
