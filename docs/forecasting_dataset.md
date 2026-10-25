# Forecast Dataset contract

This is a model-input adapter over the audited pair manifest. It does not create new
meteorological products, edit NPZ files, or train a model.

## Source roles

Each row reads three distinct paths:

| Manifest column | Role |
| --- | --- |
| `input_npz_path` | Current Signature A radar reflectivity |
| `satellite_npz_path` | Previous nominal Himawari cycle that passed the header timing audit |
| `target_npz_path` | Signature A radar reflectivity exactly 600 midpoint seconds later |

Using the Himawari array stored beside the current radar input would violate the approved
forecast protocol. The Dataset therefore never infers one source path from another.

## Returned tensors

For a batch size `B`:

| Key | Radar-only shape | Multimodal shape | dtype |
| --- | --- | --- | --- |
| `inputs` | `(B, 2, 256, 256)` | `(B, 8, 256, 256)` | `torch.float32` |
| `target` | `(B, 1, 256, 256)` | `(B, 1, 256, 256)` | `torch.float32` |
| `target_mask` | `(B, 1, 256, 256)` | `(B, 1, 256, 256)` | `torch.bool` |

Radar-only channel order is radar reflectivity and radar valid mask. Multimodal order is
radar reflectivity, B08, B13, B14, B15, B16, radar valid mask and Himawari all-band valid
mask. Channel order is exposed as `dataset.channel_names`.

Each physical field uses `(x - train_mean) / train_std` with the full-precision values in
`normalization.json`. Radar input and target share radar-input training statistics. Each
Himawari band uses its own statistics and per-band validity mask.

Missing physical values remain NaN in the source NPZ. After normalization, the Dataset
places zero at an invalid position in the in-memory tensor. A normalized zero means the
training mean, so masks are required to distinguish a missing value. Target loss must be
computed only where `target_mask` is true.

When only some satellite bands are valid at a pixel, valid bands are retained and invalid
bands are zero-padded. The all-band mask is zero there. The current 8-channel contract does
not add five separate mask channels; changing that is a model-input experiment, not a silent
Dataset change.

## Safety checks

Dataset construction checks the normalization schema, band order, statistics and manifest
SHA256 provenance. Every row must be Signature A, `PASS`, have ten audited HSD headers and
an exact 600-second radar midpoint lead. Array reads enforce float32 physical data, boolean
masks, fixed shapes, finite-value/mask agreement and all-band/per-band mask agreement.
Manifest paths may not escape the supplied data root.

## Verified smoke run

On October 24, 2026, the first two real training pairs were reopened in both modes. The
multimodal batch was `(2, 8, 256, 256)`, the radar-only batch was `(2, 2, 256, 256)`, and
both targets/masks were `(2, 1, 256, 256)`. All model input and target values were finite.
The manifest contained 508 training pairs. No training was performed.

```bash
./.venv/bin/python scripts/11_smoke_forecast_dataset.py \
  --data-root /path/to/research-workspace \
  --manifest /path/to/forecast_pairs.csv \
  --normalization /path/to/normalization.json \
  --split train \
  --input-mode multimodal \
  --batch-size 2 \
  --batches 2
```
