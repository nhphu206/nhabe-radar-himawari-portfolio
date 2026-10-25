# Official training results

This page records the first official controlled training runs. “Official” means the runs use
the complete predefined train and validation partitions, fixed configuration, checkpoint
selection and reproducible artifacts. It does not mean production deployment or final
locked-test performance.

## Controlled configuration

Both Small U-Net models use seed 2025, base width 16, AdamW, learning rate 0.001, weight
decay 0.0001, batch size 4, target-valid masked MSE and gradient clipping at 1.0. Training
uses 508 pairs. Checkpoint selection uses RMSE in dBZ over 1,139,645 common-valid pixels in
141 validation pairs. Early stopping patience is seven epochs with minimum improvement
0.001 dBZ. The only experimental variable is input mode:

- radar-only: normalized reflectivity and radar-valid mask, two channels;
- multimodal: radar, B08/B13/B14/B15/B16, radar mask and all-band mask, eight channels.

## Validation comparison

| Method | Parameters | Best epoch | MAE (dBZ) | RMSE (dBZ) | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| Persistence | 0 | – | 5.539317 | 8.036566 | – |
| Radar-only Small U-Net | 482,593 | 23 | 4.494850 | 6.211665 | 646.50 s |
| Radar + Himawari Small U-Net | 483,457 | 23 | 4.421153 | 6.039912 | 823.12 s |

Relative to persistence, radar-only reduces MAE by 18.86% and RMSE by 22.71%.
Multimodal reduces MAE by 20.19% and RMSE by 24.84%. Relative to radar-only, multimodal
reduces validation MAE by 1.64% and RMSE by 2.77%.

Both runs stopped at epoch 30 after seven epochs without sufficient improvement following
their best epoch 23. Radar-only used 10 minutes 46.5 seconds; multimodal used 13 minutes
43.1 seconds on Apple MPS. Runtime includes validation and checkpoint handling and is specific
to this machine.

## Interpretation

Both learned models outperform persistence on the same validation domain. In this single
controlled run, adding the five Himawari bands produces a small additional validation gain.
The gain is evidence that the multimodal candidate is worth carrying forward; it is not yet
evidence of statistically reliable superiority. Only one seed, one month and one site are
available, and the locked test set has not been opened.

The common-valid domain makes comparison with persistence fair, because persistence is only
defined where the current radar observation exists. Training loss uses all target-valid
pixels, while model selection and the table above use the stricter common mask. No dBZ
threshold, rainfall conversion or storm-class filter is applied.

## Artifact verification

Each canonical run contains `best_model.pt`, `training_history.csv` and
`training_summary.json` outside Git. Both summaries have status PASS and
`test_split_opened=false`. Checkpoints reopen with PyTorch's restricted
`weights_only=True` loader and reproduce the recorded best validation metric. SHA256 values
cover the checkpoint, history, manifest, normalization artifact, model config and training
config.

Two failed diagnostic histories are intentionally retained outside Git: the first overfit
run exposed an MPS transfer/cast issue, and radar-only v1 exposed an unsafe checkpoint
metadata type. They are documented in the journal and are not reported as model results.

## Remaining evaluation boundary

The official training milestone is complete. The 173-pair test split remains locked until
the user explicitly begins final evaluation. No inference API, production monitoring,
probabilistic uncertainty or multi-month generalization claim is part of this result.
