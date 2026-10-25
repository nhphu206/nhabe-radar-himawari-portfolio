# Forecast training journal

This journal records what was actually run, why it was run, the resulting evidence and what
remains unknown. It separates pipeline diagnostics from official experiments so smoke losses
cannot be mistaken for model skill.

## Day 17 — small-subset overfit gate

Planned gate: cache the first 16 audited training pairs, train the multimodal Small U-Net for
25 epochs with AdamW, and require the final same-subset masked MSE to be no more than 50% of
its random-initialization value. The purpose is to verify data → model → masked loss →
gradient → optimizer behavior before any official training. It uses no validation/test data
and writes no checkpoint.

Configuration: seed 2025, batch size 2, learning rate 0.001, weight decay 0.0001, gradient
clip norm 1.0, target-valid masked MSE, Apple MPS when available. Results will be appended
only after the run completes and its artifacts reopen successfully.

### First run: failed metric transfer diagnostic

The first 25-epoch attempt produced NaN in the recorded evaluation MSE. The run was kept at
`output/forecasting/overfit/multimodal_16/` with status FAIL. A separate read-only diagnostic
showed that model predictions and selected errors were finite. The corrupt values appeared
only when one `.to(...)` call simultaneously transferred an MPS tensor to CPU and converted
float32 to float64. For example, batch 1 produced infinite squared-error sum with the combined
operation but `767205.5466` when transfer and conversion were separate operations.

The metric path was changed to `tensor.detach().cpu().to(torch.float64)`: transfer first,
cast second. Training/evaluation math, source samples and scientific configuration were not
changed. The failed directory will not be overwritten; the corrected attempt uses a new
versioned output directory.

### Second run: learning succeeded, gate bookkeeping rejected it

The corrected metric run at `multimodal_16_v2/` reduced evaluation MSE from `69.030397` to
`0.166506` in 25 epochs, a final/initial ratio of `0.002412`. Epoch 24 was slightly lower at
`0.165023`. The script incorrectly required both the documented ratio threshold and the last
epoch to equal the exact minimum, so it wrote FAIL even though the published ≤0.5 gate passed.
Small minibatch optimization can oscillate slightly; an exact final-equals-best condition was
never part of `training.yaml` or the planned gate. The script was corrected to evaluate only
the configured final/initial ratio and to report best epoch separately. A third versioned run
will produce the canonical gate artifact.

### Canonical third run: PASS

The canonical `multimodal_16_v3/` run reproduced the same history and passed. Initial MSE
was `69.030397`, final MSE `0.166506`, best MSE `0.165023` at epoch 24, and final/initial
ratio `0.002412` against the required maximum `0.5`. Runtime was 21.84 seconds on Apple MPS.
All 16 pair identifiers and input/config fingerprints are stored in the summary. No
checkpoint was written because this run intentionally memorizes a tiny subset and must not
be presented as a forecasting model.

Conclusion: the model can receive real tensors, propagate masked loss, update parameters and
substantially reduce error. This clears the overfit gate for official train/validation runs.

## Official training protocol

The two controlled official runs use the same seed, architecture family, optimizer, learning
rate, weight decay, batch size and split. Only input mode changes: radar-only has two channels;
multimodal has eight. Training uses all 508 train pairs. Model selection uses all 141
validation pairs and global RMSE in dBZ on the common input-radar/target mask, matching the
persistence domain. The 173 test pairs remain unopened.

AdamW uses learning rate 0.001 and weight decay 0.0001; batch size is 4; maximum is 40 epochs;
gradient norm is clipped to 1.0. The best validation checkpoint is saved whenever RMSE
improves by more than 0.001 dBZ. Training stops after seven consecutive non-improving epochs.
The checkpoint, history and summary remain outside Git. These are fixed in
`configs/training.yaml` before the official runs begin.

### Radar-only attempt v1: training finished, artifact reopen failed

Radar-only v1 trained through epoch 30 and reached its best validation RMSE of approximately
`6.2117 dBZ` at epoch 23 before early stopping. The final checkpoint reopen failed because
`torch.__version__` had been stored as PyTorch's `TorchVersion` object. The restricted
`weights_only=True` loader correctly rejected that non-allowlisted Python type. Loading with
unsafe pickle was not used.

The v1 directory is retained as a failed attempt and is not the canonical result. Checkpoint
provenance now stores `str(torch.__version__)`; save/load moved into tested helpers; a unit
test reopens the exact checkpoint schema with `weights_only=True`. Training history is now
written atomically after every epoch, so a later artifact failure cannot erase epoch records.
Radar-only will be rerun from seed 2025 in a new `radar_v2/` directory.

## Official run 1 — radar-only v2: PASS

Radar-only v2 completed 30 epochs and stopped after patience reached seven. The best
checkpoint is epoch 23. It reopens with `weights_only=True` and reproduces validation MAE
`4.494850 dBZ` and RMSE `6.211665 dBZ` on 1,139,645 common-valid pixels. Target-mask
normalized MSE is `0.305170`. Runtime was 646.50 seconds on Apple MPS. The run used all 508
train pairs and 141 validation pairs; `test_split_opened=false`.

Compared with persistence, validation MAE improves 18.86% and RMSE improves 22.71%.

## Official run 2 — radar + Himawari v1: PASS

Multimodal v1 also completed 30 epochs and selected epoch 23. The checkpoint reopens safely
and reproduces validation MAE `4.421153 dBZ` and RMSE `6.039912 dBZ`; target-mask normalized
MSE is `0.291045`. Runtime was 823.12 seconds on Apple MPS. It used exactly the same pair
partitions, metric domain, seed and training policy as radar-only. Only the input channels and
the resulting 864 first-layer parameters differ.

Compared with persistence, validation MAE improves 20.19% and RMSE improves 24.84%.
Compared with radar-only, validation MAE improves 1.64% and RMSE improves 2.77%.

## Milestone conclusion

The official training milestone is complete. Both canonical summaries, histories and
checkpoints passed hash and reopen checks. The multimodal validation advantage is small and
must not be described as final superiority: this is one seed, one month and one radar site.
All 173 test pairs remain locked for the next separately documented final-evaluation step.

## Day 19 — locked-test protocol prepared

Before reading any test NPZ, the final evaluator was implemented and covered by synthetic
tests. It requires a clean Git worktree, records the exact source commit, rehashes the
manifest/normalization/configs/checkpoints/histories, and loads checkpoints with
`weights_only=True`. It evaluates persistence, radar-only U-Net and multimodal U-Net on the
same ordered test pairs and common input-radar/target mask.

The frozen outputs are `final_test_summary.json`, `final_test_samples.csv`,
`method_comparison.csv` and `checkpoint_audit.json`. Source and docs must be committed before
the evaluator is allowed to construct a test Dataset. No parameter or model selection change
is allowed after results are visible.
