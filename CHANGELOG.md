# Changelog

This project records research milestones rather than claiming production readiness.

## Unreleased

- Added a dependency-free interactive FastAPI dashboard for audited validation pairs.
- Kept target-free inference separate from the optional historical observation/error view.
- Added safe pair listing, existing-prediction metadata and validated 256 × 256 PNG endpoints.
- Extended tests and container CI checks for the dashboard and visualization contract.
- Added a one-command Compose demo with explicit synthetic-data and untrained-weight labels.

## v1.0.0 — End-to-end research pipeline

- Built leakage-aware, observation-timed forecast pairs for a fixed ten-minute radar target.
- Fitted normalization from training inputs only.
- Compared persistence, radar-only Small U-Net and radar + Himawari Small U-Net under one
  controlled protocol.
- Opened the 173-pair test set once after model selection and recorded auditable metrics.
- Added target-free single-sample inference with NPZ/JSON validation.
- Added a local FastAPI wrapper and a separate CPU-only AI Docker image.
- Added 30 AI/API tests and a Linux AMD64 CI job using an explicitly synthetic, untrained,
  target-free fixture.

Research result: the multimodal model reached MAE 4.215 dBZ and RMSE 5.904 dBZ on the locked
test set, compared with persistence RMSE 7.408 dBZ and radar-only RMSE 5.956 dBZ. The result is
limited to one seed, one station and August 2025.

## v0.1-data-pipeline — Preprocessing milestone

- Standardized radar and Himawari preprocessing on a shared 256 × 256 grid.
- Validated 8,916 radar volumes, 4,393 available Himawari cycles, 8,774 aligned samples and
  4,387 leakage groups.
- Added synthetic sample validation, Docker, Compose, CI, data dictionaries and licensing.

Raw observations, generated datasets, trained checkpoints and personal reports are excluded
from every release.
