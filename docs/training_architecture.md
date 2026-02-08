# Training architecture

This document describes how training is structured: loop, losses, balancing, validation, and configuration.

## Overview

Training is implemented in `ml/training/` and driven by scripts such as `scripts/train_maxsight.py` and `scripts/smoke_train.py`. The loop loads data via the data pipeline, runs forward and backward passes, computes a multi-task loss, and optionally runs validation and checkpointing.

## Training loop

- **`ml/training/train_loop.py`:** Core training loop (e.g. ProductionTrainLoop or equivalent). Iterates over epochs and batches, calls the model, computes loss, backpropagates, and updates optimizer. May support gradient accumulation, mixed precision, and checkpointing.
- **Scripts:** `train_maxsight.py` parses args (annotations, image dir, device, epochs, tier, condition), builds data loaders and model, and runs the loop. `smoke_train.py` runs a short training run to verify the pipeline.

## Losses

- **`ml/training/losses.py`:** Defines per-head losses (classification, box regression, objectness, urgency, distance, etc.) and combiners. Total loss is typically a weighted sum over heads.
- **`ml/training/head_losses.py`:** Head-specific loss helpers (e.g. for detection, therapy heads).
- **`ml/training/task_balancing.py`:** Task balancing (e.g. GradNorm) to weight multiple tasks so no single task dominates. Used when training many heads jointly.
- **`ml/training/matching.py`:** Label assignment for detection (e.g. Hungarian matcher) so targets are matched to predictions for loss computation.

## Validation and metrics

- **`ml/training/validation.py`:** Validation step: run model on val data, compute metrics (e.g. loss, mAP).
- **`ml/training/metrics.py`:** Metric aggregation (precision, recall, mAP, etc.).
- **`ml/training/evaluation.py`:** Evaluation reports (e.g. lighting-aware or condition-specific metrics).
- **`ml/training/scene_metrics.py`:** Scene-level or retrieval-related metrics if used.

## Configuration

- **Tier and condition:** Training uses a capability tier (T0–T5) and optionally a condition mode. These control which parts of the model are active and how preprocessing behaves.
- **Config files:** `ml/training/configs/` holds YAML or similar configs (e.g. for tiers, learning rates, data paths). Scripts may load these or take overrides from the command line.
- **Hyperparameters:** `scripts/AutoMLType.py` (Optuna-based) can search hyperparameters and write a best config (e.g. `best_hyperparameters.json`) for use in production training.

## Checkpointing and resume

- Checkpoints are usually saved under `checkpoints/` or `checkpoints_<condition>/` (e.g. `best_model.pt`, `last_checkpoint.pt`). Format is typically a dict with `model_state_dict`, `optimizer_state_dict`, `epoch`, and optional `val_loss`.
- Resume is supported by loading a checkpoint and restoring model (and optionally optimizer) before continuing the loop. Scripts may accept `--resume-from` or similar.

## Stability and regularization

- **`ml/training/stability_manager.py`:** Helps stabilize training (e.g. gradient clipping, loss scaling).
- **`ml/training/regularization.py`:** Regularization (e.g. weight decay, auxiliary losses).
- **`ml/training/quantization.py`:** Quantization-aware training if you are exporting quantized models.

## Summary

Training is a standard PyTorch loop with multi-task loss, task balancing, and validation. Entry points are `train_maxsight.py` and `smoke_train.py`; core logic is in `ml/training/` (train_loop, losses, task_balancing, validation, metrics). Use configs and AutoMLType for hyperparameters and tier/condition setup. For data loading and dataset layout, see `docs/training-data-loading.md`; for transfer from one tier to another, see `docs/transferlearning.md`.
