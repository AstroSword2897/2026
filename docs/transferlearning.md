# Transfer learning

This document describes how transfer learning is used in MaxSight: tier-to-tier transfer, checkpoint loading, and fine-tuning.

## Overview

Transfer learning lets you reuse weights from a trained model (e.g. T2) when training a larger or different model (e.g. T5). It speeds up training and can improve stability. MaxSight supports this via a tier transfer manager and by loading checkpoints into models with compatible (or partially compatible) state dicts.

## TierTransferManager

- **`ml/training/transfer_learning.py`:** Defines **TierTransferManager**, which handles transfer from one tier’s checkpoint to another (e.g. T2 → T5).
- **Initialization:** It takes a source checkpoint path, the target model (e.g. T5), and a transfer config. The source checkpoint must exist and contain at least `model_state_dict`, `epoch`, and `val_loss`.
- **Validation:** **validate_source_checkpoint()** checks that required keys exist and that the state dict has no NaNs. It may warn if the source epoch count is low (e.g. &lt; 50).
- **Weight transfer:** **transfer_weights(strict=False)** loads the source state dict and copies weights into the target model for matching parameter names. Mismatched or missing keys are skipped when `strict=False`. Return value can report how many parameters were transferred.
- **Transfer patterns:** The manager targets shared components such as backbone, FPN, ViT blocks, attention modules, detection/classification/box/distance heads, etc. New or renamed modules in the target (e.g. temporal encoder in T5) are left randomly initialized unless they share names with the source.

## Checkpoint format

- Checkpoints are typically saved as `.pt` or `.pth` with at least:
  - `model_state_dict`: state dict of the model
  - `epoch`: training epoch
  - `val_loss`: validation loss (optional but used for validation)
- Loading is done with `torch.load(..., map_location='cpu', weights_only=True)` (or equivalent) to avoid loading onto GPU by default and to stay compatible with safe loading.

## Using transfer in training

1. **Train a source model** (e.g. T2) to a reasonable epoch count and save the checkpoint.
2. **Create the target model** (e.g. T5) with the desired tier config.
3. **Instantiate TierTransferManager** with the source checkpoint, target model, and config.
4. **Call validate_source_checkpoint()** to ensure the source is valid.
5. **Call transfer_weights()** to copy compatible weights into the target model.
6. **Train the target model** (optionally with a lower learning rate or frozen backbone for a few epochs). The rest of the training loop is unchanged.

## Fine-tuning and partial loading

- **Fine-tuning:** After transfer, you can fine-tune all parameters or only the new heads (e.g. temporal, retrieval) by setting `requires_grad=False` on the transferred backbone and training only the rest. This is script-specific; the transfer module itself only copies weights.
- **Partial load in scripts:** Training or export scripts often load a checkpoint with `model.load_state_dict(state, strict=False)` so that only matching keys are loaded. This allows loading a T2 checkpoint into a T5 model for inference or further training even without using TierTransferManager explicitly.

## Best practices

- Validate the source checkpoint before long runs. Use `validate_source_checkpoint()` or equivalent checks (required keys, no NaNs).
- Prefer transferring from a model trained long enough (e.g. ≥50 epochs) so representations are stable.
- After transfer, consider lower learning rates or a short warmup so the new heads adapt without destroying transferred features.
- When adding new modules (e.g. temporal encoder), ensure their names do not conflict with the source state dict; they will remain randomly initialized after transfer.

## Related code

- **Transfer manager:** `ml/training/transfer_learning.py`
- **Training loop and checkpointing:** `ml/training/train_loop.py`, `scripts/train_maxsight.py`
- **Model creation and tier config:** `ml/models/maxsight_cnn.py` (e.g. `create_model()`, `TierConfig.for_tier()`)

For training setup and data, see `docs/training_architecture.md` and `docs/training-data-loading.md`. For model tiers and architecture, see `docs/architecture.md`.
