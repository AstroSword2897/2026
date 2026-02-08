# Project status and health

This document summarizes the current status of the MaxSight repo: what is implemented, what is tested, and what to watch when running or deploying.

## Implementation status

- **Model (MaxSightCNN):** Implemented in `ml/models/maxsight_cnn.py` with tiered capabilities (T0–T5). Backbone (ResNet50 + FPN), hybrid ViT, temporal encoder, and 30+ heads are present and gated by tier and condition.
- **Training:** Training loop, losses, task balancing (e.g. GradNorm), and validation live in `ml/training/`. Scripts such as `scripts/train_maxsight.py` and `scripts/smoke_train.py` are the main entry points. Data pipeline and dataset are in `ml/data/`.
- **Export:** JIT, ExecuTorch (.pte), CoreML, and ONNX export are in `ml/training/export.py`. iOS bundle export and top-7 deploy are supported via `scripts/deploy_top7.py` and `scripts/export_top7_to_xcode.py`.
- **Therapy:** Session management, task generation, and therapy integration are in `ml/therapy/`. See `docs/therapy_system.md`.
- **Retrieval:** Encoders, indexing, and two-stage retrieval are in `ml/retrieval/`. Used when the tier enables retrieval.
- **Simulation and tooling:** Simulation, quantization, and benchmarking live under `tools/` and `ml/training/benchmark.py`.

## Tests

- **Phase tests:** `tests/test_phase0_backbone.py` through `test_phase5_training.py` cover backbone, fusion, heads, retrieval, knowledge, and training wiring.
- **Integration and other:** `tests/test_model.py`, `tests/test_comprehensive_system.py`, and related files cover model creation, forward passes, and integration. Run with `pytest tests/`.
- **Benchmark:** `python -m ml.training.benchmark` runs inference benchmarks.

## Known limitations and risks

- **JIT / ExecuTorch export:** Tracing can hit unsupported ops or segfaults (e.g. with CLIP or scene graph). The export pipeline stubs `global_encoder` (CLIP) when needed; if export still exits (e.g. 139), the failure may be in another submodule or the environment. Using CPU and JIT-only (`--device cpu`, `--quick`) often improves stability.
- **Checkpoints and conditions:** Top-7 export expects checkpoints under `checkpoints_<condition>/best_model.pt`. If a condition is missing, that condition is skipped. Use `scripts/find_trained_checkpoints.py` to discover paths.
- **Data:** Training and evaluation assume COCO (or compatible) annotations and image layout. Incorrect paths or missing annotations will cause failures; verify with the data pipeline or download docs (`docs/downloads.md`).

## Branch and deployment

- **Default branch:** Check your current branch (e.g. `feature/multimodal_refactor`). Main deployment and export instructions in README apply to the state on that branch.
- **CI:** `.github/workflows/ci.yml` runs lint and tests on push/PR. Green CI indicates tests and lint pass for that commit.

## How to use this doc

- **Before training:** Ensure data is prepared (see `docs/downloads.md`, `docs/training-data-loading.md`) and that you have the right annotation and image paths.
- **Before export:** Run a quick validation (e.g. `scripts/deploy_top7.py --validate-only`) and use CPU + JIT-only if you see crashes.
- **After changes:** Run `pytest tests/` and, if relevant, `python -m ml.training.benchmark` to confirm nothing regressed.

For deeper technical detail, see `docs/architecture.md`, `docs/training_architecture.md`, and `docs/therapy_system.md`.
