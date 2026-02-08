# Dataset and asset downloads

This document covers how to obtain and verify datasets and other assets used for training and evaluation.

## Overview

MaxSight training and evaluation rely on annotated vision (and optionally audio) datasets. Scripts and helpers support COCO, Open Images, and other public datasets, plus project-specific splits and annotations.

## Primary scripts and modules

- **`ml/data/download_datasets.py`:** Helpers to verify and download COCO (and related) data. Defines `verify_coco_dataset()` and download/extract logic. Uses paths such as `datasets/coco` or `datasets/coco_raw` (train2017, val2017, annotations).
- **`scripts/gather_training_data.py`:** High-level script to prepare training data: can download COCO, create or use train/val/test splits, and write annotation files (e.g. `maxsight_train.json`, `maxsight_val.json`) in cleaned form.
- **`scripts/download_inference_datasets.py`:** Prepares data used for inference evaluation (e.g. for mAP runs).
- **Open Images:** Scripts such as `download_open_images_direct.py`, `download_open_images_fiftyone.py`, `download_open_images_s3.py` support Open Images in different ways (direct HTTP, FiftyOne, S3). Use one that matches your environment and credentials.

## Directory layout (typical)

- **`datasets/coco_raw/`** (or **`datasets/coco/`**): Raw COCO images and annotations (e.g. `train2017/`, `val2017/`, `annotations/`).
- **`datasets/cleaned_splits/`** (or similar): Project-specific JSON annotations (e.g. COCO-format with `file_name`, `image_id`, boxes, categories) used by `MaxSightDataset` and the data pipeline.
- **`checkpoints/`**: Model checkpoints (e.g. `checkpoints_<condition>/best_model.pt`). Not “downloaded” in the same sense but often synced from another machine or Drive.

## Verification

- **COCO:** Run `verify_coco_dataset()` (or the equivalent checks in `gather_training_data.py`) to ensure train/val image directories and annotation files exist and have expected counts.
- **Annotations:** Training scripts expect annotation paths (e.g. `--train-annotation`, `--val-annotation`) pointing to JSON that matches the dataset class (e.g. `MaxSightDataset` in `ml/data/dataset.py`). Missing or malformed annotations will cause training or validation to fail early.

## Environment and credentials

- **Open Images / S3:** Some download paths require AWS credentials or specific environment variables; see the script you use (e.g. S3 script) for required env vars.
- **Hugging Face:** If any part of the pipeline downloads models or datasets from Hugging Face, set `HF_TOKEN` or the appropriate env for authenticated access and higher rate limits.

## Colab and cloud

- On Colab, data is often on Google Drive. Mount Drive and set `DATA_DIR`, `SPLITS_DIR`, or equivalent to Drive paths so that downloaded or generated datasets persist across sessions.
- `COLAB_DOWNLOAD_DATASETS.py` and Colab-oriented docs (if present in the repo) describe how to run download/setup steps in a notebook.

## Summary

Use `ml/data/download_datasets.py` and `scripts/gather_training_data.py` as the main entry points for dataset download and layout. Verify COCO (or your chosen dataset) with the provided verification helpers, then point training and inference scripts at the correct annotation and image paths. For Open Images or other sources, use the matching download script and the required credentials or env vars.
