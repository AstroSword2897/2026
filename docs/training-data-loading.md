# Training data loading

This document describes how training data is loaded, batched, and fed to the model: dataset class, data pipeline, and preprocessing.

## Dataset

- **`ml/data/dataset.py`:** Defines **MaxSightDataset**, the main dataset used for training. It loads images and annotations (COCO-format JSON), applies preprocessing, and returns items with keys such as `images`, `labels`, `boxes`, `distance`, `urgency`, `num_objects`, and optionally `audio` and `condition_mode`.
- **Annotations:** Expected format is COCO-style: images with `file_name`, `id`; annotations with `image_id`, `category_id`, `bbox` (e.g. x, y, w, h), and any extra fields (e.g. distance, urgency). Paths in annotations are resolved relative to a root image directory.
- **Condition mode:** The dataset can be built with a `condition_mode` (e.g. glaucoma, AMD) so that preprocessing and augmentation match that condition.

## Preprocessing

- **`ml/utils/preprocessing.py`:** **ImagePreprocessor** applies condition-specific transforms: normalization (e.g. ImageNet mean/std), resizing (e.g. 224×224), and condition-based augmentation (e.g. contrast, blur, central/peripheral emphasis for different vision conditions). Used inside the dataset or in the data pipeline.
- **Augmentation:** **`ml/data/advanced_augmentation.py`** (and related) provide stronger augmentation for training (e.g. random crop, color jitter, blur). Usually applied only to training, not validation.

## Data pipeline and loaders

- **`ml/data/data_pipeline.py`:** Builds DataLoaders from MaxSightDataset. Key functions:
  - **create_data_loaders()** (or equivalent): Takes train/val annotation paths, image dir, batch size, num workers, and optional condition_mode. Returns train and val DataLoaders.
  - **collate_fn:** Custom collate that stacks images and pads variable-length targets (labels, boxes, distance, etc.) so each batch is a dict of tensors. Handles optional audio.
- **Sampling:** May use **WeightedRandomSampler** for class balancing if configured. Shuffling is typically enabled for training and disabled for validation.

## Typical usage in scripts

1. **Paths:** Scripts (e.g. `train_maxsight.py`) accept `--train-annotation`, `--val-annotation`, `--image-dir` (and optionally `--condition-mode`).
2. **Creation:** Data pipeline creates MaxSightDataset instances for train and val, then wraps them in DataLoader with the appropriate collate_fn, batch size, and workers.
3. **Batch shape:** Batches are dicts: `images` [B, 3, H, W], `labels` [B, max_objects], `boxes` [B, max_objects, 4], `num_objects` [B], `urgency` [B], etc. The model and losses expect these keys and shapes.

## Splits and annotation generation

- **Splits:** Train/val (and optionally test) splits are defined by separate annotation JSON files. They can be produced by `scripts/gather_training_data.py` or by splitting a single COCO JSON (e.g. `ml/data/coco_dataset_splitter.py`).
- **Inference datasets:** For evaluation (e.g. mAP), inference datasets and annotations may be prepared by `scripts/download_inference_datasets.py` or similar; those are separate from the training data pipeline but follow similar path and annotation conventions.

## Best practices

- Ensure annotation paths and image directories are consistent (e.g. `file_name` in JSON matches files under `image_dir`).
- Use enough workers for DataLoader to avoid CPU bottleneck, but not so many that memory is exhausted.
- Keep validation preprocessing consistent with training (same resize, normalization), but usually without heavy augmentation.
- If you add new keys to the dataset (e.g. new targets), update the collate_fn and the model/loss to consume them.

## Summary

Training data loading is centered on **MaxSightDataset** and **create_data_loaders()** in `ml/data/`. Annotations are COCO-format; preprocessing is condition-aware; batching uses a custom collate. Use `--train-annotation`, `--val-annotation`, and `--image-dir` in training scripts and ensure splits are prepared (e.g. via `gather_training_data.py`). For how to obtain and verify datasets, see `docs/downloads.md`; for the training loop and losses, see `docs/training_architecture.md`.
