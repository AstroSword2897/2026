# MaxSight Data Module - Dataset loading, downloading, and management for training
# Exports: download_datasets (COCO/AudioSet downloaders), Dataset classes (PyTorch implementations)
# Supports: COCO (80 classes, ~200K images), AudioSet (audio), synthetic data (testing)
# Complexity: O(N) dataset size, but lazy loading - only batches in memory (critical for large datasets)
# Relationship: Provides training data pipeline - required for training on real-world environmental scenes
# Usage: from ml.data.dataset import MaxSightDataset

from .dataset import MaxSightDataset
from .generate_annotations import generate_annotations_from_coco

__all__ = [
    'MaxSightDataset',
    'generate_annotations_from_coco',
]

