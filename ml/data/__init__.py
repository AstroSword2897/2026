# Usage: from ml.data.dataset import MaxSightDataset.

from ml.data.dataset import MaxSightDataset
from ml.data.generate_annotations import generate_annotations_from_coco
from ml.data.data_pipeline import (
    create_data_loaders,
    collate_fn,
    compute_class_weights,
    get_data_info
)

# Production accessibility dataset (therapy-oriented)
# Note: AccessibilityDataset and related functions are available via direct import:.
# From ml.data.create_accessibility_dataset import AccessibilityDataset.
# They are not re-exported here to avoid unused import warnings.

__all__ = [
    'MaxSightDataset',
    'generate_annotations_from_coco',
    'create_data_loaders',
    'collate_fn',
    'compute_class_weights',
    'get_data_info',
]







