# Exports: MaxSightCNN (main detection model), create_model (factory function)
# Core of MaxSight system - all training/inference depends on these definitions.
# Complexity: ~29M params, O(H*W*C) forward pass (H/W=image size, C=channels)
# Usage: from ml.models import create_model, MaxSightCNN.

from .maxsight_cnn import MaxSightCNN, create_model, COCO_CLASSES, ACCESSIBILITY_CLASSES

__all__ = [
    'MaxSightCNN',
    'create_model',
    'COCO_CLASSES',
    'ACCESSIBILITY_CLASSES',
]







