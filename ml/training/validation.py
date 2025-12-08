"""
Production-grade validation utilities for MaxSight.

Provides:
- Input validation
- Model validation
- Data validation
- Configuration validation
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def validate_model_inputs(
    images: torch.Tensor,
    expected_shape: Tuple[int, ...] = (4,)
) -> bool:
    """
    Validate model input tensors.
    
        Arguments:
        images: Input image tensor
        expected_shape: Expected tensor dimensions (e.g., (4,) for [B, C, H, W])
    
    Returns:
        True if valid, raises ValueError if invalid
    """
    if not torch.is_tensor(images):
        raise ValueError(f"Images must be a tensor, got {type(images)}")
    
    if images.dim() != len(expected_shape):
        raise ValueError(
            f"Images must have {len(expected_shape)} dimensions, got {images.dim()}"
        )
    
    if images.shape[1] != 3:
        raise ValueError(f"Images must have 3 channels, got {images.shape[1]}")
    
    if images.min() < 0 or images.max() > 1:
        logger.warning(
            f"Image values outside [0, 1] range: min={images.min():.3f}, max={images.max():.3f}"
        )
    
    return True

def validate_model_outputs(
    outputs: Dict[str, torch.Tensor],
    expected_keys: Optional[list] = None
) -> bool:
    """
    Validate model output dictionary.
    
        Arguments:
        outputs: Model outputs dictionary
        expected_keys: Optional list of expected keys
    
    Returns:
        True if valid, raises ValueError if invalid
    """
    if not isinstance(outputs, dict):
        raise ValueError(f"Outputs must be a dictionary, got {type(outputs)}")
    
    if expected_keys:
        missing_keys = set(expected_keys) - set(outputs.keys())
        if missing_keys:
            raise ValueError(f"Missing output keys: {missing_keys}")
    
    for key, value in outputs.items():
        if not torch.is_tensor(value):
            raise ValueError(f"Output '{key}' must be a tensor, got {type(value)}")
    
    return True

def validate_batch(
    batch: Any,
    required_keys: Optional[list] = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Validate and parse training batch.
    
        Arguments:
        batch: Batch from DataLoader
        required_keys: Optional list of required keys in targets
    
    Returns:
        Tuple of (images, targets)
    
    Raises:
        ValueError: If batch is invalid
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) < 1:
            raise ValueError("Batch must contain at least images")
        images = batch[0]
        targets = batch[1] if len(batch) > 1 else {}
    elif isinstance(batch, dict):
        images = batch.get('images') or batch.get('image')
        if images is None:
            raise ValueError("Batch must contain 'images' or 'image' key")
        targets = {k: v for k, v in batch.items() if k not in ['images', 'image']}
    else:
        raise ValueError(f"Unsupported batch format: {type(batch)}")
    
    # Validate images
    validate_model_inputs(images)
    
    # Validate targets if required keys specified
    if required_keys:
        missing_keys = set(required_keys) - set(targets.keys())
        if missing_keys:
            raise ValueError(f"Missing required target keys: {missing_keys}")
    
    return images, targets

def validate_checkpoint(
    checkpoint: Dict[str, Any],
    required_keys: Optional[list] = None
) -> bool:
    """
    Validate checkpoint dictionary.
    
        Arguments:
        checkpoint: Checkpoint dictionary
        required_keys: Optional list of required keys
    
    Returns:
        True if valid, raises ValueError if invalid
    """
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must be a dictionary, got {type(checkpoint)}")
    
    if required_keys:
        missing_keys = set(required_keys) - set(checkpoint.keys())
        if missing_keys:
            raise ValueError(f"Missing checkpoint keys: {missing_keys}")
    
    # Validate model_state_dict exists
    if 'model_state_dict' not in checkpoint:
        raise ValueError("Checkpoint must contain 'model_state_dict'")
    
    return True

