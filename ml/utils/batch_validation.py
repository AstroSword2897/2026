"""Batch Validation Utilities Comprehensive validation for training batches to prevent Hungarian matching failures. Checks for NaN/Inf, invalid box dimensions, and other data integrity issues."""

import torch
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


def validate_boxes(boxes: torch.Tensor, name: str = "boxes") -> Tuple[bool, str]:
    """Validate bounding boxes for training."""
    # Checks for NaN/Inf.
    if torch.isnan(boxes).any():
        return False, f"{name} contains NaN values"
    if torch.isinf(boxes).any():
        return False, f"{name} contains Inf values"
    
    # Checks dimensions (width, height > 0)
    if boxes.shape[-1] != 4:
        return False, f"{name} has wrong shape: expected [..., 4], got {boxes.shape}"
    
    widths = boxes[..., 2]
    heights = boxes[..., 3]
    
    if (widths <= 0).any():
        return False, f"{name} has non-positive width: min={widths.min().item()}"
    if (heights <= 0).any():
        return False, f"{name} has non-positive height: min={heights.min().item()}"
    
    # Check centers are in valid range [0, 1].
    centers_x = boxes[..., 0]
    centers_y = boxes[..., 1]
    
    if (centers_x < 0).any() or (centers_x > 1).any():
        return False, f"{name} has out-of-range center_x: min={centers_x.min().item()}, max={centers_x.max().item()}"
    if (centers_y < 0).any() or (centers_y > 1).any():
        return False, f"{name} has out-of-range center_y: min={centers_y.min().item()}, max={centers_y.max().item()}"
    
    # Check sizes are in valid range (1e-4, 1].
    if (widths > 1).any():
        return False, f"{name} has width > 1: max={widths.max().item()}"
    if (heights > 1).any():
        return False, f"{name} has height > 1: max={heights.max().item()}"
    
    return True, ""


def validate_labels(labels: torch.Tensor, num_classes: int, name: str = "labels") -> Tuple[bool, str]:
    """Validate class labels."""
    # Checks for NaN/Inf.
    if torch.isnan(labels).any():
        return False, f"{name} contains NaN values"
    if torch.isinf(labels).any():
        return False, f"{name} contains Inf values"
    
    # Check range.
    if (labels < 0).any():
        return False, f"{name} has negative values: min={labels.min().item()}"
    if (labels >= num_classes).any():
        return False, f"{name} has values >= num_classes: max={labels.max().item()}, num_classes={num_classes}"
    
    return True, ""


def validate_batch(
    batch: Dict[str, Any],
    num_classes: int = 91,
    check_targets: bool = True
) -> Tuple[bool, str]:
    """Comprehensive batch validation before training."""
    # Validate images.
    if 'images' in batch:
        images = batch['images']
        if torch.isnan(images).any():
            return False, "images contain NaN"
        if torch.isinf(images).any():
            return False, "images contain Inf"
    
    # Validate boxes.
    if 'boxes' in batch:
        valid, msg = validate_boxes(batch['boxes'], "boxes")
        if not valid:
            return False, msg
    
    # Validate labels.
    if 'labels' in batch and check_targets:
        valid, msg = validate_labels(batch['labels'], num_classes, "labels")
        if not valid:
            return False, msg
    
    # Validate other tensors.
    for key in ['distance', 'urgency']:
        if key in batch:
            tensor = batch[key]
            if torch.isnan(tensor).any():
                return False, f"{key} contains NaN"
            if torch.isinf(tensor).any():
                return False, f"{key} contains Inf"
    
    return True, ""


def sanitize_boxes(boxes: torch.Tensor, min_size: float = 1e-4) -> torch.Tensor:
    """Sanitize boxes to ensure valid dimensions."""
    boxes = boxes.clone()
    
    # Replace NaN/Inf with defaults.
    if torch.isnan(boxes).any() or torch.isinf(boxes).any():
        mask = torch.isnan(boxes) | torch.isinf(boxes)
        default_box = torch.tensor([0.5, 0.5, 0.1, 0.1], device=boxes.device, dtype=boxes.dtype)
        for i in range(boxes.shape[-1]):
            boxes[..., i] = torch.where(mask[..., i], default_box[i], boxes[..., i])
    
    # Clamp centers to [0, 1].
    boxes[..., 0] = torch.clamp(boxes[..., 0], 0.0, 1.0)
    boxes[..., 1] = torch.clamp(boxes[..., 1], 0.0, 1.0)
    
    # Clamp sizes to [min_size, 1].
    boxes[..., 2] = torch.clamp(boxes[..., 2], min_size, 1.0)
    boxes[..., 3] = torch.clamp(boxes[..., 3], min_size, 1.0)
    
    return boxes


def validate_and_sanitize_batch(
    batch: Dict[str, Any],
    num_classes: int = 91,
    auto_fix: bool = True
) -> Tuple[Dict[str, Any], bool, str]:
    """Validate and optionally sanitize a batch."""
    # Only validate actual objects (not padding) if num_objects is available.
    if 'num_objects' in batch and 'boxes' in batch:
        batch_size = batch['boxes'].shape[0]
        batch_valid = True
        needs_fix = False
        
        for b in range(batch_size):
            num_obj = int(batch['num_objects'][b].item())
            if num_obj > 0:
                # Check only actual boxes, not padding.
                actual_boxes = batch['boxes'][b, :num_obj]
                
                # Checks for NaN/Inf.
                if torch.isnan(actual_boxes).any() or torch.isinf(actual_boxes).any():
                    if not auto_fix:
                        return batch, False, f"Sample {b} has NaN/Inf in boxes"
                    batch['boxes'][b, :num_obj] = sanitize_boxes(actual_boxes)
                    needs_fix = True
                    continue
                
                # Check for invalid dimensions (silent fix for common COCO padding issues)
                if (actual_boxes[:, 2] <= 0).any() or (actual_boxes[:, 3] <= 0).any():
                    if auto_fix:
                        # Silent fix for expected COCO data quirks.
                        actual_boxes[:, 2] = torch.clamp(actual_boxes[:, 2], min=1e-4)
                        actual_boxes[:, 3] = torch.clamp(actual_boxes[:, 3], min=1e-4)
                        batch['boxes'][b, :num_obj] = actual_boxes
                        needs_fix = True
                    else:
                        return batch, False, f"Sample {b} has non-positive box dimensions"
        
        # Return success (with or without silent fixes)
        return batch, True, "Batch valid" if not needs_fix else "Batch sanitized (silent)"
    
    # Fallback to full validation if num_objects not available.
    is_valid, msg = validate_batch(batch, num_classes)
    
    if is_valid:
        return batch, True, "Batch is valid"
    
    if not auto_fix:
        return batch, False, msg
    
    # Apply fix when possible.
    logger.warning(f"Batch validation failed: {msg}. Attempting auto-fix...")
    batch = batch.copy()
    
    # Sanitize boxes.
    if 'boxes' in batch:
        batch['boxes'] = sanitize_boxes(batch['boxes'])
    
    # Validate again.
    is_valid, msg = validate_batch(batch, num_classes)
    
    if is_valid:
        logger.info("Batch successfully sanitized")
        return batch, True, "Batch sanitized"
    else:
        logger.error(f"Failed to sanitize batch: {msg}")
        return batch, False, f"Auto-fix failed: {msg}"






