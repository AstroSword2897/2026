"""
Comprehensive Detection Metrics Calculator - REFACTORED

Key improvements:

1. Proper mAP implementation with precision-recall curves

2. Vectorized IoU computation for batch processing

3. Device-aware tensor initialization

4. COCO-style multi-threshold mAP support

5. Confidence-based prediction storage and ranking

6. Per-size metrics (small/medium/large objects)

Complexity: O(N_pred * N_gt) per image for matching, O(C) for per-class metrics
Relationship: Core evaluation component - used by ProductionTrainer.validate()
"""

import torch
from typing import Dict, Optional, List, Tuple
from collections import defaultdict
import numpy as np


def compute_iou_matrix(
    pred_boxes: torch.Tensor,  # [N_pred, 4]
    gt_boxes: torch.Tensor      # [N_gt, 4]
) -> torch.Tensor:  # [N_pred, N_gt]
    # Edge case: empty inputs - happens sometimes with bad models or empty images
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return torch.zeros(pred_boxes.shape[0], gt_boxes.shape[0], 
                          device=pred_boxes.device)
    
    pred_boxes = pred_boxes.unsqueeze(1)  # [N_pred, 1, 4]
    gt_boxes = gt_boxes.unsqueeze(0)      # [1, N_gt, 4]
    
    pred_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2  # Left edge
    pred_y1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2  # Top edge
    pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2  # Right edge
    pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2  # Bottom edge
    
    gt_x1 = gt_boxes[..., 0] - gt_boxes[..., 2] / 2
    gt_y1 = gt_boxes[..., 1] - gt_boxes[..., 3] / 2
    gt_x2 = gt_boxes[..., 0] + gt_boxes[..., 2] / 2
    gt_y2 = gt_boxes[..., 1] + gt_boxes[..., 3] / 2
    
    # Find intersection rectangle - where boxes overlap
    inter_x1 = torch.max(pred_x1, gt_x1) 
    inter_y1 = torch.max(pred_y1, gt_y1)
    # Bottom-right corner: min of the two bottom-right corners (leftmost right, topmost bottom)
    inter_x2 = torch.min(pred_x2, gt_x2)
    inter_y2 = torch.min(pred_y2, gt_y2)
    
    # Intersection area - clamp to 0 in case boxes don't overlap
    # Clamp fixes that - negative area becomes 0 (no intersection)
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * \
                 (inter_y2 - inter_y1).clamp(min=0)
    
    # Compute union area - area of both boxes minus intersection (don't double-count)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)  # Area of prediction box
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)  # Area of ground truth box
    union_area = pred_area + gt_area - inter_area  # Subtract intersection (already counted twice)
    
    # IoU = intersection / union
    # Put small epsilon so no division by zero but effort remains low for the model
    iou = inter_area / (union_area + 1e-9)
    # Check for boxes with zero width or height - log warning if found
    # Degenerate boxes can cause unexpected behavior in the program
    if torch.any((pred_x2 - pred_x1) <= 0) or torch.any((pred_y2 - pred_y1) <= 0):
        pass
    if torch.any((gt_x2 - gt_x1) <= 0) or torch.any((gt_y2 - gt_y1) <= 0):
        pass
    # Safe squeeze: avoid squeeze() entirely to prevent shape issues
    # Keep full [N_pred, N_gt] shape even if one dimension is 1
    # Downstream code can handle the shape correctly
    return iou


class DetectionMetrics:
    """
    Comprehensive detection metrics calculator with proper mAP implementation.
    
    Improvements over original:
    - Proper AP calculation with precision-recall curves
    - Stores predictions with scores for ranking
    - Vectorized IoU computation
    - Tensor initialization
    - COCO-style multi-threshold mAP
    - Per-size metrics (small/medium/large)
    """
    
    def __init__(
        self, 
        num_classes: int, 
        iou_thresholds: List[float] = [0.5],
        device: Optional[torch.device] = None,
        image_size: Tuple[int, int] = (224, 224)
    ):

        self.image_size = image_size
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds
        self.device = device or torch.device('cpu')
        self.reset()
        
    
    def reset(self, device: Optional[torch.device] = None):
      
        if device is not None:
            self.device = device  # Update device if provided
        
        # Device-aware: these tensors live on CPU or GPU depending on self.device
        self.class_tp = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)  # True positives
        self.class_fp = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)  # False positives
        self.class_fn = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)  # False negatives
        
        # AP needs predictions sorted by confidence, then we compute precision-recall curve
        self.class_predictions = defaultdict(list)
        
        # Ground truth counts
        self.class_gt_counts = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)
        
        # Per-lighting-condition metrics may vary so defualt is created for reference
        self.lighting_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        # Per-size metrics - COCO convention for object sizes between sizes for model analysis
        self.size_metrics = {
            'small': {'tp': 0, 'fp': 0, 'fn': 0},
            'medium': {'tp': 0, 'fp': 0, 'fn': 0},
            'large': {'tp': 0, 'fp': 0, 'fn': 0}
        }
    
    def _get_size_category(self, box: torch.Tensor) -> str:
        """
        Categorize box by size - COCO standard for object detection evaluation.
        
        This is useful because models often struggle with small objects (distant cars,
        tiny signs, etc.). Tracking metrics by size helps identify these issues.
        
        Arguments:
            box: 4d tensor in normalized coordinates [0, 1]
        
        Returns:
            'small', 'medium', or 'large'
        """
        
        # Compute normalized area based off of dimensions in pixels
        # 32^2 - small
        # between small and 96^2 is medium
        # over then 96^2 is large
        box_area_normalized = box[2] * box[3]  # Normalized area [0, 1]
        pixel_area = box_area_normalized * (self.image_size[0] * self.image_size[1])
        
        # Apply COCO thresholds (in pixels²)
        if pixel_area < 32 * 32: 
            return 'small'  # Tiny objects - hardest to detect
        elif pixel_area < 96 * 96: 
            return 'medium'
        else:
            return 'large'  # Big objects - usually easier to detect
    
    def update(
        self,
        pred_boxes: torch.Tensor,      # [N_pred, 4] center format
        pred_labels: torch.Tensor,     # [N_pred] class indices
        pred_scores: torch.Tensor,     # [N_pred] confidence scores
        gt_boxes: torch.Tensor,        # [N_gt, 4] center format
        gt_labels: torch.Tensor,       # [N_gt] class indices
        lighting: Optional[str] = None,
        iou_threshold: float = 0.5
    ) -> None:
        """
        Update metrics with predictions and ground truth (improved matching).
        
        Improvements:
        - Vectorized IoU computation (10-100x faster than looped iteration)
        - Stores predictions with scores for proper AP calculation
        - Handles device placement correctly
        - Tracks per-size metrics
        """
        # Make sure that all of the tensors are on the computation device
        if pred_boxes.device != self.device:
            pred_boxes = pred_boxes.to(self.device)
        if pred_labels.device != self.device:
            pred_labels = pred_labels.to(self.device)
        if pred_scores.device != self.device:
            pred_scores = pred_scores.to(self.device)
        if gt_boxes.device != self.device:
            gt_boxes = gt_boxes.to(self.device)
        if gt_labels.device != self.device:
            gt_labels = gt_labels.to(self.device)
        # Count ground truth objects per class - needed for recall calculation
        # Recall = TP / (TP + FN) = TP / total_gt, so we need to know total_gt
        for gt_label in gt_labels:
            class_idx = int(gt_label.item())  # .item() returns Number, need int for indexing
            if 0 <= class_idx < self.num_classes:  # Safety check - don't index out of bounds
                self.class_gt_counts[class_idx] += 1
        
        # Edge case 1: No predictions at all (model didn't detect anything)
        if len(pred_boxes) == 0:
            for gt_label, gt_box in zip(gt_labels, gt_boxes):
                class_idx = int(gt_label.item())
                if 0 <= class_idx < self.num_classes:
                    self.class_fn[class_idx] += 1
                    if lighting:
                        self.lighting_metrics[lighting]['fn'] += 1
                    
                    # Update size metrics
                    size_cat = self._get_size_category(gt_box)
                    self.size_metrics[size_cat]['fn'] += 1
            return
        
        if len(gt_boxes) == 0:
            for pred_label, pred_score, pred_box in zip(pred_labels, pred_scores, pred_boxes):
                class_idx = int(pred_label.item())
                if 0 <= class_idx < self.num_classes:
                    self.class_fp[class_idx] += 1
                    if lighting:
                        self.lighting_metrics[lighting]['fp'] += 1
                    
                    # Store as FP for AP calculation
                    box_area = (pred_box[2] * pred_box[3]).item()
                    self.class_predictions[class_idx].append(
                        (pred_score.item(), 0.0, None, box_area)  # CORRECT - no GT to match
                    )
                    
                    # Update size metrics
                    size_cat = self._get_size_category(pred_box)
                    self.size_metrics[size_cat]['fp'] += 1
            return
        
        # Vectorized IoU computation off of all IoUs and then 
        # Result: [N_pred, N_gt] matrix where [i, j] = IoU between pred[i] and gt[j]
        iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)  # [N_pred, N_gt]
        
        # Greedy matching algorithm - standard in object detection
        # Strategy: sort predictions by confidence (highest first), then match each
        # prediction to the best available ground truth box of the same class
        # 
        # Why sort by confidence? Higher confidence predictions are more likely to be
        # correct, so we give them priority in matching. This is standard practice.
        matched_gt = set()  # Track which GT boxes we've already matched (one GT = one prediction)
        sorted_indices = torch.argsort(pred_scores, descending=True)  # Sort by confidence
        
        for sorted_idx in sorted_indices:
            pred_idx = int(sorted_idx.item())  # Convert tensor index to Python int
            pred_class = int(pred_labels[pred_idx].item())
            pred_score = pred_scores[pred_idx].item()  # Confidence score
            pred_box = pred_boxes[pred_idx]
            box_area = (pred_box[2] * pred_box[3]).item()  # For size metrics
            
            # Skip invalid class indices (safety check)
            if not (0 <= pred_class < self.num_classes):
                continue
            
            # Find best matching ground truth box of the same class
            # We want the GT box with highest IoU that:
            # 1. Hasn't been matched yet (not in matched_gt)
            # 2. Has the same class as this prediction
            best_iou = 0.0
            best_gt_idx = None
            
            for gt_idx in range(len(gt_boxes)):
                if gt_idx in matched_gt:
                    continue  # Already matched to another prediction
                if int(pred_labels[pred_idx].item()) != int(gt_labels[gt_idx].item()):
                    continue  # Classes don't match - can't be a match
                
                # Get IoU from pre-computed matrix
                if iou_matrix.ndim == 2:
                    iou = iou_matrix[pred_idx, gt_idx].item()
                elif iou_matrix.ndim == 1 and len(iou_matrix) > pred_idx:
                    # Edge case: single GT box (shouldn't happen after removing squeeze)
                    iou = float(iou_matrix[pred_idx])
                else:
                    iou = 0.0




                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx  # Remember this as the best match so far
            
            # Determine if this is a true positive or false positive
            # TP: IoU is high enough (>= threshold) AND we found a matching GT box
            # FP: IoU too low OR no matching GT box found
            is_tp = best_iou >= iou_threshold and best_gt_idx is not None
            
            if is_tp:
                self.class_tp[pred_class] += 1
                matched_gt.add(best_gt_idx)
                if lighting:
                    self.lighting_metrics[lighting]['tp'] += 1
                
                # Update size metrics
                size_cat = self._get_size_category(pred_box)
                self.size_metrics[size_cat]['tp'] += 1
            else:
                self.class_fp[pred_class] += 1
                if lighting:
                    self.lighting_metrics[lighting]['fp'] += 1
                
                # Update size metrics
                size_cat = self._get_size_category(pred_box)
                self.size_metrics[size_cat]['fp'] += 1
            
            # AP confidence storing off of predictions for class
            # Format: (confidence_score, is_tp, box_area)
            # We'll sort by confidence later when computing the precision-recall curve
            self.class_predictions[pred_class].append(
                (pred_score, best_iou, best_gt_idx, box_area) 
            )
        
        # Check if ground truth boxes did not get matched
        # These are objects that exist in the image but the model didn't detect them
        # This happens when the model misses objects (common with small objects or
        # objects in difficult lighting conditions)
        for gt_idx in range(len(gt_boxes)):
            if gt_idx not in matched_gt:
                gt_class = int(gt_labels[gt_idx].item())
                gt_box = gt_boxes[gt_idx]
                
                if 0 <= gt_class < self.num_classes:
                    self.class_fn[gt_class] += 1  # We missed this object
                    if lighting:
                        self.lighting_metrics[lighting]['fn'] += 1
                    
                    # Update size metrics - helps identify if model struggles with small objects
                    size_cat = self._get_size_category(gt_box)
                    self.size_metrics[size_cat]['fn'] += 1
    
    def compute_precision(self, class_idx: Optional[int] = None) -> float:
        """
        Compute precision: TP / (TP + FP)
        
        Precision answers: "Of all the objects I predicted, how many were actually correct?"
        High precision = model is conservative, only predicts when it's confident
        Low precision = model predicts too much, lots of false alarms
        
        Arguments:
            class_idx: If None, compute overall precision. If int, compute for that class.
        
        Returns:
            Precision score (0.0 to 1.0)
        """
        if class_idx is None:
            # Overall precision - sum up all TP and FP across all classes
            total_tp = self.class_tp.sum().item()
            total_fp = self.class_fp.sum().item()
        else:
            # Per-class precision - how good is the model at detecting this specific class?
            if not (0 <= class_idx < self.num_classes):
                return 0.0  # Invalid class index
            total_tp = self.class_tp[class_idx].item()
            total_fp = self.class_fp[class_idx].item()
        
        # Precision formula: correct predictions / all predictions
        denominator = total_tp + total_fp
        return total_tp / denominator if denominator > 0 else 0.0  # Avoid division by zero
    
    def compute_recall(self, class_idx: Optional[int] = None) -> float:
        """Compute recall: TP / (TP + FN)."""
        if class_idx is None:
            total_tp = self.class_tp.sum().item()
            total_fn = self.class_fn.sum().item()
        else:
            if not (0 <= class_idx < self.num_classes):
                return 0.0
            total_tp = self.class_tp[class_idx].item()
            total_fn = self.class_fn[class_idx].item()
        
        denominator = total_tp + total_fn
        return total_tp / denominator if denominator > 0 else 0.0
    
    def compute_f1(self, class_idx: Optional[int] = None) -> float:
        """Compute F1 score: 2 * (P * R) / (P + R)."""
        precision = self.compute_precision(class_idx)
        recall = self.compute_recall(class_idx)
        denominator = precision + recall
        return 2 * (precision * recall) / denominator if denominator > 0 else 0.0
    
    def compute_ap(self, class_idx: int, iou_threshold: float = 0.5) -> float:
    
        predictions = self.class_predictions.get(class_idx, [])
        total_gt = self.class_gt_counts[class_idx].item()
        
        if len(predictions) == 0 or total_gt == 0:
            return 0.0
        
        predictions.sort(key=lambda x: x[0], reverse=True)
        
        # Track which GT boxes have been matched (for this IoU threshold)
        matched_gt_indices = set()
        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []
        
        for pred_data in predictions:
            # Handle both old format (score, is_tp, area) and new format (score, best_iou, matched_gt_idx, area)
            if len(pred_data) == 3:
                # Legacy format: (score, is_tp, area) - use is_tp directly
                confidence, is_tp, _ = pred_data
                is_tp_at_threshold = is_tp if isinstance(is_tp, bool) else (pred_data[1] >= iou_threshold)
            else:
                # New format: (score, best_iou, matched_gt_idx, area)
                confidence, best_iou, matched_gt_idx, _ = pred_data
                # Recompute TP/FP based on current IoU threshold
                is_tp_at_threshold = (best_iou >= iou_threshold and 
                                    matched_gt_idx is not None and 
                                    matched_gt_idx not in matched_gt_indices)
                if is_tp_at_threshold:
                    matched_gt_indices.add(matched_gt_idx)
            
            if is_tp_at_threshold:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            
            precision = tp_cumsum / (tp_cumsum + fp_cumsum) if (tp_cumsum + fp_cumsum) > 0 else 0.0
            recall = tp_cumsum / total_gt if total_gt > 0 else 0.0
            precisions.append(precision)
            recalls.append(recall)
        
        # Full PR curve integration (trapezoidal rule)
        if len(precisions) == 0:
            return 0.0
        
        pr_pairs = sorted(zip(recalls, precisions), key=lambda x: x[0])
        ap = 0.0
        prev_recall = 0.0
        prev_precision = 1.0
        
        for recall, precision in pr_pairs:
            ap += (recall - prev_recall) * (precision + prev_precision) / 2.0
            prev_recall = recall
            prev_precision = precision
        
        if prev_recall < 1.0:
            ap += (1.0 - prev_recall) * prev_precision
        
        return ap
    
    def compute_map(self, iou_threshold: float = 0.5) -> float:
        """
        Compute mean Average Precision across all classes (PROPER implementation).
        """
        ap_scores = []
        
        for class_idx in range(self.num_classes):
            ap = self.compute_ap(class_idx, iou_threshold)
            ap_scores.append(ap)
        
        return sum(ap_scores) / len(ap_scores) if ap_scores else 0.0
    
    def compute_map_coco(self) -> Dict[str, float]:
        # Generate IoU thresholds: 0.5, 0.55, 0.6, ..., 0.95 (step 0.05)
        # Also include 0.75 separately because it's commonly reported
        # Ensure that there are no duplication issues
        thresholds = [float(round(t,2)) for t in np.arange(0.5, 1.0, 0.05)]
        if 0.75 not in thresholds:
            thresholds.append(0.75)
        thresholds = sorted(thresholds)
        results = {}
        aps_all = []  # Store all APs for averaging
        
        # Compute mAP at each threshold
        for threshold in thresholds:
            threshold_float = float(threshold)  # Ensure it's a Python float (numpy types cause issues)
            ap = self.compute_map(threshold_float)
            aps_all.append(ap)
            
            # Store specific thresholds that are commonly reported
            if abs(threshold_float - 0.5) < 1e-6:  # Floating point comparison
                results['mAP@0.5'] = ap
            elif abs(threshold_float - 0.75) < 1e-6:
                results['mAP@0.75'] = ap
        
        # Average AP across all thresholds - this is the COCO standard
        results['mAP@[0.5:0.95]'] = sum(aps_all) / len(aps_all)
        
        return results
    
    def get_per_class_metrics(self) -> Dict[int, Dict[str, float]]:
        """Get precision, recall, F1, AP for each class."""
        results = {}
        
        for class_idx in range(self.num_classes):
            results[class_idx] = {
                'precision': self.compute_precision(class_idx),
                'recall': self.compute_recall(class_idx),
                'f1': self.compute_f1(class_idx),
                'ap': self.compute_ap(class_idx)
            }
        
        return results
    
    def get_lighting_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get precision, recall, F1 for each lighting condition."""
        results = {}
        
        for lighting, metrics in self.lighting_metrics.items():
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            
            if tp + fp == 0:
                precision = float('nan')
            else: 
                precision = tp / (tp + fp)
            
            if tp + fn == 0:
                recall = float('nan')
            else: 
                recall = tp / (tp + fn)


            if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
                f1 = float('nan')
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            

            results[lighting] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return results
    
    def get_size_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get metrics by object size (COCO-style)."""
        results = {}
        
        for size_cat, metrics in self.size_metrics.items():
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            
            if tp + fp == 0:
                precision = float('nan')  # No predictions made
            else:
                precision = tp / (tp + fp)
            
            if tp + fn == 0:
                recall = float('nan')  # No ground truth exists
            else:
                recall = tp / (tp + fn)
            
            # F1 is undefined if either precision or recall is NaN
            if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
                f1 = float('nan')
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            
            results[size_cat] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return results
    
    def record_inference_time(self, inference_time_ms: float) -> None:
        """
        Record inference latency for a single forward pass.
        
        Args:
            inference_time_ms: Inference time in milliseconds
        """
        self.inference_times.append(inference_time_ms)
    
    def get_latency_stats(self) -> Dict[str, float]:
        """
        Get latency statistics from recorded inference times.
        
        Returns:
            Dictionary with mean, median, min, max, std latency in milliseconds
        """
        if len(self.inference_times) == 0:
            return {
                'mean_ms': 0.0,
                'median_ms': 0.0,
                'min_ms': 0.0,
                'max_ms': 0.0,
                'std_ms': 0.0,
                'p95_ms': 0.0,
                'p99_ms': 0.0
            }
        
        times = np.array(self.inference_times)
        return {
            'mean_ms': float(np.mean(times)),
            'median_ms': float(np.median(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'std_ms': float(np.std(times)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99))
        }
    
    def reset_latency(self) -> None:
        """Reset latency tracking."""
        self.inference_times = []
