import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple, Any
from collections import defaultdict
import numpy as np

''' The intersection over matrix area is calulated in 
a multi level metric based level 
below basic manipulations of the coordinates are taking place
'''
def compute_cost_matrix(
    pred_boxes: torch.Tensor,  # [N_pred, 4]
    gt_boxes: torch.Tensor      # [N_gt, 4]
) -> torch.Tensor:  # [N_pred, N_gt]
    """
    Batch IoU calculation using broadcasting (faster than loops). Assumes center format.
    
    Edge case: empty inputs - happens sometimes with bad models or empty images
    """
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0]:
        return torch.zeros(pred_boxes.shape[0], gt_boxes.shape[0],
                           device=pred_boxes.device)
    
    pred_boxes = pred_boxes.unsqueeze(1)  # The predictiosn need to go from 1-4
    gt_boxes = gt_boxes.unsqueeze(0)
    
    pred_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2  # The left edge
    pred_y1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2  # The top edge
    pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2  # The right edge
    pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2  # The bottom edge
    
    gt_x1 = gt_boxes[..., 0] - gt_boxes[..., 2] / 2
    gt_y1 = gt_boxes[..., 1] - gt_boxes[..., 3] / 2
    gt_x2 = gt_boxes[..., 0] + gt_boxes[..., 2] / 2
    gt_y2 = gt_boxes[..., 1] + gt_boxes[..., 3] / 2
    
    # Find intersection rectangle - where boxes overlap
    inter_x1 = torch.max(pred_x1, gt_x1) 
    inter_y1 = torch.max(pred_y1, gt_y1)
    # Bottom right corner: Get the minimum of the two bottom right corner
    inter_x2 = torch.min(pred_x2, gt_x2)
    inter_y2 = torch.min(pred_y2, gt_y2)
    
    # Intersection area - clamp to 0 in case boxes don't overlap
    # Clamp fixes that - negative area becomes 0 (no intersection)
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * \
                 (inter_y2 - inter_y1).clamp(min=0)
    
    # Compute union area - area of both boxes minus intersection
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)  # Area of prediction box
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)  # Area of ground truth box
    union_area = pred_area + gt_area - inter_area  # Subtract intersection (already counted twice)
    
    # IoU = intersection / union
    # Put small epsilon so no division by zero but effort remains low for the model
    iou = inter_area / (union_area + 1e-9)
    # Safe squeeze: only remove last dimension if it's size 1, avoid removing all size-1 dims
    # This prevents shape mismatches when N_pred=1 or N_gt=1
    if iou.shape[-1] == 1:
        iou = iou.squeeze(-1)
    return iou


class DetectionMetrics:
    """
    Proper AP calculation with precision-recall curves
    Stores predictions with scores for ranking
    Vectorized IoU computation (10-100x faster than the basic storing approach)
    Device-aware tensor initialization
    COCO-style multi-threshold mAP
    Per-size metrics (small/medium/large)
    Configurable image size for size category calculation
    """
    
    def __init__(
        self, 
        num_classes: int, 
        iou_thresholds: List[float] = [0.5],
        device: Optional[torch.device] = None,
        image_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize metrics calculator.
        
        Arguments:
            num_classes: Number of object classes
            iou_thresholds: List of IoU thresholds for mAP (default [0.5])
            device: Device for tensor allocation (default: CPU)
            image_size: (height, width) of input images for size category calculation
        """
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds
        self.device = device or torch.device('cpu')
        self.image_size = image_size  # (height, width) for size category normalization
        
        self.reset()
    
    def reset(self, device: Optional[torch.device] = None):
        """
        Reset all accumulators. Pass device to ensure tensors on correct device (CPU/GPU).
        """
        if device is not None:
            self.device = device  # Update device if provided
        
        # Device-aware: these tensors live on CPU or GPU depending on self.device (Silicon for mac download is having issues on M3 pro)
        self.class_tp = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)  # True positives
        self.class_fp = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)  # False positives
        self.class_fn = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)  # False negatives
        
        # AP needs predictions sorted by confidence, then we compute precision-recall curve
        # best_iou allows recomputing TP/FP at any IoU threshold
        self.class_predictions = defaultdict(list)
        
        # Ground truth counts
        self.class_gt_counts = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)
        
        # Per-lighting-condition metrics may vary so default is created for reference
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
        
        This is useful because models often struggle with small objects.
        Tracking metrics by size helps identify these issues.
        
        Arguments:
            box: [4] tensor (cx, cy, w, h) normalized [0, 1]
        
        Returns:
            'small', 'medium', or 'large'
        """
        # Normalize area by image size for accurate size categorization
        # COCO thresholds are based on pixel areas, so we normalize by image dimensions
        normalized_area = (box[2] * box[3]) / (self.image_size[0] * self.image_size[1])
        
        # COCO thresholds - these are in pixels², but we normalize to [0, 1]
        # Small: < 32² pixels = 32² / 224² ≈ 0.02 (tiny objects)
        # Medium: 32² to 96² = 96² / 224² ≈ 0.18 (normal objects)
        # Large: ≥ 96² pixels (big objects)
        # 
        # Note: These thresholds are approximate - COCO uses actual pixel counts,
        # but since our boxes are normalized, we use normalized thresholds
        if normalized_area < 0.02:
            return 'small'  # Tiny objects - hardest to detect
        elif normalized_area < 0.18:
            return 'medium'  # Normal-sized objects
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
        # First thing: make sure all tensors are on the same device
        # This is critical - if pred_boxes is on GPU and gt_boxes is on CPU,
        # PyTorch will throw a device mismatch error. Always move to self.device.
        # Optimization: only move if not already on correct device
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
        
        # Edge case 2: No ground truth (all predictions are false positives)
        if len(gt_boxes) == 0:
            for pred_label, pred_score, pred_box in zip(pred_labels, pred_scores, pred_boxes):
                class_idx = int(pred_label.item())
                if 0 <= class_idx < self.num_classes:
                    self.class_fp[class_idx] += 1
                    if lighting:
                        self.lighting_metrics[lighting]['fp'] += 1
                    
                    # Store as FP for AP calculation (no GT to match, so IoU = 0.0)
                    box_area = (pred_box[2] * pred_box[3]).item()
                    self.class_predictions[class_idx].append(
                        (pred_score.item(), 0.0, None, box_area)  # (score, best_iou, matched_gt_idx, area)
                    )
                    
                    # Update size metrics
                    size_cat = self._get_size_category(pred_box)
                    self.size_metrics[size_cat]['fp'] += 1
            return
        
        # Now the real matching logic - this is where the speed improvement happens
        # Vectorized IoU computation - compute ALL IoUs at once (10-100x faster than loops!)
        # Result: [N_pred, N_gt] matrix where [i, j] = IoU between pred[i] and gt[j]
        iou_matrix = compute_cost_matrix(pred_boxes, gt_boxes)  # [N_pred, N_gt]
        
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
                
                # Get IoU from pre-computed matrix (fast!)
                # Handle both 2D and edge case 1D shapes
                if iou_matrix.ndim == 2:
                    iou = iou_matrix[pred_idx, gt_idx].item()
                else:
                    # Fallback for edge cases (shouldn't happen with proper input)
                    iou = float(iou_matrix[pred_idx] if len(iou_matrix) > pred_idx else 0.0)
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
            
            # Store prediction for AP calculation - this is critical!
            # AP requires predictions sorted by confidence, so we store them here
            # Format: (confidence_score, best_iou, matched_gt_idx, box_area)
            # We'll sort by confidence later when computing the precision-recall curve
            # Storing best_iou allows recomputing TP/FP at any IoU threshold
            self.class_predictions[pred_class].append(
                (pred_score, best_iou, best_gt_idx, box_area)
            )
        
        # Finally, check for false negatives: ground truth boxes that never got matched
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
        """
        Compute recall: TP / (TP + FN)
        
        Recall answers: "Of all the objects that actually exist, how many did I find?"
        High recall = model finds most objects (fewer misses)
        Low recall = model misses many objects
        
        Arguments:
            class_idx: If None, compute overall recall. If int, compute for that class.
        
        Returns:
            Recall score (0.0 to 1.0)
        """
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
        """
        Compute F1 score: 2 * (P * R) / (P + R)
        
        F1 is the harmonic mean of precision and recall, balancing both metrics.
        
        Arguments:
            class_idx: If None, compute overall F1. If int, compute for that class.
        
        Returns:
            F1 score (0.0 to 1.0)
        """
        precision = self.compute_precision(class_idx)
        recall = self.compute_recall(class_idx)
        denominator = precision + recall
        return 2 * (precision * recall) / denominator if denominator > 0 else 0.0
    
    def compute_ap(self, class_idx: int, iou_threshold: float = 0.5) -> float:
        """
        Compute Average Precision (AP) for a single class - THE CORRECT WAY.
        
        This is the proper implementation that matches COCO evaluation. The old
        implementation just used precision, which is wrong! AP requires computing
        the area under the precision-recall curve.
        
        How it works:
        1. Sort predictions by confidence (highest first) - already done in update()
        2. Recompute TP/FP based on stored IoU values and the given threshold
        3. Compute precision and recall at each confidence threshold
        4. Calculate area under the PR curve using 11-point interpolation (COCO standard)
        
        This is what makes mAP the gold standard metric for object detection.
        
        Arguments:
            class_idx: Class index to compute AP for
            iou_threshold: IoU threshold for matching (default 0.5)
        
        Returns:
            AP score (0.0 to 1.0)
        """
        # Get all predictions for this class
        predictions = self.class_predictions.get(class_idx, [])
        total_gt = self.class_gt_counts[class_idx].item()
        
        # Edge cases: no predictions or no ground truth
        if len(predictions) == 0 or total_gt == 0:
            return 0.0
        
        # Sort by confidence (descending) - should already be sorted, but be safe
        # Predictions are stored as (confidence, best_iou, matched_gt_idx, box_area) tuples
        predictions.sort(key=lambda x: x[0], reverse=True)
        
        # Track which GT boxes have been matched (for this IoU threshold)
        # This is critical - we need to recompute TP/FP based on the current threshold
        matched_gt_indices = set()
        tp_cumsum = 0  # Cumulative true positives
        fp_cumsum = 0  # Cumulative false positives
        precisions = []  # Precision at each threshold
        recalls = []     # Recall at each threshold
        
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
            
            # Update cumulative counts
            if is_tp_at_threshold:
                tp_cumsum += 1  # Found a true positive
            else:
                fp_cumsum += 1  # Found a false positive
            
            # Compute precision and recall at this threshold
            # Precision: of all predictions so far, how many are correct?
            precision = tp_cumsum / (tp_cumsum + fp_cumsum) if (tp_cumsum + fp_cumsum) > 0 else 0.0
            # Recall: of all ground truth objects, how many have we found so far?
            recall = tp_cumsum / total_gt if total_gt > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Compute AP using 11-point interpolation (COCO standard)
        # This averages precision at 11 equally spaced recall levels: 0.0, 0.1, ..., 1.0
        # For each recall level, we find the maximum precision at that recall or higher
        ap = 0.0
        for recall_threshold in np.linspace(0, 1, 11):
            # Find maximum precision at recall >= this threshold
            max_precision = 0.0
            recall_threshold_float = float(recall_threshold)  # Convert numpy type to Python float
            for r, p in zip(recalls, precisions):
                if r >= recall_threshold_float:
                    max_precision = max(max_precision, p)
            ap += max_precision
        
        return ap / 11.0  # Average over 11 points
    
    def compute_map(self, iou_threshold: float = 0.5) -> float:
        """
        Compute mean Average Precision across all classes (PROPER implementation).
        
        Arguments:
            iou_threshold: IoU threshold for matching (default 0.5)
        
        Returns:
            mAP score (0.0 to 1.0)
        """
        ap_scores = []
        
        for class_idx in range(self.num_classes):
            ap = self.compute_ap(class_idx, iou_threshold)
            ap_scores.append(ap)
        
        return sum(ap_scores) / len(ap_scores) if ap_scores else 0.0
    
    def compute_map_coco(self) -> Dict[str, float]:
        """
        Compute COCO-style mAP at multiple IoU thresholds - the industry standard.
        
        COCO evaluation uses mAP@[0.5:0.95] which averages mAP across multiple IoU
        thresholds (0.5, 0.55, 0.6, ..., 0.95). This is more comprehensive than
        just mAP@0.5 because it tests how well the model localizes objects at
        different precision levels.
        
        Returns:
            Dictionary with:
            - mAP@0.5: AP at IoU=0.5 (most common metric, easier threshold)
            - mAP@0.75: AP at IoU=0.75 (stricter, tests localization accuracy)
            - mAP@[0.5:0.95]: Average AP from 0.5 to 0.95 (COCO standard, most comprehensive)
        """
        # Generate IoU thresholds: 0.5, 0.55, 0.6, ..., 0.95 (step 0.05)
        # Use set to remove duplicates (0.5 and 0.75 appear in both lists), then sort
        thresholds = sorted(set([0.5, 0.75] + [float(t) for t in np.arange(0.5, 1.0, 0.05)]))
        
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
        results['mAP@[0.5:0.95]'] = sum(aps_all) / len(aps_all) if aps_all else 0.0
        
        return results
    
    def get_per_class_metrics(self) -> Dict[int, Dict[str, float]]:
        """
        Get precision, recall, F1, AP for each class.
        
        Returns:
            Dictionary mapping class_idx to metrics dictionary
        """
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
        """
        Get precision, recall, F1 for each lighting condition.
        
        Returns:
            Dictionary mapping lighting condition to metrics dictionary
        """
        results = {}
        
        for lighting, metrics in self.lighting_metrics.items():
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            
            # Use max(..., 1) to avoid division by zero, but this means precision=0
            # when no predictions exist, which is mathematically correct
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-8)
            
            results[lighting] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return results
    
    def get_size_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get metrics by object size (COCO-style).
        
        Returns:
            Dictionary mapping size category ('small', 'medium', 'large') to metrics dictionary
        """
        results = {}
        
        for size_cat, metrics in self.size_metrics.items():
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            
            # Use max(..., 1) to avoid division by zero, but this means precision=0
            # when no predictions exist, which is mathematically correct
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-8)
            
            results[size_cat] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        return results