import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class DetectionPrediction:
    score: float
    iou: float
    matched_gt_idx: Optional[int]
    area: float


def compute_iou_matrix(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """Compute IoU matrix [P, G] for predicted vs ground-truth boxes (cx, cy, w, h)."""
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return torch.zeros(pred_boxes.shape[0], gt_boxes.shape[0], device=pred_boxes.device)

    pred_boxes = pred_boxes.unsqueeze(1)
    gt_boxes = gt_boxes.unsqueeze(0)

    pred_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
    pred_y1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
    pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
    pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2

    gt_x1 = gt_boxes[..., 0] - gt_boxes[..., 2] / 2
    gt_y1 = gt_boxes[..., 1] - gt_boxes[..., 3] / 2
    gt_x2 = gt_boxes[..., 0] + gt_boxes[..., 2] / 2
    gt_y2 = gt_boxes[..., 1] + gt_boxes[..., 3] / 2

    inter_x1 = torch.max(pred_x1, gt_x1)
    inter_y1 = torch.max(pred_y1, gt_y1)
    inter_x2 = torch.min(pred_x2, gt_x2)
    inter_y2 = torch.min(pred_y2, gt_y2)

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    union_area = pred_area + gt_area - inter_area

    return inter_area / (union_area + 1e-8)


class DetectionMetrics:
    """Accumulates per-class and per-condition detection TP/FP/FN for mAP and precision/recall."""

    def __init__(self, num_classes: int, iou_thresholds: Optional[List[float]] = None,
                 device: Optional[torch.device] = None, image_size: Tuple[int, int] = (640, 640),
                 store_predictions: bool = True):
        """Initialize counters and optional prediction storage."""
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [0.5]
        self.device = device or torch.device('cpu')
        self.image_size = image_size
        self.store_predictions = store_predictions
        self.reset()

    def reset(self, device: Optional[torch.device] = None) -> None:
        """Reset all TP/FP/FN counters and optional prediction lists."""
        if device:
            self.device = device
        self.class_tp = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)
        self.class_fp = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)
        self.class_fn = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)
        self.class_gt_counts = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)

        self.class_predictions: Dict[int, List[DetectionPrediction]] = defaultdict(list)
        self.condition_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        self.size_metrics = {'small': {'tp': 0, 'fp': 0, 'fn': 0},
                             'medium': {'tp': 0, 'fp': 0, 'fn': 0},
                             'large': {'tp': 0, 'fp': 0, 'fn': 0}}
        self.inference_times: List[float] = []

    def _get_size_category(self, box: torch.Tensor) -> str:
        """Return 'small', 'medium', or 'large' from normalized box area."""
        normalized_area = box[2] * box[3]
        pixel_area = normalized_area * (self.image_size[0] * self.image_size[1])
        if pixel_area < 32 * 32:
            return 'small'
        elif pixel_area < 96 * 96:
            return 'medium'
        else:
            return 'large'

    def update(self, pred_boxes: torch.Tensor, pred_labels: torch.Tensor, pred_scores: torch.Tensor,
               gt_boxes: torch.Tensor, gt_labels: torch.Tensor, condition: Optional[str] = None,
               iou_threshold: float = 0.5, lighting: Optional[str] = None) -> None:
        """Update TP/FP/FN from one batch of predictions and ground truth."""
        if condition is None and lighting is not None:
            condition = lighting

        pred_boxes = pred_boxes.to(self.device)
        pred_labels = pred_labels.to(self.device)
        pred_scores = pred_scores.to(self.device)
        gt_boxes = gt_boxes.to(self.device)
        gt_labels = gt_labels.to(self.device)

        for label in gt_labels:
            class_idx = int(label.item())
            if 0 <= class_idx < self.num_classes:
                self.class_gt_counts[class_idx] += 1

        if len(pred_boxes) == 0:
            self._handle_no_predictions(gt_boxes, gt_labels, condition)
            return
        if len(gt_boxes) == 0:
            self._handle_no_ground_truth(pred_boxes, pred_labels, pred_scores, condition)
            return

        # Vectorized matching: compute IoU matrix and match efficiently.
        iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)  # [N_pred, M_gt].
        pred_classes = pred_labels.long()
        gt_classes = gt_labels.long()
        
        # Filter valid classes.
        valid_pred_mask = (pred_classes >= 0) & (pred_classes < self.num_classes)
        valid_gt_mask = (gt_classes >= 0) & (gt_classes < self.num_classes)
        
        if not valid_pred_mask.any() or not valid_gt_mask.any():
            return
        
        # Sort predictions by score (descending)
        sorted_indices = torch.argsort(pred_scores, descending=True)
        matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool, device=self.device)
        
        # Vectorized size category computation.
        pred_areas = pred_boxes[:, 2] * pred_boxes[:, 3] * (self.image_size[0] * self.image_size[1])
        gt_areas = gt_boxes[:, 2] * gt_boxes[:, 3] * (self.image_size[0] * self.image_size[1])
        
        def get_size_category_vectorized(areas: torch.Tensor) -> torch.Tensor:
            """Vectorized size category: 0=small, 1=medium, 2=large."""
            small = areas < (32 * 32)
            medium = (areas >= (32 * 32)) & (areas < (96 * 96))
            large = areas >= (96 * 96)
            return torch.where(small, 0, torch.where(medium, 1, 2))
        
        pred_size_cats = get_size_category_vectorized(pred_areas)
        gt_size_cats = get_size_category_vectorized(gt_areas)
        
        # Match predictions to GT (vectorized where possible)
        for idx in sorted_indices:
            pred_idx = int(idx.item())
            if not valid_pred_mask[pred_idx]:
                continue
            
            pred_class = int(pred_classes[pred_idx].item())
            pred_score = float(pred_scores[pred_idx].item())
            pred_box = pred_boxes[pred_idx]
            
            # Find best matching GT of same class.
            class_match = (gt_classes == pred_class) & valid_gt_mask & (~matched_gt)
            if not class_match.any():
                # FP: no matching GT.
                self.class_fp[pred_class] += 1
                if condition:
                    self.condition_metrics[condition]['fp'] += 1
                size_cat_idx = int(pred_size_cats[pred_idx].item())
                size_cat = ['small', 'medium', 'large'][size_cat_idx]
                self.size_metrics[size_cat]['fp'] += 1
                if self.store_predictions:
                    self.class_predictions[pred_class].append(
                        DetectionPrediction(pred_score, 0.0, None, float(pred_areas[pred_idx].item()))
                    )
                continue
            
            # Get IoU with matching GTs.
            matching_ious = iou_matrix[pred_idx, class_match]
            best_match_idx = torch.argmax(matching_ious)
            best_gt_idx = torch.where(class_match)[0][best_match_idx]
            best_iou = float(matching_ious[best_match_idx].item())
            
            is_tp = best_iou >= iou_threshold
            if is_tp:
                self.class_tp[pred_class] += 1
                matched_gt[best_gt_idx] = True
                if condition:
                    self.condition_metrics[condition]['tp'] += 1
                size_cat_idx = int(pred_size_cats[pred_idx].item())
                size_cat = ['small', 'medium', 'large'][size_cat_idx]
                self.size_metrics[size_cat]['tp'] += 1
                if self.store_predictions:
                    self.class_predictions[pred_class].append(
                        DetectionPrediction(pred_score, best_iou, int(best_gt_idx.item()), float(pred_areas[pred_idx].item()))
                    )
            else:
                self.class_fp[pred_class] += 1
                if condition:
                    self.condition_metrics[condition]['fp'] += 1
                size_cat_idx = int(pred_size_cats[pred_idx].item())
                size_cat = ['small', 'medium', 'large'][size_cat_idx]
                self.size_metrics[size_cat]['fp'] += 1
                if self.store_predictions:
                    self.class_predictions[pred_class].append(
                        DetectionPrediction(pred_score, best_iou, None, float(pred_areas[pred_idx].item()))
                    )
        
        # Count unmatched GTs (FN)
        unmatched_gt_mask = ~matched_gt & valid_gt_mask
        if unmatched_gt_mask.any():
            unmatched_classes = gt_classes[unmatched_gt_mask]
            unmatched_sizes = gt_size_cats[unmatched_gt_mask]
            for gt_class, size_cat_idx in zip(unmatched_classes, unmatched_sizes):
                gt_class_int = int(gt_class.item())
                if 0 <= gt_class_int < self.num_classes:
                    self.class_fn[gt_class_int] += 1
                    if condition:
                        self.condition_metrics[condition]['fn'] += 1
                    size_cat = ['small', 'medium', 'large'][int(size_cat_idx.item())]
                    self.size_metrics[size_cat]['fn'] += 1

    def _handle_no_predictions(self, gt_boxes, gt_labels, condition) -> None:
        """Count all ground-truth boxes as false negatives when there are no predictions."""
        for gt_label, gt_box in zip(gt_labels, gt_boxes):
            class_idx = int(gt_label.item())
            if 0 <= class_idx < self.num_classes:
                self.class_fn[class_idx] += 1
                if condition:
                    self.condition_metrics[condition]['fn'] += 1
                size_cat = self._get_size_category(gt_box)
                self.size_metrics[size_cat]['fn'] += 1

    def _handle_no_ground_truth(self, pred_boxes, pred_labels, pred_scores, condition) -> None:
        """Count all predictions as false positives when there is no ground truth."""
        for pred_label, pred_score, pred_box in zip(pred_labels, pred_scores, pred_boxes):
            class_idx = int(pred_label.item())
            if 0 <= class_idx < self.num_classes:
                self.class_fp[class_idx] += 1
                if condition:
                    self.condition_metrics[condition]['fp'] += 1
                box_area = (pred_box[2] * pred_box[3]).item()
                self.class_predictions[class_idx].append(
                    DetectionPrediction(pred_score, 0.0, None, box_area)
                )
                size_cat = self._get_size_category(pred_box)
                self.size_metrics[size_cat]['fp'] += 1

    def compute_precision(self, class_idx: Optional[int] = None) -> float:
        """Precision TP/(TP+FP) for one class or overall."""
        if class_idx is None:
            tp, fp = self.class_tp.sum().item(), self.class_fp.sum().item()
        else:
            if not (0 <= class_idx < self.num_classes):
                return 0.0
            tp, fp = self.class_tp[class_idx].item(), self.class_fp[class_idx].item()
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def compute_recall(self, class_idx: Optional[int] = None) -> float:
        """Recall TP/(TP+FN) for one class or overall."""
        if class_idx is None:
            tp, fn = self.class_tp.sum().item(), self.class_fn.sum().item()
        else:
            if not (0 <= class_idx < self.num_classes):
                return 0.0
            tp, fn = self.class_tp[class_idx].item(), self.class_fn[class_idx].item()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def compute_f1(self, class_idx: Optional[int] = None) -> float:
        """F1 score (harmonic mean of precision and recall) for one class or overall."""
        precision, recall = self.compute_precision(class_idx), self.compute_recall(class_idx)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    def compute_ap_vectorized(self, class_idx: int, iou_threshold: float = 0.5) -> float:
        """Average precision for one class from stored predictions (precision-recall curve)."""
        preds = self.class_predictions.get(class_idx, [])
        if not preds:
            return 0.0
        scores = np.array([p.score for p in preds], dtype=np.float32)
        ious = np.array([p.iou for p in preds], dtype=np.float32)
        matched = np.array([p.matched_gt_idx is not None for p in preds], dtype=bool)

        sort_idx = np.argsort(-scores)
        ious, matched = ious[sort_idx], matched[sort_idx]
        tp = (ious >= iou_threshold) & matched
        fp = ~tp
        tp_cumsum, fp_cumsum = np.cumsum(tp, dtype=np.float32), np.cumsum(fp, dtype=np.float32)

        total_gt = self.class_gt_counts[class_idx].item()
        if total_gt == 0:
            return 0.0

        recall = tp_cumsum / total_gt
        precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-9)
        precision = np.maximum.accumulate(precision[::-1])[::-1]

        recall_unique, unique_idx = np.unique(recall, return_index=True)
        precision_unique = precision[unique_idx]
        return float(np.trapz(precision_unique, recall_unique))

    def compute_map_vectorized(self, iou_threshold: Optional[float] = None) -> Dict[str, float]:
        """Mean average precision over classes; optional per-threshold and per-class AP."""
        thresholds = [iou_threshold] if iou_threshold else self.iou_thresholds
        per_threshold = {float(thresh): float(np.mean([self.compute_ap_vectorized(c, float(thresh)) 
                                                       for c in range(self.num_classes)])) 
                         for thresh in thresholds}

        # Convert dict values to list for proper type inference.
        threshold_values = list(per_threshold.values()) if per_threshold else []
        mean_map = float(np.mean(threshold_values)) if threshold_values else 0.0
        result = {
            'mAP': mean_map,
            'per_threshold': per_threshold,
            'per_class': [self.compute_ap_vectorized(c, float(thresholds[0])) for c in range(self.num_classes)]
        }
        if 0.5 in per_threshold:
            result['mAP@0.5'] = per_threshold[0.5]
        if 0.75 in per_threshold:
            result['mAP@0.75'] = per_threshold[0.75]
        return result

    def compute_coco_map(self) -> Dict[str, Any]:
        """COCO-style mAP at 0.5, 0.75, and [0.5:0.95] with per-threshold and per-class AP."""
        coco_thresholds = [round(t, 2) for t in np.arange(0.5, 1.0, 0.05)]
        original_thresholds = self.iou_thresholds
        self.iou_thresholds = coco_thresholds
        map_results = self.compute_map_vectorized()
        self.iou_thresholds = original_thresholds
        return {
            'mAP@0.5': map_results.get('mAP@0.5', 0.0),
            'mAP@0.75': map_results.get('mAP@0.75', 0.0),
            'mAP@[0.5:0.95]': map_results['mAP'],
            'per_threshold': map_results.get('per_threshold', {}),
            'per_class': map_results.get('per_class', [])
        }

    def get_metrics_by_condition(self) -> Dict[str, Dict[str, float]]:
        """Precision, recall, F1 per condition (e.g. lighting or impairment)."""
        results = {}
        for condition, counts in self.condition_metrics.items():
            tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Return 0.0 instead of nan.
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            results[condition] = {'precision': precision, 'recall': recall, 'f1': f1}
        return results

    def get_metrics_by_size(self) -> Dict[str, Dict[str, float]]:
        """Precision, recall, F1 for small/medium/large object sizes."""
        results = {}
        for size_cat, counts in self.size_metrics.items():
            tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Return 0.0 instead of nan.
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            results[size_cat] = {'precision': precision, 'recall': recall, 'f1': f1}
        return results

    def record_inference_time(self, time_ms: float) -> None:
        """Append one inference latency sample (ms) for later stats."""
        self.inference_times.append(time_ms)

    def get_latency_stats(self) -> Dict[str, float]:
        """Return mean, median, min, max, std, p95, p99 of recorded inference times (ms)."""
        if not self.inference_times:
            return {k: 0.0 for k in ['mean_ms','median_ms','min_ms','max_ms','std_ms','p95_ms','p99_ms']}
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
        """Clear recorded inference times."""
        self.inference_times = []

    def compute_ap(self, class_idx: int, iou_threshold: float = 0.5) -> float:
        """Average precision for one class; delegates to compute_ap_vectorized."""
        return self.compute_ap_vectorized(class_idx, iou_threshold)

    def compute_map(self, iou_threshold: Optional[float] = None) -> Dict[str, float]:
        """Mean average precision; delegates to compute_map_vectorized."""
        return self.compute_map_vectorized(iou_threshold)

    def get_per_class_metrics(self) -> Dict[int, Dict[str, float]]:
        """Precision, recall, F1, and AP per class index."""
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
        """Alias for get_metrics_by_condition (lighting/condition stratification)."""
        return self.get_metrics_by_condition()

    def compute_map_coco(self) -> Dict[str, float]:
        """COCO-style mAP dict; delegates to compute_coco_map."""
        return self.compute_coco_map()






