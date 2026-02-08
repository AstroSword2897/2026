"""Per-Class Metrics and Confusion Matrix Analysis."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ClassMetrics:
    """Metrics for a single class."""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    confidences: List[float] = field(default_factory=list)
    
    @property
    def precision(self) -> float:
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    @property
    def recall(self) -> float:
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)
    
    @property
    def accuracy(self) -> float:
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total
    
    @property
    def support(self) -> int:
        """Number of actual positives."""
        return self.true_positives + self.false_negatives
    
    def to_dict(self) -> Dict:
        return {
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'accuracy': self.accuracy,
            'support': self.support,
            'tp': self.true_positives,
            'fp': self.false_positives,
            'fn': self.false_negatives,
            'tn': self.true_negatives,
            'avg_confidence': np.mean(self.confidences) if self.confidences else 0.0
        }


class ConfusionMatrix:
    """Confusion matrix with analysis capabilities."""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update confusion matrix with batch predictions."""
        preds = predictions.cpu().numpy()
        targs = targets.cpu().numpy()
        
        for p, t in zip(preds, targs):
            if 0 <= p < self.num_classes and 0 <= t < self.num_classes:
                self.matrix[t, p] += 1
                
    def get_per_class_metrics(self) -> Dict[str, ClassMetrics]:
        """Calculate per-class metrics from confusion matrix."""
        metrics = {}
        
        for i, name in enumerate(self.class_names):
            tp = self.matrix[i, i]
            fp = self.matrix[:, i].sum() - tp
            fn = self.matrix[i, :].sum() - tp
            tn = self.matrix.sum() - tp - fp - fn
            
            metrics[name] = ClassMetrics(
                true_positives=int(tp),
                false_positives=int(fp),
                false_negatives=int(fn),
                true_negatives=int(tn)
            )
            
        return metrics
    
    def get_most_confused_pairs(self, top_k: int = 10) -> List[Tuple[str, str, int]]:
        """Get the most commonly confused class pairs."""
        confused = []
        
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j and self.matrix[i, j] > 0:
                    confused.append((
                        self.class_names[i],  # True class.
                        self.class_names[j],  # Predicted class.
                        int(self.matrix[i, j])
                    ))
                    
        confused.sort(key=lambda x: x[2], reverse=True)
        return confused[:top_k]
    
    def get_worst_classes(self, metric: str = 'f1', top_k: int = 10) -> List[Tuple[str, float]]:
        """Get classes with worst performance."""
        metrics = self.get_per_class_metrics()
        
        class_scores = []
        for name, m in metrics.items():
            if m.support > 0:  # Only include classes with samples.
                score = getattr(m, metric)
                class_scores.append((name, score))
                
        class_scores.sort(key=lambda x: x[1])
        return class_scores[:top_k]
    
    def to_dict(self) -> Dict:
        """Export to dictionary."""
        return {
            'matrix': self.matrix.tolist(),
            'class_names': self.class_names,
            'num_classes': self.num_classes
        }
    
    def reset(self):
        """Reset the confusion matrix."""
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)


class PerClassMetricsTracker:
    """Comprehensive per-class metrics tracking. Tracks performance across classes, scenarios, impairments, and urgency levels."""
    
    def __init__(self, 
                 class_names: List[str],
                 scenario_names: Optional[List[str]] = None,
                 impairment_names: Optional[List[str]] = None,
                 urgency_levels: int = 4):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.scenario_names = scenario_names or []
        self.impairment_names = impairment_names or []
        self.urgency_levels = urgency_levels
        
        # Main confusion matrix.
        self.confusion_matrix = ConfusionMatrix(self.num_classes, class_names)
        
        # Per-class metrics.
        self.class_metrics: Dict[str, ClassMetrics] = {
            name: ClassMetrics() for name in class_names
        }
        
        # Per-scenario metrics.
        self.scenario_metrics: Dict[str, Dict[str, ClassMetrics]] = {
            scenario: {name: ClassMetrics() for name in class_names}
            for scenario in self.scenario_names
        }
        
        # Per-impairment metrics.
        self.impairment_metrics: Dict[str, Dict[str, ClassMetrics]] = {
            imp: {name: ClassMetrics() for name in class_names}
            for imp in self.impairment_names
        }
        
        # Per-urgency metrics.
        self.urgency_metrics: Dict[int, ClassMetrics] = {
            level: ClassMetrics() for level in range(urgency_levels)
        }
        
        # Prediction logging.
        self.predictions_log: List[Dict] = []
        self.misclassifications_log: List[Dict] = []
        
    def update(self, 
               predictions: torch.Tensor,
               targets: torch.Tensor,
               confidences: Optional[torch.Tensor] = None,
               metadata: Optional[List[Dict]] = None):
        """Update metrics with batch predictions."""
        batch_size = predictions.shape[0]
        
        # Update confusion matrix.
        self.confusion_matrix.update(predictions, targets)
        
        # Update per-class metrics.
        preds = predictions.cpu().numpy()
        targs = targets.cpu().numpy()
        confs = confidences.cpu().numpy() if confidences is not None else [None] * batch_size
        
        for i in range(batch_size):
            pred = int(preds[i])
            target = int(targs[i])
            conf = float(confs[i]) if confs[i] is not None else 0.0
            
            pred_name = self.class_names[pred] if pred < len(self.class_names) else f"class_{pred}"
            target_name = self.class_names[target] if target < len(self.class_names) else f"class_{target}"
            
            # Update class metrics.
            if target_name in self.class_metrics:
                if pred == target:
                    self.class_metrics[target_name].true_positives += 1
                else:
                    self.class_metrics[target_name].false_negatives += 1
                if conf:
                    self.class_metrics[target_name].confidences.append(conf)
                    
            if pred_name in self.class_metrics and pred != target:
                self.class_metrics[pred_name].false_positives += 1
                
            # Log prediction.
            log_entry = {
                'prediction': pred_name,
                'target': target_name,
                'correct': pred == target,
                'confidence': conf
            }
            
            # Add metadata.
            if metadata and i < len(metadata):
                meta = metadata[i]
                log_entry.update(meta)
                
                # Update scenario metrics.
                scenario = meta.get('scenario')
                if scenario and scenario in self.scenario_metrics:
                    if pred == target:
                        self.scenario_metrics[scenario][target_name].true_positives += 1
                    else:
                        self.scenario_metrics[scenario][target_name].false_negatives += 1
                        
                # Update impairment metrics.
                impairment = meta.get('impairment')
                if impairment and impairment in self.impairment_metrics:
                    if pred == target:
                        self.impairment_metrics[impairment][target_name].true_positives += 1
                    else:
                        self.impairment_metrics[impairment][target_name].false_negatives += 1
                        
                # Update urgency metrics.
                urgency = meta.get('urgency')
                if urgency is not None and urgency in self.urgency_metrics:
                    if pred == target:
                        self.urgency_metrics[urgency].true_positives += 1
                    else:
                        self.urgency_metrics[urgency].false_negatives += 1
                        
            self.predictions_log.append(log_entry)
            
            if pred != target:
                self.misclassifications_log.append(log_entry)
                
    def get_summary(self) -> Dict:
        """Get comprehensive metrics summary."""
        # Per-class summary.
        class_summary = {}
        for name, metrics in self.class_metrics.items():
            if metrics.support > 0:
                class_summary[name] = metrics.to_dict()
                
        # Worst classes.
        worst_classes = self.confusion_matrix.get_worst_classes('f1', 10)
        
        # Most confused pairs.
        confused_pairs = self.confusion_matrix.get_most_confused_pairs(10)
        
        # Aggregate metrics.
        total_tp = sum(m.true_positives for m in self.class_metrics.values())
        total_fp = sum(m.false_positives for m in self.class_metrics.values())
        total_fn = sum(m.false_negatives for m in self.class_metrics.values())
        
        macro_precision = np.mean([m.precision for m in self.class_metrics.values() if m.support > 0])
        macro_recall = np.mean([m.recall for m in self.class_metrics.values() if m.support > 0])
        macro_f1 = np.mean([m.f1 for m in self.class_metrics.values() if m.support > 0])
        
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        
        # Per-scenario summary.
        scenario_summary = {}
        for scenario, metrics in self.scenario_metrics.items():
            total_correct = sum(m.true_positives for m in metrics.values())
            total_samples = sum(m.support for m in metrics.values())
            scenario_summary[scenario] = {
                'accuracy': total_correct / total_samples if total_samples > 0 else 0,
                'samples': total_samples
            }
            
        # Per-impairment summary.
        impairment_summary = {}
        for imp, metrics in self.impairment_metrics.items():
            total_correct = sum(m.true_positives for m in metrics.values())
            total_samples = sum(m.support for m in metrics.values())
            impairment_summary[imp] = {
                'accuracy': total_correct / total_samples if total_samples > 0 else 0,
                'samples': total_samples
            }
            
        # Per-urgency summary.
        urgency_summary = {}
        for level, metrics in self.urgency_metrics.items():
            urgency_summary[f"urgency_{level}"] = {
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1': metrics.f1,
                'support': metrics.support
            }
            
        return {
            'aggregate': {
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'macro_f1': macro_f1,
                'micro_precision': micro_precision,
                'micro_recall': micro_recall,
                'micro_f1': micro_f1,
                'total_predictions': len(self.predictions_log),
                'total_misclassifications': len(self.misclassifications_log),
                'accuracy': 1 - len(self.misclassifications_log) / max(len(self.predictions_log), 1)
            },
            'per_class': class_summary,
            'worst_classes': worst_classes,
            'confused_pairs': confused_pairs,
            'per_scenario': scenario_summary,
            'per_impairment': impairment_summary,
            'per_urgency': urgency_summary
        }
    
    def get_critical_failures(self) -> Dict:
        """Identify critical failures for high-urgency classes."""
        critical = {
            'high_urgency_misses': [],
            'safety_critical_errors': [],
            'low_confidence_correct': []
        }
        
        for log in self.misclassifications_log:
            urgency = log.get('urgency', 0)
            
            # High urgency misclassifications.
            if urgency >= 2:
                critical['high_urgency_misses'].append(log)
                
            # Safety-critical categories.
            safety_classes = ['stairs', 'exit_sign', 'fire_hydrant', 'stop sign', 
                            'traffic light', 'car', 'truck', 'bus', 'person']
            if log['target'] in safety_classes:
                critical['safety_critical_errors'].append(log)
                
        # Low confidence correct predictions (potential fragile predictions)
        for log in self.predictions_log:
            if log['correct'] and log.get('confidence', 1.0) < 0.5:
                critical['low_confidence_correct'].append(log)
                
        return critical
    
    def save_report(self, output_path: Path):
        """Save comprehensive metrics report."""
        summary = self.get_summary()
        critical = self.get_critical_failures()
        
        report = {
            'summary': summary,
            'critical_failures': {
                'high_urgency_misses': len(critical['high_urgency_misses']),
                'safety_critical_errors': len(critical['safety_critical_errors']),
                'low_confidence_correct': len(critical['low_confidence_correct']),
                'samples': {
                    'high_urgency': critical['high_urgency_misses'][:20],
                    'safety_critical': critical['safety_critical_errors'][:20]
                }
            },
            'confusion_matrix': self.confusion_matrix.to_dict()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Metrics report saved to {output_path}")
        
    def reset(self):
        """Reset all metrics."""
        self.confusion_matrix.reset()
        for m in self.class_metrics.values():
            m.true_positives = 0
            m.false_positives = 0
            m.false_negatives = 0
            m.true_negatives = 0
            m.confidences.clear()
        self.predictions_log.clear()
        self.misclassifications_log.clear()


def compute_map_per_class(predictions: List[Dict], 
                          targets: List[Dict],
                          iou_threshold: float = 0.5) -> Dict[str, float]:
    """Compute mAP per class for object detection."""
    # Collect all predictions and targets by class.
    class_predictions = defaultdict(list)
    class_targets = defaultdict(lambda: {'boxes': [], 'matched': []})
    
    for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
        pred_boxes = pred.get('boxes', [])
        pred_labels = pred.get('labels', [])
        pred_scores = pred.get('scores', [])
        
        target_boxes = target.get('boxes', [])
        target_labels = target.get('labels', [])
        
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            class_predictions[label].append({
                'box': box,
                'score': score,
                'img_idx': img_idx
            })
            
        for box, label in zip(target_boxes, target_labels):
            class_targets[label]['boxes'].append({
                'box': box,
                'img_idx': img_idx
            })
            class_targets[label]['matched'].append(False)
            
    # Compute AP per class.
    ap_per_class = {}
    
    for class_id in set(class_predictions.keys()) | set(class_targets.keys()):
        preds = sorted(class_predictions[class_id], 
                      key=lambda x: x['score'], reverse=True)
        targets_info = class_targets[class_id]
        
        if not targets_info['boxes']:
            ap_per_class[class_id] = 0.0 if preds else 1.0
            continue
            
        if not preds:
            ap_per_class[class_id] = 0.0
            continue
            
        # Match predictions to targets.
        tp = []
        fp = []
        matched = [False] * len(targets_info['boxes'])
        
        for pred in preds:
            best_iou = 0
            best_idx = -1
            
            for idx, target in enumerate(targets_info['boxes']):
                if target['img_idx'] != pred['img_idx']:
                    continue
                if matched[idx]:
                    continue
                    
                iou = compute_iou(pred['box'], target['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
                    
            if best_iou >= iou_threshold and best_idx >= 0:
                tp.append(1)
                fp.append(0)
                matched[best_idx] = True
            else:
                tp.append(0)
                fp.append(1)
                
        # Compute precision-recall curve.
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(targets_info['boxes'])
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Compute AP using 11-point interpolation.
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            prec_at_recall = [p for p, r in zip(precisions, recalls) if r >= t]
            ap += max(prec_at_recall) if prec_at_recall else 0
        ap /= 11
        
        ap_per_class[class_id] = ap
        
    return ap_per_class


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
        
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0







