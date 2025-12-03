"""Scene-level metrics: distance estimation accuracy, urgency prediction accuracy."""

import torch
import time
import numpy as np
from typing import Dict, Optional
from collections import defaultdict


class SceneMetrics:
    """Tracks scene-level metrics: distance zones and urgency levels."""
    
    def __init__(self, num_urgency_levels: int = 4, num_distance_zones: int = 3):
        self.num_urgency_levels = num_urgency_levels
        self.num_distance_zones = num_distance_zones
        self.reset()
    
    def reset(self):  # ← Must be indented 4 spaces (class method level)
        """Reset all accumulators."""
        # Urgency metrics
        self.urgency_correct = 0
        self.urgency_total = 0
        self.urgency_confusion = torch.zeros(
            self.num_urgency_levels, 
            self.num_urgency_levels, 
            dtype=torch.long
        )
        
        # Distance metrics (per-object)
        self.distance_correct = 0
        self.distance_total = 0
        self.distance_confusion = torch.zeros(
            self.num_distance_zones,
            self.num_distance_zones,
            dtype=torch.long
        )
        
        # Per-urgency-level accuracy
        self.urgency_level_correct = defaultdict(int)
        self.urgency_level_total = defaultdict(int)
        
        # Latency tracking
        self.inference_times = []
    
    def update_urgency(  # ← Must be at class level (4 spaces), NOT inside reset()
        self,
        pred_urgency: torch.Tensor,
        gt_urgency: torch.Tensor
    ) -> None:
        """Update urgency prediction metrics. Expects [B] tensors."""
        pred_urgency = pred_urgency.cpu()
        gt_urgency = gt_urgency.cpu()
        
        # Count correct predictions
        correct = (pred_urgency == gt_urgency).sum().item()
        total = pred_urgency.numel()
        
        self.urgency_correct += correct
        self.urgency_total += total
        
        # Update confusion matrix
        for p, g in zip(pred_urgency.flatten(), gt_urgency.flatten()):
            p_idx = int(p.item())
            g_idx = int(g.item())
            if 0 <= p_idx < self.num_urgency_levels and 0 <= g_idx < self.num_urgency_levels:
                self.urgency_confusion[g_idx, p_idx] += 1
                self.urgency_level_total[g_idx] += 1
                if p_idx == g_idx:
                    self.urgency_level_correct[g_idx] += 1
    
    def update_distance(  # ← Must be at class level
        self,
        pred_distance: torch.Tensor,
        gt_distance: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> None:
        """Update distance zone prediction metrics. Expects [B, N] tensors."""
        pred_distance = pred_distance.cpu()
        gt_distance = gt_distance.cpu()
        
        if valid_mask is not None:
            valid_mask = valid_mask.cpu()
            pred_distance = pred_distance[valid_mask]
            gt_distance = gt_distance[valid_mask]
        
        # Count correct predictions
        correct = (pred_distance == gt_distance).sum().item()
        total = pred_distance.numel()
        
        self.distance_correct += correct
        self.distance_total += total
        
        # Update confusion matrix
        for p, g in zip(pred_distance.flatten(), gt_distance.flatten()):
            p_idx = int(p.item())
            g_idx = int(g.item())
            if 0 <= p_idx < self.num_distance_zones and 0 <= g_idx < self.num_distance_zones:
                self.distance_confusion[g_idx, p_idx] += 1
    
    def compute_urgency_accuracy(self) -> float:
        """Overall urgency prediction accuracy."""
        if self.urgency_total == 0:
            return 0.0
        return self.urgency_correct / self.urgency_total
    
    def compute_distance_accuracy(self) -> float:
        """Overall distance zone prediction accuracy."""
        if self.distance_total == 0:
            return 0.0
        return self.distance_correct / self.distance_total
    
    def get_per_urgency_accuracy(self) -> Dict[int, float]:
        """Accuracy per urgency level."""
        results = {}
        for level in range(self.num_urgency_levels):
            total = self.urgency_level_total.get(level, 0)
            correct = self.urgency_level_correct.get(level, 0)
            results[level] = correct / total if total > 0 else 0.0
        return results
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all scene metrics."""
        metrics = {
            'urgency_accuracy': self.compute_urgency_accuracy(),
            'distance_accuracy': self.compute_distance_accuracy(),
        }
        
        # Add per-urgency-level accuracies
        per_urgency = self.get_per_urgency_accuracy()
        for level, acc in per_urgency.items():
            metrics[f'urgency_level_{level}_accuracy'] = acc
        
        # Add latency stats if available
        if len(self.inference_times) > 0:
            latency_stats = self.get_latency_stats()
            metrics.update(latency_stats)
        
        return metrics
    
    def record_inference_time(self, inference_time_ms: float) -> None:
        """Record inference latency for a single forward pass."""
        self.inference_times.append(inference_time_ms)
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics from recorded inference times."""
        if len(self.inference_times) == 0:
            return {}
        
        times = np.array(self.inference_times)
        return {
            'mean_latency_ms': float(np.mean(times)),
            'median_latency_ms': float(np.median(times)),
            'min_latency_ms': float(np.min(times)),
            'max_latency_ms': float(np.max(times)),
            'p95_latency_ms': float(np.percentile(times, 95)),
            'p99_latency_ms': float(np.percentile(times, 99))
        }
    
    def reset_latency(self) -> None:
        """Reset latency tracking."""
        self.inference_times = []