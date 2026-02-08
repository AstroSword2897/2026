"""Evaluation Metrics for Phase 9: Evaluation & Metrics Includes: - Multi-modal metrics - Accessibility-specific metrics - Robustness evaluation."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class MultiModalMetrics:
    """Multi-modal evaluation metrics for Phase 9."""
    vision_accuracy: float = 0.0
    audio_accuracy: float = 0.0
    haptic_accuracy: float = 0.0
    fusion_improvement: float = 0.0  # Improvement from fusion vs single modality.
    cross_modal_alignment: float = 0.0  # Alignment between modalities.


@dataclass
class AccessibilityMetrics:
    """Accessibility-specific metrics for Phase 9."""
    detection_rate: float = 0.0  # % of critical objects detected.
    false_positive_rate: float = 0.0
    response_time_ms: float = 0.0
    navigation_success_rate: float = 0.0
    text_readability_score: float = 0.0
    scene_description_quality: float = 0.0


@dataclass
class RobustnessMetrics:
    """Robustness evaluation metrics for Phase 9."""
    lighting_robustness: float = 0.0  # Performance across lighting conditions.
    occlusion_robustness: float = 0.0  # Performance with occlusions.
    motion_robustness: float = 0.0  # Performance with motion blur.
    noise_robustness: float = 0.0  # Performance with noise.
    adversarial_robustness: float = 0.0  # Performance against adversarial examples.


class EvaluationMetrics:
    """Comprehensive evaluation metrics for Phase 9. Provides: - Multi-modal metrics - Accessibility-specific metrics - Robustness evaluation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.multi_modal = MultiModalMetrics()
        self.accessibility = AccessibilityMetrics()
        self.robustness = RobustnessMetrics()
    
    def compute_multi_modal_metrics(
        self,
        vision_outputs: Dict[str, torch.Tensor],
        fused_outputs: Dict[str, torch.Tensor],
        ground_truth: Dict[str, torch.Tensor],
        audio_outputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> MultiModalMetrics:
        """Compute multi-modal evaluation metrics."""
        metrics = MultiModalMetrics()
        
        # Vision accuracy.
        if 'classifications' in vision_outputs and 'labels' in ground_truth:
            vision_pred = vision_outputs['classifications'].argmax(dim=-1)
            vision_acc = (vision_pred == ground_truth['labels']).float().mean()
            metrics.vision_accuracy = vision_acc.item()
        
        # Fusion improvement.
        if 'classifications' in fused_outputs:
            fused_pred = fused_outputs['classifications'].argmax(dim=-1)
            fused_acc = (fused_pred == ground_truth['labels']).float().mean()
            metrics.fusion_improvement = max(0.0, fused_acc.item() - metrics.vision_accuracy)
        
        return metrics
    
    def compute_accessibility_metrics(
        self,
        detections: List[Dict],
        ground_truth_detections: List[Dict],
        response_times: List[float],
        navigation_success: List[bool]
    ) -> AccessibilityMetrics:
        """Compute accessibility-specific metrics."""
        metrics = AccessibilityMetrics()
        
        # Detection rate.
        if ground_truth_detections:
            detected = sum(1 for gt in ground_truth_detections 
                          if any(self._iou_overlap(gt['box'], det['box']) > 0.5 
                                for det in detections))
            metrics.detection_rate = detected / len(ground_truth_detections)
        
        # False positive rate.
        if detections:
            false_positives = sum(1 for det in detections
                                if not any(self._iou_overlap(det['box'], gt['box']) > 0.5
                                          for gt in ground_truth_detections))
            metrics.false_positive_rate = false_positives / len(detections)
        
        # Response time.
        if response_times:
            metrics.response_time_ms = np.mean(response_times)
        
        # Navigation success rate.
        if navigation_success:
            metrics.navigation_success_rate = sum(navigation_success) / len(navigation_success)
        
        return metrics
    
    def compute_robustness_metrics(
        self,
        baseline_performance: float,
        lighting_performance: Dict[str, float],
        occlusion_performance: Dict[str, float],
        noise_performance: Dict[str, float]
    ) -> RobustnessMetrics:
        """Compute robustness evaluation metrics."""
        metrics = RobustnessMetrics()
        
        # Lighting robustness (performance retention)
        if lighting_performance:
            avg_lighting = np.mean(list(lighting_performance.values()))
            metrics.lighting_robustness = avg_lighting / baseline_performance if baseline_performance > 0 else 0.0
        
        # Occlusion robustness.
        if occlusion_performance:
            avg_occlusion = np.mean(list(occlusion_performance.values()))
            metrics.occlusion_robustness = avg_occlusion / baseline_performance if baseline_performance > 0 else 0.0
        
        # Noise robustness.
        if noise_performance:
            avg_noise = np.mean(list(noise_performance.values()))
            metrics.noise_robustness = avg_noise / baseline_performance if baseline_performance > 0 else 0.0
        
        return metrics
    
    def _iou_overlap(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two boxes."""
        # Convert to [x1, y1, x2, y2] format if needed.
        if len(box1) == 4 and len(box2) == 4:
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
            
            # Intersection.
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            
            if x2_i <= x1_i or y2_i <= y1_i:
                return 0.0
            
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        return 0.0
    
    def generate_report(self) -> Dict[str, Dict]:
        """Generate comprehensive evaluation report. Returns: Dictionary with all metrics."""
        return {
            'multi_modal': {
                'vision_accuracy': self.multi_modal.vision_accuracy,
                'fusion_improvement': self.multi_modal.fusion_improvement,
                'cross_modal_alignment': self.multi_modal.cross_modal_alignment
            },
            'accessibility': {
                'detection_rate': self.accessibility.detection_rate,
                'false_positive_rate': self.accessibility.false_positive_rate,
                'response_time_ms': self.accessibility.response_time_ms,
                'navigation_success_rate': self.accessibility.navigation_success_rate
            },
            'robustness': {
                'lighting_robustness': self.robustness.lighting_robustness,
                'occlusion_robustness': self.robustness.occlusion_robustness,
                'noise_robustness': self.robustness.noise_robustness
            }
        }







