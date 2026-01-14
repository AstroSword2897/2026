"""
Stress Testing and Edge Case Evaluation

Implements:
- Stress test dataset generation
- Edge case scenarios
- Robustness evaluation
- Performance degradation measurement
- Inference benchmarking
- Fallback system for uncertain predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


# ==================== Edge Case Scenarios ====================

@dataclass
class EdgeCaseScenario:
    """Definition of an edge case test scenario."""
    name: str
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    transform_fn: Optional[Callable] = None
    expected_drop: float = 0.0  # Expected accuracy drop
    
    
EDGE_CASE_SCENARIOS = [
    # Lighting extremes
    EdgeCaseScenario(
        name="extreme_overexposure",
        description="Heavily overexposed images (bright sunlight, flash)",
        severity="high",
        expected_drop=0.15
    ),
    EdgeCaseScenario(
        name="extreme_underexposure",
        description="Very dark images (night, dim indoor)",
        severity="high",
        expected_drop=0.20
    ),
    EdgeCaseScenario(
        name="harsh_backlighting",
        description="Subject silhouetted against bright background",
        severity="medium",
        expected_drop=0.12
    ),
    
    # Motion and blur
    EdgeCaseScenario(
        name="severe_motion_blur",
        description="Heavy motion blur from camera shake or fast movement",
        severity="high",
        expected_drop=0.25
    ),
    EdgeCaseScenario(
        name="focus_blur",
        description="Out-of-focus images",
        severity="medium",
        expected_drop=0.18
    ),
    
    # Occlusion
    EdgeCaseScenario(
        name="heavy_occlusion",
        description="Objects 50%+ occluded by other objects",
        severity="high",
        expected_drop=0.30
    ),
    EdgeCaseScenario(
        name="partial_frame",
        description="Objects partially outside frame",
        severity="medium",
        expected_drop=0.15
    ),
    
    # Viewpoint
    EdgeCaseScenario(
        name="extreme_angle",
        description="Objects viewed from unusual angles",
        severity="medium",
        expected_drop=0.20
    ),
    EdgeCaseScenario(
        name="very_small_objects",
        description="Objects occupying <5% of frame",
        severity="high",
        expected_drop=0.25
    ),
    EdgeCaseScenario(
        name="very_large_objects",
        description="Objects filling >80% of frame",
        severity="low",
        expected_drop=0.08
    ),
    
    # Environmental
    EdgeCaseScenario(
        name="rain_droplets",
        description="Camera lens with rain droplets",
        severity="medium",
        expected_drop=0.15
    ),
    EdgeCaseScenario(
        name="fog_haze",
        description="Heavy fog or atmospheric haze",
        severity="high",
        expected_drop=0.22
    ),
    EdgeCaseScenario(
        name="reflections",
        description="Strong reflections on glass/water",
        severity="medium",
        expected_drop=0.18
    ),
    
    # Camera artifacts
    EdgeCaseScenario(
        name="heavy_compression",
        description="Severe JPEG compression artifacts",
        severity="medium",
        expected_drop=0.12
    ),
    EdgeCaseScenario(
        name="sensor_noise",
        description="High ISO noise",
        severity="medium",
        expected_drop=0.15
    ),
    EdgeCaseScenario(
        name="lens_distortion",
        description="Strong barrel/pincushion distortion",
        severity="low",
        expected_drop=0.08
    ),
    
    # Out of distribution
    EdgeCaseScenario(
        name="unusual_colors",
        description="Objects with unusual/unexpected colors",
        severity="medium",
        expected_drop=0.20
    ),
    EdgeCaseScenario(
        name="crowded_scene",
        description="Many overlapping objects",
        severity="high",
        expected_drop=0.25
    ),
]


def generate_edge_case_transforms() -> Dict[str, Callable]:
    """Generate transform functions for each edge case."""
    transforms = {}
    
    def extreme_overexposure(x: torch.Tensor) -> torch.Tensor:
        return (x * 2.5).clamp(0, 1)
    
    def extreme_underexposure(x: torch.Tensor) -> torch.Tensor:
        return x * 0.15
    
    def severe_motion_blur(x: torch.Tensor) -> torch.Tensor:
        kernel_size = 15
        kernel = torch.zeros(kernel_size, kernel_size, device=x.device)
        kernel[kernel_size//2, :] = 1 / kernel_size
        kernel = kernel.reshape(1, 1, kernel_size, kernel_size)
        
        if x.dim() == 3:
            x = x.unsqueeze(0)
        channels = x.shape[1]
        kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
        return F.conv2d(x, kernel, padding=kernel_size//2, groups=channels).squeeze(0)
    
    def heavy_occlusion(x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        mask = torch.ones_like(x)
        mask[..., h//4:3*h//4, w//4:3*w//4] = 0.2
        return x * mask
    
    def fog_haze(x: torch.Tensor) -> torch.Tensor:
        fog = torch.ones_like(x) * 0.7
        return x * 0.4 + fog * 0.6
    
    def sensor_noise(x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * 0.15
        return (x + noise).clamp(0, 1)
    
    def heavy_compression(x: torch.Tensor) -> torch.Tensor:
        block = 16
        h, w = x.shape[-2:]
        new_h = (h // block) * block
        new_w = (w // block) * block
        
        x_small = F.interpolate(x.unsqueeze(0), size=(new_h//4, new_w//4), mode='bilinear')
        x_back = F.interpolate(x_small, size=(new_h, new_w), mode='bilinear').squeeze(0)
        
        result = x.clone()
        result[..., :new_h, :new_w] = x_back
        return result
    
    transforms = {
        'extreme_overexposure': extreme_overexposure,
        'extreme_underexposure': extreme_underexposure,
        'severe_motion_blur': severe_motion_blur,
        'heavy_occlusion': heavy_occlusion,
        'fog_haze': fog_haze,
        'sensor_noise': sensor_noise,
        'heavy_compression': heavy_compression,
    }
    
    return transforms


# ==================== Stress Test Evaluation ====================

class StressTestEvaluator:
    """
    Evaluate model robustness under stress conditions.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 baseline_accuracy: float = 0.0,
                 device: str = 'cpu'):
        self.model = model
        self.baseline_accuracy = baseline_accuracy
        self.device = device
        self.edge_transforms = generate_edge_case_transforms()
        self.results: Dict[str, Dict] = {}
        
    def evaluate_scenario(self,
                         scenario: EdgeCaseScenario,
                         dataloader: torch.utils.data.DataLoader) -> Dict:
        """Evaluate model on a specific edge case scenario."""
        self.model.eval()
        
        correct = 0
        total = 0
        confidences = []
        predictions = []
        
        transform_fn = self.edge_transforms.get(scenario.name)
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    images = batch.get('images') or batch.get('image')
                    targets = batch.get('labels') or batch.get('targets', {}).get('labels')
                else:
                    images, targets = batch[0], batch[1]
                    
                images = images.to(self.device)
                
                # Apply stress transform
                if transform_fn:
                    images = torch.stack([transform_fn(img) for img in images])
                    
                outputs = self.model(images)
                
                # Handle different output formats
                if isinstance(outputs, dict):
                    logits = outputs.get('logits') or outputs.get('classifications')
                else:
                    logits = outputs
                    
                if logits is not None:
                    probs = F.softmax(logits, dim=-1)
                    max_probs, preds = probs.max(dim=-1)
                    
                    confidences.extend(max_probs.cpu().tolist())
                    predictions.extend(preds.cpu().tolist())
                    
                    if targets is not None:
                        if isinstance(targets, dict):
                            labels = targets.get('labels', targets.get('classifications'))
                        else:
                            labels = targets
                        if labels is not None:
                            labels = labels.to(self.device)
                            correct += (preds == labels).sum().item()
                            total += labels.size(0)
                            
        accuracy = correct / total if total > 0 else 0
        accuracy_drop = self.baseline_accuracy - accuracy if self.baseline_accuracy > 0 else 0
        
        result = {
            'scenario': scenario.name,
            'severity': scenario.severity,
            'accuracy': accuracy,
            'accuracy_drop': accuracy_drop,
            'expected_drop': scenario.expected_drop,
            'within_tolerance': accuracy_drop <= scenario.expected_drop * 1.2,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'min_confidence': min(confidences) if confidences else 0,
            'samples_evaluated': total
        }
        
        self.results[scenario.name] = result
        return result
    
    def run_full_stress_test(self,
                            dataloader: torch.utils.data.DataLoader) -> Dict:
        """Run all stress test scenarios."""
        logger.info("Starting full stress test evaluation...")
        
        all_results = []
        
        for scenario in EDGE_CASE_SCENARIOS:
            if scenario.name in self.edge_transforms:
                logger.info(f"Testing scenario: {scenario.name}")
                result = self.evaluate_scenario(scenario, dataloader)
                all_results.append(result)
                
                status = "✓" if result['within_tolerance'] else "✗"
                logger.info(f"  {status} Accuracy: {result['accuracy']:.3f}, "
                          f"Drop: {result['accuracy_drop']:.3f}")
                
        # Summary
        passed = sum(1 for r in all_results if r['within_tolerance'])
        failed = len(all_results) - passed
        
        critical_failures = [r for r in all_results 
                           if r['severity'] == 'critical' and not r['within_tolerance']]
        
        summary = {
            'total_scenarios': len(all_results),
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / len(all_results) if all_results else 0,
            'critical_failures': len(critical_failures),
            'robustness_score': self._compute_robustness_score(all_results),
            'results': all_results
        }
        
        return summary
    
    def _compute_robustness_score(self, results: List[Dict]) -> float:
        """Compute overall robustness score (0-100)."""
        if not results:
            return 0
            
        # Weight by severity
        severity_weights = {'low': 0.5, 'medium': 1.0, 'high': 1.5, 'critical': 2.0}
        
        weighted_scores = []
        total_weight = 0
        
        for r in results:
            weight = severity_weights.get(r['severity'], 1.0)
            
            # Score based on accuracy relative to expected drop
            if r['expected_drop'] > 0:
                performance_ratio = 1 - (r['accuracy_drop'] / r['expected_drop'])
            else:
                performance_ratio = 1 if r['accuracy_drop'] == 0 else 0
                
            score = max(0, min(100, performance_ratio * 100))
            weighted_scores.append(score * weight)
            total_weight += weight
            
        return sum(weighted_scores) / total_weight if total_weight > 0 else 0


# ==================== Inference Benchmarking ====================

@dataclass
class InferenceBenchmark:
    """Results from inference benchmarking."""
    model_name: str
    device: str
    batch_size: int
    avg_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_fps: float
    memory_mb: float
    num_parameters: int
    warmup_iterations: int = 10
    benchmark_iterations: int = 100
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'device': self.device,
            'batch_size': self.batch_size,
            'avg_latency_ms': self.avg_latency_ms,
            'std_latency_ms': self.std_latency_ms,
            'min_latency_ms': self.min_latency_ms,
            'max_latency_ms': self.max_latency_ms,
            'throughput_fps': self.throughput_fps,
            'memory_mb': self.memory_mb,
            'num_parameters': self.num_parameters
        }


class InferenceBenchmarker:
    """Benchmark model inference performance."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
    def benchmark(self,
                 input_shape: Tuple[int, ...] = (1, 3, 224, 224),
                 warmup_iterations: int = 10,
                 benchmark_iterations: int = 100) -> InferenceBenchmark:
        """Run inference benchmark."""
        batch_size = input_shape[0]
        
        # Create dummy input
        x = torch.randn(input_shape, device=self.device)
        
        # Warmup
        for _ in range(warmup_iterations):
            with torch.no_grad():
                _ = self.model(x)
                
        if self.device == 'cuda':
            torch.cuda.synchronize()
            
        # Benchmark
        latencies = []
        
        for _ in range(benchmark_iterations):
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = self.model(x)
                
            if self.device == 'cuda':
                torch.cuda.synchronize()
                
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms
            
        # Memory usage
        if self.device == 'cuda':
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            import psutil
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            
        # Parameter count
        num_params = sum(p.numel() for p in self.model.parameters())
        
        return InferenceBenchmark(
            model_name=self.model.__class__.__name__,
            device=self.device,
            batch_size=batch_size,
            avg_latency_ms=np.mean(latencies),
            std_latency_ms=np.std(latencies),
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            throughput_fps=1000 / np.mean(latencies) * batch_size,
            memory_mb=memory_mb,
            num_parameters=num_params,
            warmup_iterations=warmup_iterations,
            benchmark_iterations=benchmark_iterations
        )
    
    def benchmark_batch_sizes(self,
                             input_channels: int = 3,
                             input_size: int = 224,
                             batch_sizes: List[int] = [1, 2, 4, 8, 16, 32]) -> List[InferenceBenchmark]:
        """Benchmark across different batch sizes."""
        results = []
        
        for bs in batch_sizes:
            try:
                result = self.benchmark(
                    input_shape=(bs, input_channels, input_size, input_size)
                )
                results.append(result)
                logger.info(f"Batch {bs}: {result.avg_latency_ms:.2f}ms, "
                          f"{result.throughput_fps:.1f} FPS")
            except RuntimeError as e:
                logger.warning(f"Batch size {bs} failed: {e}")
                break
                
        return results


# ==================== Fallback System ====================

@dataclass
class FallbackConfig:
    """Configuration for fallback behavior."""
    confidence_threshold: float = 0.5
    uncertainty_threshold: float = 0.3
    max_retries: int = 2
    use_ensemble: bool = True
    fallback_message: str = "Unable to make confident prediction"
    conservative_mode: bool = True


class PredictionFallbackSystem:
    """
    Handle uncertain predictions with fallback strategies.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: Optional[FallbackConfig] = None,
                 backup_model: Optional[nn.Module] = None):
        self.model = model
        self.config = config or FallbackConfig()
        self.backup_model = backup_model
        self.fallback_count = 0
        self.total_predictions = 0
        self.fallback_log: List[Dict] = []
        
    def predict_with_fallback(self,
                             image: torch.Tensor,
                             return_confidence: bool = True) -> Dict:
        """
        Make prediction with fallback handling.
        
        Returns dict with:
        - prediction: class index or None
        - confidence: prediction confidence
        - used_fallback: whether fallback was triggered
        - fallback_reason: reason for fallback if any
        """
        self.total_predictions += 1
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(image)
            
        # Extract logits
        if isinstance(outputs, dict):
            logits = outputs.get('logits') or outputs.get('classifications')
        else:
            logits = outputs
            
        if logits is None:
            return self._trigger_fallback("No logits produced", image)
            
        # Compute confidence
        probs = F.softmax(logits, dim=-1)
        max_prob, pred = probs.max(dim=-1)
        confidence = max_prob.item()
        
        # Check entropy (uncertainty)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).item()
        normalized_entropy = entropy / np.log(probs.size(-1))
        
        # Decide on fallback
        if confidence < self.config.confidence_threshold:
            return self._trigger_fallback(
                f"Low confidence: {confidence:.3f}", 
                image, 
                initial_pred=pred.item(),
                initial_conf=confidence
            )
            
        if normalized_entropy > self.config.uncertainty_threshold:
            return self._trigger_fallback(
                f"High uncertainty: {normalized_entropy:.3f}",
                image,
                initial_pred=pred.item(),
                initial_conf=confidence
            )
            
        # Confident prediction
        return {
            'prediction': pred.item(),
            'confidence': confidence,
            'entropy': normalized_entropy,
            'used_fallback': False,
            'fallback_reason': None,
            'probabilities': probs.squeeze().cpu().numpy()
        }
        
    def _trigger_fallback(self,
                         reason: str,
                         image: torch.Tensor,
                         initial_pred: Optional[int] = None,
                         initial_conf: float = 0.0) -> Dict:
        """Execute fallback strategy."""
        self.fallback_count += 1
        
        result = {
            'prediction': None,
            'confidence': initial_conf,
            'used_fallback': True,
            'fallback_reason': reason,
            'initial_prediction': initial_pred
        }
        
        # Try backup model
        if self.backup_model is not None and self.config.use_ensemble:
            with torch.no_grad():
                backup_outputs = self.backup_model(image)
                
            if isinstance(backup_outputs, dict):
                backup_logits = backup_outputs.get('logits')
            else:
                backup_logits = backup_outputs
                
            if backup_logits is not None:
                backup_probs = F.softmax(backup_logits, dim=-1)
                backup_conf, backup_pred = backup_probs.max(dim=-1)
                
                if backup_conf.item() >= self.config.confidence_threshold:
                    result['prediction'] = backup_pred.item()
                    result['confidence'] = backup_conf.item()
                    result['fallback_source'] = 'backup_model'
                    return result
                    
        # Conservative mode: return most common/safe prediction
        if self.config.conservative_mode and initial_pred is not None:
            result['prediction'] = initial_pred
            result['is_conservative'] = True
            
        # Log fallback
        self.fallback_log.append({
            'reason': reason,
            'initial_pred': initial_pred,
            'initial_conf': initial_conf,
            'resolution': 'conservative' if result['prediction'] is not None else 'no_prediction'
        })
        
        return result
    
    def get_fallback_stats(self) -> Dict:
        """Get statistics on fallback usage."""
        if self.total_predictions == 0:
            return {'fallback_rate': 0, 'total': 0}
            
        reasons = defaultdict(int)
        for log in self.fallback_log:
            # Extract reason category
            if 'Low confidence' in log['reason']:
                reasons['low_confidence'] += 1
            elif 'High uncertainty' in log['reason']:
                reasons['high_uncertainty'] += 1
            else:
                reasons['other'] += 1
                
        return {
            'fallback_rate': self.fallback_count / self.total_predictions,
            'total_predictions': self.total_predictions,
            'fallback_count': self.fallback_count,
            'reasons': dict(reasons)
        }
    
    def reset_stats(self):
        """Reset fallback statistics."""
        self.fallback_count = 0
        self.total_predictions = 0
        self.fallback_log.clear()


def generate_stress_test_report(
    stress_results: Dict,
    benchmark_results: List[InferenceBenchmark],
    fallback_stats: Dict,
    output_path: Path
):
    """Generate comprehensive stress test report."""
    report = {
        'stress_test': stress_results,
        'benchmarks': [b.to_dict() for b in benchmark_results],
        'fallback': fallback_stats,
        'summary': {
            'robustness_score': stress_results.get('robustness_score', 0),
            'pass_rate': stress_results.get('pass_rate', 0),
            'avg_latency_ms': np.mean([b.avg_latency_ms for b in benchmark_results]) if benchmark_results else 0,
            'peak_throughput_fps': max([b.throughput_fps for b in benchmark_results]) if benchmark_results else 0,
            'fallback_rate': fallback_stats.get('fallback_rate', 0)
        },
        'recommendations': []
    }
    
    # Generate recommendations
    if stress_results.get('robustness_score', 0) < 70:
        report['recommendations'].append(
            "Consider adding more augmentation for edge cases"
        )
    if fallback_stats.get('fallback_rate', 0) > 0.1:
        report['recommendations'].append(
            "High fallback rate - model may need more training data"
        )
    if benchmark_results and benchmark_results[0].avg_latency_ms > 100:
        report['recommendations'].append(
            "Consider model optimization for real-time performance"
        )
        
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"Stress test report saved to {output_path}")
    return report

