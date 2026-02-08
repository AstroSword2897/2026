"""Continuous Monitoring and Readiness Dashboard."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque, defaultdict
import json
import logging
import threading
from enum import Enum

logger = logging.getLogger(__name__)



class ReadinessStatus(Enum):
    """Overall readiness status."""
    READY = "ready"
    DEGRADED = "degraded"
    NOT_READY = "not_ready"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"



@dataclass
class PredictionLog:
    """Single prediction log entry."""
    timestamp: datetime
    prediction: int
    confidence: float
    ground_truth: Optional[int] = None
    latency_ms: float = 0.0
    scenario: Optional[str] = None
    impairment: Optional[str] = None
    used_fallback: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'prediction': self.prediction,
            'confidence': self.confidence,
            'ground_truth': self.ground_truth,
            'latency_ms': self.latency_ms,
            'scenario': self.scenario,
            'impairment': self.impairment,
            'used_fallback': self.used_fallback,
            'correct': self.prediction == self.ground_truth if self.ground_truth is not None else None
        }


@dataclass
class Alert:
    """System alert."""
    timestamp: datetime
    severity: AlertSeverity
    category: str
    message: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.value,
            'category': self.category,
            'message': self.message,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold
        }


@dataclass
class ChecklistItem:
    """Single checklist item for readiness."""
    name: str
    category: str
    status: str  # 'pass', 'fail', 'warning', 'pending'
    description: str
    value: Optional[Any] = None
    threshold: Optional[Any] = None
    recommendation: Optional[str] = None



class PredictionMonitor:
    """Real-time prediction monitoring and logging."""
    
    def __init__(self, 
                 window_size: int = 1000,
                 drift_threshold: float = 0.1,
                 latency_threshold_ms: float = 100,
                 confidence_threshold: float = 0.5):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.latency_threshold_ms = latency_threshold_ms
        self.confidence_threshold = confidence_threshold
        
        # Rolling windows.
        self.predictions = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
        self.latencies = deque(maxlen=window_size)
        self.correct = deque(maxlen=window_size)
        
        # Historical baselines.
        self.baseline_accuracy = 0.0
        self.baseline_confidence = 0.0
        self.baseline_latency = 0.0
        
        # Per-class tracking.
        self.class_counts = defaultdict(int)
        self.class_correct = defaultdict(int)
        
        # Alerts.
        self.alerts: List[Alert] = []
        self._lock = threading.Lock()
        
    def log_prediction(self, log: PredictionLog):
        """Log a prediction and check for anomalies."""
        with self._lock:
            self.predictions.append(log)
            self.confidences.append(log.confidence)
            self.latencies.append(log.latency_ms)
            
            if log.ground_truth is not None:
                is_correct = log.prediction == log.ground_truth
                self.correct.append(is_correct)
                
                self.class_counts[log.prediction] += 1
                if is_correct:
                    self.class_correct[log.prediction] += 1
                    
            # Check for anomalies.
            self._check_alerts(log)
            
    def _check_alerts(self, log: PredictionLog):
        """Check for alert conditions."""
        # Low confidence alert.
        if log.confidence < self.confidence_threshold:
            self._add_alert(
                AlertSeverity.WARNING,
                "confidence",
                f"Low confidence prediction: {log.confidence:.3f}",
                "confidence",
                log.confidence,
                self.confidence_threshold
            )
            
        # High latency alert.
        if log.latency_ms > self.latency_threshold_ms:
            self._add_alert(
                AlertSeverity.WARNING,
                "latency",
                f"High latency: {log.latency_ms:.1f}ms",
                "latency_ms",
                log.latency_ms,
                self.latency_threshold_ms
            )
            
        # Accuracy drift detection.
        if len(self.correct) >= 100:
            current_accuracy = sum(self.correct) / len(self.correct)
            if self.baseline_accuracy > 0:
                drift = self.baseline_accuracy - current_accuracy
                if drift > self.drift_threshold:
                    self._add_alert(
                        AlertSeverity.CRITICAL,
                        "drift",
                        f"Accuracy drift detected: {drift:.3f}",
                        "accuracy_drift",
                        drift,
                        self.drift_threshold
                    )
                    
    def _add_alert(self, severity: AlertSeverity, category: str, 
                   message: str, metric_name: Optional[str] = None,
                   metric_value: Optional[float] = None, threshold: Optional[float] = None):
        """Add alert with deduplication."""
        # Check for recent duplicate.
        cutoff = datetime.now() - timedelta(minutes=5)
        recent_alerts = [a for a in self.alerts 
                        if a.timestamp > cutoff and a.category == category]
        
        if len(recent_alerts) < 3:  # Max 3 alerts per category per 5 minutes.
            self.alerts.append(Alert(
                timestamp=datetime.now(),
                severity=severity,
                category=category,
                message=message,
                metric_name=metric_name,
                metric_value=metric_value,
                threshold=threshold
            ))
            
    def set_baseline(self, accuracy: float, confidence: float, latency: float):
        """Set baseline metrics for drift detection."""
        self.baseline_accuracy = accuracy
        self.baseline_confidence = confidence
        self.baseline_latency = latency
        
    def get_current_metrics(self) -> Dict:
        """Get current performance metrics."""
        with self._lock:
            if not self.predictions:
                return {}
                
            return {
                'total_predictions': len(self.predictions),
                'accuracy': sum(self.correct) / len(self.correct) if self.correct else 0,
                'avg_confidence': np.mean(list(self.confidences)) if self.confidences else 0,
                'avg_latency_ms': np.mean(list(self.latencies)) if self.latencies else 0,
                'p95_latency_ms': np.percentile(list(self.latencies), 95) if self.latencies else 0,
                'fallback_rate': sum(1 for p in self.predictions if p.used_fallback) / len(self.predictions),
                'active_alerts': len([a for a in self.alerts if a.timestamp > datetime.now() - timedelta(hours=1)])
            }
            
    def get_per_class_metrics(self) -> Dict[int, Dict]:
        """Get per-class performance metrics."""
        metrics = {}
        for cls in self.class_counts:
            count = self.class_counts[cls]
            correct = self.class_correct[cls]
            metrics[cls] = {
                'count': count,
                'correct': correct,
                'accuracy': correct / count if count > 0 else 0
            }
        return metrics
        
    def get_alerts(self, 
                  severity: Optional[AlertSeverity] = None,
                  hours: int = 24) -> List[Alert]:
        """Get recent alerts."""
        cutoff = datetime.now() - timedelta(hours=hours)
        alerts = [a for a in self.alerts if a.timestamp > cutoff]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
            
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)


class ReadinessMonitor(PredictionMonitor):
    """Alias used by MaxSightCNN for performance monitoring. Accepts alert_threshold dict for compatibility with the model constructor."""
    def __init__(
        self,
        window_size: int = 100,
        alert_threshold: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ):
        if alert_threshold is None:
            alert_threshold = {}
        super().__init__(
            window_size=window_size,
            confidence_threshold=alert_threshold.get('confidence', 0.5),
            drift_threshold=alert_threshold.get('drift', 0.1),
            **kwargs,
        )



class ReadinessDashboard:
    """Real-World Readiness Assessment Dashboard. Provides comprehensive status of deployment readiness."""
    
    def __init__(self):
        self.checklist: List[ChecklistItem] = []
        self.last_assessment: Optional[datetime] = None
        
    def assess_readiness(self,
                        model: torch.nn.Module,
                        dataset_stats: Dict,
                        training_metrics: Dict,
                        stress_test_results: Dict,
                        benchmark_results: Dict,
                        monitoring_metrics: Dict) -> Dict:
        """Perform comprehensive readiness assessment."""
        self.checklist.clear()
        self.last_assessment = datetime.now()
        
        # 1. Dataset Readiness.
        self._assess_dataset(dataset_stats)
        
        # 2. Model Performance.
        self._assess_performance(training_metrics)
        
        # 3. Architecture & Training.
        self._assess_architecture(model, training_metrics)
        
        # 4. Stress Testing.
        self._assess_robustness(stress_test_results)
        
        # 5. Class Balance.
        self._assess_class_balance(dataset_stats)
        
        # 6. Deployment Readiness.
        self._assess_deployment(benchmark_results)
        
        # 7. Monitoring.
        self._assess_monitoring(monitoring_metrics)
        
        # Calculate overall status.
        status = self._calculate_overall_status()
        
        return {
            'status': status.value,
            'timestamp': self.last_assessment.isoformat(),
            'checklist': [self._item_to_dict(item) for item in self.checklist],
            'summary': self._generate_summary(),
            'recommendations': self._generate_recommendations()
        }
        
    def _assess_dataset(self, stats: Dict):
        """Assess dataset readiness."""
        # Size.
        train_size = stats.get('train_size', 0)
        self.checklist.append(ChecklistItem(
            name="Dataset Size",
            category="Dataset",
            status='pass' if train_size >= 1000 else 'warning' if train_size >= 500 else 'fail',
            description=f"{train_size} training samples",
            value=train_size,
            threshold=1000,
            recommendation="Increase to 5000+ for production" if train_size < 5000 else None
        ))
        
        # Diversity - scenarios.
        num_scenarios = len(stats.get('scenarios', []))
        self.checklist.append(ChecklistItem(
            name="Scenario Diversity",
            category="Dataset",
            status='pass' if num_scenarios >= 8 else 'warning' if num_scenarios >= 5 else 'fail',
            description=f"{num_scenarios} scenarios covered",
            value=num_scenarios,
            threshold=8
        ))
        
        # Impairments.
        num_impairments = len(stats.get('impairments', []))
        self.checklist.append(ChecklistItem(
            name="Impairment Coverage",
            category="Dataset",
            status='pass' if num_impairments >= 10 else 'warning' if num_impairments >= 5 else 'fail',
            description=f"{num_impairments} visual impairments simulated",
            value=num_impairments,
            threshold=10
        ))
        
        # Lighting conditions.
        num_lighting = len(stats.get('lighting_conditions', []))
        self.checklist.append(ChecklistItem(
            name="Lighting Conditions",
            category="Dataset",
            status='pass' if num_lighting >= 6 else 'warning',
            description=f"{num_lighting} lighting conditions",
            value=num_lighting,
            threshold=6
        ))
        
    def _assess_performance(self, metrics: Dict):
        """Assess model performance metrics."""
        # Accuracy/mAP.
        accuracy = metrics.get('accuracy', 0) or metrics.get('map', 0)
        self.checklist.append(ChecklistItem(
            name="Model Accuracy",
            category="Performance",
            status='pass' if accuracy >= 0.7 else 'warning' if accuracy >= 0.5 else 'fail',
            description=f"{accuracy:.1%} accuracy/mAP",
            value=accuracy,
            threshold=0.7
        ))
        
        # Precision.
        precision = metrics.get('precision', 0)
        if precision > 0:
            self.checklist.append(ChecklistItem(
                name="Precision",
                category="Performance",
                status='pass' if precision >= 0.7 else 'warning' if precision >= 0.5 else 'fail',
                description=f"{precision:.1%}",
                value=precision,
                threshold=0.7
            ))
            
        # Recall.
        recall = metrics.get('recall', 0)
        if recall > 0:
            self.checklist.append(ChecklistItem(
                name="Recall",
                category="Performance",
                status='pass' if recall >= 0.7 else 'warning' if recall >= 0.5 else 'fail',
                description=f"{recall:.1%}",
                value=recall,
                threshold=0.7
            ))
            
        # Loss convergence.
        final_loss = metrics.get('final_loss', float('inf'))
        self.checklist.append(ChecklistItem(
            name="Loss Convergence",
            category="Performance",
            status='pass' if final_loss < 5 else 'warning' if final_loss < 10 else 'fail',
            description=f"Final loss: {final_loss:.3f}",
            value=final_loss,
            threshold=5
        ))
        
    def _assess_architecture(self, model: torch.nn.Module, metrics: Dict):
        """Assess architecture and training setup."""
        # Parameter count.
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        self.checklist.append(ChecklistItem(
            name="Model Capacity",
            category="Architecture",
            status='pass',
            description=f"{num_params:.1f}M parameters",
            value=num_params
        ))
        
        # Regularization.
        has_dropout = any('dropout' in name.lower() for name, _ in model.named_modules())
        self.checklist.append(ChecklistItem(
            name="Dropout Regularization",
            category="Architecture",
            status='pass' if has_dropout else 'warning',
            description="Present" if has_dropout else "Not detected",
            recommendation="Add dropout for better generalization" if not has_dropout else None
        ))
        
        # Weight decay.
        weight_decay = metrics.get('weight_decay', 0)
        self.checklist.append(ChecklistItem(
            name="Weight Decay",
            category="Architecture",
            status='pass' if weight_decay > 0 else 'warning',
            description=f"{weight_decay:.1e}" if weight_decay > 0 else "Not applied",
            value=weight_decay,
            recommendation="Add weight decay to prevent overfitting" if weight_decay == 0 else None
        ))
        
    def _assess_robustness(self, results: Dict):
        """Assess stress test robustness."""
        robustness_score = results.get('robustness_score', 0)
        self.checklist.append(ChecklistItem(
            name="Robustness Score",
            category="Robustness",
            status='pass' if robustness_score >= 70 else 'warning' if robustness_score >= 50 else 'fail',
            description=f"{robustness_score:.0f}/100",
            value=robustness_score,
            threshold=70
        ))
        
        pass_rate = results.get('pass_rate', 0)
        self.checklist.append(ChecklistItem(
            name="Stress Test Pass Rate",
            category="Robustness",
            status='pass' if pass_rate >= 0.8 else 'warning' if pass_rate >= 0.6 else 'fail',
            description=f"{pass_rate:.0%} scenarios passed",
            value=pass_rate,
            threshold=0.8
        ))
        
        critical_failures = results.get('critical_failures', 0)
        self.checklist.append(ChecklistItem(
            name="Critical Failures",
            category="Robustness",
            status='pass' if critical_failures == 0 else 'fail',
            description=f"{critical_failures} critical failures",
            value=critical_failures,
            threshold=0
        ))
        
    def _assess_class_balance(self, stats: Dict):
        """Assess class balance."""
        urgency_dist = stats.get('urgency_distribution', {})
        
        # Check high-urgency representation.
        total = sum(urgency_dist.values()) if urgency_dist else 1
        high_urgency = urgency_dist.get('2', 0) + urgency_dist.get('3', 0)
        high_urgency_ratio = high_urgency / total if total > 0 else 0
        
        self.checklist.append(ChecklistItem(
            name="High-Urgency Balance",
            category="Class Balance",
            status='pass' if high_urgency_ratio >= 0.1 else 'warning',
            description=f"{high_urgency_ratio:.1%} high-urgency samples",
            value=high_urgency_ratio,
            recommendation="Oversample high-urgency cases" if high_urgency_ratio < 0.1 else None
        ))
        
    def _assess_deployment(self, benchmark: Dict):
        """Assess deployment readiness."""
        avg_latency = benchmark.get('avg_latency_ms', 0)
        self.checklist.append(ChecklistItem(
            name="Inference Latency",
            category="Deployment",
            status='pass' if avg_latency < 50 else 'warning' if avg_latency < 100 else 'fail',
            description=f"{avg_latency:.1f}ms average",
            value=avg_latency,
            threshold=50,
            recommendation="Optimize model for faster inference" if avg_latency > 100 else None
        ))
        
        throughput = benchmark.get('throughput_fps', 0)
        self.checklist.append(ChecklistItem(
            name="Throughput",
            category="Deployment",
            status='pass' if throughput >= 30 else 'warning' if throughput >= 15 else 'fail',
            description=f"{throughput:.1f} FPS",
            value=throughput,
            threshold=30
        ))
        
        memory = benchmark.get('memory_mb', 0)
        self.checklist.append(ChecklistItem(
            name="Memory Usage",
            category="Deployment",
            status='pass' if memory < 500 else 'warning' if memory < 1000 else 'fail',
            description=f"{memory:.0f} MB",
            value=memory,
            threshold=500
        ))
        
    def _assess_monitoring(self, metrics: Dict):
        """Assess monitoring setup."""
        has_monitoring = metrics.get('monitoring_enabled', False)
        self.checklist.append(ChecklistItem(
            name="Monitoring Pipeline",
            category="Monitoring",
            status='pass' if has_monitoring else 'warning',
            description="Enabled" if has_monitoring else "Not configured"
        ))
        
        fallback_rate = metrics.get('fallback_rate', 0)
        self.checklist.append(ChecklistItem(
            name="Fallback Rate",
            category="Monitoring",
            status='pass' if fallback_rate < 0.1 else 'warning' if fallback_rate < 0.2 else 'fail',
            description=f"{fallback_rate:.1%}",
            value=fallback_rate,
            threshold=0.1
        ))
        
    def _calculate_overall_status(self) -> ReadinessStatus:
        """Calculate overall readiness status."""
        fails = sum(1 for item in self.checklist if item.status == 'fail')
        warnings = sum(1 for item in self.checklist if item.status == 'warning')
        
        if fails > 0:
            return ReadinessStatus.NOT_READY
        elif warnings > 3:
            return ReadinessStatus.DEGRADED
        else:
            return ReadinessStatus.READY
            
    def _generate_summary(self) -> Dict:
        """Generate summary statistics."""
        by_status = defaultdict(int)
        by_category = defaultdict(lambda: {'pass': 0, 'warning': 0, 'fail': 0})
        
        for item in self.checklist:
            by_status[item.status] += 1
            by_category[item.category][item.status] += 1
            
        return {
            'total_checks': len(self.checklist),
            'passed': by_status['pass'],
            'warnings': by_status['warning'],
            'failed': by_status['fail'],
            'by_category': dict(by_category)
        }
        
    def _generate_recommendations(self) -> List[str]:
        """Generate prioritized recommendations."""
        recommendations = []
        
        # Process critical items first.
        for item in self.checklist:
            if item.status == 'fail' and item.recommendation:
                recommendations.append(f"[CRITICAL] {item.recommendation}")
                
        # Then warnings.
        for item in self.checklist:
            if item.status == 'warning' and item.recommendation:
                recommendations.append(f"[WARNING] {item.recommendation}")
                
        return recommendations[:10]  # Top 10.
        
    def _item_to_dict(self, item: ChecklistItem) -> Dict:
        """Convert checklist item to dict."""
        return {
            'name': item.name,
            'category': item.category,
            'status': item.status,
            'description': item.description,
            'value': item.value,
            'threshold': item.threshold,
            'recommendation': item.recommendation
        }
        
    def export_dashboard(self, output_path: Path):
        """Export dashboard to file."""
        # Simplified export for current state.
        data = {
            'timestamp': self.last_assessment.isoformat() if self.last_assessment else None,
            'checklist': [self._item_to_dict(item) for item in self.checklist],
            'summary': self._generate_summary(),
            'recommendations': self._generate_recommendations()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Dashboard exported to {output_path}")


def create_monitoring_pipeline(model: torch.nn.Module) -> Dict:
    """Create complete monitoring pipeline."""
    return {
        'monitor': PredictionMonitor(),
        'dashboard': ReadinessDashboard()
    }



class HealthChecker:
    """Health check system for MaxSight Tier 1 heads and system reliability. Run daily to catch issues before they impact users."""
    
    # Thresholds for Tier 1 head alerts.
    TIER1_DETECTION_RATE_THRESHOLD = 0.85
    TIER1_MAP_THRESHOLD = 0.80
    TIER1_IOU_THRESHOLD = 0.70
    TIER1_FALSE_REASSURANCE_THRESHOLD = 0.01
    STAGE_A_LATENCY_THRESHOLD_MS = 150.0
    STAGE_B_LATENCY_THRESHOLD_MS = 500.0
    
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        """Initialize health checker."""
        self.device = device
        self.model = model
        self.model.eval()
    
    def check_tier1_heads(self) -> Dict[str, Any]:
        """Check Tier 1 (safety-critical) heads."""
        results = {
            'status': 'PASS',
            'checks': {},
            'failures': []
        }
        
        # Create dummy input.
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(dummy_input)
        
        # Check 1: Objectness head.
        if 'objectness' in outputs:
            obj_scores = outputs['objectness']
            detection_rate = (obj_scores > 0.5).float().mean().item()
            results['checks']['objectness'] = {
                'detection_rate': detection_rate,
                'threshold': self.TIER1_DETECTION_RATE_THRESHOLD,
                'status': 'PASS' if detection_rate >= self.TIER1_DETECTION_RATE_THRESHOLD else 'FAIL'
            }
            if detection_rate < self.TIER1_DETECTION_RATE_THRESHOLD:
                results['status'] = 'FAIL'
                results['failures'].append(f"Objectness detection rate {detection_rate:.2%} < {self.TIER1_DETECTION_RATE_THRESHOLD:.2%}")
        else:
            results['status'] = 'FAIL'
            results['failures'].append("Objectness head output missing")
        
        # Check 2: Classification head.
        if 'classifications' in outputs:
            cls_logits = outputs['classifications']
            cls_probs = torch.softmax(cls_logits, dim=-1)
            max_probs = cls_probs.max(dim=-1)[0]
            avg_confidence = max_probs.mean().item()
            results['checks']['classification'] = {
                'avg_confidence': avg_confidence,
                'threshold': self.TIER1_MAP_THRESHOLD,
                'status': 'PASS' if avg_confidence >= self.TIER1_MAP_THRESHOLD else 'FAIL'
            }
            if avg_confidence < self.TIER1_MAP_THRESHOLD:
                results['status'] = 'FAIL'
                results['failures'].append(f"Classification confidence {avg_confidence:.2%} < {self.TIER1_MAP_THRESHOLD:.2%}")
        else:
            results['status'] = 'FAIL'
            results['failures'].append("Classification head output missing")
        
        # Check 3: Box regression.
        if 'boxes' in outputs:
            boxes = outputs['boxes']
            valid_boxes = ((boxes >= 0) & (boxes <= 1)).all(dim=-1).float().mean().item()
            results['checks']['box_regression'] = {
                'valid_box_rate': valid_boxes,
                'threshold': self.TIER1_IOU_THRESHOLD,
                'status': 'PASS' if valid_boxes >= self.TIER1_IOU_THRESHOLD else 'FAIL'
            }
            if valid_boxes < self.TIER1_IOU_THRESHOLD:
                results['status'] = 'FAIL'
                results['failures'].append(f"Valid box rate {valid_boxes:.2%} < {self.TIER1_IOU_THRESHOLD:.2%}")
        else:
            results['status'] = 'FAIL'
            results['failures'].append("Box regression head output missing")
        
        # Check 4: Distance zones.
        if 'distance_zones' in outputs:
            distances = outputs['distance_zones']
            dist_probs = torch.softmax(distances, dim=-1)
            max_dist_probs = dist_probs.max(dim=-1)[0]
            avg_dist_confidence = max_dist_probs.mean().item()
            results['checks']['distance'] = {
                'avg_confidence': avg_dist_confidence,
                'status': 'PASS' if avg_dist_confidence > 0.5 else 'WARN'
            }
        else:
            results['status'] = 'FAIL'
            results['failures'].append("Distance zones output missing")
        
        # Check 5: Urgency head.
        if 'urgency_scores' in outputs:
            urgency = outputs['urgency_scores']
            urgency_probs = torch.softmax(urgency, dim=-1)
            safe_prob = urgency_probs[:, 0].mean().item()  # Assuming 0 = safe.
            results['checks']['urgency'] = {
                'safe_prob': safe_prob,
                'status': 'PASS' if safe_prob < 0.99 else 'WARN'  # Too confident = risky.
            }
        else:
            results['status'] = 'FAIL'
            results['failures'].append("Urgency head output missing")
        
        # Check 6: Uncertainty head.
        if 'uncertainty' in outputs:
            uncertainty = outputs['uncertainty']
            avg_uncertainty = uncertainty.mean().item()
            results['checks']['uncertainty'] = {
                'avg_uncertainty': avg_uncertainty,
                'status': 'PASS' if 0.1 < avg_uncertainty < 0.7 else 'WARN'
            }
        else:
            results['status'] = 'WARN'
            results['failures'].append("Uncertainty head output missing (non-critical)")
        
        return results
    
    def check_latency(self, num_runs: int = 10) -> Dict[str, Any]:
        """Check inference latency."""
        results: Dict[str, Any] = {
            'status': 'PASS',
            'latencies': [],
            'stage_a_ms': 0.0,
            'avg_ms': 0.0
        }
        
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        # Warmup.
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        # Measure latency.
        import time
        latencies = []
        
        for _ in range(num_runs):
            start = time.time()
            with torch.no_grad():
                outputs = self.model(dummy_input)
            latencies.append((time.time() - start) * 1000)  # Convert to ms.
        
        avg_latency = float(np.mean(latencies))
        min_latency = float(np.min(latencies))
        max_latency = float(np.max(latencies))
        
        results['latencies'] = latencies
        results['stage_a_ms'] = avg_latency
        results['avg_ms'] = avg_latency
        results['min_ms'] = min_latency
        results['max_ms'] = max_latency
        
        # Check thresholds.
        if avg_latency > self.STAGE_A_LATENCY_THRESHOLD_MS:
            results['status'] = 'FAIL'
            results['failures'] = [f"Average latency {avg_latency:.1f}ms > {self.STAGE_A_LATENCY_THRESHOLD_MS}ms"]
        elif avg_latency > self.STAGE_A_LATENCY_THRESHOLD_MS * 0.8:
            results['status'] = 'WARN'
            results['warnings'] = [f"Average latency {avg_latency:.1f}ms approaching threshold"]
        
        return results
    
    def check_model_integrity(self) -> Dict[str, Any]:
        """Check model integrity (no NaN, Inf, or corrupted weights)."""
        results: Dict[str, Any] = {
            'status': 'PASS',
            'checks': {},
            'failures': []
        }
        
        for name, param in self.model.named_parameters():
            # Check for NaN.
            if torch.isnan(param).any().item():
                results['status'] = 'FAIL'
                results['failures'].append(f"NaN detected in {name}")
            
            # Check for Inf.
            if torch.isinf(param).any().item():
                results['status'] = 'FAIL'
                results['failures'].append(f"Inf detected in {name}")
            
            # Check for extreme values.
            if param.abs().max() > 1e6:
                results['status'] = 'WARN'
                if 'warnings' not in results:
                    results['warnings'] = []
                results['warnings'].append(f"Extreme values in {name}: max={param.abs().max().item():.2e}")
        
        return results
    
    def run_full_check(self) -> Dict[str, Any]:
        """Run all health checks."""
        report: Dict[str, Any] = {
            'timestamp': datetime.now().isoformat(),
            'device': self.device,
            'overall_status': 'PASS',
            'checks': {}
        }
        
        # Tier 1 checks.
        logger.info("Checking Tier 1 heads...")
        tier1_results = self.check_tier1_heads()
        report['checks']['tier1'] = tier1_results
        if tier1_results['status'] == 'FAIL':
            report['overall_status'] = 'FAIL'
        
        # Latency checks.
        logger.info("Checking latency...")
        latency_results = self.check_latency()
        report['checks']['latency'] = latency_results
        if latency_results['status'] == 'FAIL':
            report['overall_status'] = 'FAIL'
        
        # Model integrity.
        logger.info("Checking model integrity...")
        integrity_results = self.check_model_integrity()
        report['checks']['integrity'] = integrity_results
        if integrity_results['status'] == 'FAIL':
            report['overall_status'] = 'FAIL'
        
        return report







