"""Inference Engine - State Machine + Circuit Breaker."""

import torch
import time
import logging
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from ml.models.maxsight_cnn import create_model
from ml.utils.preprocessing import ImagePreprocessor
from ml.utils.error_handling import HeadExecutionManager, with_fallback
from ml.utils.output_scheduler import OutputMode, Severity, create_patient_output, create_clinician_output

logger = logging.getLogger(__name__)


class InferenceState(Enum):
    """Inference engine states."""
    INIT = "init"
    WARMUP = "warmup"
    STABLE = "stable"
    DEGRADED = "degraded"
    HALTED = "halted"


@dataclass
class InferenceMetrics:
    """Tracks inference performance metrics."""
    total_inferences: int = 0
    successful_inferences: int = 0
    failed_inferences: int = 0
    fallbacks_used: int = 0
    
    latencies_ms: List[float] = field(default_factory=list)
    uncertainties: List[float] = field(default_factory=list)
    
    def add_inference(self, latency_ms: float, uncertainty: float, success: bool, fallback_used: bool):
        """Record inference metrics."""
        self.total_inferences += 1
        if success:
            self.successful_inferences += 1
        else:
            self.failed_inferences += 1
        if fallback_used:
            self.fallbacks_used += 1
        
        self.latencies_ms.append(latency_ms)
        self.uncertainties.append(uncertainty)
        
        # Keep only recent history (last 100)
        if len(self.latencies_ms) > 100:
            self.latencies_ms = self.latencies_ms[-100:]
        if len(self.uncertainties) > 100:
            self.uncertainties = self.uncertainties[-100:]
    
    def get_p95_latency(self) -> float:
        """Get 95th percentile latency."""
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]
    
    def get_avg_uncertainty(self) -> float:
        """Get average uncertainty."""
        if not self.uncertainties:
            return 0.0
        return sum(self.uncertainties) / len(self.uncertainties)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker triggers."""
    # Latency thresholds.
    p95_latency_threshold_ms: float = 200.0
    max_latency_threshold_ms: float = 500.0
    
    # Uncertainty thresholds.
    avg_uncertainty_threshold: float = 0.7
    max_uncertainty_threshold: float = 0.9
    
    # Failure thresholds.
    fallback_rate_threshold: float = 0.3  # 30% fallback rate triggers degradation.
    failure_rate_threshold: float = 0.1   # 10% failure rate triggers halt.
    
    # Warmup settings.
    warmup_frames: int = 5  # Number of frames before stable.
    stabilization_window: int = 3  # Frames to stabilize before alerts.


class ThermalThrottleDetector:
    """Detect sustained latency degradation (e.g. thermal throttling). Uses a sliding window: if current avg latency > baseline * 2.0, return True."""

    def __init__(self, window_size_seconds: float = 30.0):
        self.window_size = window_size_seconds
        self.latency_history: List[Tuple[float, float]] = []  # (timestamp, latency_ms)
        self.baseline_latency: Optional[float] = None

    def check_thermal_throttle(self, current_latency: float) -> bool:
        now = time.time()
        self.latency_history.append((now, current_latency))
        # Prune old entries.
        cutoff = now - self.window_size
        self.latency_history = [(t, L) for t, L in self.latency_history if t >= cutoff]
        if len(self.latency_history) < 10:
            return False
        # Baseline = avg of first 5.
        if self.baseline_latency is None:
            self.baseline_latency = sum(L for _, L in self.latency_history[:5]) / 5.0
        # Current = avg of last 10.
        recent = self.latency_history[-10:]
        current_avg = sum(L for _, L in recent) / len(recent)
        if self.baseline_latency <= 0:
            return False
        if current_avg > self.baseline_latency * 2.0:
            logger.warning(
                "Thermal throttling detected: current_avg=%.1fms baseline=%.1fms",
                current_avg,
                self.baseline_latency,
            )
            return True
        return False


class InferenceEngine:
    """Spine of MaxSight inference with state machine and circuit breaker."""
    
    def __init__(
        self,
        device: Optional[str] = None,
        condition_mode: Optional[str] = None,
        output_mode: OutputMode = OutputMode.PATIENT,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        checkpoint_path: Optional[str] = None,
    ):
        """Initialize inference engine."""
        self.output_mode = output_mode
        self.condition_mode = condition_mode
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        
        # State.
        self.state = InferenceState.INIT
        self.metrics = InferenceMetrics()
        
        # Device setup.
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"InferenceEngine initialized on {self.device}")
        
        # Model (lazy init)
        self.model = None
        self.preprocessor = None
        self.head_manager = HeadExecutionManager(enable_fallbacks=True)
        
        # Warmup/stabilization tracking.
        self.warmup_count = 0
        self.in_stabilization = True
        self.thermal_detector = ThermalThrottleDetector(window_size_seconds=30.0)
        self.checkpoint_path = checkpoint_path

    def initialize(self):
        """Initialize model and preprocessor."""
        if self.state != InferenceState.INIT:
            logger.warning(f"Cannot initialize from state {self.state}")
            return

        logger.info("Initializing model and preprocessor...")

        # Load model.
        self.model = create_model(condition_mode=self.condition_mode)
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            self.model.load_state_dict(state, strict=False)
            logger.info("Loaded checkpoint: %s", self.checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load preprocessor.
        self.preprocessor = ImagePreprocessor(condition_mode=self.condition_mode)
        
        # Warmup.
        self._warmup()
        
        self.state = InferenceState.WARMUP
        logger.info("Inference engine ready")
    
    def _warmup(self):
        """Warmup model with dummy inputs."""
        logger.info("Warming up model...")
        if self.model is None:
            return
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            for _ in range(3):
                _ = self.model(dummy_input)
        logger.info("Warmup complete")
    
    def infer(
        self,
        image: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run inference with state machine and circuit breaker."""
        if self.state == InferenceState.HALTED:
            logger.error("Inference engine is halted")
            return self._get_safe_fallback(), {'halted': True}
        
        if self.model is None:
            self.initialize()
        assert self.model is not None  # Narrow type after initialize()
        
        # Run inference with timing.
        device = next(self.model.parameters()).device
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        
        start_time = time.perf_counter()
        
        try:
            with torch.no_grad():
                if audio_features is not None:
                    outputs = self.model(image, audio_features)
                else:
                    outputs = self.model(image)
            
            # Synchronize GPU after inference to ensure completion.
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Compute uncertainty.
            uncertainty = self._compute_uncertainty(outputs)
            
            # Record metrics.
            self.metrics.add_inference(
                latency_ms=latency_ms,
                uncertainty=uncertainty,
                success=True,
                fallback_used=False
            )
            
            # Thermal throttling: sustained degradation -> DEGRADED.
            if self.thermal_detector.check_thermal_throttle(latency_ms):
                if self.state == InferenceState.STABLE:
                    logger.warning("Thermal throttling detected, transitioning to DEGRADED")
                    self.state = InferenceState.DEGRADED
            
            # Check state transitions.
            self._check_state_transition()
            
            # Check circuit breaker.
            self._check_circuit_breaker()
            
            metadata = {
                'latency_ms': latency_ms,
                'uncertainty': uncertainty,
                'state': self.state.value,
                'in_stabilization': self.in_stabilization
            }
            
            return outputs, metadata
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            # Synchronize GPU even on error to get accurate timing.
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            self.metrics.add_inference(
                latency_ms=latency_ms,
                uncertainty=1.0,
                success=False,
                fallback_used=True
            )
            
            self._check_circuit_breaker()
            
            return self._get_safe_fallback(), {'error': str(e), 'fallback_used': True}
    
    def _compute_uncertainty(self, outputs: Dict[str, Any]) -> float:
        """Compute uncertainty from model outputs."""
        uncertainty_tensor = outputs.get('uncertainty', torch.tensor(0.5))
        if isinstance(uncertainty_tensor, torch.Tensor):
            return uncertainty_tensor.mean().item()
        return float(uncertainty_tensor)
    
    def _check_state_transition(self):
        """Check and perform state transitions."""
        if self.state == InferenceState.WARMUP:
            self.warmup_count += 1
            if self.warmup_count >= self.circuit_breaker_config.warmup_frames:
                self.state = InferenceState.STABLE
                logger.info("Transitioned to STABLE state")
        
        if self.state == InferenceState.STABLE:
            # Check if stabilization window complete.
            if self.warmup_count >= self.circuit_breaker_config.stabilization_window:
                self.in_stabilization = False
    
    def _check_circuit_breaker(self):
        """Check circuit breaker triggers and degrade/halt if needed."""
        if self.metrics.total_inferences < 5:
            return  # Need some history first.
        
        # Check failure rate.
        failure_rate = self.metrics.failed_inferences / self.metrics.total_inferences
        if failure_rate > self.circuit_breaker_config.failure_rate_threshold:
            logger.error(f"Failure rate {failure_rate:.2%} exceeds threshold, HALTING")
            self.state = InferenceState.HALTED
            return
        
        # Check fallback rate.
        fallback_rate = self.metrics.fallbacks_used / self.metrics.total_inferences
        if fallback_rate > self.circuit_breaker_config.fallback_rate_threshold:
            if self.state == InferenceState.STABLE:
                logger.warning(f"Fallback rate {fallback_rate:.2%} exceeds threshold, DEGRADING")
                self.state = InferenceState.DEGRADED
        
        # Check latency.
        p95_latency = self.metrics.get_p95_latency()
        if p95_latency > self.circuit_breaker_config.p95_latency_threshold_ms:
            if self.state == InferenceState.STABLE:
                logger.warning(f"P95 latency {p95_latency:.1f}ms exceeds threshold, DEGRADING")
                self.state = InferenceState.DEGRADED
        
        # Check uncertainty.
        avg_uncertainty = self.metrics.get_avg_uncertainty()
        if avg_uncertainty > self.circuit_breaker_config.avg_uncertainty_threshold:
            if self.state == InferenceState.STABLE:
                logger.warning(f"Avg uncertainty {avg_uncertainty:.2f} exceeds threshold, DEGRADING")
                self.state = InferenceState.DEGRADED
    
    def _get_safe_fallback(self) -> Dict[str, Any]:
        """Get safe fallback outputs."""
        return {
            'classifications': torch.zeros(1, 196, 80),
            'boxes': torch.zeros(1, 196, 4),
            'objectness': torch.zeros(1, 196),
            'urgency_scores': torch.zeros(1, 4),
            'distance_zones': torch.zeros(1, 196, 3),
            'scene_embedding': torch.zeros(1, 256),
            'uncertainty': torch.ones(1, 196)
        }
    
    def get_state(self) -> InferenceState:
        """Get current state."""
        return self.state
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            'state': self.state.value,
            'total_inferences': self.metrics.total_inferences,
            'successful': self.metrics.successful_inferences,
            'failed': self.metrics.failed_inferences,
            'fallbacks_used': self.metrics.fallbacks_used,
            'p95_latency_ms': self.metrics.get_p95_latency(),
            'avg_uncertainty': self.metrics.get_avg_uncertainty()
        }
    
    def reset(self):
        """Reset engine to INIT state."""
        logger.info("Resetting inference engine")
        self.state = InferenceState.INIT
        self.metrics = InferenceMetrics()
        self.warmup_count = 0
        self.in_stabilization = True






