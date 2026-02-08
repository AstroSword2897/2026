"""Error Handling, Fallback Logic, Kill Switches, and Ethical Safeguards for MaxSight Handles error propagation, runtime head control, and safety mechanisms."""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, Callable, List, Set
from functools import wraps
from collections import defaultdict
import time


logger = logging.getLogger(__name__)


class MaxSightError(Exception):
    """Base exception for MaxSight errors."""
    pass


class HeadExecutionError(MaxSightError):
    """Error during head execution."""
    pass


class DependencyError(MaxSightError):
    """Error due to missing dependency."""
    pass


class TimeoutError(MaxSightError):
    """Error due to timeout."""
    pass


def with_fallback(
    fallback_value: Any = None,
    fallback_func: Optional[Callable] = None,
    log_error: bool = True
):
    """Decorator to add fallback logic to functions."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.warning(f"{func.__name__} failed: {e}, using fallback")
                
                if fallback_func:
                    try:
                        return fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        logger.error(f"Fallback for {func.__name__} also failed: {fallback_error}")
                        return fallback_value
                else:
                    return fallback_value
        return wrapper
    return decorator


def with_timeout(timeout_ms: float = 1000.0):
    """Decorator to add timeout to functions. Arguments: timeout_ms: Timeout in milliseconds."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            if elapsed_ms > timeout_ms:
                logger.warning(f"{func.__name__} took {elapsed_ms:.2f}ms (exceeded {timeout_ms}ms)")
            
            return result
        return wrapper
    return decorator


class HeadExecutionManager:
    """Manages head execution with error handling and fallbacks. Handles: - Dependency validation - Error propagation - Fallback execution - Timeout management."""
    
    def __init__(
        self,
        enable_fallbacks: bool = True,
        timeout_ms: float = 1000.0,
        uncertainty_threshold: float = 0.7
    ):
        self.enable_fallbacks = enable_fallbacks
        self.timeout_ms = timeout_ms
        self.uncertainty_threshold = uncertainty_threshold
        self.execution_log: List[Dict[str, Any]] = []
    
    def execute_head(
        self,
        head_name: str,
        head_func: Callable,
        inputs: Dict[str, Any],
        dependencies: List[str],
        fallback_func: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Execute a head with error handling and fallbacks."""
        # Validate dependencies.
        missing_deps = [dep for dep in dependencies if dep not in inputs]
        if missing_deps:
            if self.enable_fallbacks and fallback_func:
                logger.warning(f"{head_name} missing dependencies {missing_deps}, using fallback")
                try:
                    result = fallback_func(**inputs)
                    # Log fallback usage.
                    self.execution_log.append({
                        'head': head_name,
                        'success': False,
                        'error': f"Missing dependencies: {missing_deps}",
                        'fallback_used': True
                    })
                    return result
                except Exception as e:
                    logger.error(f"Fallback for {head_name} failed: {e}")
                    self.execution_log.append({
                        'head': head_name,
                        'success': False,
                        'error': f"Missing dependencies: {missing_deps}, fallback failed: {e}",
                        'fallback_used': False
                    })
                    return self._get_default_outputs(head_name)
            else:
                raise DependencyError(f"{head_name} missing dependencies: {missing_deps}")
        
        # Execute with timeout.
        start_time = time.perf_counter()
        try:
            result = head_func(**inputs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            # Check timeout.
            if elapsed_ms > self.timeout_ms:
                logger.warning(f"{head_name} exceeded timeout ({elapsed_ms:.2f}ms > {self.timeout_ms}ms)")
                if self.enable_fallbacks and fallback_func:
                    return fallback_func(**inputs)
            
            # Validate result.
            if result is None:
                raise HeadExecutionError(f"{head_name} returned None")
            
            # Check for NaN/Inf.
            if isinstance(result, torch.Tensor):
                if torch.isnan(result).any() or torch.isinf(result).any():
                    logger.warning(f"{head_name} produced NaN/Inf, using fallback")
                    if self.enable_fallbacks and fallback_func:
                        return fallback_func(**inputs)
                    return self._get_default_outputs(head_name)
            
            # Log execution.
            self.execution_log.append({
                'head': head_name,
                'success': True,
                'elapsed_ms': elapsed_ms,
                'dependencies': dependencies
            })
            
            return result
            
        except Exception as e:
            logger.error(f"{head_name} execution failed: {e}")
            
            # Try fallback.
            if self.enable_fallbacks and fallback_func:
                try:
                    fallback_result = fallback_func(**inputs)
                    self.execution_log.append({
                        'head': head_name,
                        'success': False,
                        'error': str(e),
                        'fallback_used': True
                    })
                    return fallback_result
                except Exception as fallback_error:
                    logger.error(f"Fallback for {head_name} also failed: {fallback_error}")
            
            # Return default outputs.
            self.execution_log.append({
                'head': head_name,
                'success': False,
                'error': str(e),
                'fallback_used': False
            })
            
            return self._get_default_outputs(head_name)
    
    def _get_default_outputs(self, head_name: str) -> Dict[str, Any]:
        """Get default outputs for a head when execution fails."""
        defaults = {
            'classification': torch.zeros(1, 196, 80),
            'box_regression': torch.zeros(1, 196, 4),
            'objectness': torch.zeros(1, 196),
            'text_region': torch.zeros(1, 196),
            'urgency': torch.zeros(1, 4),
            'distance': torch.zeros(1, 196, 3),
            'contrast': torch.zeros(1, 1),
            'glare': torch.zeros(1, 4),
            'findability': torch.zeros(1, 196),
            'navigation_difficulty': torch.zeros(1, 1),
            'uncertainty': torch.ones(1, 1),  # High uncertainty = low confidence.
        }
        
        if head_name in defaults:
            return {head_name: defaults[head_name]}
        return {}
    
    def check_uncertainty_fallback(
        self,
        uncertainty: torch.Tensor,
        outputs: Dict[str, Any]
    ) -> bool:
        """Check if uncertainty is high enough to trigger fallback."""
        if not self.enable_fallbacks:
            return False
        
        if uncertainty is None:
            return False
        
        uncertainty_value = uncertainty.mean().item()
        if uncertainty_value > self.uncertainty_threshold:
            logger.info(f"High uncertainty ({uncertainty_value:.3f}), using fallback")
            return True
        
        return False
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of head executions."""
        total = len(self.execution_log)
        successful = sum(1 for log in self.execution_log if log.get('success', False))
        failed = total - successful
        fallbacks_used = sum(1 for log in self.execution_log if log.get('fallback_used', False))
        
        avg_latency = 0.0
        if total > 0:
            latencies = [log.get('elapsed_ms', 0) for log in self.execution_log if 'elapsed_ms' in log]
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
        
        return {
            'total_executions': total,
            'successful': successful,
            'failed': failed,
            'fallbacks_used': fallbacks_used,
            'success_rate': successful / total if total > 0 else 0.0,
            'avg_latency_ms': avg_latency
        }


def safe_head_execution(
    head_name: str,
    head_func: Callable,
    inputs: Dict[str, Any],
    dependencies: List[str],
    manager: Optional[HeadExecutionManager] = None,
    fallback_func: Optional[Callable] = None
) -> Dict[str, Any]:
    """Safely execute a head with error handling."""
    if manager is None:
        manager = HeadExecutionManager()
    
    return manager.execute_head(
        head_name=head_name,
        head_func=head_func,
        inputs=inputs,
        dependencies=dependencies,
        fallback_func=fallback_func
    )


# Head Kill Switch System.

class HeadKillSwitchManager:
    """Runtime-configurable head disabling manager."""
    
    def __init__(
        self,
        enabled_heads: Optional[List[str]] = None,
        default_enabled: bool = True
    ):
        self.enabled_heads: Optional[Set[str]] = set(enabled_heads) if enabled_heads else None
        self.disabled_heads: Set[str] = set()
        self.default_enabled = default_enabled
        self.head_stats = defaultdict(lambda: {'enabled_count': 0, 'disabled_count': 0})
        self.fallback_outputs: Dict[str, Callable] = {}
    
    def is_enabled(self, head_name: str) -> bool:
        """Check if a head is enabled."""
        if head_name in self.disabled_heads:
            self.head_stats[head_name]['disabled_count'] += 1
            return False
        if self.enabled_heads is not None:
            enabled = head_name in self.enabled_heads
            if enabled:
                self.head_stats[head_name]['enabled_count'] += 1
            else:
                self.head_stats[head_name]['disabled_count'] += 1
            return enabled
        self.head_stats[head_name]['enabled_count'] += 1
        return True
    
    def disable_head(self, head_name: str):
        """Disable a head at runtime."""
        self.disabled_heads.add(head_name)
        logger.info(f"Head '{head_name}' disabled via kill switch")
    
    def enable_head(self, head_name: str):
        """Enable a head at runtime."""
        self.disabled_heads.discard(head_name)
        logger.info(f"Head '{head_name}' enabled via kill switch")
    
    def disable_heads_by_category(self, category: str):
        """Disable heads by category."""
        category_map = {
            'accessibility': ['contrast', 'glare', 'findability', 'navigation_difficulty'],
            'therapy': ['fatigue', 'personalization'],
            'advanced': ['motion', 'roi_priority', 'predictive_alert'],
            'optional': ['contrast', 'glare', 'findability', 'navigation_difficulty',
                        'fatigue', 'personalization', 'motion', 'roi_priority',
                        'predictive_alert', 'sound_event', 'ocr', 'scene_description'],
            'non_critical': ['contrast', 'glare', 'findability', 'navigation_difficulty',
                            'fatigue', 'personalization', 'motion', 'roi_priority',
                            'predictive_alert', 'sound_event', 'ocr', 'scene_description', 'uncertainty']
        }
        heads_to_disable = category_map.get(category, [])
        for head_name in heads_to_disable:
            self.disable_head(head_name)
    
    def register_fallback(self, head_name: str, fallback_func: Callable):
        """Register a fallback function for a disabled head."""
        self.fallback_outputs[head_name] = fallback_func
    
    def get_fallback_output(self, head_name: str, **kwargs) -> Dict[str, torch.Tensor]:
        """Get fallback output for a disabled head."""
        if head_name in self.fallback_outputs:
            return self.fallback_outputs[head_name](**kwargs)
        return self._get_default_output(head_name, **kwargs)
    
    def _get_default_output(self, head_name: str, **kwargs) -> Dict[str, torch.Tensor]:
        """Generate default output for a disabled head."""
        device = None
        dtype = torch.float32
        for value in kwargs.values():
            if torch.is_tensor(value):
                device = value.device
                dtype = value.dtype
                break
        if device is None:
            device = torch.device('cpu')
        
        # Infer shape from input if available.
        batch_size = 1
        for value in kwargs.values():
            if torch.is_tensor(value) and value.dim() > 0:
                batch_size = value.shape[0]
                break
        
        defaults = {
            'depth': {'depth_map': torch.zeros(batch_size, 1, 224, 224, device=device, dtype=dtype)},
            'contrast': {'contrast_map': torch.ones(batch_size, 1, 224, 224, device=device, dtype=dtype)},
            'motion': {'flow': torch.zeros(batch_size, 2, 224, 224, device=device, dtype=dtype)},
            'fatigue': {'fatigue_score': torch.tensor(0.0, device=device, dtype=dtype)},
            'uncertainty': {'uncertainty_score': torch.tensor(0.5, device=device, dtype=dtype)},
            'roi_priority': {'roi_scores': torch.zeros(batch_size, 10, device=device, dtype=dtype)},
        }
        return defaults.get(head_name, {})
    
    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics on head enable/disable counts."""
        return dict(self.head_stats)
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.head_stats.clear()


class KillSwitchWrapper(nn.Module):
    """Wrapper that adds kill switch functionality to a head module."""
    
    def __init__(
        self,
        head_module: nn.Module,
        head_name: str,
        kill_switch_manager: HeadKillSwitchManager
    ):
        super().__init__()
        self.head_module = head_module
        self.head_name = head_name
        self.kill_switch_manager = kill_switch_manager
    
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass with kill switch check."""
        if not self.kill_switch_manager.is_enabled(self.head_name):
            return self.kill_switch_manager.get_fallback_output(
                self.head_name, *args, **kwargs
            )
        return self.head_module(*args, **kwargs)


def wrap_heads_with_killswitch(
    model: nn.Module,
    kill_switch_manager: HeadKillSwitchManager,
    head_name_mapping: Optional[Dict[str, str]] = None
) -> nn.Module:
    """Wrap all heads in a model with kill switch functionality."""
    if head_name_mapping is None:
        head_name_mapping = {}
        for name, module in model.named_modules():
            if 'head' in name.lower() and not isinstance(module, KillSwitchWrapper):
                head_name_mapping[name] = name.split('.')[-1]
    
    for attr_name, head_name in head_name_mapping.items():
        parts = attr_name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        head_module = getattr(parent, parts[-1])
        if isinstance(head_module, nn.Module) and not isinstance(head_module, KillSwitchWrapper):
            wrapped_head = KillSwitchWrapper(head_module, head_name, kill_switch_manager)
            setattr(parent, parts[-1], wrapped_head)
            logger.info(f"Wrapped head '{head_name}' with kill switch")
    
    return model


# Ethical Safeguards.

class UncertaintySuppressor:
    """Ensures uncertainty suppresses potentially harmful actions."""
    
    def __init__(
        self,
        uncertainty_threshold: float = 0.7,
        suppression_mode: str = 'soft'  # 'soft', 'hard', 'graded'
    ):
        self.uncertainty_threshold = uncertainty_threshold
        self.suppression_mode = suppression_mode
    
    def suppress_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        uncertainty: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Suppress outputs based on uncertainty."""
        if uncertainty is None:
            uncertainty = outputs.get('uncertainty_score')
            if uncertainty is None:
                uncertainty = self._infer_uncertainty(outputs)
        
        if uncertainty is None:
            logger.warning("No uncertainty available, cannot suppress")
            return outputs
        
        if torch.is_tensor(uncertainty):
            if uncertainty.numel() > 1:
                uncertainty = uncertainty.mean()
            uncertainty_value = uncertainty.item()
        else:
            uncertainty_value = float(uncertainty)
        
        suppressed_outputs = outputs.copy()
        
        if uncertainty_value > self.uncertainty_threshold:
            if self.suppression_mode == 'hard':
                suppressed_outputs = self._hard_suppress(suppressed_outputs)
            elif self.suppression_mode == 'graded':
                suppressed_outputs = self._graded_suppress(suppressed_outputs, uncertainty_value)
            else:  # Soft.
                suppressed_outputs = self._soft_suppress(suppressed_outputs, uncertainty_value)
        
        return suppressed_outputs
    
    def _hard_suppress(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Hard suppression: zero out confidence."""
        suppressed = outputs.copy()
        if 'confidence' in suppressed:
            suppressed['confidence'] = suppressed['confidence'] * 0.0
        if 'scores' in suppressed:
            suppressed['scores'] = suppressed['scores'] * 0.0
        if 'urgency' in suppressed:
            if torch.is_tensor(suppressed['urgency']):
                suppressed['urgency'] = torch.zeros_like(suppressed['urgency'])
            else:
                suppressed['urgency'] = 0
        return suppressed
    
    def _soft_suppress(
        self,
        outputs: Dict[str, torch.Tensor],
        uncertainty_value: float
    ) -> Dict[str, torch.Tensor]:
        """Soft suppression: reduce confidence proportionally."""
        suppressed = outputs.copy()
        scale_factor = 1.0 - uncertainty_value
        if 'confidence' in suppressed:
            suppressed['confidence'] = suppressed['confidence'] * scale_factor
        if 'scores' in suppressed:
            suppressed['scores'] = suppressed['scores'] * scale_factor
        if 'urgency' in suppressed:
            if torch.is_tensor(suppressed['urgency']):
                suppressed['urgency'] = (suppressed['urgency'] * scale_factor).clamp(0, 3)
            else:
                suppressed['urgency'] = max(0, int(suppressed['urgency'] * scale_factor))
        return suppressed
    
    def _graded_suppress(
        self,
        outputs: Dict[str, torch.Tensor],
        uncertainty_value: float
    ) -> Dict[str, torch.Tensor]:
        """Graded suppression: apply suppression based on uncertainty level."""
        suppressed = outputs.copy()
        if uncertainty_value > 0.9:
            suppression_strength = 0.1
        elif uncertainty_value > 0.7:
            suppression_strength = 0.3
        else:
            suppression_strength = 0.5
        if 'confidence' in suppressed:
            suppressed['confidence'] = suppressed['confidence'] * suppression_strength
        if 'scores' in suppressed:
            suppressed['scores'] = suppressed['scores'] * suppression_strength
        return suppressed
    
    def _infer_uncertainty(self, outputs: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Try to infer uncertainty from outputs."""
        uncertainty_keys = ['uncertainty', 'uncertainty_score', 'uncertainty_map', 'confidence', 'confidence_score']
        for key in uncertainty_keys:
            if key in outputs:
                value = outputs[key]
                if torch.is_tensor(value):
                    if 'confidence' in key:
                        return 1.0 - value
                    return value
        return None


class SafetyChecker:
    """Safety checks for critical outputs."""
    
    def __init__(
        self,
        max_urgency_level: int = 3,
        min_confidence_for_action: float = 0.5,
        require_uncertainty_check: bool = True
    ):
        self.max_urgency_level = max_urgency_level
        self.min_confidence_for_action = min_confidence_for_action
        self.require_uncertainty_check = require_uncertainty_check
    
    def check_safety(
        self,
        outputs: Dict[str, torch.Tensor],
        uncertainty: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Check if outputs are safe to use."""
        safe = True
        reasons = []
        
        if 'urgency' in outputs:
            urgency = outputs['urgency']
            if torch.is_tensor(urgency):
                urgency_value = urgency.max().item() if urgency.numel() > 1 else urgency.item()
            else:
                urgency_value = int(urgency)
            if urgency_value > self.max_urgency_level:
                safe = False
                reasons.append(f"Urgency level {urgency_value} exceeds maximum {self.max_urgency_level}")
        
        if 'confidence' in outputs:
            confidence = outputs['confidence']
            if torch.is_tensor(confidence):
                min_confidence = confidence.min().item()
            else:
                min_confidence = float(confidence)
            if min_confidence < self.min_confidence_for_action:
                safe = False
                reasons.append(f"Confidence {min_confidence:.2f} below minimum {self.min_confidence_for_action}")
        
        uncertainty_value = None
        if self.require_uncertainty_check:
            if uncertainty is None:
                uncertainty = outputs.get('uncertainty_score')
            if uncertainty is not None:
                if torch.is_tensor(uncertainty):
                    uncertainty_value = uncertainty.mean().item()
                else:
                    uncertainty_value = float(uncertainty)
                if uncertainty_value > 0.8:
                    safe = False
                    reasons.append(f"Uncertainty {uncertainty_value:.2f} too high")
        
        return {'safe': safe, 'reasons': reasons, 'uncertainty': uncertainty_value}
    
    def filter_unsafe_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        uncertainty: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Filter out unsafe outputs."""
        safety_check = self.check_safety(outputs, uncertainty)
        if safety_check['safe']:
            return outputs
        
        filtered = outputs.copy()
        if 'urgency' in filtered:
            urgency = filtered['urgency']
            if torch.is_tensor(urgency):
                mask = urgency <= self.max_urgency_level
                for key in ['boxes', 'labels', 'confidence']:
                    if key in filtered and mask.numel() == filtered[key].shape[0]:
                        filtered[key] = filtered[key][mask]
        
        if 'confidence' in filtered:
            confidence = filtered['confidence']
            if torch.is_tensor(confidence):
                mask = confidence >= self.min_confidence_for_action
                for key in ['boxes', 'labels']:
                    if key in filtered and mask.numel() == filtered[key].shape[0]:
                        filtered[key] = filtered[key][mask]
        
        return filtered


class EthicalGuard:
    """Main ethical guard combining uncertainty suppression and safety checks."""
    
    def __init__(
        self,
        uncertainty_threshold: float = 0.7,
        suppression_mode: str = 'soft',
        enable_safety_checks: bool = True
    ):
        self.uncertainty_suppressor = UncertaintySuppressor(
            uncertainty_threshold=uncertainty_threshold,
            suppression_mode=suppression_mode
        )
        self.safety_checker = SafetyChecker() if enable_safety_checks else None
    
    def guard_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        uncertainty: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Apply ethical safeguards to outputs."""
        if uncertainty is None:
            uncertainty = outputs.get('uncertainty_score')
        
        suppressed_outputs = self.uncertainty_suppressor.suppress_outputs(outputs, uncertainty)
        
        safety_info = None
        if self.safety_checker is not None:
            safety_info = self.safety_checker.check_safety(suppressed_outputs, uncertainty)
            if not safety_info['safe']:
                suppressed_outputs = self.safety_checker.filter_unsafe_outputs(suppressed_outputs, uncertainty)
        
        # Check if any suppression occurred.
        suppressed = False
        if isinstance(outputs, dict) and isinstance(suppressed_outputs, dict):
            for key in set(outputs.keys()) | set(suppressed_outputs.keys()):
                if key in outputs and key in suppressed_outputs:
                    if torch.is_tensor(outputs[key]) and torch.is_tensor(suppressed_outputs[key]):
                        if not torch.equal(outputs[key], suppressed_outputs[key]):
                            suppressed = True
                            break
        
        return {
            'outputs': suppressed_outputs,
            'safety_info': safety_info,
            'uncertainty': uncertainty,
            'suppressed': suppressed
        }


def apply_ethical_guards(
    outputs: Dict[str, torch.Tensor],
    uncertainty_threshold: float = 0.7,
    enable_safety_checks: bool = True
) -> Dict[str, Any]:
    """Convenience function to apply ethical guards."""
    guard = EthicalGuard(
        uncertainty_threshold=uncertainty_threshold,
        enable_safety_checks=enable_safety_checks
    )
    return guard.guard_outputs(outputs)







