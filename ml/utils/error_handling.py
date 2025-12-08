"""
Error Handling and Fallback Logic for MaxSight
Handles error propagation and provides fallback mechanisms for dependent components.
"""

import torch
import logging
from typing import Dict, Any, Optional, Callable, List
from functools import wraps
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
    """
    Decorator to add fallback logic to functions.
    
    Arguments:
        fallback_value: Default value to return on error
        fallback_func: Function to call for fallback (takes same args as original)
        log_error: Whether to log errors
    """
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
    """
    Decorator to add timeout to functions.
    
    Arguments:
        timeout_ms: Timeout in milliseconds
    """
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
    """
    Manages head execution with error handling and fallbacks.
    
    Handles:
    - Dependency validation
    - Error propagation
    - Fallback execution
    - Timeout management
    """
    
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
        """
        Execute a head with error handling and fallbacks.
        
        Arguments:
            head_name: Name of the head
            head_func: Function to execute
            inputs: Input dictionary
            dependencies: List of required dependency keys
            fallback_func: Optional fallback function
        
        Returns:
            Head outputs dictionary
        """
        # Validate dependencies
        missing_deps = [dep for dep in dependencies if dep not in inputs]
        if missing_deps:
            if self.enable_fallbacks and fallback_func:
                logger.warning(f"{head_name} missing dependencies {missing_deps}, using fallback")
                try:
                    result = fallback_func(**inputs)
                    # Log fallback usage
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
        
        # Execute with timeout
        start_time = time.perf_counter()
        try:
            result = head_func(**inputs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            # Check timeout
            if elapsed_ms > self.timeout_ms:
                logger.warning(f"{head_name} exceeded timeout ({elapsed_ms:.2f}ms > {self.timeout_ms}ms)")
                if self.enable_fallbacks and fallback_func:
                    return fallback_func(**inputs)
            
            # Validate result
            if result is None:
                raise HeadExecutionError(f"{head_name} returned None")
            
            # Check for NaN/Inf
            if isinstance(result, torch.Tensor):
                if torch.isnan(result).any() or torch.isinf(result).any():
                    logger.warning(f"{head_name} produced NaN/Inf, using fallback")
                    if self.enable_fallbacks and fallback_func:
                        return fallback_func(**inputs)
                    return self._get_default_outputs(head_name)
            
            # Log execution
            self.execution_log.append({
                'head': head_name,
                'success': True,
                'elapsed_ms': elapsed_ms,
                'dependencies': dependencies
            })
            
            return result
            
        except Exception as e:
            logger.error(f"{head_name} execution failed: {e}")
            
            # Try fallback
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
            
            # Return default outputs
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
            'uncertainty': torch.ones(1, 1),  # High uncertainty = low confidence
        }
        
        if head_name in defaults:
            return {head_name: defaults[head_name]}
        return {}
    
    def check_uncertainty_fallback(
        self,
        uncertainty: torch.Tensor,
        outputs: Dict[str, Any]
    ) -> bool:
        """
        Check if uncertainty is high enough to trigger fallback.
        
        Arguments:
            uncertainty: Uncertainty tensor
            outputs: Current outputs
        
        Returns:
            True if fallback should be used
        """
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
    """
    Safely execute a head with error handling.
    
    Arguments:
        head_name: Name of the head
        head_func: Function to execute
        inputs: Input dictionary
        dependencies: Required dependencies
        manager: Optional HeadExecutionManager instance
        fallback_func: Optional fallback function
    
    Returns:
        Head outputs
    """
    if manager is None:
        manager = HeadExecutionManager()
    
    return manager.execute_head(
        head_name=head_name,
        head_func=head_func,
        inputs=inputs,
        dependencies=dependencies,
        fallback_func=fallback_func
    )

