"""Performance monitoring utilities for MaxSight. Provides timing decorators and performance tracking for identifying bottlenecks."""

import time
import functools
import logging
from typing import Callable, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

# Global performance stats.
_performance_stats: dict[str, list[float]] = defaultdict(list)


def timed(threshold: float = 0.1, log_level: int = logging.WARNING):
    """Decorator to time function execution and log slow operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start
                _performance_stats[func.__name__].append(elapsed)
                
                if elapsed > threshold:
                    logger.log(
                        log_level,
                        f"{func.__name__} took {elapsed:.3f}s (threshold: {threshold:.3f}s)"
                    )
        
        return wrapper
    return decorator


def get_performance_stats() -> dict[str, dict[str, float]]:
    """Get performance statistics for all timed functions. Returns: Dictionary mapping function names to stats (mean, max, min, count)"""
    stats = {}
    for func_name, times in _performance_stats.items():
        if times:
            stats[func_name] = {
                'mean': sum(times) / len(times),
                'max': max(times),
                'min': min(times),
                'count': len(times),
                'total': sum(times)
            }
    return stats


def reset_performance_stats() -> None:
    """Reset all performance statistics."""
    _performance_stats.clear()


def log_slow_operations(threshold: float = 0.1) -> None:
    """Log summary of slow operations. Arguments: threshold: Minimum time to consider slow."""
    stats = get_performance_stats()
    slow_ops = {
        name: s for name, s in stats.items()
        if s['mean'] > threshold
    }
    
    if slow_ops:
        logger.warning(f"Slow operations (>{threshold}s):")
        for name, s in sorted(slow_ops.items(), key=lambda x: x[1]['mean'], reverse=True):
            logger.warning(
                f"  {name}: mean={s['mean']:.3f}s, max={s['max']:.3f}s, count={s['count']}"
            )







