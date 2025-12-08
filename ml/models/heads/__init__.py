"""
Therapy Heads for MaxSight

This module provides specialized heads for therapy tasks and adaptive assistance.
"""

from .contrast_head import ContrastMapHead
from .depth_head import DepthHead
from .fatigue_head import FatigueHead
from .motion_head import MotionHead
from .roi_priority_head import ROIPriorityHead
from .uncertainty_head import UncertaintyHead

# Head registry for dynamic head creation
HEAD_REGISTRY = {
    'contrast': ContrastMapHead,
    'depth': DepthHead,
    'fatigue': FatigueHead,
    'motion': MotionHead,
    'roi_priority': ROIPriorityHead,
    'uncertainty': UncertaintyHead,
}

def create_head(head_type: str, **kwargs):
    """
    Create a head by type name.
    
    Arguments:
        head_type: Type of head ('contrast', 'depth', 'fatigue', 'motion', 'roi_priority', 'uncertainty')
        **kwargs: Arguments to pass to head constructor
    
    Returns:
        Head module instance
    
    Raises:
        ValueError: If head_type is not in registry
    """
    if head_type not in HEAD_REGISTRY:
        available = ', '.join(HEAD_REGISTRY.keys())
        raise ValueError(
            f"Unknown head type: '{head_type}'. "
            f"Available types: {available}"
        )
    
    return HEAD_REGISTRY[head_type](**kwargs)

def list_available_heads() -> list:
    """List all available head types."""
    return list(HEAD_REGISTRY.keys())

__all__ = [
    'ContrastMapHead',
    'DepthHead',
    'FatigueHead',
    'MotionHead',
    'ROIPriorityHead',
    'UncertaintyHead',
    'HEAD_REGISTRY',
    'create_head',
    'list_available_heads',
]
