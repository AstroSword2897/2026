"""Attention modules for MaxSight 3.0."""

from .attention import (
    CBAM,
    SEBlock,
    ChannelAttention,
    SpatialAttention,
    CrossModalAttention,
    CrossTaskAttention
)

__all__ = [
    'CBAM',
    'SEBlock',
    'ChannelAttention',
    'SpatialAttention',
    'CrossModalAttention',
    'CrossTaskAttention'
]
