"""Haptic Feedback Provides haptic feedback for therapy tasks. Phase 4: Overlay Engine & UX Guidance See docs/therapy_system_implementation_plan.md for implementation details."""

from typing import Optional
from enum import Enum
import logging

# Use structured logging instead of print.
logger = logging.getLogger(__name__)


class HapticPattern(Enum):
    """Haptic feedback patterns."""
    MICRO_PULSE = "micro_pulse"  # Target found.
    LONG_PULSE = "long_pulse"  # Wrong region.
    SUCCESS_SEQUENCE = "success_sequence"
    FAILURE_SEQUENCE = "failure_sequence"


class HapticFeedback:
    """Manages haptic feedback for therapy tasks. Patterns: - Micro pulse: target found - Longer pulse: wrong region."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
    
    def trigger(self, pattern: HapticPattern, intensity: float = 0.5):
        """Trigger haptic feedback pattern. Arguments: pattern: HapticPattern enum value intensity: Intensity [0, 1]."""
        if not self.enabled:
            return
        
        # Use logger instead of print.
        logger.info(f"Haptic {pattern.value} intensity: {intensity}")
    
    def micro_pulse(self, intensity: float = 0.3):
        """Short pulse for target found."""
        self.trigger(HapticPattern.MICRO_PULSE, intensity)
    
    def long_pulse(self, intensity: float = 0.6):
        """Longer pulse for wrong region."""
        self.trigger(HapticPattern.LONG_PULSE, intensity)
    
    def success_sequence(self):
        """Success feedback sequence."""
        self.trigger(HapticPattern.SUCCESS_SEQUENCE, 0.7)
    
    def failure_sequence(self):
        """Failure feedback sequence."""
        self.trigger(HapticPattern.FAILURE_SEQUENCE, 0.4)







