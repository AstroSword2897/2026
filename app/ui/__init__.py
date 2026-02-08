"""UI Components Module Contains user interface components: - Voice feedback - Haptic feedback - Visual guidance See docs/therapy_system_implementation_plan.md Phase 4 for implementation details."""

from .voice_feedback import VoiceFeedback, VoicePrompt
from .haptic_feedback import HapticFeedback, HapticPattern

__all__ = [
    'VoiceFeedback',
    'VoicePrompt',
    'HapticFeedback',
    'HapticPattern'
]


