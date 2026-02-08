"""Voice Feedback Provides voice prompts for therapy guidance. Phase 4: Overlay Engine & UX Guidance See docs/therapy_system_implementation_plan.md for implementation details."""

from typing import Optional
from enum import Enum
import logging

# Use structured logging instead of print.
logger = logging.getLogger(__name__)


class VoicePrompt(Enum):
    """Therapy voice prompts."""
    TRY_CLOSER = "Try the closer object."
    FOLLOW_MOVEMENT = "Follow the movement."
    SHIFT_FOCUS_FAR = "Shift focus farther."
    TAKE_REST = "Take a short rest."
    GOOD_JOB = "Good job!"
    KEEP_TRYING = "Keep trying."
    ALMOST_THERE = "Almost there."


class VoiceFeedback:
    """Manages voice feedback for therapy tasks. Short, therapy-style prompts: - Try the closer object. - Follow the movement. - Shift focus farther. - Take a short rest."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.current_prompt = None
    
    def speak(self, prompt: VoicePrompt, priority: int = 0):
        """Speak a voice prompt. Arguments: prompt: VoicePrompt enum value priority: Priority level (higher = more urgent)"""
        if not self.enabled:
            return
        
        self.current_prompt = prompt
        # Use logger instead of print.
        logger.info(f"Voice: {prompt.value}")
    
    def speak_custom(self, text: str, priority: int = 0):
        """Speak custom text. Arguments: text: Custom text to speak priority: Priority level."""
        if not self.enabled:
            return
        
        self.current_prompt = text
        # Use logger instead of print.
        logger.info(f"Voice: {text}")
    
    def stop(self):
        """Stop current voice output."""
        self.current_prompt = None







