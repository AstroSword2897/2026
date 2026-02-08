"""Output authority hierarchy for MaxSight Web Simulator. Defines clear priority system to prevent conflicting feedback."""
from enum import IntEnum
from typing import Optional, Dict, Any
from dataclasses import dataclass


class OutputAuthority(IntEnum):
    """Authority hierarchy for outputs (higher = more important). Lower layers cannot override higher layers."""
    DESCRIPTIVE_NARRATION = 1  # Lowest: General scene descriptions.
    THERAPY_PROMPTS = 2  # Therapy task instructions.
    NAVIGATION_GUIDANCE = 3  # Navigation assistance.
    SAFETY_ALERTS = 4  # Highest: Critical safety warnings.


@dataclass
class OutputRequest:
    """Represents an output request with authority level."""
    authority: OutputAuthority
    content: str
    priority: int = 0  # Within same authority level.
    suppress_lower: bool = True  # Whether to suppress lower authority outputs.
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class OutputAuthorityManager:
    """Manages output authority hierarchy. Ensures higher-priority outputs take precedence."""
    
    def __init__(self):
        self.current_output: Optional[OutputRequest] = None
        self.suppressed_outputs: list = []
    
    def request_output(self, request: OutputRequest) -> bool:
        """Request an output, respecting authority hierarchy."""
        # Allow output when no current output is active.
        if self.current_output is None:
            self.current_output = request
            return True
        
        # Compare authority levels.
        if request.authority > self.current_output.authority:
            # Higher authority: suppress current, allow new.
            self.suppressed_outputs.append(self.current_output)
            self.current_output = request
            return True
        elif request.authority < self.current_output.authority:
            # Suppress request when authority is lower than current output.
            self.suppressed_outputs.append(request)
            return False
        else:
            # Same authority: compare priority.
            if request.priority > self.current_output.priority:
                self.suppressed_outputs.append(self.current_output)
                self.current_output = request
                return True
            else:
                self.suppressed_outputs.append(request)
                return False
    
    def clear_current(self):
        """Clear current output (e.g., after completion)."""
        self.current_output = None
    
    def get_current_authority(self) -> Optional[OutputAuthority]:
        """Get current output authority level."""
        return self.current_output.authority if self.current_output else None
    
    def reset(self):
        """Reset manager state."""
        self.current_output = None
        self.suppressed_outputs.clear()







