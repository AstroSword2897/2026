"""Task Generator."""

from typing import Dict, List, Optional, Any
from enum import Enum


class TaskType(Enum):
    """Therapy task types."""
    CONTRAST_MICRO = "contrast_micro"  # Edge finding.
    MOTION_TRACKING = "motion_tracking"
    DEPTH_SHIFT = "depth_shift"  # Focus near→far→near.
    GAZE_STABILIZATION = "gaze_stabilization"
    ROI_FINDABILITY = "roi_findability"
    FATIGUE_REST = "fatigue_rest"


class TaskGenerator:
    """Generates adaptive therapy tasks."""
    
    def __init__(self, user_profile: Optional[Dict[str, Any]] = None):
        self.user_profile = user_profile or {}
        self.recent_failures = []
        self.task_history = []
    
    def generate_task(
        self,
        uncertainty: float,
        fatigue_score: float,
        recent_performance: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate next therapy task."""
        # TODO: Implement adaptive task generation logic. For now, return default task.
        
        # If fatigued, suggest rest task.
        if fatigue_score > 0.7:
            return {
                'task_type': TaskType.FATIGUE_REST,
                'difficulty': 0.0,
                'duration': 60,
                'highlight_strength': 0.0,
                'target_speed': 0.0
            }
        
        # If high uncertainty, reduce difficulty.
        base_difficulty = max(0.1, 1.0 - uncertainty)
        
        # Choose task type based on recent failures.
        task_type = self._choose_task_type(recent_performance)
        
        return {
            'task_type': task_type,
            'difficulty': base_difficulty,
            'duration': int(30 + (1.0 - base_difficulty) * 30),  # 30-60 seconds.
            'highlight_strength': base_difficulty,
            'target_speed': base_difficulty * 100.0  # Pixels/second.
        }
    
    def _choose_task_type(self, recent_performance: List[Dict[str, Any]]) -> TaskType:
        """Choose task type based on recent performance."""
        # TODO: Implement task type selection logic. For now, cycle through task types.
        if not self.task_history:
            return TaskType.CONTRAST_MICRO
        
        last_task = self.task_history[-1]
        task_order = [
            TaskType.CONTRAST_MICRO,
            TaskType.MOTION_TRACKING,
            TaskType.DEPTH_SHIFT,
            TaskType.GAZE_STABILIZATION,
            TaskType.ROI_FINDABILITY
        ]
        
        try:
            current_idx = task_order.index(last_task['task_type'])
            next_idx = (current_idx + 1) % len(task_order)
            return task_order[next_idx]
        except (ValueError, KeyError):
            return TaskType.CONTRAST_MICRO
    
    def update_performance(self, task_result: Dict[str, Any]):
        """Update task generator with task result."""
        self.task_history.append(task_result)
        if task_result.get('failed', False):
            self.recent_failures.append(task_result)
            # Keep only recent failures.
            if len(self.recent_failures) > 10:
                self.recent_failures.pop(0)







