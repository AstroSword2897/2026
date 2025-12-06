"""
Adaptive Assistance Module for MaxSight
Ties verbosity and hazard alerts to user performance from session manager/therapy tasks.

PROJECT PHILOSOPHY & APPROACH:
This module implements "Adaptive Visual Assistance" - adjusting assistance levels based on user
performance to support gradual independence. This directly addresses "Skill Development Across
Senses" by reducing assistance as users improve, encouraging skill development rather than
dependence.

WHY ADAPTIVE ASSISTANCE MATTERS:
Fixed assistance levels create dependence. Adaptive assistance:
1. Starts with detailed descriptions (supports learning)
2. Gradually reduces detail as skills improve (encourages independence)
3. Increases detail when performance drops (provides support when needed)
4. Adapts hazard alerts based on user's demonstrated awareness

This supports the problem statement's emphasis on "gradually reduced assistance" - users don't
just get help, they develop skills that reduce their need for assistance over time.

HOW IT CONNECTS TO THE PROBLEM STATEMENT:
The problem asks for ways to help users "interact with the world like those who can." Adaptive
assistance answers this by supporting skill development - as users improve their visual and
spatial abilities, the system reduces assistance, enabling more independent interaction.

RELATIONSHIP TO BARRIER REMOVAL METHODS:
1. SKILL DEVELOPMENT ACROSS SENSES: Core implementation - adapts assistance to support learning
2. ROUTINE WORKFLOW: Adapts to user's demonstrated capabilities over time
3. GRADUAL INDEPENDENCE: Reduces assistance as skills improve, increases when needed
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class PerformanceMetrics:
    """
    Performance metrics for adaptive assistance.
    
    - accuracy: How well user identifies objects (higher = less assistance needed)
    - reaction_time: How quickly user responds (faster = more skilled)
    - skill_progression: Trend over time (improving = reduce assistance)
    - hazard_awareness: How well user detects hazards (higher = reduce hazard alerts)
    
    These metrics enable data-driven adaptation that responds to actual user capabilities,
    supporting both learning and independence.
    """
    accuracy: float  # 0-1, object recognition accuracy
    reaction_time: float  # seconds, average reaction time
    skill_progression: float  # -1 to 1, trend (positive = improving)
    hazard_awareness: float  # 0-1, how well user detects hazards
    session_count: int  # Number of sessions completed


class AdaptiveAssistance:
    """
    Adapts assistance levels based on user performance.
    
    This class bridges SessionManager (performance tracking) and OutputScheduler (assistance
    levels). It enables the system to automatically adjust verbosity, frequency, and hazard
    alert levels based on demonstrated user capabilities, supporting gradual independence
    while maintaining safety.
    """
    
    def __init__(
        self,
        initial_verbosity: str = 'detailed',
        min_verbosity: str = 'brief',
        max_verbosity: str = 'detailed'
    ):
        """
        Initialize adaptive assistance.
        
        - initial_verbosity: Start with detailed (supports learning)
        - min_verbosity: Never go below brief (maintains safety)
        - max_verbosity: Can increase to detailed when needed (supports struggling users)
        
        This ensures assistance adapts within safe bounds while supporting skill development.
        
        Arguments:
            initial_verbosity: Starting verbosity level
            min_verbosity: Minimum verbosity (safety floor)
            max_verbosity: Maximum verbosity (support ceiling)
        """
        self.initial_verbosity = initial_verbosity
        self.min_verbosity = min_verbosity
        self.max_verbosity = max_verbosity
        self.performance_history: List[PerformanceMetrics] = []
    
    def update_performance(
        self,
        accuracy: float,
        reaction_time: float,
        skill_progression: float,
        hazard_awareness: float,
        session_count: int
    ) -> None:
        """
        Update performance metrics.
        
        WHY THIS MATTERS:
        Performance metrics drive adaptation. This method updates the system's understanding
        of user capabilities, enabling data-driven assistance level adjustments.
        
        Arguments:
            accuracy: Object recognition accuracy (0-1)
            reaction_time: Average reaction time (seconds)
            skill_progression: Skill trend (-1 to 1, positive = improving)
            hazard_awareness: Hazard detection accuracy (0-1)
            session_count: Number of sessions completed
        """
        metrics = PerformanceMetrics(
            accuracy=accuracy,
            reaction_time=reaction_time,
            skill_progression=skill_progression,
            hazard_awareness=hazard_awareness,
            session_count=session_count
        )
        self.performance_history.append(metrics)
        
        # Keep only recent history (last 10 sessions)
        if len(self.performance_history) > 10:
            self.performance_history = self.performance_history[-10:]
    
    def get_adaptive_verbosity(self) -> str:
        """
        Get adaptive verbosity based on performance.
        
        WHY ADAPTIVE VERBOSITY:
        Verbosity should match user needs:
        - High performance + improving = brief (encourage independence)
        - Low performance or struggling = detailed (provide support)
        - Medium performance = normal (balanced approach)
        
        This supports "Skill Development Across Senses" by providing appropriate detail levels
        that encourage learning without creating dependence.
        
        Returns:
            Adaptive verbosity level ('brief', 'normal', 'detailed')
        """
        if not self.performance_history:
            return self.initial_verbosity
        
        latest = self.performance_history[-1]
        
        # High performance + improving = reduce verbosity
        if latest.accuracy > 0.8 and latest.skill_progression > 0.2:
            # User is skilled and improving - encourage independence
            if self.min_verbosity == 'brief':
                return 'brief'
            return 'normal'
        
        # Low performance or declining = increase verbosity
        if latest.accuracy < 0.5 or latest.skill_progression < -0.2:
            # User is struggling - provide more support
            if self.max_verbosity == 'detailed':
                return 'detailed'
            return 'normal'
        
        # Medium performance = normal verbosity
        return 'normal'
    
    def get_adaptive_frequency(self) -> str:
        """
        Get adaptive alert frequency based on performance.
        
        WHY ADAPTIVE FREQUENCY:
        Alert frequency should adapt to user needs:
        - High performance = low frequency (less interruption)
        - Low performance = high frequency (more guidance)
        - Medium performance = medium frequency (balanced)
        
        This supports "Clear Multimodal Communication" by adjusting information density
        based on demonstrated user capabilities.
        
        Returns:
            Adaptive frequency level ('low', 'medium', 'high')
        """
        if not self.performance_history:
            return 'medium'
        
        latest = self.performance_history[-1]
        
        # High performance = reduce frequency
        if latest.accuracy > 0.8 and latest.reaction_time < 1.0:
            return 'low'  # User is skilled - minimal interruption
        
        # Low performance = increase frequency
        if latest.accuracy < 0.5 or latest.reaction_time > 3.0:
            return 'high'  # User needs more guidance
        
        return 'medium'
    
    def get_adaptive_hazard_threshold(self) -> int:
        """
        Get adaptive hazard alert threshold based on performance.
        
        WHY ADAPTIVE HAZARD THRESHOLDS:
        Hazard awareness varies by user:
        - High hazard awareness = only alert to high-urgency hazards (reduce false alarms)
        - Low hazard awareness = alert to all hazards (ensure safety)
        
        This supports "Safety-Oriented Visual Awareness" by adapting hazard alerts to user's
        demonstrated awareness while maintaining safety.
        
        Returns:
            Minimum urgency level for alerts (0-3, higher = fewer alerts)
        """
        if not self.performance_history:
            return 1  # Default: alert to caution and above
        
        latest = self.performance_history[-1]
        
        # High hazard awareness = only high-urgency alerts
        if latest.hazard_awareness > 0.8:
            return 2  # Only warning and danger
        
        # Low hazard awareness = alert to all hazards
        if latest.hazard_awareness < 0.5:
            return 0  # Alert to all urgency levels
        
        return 1  # Default: caution and above
    
    def get_adaptive_config(self) -> Dict[str, any]:
        """
        Get complete adaptive configuration.
        
        WHY THIS FUNCTION:
        Provides a single interface for getting all adaptive settings, enabling easy
        integration with OutputScheduler and DescriptionGenerator. This ensures all
        assistance components adapt consistently based on user performance.
        
        Returns:
            Dictionary with adaptive verbosity, frequency, and hazard threshold
        """
        return {
            'verbosity': self.get_adaptive_verbosity(),
            'frequency': self.get_adaptive_frequency(),
            'hazard_threshold': self.get_adaptive_hazard_threshold()
        }


def create_adaptive_assistance_from_session(
    session_manager,
    initial_verbosity: str = 'detailed'
) -> AdaptiveAssistance:
    """
    Create adaptive assistance from session manager.
    
    Bridges SessionManager (performance tracking) and AdaptiveAssistance (assistance adaptation).
    This enables automatic assistance level adjustment based on therapy session performance,
    supporting gradual independence through data-driven adaptation.
    
        Arguments:
        session_manager: SessionManager instance with performance data
        initial_verbosity: Starting verbosity level
    
    Returns:
        AdaptiveAssistance instance configured from session data
    """
    adaptive = AdaptiveAssistance(initial_verbosity=initial_verbosity)
    
    # Extract performance metrics from session manager
    if session_manager.current_session and session_manager.task_attempts:
        # Calculate metrics from recent tasks
        recent_tasks = session_manager.task_attempts[-10:]  # Last 10 tasks
        
        if recent_tasks:
            successes = sum(1 for t in recent_tasks if t.get('result', {}).get('success', False))
            accuracy = successes / len(recent_tasks)
            
            reaction_times = [
                t.get('result', {}).get('reaction_time', 2.0)
                for t in recent_tasks
                if t.get('result', {}).get('reaction_time') is not None
            ]
            avg_reaction_time = sum(reaction_times) / len(reaction_times) if reaction_times else 2.0
            
            # Simple skill progression (compare recent vs older tasks)
            if len(recent_tasks) >= 5:
                recent_accuracy = sum(
                    1 for t in recent_tasks[-5:]
                    if t.get('result', {}).get('success', False)
                ) / 5
                older_accuracy = sum(
                    1 for t in recent_tasks[:5]
                    if t.get('result', {}).get('success', False)
                ) / 5
                skill_progression = recent_accuracy - older_accuracy
            else:
                skill_progression = 0.0
            
            # Hazard awareness (placeholder - would need hazard-specific tasks)
            hazard_awareness = accuracy  # Use general accuracy as proxy
            
            adaptive.update_performance(
                accuracy=accuracy,
                reaction_time=avg_reaction_time,
                skill_progression=skill_progression,
                hazard_awareness=hazard_awareness,
                session_count=len(session_manager.session_history)
            )
    
    return adaptive

