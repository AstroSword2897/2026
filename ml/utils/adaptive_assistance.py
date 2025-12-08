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

TECHNICAL DESIGN DECISION:
We use performance metrics (accuracy, reaction time, skill progression) rather than time-based
reduction because:
- Different users improve at different rates
- Performance-based adaptation is more responsive to actual needs
- Supports both learning (detailed when struggling) and independence (brief when skilled)
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

@dataclass
class PerformanceMetrics:
    """
    Performance metrics for adaptive assistance.
    
    WHY THESE METRICS:
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
    
    WHY THIS CLASS EXISTS:
    This class bridges SessionManager (performance tracking) and OutputScheduler (assistance
    levels). It enables the system to automatically adjust verbosity, frequency, and hazard
    alert levels based on demonstrated user capabilities, supporting gradual independence
    while maintaining safety.
    """
    
    def __init__(
        self,
        initial_verbosity: str = 'detailed',
        min_verbosity: str = 'brief',
        max_verbosity: str = 'detailed',
        use_ewma: bool = True,
        ewma_alpha: float = 0.3,
        accuracy_threshold_high: float = 0.8,
        accuracy_threshold_low: float = 0.5,
        reaction_time_threshold_fast: float = 1.0,
        reaction_time_threshold_slow: float = 3.0,
        skill_progression_threshold: float = 0.2,
        hazard_awareness_threshold_high: float = 0.8,
        hazard_awareness_threshold_low: float = 0.5
    ):
        """
        Initialize adaptive assistance.
        
        WHY THESE PARAMETERS:
        - initial_verbosity: Start with detailed (supports learning)
        - min_verbosity: Never go below brief (maintains safety)
        - max_verbosity: Can increase to detailed when needed (supports struggling users)
        - use_ewma: Use exponential weighted moving average for stability
        - ewma_alpha: EWMA smoothing factor (0-1, higher = more responsive)
        - *_threshold_*: Configurable thresholds for adaptation decisions
        
        This ensures assistance adapts within safe bounds while supporting skill development.
        
        Arguments:
            initial_verbosity: Starting verbosity level
            min_verbosity: Minimum verbosity (safety floor)
            max_verbosity: Maximum verbosity (support ceiling)
            use_ewma: Enable EWMA smoothing for metrics
            ewma_alpha: EWMA smoothing factor (0-1)
            accuracy_threshold_high: High accuracy threshold (0-1)
            accuracy_threshold_low: Low accuracy threshold (0-1)
            reaction_time_threshold_fast: Fast reaction time threshold (seconds)
            reaction_time_threshold_slow: Slow reaction time threshold (seconds)
            skill_progression_threshold: Skill progression threshold (-1 to 1)
            hazard_awareness_threshold_high: High hazard awareness threshold (0-1)
            hazard_awareness_threshold_low: Low hazard awareness threshold (0-1)
        """
        self.initial_verbosity = initial_verbosity
        self.min_verbosity = min_verbosity
        self.max_verbosity = max_verbosity
        self.performance_history: List[PerformanceMetrics] = []
        
        # EWMA smoothing
        self.use_ewma = use_ewma
        self.ewma_alpha = ewma_alpha
        self.ewma_metrics: Optional[PerformanceMetrics] = None
        
        # Configurable thresholds
        self.accuracy_threshold_high = accuracy_threshold_high
        self.accuracy_threshold_low = accuracy_threshold_low
        self.reaction_time_threshold_fast = reaction_time_threshold_fast
        self.reaction_time_threshold_slow = reaction_time_threshold_slow
        self.skill_progression_threshold = skill_progression_threshold
        self.hazard_awareness_threshold_high = hazard_awareness_threshold_high
        self.hazard_awareness_threshold_low = hazard_awareness_threshold_low
    
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
        
        # Update EWMA metrics
        if self.use_ewma:
            if self.ewma_metrics is None:
                self.ewma_metrics = metrics
            else:
                # Exponential weighted moving average
                self.ewma_metrics = PerformanceMetrics(
                    accuracy=self.ewma_alpha * metrics.accuracy + (1 - self.ewma_alpha) * self.ewma_metrics.accuracy,
                    reaction_time=self.ewma_alpha * metrics.reaction_time + (1 - self.ewma_alpha) * self.ewma_metrics.reaction_time,
                    skill_progression=self.ewma_alpha * metrics.skill_progression + (1 - self.ewma_alpha) * self.ewma_metrics.skill_progression,
                    hazard_awareness=self.ewma_alpha * metrics.hazard_awareness + (1 - self.ewma_alpha) * self.ewma_metrics.hazard_awareness,
                    session_count=metrics.session_count
                )
        
        # Keep only recent history (last 10 sessions)
        if len(self.performance_history) > 10:
            self.performance_history = self.performance_history[-10:]
    
    def get_average_metrics(self) -> PerformanceMetrics:
        """
        Get rolling average metrics for stability.
        
        WHY THIS MATTERS:
        Prevents abrupt verbosity changes based on single session. Uses rolling average
        or EWMA for smoother adaptation.
        
        Returns:
            Averaged performance metrics
        """
        if self.use_ewma and self.ewma_metrics is not None:
            return self.ewma_metrics
        
        if not self.performance_history:
            return PerformanceMetrics(0, 2.0, 0, 0, 0)
        
        n = len(self.performance_history)
        return PerformanceMetrics(
            accuracy=sum(m.accuracy for m in self.performance_history) / n,
            reaction_time=sum(m.reaction_time for m in self.performance_history) / n,
            skill_progression=sum(m.skill_progression for m in self.performance_history) / n,
            hazard_awareness=sum(m.hazard_awareness for m in self.performance_history) / n,
            session_count=self.performance_history[-1].session_count
        )
    
    def get_adaptive_verbosity(self, use_numeric: bool = False) -> Any:
        """
        Get adaptive verbosity based on performance.
        
        WHY ADAPTIVE VERBOSITY:
        Verbosity should match user needs:
        - High performance + improving = brief (encourage independence)
        - Low performance or struggling = detailed (provide support)
        - Medium performance = normal (balanced approach)
        
        This supports "Skill Development Across Senses" by providing appropriate detail levels
        that encourage learning without creating dependence.
        
        Arguments:
            use_numeric: If True, return numeric level (0-3) instead of string
        
        Returns:
            Adaptive verbosity level ('brief', 'normal', 'detailed') or numeric (0-3)
        """
        if not self.performance_history:
            if use_numeric:
                return {'brief': 0, 'normal': 1, 'detailed': 2, 'very_detailed': 3}.get(self.initial_verbosity, 1)
            return self.initial_verbosity
        
        # Use averaged metrics for stability
        metrics = self.get_average_metrics()
        
        # High performance + improving = reduce verbosity
        if metrics.accuracy > self.accuracy_threshold_high and metrics.skill_progression > self.skill_progression_threshold:
            # User is skilled and improving - encourage independence
            if use_numeric:
                return 0  # brief
            if self.min_verbosity == 'brief':
                return 'brief'
            return 'normal'
        
        # Low performance or declining = increase verbosity
        if metrics.accuracy < self.accuracy_threshold_low or metrics.skill_progression < -self.skill_progression_threshold:
            # User is struggling - provide more support
            if use_numeric:
                return 3  # very_detailed
            if self.max_verbosity == 'detailed':
                return 'detailed'
            return 'normal'
        
        # Medium performance = normal verbosity
        if use_numeric:
            return 1  # normal
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
        
        # Use averaged metrics for stability
        metrics = self.get_average_metrics()
        
        # High performance = reduce frequency
        if metrics.accuracy > self.accuracy_threshold_high and metrics.reaction_time < self.reaction_time_threshold_fast:
            return 'low'  # User is skilled - minimal interruption
        
        # Low performance = increase frequency
        if metrics.accuracy < self.accuracy_threshold_low or metrics.reaction_time > self.reaction_time_threshold_slow:
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
        
        # Use averaged metrics for stability
        metrics = self.get_average_metrics()
        
        # High hazard awareness = only high-urgency alerts
        if metrics.hazard_awareness > self.hazard_awareness_threshold_high:
            return 2  # Only warning and danger
        
        # Low hazard awareness = alert to all hazards
        if metrics.hazard_awareness < self.hazard_awareness_threshold_low:
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
    
    WHY THIS FUNCTION:
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

