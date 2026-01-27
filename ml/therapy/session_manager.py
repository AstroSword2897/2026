"""
Session Manager

Manages therapy sessions, tracks performance, and generates progress reports.

PROJECT PHILOSOPHY & APPROACH:
This module implements "Skill Development Across Senses" and "Routine Workflow" barrier removal
methods. It's not just about logging data - it's about supporting users in developing visual and
spatial skills over time.

WHY SESSION MANAGEMENT MATTERS:
MaxSight is designed to support both immediate assistance (environmental awareness) and long-term
skill development (vision therapy). This module enables the latter by:
1. Tracking user performance over time
2. Identifying skill improvements
3. Adapting difficulty based on progress
4. Supporting gradual independence (reducing reliance on app)

This directly addresses the problem statement's emphasis on "Skill Development Across Senses" - users
don't just get information, they build skills that reduce their dependence on assistive technology.

HOW IT CONNECTS TO THE PROBLEM STATEMENT:
The problem asks: "What are ways that those who cannot see... be able to interact with the world
like those who can?" This module answers by supporting skill development - helping users improve
their visual and spatial abilities so they can interact more independently over time.

RELATIONSHIP TO BARRIER REMOVAL METHODS:
1. SKILL DEVELOPMENT ACROSS SENSES: Core implementation - tracks and supports skill development
2. ROUTINE WORKFLOW: Adapts to user patterns and needs based on session history
3. ENVIRONMENTAL STRUCTURING: Provides structured feedback on environmental awareness tasks
4. GRADUAL INDEPENDENCE: Enables reducing assistance as skills improve

HOW IT CONTRIBUTES TO VISUAL AWARENESS GOALS:
This module supports "Adaptive Visual Assistance" and "Visual Training" goals by:
- Tracking progress in object recognition
- Monitoring improvement in spatial awareness
- Adapting assistance levels based on performance
- Providing feedback that reinforces learning

TECHNICAL DESIGN DECISION:
We track multiple metrics (reaction time, accuracy, gaze path) because different vision conditions
affect different skills. This comprehensive tracking ensures we can adapt to each user's specific
needs and support their unique skill development journey.
"""

from typing import Dict, List, Optional, Any, cast
from datetime import datetime
import json


class SessionManager:
    """
    Manages therapy sessions.
    
    This dual purpose is critical because:
    - Immediate assistance: Users need help navigating safely right now
    - Skill development: Users want to improve their abilities to reduce dependence
    
    By tracking sessions and performance, this class enables MaxSight to adapt and support both
    goals simultaneously.
    
    Logs:
    - All attempts
    - Reaction time
    - Path of gaze
    - Miss/Fail categories
    - Generates skill curve
    """
    
    def __init__(self, user_id: Optional[str] = None):
        self.user_id = user_id
        self.current_session = None
        self.session_history = []
        self.task_attempts = []
    
    def start_session(self, session_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new therapy session.
        
        Arguments:
            session_config: Optional session configuration
        
        Returns:
            Session ID
        """
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session = {
            'session_id': session_id,
            'start_time': datetime.now().isoformat(),
            'config': session_config or {},
            'tasks': [],
            'metrics': {
                'total_tasks': 0,
                'completed_tasks': 0,
                'failed_tasks': 0,
                'total_time': 0.0
            }
        }
        return session_id
    
    def log_task_attempt(
        self,
        task_type: str,
        task_config: Dict[str, Any],
        result: Dict[str, Any]
    ):
        """
        Log a task attempt.
        
        Arguments:
            task_type: Type of task
            task_config: Task configuration
            result: Task result with:
                - 'success': bool
                - 'reaction_time': float (seconds)
                - 'gaze_path': List[Tuple[float, float]] (optional)
                - 'misses': int
                - 'fails': int
        """
        if not self.current_session:
            self.start_session()
        
        # Type narrowing: current_session is guaranteed to be non-None after start_session
        if self.current_session is None:
            raise RuntimeError("Failed to initialize session")
        
        attempt = {
            'timestamp': datetime.now().isoformat(),
            'task_type': task_type,
            'task_config': task_config,
            'result': result
        }
        
        self.task_attempts.append(attempt)
        self.current_session['tasks'].append(attempt)
        
        # Update metrics
        self.current_session['metrics']['total_tasks'] += 1
        if result.get('success', False):
            self.current_session['metrics']['completed_tasks'] += 1
        else:
            self.current_session['metrics']['failed_tasks'] += 1
        
        if 'reaction_time' in result:
            self.current_session['metrics']['total_time'] += result['reaction_time']
    
    def end_session(self) -> Dict[str, Any]:
        """
        End current session and generate report.
        
        Returns:
            Session report dictionary
        """
        if not self.current_session:
            return {}
        
        self.current_session['end_time'] = datetime.now().isoformat()
        
        # Generate skill curve
        skill_curve = self._generate_skill_curve()
        
        report = {
            **self.current_session,
            'skill_curve': skill_curve,
            'summary': self._generate_summary()
        }
        
        self.session_history.append(report)
        self.current_session = None
        self.task_attempts = []
        
        return report
    
    def _generate_skill_curve(self) -> List[Dict[str, Any]]:
        """Generate skill progression curve from session tasks."""
        if self.current_session is None:
            return []
        
        curve = []
        for i, task in enumerate(self.current_session['tasks']):
            success = task['result'].get('success', False)
            reaction_time = task['result'].get('reaction_time', 0.0)
            curve.append({
                'task_index': i,
                'success': success,
                'reaction_time': reaction_time,
                'cumulative_success_rate': sum(
                    1 for t in self.current_session['tasks'][:i+1]
                    if t['result'].get('success', False)
                ) / (i + 1)
            })
        return curve
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate session summary."""
        if self.current_session is None:
            return {}
        
        metrics = self.current_session['metrics']
        total = metrics['total_tasks']
        
        if total == 0:
            return {'success_rate': 0.0, 'avg_reaction_time': 0.0}
        
        success_rate = metrics['completed_tasks'] / total
        avg_reaction_time = metrics['total_time'] / total
        
        return {
            'success_rate': success_rate,
            'avg_reaction_time': avg_reaction_time,
            'total_tasks': total,
            'completed_tasks': metrics['completed_tasks'],
            'failed_tasks': metrics['failed_tasks']
        }
    
    def save_session(self, filepath: str):
        """Save session to file."""
        if self.current_session:
            with open(filepath, 'w') as f:
                json.dump(self.current_session, f, indent=2)

