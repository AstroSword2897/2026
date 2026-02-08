"""Session Manager."""

from typing import Dict, List, Optional, Any, cast
from datetime import datetime
import json


class SessionManager:
    """Manages therapy sessions."""
    
    def __init__(self, user_id: Optional[str] = None):
        self.user_id = user_id
        self.current_session = None
        self.session_history = []
        self.task_attempts = []
    
    def start_session(self, session_config: Optional[Dict[str, Any]] = None) -> str:
        """Start a new therapy session. Arguments: session_config: Optional session configuration Returns: Session ID."""
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
        """Log a task attempt."""
        if not self.current_session:
            self.start_session()
        
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
        
        # Update metrics.
        self.current_session['metrics']['total_tasks'] += 1
        if result.get('success', False):
            self.current_session['metrics']['completed_tasks'] += 1
        else:
            self.current_session['metrics']['failed_tasks'] += 1
        
        if 'reaction_time' in result:
            self.current_session['metrics']['total_time'] += result['reaction_time']
    
    def end_session(self) -> Dict[str, Any]:
        """End current session and generate report. Returns: Session report dictionary."""
        if not self.current_session:
            return {}
        
        self.current_session['end_time'] = datetime.now().isoformat()
        
        # Generate skill curve.
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







