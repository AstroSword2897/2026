"""
Simulation Harness Module

End-to-end simulation for testing:
- Model inference simulation
- Overlay rendering
- User interaction simulation
- Performance logging

See docs/therapy_system_implementation_plan.md Phase 5 for implementation details.
"""

from .simulator import TherapySimulator

__all__ = ['TherapySimulator']

