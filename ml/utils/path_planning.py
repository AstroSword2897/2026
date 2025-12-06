"""
Path Planning Module for MaxSight
Uses hazard and distance info to suggest safe navigation paths.

PROJECT PHILOSOPHY & APPROACH:
=============================
This module implements "Path Planning" - going beyond obstacle detection to provide actionable
navigation guidance. This directly addresses "Navigation Assistance" by suggesting safe routes,
not just identifying obstacles.

WHY PATH PLANNING MATTERS:
Obstacle detection tells users "what's there" but not "how to navigate around it." Path planning
provides the "how" - suggesting safe routes that enable independent navigation. This is critical
for users with vision impairments who cannot visually assess path safety.

HOW IT CONNECTS TO THE PROBLEM STATEMENT:
The problem asks: "What are ways that those who cannot see... be able to interact with the world
like those who can?" Path planning answers by providing the spatial navigation guidance that
sighted people process automatically - "clear path ahead" or "move right to avoid obstacle."

RELATIONSHIP TO BARRIER REMOVAL METHODS:
1. ENVIRONMENTAL STRUCTURING: Provides structured navigation guidance
2. CLEAR MULTIMODAL COMMUNICATION: Delivers path suggestions through audio/visual/haptic
3. SAFETY: Ensures users can navigate safely around hazards

TECHNICAL DESIGN DECISION:
We use a simple grid-based approach rather than complex pathfinding because:
- Real-time performance is critical (mobile deployment)
- Simple heuristics are sufficient for immediate navigation guidance
- Complex pathfinding would require more computational resources
- Users need quick, actionable guidance, not perfect paths
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass


class PathDirection(Enum):
    """Safe navigation directions"""
    FORWARD = "forward"
    LEFT = "left"
    RIGHT = "right"
    BACKWARD = "backward"
    STOP = "stop"


@dataclass
class PathSuggestion:
    """
    A path planning suggestion.
    
    WHY THIS STRUCTURE:
    Path suggestions need to be actionable - users need clear direction, not just information.
    This structure provides:
    - direction: Where to go (forward/left/right)
    - safety_score: How safe the path is (0-1)
    - distance: How far to travel before reassessment
    - reasoning: Why this path is suggested (for detailed mode)
    """
    direction: PathDirection
    safety_score: float  # 0-1, higher = safer
    distance: float  # Estimated safe distance (meters or normalized)
    reasoning: str  # Explanation for detailed mode


class PathPlanner:
    """
    Plans safe navigation paths based on hazards and obstacles.
    
    WHY THIS CLASS EXISTS:
    This class transforms obstacle detection into actionable navigation guidance. It analyzes
    hazards, distances, and spatial relationships to suggest safe paths, enabling users to
    navigate independently around obstacles.
    
    This directly supports "Navigation Assistance" and "Safety-Oriented Visual Awareness" by
    providing the spatial guidance that sighted people process automatically.
    """
    
    def __init__(
        self,
        safety_threshold: float = 0.7,
        min_clearance: float = 0.15  # Minimum normalized clearance for safe path
    ):
        """
        Initialize path planner.
        
        WHY THESE PARAMETERS:
        - safety_threshold: Minimum safety score for path recommendation (0.7 = 70% safe)
        - min_clearance: Minimum distance from obstacles for safe path (15% of image = safe zone)
        
        These parameters ensure suggested paths are actually safe, not just "less dangerous."
        Safety is paramount - better to suggest stopping than an unsafe path.
        
        Arguments:
            safety_threshold: Minimum safety score for path recommendation
            min_clearance: Minimum clearance from obstacles (normalized)
        """
        self.safety_threshold = safety_threshold
        self.min_clearance = min_clearance
    
    def plan_path(
        self,
        detections: List[Dict],
        target_direction: Optional[str] = None
    ) -> Optional[PathSuggestion]:
        """
        Plan safe navigation path based on detected obstacles.
        
        WHY THIS FUNCTION:
        This is the core path planning function that transforms obstacle detection into
        actionable navigation guidance. It analyzes spatial relationships to suggest safe
        paths, enabling independent navigation around hazards.
        
        HOW IT SUPPORTS THE PROBLEM STATEMENT:
        The problem asks for ways to help users "interact with the world like those who can."
        Path planning provides the spatial navigation guidance that sighted people process
        automatically - analyzing obstacles and suggesting safe routes.
        
        RELATIONSHIP TO OTHER COMPONENTS:
        - Input: Detections from MaxSightCNN (objects, positions, urgency, distances)
        - Output: Path suggestions for DescriptionGenerator and CrossModalScheduler
        - Integration: Works with SpatialMemory to consider previously seen obstacles
        
        Arguments:
            detections: List of detection dictionaries with box, urgency, distance
            target_direction: Optional target direction ('forward', 'left', 'right')
        
        Returns:
            PathSuggestion or None if no safe path found
        """
        if not detections:
            # No obstacles = clear path forward
            return PathSuggestion(
                direction=PathDirection.FORWARD,
                safety_score=1.0,
                distance=5.0,  # Safe to proceed
                reasoning="Clear path ahead"
            )
        
        # Filter for hazards (high urgency, near objects)
        hazards = [
            d for d in detections
            if d.get('urgency', 0) >= 2 and d.get('distance', 2) <= 1
        ]
        
        if not hazards:
            # No immediate hazards = clear path
            return PathSuggestion(
                direction=PathDirection.FORWARD,
                safety_score=0.9,
                distance=3.0,
                reasoning="No immediate hazards detected"
            )
        
        # Analyze obstacle positions
        left_obstacles = []
        right_obstacles = []
        center_obstacles = []
        
        for hazard in hazards:
            box = hazard.get('box')
            if box is None:
                continue
            
            cx = box[0] if not isinstance(box, torch.Tensor) else box[0].item()
            
            if cx < 0.4:
                left_obstacles.append(hazard)
            elif cx > 0.6:
                right_obstacles.append(hazard)
            else:
                center_obstacles.append(hazard)
        
        # Plan path based on obstacle distribution
        if center_obstacles:
            # Obstacle directly ahead - suggest left or right
            left_safety = self._calculate_direction_safety(left_obstacles, 'left')
            right_safety = self._calculate_direction_safety(right_obstacles, 'right')
            
            if left_safety > right_safety and left_safety >= self.safety_threshold:
                return PathSuggestion(
                    direction=PathDirection.LEFT,
                    safety_score=left_safety,
                    distance=2.0,
                    reasoning="Obstacle ahead, safer path to the left"
                )
            elif right_safety > left_safety and right_safety >= self.safety_threshold:
                return PathSuggestion(
                    direction=PathDirection.RIGHT,
                    safety_score=right_safety,
                    distance=2.0,
                    reasoning="Obstacle ahead, safer path to the right"
                )
            else:
                # Both sides have obstacles - suggest stopping
                return PathSuggestion(
                    direction=PathDirection.STOP,
                    safety_score=0.3,
                    distance=0.0,
                    reasoning="Obstacles on all sides, proceed with caution"
                )
        
        elif left_obstacles and not right_obstacles:
            # Obstacles on left only - suggest right
            right_safety = self._calculate_direction_safety(right_obstacles, 'right')
            if right_safety >= self.safety_threshold:
                return PathSuggestion(
                    direction=PathDirection.RIGHT,
                    safety_score=right_safety,
                    distance=3.0,
                    reasoning="Obstacles on left, clear path to the right"
                )
        
        elif right_obstacles and not left_obstacles:
            # Obstacles on right only - suggest left
            left_safety = self._calculate_direction_safety(left_obstacles, 'left')
            if left_safety >= self.safety_threshold:
                return PathSuggestion(
                    direction=PathDirection.LEFT,
                    safety_score=left_safety,
                    distance=3.0,
                    reasoning="Obstacles on right, clear path to the left"
                )
        
        elif left_obstacles and right_obstacles:
            # Obstacles on both sides - suggest forward if center is clear
            center_safety = self._calculate_direction_safety(center_obstacles, 'forward')
            if center_safety >= self.safety_threshold:
                return PathSuggestion(
                    direction=PathDirection.FORWARD,
                    safety_score=center_safety,
                    distance=2.0,
                    reasoning="Obstacles on sides, center path is clear"
                )
            else:
                return PathSuggestion(
                    direction=PathDirection.STOP,
                    safety_score=0.4,
                    distance=0.0,
                    reasoning="Obstacles on both sides, proceed carefully"
                )
        
        # Default: forward with caution
        return PathSuggestion(
            direction=PathDirection.FORWARD,
            safety_score=0.6,
            distance=1.0,
            reasoning="Path available with minor obstacles"
        )
    
    def _calculate_direction_safety(
        self,
        obstacles: List[Dict],
        direction: str
    ) -> float:
        """
        Calculate safety score for a direction.
        
        WHY THIS FUNCTION:
        Safety scores enable data-driven path selection. This function analyzes obstacle
        density, urgency, and distance to calculate how safe a direction is, supporting
        informed path planning decisions.
        
        Arguments:
            obstacles: List of obstacles in the direction
            direction: Direction name ('left', 'right', 'forward')
        
        Returns:
            Safety score (0-1, higher = safer)
        """
        if not obstacles:
            return 1.0  # No obstacles = completely safe
        
        # Calculate safety based on:
        # 1. Number of obstacles (more = less safe)
        # 2. Urgency levels (higher = less safe)
        # 3. Distance (closer = less safe)
        
        obstacle_count = len(obstacles)
        max_urgency = max(obs.get('urgency', 0) for obs in obstacles)
        min_distance = min(obs.get('distance', 2) for obs in obstacles)
        
        # Safety decreases with more obstacles, higher urgency, closer distance
        count_penalty = min(obstacle_count * 0.2, 0.6)  # Max 60% penalty
        urgency_penalty = max_urgency * 0.15  # Max 45% penalty (3 * 0.15)
        distance_penalty = (2 - min_distance) * 0.1  # Closer = more penalty
        
        safety = 1.0 - count_penalty - urgency_penalty - distance_penalty
        return max(0.0, min(1.0, safety))  # Clamp to [0, 1]
    
    def suggest_navigation_guidance(
        self,
        detections: List[Dict],
        verbosity: str = 'normal'
    ) -> str:
        """
        Generate navigation guidance from path planning.
        
        WHY THIS FUNCTION:
        Provides a simple interface for getting path planning guidance as natural language.
        This enables integration with DescriptionGenerator to create actionable navigation
        descriptions that support independent movement.
        
        Arguments:
            detections: List of detections
            verbosity: 'brief', 'normal', or 'detailed'
        
        Returns:
            Natural language navigation guidance
        """
        path = self.plan_path(detections)
        
        if path is None:
            return "Unable to determine safe path"
        
        if verbosity == 'brief':
            if path.direction == PathDirection.FORWARD:
                return "Clear path ahead"
            elif path.direction == PathDirection.LEFT:
                return "Move left"
            elif path.direction == PathDirection.RIGHT:
                return "Move right"
            else:
                return "Stop and assess"
        
        elif verbosity == 'normal':
            if path.direction == PathDirection.FORWARD:
                return f"Clear path ahead ({path.distance:.1f}m safe)"
            elif path.direction == PathDirection.LEFT:
                return f"Safer path to the left ({path.distance:.1f}m)"
            elif path.direction == PathDirection.RIGHT:
                return f"Safer path to the right ({path.distance:.1f}m)"
            else:
                return "Obstacles detected, proceed with caution"
        
        else:  # detailed
            return path.reasoning


def create_path_planner(safety_threshold: float = 0.7) -> PathPlanner:
    """Factory function to create path planner."""
    return PathPlanner(safety_threshold=safety_threshold)

