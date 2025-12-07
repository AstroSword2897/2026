"""
Enhanced Description Generator for MaxSight
Generates natural, actionable descriptions with direction, distance, and context.

PROJECT PHILOSOPHY & APPROACH:
=============================
This module is central to MaxSight's core mission: "Removing Barriers for Vision & Hearing Disabilities."

WHY THIS APPROACH:
People with vision impairments need information about their environment in a format they can process.
Raw bounding boxes and class names are meaningless - users need actionable, spatial descriptions that
help them navigate and understand their surroundings. This module transforms technical ML outputs into
human-understandable language that directly supports independent navigation.

HOW IT CONNECTS TO THE PROBLEM STATEMENT:
The problem statement asks: "What are ways that those who cannot see and hear be able to interact 
with the world like those who can?" This module answers that by providing:

1. ENVIRONMENTAL STRUCTURING (Barrier Removal Method #1):
   - Converts raw detections into structured descriptions: "Door 2 meters ahead, handle left"
   - Labels surroundings in ways users can understand and act upon
   - Provides spatial context (distance, direction, height) that sighted people take for granted

2. CLEAR MULTIMODAL COMMUNICATION (Barrier Removal Method #2):
   - Generates descriptions suitable for TTS (text-to-speech) for blind users
   - Creates concise summaries for visual overlays for deaf users
   - Adapts verbosity based on user needs (brief/normal/detailed)

3. SKILL DEVELOPMENT ACROSS SENSES (Barrier Removal Method #3):
   - Provides consistent spatial language that helps users build mental maps
   - Gradually reduces detail as users improve (via verbosity levels)
   - Reinforces spatial understanding through repeated, structured descriptions

HOW IT CONTRIBUTES TO VISUAL AWARENESS GOALS:
This directly implements the "Environmental Awareness Goals" and "Spatial Awareness & Localization"
from the comprehensive requirements. It transforms the CNN's technical outputs (bounding boxes, 
class probabilities) into the kind of information that helps users:
- Understand where objects are relative to them (not just "door detected")
- Navigate safely ("Obstacle on left, move right")
- Build cognitive maps of their environment over time

RELATIONSHIP TO OTHER COMPONENTS:
- Input: Receives detections from MaxSightCNN (object positions, classes, distances)
- Output: Feeds into CrossModalScheduler for TTS/visual/haptic presentation
- Integration: Works with SpatialMemory to provide contextual reminders
- User Experience: Enables the "Reads Environment" MVP feature ("User points phone → app says: 
  'Door 2 meters ahead, handle left'")

TECHNICAL DESIGN DECISION:
We use verbosity levels (brief/normal/detailed) rather than a single format because:
- Users with different vision conditions need different levels of detail
- CVI (Cortical Visual Impairment) users benefit from simplified, consistent formats
- Advanced users can reduce verbosity as their skills improve (gradual independence)
- This supports the "Routine Workflow" barrier removal method by adapting to user needs
"""

import torch
from typing import Dict, List, Optional, Tuple
import math


class DescriptionGenerator:
    """
    Generates natural language descriptions from model outputs.
    Provides directional cues, distance estimates, and contextual information.
    
    WHY THIS CLASS EXISTS:
    The CNN model outputs technical data (bounding boxes, class probabilities, distance zones).
    Users with vision impairments need this translated into actionable language. This class bridges
    that gap by converting technical ML outputs into natural descriptions that support independent
    navigation and environmental understanding.
    
    DESIGN PHILOSOPHY:
    Every description is designed to answer: "Where is it, how far, and what should I do?"
    This aligns with the project's focus on practical usability - descriptions must be immediately
    actionable, not just informative.
    """
    
    # Distance zone names
    DISTANCE_NAMES = ['near', 'medium', 'far']
    
    # Direction zones (relative to image center)
    DIRECTION_ZONES = {
        'left': (-1.0, -0.33),
        'center_left': (-0.33, -0.1),
        'center': (-0.1, 0.1),
        'center_right': (0.1, 0.33),
        'right': (0.33, 1.0),
        'top': (-1.0, -0.33),  # For vertical (y-axis)
        'center_top': (-0.33, -0.1),
        'center_bottom': (0.1, 0.33),
        'bottom': (0.33, 1.0)
    }
    
    # Urgency level names
    URGENCY_NAMES = ['safe', 'caution', 'warning', 'danger']
    
    def __init__(self, verbosity: str = 'normal'):
        """
        Initialize description generator.
        
        Arguments:
            verbosity: 'brief', 'normal', or 'detailed'
        """
        self.verbosity = verbosity
    
    def get_direction_from_box(
        self, 
        box: torch.Tensor, 
        image_size: Tuple[int, int] = (224, 224)
    ) -> Tuple[str, str]:
        """
        Get horizontal and vertical direction from bounding box center.
        
        WHY THIS MATTERS:
        Directional cues are critical for navigation. A sighted person can see "door on the left"
        instantly, but a blind user needs this explicitly stated. This function converts the camera's
        perspective (bounding box position) into user-relative directions that support safe navigation.
        
        This directly addresses the "Spatial Awareness & Localization" goal: helping users understand
        where things are in 3D space relative to them. Without direction, "door detected" is useless
        for navigation - "door slightly left" is actionable.
        
        Arguments:
            box: [cx, cy, w, h] in normalized coordinates [0, 1]
            image_size: (width, height) of image
        
        Returns:
            (horizontal_direction, vertical_direction)
        """
        cx, cy = box[0].item(), box[1].item()
        
        # Horizontal direction (x-axis)
        if cx < 0.33:
            h_dir = 'left'
        elif cx < 0.4:
            h_dir = 'slightly left'
        elif cx < 0.6:
            h_dir = 'ahead'
        elif cx < 0.67:
            h_dir = 'slightly right'
        else:
            h_dir = 'right'
        
        # Vertical direction (y-axis) - top is negative, bottom is positive
        if cy < 0.33:
            v_dir = 'above'
        elif cy < 0.4:
            v_dir = 'slightly above'
        elif cy < 0.6:
            v_dir = 'at eye level'
        elif cy < 0.67:
            v_dir = 'slightly below'
        else:
            v_dir = 'below'
        
        return h_dir, v_dir
    
    def get_distance_description(
        self, 
        distance_zone: int, 
        box_size: Optional[float] = None
    ) -> str:
        """
        Get distance description from zone and optional size.
        
        Arguments:
            distance_zone: 0 (near), 1 (medium), 2 (far)
            box_size: Optional box area for more precise estimation
        
        Returns:
            Distance description string
        """
        if distance_zone < 0 or distance_zone >= len(self.DISTANCE_NAMES):
            distance_zone = 1  # Default to medium
        
        base_name = self.DISTANCE_NAMES[distance_zone]
        
        # Add more precise estimates if box_size available
        if box_size is not None and self.verbosity in ['normal', 'detailed']:
            if distance_zone == 0:  # Near
                if box_size > 0.15:
                    return "very close"
                elif box_size > 0.08:
                    return "close"
                else:
                    return "near"
            elif distance_zone == 1:  # Medium
                if box_size > 0.05:
                    return "moderate distance"
                else:
                    return "medium distance"
            else:  # Far
                if box_size < 0.02:
                    return "far away"
                else:
                    return "distant"
        
        return base_name
    
    def estimate_meters(
        self, 
        distance_zone: int, 
        box_size: float,
        object_type: str = 'object'
    ) -> Optional[str]:
        """
        Estimate distance in meters (rough approximation).
        
        Arguments:
            distance_zone: 0 (near), 1 (medium), 2 (far)
            box_size: Box area (normalized)
            object_type: Type of object for size reference
        
        Returns:
            Estimated meters string or None if not available
        """
        if self.verbosity != 'detailed':
            return None
        
        # Rough size-based estimation
        # Larger boxes = closer objects
        if distance_zone == 0:  # Near
            if box_size > 0.15:
                return "1-2 meters"
            elif box_size > 0.08:
                return "2-3 meters"
            else:
                return "3-5 meters"
        elif distance_zone == 1:  # Medium
            if box_size > 0.05:
                return "5-7 meters"
            else:
                return "7-10 meters"
        else:  # Far
            if box_size < 0.02:
                return "10+ meters"
            else:
                return "8-12 meters"
    
    def get_relative_height(
        self, 
        box: torch.Tensor,
        image_size: Tuple[int, int] = (224, 224)
    ) -> str:
        """
        Get relative height description.
        
        Arguments:
            box: [cx, cy, w, h] in normalized coordinates
            image_size: (width, height)
        
        Returns:
            Height description
        """
        if self.verbosity != 'detailed':
            return ""
        
        cy, h = box[1].item(), box[3].item()
        
        # Vertical position
        if cy < 0.3:
            position = "overhead"
        elif cy < 0.4:
            position = "above eye level"
        elif cy < 0.6:
            position = "at eye level"
        elif cy < 0.7:
            position = "below eye level"
        else:
            position = "near ground level"
        
        # Size relative to typical object
        if h > 0.3:
            size_desc = "large"
        elif h > 0.15:
            size_desc = "medium-sized"
        else:
            size_desc = "small"
        
        return f"{size_desc}, {position}"
    
    def generate_object_description(
        self,
        class_name: str,
        box: torch.Tensor,
        distance_zone: int,
        urgency: int = 0,
        priority: Optional[int] = None,
        verbosity: Optional[str] = None
    ) -> str:
        """
        Generate natural language description for a single object.
        
        CORE FUNCTION - WHY THIS EXISTS:
        This is the heart of MaxSight's "Environmental Structuring" approach. It transforms a technical
        detection (class="door", box=[0.3, 0.5, 0.2, 0.3], distance=0) into actionable information:
        "Door 2 meters ahead, slightly left, at eye level".
        
        This directly implements the MVP feature: "User points phone → app says: 'Door 2 meters ahead,
        handle left' or 'Stop sign.'" Without this transformation, users get raw technical data they
        cannot act upon.
        
        HOW IT SUPPORTS DIFFERENT VISION CONDITIONS:
        - Brief mode: For users who need minimal information (CVI, cognitive overload)
        - Normal mode: Standard actionable descriptions (most users)
        - Detailed mode: For users learning spatial relationships or needing full context
        
        This adaptive verbosity supports "Skill Development Across Senses" - users can start with
        detailed descriptions and gradually reduce to brief as they build spatial awareness.
        
        RELATIONSHIP TO SAFETY:
        Urgency levels are prominently featured because safety is paramount. A "door" is different
        from a "vehicle approaching" - this function ensures hazards are clearly communicated,
        supporting the "Safety-Oriented Visual Awareness" goal.
        
        Implements: "Stairs 3 meters ahead, slightly left"
        
        Arguments:
            class_name: Object class name
            box: [cx, cy, w, h] normalized bounding box
            distance_zone: 0 (near), 1 (medium), 2 (far)
            urgency: Urgency level (0-3)
            priority: Optional priority score (0-100)
            verbosity: Override default verbosity
        
        Returns:
            Natural language description
        """
        verbosity = verbosity or self.verbosity
        
        # Get direction
        h_dir, v_dir = self.get_direction_from_box(box)
        
        # Get distance
        box_size = box[2].item() * box[3].item()  # w * h
        distance_desc = self.get_distance_description(distance_zone, box_size)
        
        # Get meters estimate if detailed
        meters = self.estimate_meters(distance_zone, box_size, class_name)
        
        # Get height if detailed
        height_desc = self.get_relative_height(box)
        
        # Build description based on verbosity
        if verbosity == 'brief':
            # Minimal: "Door ahead" or "Stairs left"
            if h_dir == 'ahead':
                return f"{class_name} ahead"
            else:
                return f"{class_name} {h_dir}"
        
        elif verbosity == 'normal':
            # Standard: "Door 2 meters ahead, slightly left"
            parts = [class_name]
            
            # Add distance
            if meters:
                parts.append(meters)
            else:
                parts.append(distance_desc)
            
            # Add direction
            if h_dir != 'ahead':
                parts.append(h_dir)
            
            # Add urgency if high
            if urgency >= 2:
                urgency_name = self.URGENCY_NAMES[urgency] if urgency < len(self.URGENCY_NAMES) else 'warning'
                parts.append(f"({urgency_name})")
            
            return " ".join(parts)
        
        else:  # detailed
            # Full: "Wooden door, 2 meters ahead, slightly left, at eye level, brass handle on left side"
            parts = [class_name]
            
            # Add distance with meters
            if meters:
                parts.append(f"{meters} away")
            else:
                parts.append(f"at {distance_desc}")
            
            # Add direction
            if h_dir != 'ahead':
                parts.append(h_dir)
            if v_dir != 'at eye level' and v_dir != 'ahead':
                parts.append(v_dir)
            
            # Add height if available
            if height_desc:
                parts.append(f"({height_desc})")
            
            # Add urgency
            if urgency >= 1:
                urgency_name = self.URGENCY_NAMES[urgency] if urgency < len(self.URGENCY_NAMES) else 'caution'
                parts.append(f"- {urgency_name}")
            
            return ", ".join(parts)
    
    def generate_scene_description(
        self,
        detections: List[Dict],
        urgency_score: int = 0,
        verbosity: Optional[str] = None
    ) -> str:
        """
        Generate overall scene description from multiple detections.
        
        Implements: "Three people approaching from left, vehicle approaching right"
        
        Arguments:
            detections: List of detection dictionaries with class_name, box, distance, urgency
            urgency_score: Overall scene urgency
            verbosity: Override default verbosity
        
        Returns:
            Scene description string
        """
        verbosity = verbosity or self.verbosity
        
        if not detections:
            return "No objects detected"
        
        # Sort by priority/urgency (highest first)
        sorted_dets = sorted(
            detections,
            key=lambda d: (d.get('urgency', 0), d.get('priority', 0)),
            reverse=True
        )
        
        # Limit number of objects based on verbosity
        if verbosity == 'brief':
            max_objects = 2
        elif verbosity == 'normal':
            max_objects = 4
        else:  # detailed
            max_objects = 6
        
        objects_to_describe = sorted_dets[:max_objects]
        
        # Group by direction for better organization
        descriptions = []
        for det in objects_to_describe:
            class_name = det.get('class_name', 'object')
            box = det.get('box')
            distance_zone = det.get('distance', 1)
            urgency = det.get('urgency', 0)
            priority = det.get('priority', 0)
            
            if box is not None:
                box_tensor = torch.tensor(box) if not isinstance(box, torch.Tensor) else box
                desc = self.generate_object_description(
                    class_name, box_tensor, distance_zone, urgency, priority, verbosity
                )
                descriptions.append(desc)
            else:
                descriptions.append(class_name)
        
        # Combine descriptions
        if verbosity == 'brief':
            return "; ".join(descriptions)
        elif verbosity == 'normal':
            return ". ".join(descriptions) + "."
        else:  # detailed
            # Add scene context
            urgency_name = self.URGENCY_NAMES[urgency_score] if urgency_score < len(self.URGENCY_NAMES) else 'normal'
            scene_context = f"Scene: {len(detections)} objects detected. Overall safety: {urgency_name}."
            return scene_context + " " + ". ".join(descriptions) + "."
    
    def generate_navigation_guidance(
        self,
        detections: List[Dict],
        target_direction: Optional[str] = None
    ) -> str:
        """
        Generate navigation guidance with path suggestions.
        
        WHY NAVIGATION GUIDANCE IS CRITICAL:
        This function directly addresses the core problem: helping users navigate safely when they
        cannot see obstacles. A sighted person can instantly see "obstacle on left, clear path right"
        - this function provides that same information through language.
        
        This is not just about detecting objects - it's about providing actionable navigation advice
        that prevents collisions, falls, and disorientation. This supports the "Safety-Oriented Visual
        Awareness" goal and is essential for independent mobility.
        
        HOW IT CONNECTS TO THE PROBLEM STATEMENT:
        The problem asks: "What are ways that those who cannot see... be able to interact with the
        world like those who can?" Navigation guidance is a direct answer - it provides the spatial
        awareness that sighted people take for granted, enabling safe, independent movement.
        
        RELATIONSHIP TO OTHER FEATURES:
        - Works with urgency scoring to prioritize hazards
        - Integrates with distance estimation to focus on immediate obstacles
        - Feeds into CrossModalScheduler for haptic/audio alerts
        - Supports the "Navigation Assistance" feature from Sprint 3
        
        Implements: "Clear path ahead" or "Obstacle on left, move right"
        
        Arguments:
            detections: List of detections
            target_direction: Optional target direction ('forward', 'left', 'right')
        
        Returns:
            Navigation guidance string
        """
        if not detections:
            return "Clear path ahead"
        
        # Filter for obstacles (high urgency, near objects)
        obstacles = [
            d for d in detections
            if d.get('urgency', 0) >= 2 and d.get('distance', 2) <= 1
        ]
        
        if not obstacles:
            return "Clear path ahead"
        
        # Analyze obstacle positions
        left_obstacles = []
        right_obstacles = []
        center_obstacles = []
        
        for obs in obstacles:
            box = obs.get('box')
            if box is not None:
                cx = box[0] if isinstance(box, (list, tuple)) else box[0].item()
                if cx < 0.4:
                    left_obstacles.append(obs)
                elif cx > 0.6:
                    right_obstacles.append(obs)
                else:
                    center_obstacles.append(obs)
        
        # Generate guidance
        if center_obstacles:
            return "Obstacle directly ahead, move left or right"
        elif left_obstacles and not right_obstacles:
            return "Obstacle on left, move right"
        elif right_obstacles and not left_obstacles:
            return "Obstacle on right, move left"
        elif left_obstacles and right_obstacles:
            return "Obstacles on both sides, proceed with caution"
        else:
            return "Path clear with minor obstacles"
    
    def generate_hazard_alert(
        self,
        class_name: str,
        box: torch.Tensor,
        distance_zone: int,
        urgency: int
    ) -> str:
        """
        Generate urgent hazard alert.
        
        Implements: "Warning: Vehicle approaching from right"
        
        Arguments:
            class_name: Object class
            box: Bounding box
            distance_zone: Distance zone
            urgency: Urgency level
        
        Returns:
            Alert string
        """
        h_dir, _ = self.get_direction_from_box(box)
        distance_desc = self.get_distance_description(distance_zone)
        
        urgency_word = self.URGENCY_NAMES[urgency] if urgency < len(self.URGENCY_NAMES) else 'warning'
        
        if urgency >= 3:  # Danger
            return f"DANGER: {class_name} {distance_desc} {h_dir}"
        elif urgency >= 2:  # Warning
            return f"Warning: {class_name} {distance_desc} {h_dir}"
        else:  # Caution
            return f"Caution: {class_name} {distance_desc} {h_dir}"


def create_description_generator(verbosity: str = 'normal') -> DescriptionGenerator:
    """Factory function to create description generator."""
    return DescriptionGenerator(verbosity=verbosity)

