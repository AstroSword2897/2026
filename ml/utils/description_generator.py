"""Enhanced Description Generator for MaxSight Generates natural, actionable descriptions with direction, distance, and context."""

import torch
from typing import Dict, List, Optional, Tuple
import math


class DescriptionGenerator:
    """Generates natural language descriptions from model outputs."""
    
    # Distance zone names.
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
    
    # Urgency level names.
    URGENCY_NAMES = ['safe', 'caution', 'warning', 'danger']
    
    def __init__(self, verbosity: str = 'normal'):
        """Initialize description generator. Arguments: verbosity: 'brief', 'normal', or 'detailed'"""
        self.verbosity = verbosity
    
    def get_direction_from_box(
        self, 
        box: torch.Tensor, 
        image_size: Tuple[int, int] = (224, 224)
    ) -> Tuple[str, str]:
        """Get horizontal and vertical direction from bounding box center."""
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
        
        # Vertical direction (y-axis) - top is negative, bottom is positive.
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
        """Get distance description from zone and optional size."""
        # Handle string distance zones (convert to int)
        if isinstance(distance_zone, str):
            distance_map = {'near': 0, 'medium': 1, 'far': 2}
            distance_zone = distance_map.get(distance_zone.lower(), 1)
        
        # Ensure distance_zone is int and in valid range.
        distance_zone = int(distance_zone)
        if distance_zone < 0 or distance_zone >= len(self.DISTANCE_NAMES):
            distance_zone = 1  # Default to medium.
        
        base_name = self.DISTANCE_NAMES[distance_zone]
        
        # Add more precise estimates if box_size available.
        if box_size is not None and self.verbosity in ['normal', 'detailed']:
            if distance_zone == 0:  # Near.
                if box_size > 0.15:
                    return "very close"
                elif box_size > 0.08:
                    return "close"
                else:
                    return "near"
            elif distance_zone == 1:  # Medium.
                if box_size > 0.05:
                    return "moderate distance"
                else:
                    return "medium distance"
            else:  # Far.
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
        """Estimate distance in meters (rough approximation)."""
        if self.verbosity != 'detailed':
            return None
        
        # Rough size-based estimation. Larger boxes = closer objects.
        if distance_zone == 0:  # Near.
            if box_size > 0.15:
                return "1-2 meters"
            elif box_size > 0.08:
                return "2-3 meters"
            else:
                return "3-5 meters"
        elif distance_zone == 1:  # Medium.
            if box_size > 0.05:
                return "5-7 meters"
            else:
                return "7-10 meters"
        else:  # Far.
            if box_size < 0.02:
                return "10+ meters"
            else:
                return "8-12 meters"
    
    def get_relative_height(
        self, 
        box: torch.Tensor,
        image_size: Tuple[int, int] = (224, 224)
    ) -> str:
        """Get relative height description. Arguments: box: [cx, cy, w, h] in normalized coordinates image_size: (width, height) Returns: Height description."""
        if self.verbosity != 'detailed':
            return ""
        
        cy, h = box[1].item(), box[3].item()
        
        # Vertical position.
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
        
        # Size relative to typical object.
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
        # Handle string distance zones (convert to int)
        if isinstance(distance_zone, str):
            distance_map = {'near': 0, 'medium': 1, 'far': 2}
            distance_zone = distance_map.get(distance_zone.lower(), 1)
        distance_zone = int(distance_zone)
        """Generate natural language description for a single object."""
        verbosity = verbosity or self.verbosity
        
        # Get direction.
        h_dir, v_dir = self.get_direction_from_box(box)
        
        # Get distance.
        box_size = box[2].item() * box[3].item()  # W * h.
        distance_desc = self.get_distance_description(distance_zone, box_size)
        
        # Get meters estimate if detailed.
        meters = self.estimate_meters(distance_zone, box_size, class_name)
        
        # Get height if detailed.
        height_desc = self.get_relative_height(box)
        
        # Build description based on verbosity.
        if verbosity == 'brief':
            # Minimal: "Door ahead" or "Stairs left"
            if h_dir == 'ahead':
                return f"{class_name} ahead"
            else:
                return f"{class_name} {h_dir}"
        
        elif verbosity == 'normal':
            # Standard: "Door 2 meters ahead, slightly left"
            parts = [class_name]
            
            # Add distance.
            if meters:
                parts.append(meters)
            else:
                parts.append(distance_desc)
            
            # Add direction.
            if h_dir != 'ahead':
                parts.append(h_dir)
            
            # Add urgency if high.
            if urgency >= 2:
                urgency_name = self.URGENCY_NAMES[urgency] if urgency < len(self.URGENCY_NAMES) else 'warning'
                parts.append(f"({urgency_name})")
            
            return " ".join(parts)
        
        else:  # Detailed.
            # Full: "Wooden door, 2 meters ahead, slightly left, at eye level, brass handle on left side"
            parts = [class_name]
            
            # Add distance with meters.
            if meters:
                parts.append(f"{meters} away")
            else:
                parts.append(f"at {distance_desc}")
            
            # Add direction.
            if h_dir != 'ahead':
                parts.append(h_dir)
            if v_dir != 'at eye level' and v_dir != 'ahead':
                parts.append(v_dir)
            
            # Add height if available.
            if height_desc:
                parts.append(f"({height_desc})")
            
            # Add urgency.
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
        """Generate overall scene description from multiple detections."""
        verbosity = verbosity or self.verbosity
        
        if not detections:
            return "No objects detected"
        
        # Sort by priority/urgency (highest first)
        sorted_dets = sorted(
            detections,
            key=lambda d: (d.get('urgency', 0), d.get('priority', 0)),
            reverse=True
        )
        
        # Limit number of objects based on verbosity.
        if verbosity == 'brief':
            max_objects = 2
        elif verbosity == 'normal':
            max_objects = 4
        else:  # Detailed.
            max_objects = 6
        
        objects_to_describe = sorted_dets[:max_objects]
        
        # Group by direction for better organization.
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
        
        # Combine descriptions.
        if verbosity == 'brief':
            return "; ".join(descriptions)
        elif verbosity == 'normal':
            return ". ".join(descriptions) + "."
        else:  # Detailed.
            # Add scene context.
            urgency_name = self.URGENCY_NAMES[urgency_score] if urgency_score < len(self.URGENCY_NAMES) else 'normal'
            scene_context = f"Scene: {len(detections)} objects detected. Overall safety: {urgency_name}."
            return scene_context + " " + ". ".join(descriptions) + "."
    
    def generate_navigation_guidance(
        self,
        detections: List[Dict],
        target_direction: Optional[str] = None
    ) -> str:
        """Generate navigation guidance with path suggestions."""
        if not detections:
            return "Clear path ahead"
        
        # Filter for obstacles (high urgency, near objects) Handle distance as either int (zone) or string (name)
        obstacles = []
        for d in detections:
            urgency = d.get('urgency', 0)
            distance = d.get('distance', 2)
            # Convert string distance to zone if needed.
            if isinstance(distance, str):
                distance_map = {'near': 0, 'medium': 1, 'far': 2}
                distance = distance_map.get(distance.lower(), 1)
            if urgency >= 2 and distance <= 1:
                obstacles.append(d)
        
        if not obstacles:
            return "Clear path ahead"
        
        # Analyze obstacle positions.
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
        
        # Generate guidance.
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
        """Generate urgent hazard alert."""
        h_dir, _ = self.get_direction_from_box(box)
        distance_desc = self.get_distance_description(distance_zone)
        
        urgency_word = self.URGENCY_NAMES[urgency] if urgency < len(self.URGENCY_NAMES) else 'warning'
        
        if urgency >= 3:  # Danger.
            return f"DANGER: {class_name} {distance_desc} {h_dir}"
        elif urgency >= 2:  # Warning.
            return f"Warning: {class_name} {distance_desc} {h_dir}"
        else:  # Caution.
            return f"Caution: {class_name} {distance_desc} {h_dir}"
    
    def generate_description(
        self,
        detections: List[Dict],
        urgency_score: int = 0,
        verbosity: Optional[str] = None
    ) -> str:
        """Generate description from detections (wrapper for generate_scene_description)."""
        return self.generate_scene_description(detections, urgency_score, verbosity)


def create_description_generator(verbosity: str = 'normal') -> DescriptionGenerator:
    """Factory function to create description generator."""
    return DescriptionGenerator(verbosity=verbosity)







