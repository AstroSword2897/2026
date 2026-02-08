"""Therapy Integration Module for MaxSight."""

from typing import Dict, List, Optional, Tuple
from enum import Enum
import torch


class TherapyTaskType(Enum):
    """Therapy task types that use scene descriptions."""
    ATTENTION_TRAINING = "attention"  # Focus on specific objects in scene.
    CONTRAST_RECOGNITION = "contrast"  # Identify objects with different contrast.
    EDGE_DETECTION = "edge"  # Identify edges and boundaries.
    SPATIAL_AWARENESS = "spatial"  # Understand spatial relationships.
    WARNING_RECOGNITION = "warning"  # Learn to recognize hazard cues over time.


class TherapyTaskIntegrator:
    """Integrates scene descriptions into therapy exercises."""
    
    def __init__(self):
        """Initialize therapy task integrator."""
        self.task_history = []
    
    def create_attention_task(
        self,
        scene_description: str,
        target_objects: List[str],
        difficulty: float = 0.5
    ) -> Dict:
        """Create attention training task from scene description."""
        return {
            'task_type': TherapyTaskType.ATTENTION_TRAINING,
            'scene_description': scene_description,
            'target_objects': target_objects,
            'difficulty': difficulty,
            'instructions': f"Focus on: {', '.join(target_objects)}",
            'duration': int(30 + (1.0 - difficulty) * 30)  # 30-60 seconds.
        }
    
    def create_contrast_task(
        self,
        scene_description: str,
        contrast_levels: List[float],
        difficulty: float = 0.5
    ) -> Dict:
        """Create contrast recognition task from scene description."""
        return {
            'task_type': TherapyTaskType.CONTRAST_RECOGNITION,
            'scene_description': scene_description,
            'contrast_levels': contrast_levels,
            'difficulty': difficulty,
            'instructions': "Identify objects with different contrast levels",
            'duration': int(30 + (1.0 - difficulty) * 30)
        }
    
    def create_edge_task(
        self,
        scene_description: str,
        edge_types: List[str],
        difficulty: float = 0.5
    ) -> Dict:
        """Create edge detection task from scene description."""
        return {
            'task_type': TherapyTaskType.EDGE_DETECTION,
            'scene_description': scene_description,
            'edge_types': edge_types,
            'difficulty': difficulty,
            'instructions': f"Identify edges: {', '.join(edge_types)}",
            'duration': int(30 + (1.0 - difficulty) * 30)
        }
    
    def create_spatial_task(
        self,
        scene_description: str,
        spatial_relationships: List[str],
        difficulty: float = 0.5
    ) -> Dict:
        """Create spatial awareness task from scene description."""
        return {
            'task_type': TherapyTaskType.SPATIAL_AWARENESS,
            'scene_description': scene_description,
            'spatial_relationships': spatial_relationships,
            'difficulty': difficulty,
            'instructions': f"Identify relationships: {', '.join(spatial_relationships)}",
            'duration': int(30 + (1.0 - difficulty) * 30)
        }
    
    def create_warning_recognition_task(
        self,
        hazard_type: str,
        urgency_level: int,
        cue_description: str,
        difficulty: float = 0.5
    ) -> Dict:
        """Create warning recognition task so the user learns to associate cues with hazards."""
        return {
            'task_type': TherapyTaskType.WARNING_RECOGNITION,
            'hazard_type': hazard_type,
            'urgency_level': urgency_level,
            'cue_description': cue_description,
            'difficulty': difficulty,
            'instructions': (
                f"Learn the cue for {hazard_type}. "
                f"You will hear/feel: {cue_description}. "
                "When you hear this in real use, it means this hazard is present."
            ),
            'duration': int(20 + (1.0 - difficulty) * 25)
        }
    
    def generate_task_from_scene(
        self,
        detections: List[Dict],
        scene_description: str,
        task_type: TherapyTaskType,
        difficulty: float = 0.5
    ) -> Dict:
        """Generate therapy task from scene detections and description."""
        if task_type == TherapyTaskType.ATTENTION_TRAINING:
            # Extract target objects from detections.
            target_objects = [d.get('class_name', 'object') for d in detections[:3]]
            return self.create_attention_task(scene_description, target_objects, difficulty)
        
        elif task_type == TherapyTaskType.CONTRAST_RECOGNITION:
            # Extract contrast levels from detections.
            contrast_levels = [d.get('contrast', 0.5) for d in detections if 'contrast' in d]
            if not contrast_levels:
                contrast_levels = [0.3, 0.5, 0.7]  # Default levels.
            return self.create_contrast_task(scene_description, contrast_levels, difficulty)
        
        elif task_type == TherapyTaskType.EDGE_DETECTION:
            # Extract edge types from detections.
            edge_types = ['door_edge', 'stair_edge', 'obstacle_edge']
            return self.create_edge_task(scene_description, edge_types, difficulty)
        
        elif task_type == TherapyTaskType.SPATIAL_AWARENESS:
            # Extract spatial relationships from detections.
            relationships = ['left_of', 'right_of', 'near', 'far']
            return self.create_spatial_task(scene_description, relationships, difficulty)
        
        elif task_type == TherapyTaskType.WARNING_RECOGNITION:
            # Use first high-urgency detection for warning recognition drill.
            hazard = next((d for d in detections if d.get('urgency', 0) >= 1), detections[0] if detections else {})
            hazard_type = hazard.get('class_name', 'obstacle')
            urgency_level = hazard.get('urgency', 1)
            return self.create_warning_recognition_task(
                hazard_type=hazard_type,
                urgency_level=urgency_level,
                cue_description=f"Alert for {hazard_type} (urgency {urgency_level})",
                difficulty=difficulty
            )
        
        else:
            # Default: attention task.
            target_objects = [d.get('class_name', 'object') for d in detections[:3]]
            return self.create_attention_task(scene_description, target_objects, difficulty)


def create_therapy_integrator() -> TherapyTaskIntegrator:
    """Factory function to create therapy task integrator."""
    return TherapyTaskIntegrator()







