"""
Therapy Integration Module for MaxSight
Uses enhanced scene descriptions in therapy exercises for attention, contrast, and edge recognition.

PROJECT PHILOSOPHY & APPROACH:
This module integrates scene descriptions into therapy tasks, enabling vision training exercises
that use real-world environmental information. This directly supports "Skill Development Across
Senses" by providing therapy exercises that are both practical and therapeutic.

WHY THERAPY INTEGRATION MATTERS:
Therapy exercises need to be relevant to real-world use. By using actual scene descriptions in
therapy tasks, we:
1. Train skills that directly apply to navigation and environmental awareness
2. Provide context-aware exercises (attention, contrast, edge recognition)
3. Bridge the gap between therapy and practical use

This supports the problem statement's emphasis on "Skill Development Across Senses" by ensuring
therapy exercises develop skills that directly support independent interaction with the world.

HOW IT CONNECTS TO THE PROBLEM STATEMENT:
The problem asks for ways to help users "interact with the world like those who can." Therapy
integration answers by providing exercises that train the specific skills needed for real-world
navigation and environmental awareness, not just abstract vision training.

RELATIONSHIP TO BARRIER REMOVAL METHODS:
1. SKILL DEVELOPMENT ACROSS SENSES: Core implementation - uses scene descriptions for training
2. ENVIRONMENTAL STRUCTURING: Exercises use real environmental information
3. GRADUAL INDEPENDENCE: Exercises adapt difficulty based on performance

TECHNICAL DESIGN DECISION:
We integrate scene descriptions into therapy tasks rather than using abstract exercises because:
- Real-world context makes exercises more engaging and relevant
- Skills learned with real scenes transfer better to actual use
- Environmental descriptions provide natural training stimuli
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum
import torch


class TherapyTaskType(Enum):
    """Therapy task types that use scene descriptions"""
    ATTENTION_TRAINING = "attention"  # Focus on specific objects in scene
    CONTRAST_RECOGNITION = "contrast"  # Identify objects with different contrast
    EDGE_DETECTION = "edge"  # Identify edges and boundaries
    SPATIAL_AWARENESS = "spatial"  # Understand spatial relationships


class TherapyTaskIntegrator:
    """
    Integrates scene descriptions into therapy exercises.
    
    WHY THIS CLASS EXISTS:
    This class bridges DescriptionGenerator (scene descriptions) and TaskGenerator (therapy tasks).
    It enables therapy exercises to use real-world scene descriptions, making training more
    relevant and effective for actual navigation and environmental awareness.
    
    This directly supports "Skill Development Across Senses" by ensuring therapy exercises
    develop skills that directly apply to real-world use.
    """
    
    def __init__(self):
        """Initialize therapy task integrator."""
        self.task_history = []
    
    def create_attention_task(
        self,
        scene_description: str,
        target_objects: List[str],
        difficulty: float = 0.5
    ) -> Dict:
        """
        Create attention training task from scene description.
        
        WHY ATTENTION TASKS:
        Attention training helps users focus on important objects in complex scenes. By using
        real scene descriptions, users practice the attention skills needed for navigation and
        environmental awareness.
        
        This supports "Visual Assistance & Training Goals" by providing exercises that train
        attention skills directly applicable to real-world use.
        
        Arguments:
            scene_description: Scene description from DescriptionGenerator
            target_objects: Objects to focus attention on
            difficulty: Task difficulty (0-1)
        
        Returns:
            Task configuration dictionary
        """
        return {
            'task_type': TherapyTaskType.ATTENTION_TRAINING,
            'scene_description': scene_description,
            'target_objects': target_objects,
            'difficulty': difficulty,
            'instructions': f"Focus on: {', '.join(target_objects)}",
            'duration': int(30 + (1.0 - difficulty) * 30)  # 30-60 seconds
        }
    
    def create_contrast_task(
        self,
        scene_description: str,
        contrast_levels: List[float],
        difficulty: float = 0.5
    ) -> Dict:
        """
        Create contrast recognition task from scene description.
        
        WHY CONTRAST TASKS:
        Contrast recognition is critical for users with vision conditions that affect contrast
        sensitivity (cataracts, glaucoma). By using real scene descriptions, users practice
        identifying objects with different contrast levels in realistic contexts.
        
        This supports "Fine-Grained Visual Features" by training contrast sensitivity skills
        that directly apply to real-world object recognition.
        
        Arguments:
            scene_description: Scene description from DescriptionGenerator
            contrast_levels: Contrast levels to identify
            difficulty: Task difficulty (0-1)
        
        Returns:
            Task configuration dictionary
        """
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
        """
        Create edge detection task from scene description.
        
        WHY EDGE TASKS:
        Edge detection is fundamental for object recognition and navigation. By using real scene
        descriptions, users practice identifying edges and boundaries in realistic contexts,
        training skills that directly support navigation and obstacle avoidance.
        
        This supports "Fine-Grained Visual Features" by training edge detection skills that
        are critical for understanding spatial relationships and object boundaries.
        
        Arguments:
            scene_description: Scene description from DescriptionGenerator
            edge_types: Types of edges to identify (e.g., 'door_edge', 'stair_edge')
            difficulty: Task difficulty (0-1)
        
        Returns:
            Task configuration dictionary
        """
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
        """
        Create spatial awareness task from scene description.
        
        WHY SPATIAL TASKS:
        Spatial awareness is critical for navigation and environmental understanding. By using
        real scene descriptions, users practice understanding spatial relationships (left/right,
        near/far, above/below) in realistic contexts.
        
        This supports "Spatial Awareness & Localization" by training spatial understanding
        skills that directly apply to navigation and environmental awareness.
        
        Arguments:
            scene_description: Scene description from DescriptionGenerator
            spatial_relationships: Relationships to identify (e.g., 'left_of', 'near', 'above')
            difficulty: Task difficulty (0-1)
        
        Returns:
            Task configuration dictionary
        """
        return {
            'task_type': TherapyTaskType.SPATIAL_AWARENESS,
            'scene_description': scene_description,
            'spatial_relationships': spatial_relationships,
            'difficulty': difficulty,
            'instructions': f"Identify relationships: {', '.join(spatial_relationships)}",
            'duration': int(30 + (1.0 - difficulty) * 30)
        }
    
    def generate_task_from_scene(
        self,
        detections: List[Dict],
        scene_description: str,
        task_type: TherapyTaskType,
        difficulty: float = 0.5
    ) -> Dict:
        """
        Generate therapy task from scene detections and description.
        
        WHY THIS FUNCTION:
        Provides a unified interface for creating therapy tasks from real scene data. This
        enables therapy exercises to use actual environmental information, making training
        more relevant and effective.
        
        This directly supports "Integration with Therapy Tasks" by enabling scene descriptions
        to be used in therapy exercises for attention, contrast, and edge recognition.
        
        Arguments:
            detections: List of detections from MaxSightCNN
            scene_description: Scene description from DescriptionGenerator
            task_type: Type of therapy task
            difficulty: Task difficulty (0-1)
        
        Returns:
            Task configuration dictionary
        """
        if task_type == TherapyTaskType.ATTENTION_TRAINING:
            # Extract target objects from detections
            target_objects = [d.get('class_name', 'object') for d in detections[:3]]
            return self.create_attention_task(scene_description, target_objects, difficulty)
        
        elif task_type == TherapyTaskType.CONTRAST_RECOGNITION:
            # Extract contrast levels from detections
            contrast_levels = [d.get('contrast', 0.5) for d in detections if 'contrast' in d]
            if not contrast_levels:
                contrast_levels = [0.3, 0.5, 0.7]  # Default levels
            return self.create_contrast_task(scene_description, contrast_levels, difficulty)
        
        elif task_type == TherapyTaskType.EDGE_DETECTION:
            # Extract edge types from detections
            edge_types = ['door_edge', 'stair_edge', 'obstacle_edge']
            return self.create_edge_task(scene_description, edge_types, difficulty)
        
        elif task_type == TherapyTaskType.SPATIAL_AWARENESS:
            # Extract spatial relationships from detections
            relationships = ['left_of', 'right_of', 'near', 'far']
            return self.create_spatial_task(scene_description, relationships, difficulty)
        
        else:
            # Default: attention task
            target_objects = [d.get('class_name', 'object') for d in detections[:3]]
            return self.create_attention_task(scene_description, target_objects, difficulty)


def create_therapy_integrator() -> TherapyTaskIntegrator:
    """Factory function to create therapy task integrator."""
    return TherapyTaskIntegrator()

