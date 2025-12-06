"""
Semantic Grouping Module for MaxSight
Groups related objects into semantic clusters for concise scene descriptions.

PROJECT PHILOSOPHY & APPROACH:
=============================
This module implements "Semantic Grouping" - combining related objects into single descriptions
for brevity and clarity. This directly addresses the "Clear, Concise Outputs" goal by preventing
information overload while maintaining useful information.

WHY SEMANTIC GROUPING MATTERS:
Without grouping, users hear: "Chair. Chair. Chair. Table. Chair. Table." This is overwhelming
and unhelpful. With semantic grouping: "Three chairs and two tables" - concise and actionable.

This supports "Environmental Structuring" by organizing information in ways that match how humans
naturally understand scenes - not as individual objects, but as semantic groups (furniture, people,
vehicles, etc.).

HOW IT CONNECTS TO THE PROBLEM STATEMENT:
The problem emphasizes "Clear Multimodal Communication" - this module ensures information is
presented clearly and concisely, preventing cognitive overload while maintaining usefulness.
This is especially important for users with CVI (Cortical Visual Impairment) who benefit from
simplified, structured information.

RELATIONSHIP TO BARRIER REMOVAL METHODS:
1. ENVIRONMENTAL STRUCTURING: Groups objects semantically for clearer understanding
2. CLEAR MULTIMODAL COMMUNICATION: Reduces information density while maintaining clarity
3. SKILL DEVELOPMENT: Helps users learn to recognize object groups, not just individual items

TECHNICAL DESIGN DECISION:
We group by:
- Object class (same type = same group)
- Spatial proximity (nearby objects = same group)
- Semantic category (furniture, people, vehicles, etc.)

This multi-level grouping ensures descriptions are both concise and meaningful.
"""

import torch
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np


class SemanticGrouper:
    """
    Groups related objects semantically for concise scene descriptions.
    
    WHY THIS CLASS EXISTS:
    Individual object descriptions can be overwhelming. This class groups related objects
    (e.g., "three chairs", "clustered furniture") to create concise, actionable descriptions
    that support "Clear Multimodal Communication" without information overload.
    """
    
    # Semantic categories for grouping
    SEMANTIC_CATEGORIES = {
        'furniture': ['chair', 'table', 'couch', 'bed', 'desk', 'bench', 'stool'],
        'people': ['person', 'people', 'crowd'],
        'vehicles': ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'vehicle'],
        'doors': ['door', 'door_open', 'door_closed', 'sliding_door', 'automatic_door'],
        'stairs': ['stairs', 'staircase', 'stairway', 'escalator', 'ramp'],
        'signs': ['stop_sign', 'exit_sign', 'information_sign', 'warning_sign'],
        'obstacles': ['fire_hydrant', 'traffic_cone', 'barrier', 'fence'],
        'appliances': ['refrigerator', 'microwave', 'oven', 'sink', 'toaster'],
        'electronics': ['tv', 'laptop', 'cell_phone', 'keyboard', 'mouse', 'remote']
    }
    
    def __init__(self, spatial_threshold: float = 0.2):
        """
        Initialize semantic grouper.
        
        WHY THESE PARAMETERS:
        - spatial_threshold: Maximum normalized distance for objects to be considered "nearby"
          (0.2 = 20% of image width/height). This ensures objects are grouped only if they're
          actually close together, supporting accurate spatial understanding.
        
        Arguments:
            spatial_threshold: Maximum distance for spatial grouping (normalized)
        """
        self.spatial_threshold = spatial_threshold
    
    def get_semantic_category(self, class_name: str) -> str:
        """
        Get semantic category for a class name.
        
        WHY THIS MATTERS:
        Grouping by semantic category (furniture, people, vehicles) creates more meaningful
        descriptions than grouping by exact class name. "Three pieces of furniture" is more
        useful than "chair, table, couch" when objects are clustered together.
        
        Arguments:
            class_name: Object class name
        
        Returns:
            Semantic category name or 'other'
        """
        class_lower = class_name.lower()
        
        for category, classes in self.SEMANTIC_CATEGORIES.items():
            if any(cls in class_lower for cls in classes):
                return category
        
        return 'other'
    
    def group_objects(
        self,
        detections: List[Dict],
        group_by_category: bool = True,
        group_by_proximity: bool = True
    ) -> List[Dict]:
        """
        Group objects semantically.
        
        WHY THIS FUNCTION EXISTS:
        This transforms a list of individual detections into semantically grouped clusters,
        enabling concise descriptions like "Three chairs and two tables" instead of listing
        each object individually. This directly supports "Clear Multimodal Communication"
        by reducing information density while maintaining usefulness.
        
        HOW IT SUPPORTS THE PROBLEM STATEMENT:
        The problem asks for information that helps users "interact with the world like those
        who can." Sighted people naturally group objects ("some chairs over there") - this
        function provides that same semantic understanding for users with vision impairments.
        
        Arguments:
            detections: List of detection dictionaries with class_name, box, etc.
            group_by_category: Group objects by semantic category
            group_by_proximity: Group nearby objects of same type
        
        Returns:
            List of grouped detection dictionaries with 'count', 'grouped_objects', etc.
        """
        if not detections:
            return []
        
        if group_by_category:
            # Group by semantic category first
            category_groups = defaultdict(list)
            for det in detections:
                category = self.get_semantic_category(det.get('class_name', 'object'))
                category_groups[category].append(det)
            
            grouped = []
            for category, objects in category_groups.items():
                if group_by_proximity:
                    # Further group by spatial proximity within category
                    proximity_groups = self._group_by_proximity(objects)
                    for group in proximity_groups:
                        grouped.append(self._create_group_dict(group, category))
                else:
                    # Just group all objects of same category
                    grouped.append(self._create_group_dict(objects, category))
            
            return grouped
        else:
            # Group only by proximity (same class, nearby)
            return [self._create_group_dict(group) for group in self._group_by_proximity(detections)]
    
    def _group_by_proximity(self, objects: List[Dict]) -> List[List[Dict]]:
        """
        Group objects by spatial proximity.
        
        WHY PROXIMITY GROUPING:
        Objects that are close together should be described together ("three chairs clustered
        together" vs "chair, chair, chair"). This supports spatial understanding and prevents
        repetitive descriptions.
        
        Arguments:
            objects: List of detection dictionaries
        
        Returns:
            List of object groups (each group is a list of detections)
        """
        if not objects:
            return []
        
        groups = []
        used = set()
        
        for i, obj1 in enumerate(objects):
            if i in used:
                continue
            
            # Start new group
            group = [obj1]
            used.add(i)
            class_name = obj1.get('class_name', 'object')
            box1 = obj1.get('box')
            
            if box1 is None:
                continue
            
            # Find nearby objects of same class
            if isinstance(box1, torch.Tensor):
                cx1, cy1 = box1[0].item(), box1[1].item()
            else:
                cx1, cy1 = box1[0], box1[1]
            
            for j, obj2 in enumerate(objects):
                if j in used or j == i:
                    continue
                
                if obj2.get('class_name', 'object') != class_name:
                    continue
                
                box2 = obj2.get('box')
                if box2 is None:
                    continue
                
                if isinstance(box2, torch.Tensor):
                    cx2, cy2 = box2[0].item(), box2[1].item()
                else:
                    cx2, cy2 = box2[0], box2[1]
                
                # Calculate distance
                distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                
                if distance < self.spatial_threshold:
                    group.append(obj2)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _create_group_dict(self, objects: List[Dict], category: Optional[str] = None) -> Dict:
        """
        Create a grouped detection dictionary.
        
        WHY THIS STRUCTURE:
        Grouped detections need to maintain all original information while adding group-level
        metadata (count, representative object, etc.). This structure enables both individual
        object access and group-level descriptions.
        
        Arguments:
            objects: List of detections in the group
            category: Optional semantic category
        
        Returns:
            Grouped detection dictionary
        """
        if not objects:
            return {}
        
        # Use first object as representative
        representative = objects[0]
        
        # Calculate group center and average confidence
        boxes = [obj.get('box') for obj in objects if obj.get('box') is not None]
        if boxes and boxes[0] is not None:
            if isinstance(boxes[0], torch.Tensor):
                centers = [(b[0].item(), b[1].item()) for b in boxes if b is not None]
            else:
                centers = [(b[0], b[1]) for b in boxes if b is not None]
            
            if centers:
                avg_cx = sum(c[0] for c in centers) / len(centers)
                avg_cy = sum(c[1] for c in centers) / len(centers)
                group_box = [avg_cx, avg_cy, 0.1, 0.1]  # Approximate group size
            else:
                group_box = representative.get('box', [0.5, 0.5, 0.1, 0.1])
        else:
            group_box = representative.get('box', [0.5, 0.5, 0.1, 0.1])
        
        avg_confidence = sum(obj.get('confidence', 0.5) for obj in objects) / len(objects)
        avg_urgency = max(obj.get('urgency', 0) for obj in objects)  # Use max urgency
        
        return {
            'class_name': representative.get('class_name', 'object'),
            'count': len(objects),
            'grouped_objects': objects,
            'box': group_box,
            'confidence': avg_confidence,
            'urgency': avg_urgency,
            'distance': representative.get('distance', 1),
            'category': category or self.get_semantic_category(representative.get('class_name', 'object')),
            'is_grouped': True
        }
    
    def create_grouped_description(
        self,
        grouped_detections: List[Dict],
        verbosity: str = 'normal'
    ) -> str:
        """
        Create description from grouped detections.
        
        WHY GROUPED DESCRIPTIONS:
        Grouped descriptions are more concise and natural: "Three chairs and two tables" vs
        "Chair. Chair. Chair. Table. Table." This supports "Clear Multimodal Communication"
        by reducing information density while maintaining clarity.
        
        Arguments:
            grouped_detections: List of grouped detection dictionaries
            verbosity: 'brief', 'normal', or 'detailed'
        
        Returns:
            Natural language description
        """
        if not grouped_detections:
            return "No objects detected"
        
        descriptions = []
        
        for group in grouped_detections:
            count = group.get('count', 1)
            class_name = group.get('class_name', 'object')
            category = group.get('category', 'other')
            
            if count > 1:
                if verbosity == 'brief':
                    descriptions.append(f"{count} {class_name}s")
                elif verbosity == 'normal':
                    if category != 'other':
                        descriptions.append(f"{count} {category}")
                    else:
                        descriptions.append(f"{count} {class_name}s")
                else:  # detailed
                    if category != 'other':
                        descriptions.append(f"{count} {category} items ({class_name}s)")
                    else:
                        descriptions.append(f"{count} {class_name}s")
            else:
                descriptions.append(class_name)
        
        if verbosity == 'brief':
            return ", ".join(descriptions[:3])
        elif verbosity == 'normal':
            return ", ".join(descriptions[:5])
        else:  # detailed
            return ". ".join(descriptions) + "."


def group_detections_semantically(
    detections: List[Dict],
    group_by_category: bool = True,
    group_by_proximity: bool = True
) -> List[Dict]:
    """
    Convenience function to group detections semantically.
    
    WHY THIS FUNCTION:
    Provides a simple interface for semantic grouping that can be integrated into the
    description generation pipeline. This enables concise scene descriptions that support
    "Clear Multimodal Communication" without overwhelming users with individual object lists.
    
        Arguments:
        detections: List of detection dictionaries
        group_by_category: Group by semantic category
        group_by_proximity: Group nearby objects
    
    Returns:
        List of grouped detection dictionaries
    """
    grouper = SemanticGrouper()
    return grouper.group_objects(detections, group_by_category, group_by_proximity)

