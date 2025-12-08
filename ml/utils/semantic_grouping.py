"""
Semantic Grouping Module for MaxSight
Groups related objects into semantic clusters for concise scene descriptions.

PROJECT PHILOSOPHY & APPROACH:
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
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

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
    
    def __init__(
        self,
        spatial_threshold: float = 0.2,
        enable_cross_category: bool = True,
        cross_category_threshold: float = 0.15,
        use_confidence_weighting: bool = True
    ):
        """
        Initialize semantic grouper.
        
        WHY THESE PARAMETERS:
        - spatial_threshold: Maximum normalized distance for objects to be considered "nearby"
          (0.2 = 20% of image width/height). This ensures objects are grouped only if they're
          actually close together, supporting accurate spatial understanding.
        - enable_cross_category: Enable cross-category proximity clusters (e.g., chairs + tables = "dining area")
        - cross_category_threshold: Threshold for cross-category grouping (typically tighter than same-category)
        - use_confidence_weighting: Weight descriptions by detection confidence
        
        Arguments:
            spatial_threshold: Maximum distance for spatial grouping (normalized)
            enable_cross_category: Enable cross-category proximity clusters
            cross_category_threshold: Threshold for cross-category grouping
            use_confidence_weighting: Weight descriptions by confidence
        """
        self.spatial_threshold = spatial_threshold
        self.enable_cross_category = enable_cross_category
        self.cross_category_threshold = cross_category_threshold
        self.use_confidence_weighting = use_confidence_weighting
    
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
            
            # Add cross-category proximity clusters if enabled
            if self.enable_cross_category and group_by_proximity:
                cross_category_groups = self._group_cross_category_proximity(detections)
                for group in cross_category_groups:
                    grouped.append(self._create_group_dict(group, 'mixed'))
            
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
    
    def _group_cross_category_proximity(self, detections: List[Dict]) -> List[List[Dict]]:
        """
        Group objects from different categories that are spatially close (e.g., chairs + tables = dining area).
        
        WHY CROSS-CATEGORY GROUPING:
        Objects from different categories that are close together often form functional groups
        (dining area, workspace, etc.). This enables more natural descriptions like "dining area
        with chairs and tables" instead of listing each category separately.
        
        Arguments:
            detections: List of detection dictionaries
        
        Returns:
            List of cross-category groups
        """
        if not detections:
            return []
        
        # Get category for each detection
        detections_with_category = [
            (det, self.get_semantic_category(det.get('class_name', 'object')))
            for det in detections
        ]
        
        groups = []
        used = set()
        
        for i, (obj1, cat1) in enumerate(detections_with_category):
            if i in used:
                continue
            
            box1 = obj1.get('box')
            if box1 is None:
                continue
            
            if isinstance(box1, torch.Tensor):
                cx1, cy1 = box1[0].item(), box1[1].item()
            else:
                cx1, cy1 = box1[0], box1[1]
            
            # Start new cross-category group
            group = [obj1]
            used.add(i)
            
            # Find nearby objects from different categories
            for j, (obj2, cat2) in enumerate(detections_with_category):
                if j in used or j == i or cat1 == cat2:
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
                
                # Use tighter threshold for cross-category (must be very close)
                if distance < self.cross_category_threshold:
                    group.append(obj2)
                    used.add(j)
            
            # Only create group if it has multiple categories
            if len(group) > 1:
                categories = {self.get_semantic_category(obj.get('class_name', 'object')) for obj in group}
                if len(categories) > 1:  # Multiple categories
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
        max_confidence = max(obj.get('confidence', 0.5) for obj in objects)
        avg_urgency = max(obj.get('urgency', 0) for obj in objects)  # Use max urgency
        
        # Confidence-weighted average (higher confidence objects have more weight)
        if self.use_confidence_weighting:
            confidences = [obj.get('confidence', 0.5) for obj in objects]
            total_weight = sum(confidences)
            if total_weight > 0:
                weighted_confidence = sum(c * c for c in confidences) / total_weight
            else:
                weighted_confidence = avg_confidence
        else:
            weighted_confidence = avg_confidence
        
        return {
            'class_name': representative.get('class_name', 'object'),
            'count': len(objects),
            'grouped_objects': objects,
            'box': group_box,
            'confidence': weighted_confidence,
            'max_confidence': max_confidence,
            'avg_confidence': avg_confidence,
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
        
        # Sort by confidence if using confidence weighting
        if self.use_confidence_weighting:
            sorted_groups = sorted(
                grouped_detections,
                key=lambda g: g.get('confidence', 0.5),
                reverse=True
            )
        else:
            sorted_groups = grouped_detections
        
        for group in sorted_groups:
            count = group.get('count', 1)
            class_name = group.get('class_name', 'object')
            category = group.get('category', 'other')
            confidence = group.get('confidence', 0.5)
            
            # Confidence-weighted prefix for high-confidence groups
            confidence_prefix = ""
            if self.use_confidence_weighting and confidence > 0.8:
                confidence_prefix = "clearly visible "
            elif self.use_confidence_weighting and confidence < 0.5:
                confidence_prefix = "possibly "
            
            if count > 1:
                if verbosity == 'brief':
                    desc = f"{count} {class_name}s"
                    if confidence > 0.8:
                        desc = confidence_prefix + desc
                    descriptions.append(desc)
                elif verbosity == 'normal':
                    if category == 'mixed':
                        # Cross-category group - describe as functional area
                        categories = {self.get_semantic_category(obj.get('class_name', 'object')) 
                                    for obj in group.get('grouped_objects', [])}
                        if 'furniture' in categories:
                            desc = f"{count} furniture items (dining/work area)"
                        else:
                            desc = f"{count} items"
                        if confidence > 0.8:
                            desc = confidence_prefix + desc
                        descriptions.append(desc)
                    elif category != 'other':
                        desc = f"{count} {category}"
                        if confidence > 0.8:
                            desc = confidence_prefix + desc
                        descriptions.append(desc)
                    else:
                        desc = f"{count} {class_name}s"
                        if confidence > 0.8:
                            desc = confidence_prefix + desc
                        descriptions.append(desc)
                else:  # detailed
                    if category == 'mixed':
                        categories = {self.get_semantic_category(obj.get('class_name', 'object')) 
                                    for obj in group.get('grouped_objects', [])}
                        desc = f"{count} items from {len(categories)} categories: {', '.join(categories)}"
                        if confidence > 0.8:
                            desc = confidence_prefix + desc
                        descriptions.append(desc)
                    elif category != 'other':
                        desc = f"{count} {category} items ({class_name}s)"
                        if confidence > 0.8:
                            desc = confidence_prefix + desc
                        descriptions.append(desc)
                    else:
                        desc = f"{count} {class_name}s"
                        if confidence > 0.8:
                            desc = confidence_prefix + desc
                        descriptions.append(desc)
            else:
                desc = class_name
                if confidence > 0.8:
                    desc = confidence_prefix + desc
                descriptions.append(desc)
        
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

def visualize_semantic_groups(
    image: Any,
    grouped_detections: List[Dict],
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Visualize semantic groups on an image for debugging.
    
    WHY THIS HELPS:
    Visualizing semantic groups helps developers understand how objects are being grouped,
    enabling debugging and refinement of grouping logic. This is especially useful for
    validating cross-category proximity clusters and confidence weighting.
    
    Arguments:
        image: Input image (PIL Image, numpy array, or torch Tensor)
        grouped_detections: List of grouped detection dictionaries
        save_path: Optional path to save visualization
        show: If True, display the visualization
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping visualization")
        return
    
    # Convert image to numpy array
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image[0]  # Remove batch dimension
        if image.dim() == 3:
            image = image.permute(1, 2, 0).cpu().numpy()
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
    elif hasattr(image, 'numpy'):
        image = np.array(image)
    else:
        image = np.array(image)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    # Color map for categories
    category_colors = {
        'furniture': 'blue',
        'people': 'red',
        'vehicles': 'green',
        'doors': 'yellow',
        'stairs': 'orange',
        'signs': 'purple',
        'obstacles': 'cyan',
        'mixed': 'magenta',
        'other': 'gray'
    }
    
    for group in grouped_detections:
        box = group.get('box', [0.5, 0.5, 0.1, 0.1])
        category = group.get('category', 'other')
        count = group.get('count', 1)
        confidence = group.get('confidence', 0.5)
        
        # Convert normalized box to pixel coordinates
        h, w = image.shape[:2]
        if isinstance(box, torch.Tensor):
            cx, cy = box[0].item() * w, box[1].item() * h
            width, height = box[2].item() * w, box[3].item() * h
        else:
            cx, cy = box[0] * w, box[1] * h
            width, height = box[2] * w, box[3] * h
        
        x = cx - width / 2
        y = cy - height / 2
        
        # Draw bounding box
        color = category_colors.get(category, 'gray')
        rect = patches.Rectangle(
            (x, y), width, height,
            linewidth=2,
            edgecolor=color,
            facecolor='none',
            alpha=0.7
        )
        ax.add_patch(rect)
        
        # Add label
        label = f"{category} ({count})"
        if confidence < 0.5:
            label += " [low conf]"
        ax.text(
            x, y - 5,
            label,
            color=color,
            fontsize=10,
            weight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
    
    ax.set_title(f"Semantic Groups ({len(grouped_detections)} groups)", fontsize=14, weight='bold')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

