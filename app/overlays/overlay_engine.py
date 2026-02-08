"""Overlay Engine Renders visual overlays for therapy guidance. Phase 4: Overlay Engine & UX Guidance See docs/therapy_system_implementation_plan.md for implementation details."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
from PIL import Image


class OverlayEngine:
    """Renders visual overlays for therapy guidance."""
    
    def __init__(self, screen_size: Tuple[int, int] = (224, 224)):
        self.screen_size = screen_size
        self.max_overlay_percent = 0.10  # 10% max screen coverage.
        self.active_overlays = []
    
    def add_halo(
        self,
        center: Tuple[float, float],
        radius: float,
        intensity: float = 0.3
    ) -> Dict[str, Any]:
        """Add subtle halo overlay."""
        overlay = {
            'type': 'halo',
            'center': center,
            'radius': radius,
            'intensity': min(intensity, 0.5),  # Cap at 50% opacity.
            'color': (255, 255, 255)  # White, subtle.
        }
        self.active_overlays.append(overlay)
        return overlay
    
    def add_edge_glow(
        self,
        edges: List[Tuple[float, float]],
        width: float = 2.0,
        intensity: float = 0.4
    ) -> Dict[str, Any]:
        """Add edge glow for contrast reinforcement."""
        overlay = {
            'type': 'edge_glow',
            'edges': edges,
            'width': width,
            'intensity': min(intensity, 0.5),
            'color': (200, 200, 255)  # Subtle blue-white.
        }
        self.active_overlays.append(overlay)
        return overlay
    
    def add_depth_fog(
        self,
        depth_map: np.ndarray,
        near_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """Add depth "soft fog" for near objects."""
        overlay = {
            'type': 'depth_fog',
            'depth_map': depth_map,
            'near_threshold': near_threshold,
            'intensity': 0.2,  # Very subtle.
            'color': (150, 150, 150)  # Gray fog.
        }
        self.active_overlays.append(overlay)
        return overlay
    
    def add_motion_trace(
        self,
        path: List[Tuple[float, float]],
        width: float = 3.0
    ) -> Dict[str, Any]:
        """Add motion trace for tracking tasks."""
        overlay = {
            'type': 'motion_trace',
            'path': path,
            'width': width,
            'intensity': 0.6,
            'color': (100, 200, 255)  # Light blue.
        }
        self.active_overlays.append(overlay)
        return overlay
    
    def add_gaze_indicator(
        self,
        position: Tuple[float, float],
        size: float = 10.0
    ) -> Dict[str, Any]:
        """Add gaze position indicator."""
        overlay = {
            'type': 'gaze_indicator',
            'position': position,
            'size': size,
            'intensity': 0.5,
            'color': (255, 200, 100)  # Orange-yellow.
        }
        self.active_overlays.append(overlay)
        return overlay
    
    def add_guidance_arrow(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        width: float = 5.0
    ) -> Dict[str, Any]:
        """Add gentle arrow for guidance."""
        overlay = {
            'type': 'guidance_arrow',
            'start': start,
            'end': end,
            'width': width,
            'intensity': 0.4,
            'color': (150, 255, 150)  # Light green.
        }
        self.active_overlays.append(overlay)
        return overlay
    
    def create_overlay(
        self,
        base_image: Image.Image,
        detections: List[Dict[str, Any]],
        urgency_scores: Optional[np.ndarray] = None,
        text_regions: Optional[List[Dict[str, Any]]] = None
    ) -> Image.Image:
        """Create overlay with bounding boxes, labels, and text regions."""
        if not CV2_AVAILABLE:
            return base_image
        
        # Convert PIL to OpenCV format.
        img_array = np.array(base_image)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        h, w = img_array.shape[:2]
        
        # Draw detection bounding boxes.
        for det in detections:
            bbox = det.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                x1, y1 = int(x1 * w), int(y1 * h)
                x2, y2 = int(x2 * w), int(y2 * h)
                
                # Color based on urgency or confidence.
                confidence = det.get('confidence', 0.5)
                if urgency_scores is not None and len(urgency_scores) > 0:
                    urgency = int(urgency_scores.argmax())
                    if urgency >= 3:
                        color = (0, 0, 255)  # Red for danger.
                    elif urgency >= 2:
                        color = (0, 165, 255)  # Orange for warning.
                    else:
                        color = (0, 255, 0)  # Green for safe.
                else:
                    color = (0, 255, 255) if confidence > 0.7 else (255, 255, 0)
                
                # Draw bounding box.
                cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
                
                # Draw label.
                class_name = det.get('class_name', 'object')
                label = f"{class_name} {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img_array, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(img_array, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw text regions.
        if text_regions:
            for text_region in text_regions:
                bbox = text_region.get('bbox', [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    x1, y1 = int(x1 * w), int(y1 * h)
                    x2, y2 = int(x2 * w), int(y2 * h)
                    cv2.rectangle(img_array, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    text = text_region.get('text', '')
                    if text:
                        cv2.putText(img_array, text[:20], (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        # Convert back to PIL Image.
        return Image.fromarray(img_array)
    
    def render_overlays(self, base_image: np.ndarray) -> np.ndarray:
        """Render all active overlays onto base image. Arguments: base_image: Base image [H, W, 3] Returns: Image with overlays [H, W, 3]."""
        if not CV2_AVAILABLE:
            return base_image.copy()
        
        result = base_image.copy().astype(np.uint8)
        
        for overlay in self.active_overlays:
            overlay_type = overlay.get('type')
            intensity = overlay.get('intensity', 0.5)
            color = overlay.get('color', (255, 255, 255))
            
            if overlay_type == 'halo':
                center = overlay.get('center')
                radius = overlay.get('radius', 10)
                if center:
                    cx, cy = int(center[0] * result.shape[1]), int(center[1] * result.shape[0])
                    cv2.circle(result, (cx, cy), int(radius), color, -1)
                    result = cv2.addWeighted(result, 1 - intensity, base_image, intensity, 0)
            
            elif overlay_type == 'edge_glow':
                edges = overlay.get('edges', [])
                width = int(overlay.get('width', 2))
                if edges:
                    points = np.array([(int(x * result.shape[1]), int(y * result.shape[0])) 
                                     for x, y in edges], dtype=np.int32)
                    cv2.polylines(result, [points], False, color, width)
        
        return result
    
    def clear_overlays(self):
        """Clear all active overlays."""
        self.active_overlays = []
    
    def fade_overlays(self, fade_factor: float = 0.9):
        """Fade all overlays (called after task ends)."""
        for overlay in self.active_overlays:
            overlay['intensity'] *= fade_factor
            if overlay['intensity'] < 0.01:
                self.active_overlays.remove(overlay)







