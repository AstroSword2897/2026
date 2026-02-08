"""Spatial Memory System for MaxSight."""

import torch
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import time
from dataclasses import dataclass, field
from threading import Lock
import numpy as np
try:
    from scipy.spatial import KDTree  # type: ignore
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    KDTree = None


@dataclass
class SpatialObject:
    """Represents an object in spatial memory."""
    class_name: str
    position: Tuple[float, float]  # (cx, cy) normalized.
    size: Tuple[float, float]  # (w, h) normalized.
    distance_zone: int
    first_seen: float
    last_seen: float
    seen_count: int
    stability: float  # 0-1, how stable the position is.
    # Compute stability incrementally.
    position_sum: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    position_sq_sum: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))


class SpatialMemory:
    """Maintains spatial memory of objects for cognitive mapping. Tracks object positions over time to help users build mental models."""
    
    def __init__(
        self,
        memory_duration: float = 30.0,  # Seconds.
        position_threshold: float = 0.1,  # Normalized distance for "same" position.
        stability_threshold: float = 0.7  # Minimum stability for "stable" objects.
    ):
        """Initialize spatial memory."""
        self.memory_duration = memory_duration
        self.position_threshold = position_threshold
        self.stability_threshold = stability_threshold
        
        # Store objects by class name.
        self.objects: Dict[str, List[SpatialObject]] = defaultdict(list)
        
        # Track object positions over time for stability calculation.
        self.position_history: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
        # Format: (cx, cy, timestamp)
        
        # Spatial indexing for O(log N) lookups instead of O(N)
        self.spatial_index: Dict[str, Optional[Any]] = {}  # KDTree per class.
        self._spatial_index_dirty: Dict[str, bool] = defaultdict(lambda: True)
        
        # Thread safety.
        self._lock = Lock()
    
    def update(
        self,
        detections: List[Dict],
        timestamp: Optional[float] = None
    ) -> None:
        """Update spatial memory with new detections."""
        with self._lock:
            if timestamp is None:
                timestamp = time.time()
            
            # Clean up old objects.
            self._cleanup_old_objects(timestamp)
            
            # Process new detections.
            for det in detections:
                class_name = det.get('class_name', 'object')
                box = det.get('box')
                distance_zone = det.get('distance', 1)
                
                if box is None:
                    continue
                
                # Extract position and size.
                if isinstance(box, torch.Tensor):
                    cx, cy = box[0].item(), box[1].item()
                    w, h = box[2].item(), box[3].item()
                else:
                    cx, cy = box[0], box[1]
                    w, h = box[2], box[3]
                
                # Validate coordinates.
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                    continue
                
                position = (cx, cy)
                size = (w, h)
                
                # Check if this matches an existing object.
                matched_obj = self._find_matching_object(class_name, position)
                
                if matched_obj:
                    # Update existing object stability incrementally.
                    matched_obj.last_seen = timestamp
                    matched_obj.seen_count += 1
                    old_pos = matched_obj.position
                    matched_obj.position = position
                    matched_obj.size = size
                    matched_obj.distance_zone = distance_zone
                    
                    # Update incremental statistics.
                    matched_obj.position_sum = (
                        matched_obj.position_sum[0] + cx,
                        matched_obj.position_sum[1] + cy
                    )
                    matched_obj.position_sq_sum = (
                        matched_obj.position_sq_sum[0] + cx * cx,
                        matched_obj.position_sq_sum[1] + cy * cy
                    )
                    
                    # Recalculate stability from running statistics.
                    matched_obj.stability = self._calculate_stability_incremental(matched_obj)
                    self._spatial_index_dirty[class_name] = True
                else:
                    # Create new object.
                    new_obj = SpatialObject(
                        class_name=class_name,
                        position=position,
                        size=size,
                        distance_zone=distance_zone,
                        first_seen=timestamp,
                        last_seen=timestamp,
                        seen_count=1,
                        stability=0.5,  # Initial stability.
                        position_sum=(cx, cy),
                        position_sq_sum=(cx * cx, cy * cy)
                    )
                    self.objects[class_name].append(new_obj)
                    self._spatial_index_dirty[class_name] = True
                
                # Update position history.
                self.position_history[class_name].append((cx, cy, timestamp))
                # Keep only recent history (last 10 seconds)
                cutoff_time = timestamp - 10.0
                self.position_history[class_name] = [
                    p for p in self.position_history[class_name]
                    if p[2] > cutoff_time
                ]
    
    def _rebuild_spatial_index(self, class_name: str) -> None:
        """Rebuild spatial index for a class using KDTree."""
        if not SCIPY_AVAILABLE:
            return
        
        if class_name not in self.objects or not self.objects[class_name]:
            if class_name in self.spatial_index:
                del self.spatial_index[class_name]
            return
        
        objects = self.objects[class_name]
        if len(objects) == 0:
            return
        
        positions = np.array([obj.position for obj in objects])
        if len(positions) > 0 and SCIPY_AVAILABLE and KDTree is not None:
            self.spatial_index[class_name] = KDTree(positions)
        else:
            self.spatial_index[class_name] = None
        self._spatial_index_dirty[class_name] = False
    
    def _find_matching_object(
        self,
        class_name: str,
        position: Tuple[float, float]
    ) -> Optional[SpatialObject]:
        """Find object of same class near the given position using spatial indexing."""
        if class_name not in self.objects or not self.objects[class_name]:
            return None
        
        # Rebuild index if dirty.
        if self._spatial_index_dirty.get(class_name, True):
            self._rebuild_spatial_index(class_name)
        
        # Use KDTree if available, fallback to linear search.
        if SCIPY_AVAILABLE and class_name in self.spatial_index:
            tree = self.spatial_index[class_name]
            if tree is not None:
                try:
                    distances, indices = tree.query([position], k=1, distance_upper_bound=self.position_threshold)
                    if distances[0] < self.position_threshold and indices[0] < len(self.objects[class_name]):
                        return self.objects[class_name][indices[0]]
                except (ValueError, IndexError):
                    # Fallback to linear search on error.
                    pass
        
        # Fallback: linear search (for small lists or when scipy unavailable)
        cx, cy = position
        for obj in self.objects[class_name]:
            obj_cx, obj_cy = obj.position
            distance = ((cx - obj_cx)**2 + (cy - obj_cy)**2)**0.5
            
            if distance < self.position_threshold:
                return obj
        
        return None
    
    def _calculate_stability_incremental(self, obj: SpatialObject) -> float:
        """Calculate stability score using incremental statistics (O(1) instead of O(N)). Returns: Stability score 0-1 (1 = very stable, 0 = moving)"""
        if obj.seen_count < 2:
            return 0.5
        
        n = obj.seen_count
        mean_x = obj.position_sum[0] / n
        mean_y = obj.position_sum[1] / n
        
        # Variance = E[X²] - E[X]².
        var_x = (obj.position_sq_sum[0] / n) - (mean_x ** 2)
        var_y = (obj.position_sq_sum[1] / n) - (mean_y ** 2)
        variance = var_x + var_y
        
        # Convert variance to stability (lower variance = higher stability) Normalize to 0-1 range (assuming max variance of 0.1)
        stability = max(0.0, 1.0 - min(1.0, variance / 0.1))
        
        return stability
    
    def _calculate_stability(
        self,
        class_name: str,
        current_position: Tuple[float, float]
    ) -> float:
        """Calculate stability score based on position history (legacy method). Returns: Stability score 0-1 (1 = very stable, 0 = moving)"""
        if class_name not in self.position_history:
            return 0.5
        
        history = self.position_history[class_name]
        if len(history) < 2:
            return 0.5
        
        # Calculate variance in position.
        positions = [(p[0], p[1]) for p in history]
        cx_mean = sum(p[0] for p in positions) / len(positions)
        cy_mean = sum(p[1] for p in positions) / len(positions)
        
        variance = sum(
            ((p[0] - cx_mean)**2 + (p[1] - cy_mean)**2)
            for p in positions
        ) / len(positions)
        
        # Convert variance to stability (lower variance = higher stability) Normalize to 0-1 range (assuming max variance of 0.1)
        stability = max(0.0, 1.0 - min(1.0, variance / 0.1))
        
        return stability
    
    def _cleanup_old_objects(self, current_time: float) -> None:
        """Remove objects that haven't been seen recently."""
        for class_name in list(self.objects.keys()):
            self.objects[class_name] = [
                obj for obj in self.objects[class_name]
                if (current_time - obj.last_seen) < self.memory_duration
            ]
            
            # Remove empty lists.
            if not self.objects[class_name]:
                del self.objects[class_name]
                if class_name in self.spatial_index:
                    del self.spatial_index[class_name]
                self._spatial_index_dirty[class_name] = True
            else:
                self._spatial_index_dirty[class_name] = True
            
            # Clean position history too (fix memory leak)
            if class_name in self.position_history:
                cutoff_time = current_time - 10.0
                self.position_history[class_name] = [
                    p for p in self.position_history[class_name]
                    if p[2] > cutoff_time
                ]
                if not self.position_history[class_name]:
                    del self.position_history[class_name]
    
    def get_stable_objects(self) -> List[SpatialObject]:
        """Get objects that are stable (not moving, frequently seen). Returns: List of stable spatial objects."""
        with self._lock:
            stable = []
            for objects_list in self.objects.values():
                for obj in objects_list:
                    if (obj.stability >= self.stability_threshold and 
                        obj.seen_count >= 3):  # Seen at least 3 times.
                        stable.append(obj)
            
            return stable
    
    def get_recent_objects(self, time_window: float = 5.0) -> List[SpatialObject]:
        """Get objects seen within the time window. Arguments: time_window: Time window in seconds Returns: List of recent spatial objects."""
        with self._lock:
            current_time = time.time()
            recent = []
            
            for objects_list in self.objects.values():
                for obj in objects_list:
                    if (current_time - obj.last_seen) <= time_window:
                        recent.append(obj)
            
            return recent
    
    def get_contextual_reminder(
        self,
        current_detections: List[Dict]
    ) -> Optional[str]:
        """Generate contextual reminder based on spatial memory."""
        with self._lock:
            if not current_detections:
                return None
            
            # LIMIT: Only process if we have stable objects.
            stable_objects = self.get_stable_objects()
            if not stable_objects:
                return None
            
            current_classes = {det.get('class_name') for det in current_detections}
            reminders = []
            
            # LIMIT: Only check first 5 stable objects for performance.
            for stable_obj in stable_objects[:5]:
                if stable_obj.class_name not in current_classes:
                    # Object was here before but is now gone.
                    time_since = time.time() - stable_obj.last_seen
                    if time_since < 10.0:  # Recently disappeared.
                        reminders.append(f"{stable_obj.class_name} you just passed")
                        if len(reminders) >= 2:  # Early exit.
                            break
            
            # Check for objects that are consistently in the same position.
            if len(reminders) < 2:
                for det in current_detections[:10]:  # Limit detections processed.
                    class_name = det.get('class_name')
                    box = det.get('box')
                    
                    if box is None or class_name is None:
                        continue
                    
                    if isinstance(box, torch.Tensor):
                        position = (box[0].item(), box[1].item())
                    else:
                        position = (box[0], box[1])
                    
                    matched_obj = self._find_matching_object(class_name, position)
                    if matched_obj and matched_obj.stability >= self.stability_threshold:
                        if matched_obj.seen_count >= 5:  # Frequently seen.
                            reminders.append(f"{class_name} ahead as before")
                            if len(reminders) >= 2:  # Early exit.
                                break
            
            if reminders:
                return ". ".join(reminders[:2]) + "."  # Limit to 2 reminders.
            
            return None
    
    def get_spatial_summary(self) -> Dict[str, Any]:
        """Get summary of spatial memory state. Returns: Dictionary with memory statistics."""
        with self._lock:
            total_objects = sum(len(objs) for objs in self.objects.values())
            stable_count = len(self.get_stable_objects())
            recent_count = len(self.get_recent_objects())
            
            return {
                'total_objects': total_objects,
                'stable_objects': stable_count,
                'recent_objects': recent_count,
                'memory_duration': self.memory_duration,
                'object_types': list(self.objects.keys())
            }


class SpatialMemorySystem(SpatialMemory):
    """Alias for SpatialMemory used by MaxSightCNN. Accepts image_size for compatibility with the model's constructor."""
    def __init__(
        self,
        memory_duration: float = 30.0,
        stability_threshold: float = 0.7,
        image_size: Tuple[int, int] = (640, 480),
        position_threshold: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__(
            memory_duration=memory_duration,
            position_threshold=position_threshold,
            stability_threshold=stability_threshold,
            **kwargs,
        )
        self.image_size = image_size







