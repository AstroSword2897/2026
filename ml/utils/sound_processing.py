"""Sound Processing Utilities for MaxSight Enhanced sound classification, directional detection, and prioritization. Sprint 3 Day 26: Advanced Sound Features."""

from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from enum import Enum


class SoundClass(Enum):
    """15 sound categories for environmental awareness."""
    ALARM = "alarm"
    SIREN = "siren"
    VEHICLE = "vehicle"
    FOOTSTEPS = "footsteps"
    SPEECH = "speech"
    DOOR = "door"
    BELL = "bell"
    CONSTRUCTION = "construction"
    WATER = "water"
    WIND = "wind"
    MUSIC = "music"
    CROWD = "crowd"
    ANIMAL = "animal"
    EMERGENCY = "emergency"
    BACKGROUND = "background"


class SoundDirection(Enum):
    """Directional audio detection."""
    LEFT = "left"
    RIGHT = "right"
    FRONT = "front"
    BACK = "back"
    CENTER = "center"
    UNKNOWN = "unknown"


class SoundProcessor:
    """Enhanced sound processing with classification refinement, directional detection, and prioritization. Sprint 3 Day 26: Advanced Sound Features."""
    
    # Sound urgency mapping (0-3: safe, caution, warning, danger)
    SOUND_URGENCY_MAP = {
        SoundClass.EMERGENCY: 3,  # Highest urgency.
        SoundClass.ALARM: 3,
        SoundClass.SIREN: 3,
        SoundClass.CONSTRUCTION: 2,  # Warning.
        SoundClass.VEHICLE: 2,
        SoundClass.DOOR: 1,  # Caution.
        SoundClass.BELL: 1,
        SoundClass.FOOTSTEPS: 1,
        SoundClass.SPEECH: 0,  # Safe.
        SoundClass.MUSIC: 0,
        SoundClass.WATER: 0,
        SoundClass.WIND: 0,
        SoundClass.CROWD: 0,
        SoundClass.ANIMAL: 1,
        SoundClass.BACKGROUND: 0
    }
    
    # Sound priority weights (for prioritization)
    SOUND_PRIORITY_WEIGHTS = {
        SoundClass.EMERGENCY: 100,
        SoundClass.ALARM: 90,
        SoundClass.SIREN: 85,
        SoundClass.CONSTRUCTION: 60,
        SoundClass.VEHICLE: 55,
        SoundClass.DOOR: 40,
        SoundClass.BELL: 35,
        SoundClass.FOOTSTEPS: 30,
        SoundClass.SPEECH: 20,
        SoundClass.MUSIC: 10,
        SoundClass.WATER: 5,
        SoundClass.WIND: 5,
        SoundClass.CROWD: 15,
        SoundClass.ANIMAL: 25,
        SoundClass.BACKGROUND: 1
    }
    
    def __init__(
        self,
        overlap_threshold: float = 0.3,
        directional_sensitivity: float = 0.2,
        enable_temporal_smoothing: bool = True
    ):
        """Initialize sound processor."""
        self.overlap_threshold = overlap_threshold
        self.directional_sensitivity = directional_sensitivity
        self.enable_temporal_smoothing = enable_temporal_smoothing
        self.sound_history: List[Dict] = []
        self.max_history = 10  # Keep last 10 sound detections.
    
    def refine_sound_classification(
        self,
        audio_features: torch.Tensor,
        raw_predictions: Optional[torch.Tensor] = None,
        confidence_threshold: float = 0.5
    ) -> List[Dict]:
        """Refine sound classification with better handling of overlapping sounds."""
        detections = []
        
        # If raw predictions provided, use them.
        if raw_predictions is not None:
            # Apply softmax to get probabilities.
            probs = torch.softmax(raw_predictions, dim=-1)
            
            # Find all sounds above threshold.
            for class_idx, sound_class in enumerate(SoundClass):
                confidence = probs[0, class_idx].item()
                if confidence > confidence_threshold:
                    detections.append({
                        'class': sound_class,
                        'confidence': confidence,
                        'class_idx': class_idx
                    })
        else:
            # Fallback: use audio features to estimate (simplified)
            # In production, use a trained sound classifier here.
            # For now, return empty list (requires model predictions)
            pass
        
        # Handle overlapping sounds.
        if len(detections) > 1:
            # Sort by confidence.
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Check for overlapping high-confidence sounds.
            primary = detections[0]
            overlapping = []
            
            for det in detections[1:]:
                # Confidence close to primary is treated as overlapping.
                if det['confidence'] > primary['confidence'] * (1 - self.overlap_threshold):
                    overlapping.append(det)
            
            # If we have overlapping sounds, mark them.
            if overlapping:
                primary['overlapping'] = [d['class'].value for d in overlapping]
                primary['overlap_count'] = len(overlapping)
        
        # Temporal smoothing.
        if self.enable_temporal_smoothing and self.sound_history:
            # Boost confidence if sound was detected in recent history.
            for det in detections:
                for hist in self.sound_history[-3:]:  # Check last 3 frames.
                    if hist.get('class') == det['class']:
                        # Boost confidence by 10% if seen recently.
                        det['confidence'] = min(1.0, det['confidence'] * 1.1)
                        det['temporal_boost'] = True
                        break
        
        # Update history.
        self.sound_history.extend(detections)
        if len(self.sound_history) > self.max_history:
            self.sound_history = self.sound_history[-self.max_history:]
        
        return detections
    
    def detect_direction(
        self,
        audio_features: torch.Tensor,
        stereo_channels: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> SoundDirection:
        """Detect sound direction (left/right/front/back)."""
        if stereo_channels is not None:
            left, right = stereo_channels
            
            # Calculate energy difference.
            left_energy = torch.mean(left ** 2).item()
            right_energy = torch.mean(right ** 2).item()
            
            energy_diff = abs(left_energy - right_energy) / (left_energy + right_energy + 1e-8)
            
            if energy_diff < self.directional_sensitivity:
                # Balanced - likely center or front/back.
                # Use phase difference to distinguish front/back.
                # (Simplified - in production would use more sophisticated analysis)
                return SoundDirection.CENTER
            elif left_energy > right_energy:
                return SoundDirection.LEFT
            else:
                return SoundDirection.RIGHT
        else:
            # Mono audio - cannot determine direction. Could use visual cues (object position) to infer direction.
            return SoundDirection.UNKNOWN
    
    def prioritize_sounds(
        self,
        sound_detections: List[Dict],
        visual_context: Optional[Dict] = None
    ) -> List[Dict]:
        """Prioritize sounds based on urgency, context, and user needs."""
        prioritized = []
        
        for det in sound_detections:
            sound_class = det.get('class')
            if not isinstance(sound_class, SoundClass):
                continue
            
            # Base priority from sound type.
            base_priority = self.SOUND_PRIORITY_WEIGHTS.get(sound_class, 0)
            urgency = self.SOUND_URGENCY_MAP.get(sound_class, 0)
            confidence = det.get('confidence', 0.5)
            
            # Adjust priority based on confidence.
            priority = base_priority * confidence
            
            # Contextual boost: if visual context confirms sound.
            if visual_context:
                # Check if visual detection matches sound (e.g., vehicle sound + car detected)
                visual_detections = visual_context.get('detections', [])
                for vdet in visual_detections:
                    vclass = vdet.get('class_name', '').lower()
                    sound_name = sound_class.value.lower()
                    
                    # Simple matching (in production, use semantic matching)
                    if sound_name in vclass or vclass in sound_name:
                        priority *= 1.5  # Boost if visual confirms.
                        det['visual_confirmed'] = True
                        break
            
            # Urgency boost: urgent sounds get higher priority.
            if urgency >= 2:  # Warning or danger.
                priority *= 1.3
            
            prioritized.append({
                **det,
                'priority': int(priority),
                'urgency': urgency,
                'base_priority': base_priority
            })
        
        # Sort by priority (highest first)
        prioritized.sort(key=lambda x: x.get('priority', 0), reverse=True)
        
        # Filter out very low priority sounds (background noise)
        prioritized = [p for p in prioritized if p.get('priority', 0) > 10]
        
        return prioritized
    
    def get_sound_description(
        self,
        sound_detection: Dict,
        include_direction: bool = True
    ) -> str:
        """Generate natural language description of sound."""
        sound_class = sound_detection.get('class')
        if not isinstance(sound_class, SoundClass):
            return "Unknown sound"
        
        direction = sound_detection.get('direction')
        confidence = sound_detection.get('confidence', 0.5)
        urgency = sound_detection.get('urgency', 0)
        
        # Base description.
        desc = sound_class.value.replace('_', ' ').title()
        
        # Add direction.
        if include_direction and direction:
            if isinstance(direction, SoundDirection):
                dir_str = direction.value
            else:
                dir_str = str(direction)
            
            if dir_str != 'unknown' and dir_str != 'center':
                desc = f"{desc} from the {dir_str}"
        
        # Add urgency indicator.
        if urgency >= 3:
            desc = f"URGENT: {desc}"
        elif urgency >= 2:
            desc = f"Warning: {desc}"
        
        # Add confidence indicator for low confidence.
        if confidence < 0.6:
            desc = f"Possible {desc.lower()}"
        
        return desc







