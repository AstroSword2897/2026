"""Cross-Modal Output Scheduler."""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import torch
import re

try:
    from ml.utils.sound_processing import SoundProcessor, SoundClass, SoundDirection
    SOUND_PROCESSING_AVAILABLE = True
except ImportError:
    SOUND_PROCESSING_AVAILABLE = False
    SoundProcessor = None
    SoundClass = None
    SoundDirection = None


class OutputChannel(Enum):
    """Output channel types for multimodal communication."""
    AUDIO = "audio"
    HAPTIC = "haptic"
    VISUAL = "visual"
    HYBRID = "hybrid"


class AlertFrequency(Enum):
    """Alert frequency levels for information density control."""
    LOW = "low"      # Only hazards.
    MEDIUM = "medium"  # Hazards + important objects.
    HIGH = "high"     # Continuous narration.


@dataclass
class OutputConfig:
    """Configuration for output scheduling."""
    preferred_channel: OutputChannel = OutputChannel.AUDIO
    alert_frequency: AlertFrequency = AlertFrequency.MEDIUM
    audio_volume: float = 0.7
    haptic_intensity: float = 0.8
    visual_contrast: float = 0.9
    reaction_time_ms: float = 250.0
    uncertainty_threshold: float = 0.3  # Suppress alerts if uncertainty > threshold.
    verbosity: str = 'normal'  # 'brief', 'normal', or 'detailed'


@dataclass
class ScheduledOutput:
    """A scheduled output event."""
    channel: OutputChannel
    priority: int  # 0-100.
    intensity: float  # 0-1.
    frequency: float  # Hz.
    duration: float  # Seconds.
    content: str  # Description for audio/narration.
    spatial_position: Optional[Tuple[float, float]] = None  # For spatial audio/haptic.
    distance: Optional[float] = None  # Distance in meters for volume adjustment.
    audio_pan: float = 0.0  # Left (-1.0) to right (1.0) panning for spatial audio.
    volume_multiplier: float = 1.0  # Distance-based volume adjustment (closer = higher)


class CrossModalScheduler:
    """Schedules outputs across audio, haptic, and visual channels. Manages frequency, intensity, and prioritization based on user profile and model outputs."""
    
    def __init__(self, config: OutputConfig):
        self.config = config
        self.last_output_time: Dict[str, float] = {}
        # Cross-channel rate limiting to prevent sensory overload.
        self.last_output_by_channel: Dict[OutputChannel, float] = {}
        self.min_channel_interval = 0.3  # 300ms between ANY outputs (except emergencies)
        self.output_history: List[ScheduledOutput] = []
        # Track previous outputs for smooth audio transitions (Sprint 3 Day 22)
        self.previous_outputs: Dict[str, ScheduledOutput] = {}
        # Sound processing (Sprint 3 Day 26)
        if SOUND_PROCESSING_AVAILABLE:
            self.sound_processor = SoundProcessor()
        else:
            self.sound_processor = None
        
    def schedule_outputs(
        self,
        detections: List[Dict],
        model_outputs: Dict[str, torch.Tensor],
        timestamp: float
    ) -> List[ScheduledOutput]:
        """Schedule outputs based on detections and model outputs."""
        scheduled = []
        
        # Process high-urgency items even when uncertainty is high.
        critical_detections = [d for d in detections if d.get('urgency', 0) >= 3]
        normal_detections = [d for d in detections if d.get('urgency', 0) < 3]
        
        # Get uncertainty - suppress if too high (only for normal detections) By ensuring only reliable information is presented.
        uncertainty = model_outputs.get('uncertainty', torch.tensor(0.0))
        if isinstance(uncertainty, torch.Tensor):
            uncertainty = uncertainty.item()
        
        if uncertainty > self.config.uncertainty_threshold:
            # WHY: Safety first - even with uncertainty, hazards must be communicated.
            priority_threshold = 90
        else:
            # Normal operation - use frequency-based threshold.
            # WHY: Adapts to user preferences - some users want more info, others want less.
            priority_threshold = self._get_priority_threshold()
        
        # Filter normal detections by priority and frequency settings.
        filtered_normal = [
            d for d in normal_detections
            if d.get('priority', 0) >= priority_threshold
        ]
        
        # Combine critical and filtered normal detections.
        filtered_detections = critical_detections + filtered_normal
        
        # Sort by priority (highest first)
        filtered_detections.sort(key=lambda x: x.get('priority', 0), reverse=True)
        
        # Limit number of outputs based on frequency. Supports Practical Usability & Safety Goals (counterproductive otherwise).
        max_outputs = self._get_max_outputs()
        filtered_detections = filtered_detections[:max_outputs]
        
        # Schedule each detection. Channels, timing, and descriptions.
        for det in filtered_detections:
            output = self._create_output_for_detection(det, model_outputs, timestamp)
            if output:
                scheduled.append(output)
                self.last_output_time[det.get('class_name', 'unknown')] = timestamp
        
        scene_outputs = self._create_scene_outputs(model_outputs, detections, timestamp)
        scheduled.extend(scene_outputs)
        
        # Store history. WHY: Enables analysis of information patterns and supports future improvements.
        self.output_history.extend(scheduled)
        if len(self.output_history) > 100:  # Keep last 100 outputs.
            self.output_history = self.output_history[-100:]
        
        return scheduled
    
    def _get_priority_threshold(self) -> int:
        """Get priority threshold based on alert frequency."""
        thresholds = {
            AlertFrequency.LOW: 70,      # Only hazards + navigation.
            AlertFrequency.MEDIUM: 40,    # + useful objects.
            AlertFrequency.HIGH: 0       # All objects.
        }
        return thresholds.get(self.config.alert_frequency, 40)
    
    def _get_max_outputs(self) -> int:
        """Get maximum number of outputs per frame based on frequency."""
        limits = {
            AlertFrequency.LOW: 3,
            AlertFrequency.MEDIUM: 5,
            AlertFrequency.HIGH: 10
        }
        return limits.get(self.config.alert_frequency, 5)
    
    def _create_output_for_detection(
        self,
        detection: Dict,
        model_outputs: Dict[str, torch.Tensor],
        timestamp: float
    ) -> Optional[ScheduledOutput]:
        """Create output for a single detection."""
        priority = detection.get('priority', 0)
        class_name = detection.get('class_name', 'object')
        box = detection.get('box', [0.5, 0.5, 0.1, 0.1])
        findability = detection.get('findability', 0.5)
        urgency = detection.get('urgency', 0)
        
        # Determine channel first (needed for rate limiting check)
        channel = self._select_channel(priority, urgency)
        
        # Check rate limiting (including cross-channel)
        if self._should_suppress(class_name, timestamp, priority, channel):
            return None
        
        # Calculate intensity based on priority and findability.
        intensity = self._calculate_intensity(priority, findability, urgency)
        
        # Calculate frequency (Hz) - higher priority = faster rhythm.
        frequency = self._calculate_frequency(priority, urgency)
        
        # Calculate duration.
        duration = self._calculate_duration(priority, urgency)
        
        # Generate content.
        content = self._generate_content(detection, model_outputs)
        
        # Spatial position from bounding box center.
        spatial_pos = (box[0], box[1]) if len(box) >= 2 else None
        
        # Calculate 3D audio positioning (Sprint 3 Day 22: Spatial Audio Refinement)
        audio_pan = 0.0
        distance = None
        volume_multiplier = 1.0
        
        if spatial_pos is not None:
            # Calculate left/right panning from x position (-1.0 = left, 1.0 = right)
            x_pos = spatial_pos[0]  # Normalized [0, 1].
            audio_pan = (x_pos - 0.5) * 2.0  # Convert to [-1.0, 1.0].
            
            # Get distance from detection if available.
            distance_zone = detection.get('distance_zone', None)
            if distance_zone is not None:
                # Convert distance zone to approximate meters.
                if distance_zone == 0:  # Near.
                    distance = 2.0  # ~2 meters.
                    volume_multiplier = 1.2  # 20% louder for close objects.
                elif distance_zone == 1:  # Medium.
                    distance = 6.0  # ~6 meters.
                    volume_multiplier = 1.0  # Normal volume.
                else:  # Far.
                    distance = 10.0  # ~10 meters.
                    volume_multiplier = 0.8  # 20% quieter for far objects.
            
            # Get precise distance if available.
            precise_distance = detection.get('distance_meters', None)
            if precise_distance is not None:
                distance = precise_distance
                # Distance-based volume: closer = louder (inverse square law approximation)
                if distance > 0:
                    # Normalize: 1m = 1.2x, 5m = 1.0x, 10m = 0.8x.
                    volume_multiplier = max(0.5, min(1.5, 1.0 + (5.0 - distance) / 10.0))
        
        # Smooth transitions: track previous position for smooth audio movement.
        prev_output = self.previous_outputs.get(class_name)
        if prev_output and prev_output.spatial_position and spatial_pos:
            # Smooth panning transition (avoid sudden jumps)
            prev_pan = (prev_output.spatial_position[0] - 0.5) * 2.0
            pan_diff = abs(audio_pan - prev_pan)
            if pan_diff > 0.3:  # Large jump - smooth it.
                audio_pan = prev_pan + (audio_pan - prev_pan) * 0.5  # 50% of the way.
            
            # Smooth volume transitions.
            if prev_output.volume_multiplier:
                volume_diff = abs(volume_multiplier - prev_output.volume_multiplier)
                if volume_diff > 0.2:  # Large change - smooth it.
                    volume_multiplier = prev_output.volume_multiplier + (volume_multiplier - prev_output.volume_multiplier) * 0.6
        
        # Update rate limiting timestamps.
        self.last_output_time[class_name] = timestamp
        self.last_output_by_channel[channel] = timestamp
        
        output = ScheduledOutput(
            channel=channel,
            priority=priority,
            intensity=intensity,
            frequency=frequency,
            duration=duration,
            content=content,
            spatial_position=spatial_pos,
            distance=distance,
            audio_pan=audio_pan,
            volume_multiplier=volume_multiplier
        )
        
        # Store for smooth transitions.
        self.previous_outputs[class_name] = output
        
        return output
    
    def _should_suppress(self, class_name: str, timestamp: float, priority: int, channel: Optional[OutputChannel] = None) -> bool:
        """Check if output should be suppressed due to rate limiting."""
        # Emergency alerts (priority >= 90) always go through.
        if priority >= 90:
            return False
        
        # Check cross-channel rate limiting.
        if channel is not None:
            last_time = self.last_output_by_channel.get(channel, 0.0)
            if (timestamp - last_time) < self.min_channel_interval:
                return True
        
        if class_name not in self.last_output_time:
            return False
        
        last_time = self.last_output_time[class_name]
        time_since = timestamp - last_time
        
        # Rate limits based on priority (higher priority = more frequent)
        if priority >= 90:
            min_interval = 0.5  # 2 Hz max for hazards.
        elif priority >= 70:
            min_interval = 1.0  # 1 Hz max for navigation.
        else:
            min_interval = 2.0  # 0.5 Hz max for useful objects.
        
        return time_since < min_interval
    
    def _select_channel(self, priority: int, urgency: int) -> OutputChannel:
        """Select output channel based on priority and user preference."""
        # High priority/urgency -> use preferred channel or hybrid.
        if priority >= 90 or urgency >= 3:
            if self.config.preferred_channel == OutputChannel.HYBRID:
                return OutputChannel.HYBRID
            return self.config.preferred_channel
        
        # Medium priority -> use preferred channel.
        if priority >= 70:
            return self.config.preferred_channel
        
        # Low priority -> use less intrusive channel.
        if self.config.preferred_channel == OutputChannel.AUDIO:
            return OutputChannel.VISUAL  # Visual overlay instead of audio.
        return self.config.preferred_channel
    
    def _calculate_intensity(self, priority: int, findability: float, urgency: int) -> float:
        """Calculate output intensity (0-1)"""
        # Base intensity from priority.
        base_intensity = priority / 100.0
        
        # Adjust for findability (harder to find = higher intensity)
        findability_adjustment = (1.0 - findability) * 0.2
        
        # Adjust for urgency.
        urgency_adjustment = urgency / 3.0 * 0.3
        
        intensity = base_intensity + findability_adjustment + urgency_adjustment
        
        # Apply channel-specific scaling.
        if self.config.preferred_channel == OutputChannel.AUDIO:
            intensity *= self.config.audio_volume
        elif self.config.preferred_channel == OutputChannel.HAPTIC:
            intensity *= self.config.haptic_intensity
        else:
            intensity *= self.config.visual_contrast
        
        return min(1.0, max(0.0, intensity))
    
    def _calculate_frequency(self, priority: int, urgency: int) -> float:
        """Calculate output frequency in Hz."""
        # Higher priority/urgency = faster rhythm.
        if priority >= 90 or urgency >= 3:
            return 10.0  # Fast rhythm for hazards.
        elif priority >= 70:
            return 5.0   # Medium rhythm for navigation.
        else:
            return 2.0   # Slow rhythm for useful objects.
    
    def _calculate_duration(self, priority: int, urgency: int) -> float:
        """Calculate output duration in seconds."""
        # Higher priority = longer duration.
        if priority >= 90 or urgency >= 3:
            return 0.5  # Longer for hazards.
        elif priority >= 70:
            return 0.3  # Medium for navigation.
        else:
            return 0.1  # Short for useful objects.
    
    def _generate_content(self, detection: Dict, model_outputs: Dict) -> str:
        """Generate content description for output using enhanced description generator."""
        from .description_generator import DescriptionGenerator
        
        class_name = detection.get('class_name', 'object')
        box = detection.get('box')
        distance_zone = detection.get('distance', 1)
        urgency = detection.get('urgency', 0)
        priority = detection.get('priority', 0)
        
        # Get verbosity from config.
        verbosity = getattr(self.config, 'verbosity', 'normal')
        desc_gen = DescriptionGenerator(verbosity=verbosity)
        
        # Generate enhanced description.
        if box is not None:
            box_tensor = torch.tensor(box) if not isinstance(box, torch.Tensor) else box
            if urgency >= 2:
                # Use hazard alert for high urgency.
                return desc_gen.generate_hazard_alert(class_name, box_tensor, distance_zone, urgency)
            else:
                # Use standard object description.
                return desc_gen.generate_object_description(
                    class_name, box_tensor, distance_zone, urgency, priority, verbosity
                )
        else:
            # Fallback to simple description.
            if priority >= 90:
                return f"Warning: {class_name} ahead"
            elif priority >= 70:
                return f"{class_name} {distance_zone}"
            else:
                return class_name
    
    def _create_scene_outputs(
        self,
        model_outputs: Dict[str, torch.Tensor],
        detections: List[Dict],
        timestamp: float
    ) -> List[ScheduledOutput]:
        """Create scene-level outputs (navigation difficulty, glare warnings, navigation guidance)"""
        from .description_generator import DescriptionGenerator
        
        outputs = []
        desc_gen = DescriptionGenerator(verbosity=self.config.verbosity)
        
        # Navigation difficulty warning.
        nav_difficulty = model_outputs.get('navigation_difficulty', None)
        if nav_difficulty is not None:
            if isinstance(nav_difficulty, torch.Tensor):
                nav_difficulty = nav_difficulty.item()
            if nav_difficulty > 0.7:  # High difficulty.
                outputs.append(ScheduledOutput(
                    channel=self.config.preferred_channel,
                    priority=60,
                    intensity=0.6,
                    frequency=2.0,
                    duration=0.2,
                    content="Difficult navigation ahead"
                ))
        
        # Navigation guidance (if detections available)
        if detections:
            nav_guidance = desc_gen.generate_navigation_guidance(detections)
            if nav_guidance != "Clear path ahead" or self.config.verbosity == 'detailed':
                outputs.append(ScheduledOutput(
                    channel=self.config.preferred_channel,
                    priority=55,
                    intensity=0.5,
                    frequency=1.0,
                    duration=0.3,
                    content=nav_guidance
                ))
        
        # Glare warning.
        glare_level = model_outputs.get('glare_risk_level', None)
        if glare_level is not None:
            if isinstance(glare_level, torch.Tensor):
                glare_level = glare_level.item()
            if glare_level >= 2:  # Medium or high glare.
                outputs.append(ScheduledOutput(
                    channel=OutputChannel.VISUAL,  # Visual overlay for glare.
                    priority=50,
                    intensity=0.7,
                    frequency=1.0,
                    duration=0.3,
                    content="High glare detected"
                ))
        
        return outputs


def create_scheduler_from_profile(user_profile: Dict) -> CrossModalScheduler:
    """Create scheduler from user profile."""
    config = OutputConfig(
        preferred_channel=OutputChannel(user_profile.get('preferred_output_channel', 'audio')),
        alert_frequency=AlertFrequency(user_profile.get('alert_frequency', 'medium')),
        audio_volume=user_profile.get('accessibility_preferences', {}).get('audio_volume', 0.7),
        haptic_intensity=user_profile.get('accessibility_preferences', {}).get('haptic_intensity', 0.8),
        visual_contrast=user_profile.get('accessibility_preferences', {}).get('contrast_mode', 0.9),
        reaction_time_ms=user_profile.get('reaction_time_ms', 250.0)
    )
    return CrossModalScheduler(config)


# RUNTIME OUTPUT CONTRACT (Patient Safety)
"""Runtime Output Contract."""

import re
from enum import Enum
from dataclasses import dataclass


class OutputMode(Enum):
    """Output mode for different user types."""
    PATIENT = "patient"
    CLINICIAN = "clinician"
    DEV = "dev"


class Severity(Enum):
    """Message severity levels."""
    INFO = "info"
    WARNING = "warning"
    HAZARD = "hazard"
    CRITICAL = "critical"


@dataclass
class RuntimeOutput:
    """Strict runtime output contract. All user-facing messages must conform to this schema. Patient mode only receives: mode, severity, message, confidence, cooldown_applied."""
    mode: OutputMode
    severity: Severity
    message: str
    confidence: float  # 0.0-1.0.
    cooldown_applied: bool
    
    # Optional fields (only for clinician/dev modes)
    latency_ms: Optional[float] = None
    uncertainty: Optional[float] = None
    fallback_used: Optional[bool] = None
    component_breakdown: Optional[Dict[str, Any]] = None
    debug_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, respecting mode constraints."""
        result = {
            'mode': self.mode.value,
            'severity': self.severity.value,
            'message': self.message,
            'confidence': self.confidence,
            'cooldown_applied': self.cooldown_applied
        }
        
        # Only include extended fields for clinician/dev modes.
        if self.mode in (OutputMode.CLINICIAN, OutputMode.DEV):
            if self.latency_ms is not None:
                result['latency_ms'] = self.latency_ms
            if self.uncertainty is not None:
                result['uncertainty'] = self.uncertainty
            if self.fallback_used is not None:
                result['fallback_used'] = self.fallback_used
            if self.component_breakdown is not None:
                result['component_breakdown'] = self.component_breakdown
            
        # Debug info only for dev mode.
        if self.mode == OutputMode.DEV and self.debug_info is not None:
            result['debug_info'] = self.debug_info
        
        return result


class OutputValidator:
    """Validates runtime outputs against the contract. Rejects symbol characters and enforces schema constraints."""
    
    # Blocklist: checkmarks, arrows, geometric shapes (Unicode escapes to avoid emoji in source).
    FORBIDDEN_SYMBOLS_PATTERN = re.compile(
        r'[\u2705\u274C\u2713\u2717\u2714\u2716=\u00D7\u2192\u2190\u2191\u2193\u25B6\u25C0\u25B2\u25BC\u25A0\u25A1\u25CF\u25CB\u2605\u2606\u2666\u2665\u2660\u2663]'
    )
    
    @classmethod
    def validate_message(cls, message: str, mode: OutputMode) -> tuple[bool, Optional[str]]:
        """Validate a message string. Returns: (is_valid, error_message)"""
        # Check for forbidden symbols.
        if cls.FORBIDDEN_SYMBOLS_PATTERN.search(message):
            return False, "Message contains forbidden symbol characters"
        
        # Patient mode: enforce brevity.
        if mode == OutputMode.PATIENT:
            if len(message) > 200:
                return False, "Patient mode messages must be under 200 characters"
        
        # Basic sanity checks.
        if not message or not message.strip():
            return False, "Message cannot be empty"
        
        return True, None
    
    @classmethod
    def validate_confidence(cls, confidence: float) -> tuple[bool, Optional[str]]:
        """Validate confidence value."""
        if not (0.0 <= confidence <= 1.0):
            return False, f"Confidence must be in [0.0, 1.0], got {confidence}"
        return True, None
    
    @classmethod
    def validate_output(cls, output: RuntimeOutput) -> tuple[bool, Optional[str]]:
        """Validate a complete RuntimeOutput. Returns: (is_valid, error_message)"""
        # Validate message.
        is_valid, error = cls.validate_message(output.message, output.mode)
        if not is_valid:
            return False, error
        
        # Validate confidence.
        is_valid, error = cls.validate_confidence(output.confidence)
        if not is_valid:
            return False, error
        
        # Patient mode must not have debug fields.
        if output.mode == OutputMode.PATIENT:
            if output.debug_info is not None:
                return False, "Patient mode cannot include debug_info"
            # Allow component_breakdown in clinician mode only.
            if output.component_breakdown is not None:
                return False, "Patient mode cannot include component_breakdown"
        
        return True, None
    
    @classmethod
    def sanitize_message(cls, message: str) -> str:
        """Remove forbidden symbols from message."""
        return cls.FORBIDDEN_SYMBOLS_PATTERN.sub('', message)


def create_patient_output(
    message: str,
    severity: Severity,
    confidence: float,
    cooldown_applied: bool = False
) -> RuntimeOutput:
    """Create a patient-safe output. Automatically sanitizes message and validates."""
    # Sanitize message.
    message = OutputValidator.sanitize_message(message)
    
    # Truncate if too long.
    if len(message) > 200:
        message = message[:197] + "..."
    
    # Clamp confidence.
    confidence = max(0.0, min(1.0, confidence))
    
    return RuntimeOutput(
        mode=OutputMode.PATIENT,
        severity=severity,
        message=message,
        confidence=confidence,
        cooldown_applied=cooldown_applied
    )


def create_clinician_output(
    message: str,
    severity: Severity,
    confidence: float,
    cooldown_applied: bool = False,
    latency_ms: Optional[float] = None,
    uncertainty: Optional[float] = None,
    fallback_used: Optional[bool] = None,
    component_breakdown: Optional[Dict[str, Any]] = None
) -> RuntimeOutput:
    """Create a clinician-mode output with extended metrics."""
    message = OutputValidator.sanitize_message(message)
    confidence = max(0.0, min(1.0, confidence))
    
    return RuntimeOutput(
        mode=OutputMode.CLINICIAN,
        severity=severity,
        message=message,
        confidence=confidence,
        cooldown_applied=cooldown_applied,
        latency_ms=latency_ms,
        uncertainty=uncertainty,
        fallback_used=fallback_used,
        component_breakdown=component_breakdown
    )


def create_dev_output(
    message: str,
    severity: Severity,
    confidence: float,
    cooldown_applied: bool = False,
    latency_ms: Optional[float] = None,
    uncertainty: Optional[float] = None,
    fallback_used: Optional[bool] = None,
    component_breakdown: Optional[Dict[str, Any]] = None,
    debug_info: Optional[Dict[str, Any]] = None
) -> RuntimeOutput:
    """Create a dev-mode output with full debug information."""
    message = OutputValidator.sanitize_message(message)
    confidence = max(0.0, min(1.0, confidence))
    
    return RuntimeOutput(
        mode=OutputMode.DEV,
        severity=severity,
        message=message,
        confidence=confidence,
        cooldown_applied=cooldown_applied,
        latency_ms=latency_ms,
        uncertainty=uncertainty,
        fallback_used=fallback_used,
        component_breakdown=component_breakdown,
        debug_info=debug_info
    )


# Backwards-compatible alias.
class OutputScheduler(CrossModalScheduler):
    """Backwards-compatible alias for CrossModalScheduler."""
    pass








