"""Configuration for MaxSight Web Simulator. Centralizes all magic numbers and settings."""
import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from ml.utils.output_scheduler import OutputMode


@dataclass
class SimulatorConfig:
    """Centralized configuration for the simulator."""
    
    # Server settings.
    host: str = '0.0.0.0'
    port: int = 8002
    debug: bool = True
    
    # Multi-user settings.
    multi_user_enabled: bool = True
    session_timeout_seconds: int = 30 * 60  # 30 minutes.
    
    # Processing settings.
    confidence_threshold: float = 0.3
    max_ocr_texts_in_description: int = 3
    therapy_difficulty: float = 0.5
    urgency_warning_threshold: int = 2
    max_alerts_per_frame: int = 5
    alert_cooldown_frames: int = 5
    
    # Haptic/voice settings.
    haptic_intensity_high: float = 0.7
    haptic_intensity_low: float = 0.3
    
    # Baseline output.
    baseline_save_frame: int = 1
    
    # Rate limiting (requests per minute)
    rate_limit_per_session: int = 60  # 60 requests/minute per session.
    rate_limit_global: int = 1000  # 1000 requests/minute globally.
    
    # Input validation.
    max_image_size_mb: int = 10  # 10MB max.
    allowed_image_formats: tuple = ('JPEG', 'PNG', 'GIF', 'BMP', 'WEBP', 'TIFF')
    
    # Logging.
    log_level: str = 'INFO'
    enable_structured_logging: bool = True
    
    # Monitoring.
    enable_metrics: bool = True
    metrics_port: Optional[int] = None  # None = disabled.
    
    # Default output mode.
    default_output_mode: OutputMode = OutputMode.PATIENT

    model_checkpoint_path: Optional[str] = None  # Default: use glaucoma model if available.
    
    # Confidence gating (patient safety)
    min_confidence_for_patient_output: float = 0.5  # Don't show low-confidence results.
    min_confidence_for_critical_alert: float = 0.7  # Higher bar for critical alerts.
    
    # Resource caps per session.
    max_spatial_memory_entries: int = 1000  # Max objects in spatial memory.
    max_history_depth: int = 100  # Max frames to keep in history.
    max_memory_mb_per_session: int = 500  # Max memory per session (MB)
    
    # Queue settings.
    voice_queue_maxsize: int = 10  # Bounded voice queue.
    haptic_queue_maxsize: int = 10  # Bounded haptic queue.
    
    # Demo assumptions (documented)
    demo_assumptions = {
        'single_camera': True,  # Assumes single camera input.
        'single_user_per_session': True,  # One user per session.
        'stable_lighting': False,  # May vary.
        'no_adversarial_input': True,  # Trusts user input.
        'local_network_only': True,  # Not exposed to internet.
        'development_mode': True  # Not production-hardened.
    }


# Global config instance.
config = SimulatorConfig()

# Checkpoint path is taken from environment when set.
if os.getenv('MAXSIGHT_CHECKPOINT_PATH'):
    checkpoint_path = Path(os.getenv('MAXSIGHT_CHECKPOINT_PATH')).expanduser().resolve()
    if checkpoint_path.exists():
        config.model_checkpoint_path = str(checkpoint_path)
    else:
        import warnings
        warnings.warn(f"Checkpoint path from environment variable does not exist: {checkpoint_path}")







