"""
Production-grade configuration management for MaxSight.

Centralized configuration with:
- Environment variable support
- Type validation
- Default values
- Documentation
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """Training configuration with validation."""
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 100
    batch_size: int = 32
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    use_mixed_precision: bool = True
    ema_decay: float = 0.9999
    scheduler_type: str = 'cosine'
    warmup_epochs: int = 5
    freeze_backbone_epochs: int = 0
    freeze_bn_stats: bool = True
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be > 0, got {self.num_epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.scheduler_type not in ['cosine', 'onecycle', 'cosine_restarts', 'constant']:
            raise ValueError(f"Invalid scheduler_type: {self.scheduler_type}")


@dataclass
class ModelConfig:
    """Model configuration with validation."""
    num_classes: int = 48
    num_urgency_levels: int = 4
    num_distance_zones: int = 3
    use_audio: bool = True
    condition_mode: Optional[str] = None
    fpn_channels: int = 256
    detection_threshold: float = 0.5
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be > 0, got {self.num_classes}")
        if self.num_urgency_levels <= 0:
            raise ValueError(f"num_urgency_levels must be > 0, got {self.num_urgency_levels}")
        if self.detection_threshold < 0 or self.detection_threshold > 1:
            raise ValueError(f"detection_threshold must be in [0, 1], got {self.detection_threshold}")


@dataclass
class DataConfig:
    """Data configuration with validation."""
    data_dir: Path = Path("datasets")
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True
    
    def __post_init__(self):
        """Validate configuration values."""
        if not (0 < self.train_split < 1):
            raise ValueError(f"train_split must be in (0, 1), got {self.train_split}")
        if abs(self.train_split + self.val_split + self.test_split - 1.0) > 1e-6:
            raise ValueError("train_split + val_split + test_split must equal 1.0")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")


@dataclass
class SpatialMemoryConfig:
    """Spatial memory configuration."""
    memory_duration: float = 30.0  # seconds
    position_threshold: float = 0.1  # normalized distance
    stability_threshold: float = 0.7  # minimum stability score
    history_window: float = 10.0  # seconds for position history
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.memory_duration <= 0:
            raise ValueError(f"memory_duration must be > 0, got {self.memory_duration}")
        if self.position_threshold <= 0:
            raise ValueError(f"position_threshold must be > 0, got {self.position_threshold}")


@dataclass
class OCRConfig:
    """OCR configuration."""
    text_threshold: float = 0.5
    max_text_regions: int = 10
    cluster_distance: int = 10
    ocr_timeout: float = 5.0  # seconds
    max_workers: int = 4  # for parallel OCR
    
    def __post_init__(self):
        """Validate configuration values."""
        if not (0 <= self.text_threshold <= 1):
            raise ValueError(f"text_threshold must be in [0, 1], got {self.text_threshold}")
        if self.max_text_regions <= 0:
            raise ValueError(f"max_text_regions must be > 0, got {self.max_text_regions}")


@dataclass
class OutputSchedulingConfig:
    """Output scheduling configuration."""
    min_channel_interval: float = 0.3  # seconds between outputs
    uncertainty_threshold: float = 0.7
    emergency_priority: int = 90
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.min_channel_interval <= 0:
            raise ValueError(f"min_channel_interval must be > 0, got {self.min_channel_interval}")
        if not (0 <= self.uncertainty_threshold <= 1):
            raise ValueError(f"uncertainty_threshold must be in [0, 1], got {self.uncertainty_threshold}")


@dataclass
class RuntimeConfig:
    """Runtime configuration for MaxSight system."""
    spatial_memory: SpatialMemoryConfig = field(default_factory=SpatialMemoryConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    output_scheduling: OutputSchedulingConfig = field(default_factory=OutputSchedulingConfig)
    
    @classmethod
    def from_file(cls, path: Path) -> 'RuntimeConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            spatial_memory=SpatialMemoryConfig(**data.get('spatial_memory', {})),
            ocr=OCRConfig(**data.get('ocr', {})),
            output_scheduling=OutputSchedulingConfig(**data.get('output_scheduling', {}))
        )
    
    def to_file(self, path: Path) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump({
                'spatial_memory': self.spatial_memory.__dict__,
                'ocr': self.ocr.__dict__,
                'output_scheduling': self.output_scheduling.__dict__
            }, f, indent=2)


@dataclass
class Config:
    """Main configuration class."""
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    device: str = 'cuda'
    seed: int = 42
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables."""
        return cls(
            training=TrainingConfig(
                learning_rate=float(os.getenv('LEARNING_RATE', '1e-3')),
                num_epochs=int(os.getenv('NUM_EPOCHS', '100')),
                batch_size=int(os.getenv('BATCH_SIZE', '32')),
                use_mixed_precision=os.getenv('USE_MIXED_PRECISION', 'true').lower() == 'true'
            ),
            model=ModelConfig(
                num_classes=int(os.getenv('NUM_CLASSES', '48')),
                use_audio=os.getenv('USE_AUDIO', 'true').lower() == 'true',
                condition_mode=os.getenv('CONDITION_MODE', None)
            ),
            data=DataConfig(
                data_dir=Path(os.getenv('DATA_DIR', 'datasets')),
                num_workers=int(os.getenv('NUM_WORKERS', '4'))
            ),
            device=os.getenv('DEVICE', 'cuda'),
            seed=int(os.getenv('SEED', '42')),
            checkpoint_dir=Path(os.getenv('CHECKPOINT_DIR', 'checkpoints')),
            log_dir=Path(os.getenv('LOG_DIR', 'logs'))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'training': self.training.__dict__,
            'model': self.model.__dict__,
            'data': {
                **self.data.__dict__,
                'data_dir': str(self.data.data_dir)
            },
            'device': self.device,
            'seed': self.seed,
            'checkpoint_dir': str(self.checkpoint_dir),
            'log_dir': str(self.log_dir)
        }

