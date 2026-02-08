"""MaxSight Configuration and Dependency Management Centralized configuration with versioning and dependency tracking."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from datetime import datetime


@dataclass
class ModelConfig:
    """Model architecture configuration with versioning."""
    version: str = "1.0.0"
    num_classes: int = 80
    num_urgency_levels: int = 4
    num_distance_zones: int = 3
    fpn_channels: int = 256
    detection_threshold: float = 0.5
    enable_accessibility_features: bool = True
    
    # Head dependencies.
    head_dependencies: Dict[str, List[str]] = field(default_factory=lambda: {
        'classification': [],
        'box_regression': ['classification'],
        'objectness': [],
        'text_region': [],
        'urgency': ['scene_embedding'],
        'distance': ['scene_embedding', 'box_regression'],
        'contrast': ['shared_scene_embedding'],
        'glare': ['shared_scene_embedding'],
        'findability': ['detection_features'],
        'navigation_difficulty': ['shared_scene_embedding'],
        'uncertainty': ['shared_scene_embedding'],
    })
    
    # Head execution order (for dependency resolution)
    head_execution_order: List[str] = field(default_factory=lambda: [
        'scene_embedding',
        'shared_scene_embedding',
        'detection_features',
        'classification',
        'box_regression',
        'objectness',
        'text_region',
        'urgency',
        'distance',
        'contrast',
        'glare',
        'findability',
        'navigation_difficulty',
        'uncertainty',
    ])


@dataclass
class RuntimeConfig:
    """Runtime configuration for inference and deployment."""
    # Performance constraints.
    max_latency_ms: float = 500.0  # Target: <500ms for mobile.
    max_memory_mb: float = 50.0    # Target: <50MB quantized.
    
    # Head execution.
    enable_all_heads: bool = True
    enabled_heads: List[str] = field(default_factory=lambda: [
        'classification', 'box_regression', 'objectness', 'text_region',
        'urgency', 'distance', 'contrast', 'glare', 'findability',
        'navigation_difficulty', 'uncertainty'
    ])
    
    # Fallback configuration.
    enable_fallbacks: bool = True
    fallback_on_error: bool = True
    fallback_on_uncertainty: bool = True
    uncertainty_threshold: float = 0.7  # If uncertainty > 0.7, use fallback.
    
    # Error handling.
    max_retries: int = 1
    timeout_ms: float = 1000.0


@dataclass
class DependencyGraph:
    """Tracks dependencies between components."""
    components: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'model': {
            'version': '1.0.0',
            'dependencies': [],
            'outputs': ['classifications', 'boxes', 'objectness', 'text_regions', 
                       'urgency_scores', 'distance_zones', 'scene_embedding']
        },
        'preprocessing': {
            'version': '1.0.0',
            'dependencies': [],
            'outputs': ['preprocessed_image']
        },
        'ocr': {
            'version': '1.0.0',
            'dependencies': ['model.text_regions'],
            'outputs': ['text_results']
        },
        'description_generator': {
            'version': '1.0.0',
            'dependencies': ['model.detections', 'model.urgency_scores', 'ocr.text_results'],
            'outputs': ['scene_description']
        },
        'spatial_memory': {
            'version': '1.0.0',
            'dependencies': ['model.detections'],
            'outputs': ['spatial_context']
        },
        'path_planner': {
            'version': '1.0.0',
            'dependencies': ['spatial_memory.spatial_context'],
            'outputs': ['path_info']
        },
        'output_scheduler': {
            'version': '1.0.0',
            'dependencies': ['model.detections', 'model.urgency_scores', 'model.uncertainty'],
            'outputs': ['scheduled_outputs']
        },
        'therapy_integration': {
            'version': '1.0.0',
            'dependencies': ['model.detections', 'session_manager'],
            'outputs': ['therapy_feedback']
        },
    })
    
    def get_dependencies(self, component: str) -> List[str]:
        """Get dependencies for a component."""
        if component in self.components:
            return self.components[component].get('dependencies', [])
        return []
    
    def validate_dependencies(self, outputs: Dict[str, Any]) -> Dict[str, bool]:
        """Validate that all dependencies are available."""
        validation = {}
        for component, config in self.components.items():
            deps = config.get('dependencies', [])
            valid = True
            for dep in deps:
                # Check if dependency output exists.
                dep_parts = dep.split('.')
                if len(dep_parts) == 2:
                    source, output_key = dep_parts
                    if source not in outputs or output_key not in outputs[source]:
                        valid = False
                        break
            validation[component] = valid
        return validation


def get_config() -> ModelConfig:
    """Get current model configuration."""
    return ModelConfig()


def get_runtime_config() -> RuntimeConfig:
    """Get current runtime configuration."""
    return RuntimeConfig()


def save_config(config: ModelConfig, filepath: Path):
    """Save configuration with versioning."""
    config_dict = {
        'version': config.version,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'num_classes': config.num_classes,
            'num_urgency_levels': config.num_urgency_levels,
            'num_distance_zones': config.num_distance_zones,
            'fpn_channels': config.fpn_channels,
            'detection_threshold': config.detection_threshold,
            'enable_accessibility_features': config.enable_accessibility_features,
            'head_dependencies': config.head_dependencies,
            'head_execution_order': config.head_execution_order,
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config(filepath: Path) -> ModelConfig:
    """Load configuration from file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    config_data = data.get('config', {})
    return ModelConfig(
        version=data.get('version', '1.0.0'),
        num_classes=config_data.get('num_classes', 80),
        num_urgency_levels=config_data.get('num_urgency_levels', 4),
        num_distance_zones=config_data.get('num_distance_zones', 3),
        fpn_channels=config_data.get('fpn_channels', 256),
        detection_threshold=config_data.get('detection_threshold', 0.5),
        enable_accessibility_features=config_data.get('enable_accessibility_features', True),
        head_dependencies=config_data.get('head_dependencies', {}),
        head_execution_order=config_data.get('head_execution_order', [])
    )






