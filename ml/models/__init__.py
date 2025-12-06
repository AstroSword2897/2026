# Import all public training components for convenient access
from .train import Trainer  # Advanced trainer with all features
from .losses import MaxSightLoss  # Multi-task loss function
from .train_production import ProductionTrainer, create_dummy_dataloaders  # Production trainer and test data
from .export import (  # Export functions for iOS deployment
    export_model,
    export_to_jit,
    export_to_executorch,
    export_to_coreml,
    export_to_onnx
)

# Define public API - these are the symbols exported when doing: from ml.training import *
__all__ = [
    'Trainer',  # Advanced training class
    'MaxSightLoss',  # Loss function
    'ProductionTrainer',  # Production-ready trainer
    'create_dummy_dataloaders',  # Test data generator
    'export_model',  # Main export function
    'export_to_jit',  # JIT export
    'export_to_executorch',  # ExecuTorch export
    'export_to_coreml',  # CoreML export
    'export_to_onnx',  # ONNX export
]

