"""
Production quantization tools for MaxSight models.

This package provides:
- QAT (Quantization-Aware Training) for fine-tuning quantized models
- Validation and benchmarking tools for comparing FP32 vs INT8 models
"""

from .qat_finetune import QATTrainer, fuse_maxsight_model
from .validate_and_bench import QuantizationValidator, ModelBenchmark

__all__ = [
    'QATTrainer',
    'fuse_maxsight_model',
    'QuantizationValidator',
    'ModelBenchmark',
]

