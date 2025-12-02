"""
    Model quantization for mobile deployment (int8).
    The key idea:
        Reduce latency for a higher efficiency
        Fast inference is cruical for the CNN
"""

import torch
import torch.nn as nn
import torch.ao.quantization as quantization
from typing import Optional, Dict, Any, Union, List, Tuple
from pathlib import Path
import warnings
from copy import deepcopy


def fuse_maxsight_modules(model: nn.Module) -> nn.Module:
    """
    Fuse Conv+BN+ReLU patterns in MaxSight CNN architecture.
    
    Automatically detects and fuses Sequential modules with Conv+BN+ReLU patterns.
    Works with:
    - SimplifiedFPN lateral_convs and fpn_convs
    - detection_fusion, detection_head
    - cls_head, box_head, obj_head, text_head
    - ResNet backbone layers (if applicable)
    
    Args:
        model: MaxSightCNN model to fuse
    
    Returns:
        Model with fused modules
    """
    fuse_list = []
    
    # Helper to check if a Sequential has Conv+BN+ReLU pattern
    def is_fusable_conv_bn_relu(seq: nn.Sequential, start_idx: int = 0) -> bool:
        if len(seq) < start_idx + 3:
            return False
        return (isinstance(seq[start_idx], nn.Conv2d) and
                isinstance(seq[start_idx + 1], nn.BatchNorm2d) and
                isinstance(seq[start_idx + 2], (nn.ReLU, nn.ReLU6)))
    
    # Traverse model and find fusable patterns
    for name, module in model.named_modules():
        if isinstance(module, nn.Sequential):
            # Check for Conv+BN+ReLU at start of Sequential
            if is_fusable_conv_bn_relu(module, 0):
                # Create fuse pattern: ['module_name.0', 'module_name.1', 'module_name.2']
                fuse_pattern = [f"{name}.0", f"{name}.1", f"{name}.2"]
                fuse_list.append(fuse_pattern)
    
    # Also handle ResNet backbone patterns if present
    # ResNet conv1, bn1, relu can be fused
    if hasattr(model, 'conv1') and hasattr(model, 'bn1') and hasattr(model, 'relu'):
        try:
            fuse_list.append(['conv1', 'bn1', 'relu'])
        except:
            pass
    
    # Fuse all detected patterns
    fused_count = 0
    for fuse_pattern in fuse_list:
        try:
            quantization.fuse_modules(model, [fuse_pattern], inplace=True)
            fused_count += 1
        except Exception as e:
            # Some patterns might not be fusable (e.g., already fused, wrong structure)
            continue
    
    if fused_count > 0:
        print(f" Fused {fused_count} Conv+BN+ReLU patterns for better performance")
    else:
        warnings.warn("No modules were fused. Model may not have standard Conv+BN+ReLU patterns.")
    
    return model


def quantize_model_int8(
    model: nn.Module,
    calibration_data: Optional[torch.utils.data.DataLoader] = None,
    num_calibration_batches: int = 10,
    backend: str = 'qnnpack',  # Default to ARM for iOS deployment
    fuse_modules: bool = True
) -> nn.Module:
    """
    Quantize model to int8.

    - MaxSight-specific module fusion
    - Per-channel weight quantization for ARM/iOS (qnnpack, this will be useful for the Native app dev)
    - Robust error handling (off of the quantization)
    
    Arguments:
        model: MaxSightCNN 
        calibration_data: DataLoader for calibration
        num_calibration_batches: Number of batches to use for calibration
        backend: Quantization backend ('qnnpack' for ARM/iOS, 'fbgemm' for x86)
        fuse_modules: Whether to fuse common module patterns (Conv+BN+ReLU)
    
    Returns:
        Quantized model ready for ExecuTorch export
    
    Usage:
        # Week 1 pipeline integration
        model_fp32 = create_model()
        model_fp32.load_state_dict(torch.load('checkpoint.pth'))
        
        # Quantize for iOS deployment
        model_int8 = quantize_model_int8(
            model_fp32,
            calibration_data=val_loader,
            backend='qnnpack',  # ARM backend for iPhone
            num_calibration_batches=20
        )
        
        # Validate accuracy
        validation = validate_quantized_model(model_fp32, model_int8, test_images)
        
        # Export to ExecuTorch (Week 2)
        # ... export to .pte format
    """
    # Create a copy to avoid modifying original model
    model = deepcopy(model)
    model.eval()
    
    # Set quantization backend
    torch.backends.quantized.engine = backend
    
    # Fuse modules if requested (MaxSight-specific fusion)
    if fuse_modules:
        try:
            model = fuse_maxsight_modules(model)
        except Exception as e:
            warnings.warn(f"Module fusion failed: {e}. Continuing without fusion.")
    
    # Set quantization config with proper API usage
    if backend == 'qnnpack':
        # ARM/iOS backend - REQUIRES per-channel weight quantization for stability
        model.qconfig = quantization.get_default_qconfig('qnnpack')  # type: ignore
        # Critical: per-channel weight quantization for mobile CNN stability
        if model.qconfig is not None:
            model.qconfig.weight = quantization.default_per_channel_weight_observer  # type: ignore
    elif backend == 'fbgemm':
        # x86 backend
        model.qconfig = quantization.get_default_qconfig('fbgemm')  # type: ignore
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'qnnpack' (ARM) or 'fbgemm' (x86)")
    
    # Prepare model for quantization - USE MODERN API
    model_prepared = quantization.prepare(model, inplace=False)
    
    # Calibrate with sample data
    print(f"Calibrating model for quantization using {num_calibration_batches} batches...")
    
    if calibration_data is None:
        # Create dummy calibration data with realistic variations
        print("Warning: No calibration data provided. Using synthetic data.")
        dummy_inputs = [torch.randn(1, 3, 224, 224) for _ in range(num_calibration_batches)]
        calibration_data = [(inp,) for inp in dummy_inputs]  # type: ignore
    
    batch_count = 0
    with torch.no_grad():
        for batch in calibration_data:  # type: ignore
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            elif isinstance(batch, dict):
                inputs = batch.get('images') or batch.get('input') or batch.get('data')
            else:
                inputs = batch
            
            # Move to CPU if necessary (quantization typically done on CPU)
            if inputs is not None and hasattr(inputs, 'device') and inputs.device.type != 'cpu':
                inputs = inputs.cpu()
            
            try:
                model_prepared(inputs)
                batch_count += 1
                if batch_count % 5 == 0:
                    print(f"  Processed {batch_count}/{num_calibration_batches} batches")
                if batch_count >= num_calibration_batches:
                    break
            except Exception as e:
                warnings.warn(f"Error processing batch {batch_count}: {e}")
                continue
    
    if batch_count == 0:
        raise RuntimeError("No batches successfully processed during calibration")
    
    # Convert to quantized model - USE MODERN API
    print("Converting to int8...")
    model_int8 = quantization.convert(model_prepared, inplace=False)
    
    print(f"✓ Quantization complete ({batch_count} calibration batches used)")
    print(f"  Backend: {backend} ({'ARM/iOS ready' if backend == 'qnnpack' else 'x86'})")
    return model_int8


def compare_model_sizes(
    model_fp32: nn.Module,
    model_int8: Optional[nn.Module] = None,
    save_models: bool = False,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Compare model sizes (FP32 vs INT8) with optional disk size measurement.
    
    Args:
        model_fp32: Original FP32 model
        model_int8: Quantized INT8 model
        save_models: Whether to save models and measure actual disk size
        output_dir: Directory to save models (if save_models=True)
    
    Returns:
        Dictionary with size information
    """
    # Count parameters
    total_params = sum(p.numel() for p in model_fp32.parameters())
    trainable_params = sum(p.numel() for p in model_fp32.parameters() if p.requires_grad)
    
    # Estimate FP32 size (4 bytes per parameter)
    fp32_size_mb = total_params * 4 / (1024 * 1024)
    
    results = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'fp32_size_mb': fp32_size_mb,
        'fp32_size_estimate': f"{fp32_size_mb:.2f} MB",
        'target_size_mb': 50.0,
        'compression_ratio': None,
        'int8_size_mb': None,
        'disk_sizes': {}
    }
    
    if model_int8 is not None:
        # INT8 uses 1 byte per parameter (theoretical)
        int8_size_mb = total_params / (1024 * 1024)
        results['int8_size_mb'] = int8_size_mb
        results['int8_size_estimate'] = f"{int8_size_mb:.2f} MB"
        results['compression_ratio'] = fp32_size_mb / int8_size_mb
        results['meets_target'] = int8_size_mb < 50.0
        results['size_reduction'] = f"{((fp32_size_mb - int8_size_mb) / fp32_size_mb * 100):.1f}%"
    
    # Measure actual disk sizes if requested
    if save_models and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FP32 model
        fp32_path = output_dir / "model_fp32.pth"
        torch.save(model_fp32.state_dict(), fp32_path)
        fp32_disk_size = fp32_path.stat().st_size / (1024 * 1024)
        results['disk_sizes']['fp32'] = f"{fp32_disk_size:.2f} MB"
        
        # Save INT8 model if available
        if model_int8 is not None:
            int8_path = output_dir / "model_int8.pth"
            torch.save(model_int8.state_dict(), int8_path)
            int8_disk_size = int8_path.stat().st_size / (1024 * 1024)
            results['disk_sizes']['int8'] = f"{int8_disk_size:.2f} MB"
            results['disk_compression_ratio'] = fp32_disk_size / int8_disk_size
    
    return results

# Find the various computation metrics between two tensors
def compute_tensor_differences(
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
        metrics: List[str]
) -> Dict[str, float]: 
    t1_float = tensor1.float()
    t2_float = tensor2.float()
    
    # The differences in metric quantity
    diff = torch.abs(t1_float - t2_float)
    rel_diff = diff / (torch.abs(t1_float) + 1e-8)
    max_rel_diff = rel_diff.max().item()

    result = {"max_relative_dif": max_rel_diff}

    if 'mse' in metrics:
        mse = torch.mean((t1_float - t2_float) ** 2).item()
        result['mse'] = mse

    if 'mae' in metrics:
        mae = torch.mean((t1_float - t2_float) ** 2).item()
        result['mae'] = mae

    # Cosine similarity
    if 'cosine' in metrics:
        t1_flat = t1_float.flatten()
        t2_flat = t2_float.flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(
            t1_flat.unsqueeze(0), t2_flat.unsqueeze(0)
        ).item()
        result['cosine_similarity'] = cosine_sim

    return result

def print_quantization(
        size_info: Dict[str, Any],
        validation: Dict[str, Any],
        verbose: bool = True
) -> None:
    print("\nModel Quantization Results")
    print("\nModel Statistics")
    print(f" Total Parameters: {size_info['total_parameters']:,}")
    print(f" Trainable Parameters: {size_info['trainable_parameters']:,}")

    print("\nSize Comparison FP32 size:")
    if size_info['int8_size_mb'] is not None:
        print(f"  INT8 Size (est):  {size_info['int8_size_mb']:.1f} MB")
        print(f"  Compression:      {size_info['compression_ratio']:.1f}x")
        print(f"  Size Reduction:   {size_info.get('size_reduction', 'N/A')}")
        print(f"  Target:           <50 MB")
        print(f"  Status:           {'Match' if size_info.get('meets_target', False) else 'No match'}")

    if size_info.get('disk_sizes'):
        print("\nActual Disk Sizes:")
        for model_type, size in size_info['disk_sizes'].items():
            print(f" {model_type.upper()}: {size}")
    
    print("\nAccuracy Validation:")
    if 'accuracy_loss_percent' in validation:
        if 'avg_relative_difference' in validation:
            print(f"  Max Accuracy Loss:  {validation['accuracy_loss_percent']:.2f}%")
            print(f"  Avg Accuracy Loss:  {validation['avg_relative_difference'] * 100:.2f}%")
        else:
            print(f"  Accuracy Loss:      {validation['accuracy_loss_percent']:.2f}%")
        print(f"  Target:             <1%")
        print(f"  Status:             {'Met' if validation.get('meets_tolerance', False) else 'Not met'}")
        
        if verbose and 'additional_metrics' in validation:
            print("\nDetailed Metrics:")
            for metric, values in validation['additional_metrics'].items():
                if isinstance(values, dict):
                    print(f"  {metric.upper()}:")
                    for key, val in values.items():
                        print(f"    {key}: {val:.6f}")
                elif values is not None:
                    print(f"  {metric.upper()}: {values:.6f}")
    else:
        print(f"  Error: {validation.get('error', 'Unknown error')}")    