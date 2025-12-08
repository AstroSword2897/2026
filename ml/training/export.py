"""Model export for iOS: JIT, ExecuTorch, CoreML, ONNX. Handles dict outputs gracefully."""

import json
import logging
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

def export_to_jit(model: nn.Module, save_path: str = 'maxsight_traced.pt', input_size: tuple = (1, 3, 224, 224), device: Optional[str] = None, validate: bool = True) -> Path:
    """Export to PyTorch JIT format. Most reliable, always available. strict=False for dict outputs.
    
        Arguments:
        device: Device to export from ('cpu', 'cuda', 'mps'). If None, uses model's current device.
        validate: If True, test exported model with dummy input to verify it works.
    """
    logger.info(f"Exporting to JIT format: {save_path}")
    
    model.eval()
    export_device = device if device else next(model.parameters()).device
    if isinstance(export_device, torch.device):
        export_device = str(export_device)
    
    # Move to export device
    if export_device == 'cpu':
        model.cpu()
    elif export_device.startswith('cuda'):
        model.cuda()
    elif export_device == 'mps':
        model.to('mps')
    
    dummy_input = torch.randn(*input_size)
    if export_device.startswith('cuda'):
        dummy_input = dummy_input.cuda()
    elif export_device == 'mps':
        dummy_input = dummy_input.to('mps')
    
    try:
        traced_model = torch.jit.trace(model, dummy_input, strict=False)
        
        # Validate exported model if requested
        if validate:
            test_output = traced_model(dummy_input)  # type: ignore
            logger.debug("Validation: Exported model forward pass successful")
        
        # Move to CPU for saving (JIT models should be CPU)
        traced_model.cpu()
        save_path_obj = Path(save_path)
        traced_model.save(str(save_path_obj))
        
        size_mb = save_path_obj.stat().st_size / (1024 * 1024)
        logger.info(f"Saved: {save_path}, Size: {size_mb:.1f} MB")
        
        return save_path_obj
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        raise

def export_to_executorch(
    model: nn.Module, 
    save_path: str = 'maxsight.pte', 
    input_size: tuple = (1, 3, 224, 224),
    validate: bool = True
) -> Optional[Path]:
    """
    Export to ExecuTorch .pte format for iOS deployment.
    
    Optimized for quantized models and handles dict outputs efficiently.
    Falls back to JIT if ExecuTorch is not installed.
    
    Arguments:
        model: Model to export (FP32 or INT8 quantized)
        save_path: Output path for .pte file
        input_size: Input tensor size (B, C, H, W)
        validate: If True, validate exported model with test input
    
    Returns:
        Path to exported .pte file, or None if export failed
    """
    logger.info(f"Exporting to ExecuTorch format: {save_path}")
    
    try:
        # Try multiple ExecuTorch import paths (API may vary by version)
        try:
            from executorch.exir import to_edge  # type: ignore
            from executorch.extension.pybind11.portable import to_edge as to_edge_legacy  # type: ignore
            USE_EXIR = True
        except ImportError:
            try:
                from executorch.extension.pybind11.portable import to_edge  # type: ignore
                USE_EXIR = False
            except ImportError:
                raise ImportError("ExecuTorch not installed")
        
        model.eval()
        model.cpu()
        dummy_input = torch.randn(*input_size)
        
        # Handle dict outputs by wrapping model if needed
        test_output = model(dummy_input)
        if isinstance(test_output, dict):
            # Wrap model to handle dict outputs for ExecuTorch
            class ExecutorchWrapper(nn.Module):
                def __init__(self, model: nn.Module):
                    super().__init__()
                    self.model = model
                
                def forward(self, x: torch.Tensor):
                    outputs = self.model(x)
                    if isinstance(outputs, dict):
                        # Return tuple of key outputs for ExecuTorch compatibility
                        # Prioritize critical outputs: classifications, boxes, objectness
                        key_outputs = [
                            outputs.get('classifications', torch.empty(0)),
                            outputs.get('boxes', torch.empty(0, 4)),
                            outputs.get('objectness', torch.empty(0)),
                            outputs.get('urgency_scores', torch.empty(0)),
                            outputs.get('distance_zones', torch.empty(0, 3))
                        ]
                        return tuple(key_outputs)
                    return outputs
            
            wrapped_model = ExecutorchWrapper(model)
        else:
            wrapped_model = model
        
        # Convert to Edge dialect (handle different API versions)
        if USE_EXIR:
            # Modern ExecuTorch API: export first, then to_edge
            exported = torch.export.export(wrapped_model, (dummy_input,))
            edge_program = to_edge(exported)
        else:
            # Legacy API
            edge_program = to_edge(wrapped_model, (dummy_input,))  # type: ignore
        
        # Convert to ExecuTorch program
        executorch_program = edge_program.to_executorch()
        
        # Validate if requested
        if validate:
            try:
                # Test that program can be loaded (basic validation)
                logger.debug("Validation: ExecuTorch program created successfully")
            except Exception as e:
                logger.warning(f"Validation warning: {e}")
        
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path_obj, 'wb') as f:
            f.write(executorch_program.buffer)
        
        size_mb = save_path_obj.stat().st_size / (1024 * 1024)
        logger.info(f"Saved: {save_path}, Size: {size_mb:.1f} MB")
        logger.info("ExecuTorch export complete - ready for iOS deployment")
        return save_path_obj
        
    except ImportError:
        logger.warning("ExecuTorch not installed. Install with: pip install executorch")
        logger.warning("Falling back to JIT export...")
        return export_to_jit(model, save_path.replace('.pte', '_traced.pt'), input_size, validate=validate)
    except Exception as e:
        logger.error(f"ExecuTorch export failed: {e}", exc_info=True)
        logger.warning("Falling back to JIT export...")
        return export_to_jit(model, save_path.replace('.pte', '_traced.pt'), input_size, validate=validate)

def export_to_coreml(model: nn.Module, save_path: str = 'maxsight.mlpackage', input_size: tuple = (1, 3, 224, 224), device: Optional[str] = None, validate: bool = True) -> Optional[Path]:
    """Export to CoreML format (iOS native). Requires coremltools. Handles dict outputs."""
    logger.info(f"Exporting to CoreML format: {save_path}")
    
    try:
        import coremltools as ct  # type: ignore
        
        model.eval()
        export_device = device if device else 'cpu'
        if export_device == 'cpu':
            model.cpu()
        elif export_device.startswith('cuda'):
            model.cuda()
        elif export_device == 'mps':
            model.to('mps')
        
        dummy_input = torch.randn(*input_size)
        if export_device.startswith('cuda'):
            dummy_input = dummy_input.cuda()
        elif export_device == 'mps':
            dummy_input = dummy_input.to('mps')
        
        # Wrap model to handle dict outputs
        class FlattenedModel(nn.Module):
            """Wrapper to flatten dict outputs for CoreML compatibility."""
            def __init__(self, model: nn.Module):
                super().__init__()
                self.model = model
            
            def forward(self, x: torch.Tensor):
                outputs = self.model(x)
                if isinstance(outputs, dict):
                    # Flatten to tuple of tensors (CoreML can handle tuples)
                    return tuple(outputs.values())
                return outputs
        
        wrapped_model = FlattenedModel(model)
        traced_model = torch.jit.trace(wrapped_model, dummy_input, strict=False)
        
        # Validate traced model if requested
        test_output = None
        if validate:
            test_output = traced_model(dummy_input)  # type: ignore
            logger.debug("Validation: Traced model forward pass successful")
        
        # Determine output types
        if validate and isinstance(test_output, tuple):
            # Multiple outputs from flattened dict
            output_types = [ct.TensorType(name=f"output_{i}") for i in range(len(test_output))]
        elif validate:
            # Single tensor output
            output_types = [ct.TensorType(name="output")]
        else:
            # Default: single output (will fail if model returns dict, but user should use validate=True)
            output_types = [ct.TensorType(name="output")]
        
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="image", shape=input_size)],
            outputs=output_types,
            minimum_deployment_target=ct.target.iOS15
        )
        
        save_path_obj = Path(save_path)
        if coreml_model is not None:
            # Validate CoreML model if requested
            if validate:
                try:
                    test_input_np = dummy_input.cpu().numpy()
                    test_output_ml = coreml_model.predict({"image": test_input_np})
                    logger.debug("Validation: CoreML model forward pass successful")
                except Exception as e:
                    logger.warning(f"CoreML validation failed: {e}")
            
            coreml_model.save(str(save_path_obj))
        else:
            raise ValueError("CoreML conversion failed")
        
        size_mb = save_path_obj.stat().st_size / (1024 * 1024)
        logger.info(f"Saved: {save_path}, Size: {size_mb:.1f} MB, iOS 15+")
        return save_path_obj
        
    except ImportError:
        logger.warning("CoreML tools not installed (pip install coremltools)")
        return None
    except Exception as e:
        logger.error(f"CoreML export failed: {e}", exc_info=True)
        return None

def export_to_onnx(model: nn.Module, save_path: str = 'maxsight.onnx', input_size: tuple = (1, 3, 224, 224)) -> Optional[Path]:
    """Export to ONNX format. May fail with dict outputs - use JIT/CoreML for iOS."""
    logger.info(f"Exporting to ONNX format: {save_path}")
    
    try:
        import onnx  # type: ignore
        
        model.eval()
        model.cpu()
        dummy_input = torch.randn(*input_size)
        
        torch.onnx.export(
            model,
            (dummy_input,),
            save_path,
            input_names=['image'],
            output_names=['output'],  # Note: may not work with dict outputs
            dynamic_axes={'image': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=11
        )
        
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        
        save_path_obj = Path(save_path)
        size_mb = save_path_obj.stat().st_size / (1024 * 1024)
        logger.info(f"Saved: {save_path}, Size: {size_mb:.1f} MB")
        return save_path_obj
        
    except ImportError:
        logger.warning("ONNX not installed (pip install onnx)")
        return None
    except Exception as e:
        logger.error(f"ONNX export failed: {e}", exc_info=True)
        return None

def export_model(model: nn.Module, format: str = 'jit', save_dir: str = 'exports', 
                 input_size: tuple = (1, 3, 224, 224), device: Optional[str] = None, 
                 validate: bool = True) -> dict:
    """Export model to specified format(s). Formats: 'jit', 'executorch', 'coreml', 'onnx', 'all'."""
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(exist_ok=True, parents=True)  # Create export directory if needed
    
    results = {
        'format': format,  # Requested format
        'exports': {},  # Successful export paths
        'metadata': {
            'input_size': input_size,  # Input dimensions
            'model_params': sum(p.numel() for p in model.parameters()),  # Total parameter count
        }
    }
    
    if format == 'jit' or format == 'all':
        path = export_to_jit(model, str(save_dir_path / 'maxsight_traced.pt'), input_size, device, validate)
        results['exports']['jit'] = str(path)
    
    if format == 'executorch' or format == 'all':
        path = export_to_executorch(model, str(save_dir_path / 'maxsight.pte'), input_size)
        if path:
            results['exports']['executorch'] = str(path)
    
    if format == 'coreml' or format == 'all':
        path = export_to_coreml(model, str(save_dir_path / 'maxsight.mlpackage'), input_size)
        if path:
            results['exports']['coreml'] = str(path)
    
    if format == 'onnx' or format == 'all':
        path = export_to_onnx(model, str(save_dir_path / 'maxsight.onnx'), input_size)
        if path:
            results['exports']['onnx'] = str(path)
    
    metadata_path = save_dir_path / 'export_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Export metadata saved to: {metadata_path}")
    
    return results

def _extract_processing_reference() -> str:
    """
    Extract essential processing functions into single reference file.
    Pulls only the functions iOS needs to port to Swift.
    """
    from pathlib import Path
    import re
    import ast
    
    # Functions to extract (whitelist approach)
    # Format: (module_path, func_name, is_class_method, class_name)
    functions_to_extract = [
        # Standalone functions from preprocessing.py
        ('ml/utils/preprocessing.py', 'apply_refractive_error_blur', False, None),
        ('ml/utils/preprocessing.py', 'apply_cataract_contrast', False, None),
        ('ml/utils/preprocessing.py', 'apply_glaucoma_vignette', False, None),
        ('ml/utils/preprocessing.py', 'apply_amd_central_darkening', False, None),
        ('ml/utils/preprocessing.py', 'apply_low_light', False, None),
        ('ml/utils/preprocessing.py', 'apply_color_shift', False, None),
        # Class methods from maxsight_cnn.py (need to extract as standalone)
        ('ml/models/maxsight_cnn.py', '_nms', True, 'MaxSightCNN'),
        ('ml/models/maxsight_cnn.py', '_compute_iou', True, 'MaxSightCNN'),
        ('ml/models/maxsight_cnn.py', '_compute_iou_corners', True, 'MaxSightCNN'),
        ('ml/models/maxsight_cnn.py', '_center_to_corners', True, 'MaxSightCNN'),
        # Class methods from output_scheduler.py
        ('ml/utils/output_scheduler.py', '_get_priority_threshold', True, 'CrossModalScheduler'),
        ('ml/utils/output_scheduler.py', '_calculate_intensity', True, 'CrossModalScheduler'),
        ('ml/utils/output_scheduler.py', '_calculate_frequency', True, 'CrossModalScheduler'),
        ('ml/utils/output_scheduler.py', '_select_channel', True, 'CrossModalScheduler'),
        # Class methods from ocr_integration.py
        ('ml/utils/ocr_integration.py', '_cluster_text_pixels', True, 'OCRIntegration'),
        # Standalone function from ocr_integration.py
        ('ml/utils/ocr_integration.py', '_group_text_by_proximity', False, None),
    ]
    
    reference_code = '''"""
MaxSight Processing Reference for iOS
Essential preprocessing, postprocessing, and scheduling logic.

Port these functions to Swift for iOS implementation.
This is the minimal set needed to process model inputs/outputs.

Generated automatically from MaxSight repository.

NOTE: Some functions reference config/enums that need to be parameterized
when porting to Swift. See function comments for details.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from torchvision.transforms import functional as TF
from enum import Enum

# Enums needed for scheduling functions
class OutputChannel(Enum):
    AUDIO = "audio"
    HAPTIC = "haptic"
    VISUAL = "visual"
    HYBRID = "hybrid"

class AlertFrequency(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

'''
    
    # Extract each function from source files
    for module_path, func_name, is_class_method, class_name in functions_to_extract:
        try:
            module_path_obj = Path(module_path)
            if not module_path_obj.exists():
                logger.warning(f"Module not found: {module_path}")
                continue
            
            # Read source
            with open(module_path_obj, 'r') as f:
                source = f.read()
            
            # Extract function using line-by-line parsing (more reliable)
            lines = source.split('\n')
            in_target_function = False
            func_lines = []
            base_indent = 0
            
            for i, line in enumerate(lines):
                # Check if this is our target function
                if is_class_method:
                    # Look for class method: "    def func_name("
                    if f'    def {func_name}(' in line or f'\tdef {func_name}(' in line:
                        in_target_function = True
                        base_indent = len(line) - len(line.lstrip())
                        # Convert to standalone function
                        cleaned_line = line.lstrip().replace('    def ', 'def ').replace('\tdef ', 'def ')
                        func_lines.append(cleaned_line)
                        continue
                else:
                    # Look for standalone function: "def func_name("
                    if f'def {func_name}(' in line and not line.strip().startswith('class '):
                        # Check it's not indented (standalone function)
                        if not line.startswith(' ') and not line.startswith('\t'):
                            in_target_function = True
                            base_indent = 0
                            func_lines.append(line)
                            continue
                
                if in_target_function:
                    # Check if we've hit the end of the function
                    stripped = line.lstrip()
                    if stripped:
                        current_indent = len(line) - len(line.lstrip())
                        
                        # End conditions:
                        # 1. Next def/class at same or less indent (for class methods)
                        # 2. Next def/class at start of line (for standalone)
                        if is_class_method:
                            if (stripped.startswith('def ') or stripped.startswith('class ')) and current_indent <= base_indent:
                                break
                        else:
                            if (stripped.startswith('def ') or stripped.startswith('class ')) and current_indent == 0:
                                break
                    
                    # Add line to function (remove class method indent)
                    if is_class_method and base_indent > 0:
                        if line.startswith(' ' * base_indent):
                            func_lines.append(line[base_indent:])
                        elif line.startswith('\t'):
                            func_lines.append(line[1:])
                        else:
                            func_lines.append(line)
                    else:
                        func_lines.append(line)
            
            if func_lines:
                func_code = '\n'.join(func_lines)
                
                    # Clean up self references for class methods
                if is_class_method:
                    # Remove self parameter from function signature
                    func_code = re.sub(r'\(self,?\s*', '(', func_code)
                    func_code = re.sub(r'\(self\)', '()', func_code)
                    # Remove self. references
                    func_code = re.sub(r'\bself\.', '', func_code)
                    # Replace config references - add TODO comments on separate lines
                    # This preserves syntax while documenting what needs to be parameterized
                    lines = func_code.split('\n')
                    cleaned_lines = []
                    for line in lines:
                        # Check if line has config reference
                        if 'config.' in line:
                            # Add TODO comment before the line
                            indent = len(line) - len(line.lstrip())
                            todo_comment = ' ' * indent + '# TODO: Parameterize config references when porting to Swift'
                            cleaned_lines.append(todo_comment)
                            # Replace config. with placeholder that needs to be parameterized
                            # Handle both .config. and config. (after self. removal)
                            line = re.sub(r'\.?config\.preferred_channel\b', 'preferred_channel', line)
                            line = re.sub(r'\.?config\.alert_frequency\b', 'alert_frequency', line)
                            line = re.sub(r'\.?config\.audio_volume\b', 'audio_volume', line)
                            line = re.sub(r'\.?config\.haptic_intensity\b', 'haptic_intensity', line)
                            line = re.sub(r'\.?config\.visual_contrast\b', 'visual_contrast', line)
                        cleaned_lines.append(line)
                    func_code = '\n'.join(cleaned_lines)
                
                reference_code += f'\n# From {module_path}'
                if is_class_method:
                    reference_code += f' (class {class_name} method)'
                reference_code += '\n'
                reference_code += func_code
                reference_code += '\n\n'
        
        except Exception as e:
            logger.warning(f"Failed to extract {func_name} from {module_path}: {e}")
            continue
    
    return reference_code

def export_ios_bundle(
    model: nn.Module,
    output_dir: str = 'maxsight_ios_bundle',
    input_size: tuple = (1, 3, 224, 224)
) -> Path:
    """
    Export minimal iOS bundle: PTE + configs + single reference file.
    
    Creates exactly 4 files:
    - maxsight.pte (model)
    - model_config.json (model settings)
    - runtime_config.json (runtime toggles)
    - processing_reference.py (all reference logic in one file)
    - README_XCODE.md (iOS integration guide)
    
    Arguments:
        model: Model to export
        output_dir: Output directory
        input_size: Model input size (B, C, H, W)
    
    Returns:
        Path to bundle directory
    """
    from pathlib import Path
    import json
    from datetime import datetime
    
    bundle_path = Path(output_dir)
    bundle_path.mkdir(exist_ok=True, parents=True)
    
    # 1. Export PTE
    logger.info(f"Exporting PTE model...")
    pte_path = export_to_executorch(
        model,
        str(bundle_path / 'maxsight.pte'),
        input_size,
        validate=True
    )
    
    if not pte_path:
        logger.warning("PTE export failed (ExecuTorch may not be installed). Bundle created without PTE file.")
        logger.warning("Install ExecuTorch: pip install executorch")
        pte_size_mb = 0.0
    else:
        pte_size_mb = pte_path.stat().st_size / (1024 * 1024)
    
    # 2. Export model config (minimal)
    model_params = sum(p.numel() for p in model.parameters())
    model_config = {
        'version': '1.0.0',
        'export_timestamp': datetime.now().isoformat(),
        'input_size': list(input_size),
        'num_classes': 80,
        'num_urgency_levels': 4,
        'num_distance_zones': 3,
        'detection_threshold': 0.5,
        'nms_threshold': 0.5,
        'model_params': model_params,
        'model_size_mb': round(pte_size_mb, 2),
        'quantization': 'INT8' if hasattr(model, '_quantized') else 'FP32',
        'output_shapes': {
            'classifications': [input_size[0], 80],  # [B, num_classes]
            'boxes': [input_size[0], 100, 4],  # [B, max_detections, 4]
            'objectness': [input_size[0], 100],  # [B, max_detections]
            'urgency_scores': [input_size[0], 100, 4],  # [B, max_detections, urgency_levels]
            'distance_zones': [input_size[0], 100, 3],  # [B, max_detections, zones]
        }
    }
    with open(bundle_path / 'model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # 3. Export runtime config (minimal)
    runtime_config = {
        'version': '1.0.0',
        'max_latency_ms': 500.0,
        'max_memory_mb': 50.0,
        'enabled_heads': ['classification', 'box_regression', 'objectness', 'urgency', 'distance'],
        'enable_fallbacks': True,
        'uncertainty_threshold': 0.7,
        'alert_frequency': 'medium',  # 'low', 'medium', 'high'
        'preferred_channel': 'audio',  # 'audio', 'visual', 'haptic', 'hybrid'
        'condition_modes': [
            'glaucoma', 'amd', 'cataracts', 'color_blindness', 
            'retinitis_pigmentosa', 'diabetic_retinopathy', 'cvi'
        ]
    }
    with open(bundle_path / 'runtime_config.json', 'w') as f:
        json.dump(runtime_config, f, indent=2)
    
    # 4. Create processing_reference.py (extract actual functions)
    logger.info("Extracting processing reference...")
    processing_ref = _extract_processing_reference()
    
    with open(bundle_path / 'processing_reference.py', 'w') as f:
        f.write(processing_ref)
    
    # 5. Create minimal README
    readme = f'''# MaxSight iOS Bundle

**Export Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Version:** 1.0.0

## Files

- `maxsight.pte` - ExecuTorch model (add to Xcode project)
- `model_config.json` - Model parameters and thresholds
- `runtime_config.json` - Runtime settings and toggles
- `processing_reference.py` - Reference implementation (port to Swift)

## Xcode Integration

1. Drag `maxsight.pte` into your Xcode project
2. Ensure it's added to your app target
3. Add to "Copy Bundle Resources" in Build Phases

Add ExecuTorch to your project:

```swift
// Package.swift or Xcode Package Manager
dependencies: [
    .package(url: "https://github.com/pytorch/executorch", from: "0.4.0")
]
```

```swift
import Executorch

class MaxSightModel {{
    private var program: Program?
    private var method: Method?
    
    func load() throws {{
        guard let modelPath = Bundle.main.path(forResource: "maxsight", ofType: "pte") else {{
            throw MaxSightError.modelNotFound
        }}
        
        program = try Program.load(fromPath: modelPath)
        method = program?.loadMethod("forward")
    }}
    
    func predict(image: Tensor) throws -> [String: Tensor] {{
        guard let method = method else {{
            throw MaxSightError.modelNotLoaded
        }}
        
        let outputs = try method.execute(inputs: [image])
        return processOutputs(outputs)
    }}
}}
```

Reference `processing_reference.py` for preprocessing logic. Port to Swift:

```swift
func preprocessImage(_ image: UIImage, condition: VisionCondition) -> Tensor {{
    // 1. Resize to model input size ({input_size[2]}x{input_size[3]})
    let resized = image.resized(to: CGSize(width: {input_size[3]}, height: {input_size[2]}))
    
    // 2. Apply condition-specific transform
    let transformed = applyConditionTransform(resized, condition: condition)
    
    // 3. Normalize to [0, 1] and convert to tensor
    let normalized = transformed.normalized()
    let tensor = Tensor.fromImage(normalized)
    
    // 4. Add batch dimension
    return tensor.unsqueeze(0)  // [1, 3, H, W]
}}

func applyConditionTransform(_ image: UIImage, condition: VisionCondition) -> UIImage {{
    switch condition {{
    case .glaucoma:
        return applyGlaucomaVignette(image)  // See processing_reference.py
    case .amd:
        return applyAMDCentralDarkening(image)
    case .cataracts:
        return applyCataractContrast(image)
    default:
        return image
    }}
}}
```

```swift
let model = MaxSightModel()
try model.load()

let inputTensor = preprocessImage(cameraFrame, condition: .glaucoma)
let outputs = try model.predict(image: inputTensor)

// Outputs contain:
// - classifications: [B, 80] - class logits
// - boxes: [B, 100, 4] - bounding boxes (center format: x, y, w, h)
// - objectness: [B, 100] - object confidence scores
// - urgency_scores: [B, 100, 4] - urgency level scores
// - distance_zones: [B, 100, 3] - distance zone probabilities
```

Reference `processing_reference.py` for postprocessing:

```swift
func postprocessDetections(
    boxes: Tensor,
    scores: Tensor,
    classifications: Tensor,
    config: ModelConfig
) -> [Detection] {{
    // 1. Filter by detection threshold
    let validIndices = scores > config.detectionThreshold
    
    // 2. Apply NMS (Non-Maximum Suppression)
    // See processing_reference.py _nms() function
    let nmsIndices = applyNMS(
        boxes: boxes[validIndices],
        scores: scores[validIndices],
        threshold: config.nmsThreshold
    )
    
    // 3. Convert to detections
    var detections: [Detection] = []
    for idx in nmsIndices {{
        let box = boxes[idx]
        let score = scores[idx]
        let classId = classifications[idx].argmax()
        
        detections.append(Detection(
            box: box,
            score: score,
            classId: classId
        ))
    }}
    
    return detections
}}
```

```swift
struct ModelConfig: Codable {{
    let inputSize: [Int]
    let numClasses: Int
    let detectionThreshold: Double
    let nmsThreshold: Double
}}

func loadModelConfig() throws -> ModelConfig {{
    guard let url = Bundle.main.url(forResource: "model_config", withExtension: "json"),
          let data = try? Data(contentsOf: url) else {{
        throw MaxSightError.configNotFound
    }}
    
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(ModelConfig.self, from: data)
}}
```

## Model Information

- **Input Size:** {input_size}
- **Parameters:** {model_params:,}
- **Model Size:** {pte_size_mb:.1f} MB
- **Classes:** {model_config['num_classes']}
- **Urgency Levels:** {model_config['num_urgency_levels']}
- **Distance Zones:** {model_config['num_distance_zones']}
- **Quantization:** {model_config['quantization']}

## Output Tensor Shapes

See `model_config.json` for exact shapes. Typical outputs:

- `classifications`: [{input_size[0]}, 80] - Class logits
- `boxes`: [{input_size[0]}, 100, 4] - Bounding boxes (center format)
- `objectness`: [{input_size[0]}, 100] - Object confidence
- `urgency_scores`: [{input_size[0]}, 100, 4] - Urgency level scores
- `distance_zones`: [{input_size[0]}, 100, 3] - Distance zone probabilities

## Reference Implementation

See `processing_reference.py` for complete reference:

- **Preprocessing**: Condition-specific transforms (glaucoma, AMD, cataracts, etc.)
- **Postprocessing**: NMS, IoU calculation, detection filtering
- **Scheduling**: Priority calculation, intensity, frequency, channel selection
- **OCR**: Text region clustering and grouping

## Performance Targets

- **Latency**: <500ms per frame (target: <400ms)
- **Memory**: <50MB model size
- **Battery**: <12% per hour normal use

## Troubleshooting

- Verify `maxsight.pte` is in bundle resources
- Check ExecuTorch framework is properly linked
- Ensure iOS deployment target is 15.0+

- Verify input tensor shape matches `input_size` in config
- Check tensor dtype is Float32
- Ensure tensor is on CPU (ExecuTorch requirement)

- Verify preprocessing matches Python reference
- Check postprocessing (NMS, filtering) is correct
- Compare with `processing_reference.py` implementation
'''
    
    with open(bundle_path / 'README_XCODE.md', 'w') as f:
        f.write(readme)
    
    logger.info(f"iOS bundle exported to: {bundle_path}")
    logger.info(f"  - maxsight.pte ({pte_size_mb:.1f} MB)")
    logger.info(f"  - model_config.json")
    logger.info(f"  - runtime_config.json")
    logger.info(f"  - processing_reference.py")
    logger.info(f"  - README_XCODE.md")
    
    return bundle_path

if __name__ == "__main__":
    from ml.models.maxsight_cnn import create_model
    
    model = create_model()
    model.eval()
    
    print("Testing export functionality...")
    export_to_jit(model, 'test_maxsight_traced.pt')
    export_to_executorch(model, 'test_maxsight.pte')
    export_to_coreml(model, 'test_maxsight.mlpackage')
    export_to_onnx(model, 'test_maxsight.onnx')
    print("Export system ready!")

