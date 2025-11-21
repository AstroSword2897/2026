# MaxSight Model Export for iOS Deployment
# This module provides functions to export trained MaxSight CNN models to various formats suitable for iOS deployment
# Each export format has different advantages:
# - PyTorch JIT: Universal format, always available, good for testing
# - ExecuTorch: Facebook's mobile-optimized format, best performance on iOS
# - CoreML: Apple's native format, seamless iOS integration, best for production
# - ONNX: Industry standard, good for cross-platform compatibility
# All export functions handle the model's dict-based output format and provide graceful fallbacks when optional dependencies are missing
# Supports: PyTorch JIT tracing, ExecuTorch export (when available), CoreML export (for iOS), ONNX export (alternative)

# Standard library imports
import json  # For saving export metadata as JSON
from pathlib import Path  # For cross-platform path handling
from typing import Optional  # For optional return types

# PyTorch imports
import torch  # Core PyTorch library
import torch.nn as nn  # Neural network module base class


def export_to_jit(
    model: nn.Module,
    save_path: str = 'maxsight_traced.pt',
    input_size: tuple = (1, 3, 224, 224)
) -> Path:
    # Export model to PyTorch JIT traced format - converts dynamic model to static computation graph
    # Most reliable export method, always available (core PyTorch) - traced model loads without original code
    # strict=False allows dict outputs (MaxSight returns dict) - safe since output structure is constant
    # Complexity: O(1) export time - just traces forward pass once
    # Relationship: Base export format for iOS deployment - other formats may fall back to this
    # Returns: Path to saved .pt file, raises Exception if tracing fails
    print("Exporting MaxSight to PyTorch JIT Format")
    
    model.eval()  # Eval mode for consistent tracing (dropout/batch norm deterministic)
    model.cpu()  # Move to CPU - tracing works best on CPU, iOS will use Neural Engine anyway
    dummy_input = torch.randn(*input_size)  # Random dummy input - values don't matter, just capturing graph structure
    
    try:
        traced_model = torch.jit.trace(model, dummy_input, strict=False)  # Trace forward pass to static graph, strict=False allows dict outputs
        test_output = traced_model(dummy_input)  # type: ignore  # Verify trace works before saving
        
        print(f" Model traced successfully")
        print(f" Input shape: {dummy_input.shape}")
        if isinstance(test_output, dict):
            print(f"  Output keys: {list(test_output.keys())}")  # MaxSight returns dict with multiple outputs
        else:
            print(f"  Output type: {type(test_output)}")
        
        save_path_obj = Path(save_path)
        traced_model.save(str(save_path_obj))  # Save standalone model file (loadable without original code)
        
        size_mb = save_path_obj.stat().st_size / (1024 * 1024)  # Calculate file size for mobile deployment check
        print(f"\n Export complete!")
        print(f"  Saved to: {save_path}")
        print(f"  Model size: {size_mb:.1f} MB")
        print(f"  Target <50MB: {'✓' if size_mb < 50 else '✗'}")  # Mobile target: FP32 usually >50MB, INT8 should be <50MB
        
        return save_path_obj
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        raise  # Re-raise for calling code to handle


def export_to_executorch(
    model: nn.Module,
    save_path: str = 'maxsight.pte',
    input_size: tuple = (1, 3, 224, 224)
) -> Optional[Path]:
    # Export model to ExecuTorch format (.pte) - Facebook's mobile-optimized runtime for better performance/lower memory on iOS
    # Converts PyTorch model to Edge IR then compiles to .pte format for edge deployment
    # Complexity: O(1) export time, but requires executorch package (optional dependency)
    # Relationship: Part of iOS deployment pipeline - provides mobile-optimized format alternative to CoreML
    # Falls back to JIT if executorch not installed - ensures export pipeline always works
    # Returns: Path to .pte file, or JIT traced model path if fallback used
    print("Exporting MaxSight to ExecuTorch Format")
    
    try:
        import executorch.exir as exir  # type: ignore  # ExecuTorch IR module (optional dependency)
        from executorch.extension.pybind11.portable import to_edge  # type: ignore  # Edge conversion function
        
        model.eval()  # Eval mode for consistent export
        model.cpu()  # ExecuTorch export works on CPU
        dummy_input = torch.randn(*input_size)  # Dummy input for model structure tracing
        
        edge_program = to_edge(model, (dummy_input,))  # Convert PyTorch to Edge IR (ExecuTorch's internal format) - tuple required
        executorch_program = edge_program.to_executorch()  # Compile Edge IR to .pte format for mobile devices
        
        save_path_obj = Path(save_path)
        with open(save_path_obj, 'wb') as f:  # Binary write mode for .pte file
            f.write(executorch_program.buffer)  # Write serialized model data to disk
        
        size_mb = save_path_obj.stat().st_size / (1024 * 1024)
        print(f" ExecuTorch export complete!")
        print(f"  Saved to: {save_path}")
        print(f"  Model size: {size_mb:.1f} MB")
        
        return save_path_obj
        
    except ImportError:
        print("ExecuTorch not installed")
        print("Install: pip install executorch")
        print("Falling back to JIT trace...")
        return export_to_jit(model, save_path.replace('.pte', '_traced.pt'), input_size)  # Graceful fallback ensures export always works
        
    except Exception as e:
        print(f"✗ ExecuTorch export failed: {e}")
        print("Falling back to JIT trace...")
        return export_to_jit(model, save_path.replace('.pte', '_traced.pt'), input_size)  # Fallback for unsupported operations


def export_to_coreml(
    model: nn.Module,
    save_path: str = 'maxsight.mlpackage',
    input_size: tuple = (1, 3, 224, 224)
) -> Optional[Path]:
    # Export model to CoreML format (native iOS) - Apple's ML framework, recommended for production iOS deployment
    # CoreML integrates with iOS apps and leverages Neural Engine for optimal performance
    # .mlpackage is directory-based format with metadata/weights - CoreML auto-optimizes for device (CPU/GPU/Neural Engine)
    # Complexity: O(1) export time, but conversion may take time depending on model size
    # Relationship: Primary iOS deployment format - enables seamless integration with MaxSight iOS app
    # Returns: Path to .mlpackage directory, or None if export failed (requires coremltools package)
    print("Exporting MaxSight to CoreML Format")
    
    try:
        import coremltools as ct  # CoreML tools (optional dependency)
        
        model.eval()  # Eval mode for consistent tracing
        model.cpu()  # CoreML conversion works on CPU
        dummy_input = torch.randn(*input_size)  # Dummy input for tracing
        
        traced_model = torch.jit.trace(model, dummy_input, strict=False)  # Trace to JIT first (CoreML requires traced model, not original)
        test_output = traced_model(dummy_input)  # type: ignore  # Test output structure to configure CoreML properly
        
        if isinstance(test_output, dict):
            output_types = [ct.TensorType(name=key) for key in test_output.keys()]  # Create TensorType for each dict key (MaxSight has multiple outputs)
        else:
            output_types = [ct.TensorType(name="output")]  # Fallback for single tensor (shouldn't happen)
        
        coreml_model = ct.convert(
            traced_model,  # Traced PyTorch model
            inputs=[ct.TensorType(name="image", shape=input_size)],  # Input spec
            outputs=output_types,  # Output specs (one per dict key)
            minimum_deployment_target=ct.target.iOS15  # Minimum iOS version
        )
        
        save_path_obj = Path(save_path)
        if coreml_model is not None:
            coreml_model.save(str(save_path_obj))  # Save .mlpackage directory (CoreML handles directory creation)
        else:
            raise ValueError("CoreML model conversion failed")
        
        size_mb = save_path_obj.stat().st_size / (1024 * 1024)  # Calculate .mlpackage directory size
        print(f" CoreML export complete!")
        print(f" Saved to: {save_path}")
        print(f"  Model size: {size_mb:.1f} MB")
        print(f"  Deployment target: iOS 15+")
        
        return save_path_obj
        
    except ImportError:
        print("CoreML tools not installed")
        print("Install: pip install coremltools")
        return None  # Graceful failure - calling code can try other formats
        
    except Exception as e:
        print(f"✗ CoreML export failed: {e}")
        return None  # Return None for unsupported operations or conversion issues


def export_to_onnx(
    model: nn.Module,
    save_path: str = 'maxsight.onnx',
    input_size: tuple = (1, 3, 224, 224)
) -> Optional[Path]:
    # Export model to ONNX format - industry-standard format for cross-platform deployment
    # ONNX models can be converted to other formats or run with ONNX Runtime
    # Complexity: O(1) export time, but may fail with unsupported PyTorch operations
    # Relationship: Alternative export format for cross-platform compatibility - not primary iOS format
    # Note: May have limitations with some PyTorch operations - try JIT or CoreML if export fails
    # Returns: Path to .onnx file, or None if export failed (requires onnx package)
    print("Exporting MaxSight to ONNX Format")
    
    try:
        import onnx  # type: ignore  # ONNX library (optional dependency)
        
        model.eval()  # Eval mode for consistent export
        model.cpu()  # ONNX export works on CPU
        dummy_input = torch.randn(*input_size)  # Dummy input for export
        
        torch.onnx.export(
            model,  # PyTorch model to export
            (dummy_input,),  # Inputs as tuple (ONNX requires tuple, not single tensor - different from JIT)
            save_path,  # Output file path
            input_names=['image'],  # Input tensor name (useful for inference)
            output_names=['output'],  # Output name (may not work with dict outputs)
            dynamic_axes={
                'image': {0: 'batch_size'},  # Dynamic batch dimension enables variable batch sizes
                'output': {0: 'batch_size'}  # Output batch matches input
            },
            opset_version=11  # ONNX opset version 11 (widely supported)
        )
        
        onnx_model = onnx.load(save_path)  # Load exported model
        onnx.checker.check_model(onnx_model)  # Validate model structure and types
        
        save_path_obj = Path(save_path)
        size_mb = save_path_obj.stat().st_size / (1024 * 1024)
        print(f"✓ ONNX export complete!")
        print(f"  Saved to: {save_path}")
        print(f"  Model size: {size_mb:.1f} MB")
        print(f"  Opset version: 11")
        
        return save_path_obj
        
    except ImportError:
        print("ONNX not installed")
        print("Install: pip install onnx")
        return None  # Graceful failure
        
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        return None  # May fail with unsupported PyTorch operations


def export_model(
    model: nn.Module,
    format: str = 'jit',
    save_dir: str = 'exports',
    input_size: tuple = (1, 3, 224, 224)
) -> dict:
    # Main entry point for model export - handles multiple formats, can export to all formats at once
    # Formats: 'jit' (always available), 'executorch' (mobile-optimized), 'coreml' (iOS native, recommended), 'onnx' (cross-platform), 'all' (all formats)
    # Complexity: O(N) where N is number of formats - each export is independent
    # Relationship: Central export function used by training pipeline to prepare models for iOS deployment
    # Saves all exports to common directory with metadata JSON file for tracking
    # Returns: Dict with 'format', 'exports' (successful export paths), 'metadata' (input_size, param count)
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
        path = export_to_jit(model, str(save_dir_path / 'maxsight_traced.pt'), input_size)  # JIT always available (core PyTorch)
        results['exports']['jit'] = str(path)  # JIT always succeeds or raises
    
    if format == 'executorch' or format == 'all':
        path = export_to_executorch(model, str(save_dir_path / 'maxsight.pte'), input_size)  # May fail if executorch not installed
        if path:
            results['exports']['executorch'] = str(path)  # Only add if succeeded
    
    if format == 'coreml' or format == 'all':
        path = export_to_coreml(model, str(save_dir_path / 'maxsight.mlpackage'), input_size)  # May fail if coremltools not installed
        if path:
            results['exports']['coreml'] = str(path)  # Only add if succeeded
    
    if format == 'onnx' or format == 'all':
        path = export_to_onnx(model, str(save_dir_path / 'maxsight.onnx'), input_size)  # May fail with unsupported operations
        if path:
            results['exports']['onnx'] = str(path)  # Only add if succeeded
    
    metadata_path = save_dir_path / 'export_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(results, f, indent=2)  # Save metadata JSON for tracking exports
    
    print(f"\n✓ Export metadata saved to: {metadata_path}")
    
    return results


# MAIN ENTRY POINT - For testing the export system

if __name__ == "__main__":
    print("MaxSight Model Export System")
    
    from ml.models.maxsight_cnn import create_model  # Import model factory
    
    model = create_model()  # Create dummy model for testing (untrained, but export doesn't need trained weights)
    model.eval()  # Eval mode required for consistent export
    
    print("\nTesting export functionality...")
    
    export_to_jit(model, 'test_maxsight_traced.pt')  # Test JIT (always works - core PyTorch)
    export_to_executorch(model, 'test_maxsight.pte')  # Test ExecuTorch (may fail if not installed)
    export_to_coreml(model, 'test_maxsight.mlpackage')  # Test CoreML (may fail if not installed)
    export_to_onnx(model, 'test_maxsight.onnx')  # Test ONNX (may fail if not installed)
    
    print("\n Export system ready!")

