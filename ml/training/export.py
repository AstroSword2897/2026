"""Model export for iOS: JIT, ExecuTorch, CoreML, ONNX. Handles dict outputs gracefully."""

import json
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn


def export_to_jit(model: nn.Module, save_path: str = 'maxsight_traced.pt', input_size: tuple = (1, 3, 224, 224)) -> Path:
    print(f"Exporting to JIT format: {save_path}")
    
    model.eval()
    model.cpu()
    dummy_input = torch.randn(*input_size)
    
    try:
        traced_model = torch.jit.trace(model, dummy_input, strict=False)
        test_output = traced_model(dummy_input)  # type: ignore
        
        save_path_obj = Path(save_path)
        traced_model.save(str(save_path_obj))
        
        size_mb = save_path_obj.stat().st_size / (1024 * 1024)
        print(f"  Saved: {save_path}, Size: {size_mb:.1f} MB")
        
        return save_path_obj
    except Exception as e:
        print(f"Export failed: {e}")
        raise


def export_to_executorch(model: nn.Module, save_path: str = 'maxsight.pte', input_size: tuple = (1, 3, 224, 224)) -> Optional[Path]:
    print(f"Exporting to ExecuTorch format: {save_path}")
    
    try:
        from executorch.extension.pybind11.portable import to_edge  # type: ignore
        
        model.eval()
        model.cpu()
        dummy_input = torch.randn(*input_size)
        
        edge_program = to_edge(model, (dummy_input,))
        executorch_program = edge_program.to_executorch()
        
        save_path_obj = Path(save_path)
        with open(save_path_obj, 'wb') as f:
            f.write(executorch_program.buffer)
        
        size_mb = save_path_obj.stat().st_size / (1024 * 1024)
        print(f"  Saved: {save_path}, Size: {size_mb:.1f} MB")
        return save_path_obj
        
    except ImportError:
        print("ExecuTorch not installed, falling back to JIT...")
        return export_to_jit(model, save_path.replace('.pte', '_traced.pt'), input_size)
    except Exception as e:
        print(f"ExecuTorch export failed: {e}, falling back to JIT...")
        return export_to_jit(model, save_path.replace('.pte', '_traced.pt'), input_size)


def export_to_coreml(model: nn.Module, save_path: str = 'maxsight.mlpackage', input_size: tuple = (1, 3, 224, 224)) -> Optional[Path]:
    """Export to CoreML format (iOS native). Requires coremltools."""
    print(f"Exporting to CoreML format: {save_path}")
    
    try:
        import coremltools as ct
        
        model.eval()
        model.cpu()
        dummy_input = torch.randn(*input_size)
        
        traced_model = torch.jit.trace(model, dummy_input, strict=False)
        test_output = traced_model(dummy_input)  # type: ignore
        
        if isinstance(test_output, dict):
            output_types = [ct.TensorType(name=key) for key in test_output.keys()]
        else:
            output_types = [ct.TensorType(name="output")]
        
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="image", shape=input_size)],
            outputs=output_types,
            minimum_deployment_target=ct.target.iOS15
        )
        
        save_path_obj = Path(save_path)
        if coreml_model is not None:
            coreml_model.save(str(save_path_obj))
        else:
            raise ValueError("CoreML conversion failed")
        
        size_mb = save_path_obj.stat().st_size / (1024 * 1024)
        print(f"  Saved: {save_path}, Size: {size_mb:.1f} MB, iOS 15+")
        return save_path_obj
        
    except ImportError:
        print("CoreML tools not installed (pip install coremltools)")
        return None
    except Exception as e:
        print(f"CoreML export failed: {e}")
        return None


def export_to_onnx(model: nn.Module, save_path: str = 'maxsight.onnx', input_size: tuple = (1, 3, 224, 224)) -> Optional[Path]:
    """Export to ONNX format. May fail with dict outputs - use JIT/CoreML for iOS."""
    print(f"Exporting to ONNX format: {save_path}")
    
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
        print(f"  Saved: {save_path}, Size: {size_mb:.1f} MB")
        return save_path_obj
        
    except ImportError:
        print("ONNX not installed (pip install onnx)")
        return None
    except Exception as e:
        print(f"ONNX export failed: {e}")
        return None


def export_model(model: nn.Module, format: str = 'jit', save_dir: str = 'exports', input_size: tuple = (1, 3, 224, 224)) -> dict:
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
        path = export_to_jit(model, str(save_dir_path / 'maxsight_traced.pt'), input_size)
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
    
    print(f"Export metadata saved to: {metadata_path}\n")
    
    return results


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

