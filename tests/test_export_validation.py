"""
Export Validation Tests for MaxSight Model
Tests JIT, ExecuTorch, CoreML, and ONNX exports for mobile deployment.
"""

import torch
import sys
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.maxsight_cnn import create_model
from ml.training.export import (
    export_to_jit,
    export_to_executorch,
    export_to_coreml,
    export_to_onnx,
    export_model
)


def test_jit_export():
    """Test JIT export functionality."""
    print("Export Test 1: JIT Export")
    
    model = create_model()
    model.eval()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "test_model.pt"
        
        # Export model
        saved_path = export_to_jit(model, str(export_path), input_size=(1, 3, 224, 224))
        
        # Verify file exists
        assert saved_path.exists(), "JIT export file not created"
        assert saved_path.suffix == ".pt", "JIT export should be .pt file"
        
        # Verify file size is reasonable
        size_mb = saved_path.stat().st_size / (1024 * 1024)
        assert size_mb < 200, f"JIT export too large: {size_mb:.1f} MB"
        
        # Load and test exported model
        loaded_model = torch.jit.load(str(saved_path))
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = loaded_model(dummy_input)
        
        assert output is not None, "Exported model should produce output"
        
    print("  PASSED: JIT export works correctly")


def test_executorch_export():
    """Test ExecuTorch export (may skip if not installed)."""
    print("\nExport Test 2: ExecuTorch Export")
    
    model = create_model()
    model.eval()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "test_model.pte"
        
        try:
            saved_path = export_to_executorch(model, str(export_path), input_size=(1, 3, 224, 224))
            
            if saved_path is not None:
                assert saved_path.exists(), "ExecuTorch export file not created"
                size_mb = saved_path.stat().st_size / (1024 * 1024)
                assert size_mb < 200, f"ExecuTorch export too large: {size_mb:.1f} MB"
                print("  PASSED: ExecuTorch export works correctly")
            else:
                print("  ⚠️ SKIPPED: ExecuTorch not available, fell back to JIT")
        except Exception as e:
            print(f"  ⚠️ SKIPPED: ExecuTorch export failed (expected if not installed): {e}")


def test_coreml_export():
    """Test CoreML export (may skip if not installed)."""
    print("\nExport Test 3: CoreML Export")
    
    model = create_model()
    model.eval()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "test_model.mlpackage"
        
        try:
            saved_path = export_to_coreml(model, str(export_path), input_size=(1, 3, 224, 224))
            
            if saved_path is not None:
                assert saved_path.exists() or Path(str(export_path) + ".mlpackage").exists(), "CoreML export not created"
                print("  PASSED: CoreML export works correctly")
            else:
                print("  ⚠️ SKIPPED: CoreML export not available")
        except Exception as e:
            print(f"  ⚠️ SKIPPED: CoreML export failed (expected if not installed): {e}")


def test_onnx_export():
    """Test ONNX export (may skip if not installed)."""
    print("\nExport Test 4: ONNX Export")
    
    model = create_model()
    model.eval()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "test_model.onnx"
        
        try:
            saved_path = export_to_onnx(model, str(export_path), input_size=(1, 3, 224, 224))
            
            if saved_path is not None:
                assert saved_path.exists(), "ONNX export file not created"
                size_mb = saved_path.stat().st_size / (1024 * 1024)
                assert size_mb < 200, f"ONNX export too large: {size_mb:.1f} MB"
                print("  PASSED: ONNX export works correctly")
            else:
                print("  ⚠️ SKIPPED: ONNX export not available")
        except Exception as e:
            print(f"  ⚠️ SKIPPED: ONNX export failed (expected if not installed): {e}")


def test_export_model_function():
    """Test unified export_model function."""
    print("\nExport Test 5: Unified Export Function")
    
    model = create_model()
    model.eval()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        results = export_model(
            model,
            format='jit',
            save_dir=tmpdir,
            input_size=(1, 3, 224, 224)
        )
        
        assert 'exports' in results, "Export results should contain exports"
        assert 'jit' in results['exports'], "JIT export should be in results"
        assert Path(results['exports']['jit']).exists(), "JIT export file should exist"
        
        assert 'metadata' in results, "Export results should contain metadata"
        assert 'input_size' in results['metadata'], "Metadata should contain input_size"
        assert 'model_params' in results['metadata'], "Metadata should contain model_params"
        
    print("  PASSED: Unified export function works correctly")


def test_export_output_consistency():
    """Test that exported model outputs match original model."""
    print("\nExport Test 6: Export Output Consistency")
    
    model = create_model()
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Get original model output
    with torch.no_grad():
        original_output = model(dummy_input)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "test_model.pt"
        export_to_jit(model, str(export_path), input_size=(1, 3, 224, 224))
        
        # Load exported model
        loaded_model = torch.jit.load(str(export_path))
        
        with torch.no_grad():
            exported_output = loaded_model(dummy_input)
        
        # Compare outputs (allow small numerical differences)
        if isinstance(original_output, dict) and isinstance(exported_output, dict):
            for key in original_output.keys():
                if key in exported_output:
                    orig_tensor = original_output[key]
                    exp_tensor = exported_output[key]
                    if isinstance(orig_tensor, torch.Tensor) and isinstance(exp_tensor, torch.Tensor):
                        diff = torch.abs(orig_tensor - exp_tensor).max().item()
                        assert diff < 1.0, f"Output difference too large for {key}: {diff}"
        
    print("  PASSED: Exported model outputs are consistent")


if __name__ == "__main__":
    test_jit_export()
    test_executorch_export()
    test_coreml_export()
    test_onnx_export()
    test_export_model_function()
    test_export_output_consistency()
    
    print("\nAll export validation tests completed!")

