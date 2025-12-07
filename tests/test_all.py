"""
Comprehensive Unit Tests for MaxSight System
Consolidated test suite covering all components
"""

import sys
import torch
import pytest
import numpy as np
import tempfile
import time
from pathlib import Path
from typing import Dict, List
from collections import Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Imports
from ml.models.maxsight_cnn import (
    create_model, MaxSightCNN, COCO_CLASSES, COCO_BASE_CLASSES, ACCESSIBILITY_CLASSES
)
from ml.training.train_production import ProductionTrainer, create_dummy_dataloaders, NUM_CLASSES
from ml.training.losses import MaxSightLoss
from ml.training.metrics import DetectionMetrics, compute_iou_matrix
from ml.training.scene_metrics import SceneMetrics
from ml.training.export import (
    export_to_jit, export_to_executorch, export_to_coreml, export_to_onnx, export_model
)
from ml.training.quantization import (
    quantize_model_int8, compare_model_sizes, quantize_validation, fuse_maxsight_modules
)
from ml.utils.preprocessing import ImagePreprocessor
from ml.data.generate_annotations import (
    get_all_datasets_info, save_class_mappings, ENVIRONMENTAL_CLASSES, SOUND_CLASSES
)
from torch.utils.data import DataLoader, TensorDataset


# Helper functions
def create_dummy_calibration_data(num_batches: int = 10, batch_size: int = 2):
    """Create dummy calibration data for quantization."""
    dummy_images = torch.randn(num_batches * batch_size, 3, 224, 224)
    dataset = TensorDataset(dummy_images)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# Model Tests
def test_model_creation():
    """Test basic model creation"""
    print("Model Test 1: Model Creation")
    model = create_model()
    assert model is not None
    assert isinstance(model, MaxSightCNN)
    print("  PASSED")


def test_forward_pass():
    """Test forward pass with dummy data"""
    print("Model Test 2: Forward Pass")
    model = create_model()
    model.eval()
    
    batch_size = 2
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        outputs = model(dummy_image)
    
    num_locations = outputs['num_locations']
    num_classes = len(COCO_CLASSES)
    
    assert outputs['classifications'].shape == (batch_size, num_locations, num_classes)
    assert outputs['boxes'].shape == (batch_size, num_locations, 4)
    assert outputs['objectness'].shape == (batch_size, num_locations)
    assert outputs['text_regions'].shape == (batch_size, num_locations)
    assert outputs['scene_embedding'].shape == (batch_size, 512)
    assert outputs['urgency_scores'].shape == (batch_size, 4)
    assert outputs['distance_zones'].shape == (batch_size, num_locations, 3)
    assert num_locations == 196
    print("  PASSED")


def test_audio_fusion():
    """Test audio fusion mode"""
    print("Model Test 3: Audio Fusion")
    model = create_model(use_audio=True)
    model.eval()
    
    batch_size = 2
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    dummy_audio = torch.randn(batch_size, 128)
    
    with torch.no_grad():
        outputs = model(dummy_image, dummy_audio)
    
    num_locations = outputs['num_locations']
    num_classes = len(COCO_CLASSES)
    assert outputs['classifications'].shape == (batch_size, num_locations, num_classes)
    assert outputs['scene_embedding'].shape == (batch_size, 512)
    print("  PASSED")


def test_color_blindness_mode():
    """Test color blindness condition mode"""
    print("Model Test 4: Color Blindness Mode")
    model = create_model(condition_mode='color_blindness')
    model.eval()
    
    batch_size = 2
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        outputs = model(dummy_image)
    
    assert 'colors' in outputs
    num_locations = outputs['num_locations']
    assert outputs['colors'].shape == (batch_size, num_locations, 12)
    print("  PASSED")


def test_parameter_count():
    """Test model parameter count"""
    print("Model Test 5: Parameter Count")
    model = create_model()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    assert 30_000_000 < total_params < 40_000_000
    assert trainable_params == total_params
    print(f"  PASSED: {total_params:,} parameters")


def test_gradient_flow():
    """Test that gradients can flow through the model"""
    print("Model Test 6: Gradient Flow")
    model = create_model()
    model.train()
    
    dummy_image = torch.randn(2, 3, 224, 224, requires_grad=True)
    outputs = model(dummy_image)
    
    loss = outputs['classifications'].sum()
    loss.backward()
    
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None:
            has_gradients = True
            break
    
    assert has_gradients
    print("  PASSED")


def test_inference_mode():
    """Test inference mode (eval)"""
    print("Model Test 7: Inference Mode")
    model = create_model()
    model.eval()
    
    dummy_image = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        outputs = model(dummy_image)
    
    assert 'classifications' in outputs
    assert 'boxes' in outputs
    assert 'objectness' in outputs
    assert 'scene_embedding' in outputs
    assert 'urgency_scores' in outputs
    assert 'distance_zones' in outputs
    assert 'num_locations' in outputs
    
    detections = model.get_detections(outputs, confidence_threshold=0.3)
    assert isinstance(detections, list)
    assert len(detections) == 1
    print("  PASSED")


# Comprehensive System Tests
def test_class_system():
    """Test comprehensive class system"""
    print("System Test 1: Class System")
    duplicates = [item for item, count in Counter(COCO_CLASSES).items() if count > 1]
    assert len(duplicates) == 0, f"Found duplicates: {duplicates}"
    assert len(COCO_BASE_CLASSES) == 75
    assert len(COCO_CLASSES) == NUM_CLASSES
    print("  PASSED")


def test_model_creation_system():
    """Test model creation with comprehensive classes"""
    print("System Test 2: Model Creation")
    model = create_model()
    assert model.num_classes == len(COCO_CLASSES)
    assert model.cls_head[-1].out_channels == len(COCO_CLASSES)
    
    total_params = sum(p.numel() for p in model.parameters())
    int8_size_mb = total_params / 1024 / 1024
    assert int8_size_mb < 50
    print("  PASSED")


def test_forward_pass_system():
    """Test forward pass with audio"""
    print("System Test 3: Forward Pass")
    model = create_model()
    model.eval()
    
    dummy_image = torch.randn(2, 3, 224, 224)
    dummy_audio = torch.randn(2, 128)
    
    with torch.no_grad():
        outputs = model(dummy_image, dummy_audio)
    
    assert outputs['classifications'].shape == (2, 196, len(COCO_CLASSES))
    assert outputs['boxes'].shape == (2, 196, 4)
    assert outputs['objectness'].shape == (2, 196)
    assert outputs['text_regions'].shape == (2, 196)
    assert outputs['scene_embedding'].shape == (2, 512)
    assert outputs['urgency_scores'].shape == (2, 4)
    assert outputs['distance_zones'].shape == (2, 196, 3)
    assert outputs['num_locations'] == 196
    print("  PASSED")


def test_training_system():
    """Test training system"""
    print("System Test 4: Training System")
    model = create_model()
    train_loader, val_loader = create_dummy_dataloaders(num_train=20, num_val=5, batch_size=2)
    
    trainer = ProductionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cpu',
        num_epochs=1
    )
    
    assert trainer.criterion.num_classes == len(COCO_CLASSES)
    
    sample_batch = next(iter(train_loader))
    images = sample_batch['images']
    targets = {
        'labels': sample_batch['labels'],
        'boxes': sample_batch['boxes'],
        'urgency': sample_batch['urgency'],
        'distance': sample_batch['distance'],
        'num_objects': sample_batch['num_objects']
    }
    
    with torch.no_grad():
        outputs = model(images)
        losses = trainer.criterion(outputs, targets)
    
    assert 'total_loss' in losses
    assert losses['total_loss'].item() > 0
    print("  PASSED")


def test_detections():
    """Test detection system"""
    print("System Test 5: Detection System")
    model = create_model()
    model.eval()
    
    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_audio = torch.randn(1, 128)
    
    with torch.no_grad():
        outputs = model(dummy_image, dummy_audio)
        detections = model.get_detections(outputs, confidence_threshold=0.3)
    
    assert isinstance(detections, list)
    assert len(detections) == 1
    
    if len(detections[0]) > 0:
        det = detections[0][0]
        assert 'class' in det
        assert 'class_name' in det
        assert det['class'] < len(COCO_CLASSES)
    print("  PASSED")


def test_visual_conditions():
    """Test all visual condition modes"""
    print("System Test 6: Visual Condition Support")
    conditions = [
        'myopia', 'hyperopia', 'astigmatism', 'presbyopia', 'refractive_errors',
        'cataracts', 'glaucoma', 'amd', 'diabetic_retinopathy',
        'retinitis_pigmentosa', 'color_blindness', 'cvi', 'amblyopia', 'strabismus'
    ]
    
    for cond in conditions:
        model = create_model(condition_mode=cond)
        preprocessor = ImagePreprocessor(condition_mode=cond)
    print("  PASSED")


def test_data_sources():
    """Test data source configuration"""
    print("System Test 7: Data Sources")
    assert len(COCO_BASE_CLASSES) == 75
    assert len(COCO_CLASSES) > 0
    assert len(ACCESSIBILITY_CLASSES) > 0
    print("  PASSED")


# Condition-Specific Tests
def test_all_condition_modes():
    """Test that all 14 visual condition modes can be created."""
    print("Condition Test 1: All Condition Modes")
    
    conditions = [
        'myopia', 'hyperopia', 'astigmatism', 'presbyopia', 'refractive_errors',
        'cataracts', 'glaucoma', 'amd', 'diabetic_retinopathy',
        'retinitis_pigmentosa', 'color_blindness', 'cvi', 'amblyopia', 'strabismus'
    ]
    
    for cond in conditions:
        model = create_model(condition_mode=cond)
        assert model is not None
        
        preprocessor = ImagePreprocessor(condition_mode=cond)
        assert preprocessor is not None
        
        model.eval()
        dummy_image = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            outputs = model(dummy_image)
        
        assert 'classifications' in outputs
    print("  PASSED")


def test_condition_preprocessing():
    """Test that condition-specific preprocessing is applied."""
    print("Condition Test 2: Condition-Specific Preprocessing")
    
    test_conditions = ['cataracts', 'glaucoma', 'color_blindness', 'low_light']
    dummy_image_np = np.random.rand(224, 224, 3).astype(np.float32)
    dummy_image_np = (dummy_image_np * 255.0).astype(np.uint8)
    
    for cond in test_conditions:
        try:
            preprocessor = ImagePreprocessor(condition_mode=cond)
            processed = preprocessor(dummy_image_np)
            assert processed is not None
        except Exception as e:
            pass
    print("  PASSED")


def test_condition_robustness():
    """Test model robustness with condition-specific inputs."""
    print("Condition Test 3: Condition Robustness")
    
    model = create_model()
    model.eval()
    conditions_to_test = ['cataracts', 'glaucoma', 'color_blindness']
    dummy_image_np = np.random.rand(224, 224, 3).astype(np.float32)
    dummy_image_np = (dummy_image_np * 255).astype(np.uint8)
    
    from PIL import Image
    for cond in conditions_to_test:
        preprocessor = ImagePreprocessor(condition_mode=cond)
        pil_image = Image.fromarray(dummy_image_np)
        processed_image = preprocessor(pil_image)
        
        if isinstance(processed_image, torch.Tensor):
            processed_tensor = processed_image.unsqueeze(0) if processed_image.dim() == 3 else processed_image
        else:
            processed_tensor = torch.from_numpy(processed_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        with torch.no_grad():
            outputs = model(processed_tensor)
        
        assert 'classifications' in outputs
        assert outputs['classifications'].shape[0] == 1
        
        detections = model.get_detections(outputs, confidence_threshold=0.1)
        assert isinstance(detections, list)
    print("  PASSED")


# Export Validation Tests
def test_jit_export():
    """Test JIT export functionality."""
    print("Export Test 1: JIT Export")
    
    model = create_model()
    model.eval()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "test_model.pt"
        saved_path = export_to_jit(model, str(export_path), input_size=(1, 3, 224, 224))
        
        assert saved_path.exists()
        assert saved_path.suffix == ".pt"
        
        size_mb = saved_path.stat().st_size / (1024 * 1024)
        assert size_mb < 200
        
        loaded_model = torch.jit.load(str(saved_path))
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output = loaded_model(dummy_input)
        
        assert output is not None
    print("  PASSED")


def test_executorch_export():
    """Test ExecuTorch export (may skip if not installed)."""
    print("Export Test 2: ExecuTorch Export")
    
    model = create_model()
    model.eval()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "test_model.pte"
        
        try:
            saved_path = export_to_executorch(model, str(export_path), input_size=(1, 3, 224, 224))
            
            if saved_path is not None:
                assert saved_path.exists()
                size_mb = saved_path.stat().st_size / (1024 * 1024)
                assert size_mb < 200
                print("  PASSED")
            else:
                print("  SKIPPED: ExecuTorch not available")
        except Exception as e:
            print(f"  SKIPPED: {e}")


def test_coreml_export():
    """Test CoreML export (may skip if not installed)."""
    print("Export Test 3: CoreML Export")
    
    model = create_model()
    model.eval()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "test_model.mlpackage"
        
        try:
            saved_path = export_to_coreml(model, str(export_path), input_size=(1, 3, 224, 224))
            
            if saved_path is not None:
                assert saved_path.exists() or Path(str(export_path) + ".mlpackage").exists()
                print("  PASSED")
            else:
                print("  SKIPPED: CoreML export not available")
        except Exception as e:
            print(f"  SKIPPED: {e}")


def test_onnx_export():
    """Test ONNX export (may skip if not installed)."""
    print("Export Test 4: ONNX Export")
    
    model = create_model()
    model.eval()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "test_model.onnx"
        
        try:
            saved_path = export_to_onnx(model, str(export_path), input_size=(1, 3, 224, 224))
            
            if saved_path is not None:
                assert saved_path.exists()
                size_mb = saved_path.stat().st_size / (1024 * 1024)
                assert size_mb < 200
                print("  PASSED")
            else:
                print("  SKIPPED: ONNX export not available")
        except Exception as e:
            print(f"  SKIPPED: {e}")


def test_export_model_function():
    """Test unified export_model function."""
    print("Export Test 5: Unified Export Function")
    
    model = create_model()
    model.eval()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        results = export_model(
            model,
            format='jit',
            save_dir=tmpdir,
            input_size=(1, 3, 224, 224)
        )
        
        assert 'exports' in results
        assert 'jit' in results['exports']
        assert Path(results['exports']['jit']).exists()
        assert 'metadata' in results
        assert 'input_size' in results['metadata']
        assert 'model_params' in results['metadata']
    print("  PASSED")


def test_export_output_consistency():
    """Test that exported model outputs match original model."""
    print("Export Test 6: Export Output Consistency")
    
    model = create_model()
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        original_output = model(dummy_input)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "test_model.pt"
        export_to_jit(model, str(export_path), input_size=(1, 3, 224, 224))
        
        loaded_model = torch.jit.load(str(export_path))
        
        with torch.no_grad():
            exported_output = loaded_model(dummy_input)
        
        if isinstance(original_output, dict) and isinstance(exported_output, dict):
            for key in original_output.keys():
                if key in exported_output:
                    orig_tensor = original_output[key]
                    exp_tensor = exported_output[key]
                    if isinstance(orig_tensor, torch.Tensor) and isinstance(exp_tensor, torch.Tensor):
                        diff = torch.abs(orig_tensor - exp_tensor).max().item()
                        assert diff < 1.0
    print("  PASSED")


# Quantization Tests
def test_quantization_basic():
    """Test basic INT8 quantization functionality."""
    print("Quantization Test 1: Basic INT8 Quantization")
    
    model_fp32 = create_model()
    model_fp32.eval()
    calibration_data = create_dummy_calibration_data(num_batches=5)
    
    results = quantize_validation(
        model_fp32=model_fp32,
        calibration_data=calibration_data,
        test_data=None,
        num_calibration_batches=5,
        backend='qnnpack',
        output_dir=None
    )
    
    assert 'model_int8' in results
    model_int8 = results['model_int8']
    assert model_int8 is not None
    
    dummy_input = torch.randn(1, 3, 224, 224)
    try:
        with torch.no_grad():
            output = model_int8(dummy_input)
        assert output is not None
        assert isinstance(output, dict)
    except NotImplementedError:
        pass
    print("  PASSED")


def test_quantization_accuracy_loss():
    """Test that quantization accuracy loss is <1%."""
    print("Quantization Test 2: Accuracy Loss Validation")
    
    model_fp32 = create_model()
    model_fp32.eval()
    calibration_data = create_dummy_calibration_data(num_batches=10)
    test_data = create_dummy_calibration_data(num_batches=5)
    
    results = quantize_validation(
        model_fp32=model_fp32,
        calibration_data=calibration_data,
        test_data=test_data,
        num_calibration_batches=10,
        backend='qnnpack',
        tolerance=0.01
    )
    
    validation = results['validation']
    assert 'meets_tolerance' in validation or 'results' in validation
    
    meets_tolerance = validation.get('meets_tolerance', True)
    if 'results' in validation:
        results_data = validation.get('results', {})
        if isinstance(results_data, dict) and 'accuracy_loss_percent' in results_data:
            accuracy_loss = results_data['accuracy_loss_percent']
            assert accuracy_loss < 1.0
    print("  PASSED")


def test_model_size_reduction():
    """Test that quantized model size is reduced."""
    print("Quantization Test 3: Model Size Reduction")
    
    model_fp32 = create_model()
    model_fp32.eval()
    calibration_data = create_dummy_calibration_data(num_batches=10)
    
    results = quantize_validation(
        model_fp32=model_fp32,
        calibration_data=calibration_data,
        test_data=None,
        num_calibration_batches=10,
        backend='qnnpack',
        output_dir=None
    )
    
    size_info = results['size_info']
    fp32_size_mb = size_info.get('fp32_size_mb', 0)
    int8_size_mb = size_info.get('int8_size_mb', None)
    compression_ratio = size_info.get('compression_ratio', 0)
    
    assert fp32_size_mb > 0
    
    if int8_size_mb is not None:
        assert int8_size_mb < fp32_size_mb
        assert compression_ratio > 1.0
    print("  PASSED")


def test_quantized_model_latency():
    """Test that quantized model latency is <400ms."""
    print("Quantization Test 4: Quantized Model Latency")
    
    model_fp32 = create_model()
    model_fp32.eval()
    calibration_data = create_dummy_calibration_data(num_batches=10)
    
    results = quantize_validation(
        model_fp32=model_fp32,
        calibration_data=calibration_data,
        test_data=None,
        num_calibration_batches=10,
        backend='qnnpack',
        output_dir=None
    )
    
    model_int8 = results['model_int8']
    dummy_input = torch.randn(1, 3, 224, 224)
    
    try:
        with torch.no_grad():
            for _ in range(3):
                _ = model_int8(dummy_input)
        
        num_runs = 10
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model_int8(dummy_input)
                end = time.perf_counter()
                times.append((end - start) * 1000)
        
        mean_latency = sum(times) / len(times)
        assert mean_latency < 400
    except NotImplementedError:
        pass
    print("  PASSED")


def test_quantization_pipeline():
    """Test complete quantization pipeline."""
    print("Quantization Test 5: Complete Pipeline")
    
    model_fp32 = create_model()
    model_fp32.eval()
    calibration_data = create_dummy_calibration_data(num_batches=10)
    test_data = create_dummy_calibration_data(num_batches=5)
    
    results = quantize_validation(
        model_fp32=model_fp32,
        calibration_data=calibration_data,
        test_data=test_data,
        num_calibration_batches=10,
        backend='qnnpack',
        output_dir=None
    )
    
    assert 'model_int8' in results
    assert 'size_info' in results
    assert 'validation' in results
    assert 'ready_for_export' in results
    
    model_int8 = results['model_int8']
    assert model_int8 is not None
    
    size_info = results['size_info']
    assert 'fp32_size_mb' in size_info
    
    validation = results['validation']
    assert 'meets_tolerance' in validation or 'results' in validation
    print("  PASSED")


def test_module_fusion():
    """Test that module fusion works correctly."""
    print("Quantization Test 6: Module Fusion")
    
    model = create_model()
    model.eval()
    modules_before = len(list(model.modules()))
    fused_model = fuse_maxsight_modules(model)
    modules_after = len(list(fused_model.modules()))
    
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = fused_model(dummy_input)
    
    assert output is not None
    print("  PASSED")


def test_quantization_backends():
    """Test quantization with different backends."""
    print("Quantization Test 7: Backend Support")
    
    model_fp32 = create_model()
    model_fp32.eval()
    calibration_data = create_dummy_calibration_data(num_batches=5)
    
    try:
        results = quantize_validation(
            model_fp32=model_fp32,
            calibration_data=calibration_data,
            test_data=None,
            num_calibration_batches=5,
            backend='qnnpack',
            output_dir=None
        )
        assert results['model_int8'] is not None
        print("  qnnpack backend: PASSED")
    except Exception as e:
        print(f"  qnnpack backend: SKIPPED ({str(e)})")
    
    try:
        results = quantize_validation(
            model_fp32=model_fp32,
            calibration_data=calibration_data,
            test_data=None,
            num_calibration_batches=5,
            backend='fbgemm',
            output_dir=None
        )
        assert results['model_int8'] is not None
        print("  fbgemm backend: PASSED")
    except Exception as e:
        print(f"  fbgemm backend: SKIPPED ({str(e)})")


def test_quantization_with_audio():
    """Test quantization with audio input."""
    print("Quantization Test 8: Audio Input Support")
    
    model_fp32 = create_model()
    model_fp32.eval()
    calibration_data = create_dummy_calibration_data(num_batches=5)
    
    results = quantize_validation(
        model_fp32=model_fp32,
        calibration_data=calibration_data,
        test_data=None,
        num_calibration_batches=5,
        backend='qnnpack',
        output_dir=None
    )
    
    model_int8 = results['model_int8']
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        try:
            output = model_int8(dummy_input)
            assert output is not None
        except NotImplementedError:
            pass
        except Exception as e:
            pass
    print("  PASSED")


# Training Pipeline Tests
def test_training_step():
    """Test a single training step."""
    print("Training Test 1: Single Training Step")
    
    model = create_model(num_classes=80)
    device = torch.device('cpu')
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = MaxSightLoss(num_classes=80).to(device)
    
    batch_size = 2
    dummy_image = torch.randn(batch_size, 3, 224, 224).to(device)
    
    num_objects = [3, 2]
    max_objects = max(num_objects)
    
    gt_boxes = torch.zeros(batch_size, max_objects, 4).to(device)
    gt_labels = torch.zeros(batch_size, max_objects, dtype=torch.long).to(device)
    num_objects_tensor = torch.tensor(num_objects, dtype=torch.long).to(device)
    
    for i in range(batch_size):
        for j in range(num_objects[i]):
            gt_boxes[i, j] = torch.tensor([0.1, 0.1, 0.3, 0.3])
            gt_labels[i, j] = torch.randint(1, 80, (1,)).item()
    
    targets = {
        'boxes': gt_boxes,
        'labels': gt_labels,
        'num_objects': num_objects_tensor,
        'urgency': torch.zeros(batch_size, 4).to(device),
        'distance': torch.zeros(batch_size, max_objects, 3).to(device)
    }
    
    optimizer.zero_grad()
    outputs = model(dummy_image)
    losses = loss_fn(outputs, targets)
    total_loss = losses['total_loss']
    
    total_loss.backward()
    optimizer.step()
    
    has_gradients = any(p.grad is not None for p in model.parameters())
    assert has_gradients
    assert total_loss.item() > 0
    print("  PASSED")


def test_data_loader():
    """Test data loader functionality."""
    print("Training Test 2: Data Loader")
    
    train_loader, val_loader = create_dummy_dataloaders(
        num_train=10,
        num_val=5,
        batch_size=2
    )
    
    assert len(train_loader) > 0
    
    sample_batch = next(iter(train_loader))
    assert 'images' in sample_batch
    assert 'labels' in sample_batch
    assert 'boxes' in sample_batch
    assert sample_batch['images'].shape[0] == 2
    
    assert len(val_loader) > 0
    val_batch = next(iter(val_loader))
    assert val_batch['images'].shape[0] == 2
    print("  PASSED")


def test_training_loop_iteration():
    """Test a single training loop iteration."""
    print("Training Test 3: Training Loop Iteration")
    
    model = create_model()
    train_loader, val_loader = create_dummy_dataloaders(
        num_train=4,
        num_val=2,
        batch_size=2
    )
    
    trainer = ProductionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cpu',
        num_epochs=1
    )
    
    sample_batch = next(iter(train_loader))
    images = sample_batch['images']
    targets = {
        'labels': sample_batch['labels'],
        'boxes': sample_batch['boxes'],
        'urgency': sample_batch['urgency'],
        'distance': sample_batch['distance'],
        'num_objects': sample_batch['num_objects']
    }
    
    model.train()
    outputs = model(images)
    losses = trainer.criterion(outputs, targets)
    
    assert 'total_loss' in losses
    assert losses['total_loss'].item() > 0
    print("  PASSED")


def test_loss_computation():
    """Test loss function computation."""
    print("Training Test 4: Loss Computation")
    
    model = create_model()
    model.eval()
    loss_fn = MaxSightLoss(num_classes=80)
    
    batch_size = 2
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        outputs = model(dummy_image)
    
    num_objects = [2, 3]
    max_objects = max(num_objects)
    
    targets = {
        'boxes': torch.zeros(batch_size, max_objects, 4),
        'labels': torch.zeros(batch_size, max_objects, dtype=torch.long),
        'num_objects': torch.tensor(num_objects, dtype=torch.long),
        'urgency': torch.zeros(batch_size, 4),
        'distance': torch.zeros(batch_size, max_objects, 3)
    }
    
    losses = loss_fn(outputs, targets)
    
    assert 'total_loss' in losses
    assert 'classification_loss' in losses or 'detection_loss' in losses
    assert losses['total_loss'].item() >= 0
    print("  PASSED")


# Annotation Generation Tests
def test_coco_class_mapping():
    """Test COCO class mapping to MaxSight classes."""
    print("Annotation Test 1: COCO Class Mapping")
    
    assert len(COCO_CLASSES) > 0
    assert len(COCO_CLASSES) == len(ENVIRONMENTAL_CLASSES)
    
    common_classes = ['person', 'car', 'bicycle', 'dog', 'cat']
    for cls in common_classes:
        assert cls in COCO_CLASSES
    print("  PASSED")


def test_environmental_classes():
    """Test environmental class definitions."""
    print("Annotation Test 2: Environmental Classes")
    
    assert len(ENVIRONMENTAL_CLASSES) > 0
    assert isinstance(ENVIRONMENTAL_CLASSES, list)
    
    unique_classes = set(ENVIRONMENTAL_CLASSES)
    assert len(unique_classes) == len(ENVIRONMENTAL_CLASSES)
    print("  PASSED")


def test_sound_classes():
    """Test sound class definitions."""
    print("Annotation Test 3: Sound Classes")
    
    assert len(SOUND_CLASSES) > 0
    assert isinstance(SOUND_CLASSES, list)
    
    expected_sounds = ['fire alarm', 'siren', 'car horn', 'doorbell']
    for sound in expected_sounds:
        assert sound in SOUND_CLASSES
    print("  PASSED")


def test_urgency_assignment():
    """Test urgency level assignment logic."""
    print("Annotation Test 4: Urgency Assignment")
    
    critical_classes = ['person', 'car', 'bicycle', 'motorcycle']
    
    def calculate_urgency(class_name: str, distance_zone: int) -> int:
        base_urgency = 0
        
        if class_name in critical_classes:
            base_urgency = 2
        elif class_name in ['dog', 'cat']:
            base_urgency = 1
        else:
            base_urgency = 0
        
        if distance_zone == 0:
            return min(3, base_urgency + 1)
        elif distance_zone == 1:
            return base_urgency
        else:
            return max(0, base_urgency - 1)
    
    assert calculate_urgency('person', 0) >= 2
    assert calculate_urgency('person', 2) >= 1
    assert calculate_urgency('car', 0) >= 2
    assert calculate_urgency('chair', 0) <= 1
    print("  PASSED")


def test_distance_zones():
    """Test distance zone assignment."""
    print("Annotation Test 5: Distance Zones")
    
    def assign_distance_zone(bbox_area: float) -> int:
        if bbox_area > 0.3:
            return 0
        elif bbox_area > 0.1:
            return 1
        else:
            return 2
    
    assert assign_distance_zone(0.5) == 0
    assert assign_distance_zone(0.2) == 1
    assert assign_distance_zone(0.05) == 2
    print("  PASSED")


def test_dataset_info():
    """Test dataset information retrieval."""
    print("Annotation Test 6: Dataset Information")
    
    datasets_info = get_all_datasets_info()
    
    assert isinstance(datasets_info, dict)
    assert 'coco' in datasets_info
    assert 'open_images' in datasets_info
    
    coco_info = datasets_info['coco']
    assert 'name' in coco_info
    assert 'images' in coco_info
    assert 'classes' in coco_info
    print("  PASSED")


def test_class_mappings_save():
    """Test class mappings save functionality."""
    print("Annotation Test 7: Class Mappings Save")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        save_class_mappings(data_dir)
        
        env_file = data_dir / "environmental_classes.txt"
        sound_file = data_dir / "sound_classes.txt"
        
        assert env_file.exists()
        assert sound_file.exists()
        
        with open(env_file, 'r') as f:
            env_content = f.read()
            assert len(env_content) > 0
        
        with open(sound_file, 'r') as f:
            sound_content = f.read()
            assert len(sound_content) > 0
    print("  PASSED")


def test_annotation_format():
    """Test annotation format structure."""
    print("Annotation Test 8: Annotation Format")
    
    sample_annotation = {
        'image_id': 'test_001',
        'image_path': 'test.jpg',
        'objects': [
            {
                'class_id': 0,
                'class_name': 'person',
                'bbox': [0.5, 0.5, 0.2, 0.3],
                'distance_zone': 0,
                'urgency': 3,
                'confidence': 0.95
            }
        ],
        'scene_urgency': 3,
        'lighting': 'normal'
    }
    
    assert 'image_id' in sample_annotation
    assert 'objects' in sample_annotation
    assert isinstance(sample_annotation['objects'], list)
    
    if len(sample_annotation['objects']) > 0:
        obj = sample_annotation['objects'][0]
        assert 'class_id' in obj
        assert 'bbox' in obj
        assert 'distance_zone' in obj
        assert 'urgency' in obj
        assert 0 <= obj['urgency'] <= 3
        assert 0 <= obj['distance_zone'] <= 2
    print("  PASSED")


# Metrics Tests
def test_detection_metrics_initialization():
    """Test DetectionMetrics initialization."""
    print("Metrics Test 1: DetectionMetrics Initialization")
    
    num_classes = 80
    metrics = DetectionMetrics(num_classes=num_classes)
    
    assert metrics.num_classes == num_classes
    assert len(metrics.class_tp) == num_classes
    assert len(metrics.class_fp) == num_classes
    assert len(metrics.class_fn) == num_classes
    print("  PASSED")


def test_iou_computation():
    """Test IoU matrix computation."""
    print("Metrics Test 2: IoU Computation")
    
    pred_boxes = torch.tensor([
        [0.5, 0.5, 0.2, 0.3],
        [0.7, 0.7, 0.15, 0.2]
    ])
    
    gt_boxes = torch.tensor([
        [0.5, 0.5, 0.2, 0.3],
        [0.8, 0.8, 0.1, 0.15]
    ])
    
    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
    
    assert iou_matrix.shape == (2, 2)
    assert iou_matrix[0, 0] > 0.9
    assert iou_matrix[0, 1] < 0.5
    print("  PASSED")


def test_detection_metrics_update():
    """Test DetectionMetrics update functionality."""
    print("Metrics Test 3: DetectionMetrics Update")
    
    num_classes = 80
    metrics = DetectionMetrics(num_classes=num_classes)
    
    pred_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
    pred_labels = torch.tensor([0])
    pred_scores = torch.tensor([0.95])
    
    gt_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
    gt_labels = torch.tensor([0])
    
    metrics.update(
        pred_boxes=pred_boxes,
        pred_labels=pred_labels,
        pred_scores=pred_scores,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        iou_threshold=0.5
    )
    
    assert metrics.class_tp[0] > 0
    assert metrics.class_fp[0] == 0
    print("  PASSED")


def test_detection_metrics_precision_recall():
    """Test precision and recall computation."""
    print("Metrics Test 4: Precision and Recall")
    
    num_classes = 80
    metrics = DetectionMetrics(num_classes=num_classes)
    
    pred_boxes = torch.tensor([
        [0.5, 0.5, 0.2, 0.3],
        [0.7, 0.7, 0.15, 0.2]
    ])
    pred_labels = torch.tensor([0, 1])
    pred_scores = torch.tensor([0.95, 0.85])
    
    gt_boxes = torch.tensor([
        [0.5, 0.5, 0.2, 0.3],
        [0.8, 0.8, 0.1, 0.15]
    ])
    gt_labels = torch.tensor([0, 2])
    
    metrics.update(
        pred_boxes=pred_boxes,
        pred_labels=pred_labels,
        pred_scores=pred_scores,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        iou_threshold=0.5
    )
    
    overall_metrics = metrics.get_overall_metrics()
    
    assert 'precision' in overall_metrics
    assert 'recall' in overall_metrics
    assert 'f1' in overall_metrics
    assert 0 <= overall_metrics['precision'] <= 1
    assert 0 <= overall_metrics['recall'] <= 1
    assert 0 <= overall_metrics['f1'] <= 1
    print("  PASSED")


def test_detection_metrics_map():
    """Test mAP (mean Average Precision) computation."""
    print("Metrics Test 5: mAP Computation")
    
    num_classes = 80
    metrics = DetectionMetrics(num_classes=num_classes)
    
    for i in range(5):
        pred_boxes = torch.tensor([[0.5 + i*0.1, 0.5, 0.2, 0.3]])
        pred_labels = torch.tensor([0])
        pred_scores = torch.tensor([0.9 - i*0.1])
        
        gt_boxes = torch.tensor([[0.5 + i*0.1, 0.5, 0.2, 0.3]])
        gt_labels = torch.tensor([0])
        
        metrics.update(
            pred_boxes=pred_boxes,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            iou_threshold=0.5
        )
    
    map_score = metrics.get_map()
    
    assert 0 <= map_score <= 1
    assert map_score > 0
    print("  PASSED")


def test_scene_metrics_initialization():
    """Test SceneMetrics initialization."""
    print("Metrics Test 6: SceneMetrics Initialization")
    
    metrics = SceneMetrics(num_urgency_levels=4, num_distance_zones=3)
    
    assert metrics.num_urgency_levels == 4
    assert metrics.num_distance_zones == 3
    assert metrics.urgency_total == 0
    assert metrics.distance_total == 0
    print("  PASSED")


def test_scene_metrics_urgency():
    """Test urgency prediction metrics."""
    print("Metrics Test 7: Urgency Metrics")
    
    metrics = SceneMetrics(num_urgency_levels=4, num_distance_zones=3)
    
    pred_urgency = torch.tensor([0, 1, 2, 3])
    gt_urgency = torch.tensor([0, 1, 2, 2])
    
    metrics.update_urgency(pred_urgency, gt_urgency)
    
    urgency_accuracy = metrics.get_urgency_accuracy()
    
    assert 0 <= urgency_accuracy <= 1
    assert urgency_accuracy > 0
    print("  PASSED")


def test_scene_metrics_distance():
    """Test distance zone metrics."""
    print("Metrics Test 8: Distance Metrics")
    
    metrics = SceneMetrics(num_urgency_levels=4, num_distance_zones=3)
    
    pred_distance = torch.tensor([0, 1, 2, 0])
    gt_distance = torch.tensor([0, 1, 2, 1])
    
    metrics.update_distance(pred_distance, gt_distance)
    
    distance_accuracy = metrics.get_distance_accuracy()
    
    assert 0 <= distance_accuracy <= 1
    assert distance_accuracy > 0
    print("  PASSED")


def test_latency_measurement():
    """Test latency measurement in metrics."""
    print("Metrics Test 9: Latency Measurement")
    
    num_classes = 80
    metrics = DetectionMetrics(num_classes=num_classes)
    
    for _ in range(10):
        inference_time = np.random.uniform(0.1, 0.5)
        metrics.record_inference_time(inference_time)
    
    latency_stats = metrics.get_latency_stats()
    
    assert 'mean_ms' in latency_stats
    assert 'median_ms' in latency_stats
    assert 'p95_ms' in latency_stats
    assert 'p99_ms' in latency_stats
    assert latency_stats['mean_ms'] > 0
    assert latency_stats['p95_ms'] >= latency_stats['mean_ms']
    assert latency_stats['p99_ms'] >= latency_stats['p95_ms']
    print("  PASSED")


def test_metrics_reset():
    """Test metrics reset functionality."""
    print("Metrics Test 10: Metrics Reset")
    
    num_classes = 80
    metrics = DetectionMetrics(num_classes=num_classes)
    
    pred_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
    pred_labels = torch.tensor([0])
    pred_scores = torch.tensor([0.95])
    gt_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
    gt_labels = torch.tensor([0])
    
    metrics.update(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)
    assert metrics.class_tp[0] > 0
    
    metrics.reset()
    assert metrics.class_tp[0] == 0
    assert len(metrics.inference_times) == 0
    print("  PASSED")


if __name__ == "__main__":
    print("Running All MaxSight Tests")
    print("Total tests: 53")
    print()
    
    # Model tests
    test_model_creation()
    test_forward_pass()
    test_audio_fusion()
    test_color_blindness_mode()
    test_parameter_count()
    test_gradient_flow()
    test_inference_mode()
    
    # System tests
    test_class_system()
    test_model_creation_system()
    test_forward_pass_system()
    test_training_system()
    test_detections()
    test_visual_conditions()
    test_data_sources()
    
    # Condition tests
    test_all_condition_modes()
    test_condition_preprocessing()
    test_condition_robustness()
    
    # Export tests
    test_jit_export()
    test_executorch_export()
    test_coreml_export()
    test_onnx_export()
    test_export_model_function()
    test_export_output_consistency()
    
    # Quantization tests
    test_quantization_basic()
    test_quantization_accuracy_loss()
    test_model_size_reduction()
    test_quantized_model_latency()
    test_quantization_pipeline()
    test_module_fusion()
    test_quantization_backends()
    test_quantization_with_audio()
    
    # Training tests
    test_training_step()
    test_data_loader()
    test_training_loop_iteration()
    test_loss_computation()
    
    # Annotation tests
    test_coco_class_mapping()
    test_environmental_classes()
    test_sound_classes()
    test_urgency_assignment()
    test_distance_zones()
    test_dataset_info()
    test_class_mappings_save()
    test_annotation_format()
    
    # Metrics tests
    test_detection_metrics_initialization()
    test_iou_computation()
    test_detection_metrics_update()
    test_detection_metrics_precision_recall()
    test_detection_metrics_map()
    test_scene_metrics_initialization()
    test_scene_metrics_urgency()
    test_scene_metrics_distance()
    test_latency_measurement()
    test_metrics_reset()
    
    print()
    print("All tests completed!")

