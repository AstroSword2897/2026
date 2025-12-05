"""
Comprehensive System Tests - Maximum Data & Classes
Tests the complete MaxSight system with 347 classes for user guidance
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.maxsight_cnn import (
    COCO_CLASSES, COCO_BASE_CLASSES, ACCESSIBILITY_CLASSES,
    create_model, MaxSightCNN
)
from ml.training.train_production import (
    NUM_CLASSES, ProductionTrainer, create_dummy_dataloaders
)
from ml.training.losses import MaxSightLoss
from ml.utils.preprocessing import ImagePreprocessor
from collections import Counter


def test_class_system():
    """Test comprehensive class system"""
    print("Test 1: Comprehensive Class System")
    # Check for duplicates
    duplicates = [item for item, count in Counter(COCO_CLASSES).items() if count > 1]
    assert len(duplicates) == 0, f"Found duplicates: {duplicates}"
    
    # Verify counts
    assert len(COCO_BASE_CLASSES) == 80, f"COCO base should be 80, got {len(COCO_BASE_CLASSES)}"
    assert len(COCO_CLASSES) == NUM_CLASSES, f"Class count mismatch: {len(COCO_CLASSES)} != {NUM_CLASSES}"
    
    return True


def test_model_creation():
    """Test model creation with comprehensive classes"""
    print("Test 2: Model Creation")
    model = create_model()
    assert model.num_classes == len(COCO_CLASSES), f"Model classes {model.num_classes} != {len(COCO_CLASSES)}"
    assert model.cls_head[-1].out_channels == len(COCO_CLASSES), "Classification head mismatch"
    
    total_params = sum(p.numel() for p in model.parameters())
    int8_size_mb = total_params / 1024 / 1024
    
    assert int8_size_mb < 50, f"Model size {int8_size_mb:.1f} MB exceeds target of 50 MB"
    
    return True


def test_forward_pass():
    """Test forward pass with audio"""
    print("Test 3: Forward Pass")
    model = create_model()
    model.eval()
    
    dummy_image = torch.randn(2, 3, 224, 224)
    dummy_audio = torch.randn(2, 128)
    
    with torch.no_grad():
        outputs = model(dummy_image, dummy_audio)
    
    assert outputs['classifications'].shape == (2, 196, len(COCO_CLASSES)), "Classification shape mismatch"
    assert outputs['boxes'].shape == (2, 196, 4), "Box shape mismatch"
    assert outputs['objectness'].shape == (2, 196), "Objectness shape mismatch"
    assert outputs['text_regions'].shape == (2, 196), "Text regions shape mismatch"
    assert outputs['scene_embedding'].shape == (2, 512), "Scene embedding shape mismatch"
    assert outputs['urgency_scores'].shape == (2, 4), "Urgency scores shape mismatch (should be scene-level)"
    assert outputs['distance_zones'].shape == (2, 196, 3), "Distance zones shape mismatch"
    assert outputs['num_locations'] == 196, "Num locations should be 196 (14x14)"
    
    return True


def test_training_system():
    """Test training system"""
    print("Test 4: Training System")
    model = create_model()
    train_loader, val_loader = create_dummy_dataloaders(num_train=20, num_val=5, batch_size=2)
    
    trainer = ProductionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cpu',
        num_epochs=1
    )
    
    assert trainer.criterion.num_classes == len(COCO_CLASSES), "Trainer class mismatch"
    
    # Test loss computation
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
    
    assert 'total_loss' in losses, "Missing total_loss"
    assert losses['total_loss'].item() > 0, "Loss should be positive"
    
    return True


def test_detections():
    """Test detection system"""
    print("Test 5: Detection System")
    model = create_model()
    model.eval()
    
    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_audio = torch.randn(1, 128)
    
    with torch.no_grad():
        outputs = model(dummy_image, dummy_audio)
        detections = model.get_detections(outputs, confidence_threshold=0.3)
    
    assert isinstance(detections, list), "Detections should be a list"
    assert len(detections) == 1, "Should have detections for 1 image"
    
    if len(detections[0]) > 0:
        det = detections[0][0]
        assert 'class' in det, "Detection missing class"
        assert 'class_name' in det, "Detection missing class_name"
        assert det['class'] < len(COCO_CLASSES), "Class ID out of range"
    
    return True


def test_visual_conditions():
    """Test all visual condition modes"""
    print("Test 6: Visual Condition Support")
    conditions = [
        'myopia', 'hyperopia', 'astigmatism', 'presbyopia', 'refractive_errors',
        'cataracts', 'glaucoma', 'amd', 'diabetic_retinopathy',
        'retinitis_pigmentosa', 'color_blindness', 'cvi', 'amblyopia', 'strabismus'
    ]
    
    for cond in conditions:
        model = create_model(condition_mode=cond)
        preprocessor = ImagePreprocessor(condition_mode=cond)
    
    return True


def test_data_sources():
    """Test data source configuration"""
    print("Test 7: Data Sources")
    # Verify class counts are correct
    assert len(COCO_BASE_CLASSES) == 80, "COCO base classes should be 80"
    assert len(COCO_CLASSES) > 0, "Total classes should be greater than 0"
    assert len(ACCESSIBILITY_CLASSES) > 0, "Accessibility classes should exist"
    
    return True


if __name__ == "__main__":
    tests = [
        test_class_system,
        test_model_creation,
        test_forward_pass,
        test_training_system,
        test_detections,
        test_visual_conditions,
        test_data_sources
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            failed += 1
    
    if failed == 0:
        sys.exit(0)
    else:
        sys.exit(1)

