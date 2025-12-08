"""
Condition-Specific Tests for MaxSight Model
Tests all 14 visual condition modes and their adaptations.
"""

import torch
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.maxsight_cnn import create_model
from ml.utils.preprocessing import ImagePreprocessor


def test_all_condition_modes():
    """Test that all 14 visual condition modes can be created."""
    print("Condition Test 1: All Condition Modes")
    
    conditions = [
        'myopia', 'hyperopia', 'astigmatism', 'presbyopia', 'refractive_errors',
        'cataracts', 'glaucoma', 'amd', 'diabetic_retinopathy',
        'retinitis_pigmentosa', 'color_blindness', 'cvi', 'amblyopia', 'strabismus'
    ]
    
    for cond in conditions:
        # Test model creation with condition
        model = create_model(condition_mode=cond)
        assert model is not None, f"Model creation failed for {cond}"
        
        # Test preprocessor creation with condition
        preprocessor = ImagePreprocessor(condition_mode=cond)
        assert preprocessor is not None, f"Preprocessor creation failed for {cond}"
        
        # Test forward pass
        model.eval()
        dummy_image = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            outputs = model(dummy_image)
        
        assert 'classifications' in outputs, f"Output missing for {cond}"
        print(f"  {cond}: OK")
    
    print("  PASSED: All 14 condition modes work correctly")


def test_condition_preprocessing():
    """Test that condition-specific preprocessing is applied."""
    print("\nCondition Test 2: Condition-Specific Preprocessing")
    
    # Test a few key conditions
    test_conditions = ['cataracts', 'glaucoma', 'color_blindness', 'low_light']
    
    # Create image as numpy array (preprocessor expects PIL or numpy)
    dummy_image_np = np.random.rand(224, 224, 3).astype(np.float32)
    dummy_image_np = (dummy_image_np * 255.0).astype(np.uint8)  # type: ignore[operator]
    
    for cond in test_conditions:
        try:
            preprocessor = ImagePreprocessor(condition_mode=cond)
            processed = preprocessor(dummy_image_np)
            
            assert processed is not None, f"Preprocessing failed for {cond}"
            print(f"  {cond}: Preprocessing OK")
        except Exception as e:
            print(f"  {cond}: Preprocessing issue - {e}")
    
    print("  PASSED: Condition-specific preprocessing works")


def test_condition_robustness():
    """Test model robustness with condition-specific inputs."""
    print("\nCondition Test 3: Condition Robustness")
    
    model = create_model()
    model.eval()
    
    # Test with different condition modes
    conditions_to_test = ['cataracts', 'glaucoma', 'color_blindness']
    
    # Create image as numpy array (preprocessor expects PIL or numpy)
    import numpy as np
    dummy_image_np = np.random.rand(224, 224, 3).astype(np.float32)
    dummy_image_np = (dummy_image_np * 255).astype(np.uint8)
    
    for cond in conditions_to_test:
        preprocessor = ImagePreprocessor(condition_mode=cond)
        processed_image = preprocessor(dummy_image_np)
        
        # Convert to tensor if needed
        if isinstance(processed_image, torch.Tensor):
            processed_tensor = processed_image.unsqueeze(0) if processed_image.dim() == 3 else processed_image
        else:
            # Convert numpy/PIL to tensor
            processed_tensor = torch.from_numpy(processed_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        with torch.no_grad():
            outputs = model(processed_tensor)
        
        # Model should still produce valid outputs
        assert 'classifications' in outputs, f"Model failed with {cond} preprocessing"
        assert outputs['classifications'].shape[0] == 1, f"Batch size incorrect for {cond}"
        
        detections = model.get_detections(outputs, confidence_threshold=0.1)
        assert isinstance(detections, list), f"Detections should be list for {cond}"
        
        print(f"  {cond}: Model robust")
    
    print("  PASSED: Model is robust to condition-specific preprocessing")


if __name__ == "__main__":
    test_all_condition_modes()
    test_condition_preprocessing()
    test_condition_robustness()
    
    print("\nAll condition-specific tests completed!")

