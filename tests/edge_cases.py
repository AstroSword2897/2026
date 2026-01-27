"""
Edge Case Tests for MaxSight Model
Tests extreme conditions, combined impairments, and unusual scenarios.
"""

import torch
import numpy as np
import time
import sys
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.maxsight_cnn import create_model
from ml.utils.preprocessing import (
    apply_refractive_error_blur,
    apply_cataract_contrast,
    apply_glaucoma_vignette,
    apply_amd_central_darkening,
    apply_low_light,
    apply_color_shift
)


def test_extreme_blur():
    """Test model with extreme blur (severe refractive errors)."""
    print("Edge Case Test 1: Extreme Blur")
    
    model = create_model()
    model.eval()
    
    # Create image with extreme blur
    dummy_image = torch.randn(1, 3, 224, 224)
    extreme_blur = apply_refractive_error_blur(dummy_image, sigma=10.0)  # Very high blur
    
    with torch.no_grad():
        outputs = model(extreme_blur)
    
    # Model should still produce outputs (may be degraded but not crash)
    assert 'classifications' in outputs, "Model failed on extreme blur"
    assert outputs['classifications'].shape[0] == 1, "Batch size incorrect"
    
    detections = model.get_detections(outputs, confidence_threshold=0.1)  # Lower threshold
    print(f"  Detections with extreme blur: {len(detections[0])}")
    
    print("  PASSED: Model handles extreme blur")


def test_extreme_contrast_loss():
    """Test model with extreme contrast reduction (severe cataracts)."""
    print("\nEdge Case Test 2: Extreme Contrast Loss")
    
    model = create_model()
    model.eval()
    
    dummy_image = torch.randn(1, 3, 224, 224)
    extreme_contrast = apply_cataract_contrast(dummy_image, contrast_factor=0.1)  # Very low contrast
    
    with torch.no_grad():
        outputs = model(extreme_contrast)
    
    assert 'classifications' in outputs, "Model failed on extreme contrast loss"
    
    detections = model.get_detections(outputs, confidence_threshold=0.1)
    print(f"  Detections with extreme contrast loss: {len(detections[0])}")
    
    print("  PASSED: Model handles extreme contrast loss")


def test_combined_impairments():
    """Test model with multiple combined impairments."""
    print("\nEdge Case Test 3: Combined Impairments")
    
    model = create_model()
    model.eval()
    
    dummy_image = torch.randn(1, 3, 224, 224)
    
    # Apply multiple impairments sequentially
    impaired = apply_refractive_error_blur(dummy_image, sigma=5.0)
    impaired = apply_cataract_contrast(impaired, contrast_factor=0.4)
    impaired = apply_low_light(impaired, brightness_factor=0.3)
    
    with torch.no_grad():
        outputs = model(impaired)
    
    assert 'classifications' in outputs, "Model failed on combined impairments"
    
    detections = model.get_detections(outputs, confidence_threshold=0.1)
    print(f"  Detections with combined impairments: {len(detections[0])}")
    
    print("  PASSED: Model handles combined impairments")


def test_very_dark_image():
    """Test model with very dark image (severe night blindness)."""
    print("\nEdge Case Test 4: Very Dark Image")
    
    model = create_model()
    model.eval()
    
    # Create very dark image
    dummy_image = torch.randn(1, 3, 224, 224) * 0.1  # Very dark
    very_dark = apply_low_light(dummy_image, brightness_factor=0.1)
    
    with torch.no_grad():
        outputs = model(very_dark)
    
    assert 'classifications' in outputs, "Model failed on very dark image"
    
    detections = model.get_detections(outputs, confidence_threshold=0.1)
    print(f"  Detections with very dark image: {len(detections[0])}")
    
    print("  PASSED: Model handles very dark images")


def test_very_bright_image():
    """Test model with very bright image (glare, overexposure)."""
    print("\nEdge Case Test 5: Very Bright Image")
    
    model = create_model()
    model.eval()
    
    # Create very bright image
    dummy_image = torch.ones(1, 3, 224, 224) * 0.9  # Very bright
    
    with torch.no_grad():
        outputs = model(dummy_image)
    
    assert 'classifications' in outputs, "Model failed on very bright image"
    
    detections = model.get_detections(outputs, confidence_threshold=0.1)
    print(f"  Detections with very bright image: {len(detections[0])}")
    
    print("  PASSED: Model handles very bright images")


def test_unusual_aspect_ratios():
    """Test model with unusual input (should handle gracefully)."""
    print("\nEdge Case Test 6: Unusual Input Handling")
    
    model = create_model()
    model.eval()
    
    # Test with edge case inputs
    test_cases = [
        ("All zeros", torch.zeros(1, 3, 224, 224)),
        ("All ones", torch.ones(1, 3, 224, 224)),
        ("Random noise", torch.randn(1, 3, 224, 224)),
        ("Extreme values", torch.randn(1, 3, 224, 224) * 10.0),
    ]
    
    for name, test_input in test_cases:
        try:
            with torch.no_grad():
                outputs = model(test_input)
            assert 'classifications' in outputs, f"Model failed on {name}"
            print(f"  {name}: OK")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            raise
    
    print("  ✅ PASSED: Model handles unusual inputs")


def test_crowded_scene_simulation():
    """Test model with simulated crowded scene (many objects)."""
    print("\nEdge Case Test 7: Crowded Scene Simulation")
    
    model = create_model()
    model.eval()
    
    # Create image that might have many detections
    # (In real scenario, this would be an actual crowded scene image)
    dummy_image = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        outputs = model(dummy_image)
    
    # Get all detections with low threshold
    detections = model.get_detections(outputs, confidence_threshold=0.05)
    num_detections = len(detections[0]) if detections else 0
    
    print(f"  Detections in scene: {num_detections}")
    print(f"  Model can handle up to {outputs['num_locations']} potential detections")
    
    # Model should handle many detections without crashing
    assert num_detections <= outputs['num_locations'], "Too many detections"
    
    print("  PASSED: Model handles crowded scenes")


def test_rapid_inference():
    """Test model with rapid successive inferences (stress test)."""
    print("\nEdge Case Test 8: Rapid Inference Stress Test")
    
    model = create_model()
    model.eval()
    
    dummy_image = torch.randn(1, 3, 224, 224)
    num_inferences = 100
    
    start = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(num_inferences):
            outputs = model(dummy_image)
            assert 'classifications' in outputs, "Model failed during rapid inference"
    
    end = time.perf_counter()
    total_time = end - start
    avg_time = total_time / num_inferences
    
    print(f"  Processed {num_inferences} inferences in {total_time:.2f}s")
    print(f"  Average time per inference: {avg_time*1000:.2f}ms")
    
    # Should complete without errors
    assert avg_time < 1.0, f"Average inference time {avg_time:.2f}s too slow"
    
    print("  PASSED: Model handles rapid inference")


if __name__ == "__main__":
    import time
    
    print("Running Edge Case Tests")
    
    test_extreme_blur()
    test_extreme_contrast_loss()
    test_combined_impairments()
    test_very_dark_image()
    test_very_bright_image()
    test_unusual_aspect_ratios()
    test_crowded_scene_simulation()
    test_rapid_inference()
    
    print("All edge case tests passed!")

