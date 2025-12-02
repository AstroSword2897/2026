"""Sprint 1 Demo Script - Showcase MaxSight CNN capabilities."""

import torch
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.maxsight_cnn import create_model
from ml.training.benchmark import benchmark_inference, print_benchmark_results
from ml.training.quantization import compare_model_sizes, print_quantization_results
from ml.utils.preprocessing import (
    apply_refractive_error_blur,
    apply_cataract_contrast,
    apply_glaucoma_peripheral_mask
)


def demo_basic_inference():
    #Demo 1: Basic object detection inference.
    print("DEMO 1: Basic Object Detection")
    
    model = create_model()
    model.eval()
    
    # Create the test image
    dummy_image = torch.randn(1, 3, 224, 224)
    
    print("\nRunning inference")
    with torch.no_grad():
        start = time.time()
        outputs = model(dummy_image)
        elapsed = (time.time() - start) * 1000
    
    print(f"Inference time: {elapsed:.1f} ms")
    print("\nModel Outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value).__name__}")
    
    # Get detections
    detections = model.get_detections(outputs, confidence_threshold=0.3)
    print(f"\nDetections: {len(detections[0])} objects found")
    
    if len(detections[0]) > 0:
        print("\nSample Detection:")
        sample = detections[0][0]
        print(f"  Class: {sample.get('class', 'unknown')}")
        print(f"  Confidence: {sample.get('confidence', 0.0):.3f}")
        print(f"  Box: {sample.get('box', [])}")


def demo_condition_specific():
    #Demo 2: Condition-specific adaptations.
    print("DEMO 2: Condition-Specific Adaptations")
    
    model = create_model()
    model.eval()
    
    dummy_image = torch.randn(1, 3, 224, 224)
    
    # Test different condition modes
    conditions = [
        (None, "Normal"),
        ("glaucoma", "Glaucoma"),
        ("amd", "AMD"),
        ("color_blindness", "Color Blindness")
    ]
    
    print("\nTesting condition-specific modes:")
    for condition_mode, name in conditions:
        cond_model = create_model(condition_mode=condition_mode)
        cond_model.eval()
        
        with torch.no_grad():
            outputs = cond_model(dummy_image)
        
        detections = cond_model.get_detections(outputs, confidence_threshold=0.3) 
        print(f"  {name}: {len(detections[0])} detections")


def demo_impairment_simulation():
    #Demo 3: Impairment simulation robustness.
    print("DEMO 3: Impairment Simulation Robustness")
    
    model = create_model()
    model.eval()
    
    dummy_image = torch.randn(1, 3, 224, 224)
    
    # Baseline
    with torch.no_grad():
        baseline_outputs = model(dummy_image)
    baseline_detections = model.get_detections(baseline_outputs, confidence_threshold=0.3)  # type: ignore
    baseline_count = len(baseline_detections[0])
    
    print(f"\nBaseline (normal): {baseline_count} detections")
    
    # Test impairments
    impairments = [
        ("Blur (Refractive Errors)", lambda img: apply_refractive_error_blur(img, sigma=3.0)),
        ("Contrast Reduction (Cataracts)", lambda img: apply_cataract_contrast(img, factor=0.5)),
        ("Peripheral Mask (Glaucoma)", lambda img: apply_glaucoma_peripheral_mask(img)),
    ]
    
    print("\nWith Impairments:")
    for name, transform_fn in impairments:
        impaired_image = transform_fn(dummy_image.clone())
        with torch.no_grad():
            impaired_outputs = model(impaired_image)
        impaired_detections = model.get_detections(impaired_outputs, confidence_threshold=0.3)  # type: ignore
        impaired_count = len(impaired_detections[0])
        
        degradation = abs(baseline_count - impaired_count) / max(baseline_count, 1) * 100
        print(f"  {name}: {impaired_count} detections ({degradation:.1f}% change)")


def demo_performance():
    #Demo 4: Performance benchmarks.
    print("DEMO 4: Performance Benchmarks")
    
    model = create_model()
    
    # Benchmark inference
    print("\nInference Latency:")
    results = benchmark_inference(model, num_runs=20)
    print_benchmark_results(results)
    
    # Model size
    print("\nModel Size:")
    size_info = compare_model_sizes(model)
    print(f"  Total Parameters: {size_info['total_parameters']:,}")
    print(f"  FP32 Size: {size_info['fp32_size_mb']:.1f} MB")
    print(f"  Target: <50 MB (after quantization)")


def main():
    #Run all demos.
    print("MaxSight CNN - Sprint 1 Demo")
    print("Mission: Remove barriers through environmental structuring")
    
    try:
        demo_basic_inference()
        demo_condition_specific()
        demo_impairment_simulation()
        demo_performance()
        
        print("All Demos Completed")
        
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()


