"""
Multi-Head Latency Benchmark Tests
Tests latency for different head combinations to identify bottlenecks.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.maxsight_cnn import create_model
from ml.utils.multihead_benchmark import MultiHeadBenchmark


def test_multihead_latency():
    """Test latency for different head combinations."""
    print("Multi-Head Latency Benchmark Test")
    
    model = create_model()
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    benchmark = MultiHeadBenchmark(model, device=device)
    
    # Test input
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    
    # Benchmark all head combinations
    results = benchmark.benchmark_all_heads(dummy_image)
    
    # Analyze bottlenecks
    bottleneck_analysis = benchmark.identify_bottlenecks(target_latency_ms=500.0)
    
    print("\nBenchmark Results:")
    for combination, stats in results.items():
        print(f"  {combination}: {stats['mean_ms']:.2f}ms (P95: {stats['p95_ms']:.2f}ms)")
    
    print("\nBottleneck Analysis:")
    print(f"  Target latency: {bottleneck_analysis['target_latency_ms']}ms")
    print(f"  Combinations within target: {bottleneck_analysis['combinations_within_target']}")
    print(f"  Bottlenecks found: {len(bottleneck_analysis['bottlenecks'])}")
    
    if bottleneck_analysis['bottlenecks']:
        print("\n  Bottlenecks:")
        for bottleneck in bottleneck_analysis['bottlenecks']:
            print(f"    - {bottleneck['combination']}: {bottleneck['mean_latency_ms']:.2f}ms")
    
    if bottleneck_analysis['recommendations']:
        print("\n  Recommendations:")
        for rec in bottleneck_analysis['recommendations']:
            print(f"    - {rec}")
    
    # Get optimal configuration
    optimal_config = benchmark.get_optimal_head_config(
        target_latency_ms=500.0,
        required_heads=['classification', 'box_regression', 'objectness']
    )
    
    print(f"\n  Optimal head configuration: {optimal_config}")
    
    # Assert that at least core heads meet target
    # Try different key orderings (sorted vs unsorted)
    core_key = 'classification+box_regression+objectness'
    if core_key not in results:
        # Try sorted version
        core_key = 'box_regression+classification+objectness'
    
    core_latency = results.get(core_key, {}).get('mean_ms', float('inf'))
    assert core_latency < 500.0, f"Core heads latency {core_latency:.2f}ms exceeds 500ms target"
    
    print("\n✅ Multi-head benchmark test passed")


if __name__ == "__main__":
    test_multihead_latency()

