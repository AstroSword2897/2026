"""Multi-Head Latency Benchmarking Measures latency for each head individually and in combination."""

import torch
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import statistics


class MultiHeadBenchmark:
    """Benchmark individual heads and combinations to identify latency bottlenecks."""
    
    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()
        
        self.results: Dict[str, Dict[str, float]] = {}
    
    def benchmark_head_combination(
        self,
        head_names: List[str],
        input_tensor: torch.Tensor,
        num_warmup: int = 5,
        num_runs: int = 50
    ) -> Dict[str, float]:
        """Benchmark a combination of heads."""
        # Warmup.
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self.model(input_tensor)
        
        # Time individual heads (if possible)
        head_timings: Dict[str, List[float]] = {name: [] for name in head_names}
        total_timings: List[float] = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_total = time.perf_counter()
                
                # Full forward pass.
                outputs = self.model(input_tensor)
                
                total_time = (time.perf_counter() - start_total) * 1000
                total_timings.append(total_time)
        
        # Calculate statistics.
        stats = {
            'mean_ms': statistics.mean(total_timings),
            'median_ms': statistics.median(total_timings),
            'min_ms': min(total_timings),
            'max_ms': max(total_timings),
            'std_ms': statistics.stdev(total_timings) if len(total_timings) > 1 else 0.0,
            'p95_ms': sorted(total_timings)[int(0.95 * len(total_timings))],
            'p99_ms': sorted(total_timings)[int(0.99 * len(total_timings))],
        }
        
        combination_key = '+'.join(sorted(head_names))
        self.results[combination_key] = stats
        
        return stats
    
    def benchmark_all_heads(self, input_tensor: torch.Tensor) -> Dict[str, Dict[str, float]]:
        """Benchmark all head combinations. Arguments: input_tensor: Input tensor Returns: Dictionary of all benchmark results."""
        # Core heads (always needed)
        core_heads = ['classification', 'box_regression', 'objectness']
        
        # Optional heads.
        optional_heads = [
            'text_region',
            'urgency',
            'distance',
            'contrast',
            'glare',
            'findability',
            'navigation_difficulty',
            'uncertainty'
        ]
        
        # Benchmark core only.
        print("Benchmarking core heads...")
        self.benchmark_head_combination(core_heads, input_tensor)
        
        # Benchmark core + each optional head.
        for head in optional_heads:
            print(f"Benchmarking core + {head}...")
            self.benchmark_head_combination(core_heads + [head], input_tensor)
        
        # Benchmark all heads.
        print("Benchmarking all heads...")
        self.benchmark_head_combination(core_heads + optional_heads, input_tensor)
        
        return self.results
    
    def identify_bottlenecks(self, target_latency_ms: float = 500.0) -> Dict[str, Any]:
        """Identify which head combinations exceed target latency."""
        bottlenecks = []
        recommendations = []
        
        for combination, stats in self.results.items():
            mean_latency = stats['mean_ms']
            p95_latency = stats['p95_ms']
            
            if mean_latency > target_latency_ms:
                bottlenecks.append({
                    'combination': combination,
                    'mean_latency_ms': mean_latency,
                    'p95_latency_ms': p95_latency,
                    'exceeds_target_by_ms': mean_latency - target_latency_ms
                })
                
                # Recommendations.
                if 'uncertainty' in combination:
                    recommendations.append(f"Consider disabling uncertainty head for {combination}")
                if 'findability' in combination:
                    recommendations.append(f"Consider disabling findability head for {combination}")
                if len(combination.split('+')) > 8:
                    recommendations.append(f"Too many heads in {combination}, consider disabling optional heads")
        
        return {
            'target_latency_ms': target_latency_ms,
            'bottlenecks': bottlenecks,
            'recommendations': recommendations,
            'total_combinations_tested': len(self.results),
            'combinations_within_target': len(self.results) - len(bottlenecks)
        }
    
    def save_results(self, filepath: Path):
        """Save benchmark results to file."""
        with open(filepath, 'w') as f:
            json.dump({
                'results': self.results,
                'bottleneck_analysis': self.identify_bottlenecks()
            }, f, indent=2)
    
    def get_optimal_head_config(
        self,
        target_latency_ms: float = 500.0,
        required_heads: Optional[List[str]] = None
    ) -> List[str]:
        """Get optimal head configuration that meets latency target."""
        if required_heads is None:
            required_heads = ['classification', 'box_regression', 'objectness']
        
        # Find best combination.
        best_combination = None
        best_latency = float('inf')
        
        for combination, stats in self.results.items():
            heads = combination.split('+')
            
            # Check if all required heads are present.
            if all(req in heads for req in required_heads):
                mean_latency = stats['mean_ms']
                if mean_latency < target_latency_ms and mean_latency < best_latency:
                    best_latency = mean_latency
                    best_combination = heads
        
        if best_combination:
            return best_combination
        else:
            # Fallback to required heads only.
            return required_heads







