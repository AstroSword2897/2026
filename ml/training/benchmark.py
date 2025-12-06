# Inference latency benchmarking for MaxSight model.

import time
import torch
from typing import Dict, Optional
from pathlib import Path
import statistics


def benchmark_inference(
    model: torch.nn.Module,
    input_size: tuple = (1, 3, 224, 224),
    device: Optional[torch.device] = None,
    num_warmup: int = 5,
    num_runs: int = 50,
    batch_sizes: Optional[list] = None,
    input_shapes: Optional[Dict[int, tuple]] = None,
    save_path: Optional[str] = None
) -> Dict[str, float]:
    
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    if batch_sizes is None:
        batch_sizes = [1, 4, 8]
    
    results = {}
    
    for batch_size in batch_sizes:
        # Support different input shapes per batch size
        if input_shapes and batch_size in input_shapes:
            shape = input_shapes[batch_size]
            dummy_input = torch.randn(*shape, device=device)
        else:
            dummy_input = torch.randn(batch_size, *input_size[1:], device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy_input)
        
        # Synchronize if CUDA
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Timing runs
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = model(dummy_input)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
                times.append(elapsed)
        
        # Compute statistics
        results[f'batch_{batch_size}'] = {
            'mean_ms': statistics.mean(times),
            'median_ms': statistics.median(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'std_ms': statistics.stdev(times) if len(times) > 1 else 0.0,
        }
    
    # Overall stats (batch_size=1)
    if 'batch_1' in results:
        results['overall'] = results['batch_1'].copy()
    
    # Save results if path provided
    if save_path:
        save_benchmark_results(results, save_path, format='json')
    
    return results


def print_benchmark_results(results: Dict[str, Dict[str, float]]) -> None:
    print("\nInference Latency Benchmark Results")
    
    for key, stats in results.items():
        if key == 'overall':
            continue
        
        print(f"\nBatch Size {key.replace('batch_', '')}:")
        print(f"  Mean:   {stats['mean_ms']:.2f} ms")
        print(f"  Median: {stats['median_ms']:.2f} ms")
        print(f"  Min:    {stats['min_ms']:.2f} ms")
        print(f"  Max:    {stats['max_ms']:.2f} ms")
        print(f"  Std:    {stats['std_ms']:.2f} ms")
    
    if 'overall' in results:
        overall = results['overall']
        print(f"\nOverall (batch_size=1):")
        print(f"  Mean:   {overall['mean_ms']:.2f} ms")
        print(f"  Target: <500 ms")
        print(f"  Status: {'PASS' if overall['mean_ms'] < 500 else 'FAIL'}")
    
    print()


def save_benchmark_results(results: Dict[str, Dict[str, float]], save_path: str, format: str = 'json') -> None:
    """Save benchmark results to JSON or CSV."""
    save_path_obj = Path(save_path)
    
    if format.lower() == 'json':
        import json
        with open(save_path_obj, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Benchmark results saved to JSON: {save_path}")
    elif format.lower() == 'csv':
        try:
            import pandas as pd
            # Flatten nested dict for CSV
            rows = []
            for batch_key, stats in results.items():
                if batch_key == 'overall':
                    continue
                row = {'batch_size': batch_key.replace('batch_', '')}
                # Merge stats into row dict
                row = {**row, **stats}
                rows.append(row)
            df = pd.DataFrame(rows)
            df.to_csv(save_path_obj, index=False)
            print(f"Benchmark results saved to CSV: {save_path}")
        except ImportError:
            print("pandas not available, cannot save CSV. Install pandas or use JSON format.")
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")

