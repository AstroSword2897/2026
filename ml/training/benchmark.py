"""Inference latency benchmarking for MaxSight. Latency targets (e.g. <500 ms) matter for real-time assistive use so hazards and cues can be announced as the user moves."""

import logging
import statistics
import time
from pathlib import Path
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


def benchmark_inference(
    model: torch.nn.Module,
    input_size: tuple = (1, 3, 224, 224),
    device: Optional[torch.device] = None,
    num_warmup: int = 5,
    num_runs: int = 50,
    batch_sizes: Optional[list] = None,
    input_shapes: Optional[Dict[int, tuple]] = None,
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """Measure inference latency (ms) per batch size. Ensures the model can run in real time on target devices for assistive deployment."""
    if device is None:
        device = next(model.parameters()).device
    if isinstance(device, str):
        device = torch.device(device)
    model.eval()
    if batch_sizes is None:
        batch_sizes = [1, 4, 8]
    results = {}
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    for batch_size in batch_sizes:
        if input_shapes and batch_size in input_shapes:
            shape = input_shapes[batch_size]
            dummy_input = torch.randn(*shape, device=device)
        else:
            dummy_input = torch.randn(batch_size, *input_size[1:], device=device)
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(dummy_input)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
        times_sorted = sorted(times)
        p95_idx = int(len(times_sorted) * 0.95)
        p99_idx = int(len(times_sorted) * 0.99)
        
        batch_results = {
            'mean_ms': statistics.mean(times),
            'median_ms': statistics.median(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'std_ms': statistics.stdev(times) if len(times) > 1 else 0.0,
            'p95_ms': times_sorted[p95_idx] if p95_idx < len(times_sorted) else times_sorted[-1],
            'p99_ms': times_sorted[p99_idx] if p99_idx < len(times_sorted) else times_sorted[-1],
        }
        if device.type == 'cuda':
            batch_results['peak_memory_mb'] = torch.cuda.max_memory_allocated() / (1024 * 1024)
            batch_results['memory_reserved_mb'] = torch.cuda.max_memory_reserved() / (1024 * 1024)
            # Reset for next batch size.
            torch.cuda.reset_peak_memory_stats()
        
        # Calculate throughput (FPS)
        batch_results['fps'] = 1000.0 / batch_results['mean_ms'] if batch_results['mean_ms'] > 0 else 0.0
        
        results[f'batch_{batch_size}'] = batch_results
    
    # Overall stats (batch_size=1)
    if 'batch_1' in results:
        results['overall'] = results['batch_1'].copy()
    
    # Final GPU memory stats if CUDA.
    if device.type == 'cuda':
        results['final_peak_memory_mb'] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        results['final_memory_reserved_mb'] = torch.cuda.max_memory_reserved() / (1024 * 1024)
    
    # Save results if path provided.
    if save_path:
        save_benchmark_results(results, save_path, format='json')
    
    return results


def log_benchmark_results(results: Dict[str, Dict[str, float]]) -> None:
    """Log inference latency stats per batch size and overall pass/fail vs 500 ms target."""
    logger.info("Inference Latency Benchmark Results")
    for key, stats in results.items():
        if key == 'overall':
            continue
        logger.info("Batch %s: mean=%.2f ms, median=%.2f ms, min=%.2f, max=%.2f, std=%.2f",
                    key.replace('batch_', ''), stats['mean_ms'], stats['median_ms'],
                    stats['min_ms'], stats['max_ms'], stats['std_ms'])
    if 'overall' in results:
        o = results['overall']
        status = 'PASS' if o['mean_ms'] < 500 else 'FAIL'
        logger.info("Overall (batch_size=1): mean=%.2f ms, target <500 ms: %s", o['mean_ms'], status)


def save_benchmark_results(
    results: Dict[str, Dict[str, float]],
    save_path: str,
    format: str = 'json',
) -> None:
    """Write benchmark results to JSON or CSV file."""
    import json
    save_path_obj = Path(save_path)
    if format.lower() == 'json':
        with open(save_path_obj, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info("Benchmark results saved to JSON: %s", save_path)
    elif format.lower() == 'csv':
        try:
            import pandas as pd
            rows = []
            for batch_key, stats in results.items():
                if batch_key == 'overall':
                    continue
                rows.append({'batch_size': batch_key.replace('batch_', ''), **stats})
            pd.DataFrame(rows).to_csv(save_path_obj, index=False)
            logger.info("Benchmark results saved to CSV: %s", save_path)
        except ImportError:
            logger.warning("pandas not available; use JSON format or install pandas.")
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")







