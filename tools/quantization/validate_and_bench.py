"""Production validation and benchmarking for quantized MaxSight models. Compares FP32 vs INT8 across all heads with detailed metrics."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
import time
import numpy as np
from pathlib import Path
import json
import sys

# Add parent directory to path for imports.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class QuantizationValidator:
    """Comprehensive validator for quantized models with per-head metrics. Integrated with MaxSight output format."""
    
    def __init__(
        self,
        model_fp32: nn.Module,
        model_int8: nn.Module,
        test_loader: DataLoader,
        device: str = 'cpu'
    ):
        self.model_fp32 = model_fp32.eval().to(device)
        self.model_int8 = model_int8.eval().to(device)
        self.test_loader = test_loader
        self.device = device
        
    def compute_relative_error(
        self, 
        fp32_out: torch.Tensor, 
        int8_out: torch.Tensor
    ) -> Dict[str, float]:
        """Compute various error metrics between FP32 and INT8 outputs."""
        fp32_out = fp32_out.detach().float().cpu()
        int8_out = int8_out.detach().float().cpu()
        
        # Absolute error.
        abs_diff = (fp32_out - int8_out).abs()
        
        # Relative error (avoiding division by zero)
        rel_diff = abs_diff / (fp32_out.abs() + 1e-8)
        
        # Mean Squared Error.
        mse = ((fp32_out - int8_out) ** 2).mean().item()
        
        # Mean Absolute Error.
        mae = abs_diff.mean().item()
        
        # Signal-to-Noise Ratio (with proper edge case handling)
        signal_power = (fp32_out ** 2).mean().item()
        noise_power = ((fp32_out - int8_out) ** 2).mean().item()
        if signal_power < 1e-8:
            snr_db = -float('inf')
        elif noise_power < 1e-8:
            snr_db = float('inf')
        else:
            snr_db = 10 * np.log10(signal_power / noise_power)
        
        return {
            'mse': mse,
            'mae': mae,
            'max_abs_diff': abs_diff.max().item(),
            'mean_rel_diff': rel_diff.mean().item(),
            'max_rel_diff': rel_diff.max().item(),
            'snr_db': snr_db
        }
    
    def validate_classification_head(
        self,
        fp32_logits: torch.Tensor,
        int8_logits: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Validate classification head with accuracy metrics. MaxSight format: [B, num_locations, num_classes]."""
        # Shape validation.
        if fp32_logits.shape != int8_logits.shape:
            raise ValueError(f"Shape mismatch: FP32 {fp32_logits.shape} vs INT8 {int8_logits.shape}")
        
        # Flatten for per-location classification.
        B, num_locations, num_classes = fp32_logits.shape
        fp32_flat = fp32_logits.reshape(-1, num_classes)
        int8_flat = int8_logits.reshape(-1, num_classes)
        
        # Output error.
        output_error = self.compute_relative_error(fp32_flat, int8_flat)
        
        results = {
            **{f'logits_{k}': v for k, v in output_error.items()}
        }
        
        # If targets available, compute accuracy.
        if targets is not None:
            # Targets format: [B, max_objects] with class indices.
            # For MaxSight, we need to match predictions to ground truth.
            # Simplified matching; full matching requires IoU computation.
            fp32_preds = fp32_flat.argmax(dim=1)
            int8_preds = int8_flat.argmax(dim=1)
            
            # Agreement between FP32 and INT8 predictions.
            agreement = (fp32_preds == int8_preds).float().mean().item()
            results['prediction_agreement'] = agreement
        
        return results
    
    def validate_bbox_head(
        self,
        fp32_bbox: torch.Tensor,
        int8_bbox: torch.Tensor,
        targets_bbox: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Validate bounding box head (regression). MaxSight format: [B, num_locations, 4] (cx, cy, w, h normalized)"""
        # Shape validation.
        if fp32_bbox.shape != int8_bbox.shape:
            return {'_bbox_error': f"Shape mismatch: FP32 {fp32_bbox.shape} vs INT8 {int8_bbox.shape}"}
        
        output_error = self.compute_relative_error(fp32_bbox, int8_bbox)
        
        results: Dict[str, Any] = {
            'bbox_mae': output_error['mae'],
            'bbox_mse': output_error['mse'],
            'bbox_max_rel_diff': output_error['max_rel_diff'],
            'bbox_snr_db': output_error['snr_db']
        }
        
        # If ground truth available, compute IoU approximation.
        if targets_bbox is not None:
            # MaxSight uses center format (cx, cy, w, h)
            def compute_iou_center_format(pred, target):
                # Convert to corner format.
                def center_to_corner(boxes):
                    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    return torch.stack([x1, y1, x2, y2], dim=1)
                
                pred_corners = center_to_corner(pred)
                target_corners = center_to_corner(target)
                
                # Intersection.
                x1 = torch.max(pred_corners[:, 0], target_corners[:, 0])
                y1 = torch.max(pred_corners[:, 1], target_corners[:, 1])
                x2 = torch.min(pred_corners[:, 2], target_corners[:, 2])
                y2 = torch.min(pred_corners[:, 3], target_corners[:, 3])
                
                inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
                
                # Union.
                pred_area = (pred_corners[:, 2] - pred_corners[:, 0]) * (pred_corners[:, 3] - pred_corners[:, 1])
                target_area = (target_corners[:, 2] - target_corners[:, 0]) * (target_corners[:, 3] - target_corners[:, 1])
                union_area = pred_area + target_area - inter_area
                
                iou = inter_area / (union_area + 1e-6)
                return iou.mean().item()
            
            try:
                # Flatten for comparison.
                fp32_flat = fp32_bbox.reshape(-1, 4)
                int8_flat = int8_bbox.reshape(-1, 4)
                targets_flat = targets_bbox.reshape(-1, 4)
                
                fp32_iou = compute_iou_center_format(fp32_flat.cpu(), targets_flat.cpu())
                int8_iou = compute_iou_center_format(int8_flat.cpu(), targets_flat.cpu())
                results['fp32_iou'] = fp32_iou
                results['int8_iou'] = int8_iou
                results['iou_drop'] = fp32_iou - int8_iou
            except Exception as e:
                # Store error message separately (will be filtered out during aggregation)
                results['_iou_error'] = str(e)  # type: ignore
                results['fp32_iou'] = 0.0
                results['int8_iou'] = 0.0
                results['iou_drop'] = 0.0
        
        return results
    
    def validate_embedding_head(
        self,
        fp32_embed: torch.Tensor,
        int8_embed: torch.Tensor
    ) -> Dict[str, float]:
        """Validate embedding head with cosine similarity. MaxSight format: [B, embedding_dim] (scene_embedding)"""
        # Shape validation.
        if fp32_embed.shape != int8_embed.shape:
            raise ValueError(f"Shape mismatch: FP32 {fp32_embed.shape} vs INT8 {int8_embed.shape}")
        
        # Normalize embeddings.
        fp32_norm = torch.nn.functional.normalize(fp32_embed, p=2, dim=1)
        int8_norm = torch.nn.functional.normalize(int8_embed, p=2, dim=1)
        
        # Cosine similarity between FP32 and INT8 embeddings.
        cos_sim = (fp32_norm * int8_norm).sum(dim=1)
        
        # L2 distance.
        l2_dist = torch.norm(fp32_embed - int8_embed, p=2, dim=1)
        
        output_error = self.compute_relative_error(fp32_embed, int8_embed)
        
        return {
            'embed_mean_cosine_sim': cos_sim.mean().item(),
            'embed_min_cosine_sim': cos_sim.min().item(),
            'embed_mean_l2_dist': l2_dist.mean().item(),
            'embed_max_l2_dist': l2_dist.max().item(),
            **{f'embed_{k}': v for k, v in output_error.items()}
        }
    
    def validate_urgency_head(
        self,
        fp32_urgency: torch.Tensor,
        int8_urgency: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Validate urgency classification head. MaxSight format: [B, num_urgency_levels] (typically 4)"""
        # Shape validation.
        if fp32_urgency.shape != int8_urgency.shape:
            raise ValueError(f"Shape mismatch: FP32 {fp32_urgency.shape} vs INT8 {int8_urgency.shape}")
        
        output_error = self.compute_relative_error(fp32_urgency, int8_urgency)
        
        results = {
            'urgency_mae': output_error['mae'],
            'urgency_max_rel_diff': output_error['max_rel_diff'],
            'urgency_snr_db': output_error['snr_db']
        }
        
        # Classification accuracy.
        fp32_preds = fp32_urgency.argmax(dim=1)
        int8_preds = int8_urgency.argmax(dim=1)
        
        # Agreement between FP32 and INT8.
        agreement = (fp32_preds == int8_preds).float().mean().item()
        results['urgency_prediction_agreement'] = agreement
        
        if targets is not None:
            targets = targets.to(self.device)
            fp32_acc = (fp32_preds == targets).float().mean().item()
            int8_acc = (int8_preds == targets).float().mean().item()
            results['fp32_urgency_acc'] = fp32_acc
            results['int8_urgency_acc'] = int8_acc
            results['urgency_acc_drop'] = fp32_acc - int8_acc
        
        return results
    
    def validate_objectness_head(
        self,
        fp32_obj: torch.Tensor,
        int8_obj: torch.Tensor
    ) -> Dict[str, float]:
        """Validate objectness head. MaxSight format: [B, num_locations]."""
        # Shape validation.
        if fp32_obj.shape != int8_obj.shape:
            raise ValueError(f"Shape mismatch: FP32 {fp32_obj.shape} vs INT8 {int8_obj.shape}")
        
        output_error = self.compute_relative_error(fp32_obj, int8_obj)
        
        return {
            'objectness_mae': output_error['mae'],
            'objectness_mse': output_error['mse'],
            'objectness_max_rel_diff': output_error['max_rel_diff'],
            'objectness_snr_db': output_error['snr_db']
        }
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation across all MaxSight heads. Returns detailed metrics dictionary."""
        print("Running Full Quantization Validation")
        
        from collections import defaultdict
        all_metrics: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        
        total_batches = len(self.test_loader)
        print(f"Processing {total_batches} batches for validation...")
        
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                try:
                    # Parse batch (MaxSight dataset format)
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0]
                        targets = batch[1] if len(batch) > 1 else {}
                    elif isinstance(batch, dict):
                        inputs = batch.get('images') or batch.get('image')
                        targets = {k: v for k, v in batch.items() if k not in ['images', 'image']}
                    else:
                        continue
                    
                    if inputs is None:
                        continue
                    
                    if not isinstance(inputs, torch.Tensor):
                        continue
                        
                    inputs = inputs.to(self.device)
                    
                    # Forward pass.
                    fp32_out = self.model_fp32(inputs)
                    int8_out = self.model_int8(inputs)
                    
                    # Validate each head (MaxSight output format)
                    if isinstance(fp32_out, dict):
                        # Classification head.
                        if 'classifications' in fp32_out:
                            class_targets = targets.get('labels') if isinstance(targets, dict) else None
                            if class_targets is not None and torch.is_tensor(class_targets):
                                class_targets = class_targets.to(self.device)
                            try:
                                metrics = self.validate_classification_head(
                                    fp32_out['classifications'],
                                    int8_out['classifications'],
                                    class_targets
                                )
                                all_metrics['classification'].append(metrics)
                            except Exception as e:
                                print(f"Warning: Classification validation failed: {e}")
                        
                        # Bounding box head.
                        if 'boxes' in fp32_out:
                            bbox_targets = targets.get('boxes') if isinstance(targets, dict) else None
                            if bbox_targets is not None and torch.is_tensor(bbox_targets):
                                bbox_targets = bbox_targets.to(self.device)
                            try:
                                metrics = self.validate_bbox_head(
                                    fp32_out['boxes'],
                                    int8_out['boxes'],
                                    bbox_targets
                                )
                                all_metrics['bbox'].append(metrics)
                            except Exception as e:
                                print(f"Warning: Bbox validation failed: {e}")
                        
                        # Scene embedding head.
                        if 'scene_embedding' in fp32_out:
                            try:
                                metrics = self.validate_embedding_head(
                                    fp32_out['scene_embedding'],
                                    int8_out['scene_embedding']
                                )
                                all_metrics['embedding'].append(metrics)
                            except Exception as e:
                                print(f"Warning: Embedding validation failed: {e}")
                        
                        # Urgency head.
                        if 'urgency_scores' in fp32_out:
                            urgency_targets = targets.get('urgency') if isinstance(targets, dict) else None
                            if urgency_targets is not None and torch.is_tensor(urgency_targets):
                                urgency_targets = urgency_targets.to(self.device)
                            try:
                                metrics = self.validate_urgency_head(
                                    fp32_out['urgency_scores'],
                                    int8_out['urgency_scores'],
                                    urgency_targets
                                )
                                all_metrics['urgency'].append(metrics)
                            except Exception as e:
                                print(f"Warning: Urgency validation failed: {e}")
                        
                        # Objectness head.
                        if 'objectness' in fp32_out:
                            try:
                                metrics = self.validate_objectness_head(
                                    fp32_out['objectness'],
                                    int8_out['objectness']
                                )
                                all_metrics['objectness'].append(metrics)
                            except Exception as e:
                                print(f"Warning: Objectness validation failed: {e}")
                        
                        num_batches += 1
                        
                        # Progress indication.
                        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                            progress = (batch_idx + 1) / total_batches * 100
                            print(f"  Processed {batch_idx + 1}/{total_batches} batches ({progress:.1f}%)")
                
                except Exception as e:
                    import warnings
                    warnings.warn(f"Batch {batch_idx} failed: {e}")
                    continue  # Skip failed batch, don't crash entire validation.
        
        # Aggregate metrics.
        aggregated = {}
        for head_name, metrics_list in all_metrics.items():
            if not metrics_list:
                continue
            
            head_agg = {}
            
            # Average all numeric metrics.
            all_keys = set()
            for m in metrics_list:
                all_keys.update(m.keys())
            
            for key in all_keys:
                # Skip error fields (they're strings, not floats)
                if key.startswith('_') or key.endswith('_error'):
                    continue
                values = [
                    m[key] for m in metrics_list 
                    if key in m and isinstance(m[key], (int, float)) 
                    and not (np.isnan(m[key]) or np.isinf(m[key]))
                ]
                if values:
                    head_agg[f'{key}_mean'] = float(np.mean(values))
                    head_agg[f'{key}_std'] = float(np.std(values))
                    head_agg[f'{key}_min'] = float(np.min(values))
                    head_agg[f'{key}_max'] = float(np.max(values))
            
            aggregated[head_name] = head_agg
        
        aggregated['num_batches'] = num_batches
        
        print("\nValidation Complete")
        
        return aggregated


class ModelBenchmark:
    """Benchmark inference latency and memory usage."""
    
    @staticmethod
    def benchmark_latency(
        model: nn.Module,
        test_inputs: List[torch.Tensor],
        device: str = 'cpu',
        warmup_runs: int = 10,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Benchmark model inference latency."""
        model = model.eval().to(device)
        test_inputs = [x.to(device) for x in test_inputs]
        
        # Warmup.
        with torch.no_grad():
            for _ in range(warmup_runs):
                for x in test_inputs:
                    _ = model(x)
            # Synchronize GPU operations before benchmarking.
            if device.startswith('cuda') and torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Benchmark.
        timings = []
        with torch.no_grad():
            for _ in range(num_runs):
                # Synchronize before timing.
                if device.startswith('cuda') and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for x in test_inputs:
                    _ = model(x)
                # Synchronize after inference to ensure completion.
                if device.startswith('cuda') and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                timings.append((t1 - t0) / len(test_inputs) * 1000)  # Ms per input.
        
        timings = np.array(timings)
        
        mean_ms = float(np.mean(timings))
        return {
            'mean_ms': mean_ms,
            'std_ms': float(np.std(timings)),
            'median_ms': float(np.median(timings)),
            'p95_ms': float(np.percentile(timings, 95)),
            'p99_ms': float(np.percentile(timings, 99)),
            'min_ms': float(np.min(timings)),
            'max_ms': float(np.max(timings)),
            'throughput_fps': 1000.0 / mean_ms if mean_ms > 0 else 0.0
        }
    
    @staticmethod
    def compare_models(
        model_fp32: nn.Module,
        model_int8: nn.Module,
        test_inputs: List[torch.Tensor],
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """Compare FP32 vs INT8 model performance."""
        print("\nBenchmarking FP32 model...")
        fp32_bench = ModelBenchmark.benchmark_latency(model_fp32, test_inputs, device)
        
        print("Benchmarking INT8 model...")
        int8_bench = ModelBenchmark.benchmark_latency(model_int8, test_inputs, device)
        
        speedup = fp32_bench['mean_ms'] / int8_bench['mean_ms'] if int8_bench['mean_ms'] > 0 else 0.0
        
        return {
            'fp32': fp32_bench,
            'int8': int8_bench,
            'speedup': speedup,
            'latency_reduction_percent': (1 - int8_bench['mean_ms'] / fp32_bench['mean_ms']) * 100 if fp32_bench['mean_ms'] > 0 else 0.0
        }


# CLI usage.
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate and benchmark quantized models')
    parser.add_argument('--fp32-model', type=str, required=True,
                       help='Path to FP32 model state dict')
    parser.add_argument('--int8-model', type=str, required=True,
                       help='Path to INT8 model state dict')
    parser.add_argument('--model-file', type=str, required=True,
                       help='Python file with build_model() function')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Test data directory')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu/cuda)')
    parser.add_argument('--output-file', type=str, default='validation_results.json',
                       help='Output JSON file')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run latency benchmarks')
    
    args = parser.parse_args()
    
    # Load model architecture.
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_def", args.model_file)
    if spec is None:
        raise SystemExit(f"Could not load model file: {args.model_file}")
    if spec.loader is None:
        raise SystemExit(f"Model file has no loader: {args.model_file}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model_def"] = mod
    spec.loader.exec_module(mod)
    
    if not hasattr(mod, "build_model"):
        raise SystemExit("Model file must define build_model() function")
    
    model_fp32 = mod.build_model()
    model_int8 = mod.build_model()
    
    # Load weights.
    model_fp32.load_state_dict(torch.load(args.fp32_model, map_location='cpu'))
    model_int8.load_state_dict(torch.load(args.int8_model, map_location='cpu'))
    
    # Setup test dataloader (adapt to your dataset)
    from torch.utils.data import DataLoader, TensorDataset
    test_data = TensorDataset(
        torch.randn(50, 3, 224, 224),
        torch.randint(0, 10, (50,))
    )
    test_loader = DataLoader(test_data, batch_size=args.batch_size)
    
    # Run validation.
    validator = QuantizationValidator(model_fp32, model_int8, test_loader, args.device)
    validation_results = validator.run_full_validation()
    
    # Run benchmark if requested.
    if args.benchmark:
        test_inputs = [torch.randn(1, 3, 224, 224) for _ in range(10)]
        benchmark_results = ModelBenchmark.compare_models(
            model_fp32, model_int8, test_inputs, args.device
        )
        validation_results['benchmark'] = benchmark_results
    
    # Save results.
    output_path = Path(args.output_file)
    with open(output_path, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print summary.
    print("\nVALIDATION SUMMARY")
    
    for head, metrics in validation_results.items():
        if head == 'num_batches' or head == 'benchmark':
            continue
        print(f"\n{head.upper()} HEAD:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
    
    if 'benchmark' in validation_results:
        bench = validation_results['benchmark']
        print(f"\nBENCHMARK:")
        print(f"  FP32 latency: {bench['fp32']['mean_ms']:.2f} ms")
        print(f"  INT8 latency: {bench['int8']['mean_ms']:.2f} ms")
        print(f"  Speedup: {bench['speedup']:.2f}x")
        print(f"  Latency reduction: {bench['latency_reduction_percent']:.1f}%")







