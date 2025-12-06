

# Evaluate status


from pathlib import Path
from typing import Dict, Optional, Any, List
import json
from ml.training.scene_metrics import SceneMetrics
from ml.training.benchmark import benchmark_inference


def generate_evaluation_report(metrics: Dict[str, float], save_path: Optional[Path] = None) -> str:
    """Generate text report with overall and lighting-stratified metrics."""
    report = []
    
    report.append("MaxSight Model Evaluation Report")
    
    report.append("Overall Performance Metrics:")
    report.append(f"  Loss:      {metrics.get('loss', 0.0):.4f}")
    report.append(f"  Accuracy:  {metrics.get('accuracy', 0.0):.2f}%")
    report.append(f"  Precision: {metrics.get('precision', 0.0):.4f}")
    report.append(f"  Recall:    {metrics.get('recall', 0.0):.4f}")
    report.append(f"  F1 Score:  {metrics.get('f1', 0.0):.4f}")
    report.append(f"  mAP@0.5:   {metrics.get('map', 0.0):.4f}")
    
    # Scene-level metrics
    if 'urgency_accuracy' in metrics:
        report.append(f"  Urgency Accuracy: {metrics.get('urgency_accuracy', 0.0):.4f}")
    if 'distance_accuracy' in metrics:
        report.append(f"  Distance Accuracy: {metrics.get('distance_accuracy', 0.0):.4f}")
    
    # Inference latency (integrated with benchmark results)
    if 'inference_latency_ms' in metrics:
        report.append(f"  Inference Latency: {metrics.get('inference_latency_ms', 0.0):.1f} ms")
    elif 'mean_latency_ms' in metrics:
        report.append(f"  Inference Latency: {metrics.get('mean_latency_ms', 0.0):.1f} ms")
        if 'p95_latency_ms' in metrics:
            report.append(f"  P95 Latency: {metrics.get('p95_latency_ms', 0.0):.1f} ms")
        if 'p99_latency_ms' in metrics:
            report.append(f"  P99 Latency: {metrics.get('p99_latency_ms', 0.0):.1f} ms")
    
    
    report.append("Lighting Condition Performance:")
    
    # Auto-detect lighting conditions from metrics keys
    default_lighting_conditions = ['bright', 'normal', 'dim', 'dark']
    available_lightings = [l for l in default_lighting_conditions if any(f'{l}_{m}' in metrics for m in ['precision', 'recall', 'f1'])]
    if not available_lightings:
        available_lightings = default_lighting_conditions  # Fallback to default
    
    for lighting in available_lightings:
        p = metrics.get(f'{lighting}_precision', 0.0)
        r = metrics.get(f'{lighting}_recall', 0.0)
        f = metrics.get(f'{lighting}_f1', 0.0)
        
        report.append(f"  {lighting.capitalize()}:")
        report.append(f"    Precision: {p:.4f}")
        report.append(f"    Recall:    {r:.4f}")
        report.append(f"    F1 Score:  {f:.4f}")
        report.append("")
    
    report.append("Performance Degradation Analysis:")
    
    normal_recall = metrics.get('normal_recall', 0.0)
    normal_precision = metrics.get('normal_precision', 0.0)
    normal_f1 = metrics.get('normal_f1', 0.0)
    
    if normal_recall > 0:
        for lighting in ['bright', 'dim', 'dark']:
            lighting_recall = metrics.get(f'{lighting}_recall', 0.0)
            lighting_precision = metrics.get(f'{lighting}_precision', 0.0)
            lighting_f1 = metrics.get(f'{lighting}_f1', 0.0)
            
            recall_degradation = ((normal_recall - lighting_recall) / normal_recall) * 100 if normal_recall > 0 else 0.0
            precision_degradation = ((normal_precision - lighting_precision) / normal_precision) * 100 if normal_precision > 0 else 0.0
            f1_degradation = ((normal_f1 - lighting_f1) / normal_f1) * 100 if normal_f1 > 0 else 0.0
            
            report.append(f"  {lighting.capitalize()} vs Normal:")
            report.append(f"    Recall:    {recall_degradation:+.2f}%")
            report.append(f"    Precision: {precision_degradation:+.2f}%")
            report.append(f"    F1:        {f1_degradation:+.2f}%")
            
            if recall_degradation > 10.0:
                report.append(f"    WARNING: Significant degradation in {lighting} lighting")
            report.append("")
    
    report.append("Summary:")
    report.append("-" * 70)
    
    overall_recall = metrics.get('recall', 0.0)
    dark_recall = metrics.get('dark_recall', 0.0)
    
    if overall_recall >= 0.85:
        report.append("  Overall recall: meets target (≥85%)")
    else:
        report.append(f"  Overall recall: {overall_recall:.2%} (target: ≥85%)")
    
    if dark_recall >= 0.75:
        report.append("  Dark lighting recall: meets target (≥75%)")
    else:
        report.append(f"  Dark lighting recall: {dark_recall:.2%} (target: ≥75%)")
    
    if normal_recall > 0:
        dark_degradation = ((normal_recall - dark_recall) / normal_recall) * 100
        if dark_degradation <= 10.0:
            report.append("  Dark degradation: <10% (meets requirement)")
        else:
            report.append(f"  Dark degradation: {dark_degradation:.2f}% (target: <10%)")
    
    report.append("")
    report.append("=" * 70)
    
    report_str = "\n".join(report)
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(report_str)
    
    return report_str


def plot_lighting_metrics(metrics: Dict[str, float], save_path: Optional[Path] = None) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return
    
    lightings = ['bright', 'normal', 'dim', 'dark']
    precisions = [metrics.get(f'{l}_precision', 0.0) for l in lightings]
    recalls = [metrics.get(f'{l}_recall', 0.0) for l in lightings]
    f1_scores = [metrics.get(f'{l}_f1', 0.0) for l in lightings]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = range(len(lightings))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], precisions, width, label='Precision', color='#3498db', alpha=0.8)
    ax1.bar([i + width/2 for i in x], recalls, width, label='Recall', color='#e74c3c', alpha=0.8)
    ax1.set_xlabel('Lighting Condition')
    ax1.set_ylabel('Score')
    ax1.set_title('Precision and Recall Across Lighting Conditions')
    ax1.set_xticks(x)
    ax1.set_xticklabels([l.capitalize() for l in lightings])
    ax1.legend()
    ax1.set_ylim([0, 1.05])
    ax1.grid(axis='y', alpha=0.3)
    
    for i, (p, r) in enumerate(zip(precisions, recalls)):
        ax1.text(i - width/2, p + 0.02, f'{p:.2f}', ha='center', va='bottom', fontsize=9)
        ax1.text(i + width/2, r + 0.02, f'{r:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax2.bar(x, f1_scores, width=0.6, color='#2ecc71', alpha=0.8)
    ax2.set_xlabel('Lighting Condition')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score Across Lighting Conditions')
    ax2.set_xticks(x)
    ax2.set_xticklabels([l.capitalize() for l in lightings])
    ax2.set_ylim([0, 1.05])
    ax2.grid(axis='y', alpha=0.3)
    
    for i, f1 in enumerate(f1_scores):
        ax2.text(i, f1 + 0.02, f'{f1:.2f}', ha='center', va='bottom', fontsize=9)
    
    fig.suptitle('MaxSight Model Performance Across Lighting Conditions', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close()


def analyze_lighting_degradation(metrics: Dict[str, float], lighting_conditions: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    if lighting_conditions is None:
        lighting_conditions = ['bright', 'dim', 'dark']
    
    normal_metrics = {
        'precision': metrics.get('normal_precision', 0.0),
        'recall': metrics.get('normal_recall', 0.0),
        'f1': metrics.get('normal_f1', 0.0)
    }
    
    degradations = {}
    for lighting in lighting_conditions:
        degradations[lighting] = {}
        for metric in ['precision', 'recall', 'f1']:
            lighting_metric = metrics.get(f'{lighting}_{metric}', 0.0)
            normal_metric = normal_metrics[metric]
            if normal_metric > 0:
                degradation = ((normal_metric - lighting_metric) / normal_metric) * 100
            else:
                degradation = 0.0
            degradations[lighting][metric] = degradation
    
    return degradations


def export_metrics_json(metrics: Dict[str, float], save_path: Path) -> None:
    """Export metrics to JSON. Handles torch tensors."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    json_metrics = {}
    for key, value in metrics.items():
        if hasattr(value, 'item'):
            json_metrics[key] = float(value.item())
        else:
            json_metrics[key] = float(value) if isinstance(value, (int, float)) else str(value)
    
    with open(save_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    
    print(f"Metrics exported to JSON: {save_path}")

