#Evaluation report generator
from pathlib import Path
from typing import Dict, Optional
import matplotlib
import matplotlib.pyplot as plt
import json


def generate_evaluation_report(
    metrics: Dict[str, float],
    save_path: Optional[Path] = None
) -> str:
    report = []
    
    # Header section
    report.append("MaxSight Model Evaluation Report")
    report.append("Comprehensive Performance Analysis with Lighting-Aware Metrics")
    report.append("")
    
    # Overall performance metrics section
    report.append("Overall Performance Metrics:")
    report.append(f" Loss: {metrics.get('loss', 0.0):.4f}")
    report.append(f" Accuracy: {metrics.get('accuracy', 0.0):.2f}%")
    report.append(f" Precision: {metrics.get('precision', 0.0):.4f}  (TP / (TP + FP))")
    report.append(f" Recall: {metrics.get('recall', 0.0):.4f}  (TP / (TP + FN))")
    report.append(f" F1 Score: {metrics.get('f1', 0.0):.4f}  (Harmonic mean of P & R)")
    report.append(f" mAP@0.5: {metrics.get('map', 0.0):.4f}  (Mean Average Precision)")
    report.append("")
    
    # Lighting condition performance breakdown
    # Complexity: O(N)
    report.append("Lighting Condition Performance Breakdown:")
    report.append("-" * 70)
    
    lighting_conditions = ['bright', 'normal', 'dim', 'dark']
    for lighting in lighting_conditions:
        # Extract metrics for this lighting condition
        p = metrics.get(f'{lighting}_precision', 0.0)
        r = metrics.get(f'{lighting}_recall', 0.0)
        f = metrics.get(f'{lighting}_f1', 0.0)
        
        lighting_name = lighting.capitalize()
        
        # Add metrics to report
        report.append(f" {lighting_name} Lighting:")
        report.append(f" Precision: {p:.4f}  (Fraction of predictions that are correct)")
        report.append(f" Recall:    {r:.4f}  (Fraction of objects detected)")
        report.append(f" F1 Score:  {f:.4f}  (Balanced metric)")
        report.append("")
    
    # Performance degradation analysis
    report.append("Performance Degradation Analysis:")
    report.append("-" * 70)
    
    # Compare each lighting condition to normal (baseline)
    # Complexity: O(N)
    normal_recall = metrics.get('normal_recall', 0.0)
    normal_precision = metrics.get('normal_precision', 0.0)
    normal_f1 = metrics.get('normal_f1', 0.0)
    
    if normal_recall > 0:
        # Calculate degradation for each lighting condition
        for lighting in ['bright', 'dim', 'dark']:
            lighting_recall = metrics.get(f'{lighting}_recall', 0.0)
            lighting_precision = metrics.get(f'{lighting}_precision', 0.0)
            lighting_f1 = metrics.get(f'{lighting}_f1', 0.0)
            
            # Calculate percentage degradation (positive = worse, negative = better)
            recall_degradation = ((normal_recall - lighting_recall) / normal_recall) * 100 if normal_recall > 0 else 0.0
            precision_degradation = ((normal_precision - lighting_precision) / normal_precision) * 100 if normal_precision > 0 else 0.0
            f1_degradation = ((normal_f1 - lighting_f1) / normal_f1) * 100 if normal_f1 > 0 else 0.0
            
            # Format degradation report
            lighting_name = lighting.capitalize()
            report.append(f" {lighting_name} vs Normal:")
            report.append(f" Recall degradation: {recall_degradation:+.2f}%")
            report.append(f" Precision degradation: {precision_degradation:+.2f}%")
            report.append(f" F1 degradation: {f1_degradation:+.2f}%")
            
            # Warning if degradation is significant (>10%)
            if recall_degradation > 10.0:
                report.append(f"WARNING: Significant recall degradation in {lighting} lighting!")
                report.append(f"Model may struggle to detect objects in {lighting} conditions.")
            elif recall_degradation < -5.0:
                report.append(f"Model performs better in {lighting} lighting than normal!")
            else:
                report.append(f"Performance degradation <10% - model robust to {lighting} lighting")
            report.append("")
    
    
    report.append("Summary and Recommendations:")
    
    overall_recall = metrics.get('recall', 0.0)
    dark_recall = metrics.get('dark_recall', 0.0)
    
    if overall_recall >= 0.85:
        report.append("Overall recall meets target (≥85%)")
    else:
        report.append(f"Overall recall below target: {overall_recall:.2%} (target: ≥85%)")
        report.append("Train longer or adjust loss weights")
    
    if dark_recall >= 0.75:
        report.append("Dark lighting recall meets target (≥75%)")
    else:
        report.append(f" Dark lighting recall below target: {dark_recall:.2%} (target: ≥75%)")
        report.append("Add more dark lighting training data or augmentation")
    
    # Check degradation threshold (Task 4.2 requirement from the Software Development Checklist: <10% degradation)
    if normal_recall > 0:
        dark_degradation = ((normal_recall - dark_recall) / normal_recall) * 100
        if dark_degradation <= 10.0:
            report.append("Performance degradation in dark lighting <10% (meets requirement)")
        else:
            report.append(f"Performance degradation in dark lighting: {dark_degradation:.2f}% (target: <10%)")
            report.append("Improve low-light robustness with more dark training data")
    
    report.append("")
    
    report_str = "\n".join(report)
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True) 
        save_path.write_text(report_str)
    
    return report_str


def plot_lighting_metrics(
    metrics: Dict[str, float],
    save_path: Optional[Path] = None
) -> None:
    
    try:
        matplotlib.use('Agg')  # Non-interactive backend for server environments
    except ImportError:
        print("Warning: matplotlib not installed. Skipping plot generation.")
        print("Install with: pip install matplotlib")
        return
    
    # Extract lighting conditions and metrics
    # Complexity: O(N) where L=4 - processes each lighting condition
    lightings = ['bright', 'normal', 'dim', 'dark']
    precisions = [metrics.get(f'{l}_precision', 0.0) for l in lightings]
    recalls = [metrics.get(f'{l}_recall', 0.0) for l in lightings]
    f1_scores = [metrics.get(f'{l}_f1', 0.0) for l in lightings]
    
    # Create figure with subplots
    # Complexity: O(1) - matplotlib figure creation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Precision and Recall comparison
    # Complexity: O(1) - matplotlib plotting
    x = range(len(lightings))
    width = 0.35  # Bar width
    
    ax1.bar([i - width/2 for i in x], precisions, width, label='Precision', color='#3498db', alpha=0.8)
    ax1.bar([i + width/2 for i in x], recalls, width, label='Recall', color='#e74c3c', alpha=0.8)
    
    ax1.set_xlabel('Lighting Condition', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Precision and Recall Across Lighting Conditions', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([l.capitalize() for l in lightings])
    ax1.legend(loc='upper right')
    ax1.set_ylim([0, 1.05])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    # Complexity: O(N) - adds text to each bar
    for i, (p, r) in enumerate(zip(precisions, recalls)):
        ax1.text(i - width/2, p + 0.02, f'{p:.2f}', ha='center', va='bottom', fontsize=9)
        ax1.text(i + width/2, r + 0.02, f'{r:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: F1 Score comparison
    # Complexity: O(1) - matplotlib plotting
    ax2.bar(x, f1_scores, width=0.6, color="#00ff6a", alpha=0.8)
    
    ax2.set_xlabel('Lighting Condition', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('F1 Score Across Lighting Conditions', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([l.capitalize() for l in lightings])
    ax2.set_ylim([0, 1.05])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    # Complexity: O(N) - adds text to each bar
    for i, f1 in enumerate(f1_scores):
        ax2.text(i, f1 + 0.02, f'{f1:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add overall title
    # Complexity: O(1) - matplotlib operation
    fig.suptitle('MaxSight Model Performance Across Lighting Conditions', fontsize=16, fontweight='bold', y=1.02)
    
    # Adjust layout to prevent overlap issues
    # Complexity: O(1) - matplotlib layout adjustment
    plt.tight_layout()
    
    # Save plot if path provided
    # Complexity: O(1) - file I/O
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    # Close figure to free memory
    # Complexity: O(1) - matplotlib cleanup
    plt.close()


def analyze_lighting_degradation(metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    # Get normal (baseline) metrics
    # Complexity: O(1) - dictionary access
    normal_metrics = {
        'precision': metrics.get('normal_precision', 0.0),
        'recall': metrics.get('normal_recall', 0.0),
        'f1': metrics.get('normal_f1', 0.0)
    }
    
    # Calculate degradation for each lighting condition (compared to normal)
    # Complexity: O(N*M) where (bright, dim, dark), M=3 metrics
    degradations = {}
    for lighting in ['bright', 'dim', 'dark']:
        degradations[lighting] = {}
        
        # Calculate degradation for each metric
        # Complexity: O(M) where M=3 metrics
        for metric in ['precision', 'recall', 'f1']:
            # Complexity: O(1) - dictionary access
            lighting_metric = metrics.get(f'{lighting}_{metric}', 0.0)
            normal_metric = normal_metrics[metric]
            
            if normal_metric > 0:
                degradation = ((normal_metric - lighting_metric) / normal_metric) * 100
            else:
                degradation = 0.0  # Can't calculate if normal is 0
            
            degradations[lighting][metric] = degradation
    
    return degradations


def export_metrics_json(
    metrics: Dict[str, float],
    save_path: Path
) -> None:
    #JSON file export of metrics
    # Complexity: O(1) - directory creation
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert metrics to JSON-serializable format (handle any torch tensors)
    json_metrics = {}
    for key, value in metrics.items():
        # Convert torch tensors to Python floats/ints
        if hasattr(value, 'item'):
            json_metrics[key] = float(value.item())
        else:
            json_metrics[key] = float(value) if isinstance(value, (int, float)) else str(value)
    
    # Write JSON file
    with open(save_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    
    print(f"Metrics exported to JSON: {save_path}")

