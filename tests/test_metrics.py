"""Tests for Metrics Computation - Task 4.1"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
import time
import numpy as np
from typing import Dict, List

from ml.training.metrics import DetectionMetrics, compute_iou_matrix
from ml.training.scene_metrics import SceneMetrics


def test_detection_metrics_initialization():
    """Test DetectionMetrics initialization."""
    print("\nMetrics Test 1: DetectionMetrics Initialization")
    
    num_classes = 80
    metrics = DetectionMetrics(num_classes=num_classes)
    
    assert metrics.num_classes == num_classes, "num_classes should match"
    assert len(metrics.class_tp) == num_classes, "class_tp should have num_classes elements"
    assert len(metrics.class_fp) == num_classes, "class_fp should have num_classes elements"
    assert len(metrics.class_fn) == num_classes, "class_fn should have num_classes elements"
    
    print("  DetectionMetrics initialization: PASSED")


def test_iou_computation():
    """Test IoU matrix computation."""
    print("\nMetrics Test 2: IoU Computation")
    
    # Create test boxes (normalized center format: [cx, cy, w, h])
    pred_boxes = torch.tensor([
        [0.5, 0.5, 0.2, 0.3],  # Box 1
        [0.7, 0.7, 0.15, 0.2]  # Box 2
    ])
    
    gt_boxes = torch.tensor([
        [0.5, 0.5, 0.2, 0.3],  # Matches Box 1
        [0.8, 0.8, 0.1, 0.15]  # Different box
    ])
    
    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
    
    assert iou_matrix.shape == (2, 2), "IoU matrix should be [num_pred, num_gt]"
    assert iou_matrix[0, 0] > 0.9, "Perfect match should have IoU > 0.9"
    assert iou_matrix[0, 1] < 0.5, "Non-overlapping boxes should have low IoU"
    
    print("  IoU computation: PASSED")


def test_detection_metrics_update():
    """Test DetectionMetrics update functionality."""
    print("\nMetrics Test 3: DetectionMetrics Update")
    
    num_classes = 80
    metrics = DetectionMetrics(num_classes=num_classes)
    
    # Create test data
    pred_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
    pred_labels = torch.tensor([0])  # person class
    pred_scores = torch.tensor([0.95])
    
    gt_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
    gt_labels = torch.tensor([0])  # person class
    
    # Update metrics
    metrics.update(
        pred_boxes=pred_boxes,
        pred_labels=pred_labels,
        pred_scores=pred_scores,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        iou_threshold=0.5
    )
    
    # Check that TP was recorded
    assert metrics.class_tp[0] > 0, "True positive should be recorded"
    assert metrics.class_fp[0] == 0, "False positive should not be recorded for perfect match"
    
    print("  DetectionMetrics update: PASSED")


def test_detection_metrics_precision_recall():
    """Test precision and recall computation."""
    print("\nMetrics Test 4: Precision and Recall")
    
    num_classes = 80
    metrics = DetectionMetrics(num_classes=num_classes)
    
    # Add some predictions and ground truth
    pred_boxes = torch.tensor([
        [0.5, 0.5, 0.2, 0.3],
        [0.7, 0.7, 0.15, 0.2]
    ])
    pred_labels = torch.tensor([0, 1])
    pred_scores = torch.tensor([0.95, 0.85])
    
    gt_boxes = torch.tensor([
        [0.5, 0.5, 0.2, 0.3],
        [0.8, 0.8, 0.1, 0.15]
    ])
    gt_labels = torch.tensor([0, 2])
    
    metrics.update(
        pred_boxes=pred_boxes,
        pred_labels=pred_labels,
        pred_scores=pred_scores,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        iou_threshold=0.5
    )
    
    # Get metrics
    overall_metrics = metrics.get_overall_metrics()
    
    assert 'precision' in overall_metrics, "Should have precision"
    assert 'recall' in overall_metrics, "Should have recall"
    assert 'f1' in overall_metrics, "Should have F1 score"
    
    assert 0 <= overall_metrics['precision'] <= 1, "Precision should be 0-1"
    assert 0 <= overall_metrics['recall'] <= 1, "Recall should be 0-1"
    assert 0 <= overall_metrics['f1'] <= 1, "F1 should be 0-1"
    
    print(f"  Precision: {overall_metrics['precision']:.3f}")
    print(f"  Recall: {overall_metrics['recall']:.3f}")
    print(f"  F1: {overall_metrics['f1']:.3f}")
    print("  Precision and recall: PASSED")


def test_detection_metrics_map():
    """Test mAP (mean Average Precision) computation."""
    print("\nMetrics Test 5: mAP Computation")
    
    num_classes = 80
    metrics = DetectionMetrics(num_classes=num_classes)
    
    # Add multiple predictions
    for i in range(5):
        pred_boxes = torch.tensor([[0.5 + i*0.1, 0.5, 0.2, 0.3]])
        pred_labels = torch.tensor([0])
        pred_scores = torch.tensor([0.9 - i*0.1])
        
        gt_boxes = torch.tensor([[0.5 + i*0.1, 0.5, 0.2, 0.3]])
        gt_labels = torch.tensor([0])
        
        metrics.update(
            pred_boxes=pred_boxes,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            iou_threshold=0.5
        )
    
    # Get mAP
    map_score = metrics.get_map()
    
    assert 0 <= map_score <= 1, "mAP should be 0-1"
    assert map_score > 0, "mAP should be positive for good predictions"
    
    print(f"  mAP: {map_score:.3f}")
    print("  mAP computation: PASSED")


def test_scene_metrics_initialization():
    """Test SceneMetrics initialization."""
    print("\nMetrics Test 6: SceneMetrics Initialization")
    
    metrics = SceneMetrics(num_urgency_levels=4, num_distance_zones=3)
    
    assert metrics.num_urgency_levels == 4, "num_urgency_levels should match"
    assert metrics.num_distance_zones == 3, "num_distance_zones should match"
    assert metrics.urgency_total == 0, "Should start with zero total"
    assert metrics.distance_total == 0, "Should start with zero total"
    
    print("  SceneMetrics initialization: PASSED")


def test_scene_metrics_urgency():
    """Test urgency prediction metrics."""
    print("\nMetrics Test 7: Urgency Metrics")
    
    metrics = SceneMetrics(num_urgency_levels=4, num_distance_zones=3)
    
    # Test urgency updates
    pred_urgency = torch.tensor([0, 1, 2, 3])
    gt_urgency = torch.tensor([0, 1, 2, 2])  # Last one is wrong
    
    metrics.update_urgency(pred_urgency, gt_urgency)
    
    # Check accuracy
    urgency_accuracy = metrics.get_urgency_accuracy()
    
    assert 0 <= urgency_accuracy <= 1, "Urgency accuracy should be 0-1"
    assert urgency_accuracy > 0, "Should have some correct predictions"
    
    print(f"  Urgency accuracy: {urgency_accuracy:.3f}")
    print("  Urgency metrics: PASSED")


def test_scene_metrics_distance():
    """Test distance zone metrics."""
    print("\nMetrics Test 8: Distance Metrics")
    
    metrics = SceneMetrics(num_urgency_levels=4, num_distance_zones=3)
    
    # Test distance updates
    pred_distance = torch.tensor([0, 1, 2, 0])
    gt_distance = torch.tensor([0, 1, 2, 1])  # Last one is wrong
    
    metrics.update_distance(pred_distance, gt_distance)
    
    # Check accuracy
    distance_accuracy = metrics.get_distance_accuracy()
    
    assert 0 <= distance_accuracy <= 1, "Distance accuracy should be 0-1"
    assert distance_accuracy > 0, "Should have some correct predictions"
    
    print(f"  Distance accuracy: {distance_accuracy:.3f}")
    print("  Distance metrics: PASSED")


def test_latency_measurement():
    """Test latency measurement in metrics."""
    print("\nMetrics Test 9: Latency Measurement")
    
    num_classes = 80
    metrics = DetectionMetrics(num_classes=num_classes)
    
    # Simulate inference times
    for _ in range(10):
        inference_time = np.random.uniform(0.1, 0.5)  # 100-500ms
        metrics.record_inference_time(inference_time)
    
    # Get latency stats
    latency_stats = metrics.get_latency_stats()
    
    assert 'mean_ms' in latency_stats, "Should have mean latency"
    assert 'median_ms' in latency_stats, "Should have median latency"
    assert 'p95_ms' in latency_stats, "Should have P95 latency"
    assert 'p99_ms' in latency_stats, "Should have P99 latency"
    
    assert latency_stats['mean_ms'] > 0, "Mean latency should be positive"
    assert latency_stats['p95_ms'] >= latency_stats['mean_ms'], "P95 should be >= mean"
    assert latency_stats['p99_ms'] >= latency_stats['p95_ms'], "P99 should be >= P95"
    
    print(f"  Mean latency: {latency_stats['mean_ms']:.2f} ms")
    print(f"  P95 latency: {latency_stats['p95_ms']:.2f} ms")
    print("  Latency measurement: PASSED")


def test_metrics_reset():
    """Test metrics reset functionality."""
    print("\nMetrics Test 10: Metrics Reset")
    
    num_classes = 80
    metrics = DetectionMetrics(num_classes=num_classes)
    
    # Add some data
    pred_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
    pred_labels = torch.tensor([0])
    pred_scores = torch.tensor([0.95])
    gt_boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
    gt_labels = torch.tensor([0])
    
    metrics.update(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)
    
    # Check data was added
    assert metrics.class_tp[0] > 0, "Should have recorded TP"
    
    # Reset
    metrics.reset()
    
    # Check data was cleared
    assert metrics.class_tp[0] == 0, "TP should be reset to 0"
    assert len(metrics.inference_times) == 0, "Inference times should be cleared"
    
    print("  Metrics reset: PASSED")


if __name__ == "__main__":
    print("Running Metrics Tests")
    print("=" * 70)
    
    test_detection_metrics_initialization()
    test_iou_computation()
    test_detection_metrics_update()
    test_detection_metrics_precision_recall()
    test_detection_metrics_map()
    test_scene_metrics_initialization()
    test_scene_metrics_urgency()
    test_scene_metrics_distance()
    test_latency_measurement()
    test_metrics_reset()
    
    print("\n" + "=" * 70)
    print("All metrics tests completed!")

