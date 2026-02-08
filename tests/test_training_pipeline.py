"""Training Pipeline Tests for MaxSight Model Tests training infrastructure with dummy/synthetic data."""

import torch
import torch.nn as nn
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path.
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.maxsight_cnn import create_model, COCO_CLASSES
from ml.training.losses import (
    ObjectnessLoss, 
    ClassificationLoss, 
    BoxRegressionLoss,
    MultiHeadLoss
)
from ml.training.matching import match_predictions_to_gt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


class DetectionLoss(nn.Module):
    """Combined detection loss wrapper for testing."""
    
    def __init__(self, num_classes: int):
        super().__init__()
        self.objectness_loss = ObjectnessLoss()
        self.classification_loss = ClassificationLoss(num_classes)
        self.box_loss = BoxRegressionLoss()
        self.num_classes = num_classes
    
    def forward(self, predictions: dict, targets: dict) -> dict:
        """Compute combined detection losses."""
        losses = {}
        
        B = predictions['classifications'].shape[0]
        N = predictions['classifications'].shape[1]
        
        # Convert list targets to batched tensors. Labels: list of [num_objects] -> [B, N] (pad with -1 for no object)
        if 'labels' in targets:
            labels_list = targets['labels'] if isinstance(targets['labels'], list) else [targets['labels']]
            labels_batched = torch.full((B, N), -1, dtype=torch.long, device=predictions['classifications'].device)
            for b, labels in enumerate(labels_list):
                if len(labels) > 0:
                    num_objs = min(len(labels), N)
                    labels_batched[b, :num_objs] = labels[:num_objs]
        else:
            labels_batched = None
        
        # Boxes: list of [num_objects, 4] -> [B, N, 4] (pad with 0)
        if 'boxes' in targets:
            boxes_list = targets['boxes'] if isinstance(targets['boxes'], list) else [targets['boxes']]
            boxes_batched = torch.zeros((B, N, 4), device=predictions['boxes'].device)
            for b, boxes in enumerate(boxes_list):
                if len(boxes) > 0:
                    num_objs = min(len(boxes), N)
                    boxes_batched[b, :num_objs] = boxes[:num_objs]
        else:
            boxes_batched = None
        
        # Objectness: create from labels (1 where object exists, 0 elsewhere)
        if labels_batched is not None:
            objectness_targets = (labels_batched >= 0).float()
        else:
            objectness_targets = None
        
        # Objectness loss.
        if 'objectness' in predictions and objectness_targets is not None:
            losses['objectness'] = self.objectness_loss(
                predictions['objectness'], 
                objectness_targets
            )
        
        # Classification loss (only on valid locations)
        if 'classifications' in predictions and labels_batched is not None:
            # Mask out invalid locations.
            valid_mask = labels_batched >= 0
            if valid_mask.any():
                valid_preds = predictions['classifications'][valid_mask]  # [num_valid, num_classes].
                valid_labels = labels_batched[valid_mask]  # [num_valid].
                losses['classification'] = self.classification_loss(
                    valid_preds.unsqueeze(0),  # Add batch dim.
                    valid_labels.unsqueeze(0)
                )
            else:
                losses['classification'] = torch.tensor(0.0, device=predictions['classifications'].device)
        
        # Box regression loss (only on valid locations)
        if 'boxes' in predictions and boxes_batched is not None:
            # Mask out invalid locations.
            valid_mask = (labels_batched >= 0) if labels_batched is not None else torch.ones(B, N, dtype=torch.bool, device=predictions['boxes'].device)
            if valid_mask.any():
                valid_preds = predictions['boxes'][valid_mask]  # [num_valid, 4].
                valid_targets = boxes_batched[valid_mask]  # [num_valid, 4].
                losses['box'] = self.box_loss(
                    valid_preds.unsqueeze(0),  # Add batch dim.
                    valid_targets.unsqueeze(0)
                )
            else:
                losses['box'] = torch.tensor(0.0, device=predictions['boxes'].device)
        
        # Total loss.
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses


class DummyMaxSightDataset(Dataset):
    """Dummy dataset for training pipeline tests."""
    
    def __init__(self, num_samples: int = 10, image_size: tuple = (224, 224)):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = 80  # COCO classes.
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate dummy image.
        image = torch.randn(3, *self.image_size)
        
        # Generate dummy ground truth (normalized format: x, y, w, h)
        num_objects = int(torch.randint(1, 5, (1,)).item())
        
        boxes = torch.rand(num_objects, 4)  # Normalized [0, 1].
        boxes[:, 2:] = boxes[:, 2:] * 0.3 + 0.1  # Width/height between 0.1-0.4.
        boxes[:, :2] = boxes[:, :2] * 0.7  # Position in center 70% of image.
        
        labels = torch.randint(0, self.num_classes, (num_objects,))
        objectness = torch.ones(num_objects)
        
        # Dummy scene-level targets.
        urgency_scores = torch.randint(0, 4, (4,)).float()
        distance_zones = torch.randint(0, 3, (num_objects, 3)).float()
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'objectness': objectness,
            'urgency_scores': urgency_scores,
            'distance_zones': distance_zones,
        }


def test_training_step():
    """Test a single training step with dummy data."""
    print("Training Pipeline Test 1: Single Training Step")
    
    model = create_model()
    model.train()
    
    # Create dummy data.
    batch_size = 2
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass.
    outputs = model(dummy_image)
    
    # Create dummy ground truth (normalized format: x, y, w, h)
    gt_boxes = []
    gt_labels = []
    for _ in range(batch_size):
        num_objects = 3
        boxes = torch.rand(num_objects, 4)  # Normalized [0, 1].
        boxes[:, 2:] = boxes[:, 2:] * 0.3 + 0.1  # Width/height between 0.1-0.4.
        boxes[:, :2] = boxes[:, :2] * 0.7  # Position in center 70% of image.
        gt_boxes.append(boxes)
        gt_labels.append(torch.randint(0, 80, (num_objects,)))
    
    # Create loss function.
    detection_loss_fn = DetectionLoss(num_classes=len(COCO_CLASSES))
    
    # Prepare targets in correct format.
    targets = {
        'labels': gt_labels,
        'boxes': gt_boxes,
        'num_objects': torch.tensor([len(boxes) for boxes in gt_boxes])
    }
    
    # Compute loss.
    predictions = {
        'classifications': outputs['classifications'],
        'boxes': outputs['boxes'],
        'objectness': outputs['objectness']
    }
    
    loss_dict = detection_loss_fn(predictions, targets)
    total_loss = loss_dict['total_loss']
    
    # Backward pass.
    total_loss.backward()
    
    # Check gradients.
    has_gradients = any(p.grad is not None for p in model.parameters())
    assert has_gradients, "No gradients computed"
    
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Gradients computed: {has_gradients}")
    print("  PASSED: Training step works correctly")


def test_data_loader():
    """Test data loader with dummy dataset."""
    print("\nTraining Pipeline Test 2: Data Loader")
    
    dataset = DummyMaxSightDataset(num_samples=20)
    # Use collate_fn to handle variable-sized tensors.
    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        return {
            'image': images,
            'boxes': [item['boxes'] for item in batch],  # List, not stacked.
            'labels': [item['labels'] for item in batch],  # List, not stacked.
            'objectness': [item['objectness'] for item in batch],
            'urgency_scores': torch.stack([item['urgency_scores'] for item in batch]),
            'distance_zones': [item['distance_zones'] for item in batch],
        }
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    # Test loading a batch.
    batch = next(iter(dataloader))
    
    assert 'image' in batch, "Batch missing image"
    assert 'boxes' in batch, "Batch missing boxes"
    assert 'labels' in batch, "Batch missing labels"
    
    assert batch['image'].shape[0] == 4, "Batch size incorrect"
    assert batch['image'].shape[1] == 3, "Image channels incorrect"
    
    print(f"  Batch size: {batch['image'].shape[0]}")
    print(f"  Image shape: {batch['image'].shape}")
    print("  PASSED: Data loader works correctly")


def test_training_loop_iteration():
    """Test a complete training loop iteration."""
    print("\nTraining Pipeline Test 3: Training Loop Iteration")
    
    model = create_model()
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    dataset = DummyMaxSightDataset(num_samples=10)
    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        return {
            'image': images,
            'boxes': [item['boxes'] for item in batch],
            'labels': [item['labels'] for item in batch],
            'objectness': [item['objectness'] for item in batch],
            'urgency_scores': torch.stack([item['urgency_scores'] for item in batch]),
            'distance_zones': [item['distance_zones'] for item in batch],
        }
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    detection_loss_fn = DetectionLoss(num_classes=len(COCO_CLASSES))
    
    # One epoch.
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        images = batch['image']
        
        # Forward.
        outputs = model(images)
        
        # Prepare targets.
        targets = {
            'labels': batch['labels'],
            'boxes': batch['boxes'],
            'num_objects': torch.tensor([len(boxes) for boxes in batch['boxes']])
        }
        
        # Compute loss.
        predictions = {
            'classifications': outputs['classifications'],
            'boxes': outputs['boxes'],
            'objectness': outputs['objectness']
        }
        
        loss_dict = detection_loss_fn(predictions, targets)
        loss = loss_dict['total_loss']
        
        # Backward.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    print(f"  Processed {num_batches} batches")
    print(f"  Average loss: {avg_loss:.4f}")
    print("  PASSED: Training loop iteration works")


def test_gradient_accumulation():
    """Test gradient accumulation for larger effective batch sizes."""
    print("\nTraining Pipeline Test 4: Gradient Accumulation")
    
    model = create_model()
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    dataset = DummyMaxSightDataset(num_samples=8)
    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        return {
            'image': images,
            'boxes': [item['boxes'] for item in batch],
            'labels': [item['labels'] for item in batch],
            'objectness': [item['objectness'] for item in batch],
            'urgency_scores': torch.stack([item['urgency_scores'] for item in batch]),
            'distance_zones': [item['distance_zones'] for item in batch],
        }
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    detection_loss_fn = DetectionLoss(num_classes=len(COCO_CLASSES))
    accumulation_steps = 2
    
    # Training with gradient accumulation.
    optimizer.zero_grad()
    
    for i, batch in enumerate(dataloader):
        images = batch['image']
        outputs = model(images)
        
        # Prepare targets.
        targets = {
            'labels': batch['labels'],
            'boxes': batch['boxes'],
            'num_objects': torch.tensor([len(boxes) for boxes in batch['boxes']])
        }
        
        # Compute loss.
        predictions = {
            'classifications': outputs['classifications'],
            'boxes': outputs['boxes'],
            'objectness': outputs['objectness']
        }
        
        loss_dict = detection_loss_fn(predictions, targets)
        loss = loss_dict['total_loss'] / accumulation_steps  # Scale loss.
        
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    print(f"  Gradient accumulation steps: {accumulation_steps}")
    print("  PASSED: Gradient accumulation works")


def test_fp32_training():
    """Test FP32 training (CUDA if available)."""
    print("\nTraining Pipeline Test 5: FP32 Training")
    
    if not torch.cuda.is_available():
        print("  SKIPPED: CUDA not available")
        return
    
    model = create_model()
    model = model.cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dummy_image = torch.randn(2, 3, 224, 224).cuda()
    optimizer.zero_grad()
    outputs = model(dummy_image)
    loss = outputs['classifications'].sum()
    loss.backward()
    optimizer.step()
    print("  PASSED: FP32 training works")


if __name__ == "__main__":
    print("Running Training Pipeline Tests")
    print("=" * 50)
    
    test_training_step()
    test_data_loader()
    test_training_loop_iteration()
    test_gradient_accumulation()
    test_fp32_training()
    
    print("\n" + "=" * 50)
    print("All training pipeline tests passed!")







