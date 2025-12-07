"""
Training Pipeline Tests for MaxSight Model
Tests training infrastructure, loss computation, and optimization.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.models.maxsight_cnn import create_model
from ml.training.train_production import ProductionTrainer, create_dummy_dataloaders
from ml.training.losses import MaxSightLoss


def test_training_step():
    """Test a single training step."""
    print("Training Test 1: Single Training Step")
    
    model = create_model(num_classes=80)
    device = torch.device('cpu')
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = MaxSightLoss(num_classes=80).to(device)
    
    # Create dummy batch
    batch_size = 2
    dummy_image = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Create dummy ground truth
    num_objects = [3, 2]  # Different number of objects per image
    max_objects = max(num_objects)
    
    gt_boxes = torch.zeros(batch_size, max_objects, 4).to(device)
    gt_labels = torch.zeros(batch_size, max_objects, dtype=torch.long).to(device)
    num_objects_tensor = torch.tensor(num_objects, dtype=torch.long).to(device)
    
    # Fill with dummy data
    for i in range(batch_size):
        for j in range(num_objects[i]):
            gt_boxes[i, j] = torch.tensor([0.1, 0.1, 0.3, 0.3])
            gt_labels[i, j] = torch.randint(1, 80, (1,)).item()
    
    targets = {
        'boxes': gt_boxes,
        'labels': gt_labels,
        'num_objects': num_objects_tensor,
        'urgency': torch.zeros(batch_size, 4).to(device),
        'distance': torch.zeros(batch_size, max_objects, 3).to(device)
    }
    
    # Training step
    optimizer.zero_grad()
    outputs = model(dummy_image)
    losses = loss_fn(outputs, targets)
    total_loss = losses['total_loss']
    
    total_loss.backward()
    optimizer.step()
    
    # Verify gradients computed
    has_gradients = any(p.grad is not None for p in model.parameters())
    assert has_gradients, "No gradients computed"
    assert total_loss.item() > 0, "Loss should be positive"
    
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Gradients computed: {has_gradients}")
    print("  PASSED: Training step works correctly")


def test_data_loader():
    """Test data loader functionality."""
    print("\nTraining Test 2: Data Loader")
    
    train_loader, val_loader = create_dummy_dataloaders(
        num_train=10,
        num_val=5,
        batch_size=2
    )
    
    # Test train loader
    assert len(train_loader) > 0, "Train loader should have batches"
    
    sample_batch = next(iter(train_loader))
    assert 'images' in sample_batch, "Batch should contain images"
    assert 'labels' in sample_batch, "Batch should contain labels"
    assert 'boxes' in sample_batch, "Batch should contain boxes"
    
    assert sample_batch['images'].shape[0] == 2, "Batch size should be 2"
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Batch shape: {sample_batch['images'].shape}")
    
    # Test val loader
    assert len(val_loader) > 0, "Val loader should have batches"
    val_batch = next(iter(val_loader))
    assert val_batch['images'].shape[0] == 2, "Val batch size should be 2"
    print(f"  Val batches: {len(val_loader)}")
    
    print("  PASSED: Data loaders work correctly")


def test_training_loop_iteration():
    """Test a single training loop iteration."""
    print("\nTraining Test 3: Training Loop Iteration")
    
    model = create_model()
    train_loader, val_loader = create_dummy_dataloaders(
        num_train=4,
        num_val=2,
        batch_size=2
    )
    
    trainer = ProductionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cpu',
        num_epochs=1
    )
    
    # Run one iteration
    sample_batch = next(iter(train_loader))
    images = sample_batch['images']
    targets = {
        'labels': sample_batch['labels'],
        'boxes': sample_batch['boxes'],
        'urgency': sample_batch['urgency'],
        'distance': sample_batch['distance'],
        'num_objects': sample_batch['num_objects']
    }
    
    model.train()
    outputs = model(images)
    losses = trainer.criterion(outputs, targets)
    
    assert 'total_loss' in losses, "Should have total_loss"
    assert losses['total_loss'].item() > 0, "Loss should be positive"
    
    print(f"  Loss: {losses['total_loss'].item():.4f}")
    print("  PASSED: Training loop iteration works")


def test_loss_computation():
    """Test loss function computation."""
    print("\nTraining Test 4: Loss Computation")
    
    model = create_model()
    model.eval()
    loss_fn = MaxSightLoss(num_classes=80)
    
    batch_size = 2
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        outputs = model(dummy_image)
    
    # Create dummy targets
    num_objects = [2, 3]
    max_objects = max(num_objects)
    
    targets = {
        'boxes': torch.zeros(batch_size, max_objects, 4),
        'labels': torch.zeros(batch_size, max_objects, dtype=torch.long),
        'num_objects': torch.tensor(num_objects, dtype=torch.long),
        'urgency': torch.zeros(batch_size, 4),
        'distance': torch.zeros(batch_size, max_objects, 3)
    }
    
    losses = loss_fn(outputs, targets)
    
    assert 'total_loss' in losses, "Should have total_loss"
    assert 'classification_loss' in losses or 'detection_loss' in losses, "Should have component losses"
    assert losses['total_loss'].item() >= 0, "Loss should be non-negative"
    
    print(f"  Total loss: {losses['total_loss'].item():.4f}")
    print("  PASSED: Loss computation works correctly")


if __name__ == "__main__":
    test_training_step()
    test_data_loader()
    test_training_loop_iteration()
    test_loss_computation()
    
    print("\nAll training pipeline tests completed!")

