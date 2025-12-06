"""
Production-grade training loop for MaxSight CNN.

This is the clean, minimal training loop that matches the pseudo-code specification.
Supports multi-head loss, mixed precision, and proper checkpointing.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional, Any
from pathlib import Path
import json
import time
from copy import deepcopy

try:
    from torch.amp import autocast
    from torch.cuda.amp import GradScaler
    AMP_AVAILABLE = True
except ImportError:
    class DummyAutocast:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    autocast = DummyAutocast
    GradScaler = None
    AMP_AVAILABLE = False


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_targets_to_device(targets: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    """Move all tensor targets to device."""
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in targets.items()}


def parse_batch(batch: Any) -> tuple:
    """
    Parse batch from dataloader.
    Supports tuple (images, targets) or dict format.
    """
    if isinstance(batch, (list, tuple)):
        images = batch[0]
        targets = batch[1] if len(batch) > 1 else {}
    elif isinstance(batch, dict):
        images = batch.get('images') or batch.get('image')
        targets = {k: v for k, v in batch.items() if k not in ['images', 'image']}
    else:
        raise ValueError(f"Unsupported batch format: {type(batch)}")
    return images, targets


class ProductionTrainLoop:
    """
    Production-grade training loop matching the pseudo-code specification.
    
    Features:
    - Multi-head loss support (detection, scene, depth, OCR)
    - Mixed precision training (optional)
    - Gradient clipping
    - Learning rate scheduling
    - Checkpointing
    - Validation loop
    - Backbone freezing (for fine-tuning scenarios)
    
    Modular design: Can be reused for QAT fine-tuning in Sprint 2.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        loss_fn: Optional[nn.Module] = None,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_epochs: int = 100,
        use_mixed_precision: bool = True,
        gradient_clip_norm: float = 1.0,
        log_interval: int = 50,
        checkpoint_dir: str = './checkpoints',
        save_best_only: bool = True,
        freeze_backbone: bool = False,
        freeze_backbone_epochs: int = 0,
        seed: int = 42
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.device = device
        self.num_epochs = num_epochs
        self.gradient_clip_norm = gradient_clip_norm
        self.log_interval = log_interval
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.freeze_backbone = freeze_backbone
        self.freeze_backbone_epochs = freeze_backbone_epochs
        
        # Identify backbone parameters (ResNet layers)
        self.backbone_params = []
        self.head_params = []
        for name, param in model.named_parameters():
            if any(x in name for x in ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']):
                self.backbone_params.append(param)
            else:
                self.head_params.append(param)
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
        
        # Mixed precision
        self.use_mixed_precision = use_mixed_precision and AMP_AVAILABLE and device == 'cuda'
        if self.use_mixed_precision and GradScaler is not None:
            self.scaler = GradScaler()
        else:
            self.scaler = None
            self.use_mixed_precision = False
        
        # Optimizer with separate LRs for backbone and heads (if not frozen)
        if freeze_backbone:
            # Only optimize heads if backbone is frozen
            self.optimizer = AdamW(
                self.head_params,
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            # Different learning rates for backbone (lower) and heads (higher)
            param_groups = [
                {'params': self.backbone_params, 'lr': learning_rate * 0.1},
                {'params': self.head_params, 'lr': learning_rate}
            ]
            self.optimizer = AdamW(
                param_groups,
                weight_decay=weight_decay
            )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=learning_rate * 0.01
        )
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_state = None
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
        
        # Set seed
        set_seed(seed)
    
    def _freeze_backbone(self):
        """Freeze ResNet backbone parameters."""
        for param in self.backbone_params:
            param.requires_grad = False
        print(" Backbone frozen (only heads will be trained)")
    
    def _unfreeze_backbone(self):
        """Unfreeze ResNet backbone parameters."""
        for param in self.backbone_params:
            param.requires_grad = True
        print(" Backbone unfrozen (full model training)")
    
    def compute_multihead_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute multi-head loss.
        
        Supports:
        - Detection loss (classification + bbox)
        - Scene classification loss
        - Depth regression loss
        - OCR region loss
        
        If loss_fn is provided, use it. Otherwise, compute basic losses.
        """
        if self.loss_fn is not None:
            # Use provided loss function (e.g., MaxSightLoss)
            if hasattr(self.loss_fn, 'forward'):
                loss_dict = self.loss_fn(outputs, targets)
                if isinstance(loss_dict, dict):
                    return loss_dict.get('total_loss', loss_dict.get('loss', sum(loss_dict.values())))
                return loss_dict
            else:
                return self.loss_fn(outputs, targets)
        
        # Fallback: basic loss computation
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Detection loss (if available)
        if 'classifications' in outputs and 'labels' in targets:
            cls_logits = outputs['classifications']
            cls_targets = targets['labels']
            if cls_logits.dim() == 3:  # [B, num_locations, num_classes]
                cls_logits = cls_logits.reshape(-1, cls_logits.size(-1))
                cls_targets = cls_targets.reshape(-1)
            ce_loss = nn.functional.cross_entropy(cls_logits, cls_targets, ignore_index=-1)
            total_loss += ce_loss
        
        # Bbox loss (if available)
        if 'boxes' in outputs and 'boxes' in targets:
            bbox_pred = outputs['boxes']
            bbox_target = targets['boxes']
            if bbox_pred.dim() == 3:  # [B, num_locations, 4]
                bbox_pred = bbox_pred.reshape(-1, 4)
                bbox_target = bbox_target.reshape(-1, 4)
            smooth_l1 = nn.functional.smooth_l1_loss(bbox_pred, bbox_target)
            total_loss += smooth_l1
        
        # Urgency loss (if available)
        if 'urgency_scores' in outputs and 'urgency' in targets:
            urgency_logits = outputs['urgency_scores']
            urgency_targets = targets['urgency']
            ce_loss = nn.functional.cross_entropy(urgency_logits, urgency_targets)
            total_loss += 0.5 * ce_loss
        
        return total_loss
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Parse batch
            images, targets = parse_batch(batch)
            
            # Move to device
            images = images.to(self.device)
            targets = move_targets_to_device(targets, self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_mixed_precision and self.scaler is not None:
                device_type = 'cuda' if self.device.startswith('cuda') else 'mps' if self.device == 'mps' else 'cpu'
                with autocast(device_type=device_type):  # type: ignore
                    outputs = self.model(images)
                    loss = self.compute_multihead_loss(outputs, targets)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.compute_multihead_loss(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Logging
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                print(f"  Epoch {epoch+1} [{batch_idx+1}/{len(self.train_loader)}] "
                      f"Loss: {avg_loss:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate(self, epoch: int) -> float:
        """Validate model."""
        if self.val_loader is None:
            return float('inf')
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Parse batch
                images, targets = parse_batch(batch)
                
                # Move to device
                images = images.to(self.device)
                targets = move_targets_to_device(targets, self.device)
                
                # Forward pass
                if self.use_mixed_precision and self.scalar is not None:
                    device_type = "cuda" if self.device.startswith("cuda") else "mps" if self.device == "mps" else "cpu"
                    with autocast(device_type=device_type):  # type: ignore
                        outputs = self.model(images)
                        loss = self.compute_multihead_loss(outputs, targets)
                # Backward pass with scaling
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], 
                        self.gradient_clip_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images)
                    loss = self.compute_multihead_loss(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], 
                        self.gradient_clip_norm
                    )
                    self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1 
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss
    
    def save_checkpoint(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        is_best: bool = False
    ):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
        }
        
        # Save epoch checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Clean up old checkpoints if save_best_only
        if self.save_best_only and not is_best:
            checkpoint_path.unlink(missing_ok=True)
    
    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.
        
        Returns:
            Dictionary with training history and best model path
        """
        print("Starting Production Training Loop")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_mixed_precision}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Val batches: {len(self.val_loader)}")
        
        start_time = time.time()
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            
            # Unfreeze backbone after freeze_backbone_epochs
            if self.freeze_backbone_epochs > 0 and epoch == self.freeze_backbone_epochs + 1:
                self._unfreeze_backbone()
                # Recreate optimizer with all parameters
                param_groups = [
                    {'params': self.backbone_params, 'lr': self.learning_rate * 0.1},
                    {'params': self.head_params, 'lr': self.learning_rate}
                ]
                self.optimizer = AdamW(param_groups, weight_decay=self.weight_decay)
            
            # Train
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Validate
            val_loss = self.validate(epoch)
            self.history['val_loss'].append(val_loss)
            
            # Step scheduler
            self.scheduler.step()
            
            # Save best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_state = deepcopy(self.model.state_dict())
            
            # Save checkpoint
            self.save_checkpoint(epoch, train_loss, val_loss, is_best)
            
            # Print epoch summary
            print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        elapsed_time = time.time() - start_time
        
        # Save training history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return {
            'best_val_loss': self.best_val_loss,
            'best_model_path': str(self.checkpoint_dir / 'best_model.pt'),
            'history': self.history,
            'checkpoint_dir': str(self.checkpoint_dir)
        }


# Convenience function for quick training
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    loss_fn: Optional[nn.Module] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to train a model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        loss_fn: Loss function (optional, will use basic losses if not provided)
        **kwargs: Additional arguments for ProductionTrainLoop
    
    Returns:
        Training results dictionary
    """
    trainer = ProductionTrainLoop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        **kwargs
    )
    return trainer.train()

