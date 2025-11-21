import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data 
import DataLoader
from typing import Dict, Optional
from pathlib import Path
import copy

try:
    from torch.amp import autocast
    from torch.cuda.amp import GradScaler
    AMP_AVILABLE = True
except ImportError:
    class DummyAutocast:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    autocast = DummyAutocast
    GradScaler = None
    AMP_AVILABLE = False

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
            )
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class Trainer:
    """Improved Trainer class for MaxSight CNN"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cpu',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        freeze_backbone_epochs: int = 5,
        use_mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        warmup_epochs: int = 3,
        ema_decay: float = 0.9999,
        early_stopping_patience: int = 10,
        criterion: Optional[nn.Module] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_epochs = warmup_epochs
        self.early_stopping_patience = early_stopping_patience
        self.patience_counter = 0
        
        # Mixed precision
        self.use_mixed_precision = use_mixed_precision and AMP_AVAILABLE and device == 'cuda'
        if self.use_mixed_precision and GradScaler is not None:
            self.scaler = GradScaler()
        else:
            self.scaler = None
            self.use_mixed_precision = False
        
        # Loss function
        self.criterion = criterion
        
        # EMA
        self.ema = EMA(model, decay=ema_decay) if ema_decay > 0 else None
        
        # Optimizer - different LRs for backbone and heads
        self.backbone_params = []
        self.head_params = []
        
        for name, param in self.model.named_parameters():
            if any(x in name for x in ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']):
                self.backbone_params.append(param)
            else:
                self.head_params.append(param)
        
        self.optimizer = optim.AdamW(
            [
                {'params': self.backbone_params, 'lr': learning_rate * 0.1},
                {'params': self.head_params, 'lr': learning_rate}
            ],
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scheduler
        self.base_lr = learning_rate
        self.warmup_steps = warmup_epochs * len(train_loader)
        self.total_steps = 0  # Track total training steps
        
        # Cosine annealing (will be set in train())
        self.scheduler = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        self.best_val_loss = float('inf')
    
    def _get_lr_scale(self, step: int) -> float:
        """Get learning rate scale based on warmup and cosine decay"""
        if step < self.warmup_steps:
            # Linear warmup
            return step / self.warmup_steps
        else:
            # Cosine decay after warmup
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
    
    def _update_lr(self, step: int):
        """Update learning rate with warmup + cosine schedule"""
        lr_scale = self._get_lr_scale(step)
        for i, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lr * 0.1 if i == 0 else self.base_lr  # Backbone vs heads
            param_group['lr'] = base_lr * lr_scale
    def freeze_backbone(self):
        """Freeze ResNet backbone"""
        for param in self.backbone_params:
            param.requires_grad = False
        print("✓ Backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze ResNet backbone"""
        for param in self.backbone_params:
            param.requires_grad = True
        print("✓ Backbone unfrozen for fine-tuning")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        # Freeze/unfreeze backbone
        if epoch == 1:
            self.freeze_backbone()
        elif epoch == self.freeze_backbone_epochs + 1:
            self.unfreeze_backbone()
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['images'].to(self.device)
            targets = {
                'labels': batch['labels'].to(self.device),
                'boxes': batch['boxes'].to(self.device),
                'urgency': batch.get('urgency', torch.zeros(images.size(0), dtype=torch.long)).to(self.device),
                'distance': batch.get('distance', torch.zeros_like(batch['labels'])).to(self.device),
                'num_objects': batch.get('num_objects', torch.tensor([batch['labels'].size(1)] * images.size(0)))
            }
            
            # Update learning rate
            step = (epoch - 1) * len(self.train_loader) + batch_idx
            self._update_lr(step)
            
            # Forward pass with mixed precision
            if self.criterion is None:
                raise ValueError("Criterion (loss function) must be provided to Trainer")
            
            if self.use_mixed_precision and self.scaler is not None:
                device_type = 'cuda' if self.device == 'cuda' else 'mps'  # Determine device type for autocast
                with autocast(device_type=device_type):  # type: ignore  # FP16 forward pass (new API, type stubs may be outdated)
                    outputs = self.model(images)
                    loss_dict = self.criterion(outputs, targets)
                    loss = loss_dict['total_loss'] / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict['total_loss'] / self.gradient_accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_mixed_precision and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update EMA after optimizer step
                if self.ema is not None:
                    self.ema.update()
            
            total_loss += loss_dict['total_loss'].item()
            total_samples += images.size(0)
            
            if batch_idx % 10 == 0:
                current_lr = self.optimizer.param_groups[-1]['lr']
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss_dict["total_loss"].item():.4f}, '
                      f'Cls: {loss_dict["classification_loss"].item():.4f}, '
                      f'Box: {loss_dict["localization_loss"].item():.4f}, '
                      f'Obj: {loss_dict["objectness_loss"].item():.4f}, '
                      f'LR: {current_lr:.6f}')
        
        avg_loss = total_loss / len(self.train_loader)
        return {'loss': avg_loss}
    
    def validate(self, use_ema: bool = True) -> Dict[str, float]:
        """Validate the model"""
        if self.val_loader is None:
            return {}
        
        # Use EMA weights if available
        if use_ema and self.ema is not None:
            self.ema.apply_shadow()
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['images'].to(self.device)
                targets = {
                    'labels': batch['labels'].to(self.device),
                    'boxes': batch['boxes'].to(self.device),
                    'urgency': batch.get('urgency', torch.zeros(images.size(0), dtype=torch.long)).to(self.device),
                    'distance': batch.get('distance', torch.zeros_like(batch['labels'])).to(self.device),
                    'num_objects': batch.get('num_objects', torch.tensor([batch['labels'].size(1)] * images.size(0)))
                }
                
                if self.criterion is None:
                    raise ValueError("Criterion (loss function) must be provided to Trainer")
                
                if self.use_mixed_precision and self.scaler is not None:
                    device_type = 'cuda' if self.device == 'cuda' else 'mps'  # Determine device type for autocast
                    with autocast(device_type=device_type):  # type: ignore  # FP16 forward pass (new API, type stubs may be outdated)
                        outputs = self.model(images)
                        loss_dict = self.criterion(outputs, targets)
                else:
                    outputs = self.model(images)
                    loss_dict = self.criterion(outputs, targets)
                
                total_loss += loss_dict['total_loss'].item()
                
                # Calculate accuracy (top-1 classification)
                # Use objectness to find most confident detection
                for b in range(images.size(0)):
                    obj_scores = outputs['objectness'][b]  # [N]
                    top_idx = obj_scores.argmax()
                    pred_cls = outputs['classifications'][b, top_idx].argmax()
                    
                    # Compare to first ground truth object
                    gt_cls = targets['labels'][b, 0]
                    if pred_cls == gt_cls:
                        correct += 1
                    total += 1
        
        # Restore original weights
        if use_ema and self.ema is not None:
            self.ema.restore()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(self, num_epochs: int, save_dir: str = 'checkpoints'):
        """Main training loop"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Set total steps for LR schedule
        self.total_steps = num_epochs * len(self.train_loader)
        
        print(f"\n{'='*60}")
        print(f"Starting Training - {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_mixed_precision}")
        print(f"Gradient Accumulation: {self.gradient_accumulation_steps}")
        print(f"EMA: {self.ema is not None}")
        print(f"Warmup Epochs: {self.warmup_epochs}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            
            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate(use_ema=True)
                val_loss = val_metrics.get('loss', 0)
                val_acc = val_metrics.get('accuracy', 0)
                
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                print(f"\nTrain Loss: {train_metrics['loss']:.4f}")
                print(f"Val Loss: {val_loss:.4f}")
                print(f"Val Acc: {val_acc:.2f}%")
                
                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    
                    # Save best model with EMA
                    if self.ema is not None:
                        self.ema.apply_shadow()
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': self.best_val_loss,
                        'ema_shadow': self.ema.shadow if self.ema else None,
                    }, save_path / 'best_model.pth')
                    
                    if self.ema is not None:
                        self.ema.restore()
                    
                    print(f"✓ Saved best model (val_loss: {self.best_val_loss:.4f})")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.early_stopping_patience:
                        print(f"\nEarly stopping after {self.early_stopping_patience} epochs without improvement")
                        break
            else:
                print(f"Train Loss: {train_metrics['loss']:.4f}")
            
            current_lr = self.optimizer.param_groups[-1]['lr']
            self.history['learning_rates'].append(current_lr)
            print(f"Learning Rate: {current_lr:.6f}")
        
        # Save final model
        if self.ema is not None:
            self.ema.apply_shadow()
        
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'ema_shadow': self.ema.shadow if self.ema else None,
        }, save_path / 'final_model.pth')
        
        if self.ema is not None:
            self.ema.restore()
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        return self.history
