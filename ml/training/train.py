import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
from pathlib import Path
import math

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
                    (1 - self.decay) * param.data.detach()
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
    # Training loop with mixed precision, EMA, and early stopping.
    
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
        self.use_mixed_precision = use_mixed_precision and AMP_AVAILABLE and (device == 'cuda' or str(device).startswith('cuda'))
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
        # Track total training steps
        self.total_steps = 0
        self.global_step = 0
        
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
        """LR scale: linear warmup then cosine decay."""
        if step < self.warmup_steps:
            # Linear warmup
            return max(1e-6, (step + 1) / max(1, self.warmup_steps))
        else:
            # Cosine decay after warmup
            den = max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, (step - self.warmup_steps) / den)) 
            return 0.5 * (1 + math.cos(progress * math.pi))
    
    def _update_lr(self, step: int):
        """Update LR using warmup + cosine schedule."""
        lr_scale = self._get_lr_scale(step)
        for i, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lr * 0.1 if i == 0 else self.base_lr  # Backbone vs heads
            param_group['lr'] = base_lr * lr_scale
    
    def freeze_backbone(self):
        #Freeze ResNet backbone and its BatchNorm layers.
        for param in self.backbone_params:
            param.requires_grad = False

        #Freeze BatchNorm Running stats
        for name, module in self.model.named_modules():
            if any(x in name for x in ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']):
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.eval()
                    module.track_running_stats = False
                    for p in module.parameters():
                        p.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze ResNet backbone."""
        for param in self.backbone_params:
            param.requires_grad = True
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        # Freeze/unfreeze backbone dynamically
        if self.freeze_backbone_epochs > 0:
            if epoch <= self.freeze_backbone_epochs:
                if not all(not p.requires_grad for p in self.backbone_params):
                    self.freeze_backbone()
            else:
                if not all(p.requires_grad for p in self.backbone_params):
                    self.unfreeze_backbone()
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['images'].to(self.device)
            targets = {
                'labels': batch['labels'].to(self.device),
                'boxes': batch['boxes'].to(self.device),
                'urgency': batch.get('urgency', torch.zeros(images.size(0), dtype=torch.long)).to(self.device),
                'distance': batch.get('distance', torch.zeros_like(batch['labels'])).to(self.device),
                'num_objects': batch.get('num_objects', torch.full((images.size(0),), batch['labels'].size(1), device=self.device, dtype=torch.long))
            }
            
            # Update learning rate
            step = (epoch - 1) * len(self.train_loader) + batch_idx
            self._update_lr(step)
            
            # Forward pass with mixed precision
            if self.criterion is None:
                raise ValueError("Criterion (loss function) must be provided to Trainer")
            
            # Use AMP if available (speeds up on GPU, saves memory)
            device_str = str(self.device)
            with autocast(device_type='cuda' if 'cuda' in device_str else 'cpu',
              enabled=self.use_mixed_precision):
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict['total_loss'] / self.gradient_accumulation_steps
            
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation for the final batch
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
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
        
        # Final partial accumalation - Handled
        if (batch_idx + 1) % self.gradient_accumulation_steps != 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update()

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
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['images'].to(self.device)
                targets = {
                    'labels': batch['labels'].to(self.device),
                    'boxes': batch['boxes'].to(self.device),
                    'urgency': batch.get('urgency', torch.zeros(images.size(0), dtype=torch.long)).to(self.device),
                    'distance': batch.get('distance', torch.zeros_like(batch['labels'])).to(self.device),
                    'num_objects': batch.get(
                        'num_objects',
                        torch.full(
                            (images.size(0),),
                            batch['labels'].size(1),
                            device=self.device,
                            dtype=torch.long
                        )
                    )

                }
                
                if self.criterion is None:
                    raise ValueError("Criterion (loss function) must be provided to Trainer")
                if self.use_mixed_precision:
                    device_str = str(self.device)
                    with autocast(device_type='cuda' if 'cuda' in device_str else 'cpu',
                                enabled=self.use_mixed_precision):
                        outputs = self.model(images)
                        loss_dict = self.criterion(outputs, targets)

                else:
                    outputs = self.model(images)
                    loss_dict = self.criterion(outputs, targets)
                
                total_loss += loss_dict['total_loss'].item()
                
                # TODO: Replace with mAP from DetectionMetrics (accuracy is wrong for detection)
        
        # Restore original weights
        if use_ema and self.ema is not None:
            self.ema.restore()
        
        avg_loss = total_loss / len(self.val_loader)
        return {'loss': avg_loss}
    
    def train(self, num_epochs: int, save_dir: str = 'checkpoints'):
        """Main training loop"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Set total steps for LR schedule
        self.total_steps = num_epochs * len(self.train_loader)
        
        print(f"\nStarting Training - {num_epochs} epochs")
        print(f"Device: {self.device}, Mixed Precision: {self.use_mixed_precision}")
        print(f"Gradient Accumulation: {self.gradient_accumulation_steps}, EMA: {self.ema is not None}")
        print(f"Warmup Epochs: {self.warmup_epochs}\n")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            
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
                    
                    print(f"Saved best model (val_loss: {self.best_val_loss:.4f})")
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
        
        # Save training history
        history_path = save_path / 'training_history.json'
        import json
        with open(history_path, 'w') as f:
            # JSON serialization form
            json_history = {
                'train_loss': [float(x) for x in self.history['train_loss']],
                'val_loss': [float(x) for x in self.history['val_loss']],
                'learning_rates': [float(x) for x in self.history['learning_rates']]
            }
            json.dump(json_history, f, indent=2)
        
        print(f"\nTraining Complete!")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"History saved to: {history_path}")
        
        return self.history
