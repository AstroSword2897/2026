"""
Production-grade Quantization-Aware Training (QAT) for MaxSight models.

Use this when PTQ degrades accuracy >1% on critical heads (embedding, bbox, urgency).
"""

import torch
import torch.nn as nn
import torch.ao.quantization as quantization
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import time
import warnings
from copy import deepcopy
import json
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.training.losses import MaxSightLoss

torch.backends.quantized.engine = "qnnpack"


def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fuse_maxsight_model(model: nn.Module):
    """
    Fuse conv+bn+relu patterns for MaxSight CNN architecture.
    
    MaxSight uses ResNet50 backbone with FPN, so we fuse:
    - ResNet layers (conv+bn+relu)
    - FPN layers (conv+bn+relu)
    - Detection head layers (conv+bn+relu)
    """
    patterns = []
    
    # Auto-detect fusion patterns in all modules
    for name, module in model.named_modules():
        try:
            children = list(module._modules.keys())
            for i in range(len(children) - 2):
                a, b, c = children[i], children[i+1], children[i+2]
                m_a, m_b, m_c = module._modules[a], module._modules[b], module._modules[c]
                
                if isinstance(m_a, nn.Conv2d) and \
                   isinstance(m_b, (nn.BatchNorm2d, nn.BatchNorm1d)) and \
                   isinstance(m_c, (nn.ReLU, nn.ReLU6)):
                    full_name = f"{name}.{a}" if name else a
                    patterns.append([f"{full_name}", f"{name}.{b}" if name else b, f"{name}.{c}" if name else c])
        except Exception:
            continue
    
    # Remove duplicates
    unique_patterns = []
    seen = set()
    for p in patterns:
        p_tuple = tuple(p)
        if p_tuple not in seen:
            seen.add(p_tuple)
            unique_patterns.append(p)
    
    # Apply fusion
    fused_count = 0
    for p in unique_patterns:
        try:
            quantization.fuse_modules(model, p, inplace=True)
            fused_count += 1
        except Exception as e:
            warnings.warn(f"Could not fuse pattern {p}: {e}")
    
    print(f"Fused {fused_count} module patterns")
    return model


class QATTrainer:
    """
    Production QAT trainer with multi-head loss support and validation.
    Integrated with MaxSightLoss for proper multi-task training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Optional[Callable] = None,
        backend: str = 'qnnpack',
        lr: float = 1e-5,
        epochs: int = 5,
        device: str = 'cuda',
        output_dir: str = './artifacts/qat',
        fuse: bool = True,
        warmup_epochs: int = 1,
        log_interval: int = 50,
        num_classes: int = 48,
        freeze_backbone: bool = True,  # Freeze backbone for QAT fine-tuning
        freeze_backbone_epochs: int = 2,  # Unfreeze after N epochs
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn or MaxSightLoss(num_classes=num_classes)
        self.backend = backend
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fuse = fuse
        self.warmup_epochs = warmup_epochs
        self.log_interval = log_interval
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        self.freeze_backbone_epochs = freeze_backbone_epochs
        
        # Identify backbone parameters
        self.backbone_params = []
        self.head_params = []
        for name, param in model.named_parameters():
            if any(x in name for x in ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']):
                self.backbone_params.append(param)
            else:
                self.head_params.append(param)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': [],
            'lr': []
        }
        
        set_seed(42)
        
    def prepare_model_for_qat(self) -> nn.Module:
        """Prepare model for QAT by fusing and setting qconfig."""
        model_qat = deepcopy(self.model).to('cpu')
        model_qat.eval()
        
        # Set backend
        torch.backends.quantized.engine = self.backend
        
        # Fuse modules
        if self.fuse:
            try:
                model_qat = fuse_maxsight_model(model_qat)
                print("Fusion completed")
            except Exception as e:
                warnings.warn(f"Fusion failed: {e}")
        
        # Set qconfig for QAT
        qconfig = quantization.get_default_qat_qconfig(self.backend)
        
        # Use per-channel weight quantization for qnnpack (mobile/ARM)
        if self.backend == 'qnnpack':
            try:
                qconfig = deepcopy(qconfig)
                if hasattr(qconfig, 'weight'):
                    qconfig.weight = quantization.default_per_channel_weight_observer  # type: ignore
            except Exception:
                pass
        
        # Set qconfig on model (PyTorch typing quirk - this is valid)
        setattr(model_qat, 'qconfig', qconfig)  # type: ignore
        
        # Prepare for QAT (inserts fake quantization modules)
        model_qat = quantization.prepare_qat(model_qat, inplace=False)
        
        print(f"Model prepared for QAT with {self.backend} backend")
        return model_qat.to(self.device)
    
    def train_epoch(self, model: nn.Module, optimizer, epoch: int) -> float:
        """Train for one epoch with fake quantization."""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Handle MaxSight dataset format
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
                targets = batch[1] if len(batch) > 1 else {}
            elif isinstance(batch, dict):
                inputs = batch.get('images') or batch.get('image')
                targets = {k: v for k, v in batch.items() if k not in ['images', 'image']}
            else:
                raise ValueError("Unsupported batch format")
            
            if inputs is not None:
                inputs = inputs.to(self.device)
            
            # Move targets to device
            if isinstance(targets, dict):
                targets = {k: v.to(self.device) if torch.is_tensor(v) else v 
                          for k, v in targets.items()}
            elif torch.is_tensor(targets):
                targets = targets.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss (MaxSightLoss expects outputs dict and targets dict)
            if isinstance(self.loss_fn, MaxSightLoss):
                loss_dict = self.loss_fn(outputs, targets)
                loss = loss_dict['total_loss']
            else:
                # Fallback for generic loss functions
                loss = self.loss_fn(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                print(f"  Epoch {epoch+1} [{batch_idx+1}/{len(self.train_loader)}] "
                      f"Loss: {avg_loss:.4f}")
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate(self, model: nn.Module, epoch: int) -> Dict[str, float]:
        """Validate model and compute metrics."""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Handle MaxSight dataset format
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                    targets = batch[1] if len(batch) > 1 else {}
                elif isinstance(batch, dict):
                    inputs = batch.get('images') or batch.get('image')
                    targets = {k: v for k, v in batch.items() if k not in ['images', 'image']}
                else:
                    continue
                
                if inputs is not None:
                    inputs = inputs.to(self.device)
                
                # Move targets to device
                if isinstance(targets, dict):
                    targets = {k: v.to(self.device) if torch.is_tensor(v) else v 
                              for k, v in targets.items()}
                elif torch.is_tensor(targets):
                    targets = targets.to(self.device)
                
                outputs = model(inputs)
                
                # Compute loss
                if isinstance(self.loss_fn, MaxSightLoss):
                    loss_dict = self.loss_fn(outputs, targets)
                    loss = loss_dict['total_loss']
                else:
                    loss = self.loss_fn(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        metrics = {
            'loss': avg_loss,
            'num_batches': num_batches
        }
        
        print(f"  Validation - Epoch {epoch+1}: Loss={avg_loss:.4f}")
        return metrics
    
    def train(self) -> Dict[str, Any]:
        """
        Run full QAT training loop.
        Returns dict with best model path and training history.
        """
        print("Starting Quantization-Aware Training (QAT)")
        
        # Prepare model for QAT
        model_qat = self.prepare_model_for_qat()
        
        # Setup optimizer with backbone freezing support
        if self.freeze_backbone:
            # Freeze backbone initially
            for param in self.backbone_params:
                param.requires_grad = False
            print("Backbone frozen for QAT fine-tuning (training heads only)")
            # Only optimize heads
            optimizer = AdamW(
                self.head_params,
                lr=self.lr,
                weight_decay=1e-4
            )
        else:
            # Full model training with different LRs
            param_groups = [
                {'params': self.backbone_params, 'lr': self.lr * 0.1},
                {'params': self.head_params, 'lr': self.lr}
            ]
            optimizer = AdamW(
                param_groups,
                weight_decay=1e-4
            )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=self.lr * 0.01
        )
        
        best_val_loss = float('inf')
        best_epoch = -1
        best_model_state = None
        
        # Training loop
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            print("-" * 70)
            
            # Unfreeze backbone after freeze_backbone_epochs
            if self.freeze_backbone and epoch == self.freeze_backbone_epochs:
                for param in self.backbone_params:
                    param.requires_grad = True
                print("Backbone unfrozen (full model QAT)")
                # Recreate optimizer with all parameters
                param_groups = [
                    {'params': self.backbone_params, 'lr': self.lr * 0.1},
                    {'params': self.head_params, 'lr': self.lr}
                ]
                optimizer = AdamW(param_groups, weight_decay=1e-4)
            
            # Enable/disable observer and fake quant based on warmup
            if epoch < self.warmup_epochs:
                print("  [Warmup] Enabling observers, disabling fake quantization")
                model_qat.apply(torch.ao.quantization.enable_observer)
                model_qat.apply(torch.ao.quantization.disable_fake_quant)
            else:
                print("  [QAT] Enabling observers and fake quantization")
                model_qat.apply(torch.ao.quantization.enable_observer)
                model_qat.apply(torch.ao.quantization.enable_fake_quant)
            
            # Train
            train_loss = self.train_epoch(model_qat, optimizer, epoch)
            self.history['train_loss'].append(train_loss)
            self.history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Validate
            val_metrics = self.validate(model_qat, epoch)
            val_loss = val_metrics['loss']
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = deepcopy(model_qat.state_dict())
                print(f"  New best model (loss: {best_val_loss:.4f})")
            
            # Step scheduler
            scheduler.step()
            
            # Save checkpoint
            checkpoint_path = self.output_dir / f'qat_checkpoint_epoch{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_qat.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
        
        print(f"\nQAT Training Complete! Best epoch: {best_epoch+1}")
        
        # Load best model and convert to INT8
        if best_model_state is not None:
            model_qat.load_state_dict(best_model_state)
        model_qat.eval()
        model_qat.to('cpu')
        
        # Disable observers before conversion
        model_qat.apply(torch.ao.quantization.disable_observer)
        
        # Convert to fully quantized INT8 model
        print("\nConverting QAT model to INT8...")
        model_int8 = quantization.convert(model_qat, inplace=False)
        
        # Save models
        qat_model_path = self.output_dir / 'model_qat_best.pt'
        int8_model_path = self.output_dir / 'model_int8_from_qat.pt'
        
        torch.save(best_model_state, qat_model_path)
        torch.save(model_int8.state_dict(), int8_model_path)
        
        # Save training history
        history_path = self.output_dir / 'qat_history.json'
        with open(history_path, 'w') as f:
            json.dump({
                'train_loss': self.history['train_loss'],
                'val_loss': self.history['val_loss'],
                'lr': self.history['lr'],
                'best_epoch': best_epoch,
                'best_val_loss': best_val_loss,
            }, f, indent=2)
        
        print(f"\nBest QAT model saved: {qat_model_path}")
        print(f"INT8 model saved: {int8_model_path}")
        print(f"Training history saved: {history_path}")
        
        return {
            'qat_model_path': str(qat_model_path),
            'int8_model_path': str(int8_model_path),
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'history': self.history,
            'model_int8': model_int8
        }


# CLI usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='QAT fine-tuning for MaxSight')
    parser.add_argument('--model-file', type=str, required=True,
                       help='Python file with build_model() function')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Data directory for train/val')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of QAT epochs')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--output-dir', type=str, default='./artifacts/qat',
                       help='Output directory')
    parser.add_argument('--backend', type=str, default='qnnpack',
                       choices=['qnnpack', 'fbgemm'],
                       help='Quantization backend')
    parser.add_argument('--num-classes', type=int, default=48,
                       help='Number of classes')
    
    args = parser.parse_args()
    
    # Dynamic model import
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_def", args.model_file)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Failed to load model file: {args.model_file}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model_def"] = mod
    spec.loader.exec_module(mod)
    
    if not hasattr(mod, "build_model"):
        raise SystemExit("Model file must define build_model() function")
    
    model = mod.build_model()
    
    # Setup dataloaders (adapt to your dataset)
    from torch.utils.data import DataLoader, TensorDataset
    
    # Dummy dataloaders - replace with actual MaxSight dataset
    print("Creating dummy dataloaders - replace with actual MaxSight dataset")
    train_data = TensorDataset(
        torch.randn(100, 3, 224, 224),
        torch.randint(0, 10, (100,))
    )
    val_data = TensorDataset(
        torch.randn(20, 3, 224, 224),
        torch.randint(0, 10, (20,))
    )
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    
    # Use MaxSightLoss
    loss_fn = MaxSightLoss(num_classes=args.num_classes)
    
    # Create trainer
    trainer = QATTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        backend=args.backend,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        output_dir=args.output_dir,
        num_classes=args.num_classes
    )
    
    # Run QAT
    results = trainer.train()
    
    print("\nQAT Results:")
    print(f"  Best model: {results['qat_model_path']}")
    print(f"  INT8 model: {results['int8_model_path']}")
    print(f"  Best validation loss: {results['best_val_loss']:.4f}")

