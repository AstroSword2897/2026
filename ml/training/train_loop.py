"""
Production-grade training loop for MaxSight CNN - IMPROVED VERSION

This version fixes all identified issues:
- Safe mixed precision handling
- Fixed gradient accumulation edge cases
- EMA with bias correction
- Official PyTorch schedulers
- Safe backbone freezing
- Loss dict with .get() defaults
- Integrated DetectionMetrics for mAP
- Resume capability
- Batch validation
- Comprehensive logging (production-grade)
- Proper error handling and exception management

Author: Production-grade improvements based on detailed analysis
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    CosineAnnealingWarmRestarts,
    SequentialLR,
    LinearLR,
    ConstantLR
)
from typing import Dict, Optional, Any, Tuple, List
from pathlib import Path
import json
import time
import logging
from copy import deepcopy
import numpy as np

try:
    from torch.amp import autocast
    from torch.cuda.amp import GradScaler
    AMP_AVAILABLE = True
except ImportError:
    class DummyAutocast:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    autocast = DummyAutocast
    GradScaler = None
    AMP_AVAILABLE = False

from ml.training.metrics import DetectionMetrics

# Setup logging
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.debug(f"Random seed set to {seed}")

def move_targets_to_device(targets: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    """Move all tensor targets to device."""
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in targets.items()}

def parse_batch(batch: Any) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Parse batch from dataloader with validation.
    Supports tuple (images, targets) or dict format.
    
        Arguments:
        batch: Batch from DataLoader (tuple or dict)
    
    Returns:
        Tuple of (images tensor, targets dict)
    
    Raises:
        ValueError: If batch format is invalid or images are malformed
    """
    if isinstance(batch, (list, tuple)):
        images = batch[0]
        targets = batch[1] if len(batch) > 1 else {}
    elif isinstance(batch, dict):
        images = batch.get('images') or batch.get('image')
        if images is None:
            raise ValueError("Batch must contain 'images' or 'image' key")
        targets = {k: v for k, v in batch.items() if k not in ['images', 'image']}
    else:
        raise ValueError(f"Unsupported batch format: {type(batch)}")
    
    # Validate images
    if not torch.is_tensor(images):
        raise ValueError(f"Images must be a tensor, got {type(images)}")
    if images.dim() != 4:
        raise ValueError(f"Images must be 4D [B, C, H, W], got shape {images.shape}")
    
    return images, targets

class EMA:
    """
    Exponential Moving Average with bias correction.
    
    Maintains shadow copies of model parameters with exponential moving average.
    Provides bias correction for early training steps.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999, total_steps: int = 10000):
        """
        Initialize EMA.
        
        Arguments:
            model: Model to track
            decay: EMA decay factor
            total_steps: Total training steps for bias correction
        """
        self.decay = decay
        self.total_steps = total_steps
        self.global_step = 0
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module) -> None:
        """Update shadow parameters with bias correction."""
        self.global_step += 1
        
        # Bias correction: adjust decay for early steps
        bias_correction = 1 - (self.decay ** self.global_step)
        effective_decay = self.decay / bias_correction if bias_correction > 0 else self.decay
        
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = effective_decay * self.shadow[name] + (1 - effective_decay) * param.data
    
    def apply_shadow(self, model: nn.Module) -> None:
        """Apply shadow parameters to model."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module) -> None:
        """Restore original parameters from backup."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        # Clear backup after restore
        self.backup.clear()

class ProductionTrainLoop:
    """
    Production-grade training loop with all improvements.
    
    Features:
    - Safe mixed precision (proper fallback handling)
    - Fixed gradient accumulation (no double-update)
    - EMA with bias correction
    - Official PyTorch schedulers
    - Safe backbone freezing (isinstance checks)
    - Loss dict with .get() defaults
    - Integrated DetectionMetrics for mAP
    - Resume capability
    - Batch validation
    - Comprehensive logging (production-grade)
    - Proper error handling and exception management
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
        gradient_accumulation_steps: int = 1,
        log_interval: int = 50,
        checkpoint_dir: str = './checkpoints',
        save_best_only: bool = True,
        freeze_backbone: bool = False,
        freeze_backbone_epochs: int = 0,
        freeze_bn_stats: bool = True,
        ema_decay: float = 0.9999,
        scheduler_type: str = 'cosine',  # 'cosine', 'onecycle', 'cosine_restarts'
        warmup_epochs: int = 5,
        num_classes: int = 80,  # For DetectionMetrics
        resume_from: Optional[str] = None,
        seed: int = 42,
        logger: Optional[logging.Logger] = None,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 0.0,
        early_stopping_metric: str = 'val_loss'  # 'val_loss' or 'val_map'
    ):
        """
        Initialize production training loop.
        
        Arguments:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            loss_fn: Loss function (optional, uses default if None)
            device: Device to train on ('cuda', 'cpu', 'mps')
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            num_epochs: Number of training epochs
            use_mixed_precision: Use mixed precision training
            gradient_clip_norm: Gradient clipping norm
            gradient_accumulation_steps: Steps to accumulate gradients
            log_interval: Logging interval (batches)
            checkpoint_dir: Directory to save checkpoints
            save_best_only: Only save best model
            freeze_backbone: Freeze backbone parameters
            freeze_backbone_epochs: Epochs to freeze backbone
            freeze_bn_stats: Freeze BatchNorm stats when freezing backbone
            ema_decay: EMA decay factor
            scheduler_type: LR scheduler type
            warmup_epochs: Warmup epochs
            num_classes: Number of classes for metrics
            resume_from: Path to checkpoint to resume from
            seed: Random seed
            logger: Optional logger instance
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_interval = log_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.freeze_backbone = freeze_backbone
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.freeze_bn_stats = freeze_bn_stats
        self.ema_decay = ema_decay
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.num_classes = num_classes
        self.seed = seed
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_counter = 0
        self.early_stopping_best_metric = float('inf') if early_stopping_metric == 'val_loss' else 0.0
        
        # Setup logger
        self.logger = logger or logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Set seed
        set_seed(seed)
        
        # Mixed precision
        self.use_mixed_precision = use_mixed_precision and AMP_AVAILABLE and (
            device == 'cuda' or str(device).startswith('cuda') or device == 'mps'
        )
        if self.use_mixed_precision and GradScaler is not None:
            self.scaler = GradScaler()
        else:
            self.scaler = None
            self.use_mixed_precision = False
            if use_mixed_precision:
                self.logger.warning("Mixed precision requested but not available, disabling")
        
        # Optimizer setup with discriminative learning rates
        self.backbone_params = []
        self.head_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Identify backbone vs head parameters
            if any(bb_name in name for bb_name in ['backbone', 'resnet', 'conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']):
                self.backbone_params.append(param)
            else:
                self.head_params.append(param)
        
        if self.freeze_backbone and self.freeze_backbone_epochs > 0:
            # Freeze backbone initially
            self._freeze_backbone()
            param_groups = [
                {'params': self.head_params, 'lr': learning_rate}
            ]
        else:
            param_groups = [
                {'params': self.backbone_params, 'lr': learning_rate * 0.1},
                {'params': self.head_params, 'lr': learning_rate}
            ]
        
        self.optimizer = AdamW(param_groups, weight_decay=weight_decay)
        
        # Scheduler - Use official PyTorch schedulers
        total_steps = len(train_loader) * num_epochs
        warmup_steps = warmup_epochs * len(train_loader) if warmup_epochs > 0 else 0
        
        # Ensure warmup doesn't exceed total steps
        if warmup_steps >= total_steps:
            warmup_epochs = 0
            warmup_steps = 0
            self.logger.warning(f"Warmup steps ({warmup_steps}) >= total steps, disabling warmup")
        
        if scheduler_type == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=learning_rate,
                total_steps=total_steps,
                pct_start=0.3,
                anneal_strategy='cos'
            )
        elif scheduler_type == 'cosine_restarts':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=len(train_loader) * 10,  # Restart every 10 epochs
                T_mult=2,
                eta_min=learning_rate * 0.01
            )
        elif scheduler_type == 'cosine':
            if warmup_steps > 0:
                # Warmup + Cosine
                warmup_scheduler = LinearLR(
                    self.optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_steps
                )
                cosine_steps = max(1, total_steps - warmup_steps)
                cosine_scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=cosine_steps,
                    eta_min=learning_rate * 0.01
                )
                self.scheduler = SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_steps]
                )
            else:
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=max(1, total_steps),
                    eta_min=learning_rate * 0.01
                )
        else:
            # Default: constant LR
            self.scheduler = ConstantLR(self.optimizer, factor=1.0)
        
        # EMA with bias correction
        self.ema = EMA(model, decay=ema_decay, total_steps=total_steps) if ema_decay > 0 else None
        
        # DetectionMetrics for validation
        self.detection_metrics = DetectionMetrics(
            num_classes=num_classes,
            iou_thresholds=[0.5, 0.75],
            device=torch.device(device)
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_map = 0.0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_map': [],
            'val_map_50': [],
            'val_map_75': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'learning_rates': []
        }
        
        # Resume from checkpoint if provided
        if resume_from:
            try:
                self._load_checkpoint(resume_from)
            except Exception as e:
                self.logger.error(f"Failed to load checkpoint {resume_from}: {e}")
                raise
    
    def _freeze_backbone(self) -> None:
        """Freeze backbone parameters safely using isinstance checks."""
        frozen_count = 0
        for name, module in self.model.named_modules():
            # Check if this is a backbone module
            if any(bb_name in name for bb_name in ['backbone', 'resnet', 'conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']):
                for param in module.parameters():
                    param.requires_grad = False
                    frozen_count += 1
                
                # Freeze BatchNorm stats if requested
                if self.freeze_bn_stats and isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.eval()
        
        if frozen_count > 0:
            self.logger.info(f"Backbone frozen ({frozen_count} parameters, BN stats frozen: {self.freeze_bn_stats})")
    
    def _unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        unfrozen_count = 0
        for name, module in self.model.named_modules():
            if any(bb_name in name for bb_name in ['backbone', 'resnet', 'conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']):
                for param in module.parameters():
                    param.requires_grad = True
                    unfrozen_count += 1
                
                # Unfreeze BatchNorm stats
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.train()
        
        if unfrozen_count > 0:
            self.logger.info(f"Backbone unfrozen ({unfrozen_count} parameters)")
    
    def compute_multihead_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-head loss with safe .get() defaults.
        
        Arguments:
            outputs: Model outputs dictionary
            targets: Target labels dictionary
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        if self.loss_fn is not None:
            loss_dict = self.loss_fn(outputs, targets)
        else:
            # Default loss computation
            loss_dict = {
                'total_loss': torch.tensor(0.0, device=self.device),
                'classification_loss': torch.tensor(0.0, device=self.device),
                'localization_loss': torch.tensor(0.0, device=self.device),
                'objectness_loss': torch.tensor(0.0, device=self.device)
            }
        
        # Safe access with defaults
        total_loss = loss_dict.get('total_loss', torch.tensor(0.0, device=self.device))
        
        return total_loss, loss_dict
    
    def _step_optimizer(self) -> None:
        """Unified optimizer step with safe scaler handling."""
        # Only clip gradients for trainable parameters (exclude frozen params)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if self.scaler is not None:
            # Unscale gradients before clipping
            self.scaler.unscale_(self.optimizer)
            if trainable_params:
                torch.nn.utils.clip_grad_norm_(trainable_params, self.gradient_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if trainable_params:
                torch.nn.utils.clip_grad_norm_(trainable_params, self.gradient_clip_norm)
            self.optimizer.step()
        
        self.optimizer.zero_grad()
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with fixed gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        accum_steps = 0  # Track accumulation steps
        
        # Progress bar for training
        try:
            from tqdm import tqdm
            use_tqdm = True
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        except ImportError:
            use_tqdm = False
            pbar = self.train_loader
        
        for batch_idx, batch in enumerate(pbar):
            # Parse and validate batch
            try:
                images, targets = parse_batch(batch)
            except (ValueError, KeyError) as e:
                self.logger.warning(f"Skipping invalid batch {batch_idx}: {e}")
                continue
            
            # Move to device
            try:
                images = images.to(self.device)
                targets = move_targets_to_device(targets, self.device)
            except Exception as e:
                self.logger.error(f"Failed to move batch to device: {e}")
                continue
            
            # Forward pass with mixed precision - SAFE handling
            try:
                if self.device.startswith('cuda'):
                    device_type = 'cuda'
                elif self.device == 'mps':
                    device_type = 'cpu'  # MPS uses CPU autocast
                else:
                    device_type = 'cpu'
                
                if self.use_mixed_precision:
                    with autocast(device_type=device_type):  # type: ignore
                        outputs = self.model(images)
                        loss, loss_dict = self.compute_multihead_loss(outputs, targets)
                        loss = loss / self.gradient_accumulation_steps
                else:
                    outputs = self.model(images)
                    loss, loss_dict = self.compute_multihead_loss(outputs, targets)
                    loss = loss / self.gradient_accumulation_steps
            except Exception as e:
                self.logger.error(f"Forward pass failed at batch {batch_idx}: {e}")
                continue
            
            # Backward pass - SAFE scaler handling
            try:
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            except Exception as e:
                self.logger.error(f"Backward pass failed at batch {batch_idx}: {e}")
                self.optimizer.zero_grad()
                continue
            
            accum_steps += 1
            
            # Gradient accumulation: only step when we've accumulated enough OR it's the last batch
            is_last_batch = (batch_idx + 1) == len(self.train_loader)
            should_step = (accum_steps % self.gradient_accumulation_steps == 0) or is_last_batch
            
            if should_step:
                try:
                    self._step_optimizer()
                except Exception as e:
                    self.logger.error(f"Optimizer step failed at batch {batch_idx}: {e}")
                    self.optimizer.zero_grad()
                    continue
                
                # Step scheduler if per-step
                if isinstance(self.scheduler, (OneCycleLR, SequentialLR)):
                    self.scheduler.step()
                
                # Update EMA
                if self.ema is not None:
                    self.ema.update(self.model)
                
                self.global_step += 1
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            if use_tqdm:
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                    'lr': f"{current_lr:.2e}"
                })
            
            # Logging
            if (batch_idx + 1) % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                if len(self.optimizer.param_groups) > 1:
                    backbone_lr = self.optimizer.param_groups[0]['lr']
                    head_lr = self.optimizer.param_groups[1]['lr']
                    self.logger.info(
                        f"Epoch {epoch+1} [{batch_idx+1}/{len(self.train_loader)}] "
                        f"Loss: {loss.item() * self.gradient_accumulation_steps:.4f}, "
                        f"LR: backbone={backbone_lr:.2e}, head={head_lr:.2e}"
                    )
                else:
                    self.logger.info(
                        f"Epoch {epoch+1} [{batch_idx+1}/{len(self.train_loader)}] "
                        f"Loss: {loss.item() * self.gradient_accumulation_steps:.4f}, "
                        f"LR: {current_lr:.2e}"
                    )
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return {'loss': avg_loss}
    
    def validate(self, epoch: int, use_ema: bool = True) -> Dict[str, float]:
        """
        Validate model with DetectionMetrics integration.
        
        Arguments:
            epoch: Current epoch number
            use_ema: If True, always use EMA weights for validation (default: True)
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Apply EMA weights if available
        if use_ema and self.ema is not None:
            self.ema.apply_shadow(self.model)
        
        # Reset metrics
        self.detection_metrics.reset()
        
        with torch.no_grad():
            if self.val_loader is None:
                return {'loss': float('inf')}
            for batch_idx, batch in enumerate(self.val_loader):
                try:
                    images, targets = parse_batch(batch)
                    images = images.to(self.device)
                    targets = move_targets_to_device(targets, self.device)
                except Exception as e:
                    self.logger.warning(f"Skipping invalid validation batch: {e}")
                    continue
                
                try:
                    outputs = self.model(images)
                    loss, loss_dict = self.compute_multihead_loss(outputs, targets)
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Update DetectionMetrics if we have detection outputs
                    if 'boxes' in outputs and 'labels' in targets:
                        # Extract predictions (assuming format matches DetectionMetrics)
                        pred_boxes = outputs.get('boxes', torch.empty(0, 4, device=self.device))
                        pred_labels = outputs.get('classifications', torch.empty(0, dtype=torch.long, device=self.device))
                        pred_scores = outputs.get('scores', torch.ones(pred_labels.shape[0], device=self.device))
                        
                        gt_boxes = targets.get('boxes', torch.empty(0, 4, device=self.device))
                        gt_labels = targets.get('labels', torch.empty(0, dtype=torch.long, device=self.device))
                        
                        # Convert to proper format if needed
                        if pred_boxes.dim() == 3:
                            pred_boxes = pred_boxes.reshape(-1, 4)
                        if pred_labels.dim() > 1:
                            pred_labels = pred_labels.argmax(dim=-1)
                        
                        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                            self.detection_metrics.update(
                                pred_boxes=pred_boxes,
                                pred_labels=pred_labels,
                                pred_scores=pred_scores,
                                gt_boxes=gt_boxes,
                                gt_labels=gt_labels,
                                iou_threshold=0.5
                            )
                except Exception as e:
                    self.logger.error(f"Validation failed at batch {batch_idx}: {e}")
                    continue
        
        # Restore original weights if EMA was used
        if use_ema and self.ema is not None:
            self.ema.restore(self.model)
            self.ema.backup.clear()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        # Compute mAP and other metrics
        try:
            map_results = self.detection_metrics.compute_map(iou_threshold=0.5)
            map_50 = map_results.get('mAP@0.5', map_results.get('mAP', 0.0))
            map_75 = map_results.get('mAP@0.75', 0.0)
            overall_map = map_results.get('mAP', 0.0)
            
            precision = self.detection_metrics.compute_precision()
            recall = self.detection_metrics.compute_recall()
            f1 = self.detection_metrics.compute_f1()
        except Exception as e:
            self.logger.warning(f"Failed to compute metrics: {e}")
            map_50 = 0.0
            map_75 = 0.0
            overall_map = 0.0
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        
        return {
            'loss': avg_loss,
            'map': overall_map,
            'map_50': map_50,
            'map_75': map_75,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save checkpoint with comprehensive state."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_map': self.best_val_map,
            'history': self.history,
            'config': {
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'num_epochs': self.num_epochs,
                'scheduler_type': self.scheduler_type,
                'warmup_epochs': self.warmup_epochs,
                'num_classes': self.num_classes,
                'batch_size': self.train_loader.batch_size if hasattr(self.train_loader, 'batch_size') else None,
                'device': str(self.device),
                'gradient_clip_norm': self.gradient_clip_norm,
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'use_mixed_precision': self.use_mixed_precision,
                'ema_decay': self.ema_decay,
                'freeze_backbone': self.freeze_backbone,
                'freeze_backbone_epochs': self.freeze_backbone_epochs,
                'seed': self.seed,
                'early_stopping_patience': self.early_stopping_patience,
                'early_stopping_min_delta': self.early_stopping_min_delta,
                'early_stopping_metric': self.early_stopping_metric
            },
            'metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_steps': self.global_step,
                'train_samples': len(self.train_loader.dataset) if hasattr(self.train_loader, 'dataset') and hasattr(self.train_loader.dataset, '__len__') else None,  # type: ignore[arg-type]
                'val_samples': len(self.val_loader.dataset) if self.val_loader and hasattr(self.val_loader, 'dataset') and hasattr(self.val_loader.dataset, '__len__') else None  # type: ignore[arg-type]
            }
        }
        
        # Add EMA state if available
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.shadow
            checkpoint['ema_global_step'] = self.ema.global_step
        
        # Save last checkpoint
        last_checkpoint_path = self.checkpoint_dir / 'last_checkpoint.pt'
        try:
            torch.save(checkpoint, last_checkpoint_path)
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            return
        
        # Save best model
        if is_best:
            best_checkpoint_path = self.checkpoint_dir / 'best_model.pt'
            try:
                torch.save(checkpoint, best_checkpoint_path)
                self.logger.info(
                    f"Saved best model (val_loss: {self.best_val_loss:.4f}, "
                    f"val_map: {self.best_val_map:.4f})"
                )
            except Exception as e:
                self.logger.error(f"Failed to save best model: {e}")
        
        # Save final model
        if epoch == self.num_epochs - 1:
            final_checkpoint_path = self.checkpoint_dir / 'final_model.pt'
            try:
                torch.save(checkpoint, final_checkpoint_path)
            except Exception as e:
                self.logger.error(f"Failed to save final model: {e}")
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint and resume training."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint.get('epoch', 0)
            self.global_step = checkpoint.get('global_step', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.best_val_map = checkpoint.get('best_val_map', 0.0)
            self.history = checkpoint.get('history', self.history)
            
            # Restore EMA if available
            if self.ema is not None and 'ema_state_dict' in checkpoint:
                self.ema.shadow = checkpoint['ema_state_dict']
                self.ema.global_step = checkpoint.get('ema_global_step', 0)
            
            self.logger.info(f"Resumed from checkpoint: epoch {self.current_epoch}, step {self.global_step}")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise
    
    def train(self) -> Dict[str, Any]:
        """Run full training loop."""
        self.logger.info("Starting Production Training Loop")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Mixed Precision: {self.use_mixed_precision}")
        self.logger.info(f"Gradient Accumulation: {self.gradient_accumulation_steps}")
        self.logger.info(f"EMA: {self.ema is not None}")
        self.logger.info(f"Scheduler: {self.scheduler_type}")
        self.logger.info(f"Epochs: {self.num_epochs}")
        self.logger.info(f"Train batches: {len(self.train_loader)}")
        if self.val_loader:
            self.logger.info(f"Val batches: {len(self.val_loader)}")
        
        start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, self.num_epochs):
                self.logger.info(f"\nEpoch {epoch+1}/{self.num_epochs}")
                
                # Unfreeze backbone after freeze_backbone_epochs
                if self.freeze_backbone_epochs > 0 and epoch == self.freeze_backbone_epochs:
                    self._unfreeze_backbone()
                    # Recreate optimizer with all parameters
                    param_groups = [
                        {'params': self.backbone_params, 'lr': self.learning_rate * 0.1},
                        {'params': self.head_params, 'lr': self.learning_rate}
                    ]
                    self.optimizer = AdamW(param_groups, weight_decay=self.weight_decay)
                    # Recreate scheduler
                    total_steps = len(self.train_loader) * (self.num_epochs - epoch)
                    if self.scheduler_type == 'cosine':
                        self.scheduler = CosineAnnealingLR(
                            self.optimizer,
                            T_max=total_steps,
                            eta_min=self.learning_rate * 0.01
                        )
                
                # Train
                train_metrics = self.train_epoch(epoch)
                self.history['train_loss'].append(train_metrics['loss'])
                
                # Step scheduler (if not per-step)
                if not isinstance(self.scheduler, (OneCycleLR, SequentialLR)):
                    self.scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['learning_rates'].append(current_lr)
                
                # Validate
                if self.val_loader:
                    val_metrics = self.validate(epoch, use_ema=True)
                    val_loss = val_metrics.get('loss', float('inf'))
                    val_map = val_metrics.get('map', 0.0)
                    val_map_50 = val_metrics.get('map_50', 0.0)
                    val_map_75 = val_metrics.get('map_75', 0.0)
                    val_precision = val_metrics.get('precision', 0.0)
                    val_recall = val_metrics.get('recall', 0.0)
                    val_f1 = val_metrics.get('f1', 0.0)
                    
                    self.history['val_loss'].append(val_loss)
                    self.history['val_map'].append(val_map)
                    self.history['val_map_50'].append(val_map_50)
                    self.history['val_map_75'].append(val_map_75)
                    self.history['val_precision'].append(val_precision)
                    self.history['val_recall'].append(val_recall)
                    self.history['val_f1'].append(val_f1)
                    
                    # Check if best model
                    is_best = val_loss < self.best_val_loss or val_map > self.best_val_map
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                    if val_map > self.best_val_map:
                        self.best_val_map = val_map
                    
                    # Early stopping check
                    if self.early_stopping_patience > 0:
                        if self.early_stopping_metric == 'val_loss':
                            current_metric = val_loss
                            improvement = self.early_stopping_best_metric - current_metric
                            if improvement > self.early_stopping_min_delta:
                                self.early_stopping_best_metric = current_metric
                                self.early_stopping_counter = 0
                            else:
                                self.early_stopping_counter += 1
                        else:  # val_map
                            current_metric = val_map
                            improvement = current_metric - self.early_stopping_best_metric
                            if improvement > self.early_stopping_min_delta:
                                self.early_stopping_best_metric = current_metric
                                self.early_stopping_counter = 0
                            else:
                                self.early_stopping_counter += 1
                        
                        if self.early_stopping_counter >= self.early_stopping_patience:
                            self.logger.info(
                                f"Early stopping triggered after {epoch+1} epochs. "
                                f"No improvement for {self.early_stopping_patience} epochs."
                            )
                            break
                    
                    # Save checkpoint
                    self._save_checkpoint(epoch, is_best=is_best)
                    
                    self.logger.info(
                        f"Train Loss: {train_metrics['loss']:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val mAP: {val_map:.4f}, "
                        f"Val mAP@0.5: {val_map_50:.4f}, Val mAP@0.75: {val_map_75:.4f}, "
                        f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}"
                    )
                else:
                    # No validation, just save checkpoint
                    self._save_checkpoint(epoch, is_best=False)
                    self.logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user")
            self._save_checkpoint(self.current_epoch, is_best=False)
            raise
        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            self._save_checkpoint(self.current_epoch, is_best=False)
            raise
        
        elapsed_time = time.time() - start_time
        
        # Save training history
        history_path = self.checkpoint_dir / 'training_history.json'
        try:
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save training history: {e}")
        
        # Also save as CSV for easy plotting with pandas
        try:
            import pandas as pd
            df = pd.DataFrame(self.history)
            csv_path = self.checkpoint_dir / 'training_history.csv'
            df.to_csv(csv_path, index=False)
        except ImportError:
            pass  # pandas not available, skip CSV export
        
        self.logger.info("\nTraining Complete!")
        self.logger.info(f"Best validation mAP: {self.best_val_map:.4f}")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info(f"Total time: {elapsed_time/3600:.2f} hours")
        self.logger.info(f"Checkpoints saved to: {self.checkpoint_dir}")
        
        return {
            'best_val_loss': self.best_val_loss,
            'best_val_map': self.best_val_map,
            'best_model_path': str(self.checkpoint_dir / 'best_model.pt'),
            'history': self.history,
            'checkpoint_dir': str(self.checkpoint_dir)
        }

# Convenience function
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    loss_fn: Optional[nn.Module] = None,
    **kwargs
) -> Dict[str, Any]:
    """Convenience function to train a model."""
    trainer = ProductionTrainLoop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        **kwargs
    )
    return trainer.train()
