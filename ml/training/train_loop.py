"""Production-grade training loop for MaxSight CNN - IMPROVED VERSION."""

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
import math
from copy import deepcopy
import numpy as np

from ml.training.stability_manager import StabilityManager

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

# GradNorm integration (optional)
try:
    from ml.training.task_balancing import GradNormMultiHeadLoss
    GRADNORM_AVAILABLE = True
except ImportError:
    GRADNORM_AVAILABLE = False
    GradNormMultiHeadLoss = None

# Setup logging.
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS support (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS has no explicit seed API; manual_seed covers RNG.
        pass
    logger.debug(f"Random seed set to {seed}")


def move_targets_to_device(targets: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    """Move all tensor targets to device."""
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in targets.items()}


def parse_batch(batch: Any) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Parse batch from dataloader with validation. Handles multiple batch formats: - List/tuple: (images, targets) - Dict: {'images': ..., 'labels': ..., ...} Returns: Tuple of (images tensor, targets dict)"""
    if isinstance(batch, (list, tuple)):
        images = batch[0]
        targets = batch[1] if len(batch) > 1 else {}
    elif isinstance(batch, dict):
        # Support both 'images' and 'image' keys for flexibility.
        # Check None explicitly (can't use 'or' with tensors - causes ambiguous boolean error)
        images = batch.get('images')
        if images is None:
            images = batch.get('image')
        if images is None:
            raise ValueError("Batch must contain 'images' or 'image' key")
        targets = {k: v for k, v in batch.items() if k not in ['images', 'image']}
    else:
        raise ValueError(f"Unsupported batch format: {type(batch)}")
    
    # Validate image tensor shape.
    if not torch.is_tensor(images):
        raise ValueError(f"Images must be a tensor, got {type(images)}")
    if images.dim() != 4:
        raise ValueError(f"Images must be 4D [B, C, H, W], got shape {images.shape}")
    
    return images, targets


class EMA:
    """Exponential Moving Average with bias correction. Maintains shadow copies of model parameters with exponential moving average. Provides bias correction for early training steps."""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999, total_steps: int = 10000):
        """Initialize EMA. Arguments: model: Model to track decay: EMA decay factor total_steps: Total training steps for bias correction."""
        self.decay = decay
        self.total_steps = total_steps
        self.global_step = 0
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters.
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module) -> None:
        """Update shadow parameters with bias correction."""
        self.global_step += 1
        
        # Bias correction: adjust decay for early steps.
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
        # Clear backup after restore.
        self.backup.clear()
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state dict for checkpointing."""
        return {
            'shadow': self.shadow,
            'decay': self.decay,
            'total_steps': self.total_steps,
            'global_step': self.global_step
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict from checkpoint."""
        self.shadow = state_dict.get('shadow', {})
        self.decay = state_dict.get('decay', self.decay)
        self.total_steps = state_dict.get('total_steps', self.total_steps)
        self.global_step = state_dict.get('global_step', 0)


def _validate_training_config(
    gradient_accumulation_steps: int,
    ema_decay: float,
    learning_rate: float,
    weight_decay: float,
    use_gradnorm: bool,
    num_epochs: int,
    warmup_epochs: int,
) -> None:
    """Validate training config before initialization (Fix #7)."""
    errors = []
    if gradient_accumulation_steps <= 0:
        errors.append("gradient_accumulation_steps must be positive")
    if not (0 <= ema_decay <= 1):
        errors.append(f"ema_decay must be in [0, 1], got {ema_decay}")
    if learning_rate <= 0:
        errors.append(f"learning_rate must be positive, got {learning_rate}")
    if weight_decay < 0:
        errors.append(f"weight_decay must be non-negative, got {weight_decay}")
    if warmup_epochs >= num_epochs:
        errors.append(
            f"warmup_epochs ({warmup_epochs}) must be < num_epochs ({num_epochs})"
        )
    if not use_gradnorm:
        import sys
        print(
            "WARNING: GradNorm disabled. For T5's 15-head architecture, "
            "GradNorm is STRONGLY recommended to prevent gradient warfare.",
            file=sys.stderr,
        )
    if errors:
        raise ValueError("Config validation failed:\n" + "\n".join(f"  • {e}" for e in errors))


class ProductionTrainLoop:
    """Production-grade training loop with all improvements."""
    
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
        learning_rate_backbone: Optional[float] = None,  # If set with learning_rate_head, use for param_groups.
        learning_rate_head: Optional[float] = None,
        num_classes: int = 80,  # For DetectionMetrics.
        checkpoint_interval: int = 0,  # Save an extra snapshot every N epochs (0 = only last/best)
        resume_from: Optional[str] = None,
        resume_model_only: bool = False,
        seed: int = 42,
        logger: Optional[logging.Logger] = None,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 0.0,
        early_stopping_metric: str = 'val_loss',  # 'val_loss' or 'val_map'
        use_gradnorm: bool = False,  # Enable GradNorm task balancing.
        gradnorm_alpha: float = 1.5,  # GradNorm restoring force.
        gradnorm_update_interval: int = 100  # Update task weights every N iterations.
    ):
        """Initialize production training loop."""
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn.to(device) if loss_fn is not None and hasattr(loss_fn, 'to') else loss_fn
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
        self.learning_rate_backbone = learning_rate_backbone
        self.learning_rate_head = learning_rate_head
        self.checkpoint_interval = checkpoint_interval
        self.resume_model_only = resume_model_only
        self.num_classes = num_classes
        self.seed = seed
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_counter = 0
        self.early_stopping_best_metric = None  # Set on first validation (Fix #4)
        
        # Config validation (Fix #7)
        _validate_training_config(
            gradient_accumulation_steps=gradient_accumulation_steps,
            ema_decay=ema_decay,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            use_gradnorm=use_gradnorm,
            num_epochs=num_epochs,
            warmup_epochs=warmup_epochs,
        )
        
        # Setup logger.
        self.logger = logger or logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Set seed.
        set_seed(seed)
        
        # GradNorm integration (CRITICAL: Prevents gradient warfare with 20 heads)
        self.use_gradnorm = use_gradnorm and GRADNORM_AVAILABLE
        self.gradnorm_loss = None
        self.original_loss_fn = loss_fn  # Store for fallback.
        
        if self.use_gradnorm:
            if not GRADNORM_AVAILABLE:
                self.logger.warning("GradNorm requested but not available, disabling")
                self.use_gradnorm = False
            elif loss_fn is None:
                self.logger.warning("GradNorm requires loss_fn to be provided, disabling GradNorm")
                self.use_gradnorm = False
            else:
                # Check if loss_fn is already a GradNormMultiHeadLoss or MultiHeadLoss.
                if GRADNORM_AVAILABLE and GradNormMultiHeadLoss is not None and isinstance(loss_fn, GradNormMultiHeadLoss):
                    # Already a GradNorm loss, use directly.
                    self.gradnorm_loss = loss_fn
                    self.logger.info("Using provided GradNormMultiHeadLoss")
                elif isinstance(loss_fn, nn.Module) and hasattr(loss_fn, 'loss_functions'):
                    # MultiHeadLoss with dict of head losses - wrap with GradNorm.
                    head_losses = getattr(loss_fn, 'loss_functions', {})
                    if isinstance(head_losses, dict) and len(head_losses) > 0:
                        # Get shared parameters (backbone) for GradNorm.
                        shared_params = list(self.backbone_params) if self.backbone_params else None
                        
                        if GRADNORM_AVAILABLE and GradNormMultiHeadLoss is not None:
                            self.gradnorm_loss = GradNormMultiHeadLoss(
                                head_losses=head_losses,
                                shared_params=shared_params,
                                alpha=gradnorm_alpha,
                                update_interval=gradnorm_update_interval
                            )
                            self.logger.info(f"GradNorm enabled with {len(head_losses)} head losses")
                        else:
                            self.logger.warning("GradNorm not available, using standard loss")
                            self.use_gradnorm = False
                    else:
                        self.logger.warning("Loss function doesn't have valid head losses dict, disabling GradNorm")
                        self.use_gradnorm = False
                else:
                    # Single loss function - cannot use GradNorm directly. Log warning but don't fail - will use standard loss.
                    self.logger.warning(
                        "GradNorm requires MultiHeadLoss with dict of head losses. "
                        "Using standard loss computation. To enable GradNorm, provide a "
                        "MultiHeadLoss or GradNormMultiHeadLoss instance."
                    )
                    self.use_gradnorm = False
        
        # Mixed precision.
        self.use_mixed_precision = use_mixed_precision and AMP_AVAILABLE and (
            device == 'cuda' or str(device).startswith('cuda') or device == 'mps'
        )
        if self.use_mixed_precision and GradScaler is not None:
            try:
                from torch.amp import GradScaler as AmpGradScaler
                self.scaler = AmpGradScaler('cuda')
            except (ImportError, TypeError, AttributeError):
                self.scaler = GradScaler()
        else:
            self.scaler = None
            self.use_mixed_precision = False
            if use_mixed_precision:
                self.logger.warning("Mixed precision requested but not available, disabling")
        
        # Optimizer setup with discriminative learning rates.
        self.backbone_params = []
        self.head_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Identify backbone vs head parameters.
            if any(bb_name in name for bb_name in ['backbone', 'resnet', 'conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']):
                self.backbone_params.append(param)
            else:
                self.head_params.append(param)
        
        lr_backbone = learning_rate_backbone if learning_rate_backbone is not None else learning_rate * 0.1
        lr_head = learning_rate_head if learning_rate_head is not None else learning_rate
        if self.freeze_backbone and self.freeze_backbone_epochs > 0:
            # Freeze backbone initially.
            self._freeze_backbone()
            param_groups = [
                {'params': self.head_params, 'lr': lr_head}
            ]
        else:
            param_groups = [
                {'params': self.backbone_params, 'lr': lr_backbone},
                {'params': self.head_params, 'lr': lr_head}
            ]
        
        self.optimizer = AdamW(param_groups, weight_decay=weight_decay)
        
        # Stability manager for auto-adjustment during training.
        self.stability_manager = None  # Will be initialized after scheduler.
        
        # Scheduler - Use official PyTorch schedulers.
        total_steps = len(train_loader) * num_epochs
        warmup_steps = warmup_epochs * len(train_loader) if warmup_epochs > 0 else 0
        
        # Ensure warmup doesn't exceed total steps.
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
                T_0=len(train_loader) * 10,  # Restart every 10 epochs.
                T_mult=2,
                eta_min=learning_rate * 0.001
            )
        elif scheduler_type == 'cosine':
            if warmup_steps > 0:
                # Warmup + Cosine.
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
                    eta_min=learning_rate * 0.001  # Lower min LR for finer convergence and lower final loss.
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
                    eta_min=learning_rate * 0.001  # Lower min LR for finer convergence.
                )
        else:
            # Default: constant LR.
            self.scheduler = ConstantLR(self.optimizer, factor=1.0)
        
        # EMA with bias correction.
        self.ema = EMA(model, decay=ema_decay, total_steps=total_steps) if ema_decay > 0 else None
        
        # DetectionMetrics for validation.
        self.detection_metrics = DetectionMetrics(
            num_classes=num_classes,
            iou_thresholds=[0.5, 0.75],
            device=torch.device(device)
        )
        
        # Stability manager - auto-adjusts hyperparameters during training.
        self.stability_manager = StabilityManager(
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            gradnorm_loss=self.gradnorm_loss if self.use_gradnorm else None,
            spike_threshold=0.3,  # 30% loss increase triggers adjustment.
            overfit_threshold=0.25,  # 25% val-train gap triggers regularization.
            lr_reduce_factor=0.5,  # Halve LR on spike.
            wd_increase_factor=1.5,  # 1.5x weight decay on overfit.
            max_wd=0.5,
            min_lr=1e-7,
            log_every=1,  # Log stability every epoch.
        )
        
        # Training state.
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
        
        # Resume from checkpoint if provided.
        if resume_from:
            try:
                self._load_checkpoint(resume_from, model_only=resume_model_only)
            except Exception as e:
                self.logger.error(f"Failed to load checkpoint {resume_from}: {e}")
                raise
    
    def _freeze_backbone(self) -> None:
        """Freeze backbone parameters safely using isinstance checks."""
        frozen_count = 0
        for name, module in self.model.named_modules():
            # Detect backbone module for separate learning rate or freezing.
            if any(bb_name in name for bb_name in ['backbone', 'resnet', 'conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']):
                for param in module.parameters():
                    param.requires_grad = False
                    frozen_count += 1
                
                # Freeze BatchNorm stats if requested.
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
                
                # Unfreeze BatchNorm stats.
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.train()
        
        if unfrozen_count > 0:
            self.logger.info(f"Backbone unfrozen ({unfrozen_count} parameters)")
    
    def _validate_t5_batch(
        self, images: torch.Tensor, targets: Dict[str, torch.Tensor]
    ) -> bool:
        """Validate T5-specific batch requirements (Fix #6). Returns False if batch should be skipped."""
        if images.dim() == 5:
            B, T, C, H, W = images.shape
            if T < 2:
                self.logger.warning(f"Temporal sequence too short: T={T}")
                return False
        elif images.dim() == 4:
            B, C, H, W = images.shape
        else:
            self.logger.warning(f"Invalid image dims: {images.shape}")
            return False
        if torch.isnan(images).any() or torch.isinf(images).any():
            self.logger.warning("NaN/Inf in images")
            return False
        task_keys = [
            'boxes', 'labels', 'depth', 'audio_events', 'scene_graph',
            'motion', 'ocr', 'therapy_state', 'roi_priority',
        ]
        if not any(k in targets for k in task_keys):
            self.logger.debug("No valid targets")
            return False
        if 'boxes' in targets and 'num_objects' in targets:
            boxes = targets['boxes']
            num_objects = targets['num_objects']
            for b in range(boxes.shape[0]):
                num_obj = int(num_objects[b].item())
                if num_obj > 0 and num_obj <= boxes.shape[1]:
                    actual_boxes = boxes[b, :num_obj]
                    if torch.isnan(actual_boxes).any() or torch.isinf(actual_boxes).any():
                        return False
        return True
    
    def compute_multihead_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        is_training: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute multi-head loss with safe .get() defaults."""
        if self.use_gradnorm and self.gradnorm_loss is not None and is_training:
            total_loss, loss_dict = self.gradnorm_loss(outputs, targets, model=self.model)
            if 'total_loss' not in loss_dict:
                loss_dict['total_loss'] = total_loss.item() if torch.is_tensor(total_loss) else total_loss
            if self.global_step % 100 == 0:
                task_weights = getattr(self.gradnorm_loss, 'task_weights', None)
                if task_weights is not None:
                    self.logger.debug(f"GradNorm task weights: {task_weights}")
        elif self.loss_fn is not None:
            # Standard loss computation.
            loss_result = self.loss_fn(outputs, targets)
            
            # Handle both dict and tensor returns.
            if isinstance(loss_result, dict):
                loss_dict = loss_result
            elif torch.is_tensor(loss_result):
                loss_dict = {'total_loss': loss_result}
            else:
                # Try tuple (loss, dict) format.
                if isinstance(loss_result, tuple) and len(loss_result) == 2:
                    total_loss_val, loss_dict = loss_result
                    if 'total_loss' not in loss_dict:
                        loss_dict['total_loss'] = total_loss_val
                else:
                    loss_dict = {'total_loss': torch.tensor(0.0, device=self.device)}
        else:
            if is_training and self.loss_fn is None:
                self.logger.warning(
                    "No loss function provided! Training with zero loss. "
                    "For T5's 15-head architecture, provide MultiHeadLoss or GradNormMultiHeadLoss."
                )
            loss_dict = {
                'total_loss': torch.tensor(0.0, device=self.device),
                'classification_loss': torch.tensor(0.0, device=self.device),
                'localization_loss': torch.tensor(0.0, device=self.device),
                'objectness_loss': torch.tensor(0.0, device=self.device)
            }
        
        # Safe access with defaults.
        total_loss = loss_dict.get('total_loss', torch.tensor(0.0, device=self.device))
        if not torch.is_tensor(total_loss):
            total_loss = torch.tensor(total_loss, device=self.device)
        if not total_loss.requires_grad and outputs:
            for v in outputs.values():
                if torch.is_tensor(v) and v.requires_grad and v.numel() > 0:
                    total_loss = total_loss + 1e-8 * v.pow(2).mean()
                    break
        
        return total_loss, loss_dict
    
    def _step_optimizer(self) -> None:
        """Unified optimizer step with safe scaler handling."""
        # Only clip gradients for trainable parameters (exclude frozen params)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if self.scaler is not None:
            # Unscale gradients before clipping.
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
    
    def _log_training_progress(
        self, epoch: int, batch_idx: int, batch_loss: float
    ) -> None:
        """Log training progress with T5-specific details (Fix #8)."""
        current_lr = self.optimizer.param_groups[0]['lr']
        if len(self.optimizer.param_groups) > 1:
            backbone_lr = self.optimizer.param_groups[0]['lr']
            head_lr = self.optimizer.param_groups[1]['lr']
            self.logger.info(
                f"Epoch {epoch+1} [{batch_idx+1}/{len(self.train_loader)}] "
                f"Loss: {batch_loss:.4f}, LR: backbone={backbone_lr:.2e}, head={head_lr:.2e}"
            )
        else:
            self.logger.info(
                f"Epoch {epoch+1} [{batch_idx+1}/{len(self.train_loader)}] "
                f"Loss: {batch_loss:.4f}, LR: {current_lr:.2e}"
            )
        if self.use_gradnorm and self.global_step % 500 == 0:
            if hasattr(self, 'gradnorm_loss') and self.gradnorm_loss is not None:
                weights = getattr(self.gradnorm_loss, 'task_weights', None)
                if weights is not None:
                    if isinstance(weights, dict):
                        items = sorted(weights.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)
                        top_5 = items[:5]
                        bottom_5 = items[-5:] if len(items) >= 5 else items
                        self.logger.info(
                            f"GradNorm weights - Top: {dict(top_5)}, Bottom: {dict(bottom_5)}"
                        )
                    else:
                        self.logger.info(f"GradNorm weights: {weights}")
    
    def _check_early_stopping(
        self, epoch: int, val_loss: float, val_map: float
    ) -> bool:
        """Check early stopping with proper baseline initialization (Fix #4)."""
        if self.early_stopping_patience <= 0:
            return False
        if self.early_stopping_best_metric is None:
            if self.early_stopping_metric == 'val_loss':
                self.early_stopping_best_metric = val_loss
            else:
                self.early_stopping_best_metric = val_map
            self.logger.info(
                f"Early stopping baseline ({self.early_stopping_metric}): "
                f"{self.early_stopping_best_metric:.4f}"
            )
            return False
        if self.early_stopping_metric == 'val_loss':
            current_metric = val_loss
            improvement = self.early_stopping_best_metric - current_metric
        else:
            current_metric = val_map
            improvement = current_metric - self.early_stopping_best_metric
        if improvement > self.early_stopping_min_delta:
            self.early_stopping_best_metric = current_metric
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        if self.early_stopping_counter >= self.early_stopping_patience:
            self.logger.info(
                f"Early stopping triggered after {epoch+1} epochs"
            )
            return True
        return False
    
    def train_epoch(self, epoch: int) -> Dict[str, Any]:
        """Train for one epoch with fixed gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        accum_steps = 0  # Track accumulation steps.
        epoch_losses = []  # Fix #1: track all batch losses for stability manager.
        
        # Progress bar for training.
        try:
            from tqdm import tqdm
            use_tqdm = True
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        except ImportError:
            use_tqdm = False
            pbar = self.train_loader
        
        for batch_idx, batch in enumerate(pbar):
            # Parse and validate batch.
            try:
                images, targets = parse_batch(batch)
            except (ValueError, KeyError) as e:
                self.logger.warning(f"Skipping invalid batch {batch_idx}: {e}")
                continue
            
            if not self._validate_t5_batch(images, targets):  # Fix #6.
                continue
            
            # Quick NaN/Inf check (silent - dimensions sanitized in collate_fn)
            if 'num_objects' in targets and 'boxes' in targets:
                batch_valid = True
                batch_size = targets['boxes'].shape[0]
                for b in range(batch_size):
                    num_obj = int(targets['num_objects'][b].item())
                    if num_obj > 0:
                        actual_boxes = targets['boxes'][b, :num_obj]
                        # Silent check - only skip on critical NaN/Inf issues.
                        if torch.isnan(actual_boxes).any() or torch.isinf(actual_boxes).any():
                            batch_valid = False
                            break
                if not batch_valid:
                    continue  # Skip silently - data already sanitized in collate.
            
            # Move to device.
            try:
                images = images.to(self.device)
                targets = move_targets_to_device(targets, self.device)
            except Exception as e:
                self.logger.error(f"Failed to move batch to device: {e}")
                continue
            
            # Forward pass with mixed precision - SAFE handling (Fix #3: device type)
            try:
                device_str = str(self.device)
                if 'cuda' in device_str:
                    device_type = 'cuda'
                elif 'mps' in device_str:
                    device_type = 'mps'
                else:
                    device_type = 'cpu'
                
                if self.use_mixed_precision:
                    with autocast(device_type=device_type):  # type: ignore
                        outputs = self.model(images)
                        loss, loss_dict = self.compute_multihead_loss(outputs, targets, is_training=True)
                        loss = loss / self.gradient_accumulation_steps
                else:
                    outputs = self.model(images)
                    loss, loss_dict = self.compute_multihead_loss(outputs, targets, is_training=True)
                    loss = loss / self.gradient_accumulation_steps
            except Exception as e:
                self.logger.error(f"Forward pass failed at batch {batch_idx}: {e}")
                continue
            
            # Backward pass - SAFE scaler handling.
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
            
            is_last_batch = (batch_idx + 1) == len(self.train_loader)
            should_step = (accum_steps % self.gradient_accumulation_steps == 0) or is_last_batch
            
            if should_step:
                try:
                    self._step_optimizer()
                except Exception as e:
                    self.logger.error(f"Optimizer step failed at batch {batch_idx}: {e}")
                    self.optimizer.zero_grad()
                    continue
                
                # Step scheduler if per-step.
                if isinstance(self.scheduler, (OneCycleLR, SequentialLR)):
                    self.scheduler.step()
                
                # Update EMA.
                if self.ema is not None:
                    self.ema.update(self.model)
                
                self.global_step += 1
            
            batch_loss = loss.item() * self.gradient_accumulation_steps
            epoch_losses.append(batch_loss)
            total_loss += batch_loss
            num_batches += 1
            
            # Update progress bar.
            if use_tqdm:
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{batch_loss:.4f}",
                    'lr': f"{current_lr:.2e}"
                })
            
            if (batch_idx + 1) % self.log_interval == 0:
                self._log_training_progress(epoch, batch_idx, batch_loss)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return {'loss': avg_loss, 'all_losses': epoch_losses}
    
    def validate(self, epoch: int, use_ema: bool = True) -> Dict[str, float]:
        """Validate model with DetectionMetrics integration."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Apply EMA weights if available.
        if use_ema and self.ema is not None:
            self.ema.apply_shadow(self.model)
        
        # Reset metrics.
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
                    loss, loss_dict = self.compute_multihead_loss(outputs, targets, is_training=False)
                    loss_value = loss.item() if torch.is_tensor(loss) else float(loss)
                    if not (math.isnan(loss_value) or math.isinf(loss_value)):
                        total_loss += loss_value
                        num_batches += 1
                    else:
                        self.logger.warning(
                            f"Validation batch {batch_idx} produced invalid loss: {loss_value}, skipping"
                        )
                        continue
                    
                    # Update DetectionMetrics using post-processed detections. Optimized: Process batch detections once instead of per-image.
                    if 'boxes' in outputs and 'classifications' in outputs and 'boxes' in targets:
                        try:
                            batch_size = images.shape[0]
                            
                            # Process entire batch at once (get_detections handles batches efficiently)
                            batch_detections = self.model.get_detections(
                                outputs,
                                confidence_threshold=0.3
                            )
                            
                            # Process each image's detections from batch results.
                            for b in range(batch_size):
                                gt_boxes_b = targets['boxes'][b]
                                gt_labels_b = targets['labels'][b]
                                num_objects = int(targets.get('num_objects', torch.tensor([gt_boxes_b.shape[0]]))[b].item())
                                
                                if num_objects == 0:
                                    continue
                                
                                # Extract valid ground truth boxes (only actual objects, not padding)
                                gt_boxes_valid = gt_boxes_b[:num_objects].to(self.device)
                                gt_labels_valid = gt_labels_b[:num_objects].to(self.device)
                                
                                if len(gt_boxes_valid) == 0:
                                    continue
                                
                                # Extract detections for the current image from batch results.
                                if batch_detections and len(batch_detections) > b and len(batch_detections[b]) > 0:
                                    detections_list = batch_detections[b]
                                    
                                    # Extract predictions from detection dict format.
                                    pred_boxes_list = []
                                    pred_labels_list = []
                                    pred_scores_list = []
                                    
                                    for det in detections_list:
                                        if 'box' in det:
                                            box = det['box']
                                            if isinstance(box, (list, tuple)) and len(box) == 4:
                                                pred_boxes_list.append(box)
                                                pred_labels_list.append(det.get('class', det.get('class_id', 0)))
                                                pred_scores_list.append(det.get('confidence', 0.5))
                                    
                                    if pred_boxes_list:
                                        pred_boxes = torch.tensor(pred_boxes_list, device=self.device, dtype=torch.float32)
                                        pred_labels = torch.tensor(pred_labels_list, device=self.device, dtype=torch.long)
                                        pred_scores = torch.tensor(pred_scores_list, device=self.device, dtype=torch.float32)
                                        
                                        # Update metrics with post-processed predictions.
                                        if gt_boxes_valid.dim() == 2 and gt_boxes_valid.shape[1] == 4:
                                            self.detection_metrics.update(
                                                pred_boxes=pred_boxes,
                                                pred_labels=pred_labels,
                                                pred_scores=pred_scores,
                                                gt_boxes=gt_boxes_valid,
                                                gt_labels=gt_labels_valid,
                                                iou_threshold=0.5
                                            )
                        except Exception as e:
                            self.logger.debug(f"Detection metrics update skipped for batch {batch_idx}: {e}")
                            continue
                except Exception as e:
                    self.logger.error(f"Validation failed at batch {batch_idx}: {e}")
                    continue
        
        # Restore original weights if EMA was used.
        if use_ema and self.ema is not None:
            self.ema.restore(self.model)
            self.ema.backup.clear()
        
        # Handle NaN/Inf and empty validation sets to avoid invalid metrics.
        if num_batches == 0:
            avg_loss = float('inf')
            self.logger.warning("No valid validation batches processed, returning inf loss")
        else:
            avg_loss = total_loss / num_batches
            if math.isnan(avg_loss) or math.isinf(avg_loss):
                self.logger.error(
                    f"Validation loss is NaN/Inf (total_loss={total_loss}, num_batches={num_batches}), "
                    "this indicates a serious issue with loss computation"
                )
                avg_loss = float('inf')
        
        # Compute mAP and other metrics.
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
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False, extra_path: Optional[Path] = None) -> None:
        """Save checkpoint with comprehensive state for resume on same or different GPU/machine. If extra_path is set, also save a copy there (e.g. checkpoint_epoch_N.pt)."""
        lr_current = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else self.learning_rate
        wd_current = self.optimizer.param_groups[0].get('weight_decay', 0.0) if self.optimizer.param_groups else self.weight_decay
        data_paths: Dict[str, Optional[str]] = {}
        try:
            for name, loader in [('train', self.train_loader), ('val', self.val_loader)]:
                if loader is None or not hasattr(loader, 'dataset'):
                    continue
                ds = loader.dataset
                for attr in ('annotation_file', 'annotation_path', 'ann_file', 'img_root', 'image_root', 'root', 'data_dir'):
                    try:
                        if hasattr(ds, attr):
                            val = getattr(ds, attr)
                            if val is not None:
                                data_paths[f'{name}_{attr}'] = str(Path(val).resolve()) if isinstance(val, (Path, str)) else str(val)
                            break
                    except Exception:
                        continue
                if not any(k.startswith(name + '_') for k in data_paths) and hasattr(ds, 'coco'):
                    try:
                        coco_ds = getattr(ds.coco, 'dataset', None)
                        if isinstance(coco_ds, dict) and coco_ds.get('annotation_file'):
                            data_paths[f'{name}_annotation_file'] = str(coco_ds.get('annotation_file'))
                    except Exception:
                        pass
        except Exception as e:
            self.logger.debug("Could not capture data paths for checkpoint: %s", e)
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
                'lr_current': lr_current,
                'weight_decay_current': wd_current,
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
                'early_stopping_metric': self.early_stopping_metric,
                'data_paths': data_paths if data_paths else None,
            },
            'metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_steps': self.global_step,
                'train_samples': len(self.train_loader.dataset) if hasattr(self.train_loader, 'dataset') and hasattr(self.train_loader.dataset, '__len__') else None,  # type: ignore[arg-type]
                'val_samples': len(self.val_loader.dataset) if self.val_loader and hasattr(self.val_loader, 'dataset') and hasattr(self.val_loader.dataset, '__len__') else None  # type: ignore[arg-type]
            }
        }
        
        # Add EMA state if available.
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()
        
        # Save last checkpoint.
        last_checkpoint_path = self.checkpoint_dir / 'last_checkpoint.pt'
        try:
            torch.save(checkpoint, last_checkpoint_path)
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            return
        # Write resume manifest for recovery on different GPU/machine.
        resume_info = {
            'checkpoint_path': str(last_checkpoint_path.resolve()),
            'epoch': epoch,
            'total_epochs': self.num_epochs,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'best_val_map': self.best_val_map,
            'data_paths': checkpoint['config'].get('data_paths'),
            'resume_command': 'Resume with --resume_from <path_to_last_checkpoint.pt> (and same data paths if you moved machines).',
        }
        try:
            with open(self.checkpoint_dir / 'resume_info.json', 'w') as f:
                json.dump(resume_info, f, indent=2)
        except Exception as e:
            self.logger.debug(f"Could not write resume_info.json: {e}")
        
        # Save best model.
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
        
        # Save final model.
        if epoch == self.num_epochs - 1:
            final_checkpoint_path = self.checkpoint_dir / 'final_model.pt'
            try:
                torch.save(checkpoint, final_checkpoint_path)
            except Exception as e:
                self.logger.error(f"Failed to save final model: {e}")
        
        # Optional: save to extra path (e.g. checkpoint_epoch_5.pt)
        if extra_path is not None:
            try:
                torch.save(checkpoint, extra_path)
                self.logger.info(f"Saved snapshot: {extra_path}")
            except Exception as e:
                self.logger.error(f"Failed to save snapshot to {extra_path}: {e}")
    
    def _load_checkpoint(self, checkpoint_path: str, model_only: bool = False) -> None:
        """Load checkpoint and resume training. If model_only=True, load only model + epoch (and best/history); use current optimizer/scheduler (e.g. new LRs)."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            state = checkpoint['model_state_dict']
            result = self.model.load_state_dict(state, strict=False)
            if result.missing_keys or result.unexpected_keys:
                self.logger.warning(
                    f"Checkpoint load (strict=False): missing_keys={len(result.missing_keys)}, unexpected_keys={len(result.unexpected_keys)}. "
                    "Training will continue; new layers use random init."
                )
                if result.missing_keys:
                    self.logger.debug(f"Missing: {result.missing_keys[:5]}{'...' if len(result.missing_keys) > 5 else ''}")
                if result.unexpected_keys:
                    self.logger.debug(f"Unexpected: {result.unexpected_keys[:5]}{'...' if len(result.unexpected_keys) > 5 else ''}")
            self.current_epoch = checkpoint.get('epoch', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.best_val_map = checkpoint.get('best_val_map', 0.0)
            self.history = checkpoint.get('history', self.history)
            if not model_only:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.scheduler and checkpoint.get('scheduler_state_dict'):
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.global_step = checkpoint.get('global_step', 0)
                if self.ema is not None:
                    if 'ema_state_dict' in checkpoint:
                        self.ema.load_state_dict(checkpoint['ema_state_dict'])
                    elif 'ema_shadow' in checkpoint:
                        self.ema.shadow = checkpoint['ema_shadow']
                        self.ema.global_step = checkpoint.get('ema_global_step', 0)
                self.logger.info(f"Resumed from checkpoint: epoch {self.current_epoch}, step {self.global_step}")
            else:
                self.global_step = 0  # Will be recomputed as we train.
                self.logger.info(
                    f"Resumed model only from checkpoint: epoch {self.current_epoch}. "
                    "Optimizer/scheduler use current config (new LRs, schedule)."
                )
        except EOFError as e:
            # Corrupted checkpoint (incomplete file write) - start fresh instead of crashing.
            self.logger.error(f"Checkpoint {checkpoint_path} is corrupted (EOFError). Starting training from scratch.")
            self.current_epoch = 0
            self.best_val_loss = float('inf')
            self.best_val_map = 0.0
            return
        except Exception as e:
            # Other errors (missing keys, device mismatch, etc.) - start fresh.
            self.logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            self.logger.warning("Starting training from scratch due to checkpoint load failure.")
            self.current_epoch = 0
            self.best_val_loss = float('inf')
            self.best_val_map = 0.0
            return
    
    def _run_sanity_check(self) -> None:
        """Run 1 train step + 1 val batch to fail fast on data/loss/backward issues (~30–60s)."""
        t0 = time.time()
        self.logger.info("Sanity check: 1 train step + 1 val batch (fail fast)...")
        self.model.train()
        self.optimizer.zero_grad()
        try:
            batch = next(iter(self.train_loader))
        except StopIteration:
            raise RuntimeError("Sanity check failed: train_loader is empty.")
        try:
            images, targets = parse_batch(batch)
        except Exception as e:
            raise RuntimeError(f"Sanity check failed: parse_batch error: {e}") from e
        if not self._validate_t5_batch(images, targets):
            raise RuntimeError("Sanity check failed: first train batch failed _validate_t5_batch.")
        images = images.to(self.device)
        targets = move_targets_to_device(targets, self.device)
        device_str = str(self.device)
        device_type = 'cuda' if 'cuda' in device_str else ('mps' if 'mps' in device_str else 'cpu')
        try:
            if self.use_mixed_precision and AMP_AVAILABLE:
                with autocast(device_type=device_type):  # type: ignore
                    outputs = self.model(images)
                    loss, _ = self.compute_multihead_loss(outputs, targets, is_training=True)
            else:
                outputs = self.model(images)
                loss, _ = self.compute_multihead_loss(outputs, targets, is_training=True)
        except Exception as e:
            raise RuntimeError(f"Sanity check failed: train forward/loss error: {e}") from e
        loss_val = loss.item() if torch.is_tensor(loss) else float(loss)
        if math.isnan(loss_val) or math.isinf(loss_val):
            raise RuntimeError(f"Sanity check failed: train loss is {loss_val} (NaN/Inf). Fix data or loss.")
        try:
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        except Exception as e:
            raise RuntimeError(f"Sanity check failed: train backward error: {e}") from e
        self.optimizer.zero_grad()
        if self.val_loader is not None:
            self.model.eval()
            try:
                batch_val = next(iter(self.val_loader))
            except StopIteration:
                raise RuntimeError("Sanity check failed: val_loader is empty.")
            try:
                with torch.no_grad():
                    images_v, targets_v = parse_batch(batch_val)
                    images_v = images_v.to(self.device)
                    targets_v = move_targets_to_device(targets_v, self.device)
                    outputs_v = self.model(images_v)
                    loss_v, _ = self.compute_multihead_loss(outputs_v, targets_v, is_training=False)
            except Exception as e:
                raise RuntimeError(f"Sanity check failed: val forward/loss error: {e}") from e
            loss_v_val = loss_v.item() if torch.is_tensor(loss_v) else float(loss_v)
            if math.isnan(loss_v_val) or math.isinf(loss_v_val):
                raise RuntimeError(f"Sanity check failed: val loss is {loss_v_val} (NaN/Inf). Fix data or loss.")
        elapsed = time.time() - t0
        self.logger.info(f"Sanity check passed in {elapsed:.1f}s (1 train step + 1 val batch).")

    def train(self) -> Dict[str, Any]:
        """Run full training loop."""
        self.logger.info("Starting Production Training Loop")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Mixed Precision: {self.use_mixed_precision}")
        self.logger.info(f"Gradient Accumulation: {self.gradient_accumulation_steps}")
        self.logger.info(f"EMA: {self.ema is not None}")
        self.logger.info(f"Scheduler: {self.scheduler_type}")
        self.logger.info(f"Epochs: {self.num_epochs}")
        n_train = len(self.train_loader.dataset) if hasattr(self.train_loader, 'dataset') and hasattr(self.train_loader.dataset, '__len__') else 0
        bs = getattr(self.train_loader, 'batch_size', None)
        self.logger.info(f"Train samples: {n_train}, Batch size: {bs}, Train batches: {len(self.train_loader)}")
        if self.val_loader:
            n_val = len(self.val_loader.dataset) if hasattr(self.val_loader, 'dataset') and hasattr(self.val_loader.dataset, '__len__') else 0
            self.logger.info(f"Val samples: {n_val}, Val batches: {len(self.val_loader)}")
        
        self._run_sanity_check()
        start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, self.num_epochs):
                self.logger.info(f"\nEpoch {epoch+1}/{self.num_epochs}")
                
                # Unfreeze backbone after freeze_backbone_epochs.
                if self.freeze_backbone_epochs > 0 and epoch == self.freeze_backbone_epochs:
                    self._unfreeze_backbone()
                    
                    # Preserve optimizer state when recreating.
                    old_optimizer_state = self.optimizer.state_dict() if self.optimizer else None
                    
                    # Recreate optimizer with all parameters.
                    param_groups = [
                        {'params': self.backbone_params, 'lr': self.learning_rate * 0.1},
                        {'params': self.head_params, 'lr': self.learning_rate}
                    ]
                    self.optimizer = AdamW(param_groups, weight_decay=self.weight_decay)
                    
                    # Transfer optimizer state (momentum, Adam buffers) for matching parameters.
                    if old_optimizer_state:
                        try:
                            # Map old param IDs to new param IDs.
                            old_state = old_optimizer_state.get('state', {})
                            new_state = {}
                            
                            # Create mapping from parameter IDs.
                            old_param_id_map = {}
                            for group_idx, group in enumerate(old_optimizer_state.get('param_groups', [])):
                                for param_idx, param_id in enumerate(group.get('params', [])):
                                    old_param_id_map[param_id] = (group_idx, param_idx)
                            
                            # Transfer state for matching parameters.
                            for new_group_idx, new_group in enumerate(self.optimizer.param_groups):
                                for new_param_idx, new_param in enumerate(new_group['params']):
                                    # Find matching old parameter. Simplified; full impl would match state dict by parameter name.
                                    if new_param_idx < len(old_optimizer_state.get('param_groups', [{}])[0].get('params', [])):
                                        old_param_id = old_optimizer_state['param_groups'][0]['params'][new_param_idx]
                                        if old_param_id in old_state:
                                            new_param_id = id(new_param)
                                            new_state[new_param_id] = old_state[old_param_id]
                            
                            if new_state:
                                self.optimizer.state = new_state
                                self.logger.info("Preserved optimizer state when unfreezing backbone")
                        except Exception as e:
                            self.logger.warning(f"Could not preserve optimizer state: {e}. Continuing with fresh optimizer.")
                    
                    # Recreate scheduler.
                    total_steps = len(self.train_loader) * (self.num_epochs - epoch)
                    if self.scheduler_type == 'cosine':
                        self.scheduler = CosineAnnealingLR(
                            self.optimizer,
                            T_max=total_steps,
                            eta_min=self.learning_rate * 0.01
                        )
                    elif self.scheduler_type == 'onecycle':
                        # OneCycleLR needs to be recreated with new optimizer.
                        self.scheduler = OneCycleLR(
                            self.optimizer,
                            max_lr=self.learning_rate,
                            total_steps=total_steps,
                            pct_start=0.3
                        )
                    # Note: Other schedulers will continue with existing state.
                
                # Train.
                train_metrics = self.train_epoch(epoch)
                train_loss = train_metrics['loss']
                epoch_losses: List[float] = train_metrics.get('all_losses', [])
                self.history['train_loss'].append(train_loss)
                
                # OneCycleLR and SequentialLR step per-batch in train_epoch(); others step here.
                if not isinstance(self.scheduler, (OneCycleLR, SequentialLR)):
                    self.scheduler.step()
                    self.logger.debug(f"Scheduler stepped (per-epoch). New LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['learning_rates'].append(current_lr)
                
                # Validate.
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
                    
                    # Stability check and auto-adjustment (Fix #1: use tracked epoch_losses)
                    if self.stability_manager and len(epoch_losses) > 0:
                        train_loss_avg = sum(epoch_losses) / len(epoch_losses)
                        stability_metrics = self.stability_manager.check_and_adjust(
                            epoch=epoch,
                            train_loss=train_loss_avg,
                            val_loss=val_loss,
                            train_metrics={'map': 0.0},
                            val_metrics=val_metrics,
                        )
                        if stability_metrics.lr_adjusted or stability_metrics.wd_adjusted:
                            self.logger.info(
                                f"Stability adjustment: LR={stability_metrics.new_lr if stability_metrics.lr_adjusted else 'unchanged'}, "
                                f"WD={stability_metrics.new_wd if stability_metrics.wd_adjusted else 'unchanged'}"
                            )
                    
                    # Check if best model.
                    is_best = val_loss < self.best_val_loss or val_map > self.best_val_map
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                    if val_map > self.best_val_map:
                        self.best_val_map = val_map
                    
                    if self._check_early_stopping(epoch, val_loss, val_map):
                        break
                    
                    # Save checkpoint.
                    self._save_checkpoint(epoch, is_best=is_best)
                    # Optional: save numbered snapshot every N epochs.
                    if self.checkpoint_interval > 0 and (epoch + 1) % self.checkpoint_interval == 0:
                        self._save_checkpoint(epoch, is_best=False, extra_path=self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt")
                    
                    self.logger.info(
                        f"Train Loss: {train_metrics['loss']:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val mAP: {val_map:.4f}, "
                        f"Val mAP@0.5: {val_map_50:.4f}, Val mAP@0.75: {val_map_75:.4f}, "
                        f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}"
                    )
                else:
                    # No validation, just save checkpoint.
                    self._save_checkpoint(epoch, is_best=False)
                    if self.checkpoint_interval > 0 and (epoch + 1) % self.checkpoint_interval == 0:
                        self._save_checkpoint(epoch, is_best=False, extra_path=self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt")
                    self.logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user")
            self._save_checkpoint(self.current_epoch, is_best=False)
            raise
        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            self._save_checkpoint(self.current_epoch, is_best=False)
            raise
        
        if self.checkpoint_interval > 0 and (self.current_epoch + 1) % self.checkpoint_interval != 0:
            self._save_checkpoint(
                self.current_epoch,
                is_best=False,
                extra_path=self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch + 1}.pt",
            )
        
        elapsed_time = time.time() - start_time
        
        # Save training history.
        history_path = self.checkpoint_dir / 'training_history.json'
        try:
            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save training history: {e}")
        
        # Also save as CSV for easy plotting with pandas.
        try:
            import pandas as pd
            df = pd.DataFrame(self.history)
            csv_path = self.checkpoint_dir / 'training_history.csv'
            df.to_csv(csv_path, index=False)
        except ImportError:
            pass  # Pandas not available, skip CSV export.
        
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


# Convenience function.
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






