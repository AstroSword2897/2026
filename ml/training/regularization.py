"""Regularization, Transfer Learning, and Class Weighting."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import Counter
import logging

logger = logging.getLogger(__name__)



class SpatialDropout2d(nn.Module):
    """Spatial dropout that drops entire channels."""
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
            
        # X shape: [B, C, H, W].
        mask = torch.ones(x.shape[0], x.shape[1], 1, 1, device=x.device)
        mask = F.dropout(mask, self.p, training=True)
        return x * mask


class DropConnect(nn.Module):
    """DropConnect - drops connections instead of activations."""
    
    def __init__(self, module: nn.Module, p: float = 0.5):
        super().__init__()
        self.module = module
        self.p = p
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return self.module(x)
            
        # For linear layers.
        if hasattr(self.module, 'weight'):
            mask = torch.ones_like(self.module.weight)
            mask = F.dropout(mask, self.p, training=True)
            # Temporarily modify weights.
            original_weight = self.module.weight.data.clone()
            self.module.weight.data = self.module.weight.data * mask
            output = self.module(x)
            self.module.weight.data = original_weight
            return output
            
        return self.module(x)


class StochasticDepth(nn.Module):
    """Stochastic depth - randomly drops entire layers."""
    
    def __init__(self, p: float = 0.1, mode: str = 'row'):
        super().__init__()
        self.p = p
        self.mode = mode
        
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x + residual
            
        if self.mode == 'row':
            # Different probability for each sample in batch.
            survival_rate = 1 - self.p
            if torch.rand(1).item() > survival_rate:
                return x
            return x + residual / survival_rate
        else:
            # Same probability for all.
            if torch.rand(1).item() < self.p:
                return x
            return x + residual / (1 - self.p)



class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing."""
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(-1)
        
        # One-hot encode targets.
        target_one_hot = F.one_hot(target, n_classes).float()
        
        # Apply label smoothing.
        target_smooth = target_one_hot * (1 - self.smoothing) + \
                       self.smoothing / n_classes
        
        log_probs = F.log_softmax(pred, dim=-1)
        loss = -(target_smooth * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance and hard examples. FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)"""
    
    def __init__(self, 
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha  # Per-class weights.
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = self.alpha.to(pred.device)[target]
            focal_weight = alpha_t * focal_weight
            
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class ClassWeightedLoss(nn.Module):
    """Cross entropy with class weights for imbalanced data."""
    
    def __init__(self, 
                 class_counts: Optional[Dict[int, int]] = None,
                 num_classes: int = 100,
                 weighting_strategy: str = 'inverse_freq'):
        super().__init__()
        self.num_classes = num_classes
        
        if class_counts:
            weights = self._compute_weights(class_counts, weighting_strategy)
            self.register_buffer('weights', weights)
        else:
            self.weights = None
            
    def _compute_weights(self, 
                        class_counts: Dict[int, int],
                        strategy: str) -> torch.Tensor:
        """Compute class weights from counts."""
        counts = torch.zeros(self.num_classes)
        for cls, count in class_counts.items():
            if cls < self.num_classes:
                counts[cls] = count
                
        # Avoid division by zero.
        counts = torch.clamp(counts, min=1)
        total = counts.sum()
        
        if strategy == 'inverse_freq':
            # Weight = total / (num_classes * count)
            weights = total / (self.num_classes * counts)
            
        elif strategy == 'inverse_sqrt':
            # Weight = sqrt(total / count)
            weights = torch.sqrt(total / counts)
            
        elif strategy == 'effective_samples':
            # Effective number of samples (CVPR 2019)
            beta = 0.9999
            effective_num = 1 - torch.pow(beta, counts)
            weights = (1 - beta) / effective_num
            
        else:
            weights = torch.ones(self.num_classes)
            
        # Normalize weights.
        weights = weights / weights.sum() * self.num_classes
        
        return weights
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(pred, target, weight=self.weights)
    
    def update_weights(self, class_counts: Dict[int, int], 
                      strategy: str = 'inverse_freq'):
        """Update weights based on new class counts."""
        self.weights = self._compute_weights(class_counts, strategy).to(
            next(self.parameters()).device if list(self.parameters()) else 'cpu')



class TransferLearningConfig:
    """Configuration for transfer learning."""
    
    def __init__(self,
                 backbone: str = 'resnet50',
                 pretrained: bool = True,
                 freeze_backbone: bool = True,
                 unfreeze_layers: int = 0,
                 gradual_unfreeze: bool = True,
                 unfreeze_schedule: Optional[Dict[int, int]] = None):
        self.backbone = backbone
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.unfreeze_layers = unfreeze_layers
        self.gradual_unfreeze = gradual_unfreeze
        self.unfreeze_schedule = unfreeze_schedule or {5: 2, 10: 4, 15: -1}


def load_pretrained_backbone(backbone_name: str = 'resnet50',
                            pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Load pretrained backbone for transfer learning. Returns: backbone module, output feature dimension."""
    try:
        import torchvision.models as models
    except ImportError:
        logger.warning("torchvision not available for pretrained models")
        return None, 0
        
    backbones = {
        'resnet18': (models.resnet18, 512),
        'resnet34': (models.resnet34, 512),
        'resnet50': (models.resnet50, 2048),
        'resnet101': (models.resnet101, 2048),
        'efficientnet_b0': (models.efficientnet_b0, 1280),
        'efficientnet_b4': (models.efficientnet_b4, 1792),
        'mobilenet_v3_large': (models.mobilenet_v3_large, 960),
        'mobilenet_v3_small': (models.mobilenet_v3_small, 576),
    }
    
    if backbone_name not in backbones:
        logger.warning(f"Unknown backbone: {backbone_name}, using resnet50")
        backbone_name = 'resnet50'
        
    model_fn, out_dim = backbones[backbone_name]
    
    # Load model.
    weights = 'IMAGENET1K_V1' if pretrained else None
    model = model_fn(weights=weights)
    
    # Remove classification head.
    if 'resnet' in backbone_name:
        backbone = nn.Sequential(*list(model.children())[:-2])  # Remove avgpool and fc.
    elif 'efficientnet' in backbone_name:
        backbone = model.features
    elif 'mobilenet' in backbone_name:
        backbone = model.features
    else:
        backbone = model
        
    return backbone, out_dim


def freeze_backbone(model: nn.Module, 
                   freeze: bool = True,
                   unfreeze_last_n: int = 0):
    """Freeze/unfreeze backbone parameters."""
    layers = list(model.children())
    n_layers = len(layers)
    
    for i, layer in enumerate(layers):
        should_freeze = freeze and (i < n_layers - unfreeze_last_n)
        for param in layer.parameters():
            param.requires_grad = not should_freeze
            
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    total = sum(1 for _ in model.parameters())
    logger.info(f"Frozen {frozen}/{total} parameters")


def gradual_unfreeze_step(model: nn.Module,
                         epoch: int,
                         schedule: Dict[int, int]):
    """Gradually unfreeze backbone layers according to schedule."""
    if epoch not in schedule:
        return
        
    layers_to_unfreeze = schedule[epoch]
    
    if layers_to_unfreeze == -1:
        # Unfreeze all.
        for param in model.parameters():
            param.requires_grad = True
        logger.info(f"Epoch {epoch}: Unfroze all layers")
    else:
        # Unfreeze last N layers.
        layers = list(model.children())
        for i, layer in enumerate(layers):
            if i >= len(layers) - layers_to_unfreeze:
                for param in layer.parameters():
                    param.requires_grad = True
        logger.info(f"Epoch {epoch}: Unfroze last {layers_to_unfreeze} layers")



def add_weight_decay(model: nn.Module,
                    weight_decay: float = 1e-5,
                    skip_list: Tuple[str, ...] = ('bias', 'bn', 'norm')) -> List[Dict]:
    """Create parameter groups with and without weight decay. Bias and batch norm parameters should not have weight decay."""
    decay = []
    no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if any(skip_name in name.lower() for skip_name in skip_list):
            no_decay.append(param)
        else:
            decay.append(param)
            
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ]


class WeightDecayScheduler:
    """Schedule weight decay during training."""
    
    def __init__(self,
                 optimizer: Optimizer,
                 initial_wd: float = 1e-5,
                 final_wd: float = 1e-4,
                 warmup_epochs: int = 5,
                 total_epochs: int = 100,
                 schedule: str = 'linear'):
        self.optimizer = optimizer
        self.initial_wd = initial_wd
        self.final_wd = final_wd
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.schedule = schedule
        
    def step(self, epoch: int):
        """Update weight decay for current epoch."""
        if epoch < self.warmup_epochs:
            wd = self.initial_wd
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            
            if self.schedule == 'linear':
                wd = self.initial_wd + (self.final_wd - self.initial_wd) * progress
            elif self.schedule == 'cosine':
                wd = self.final_wd + (self.initial_wd - self.final_wd) * \
                     (1 + np.cos(np.pi * progress)) / 2
            else:
                wd = self.initial_wd
                
        for param_group in self.optimizer.param_groups:
            if 'weight_decay' in param_group and param_group.get('weight_decay', 0) > 0:
                param_group['weight_decay'] = wd



class RegularizationManager:
    """Central manager for all regularization techniques."""
    
    def __init__(self,
                 model: nn.Module,
                 dropout_rate: float = 0.3,
                 weight_decay: float = 1e-5,
                 label_smoothing: float = 0.1,
                 use_focal_loss: bool = False,
                 focal_gamma: float = 2.0,
                 class_weights: Optional[Dict[int, int]] = None,
                 num_classes: int = 100):
        self.model = model
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        
        # Loss functions.
        if use_focal_loss:
            self.criterion = FocalLoss(gamma=focal_gamma)
        elif label_smoothing > 0:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        elif class_weights:
            self.criterion = ClassWeightedLoss(
                class_counts=class_weights,
                num_classes=num_classes
            )
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        # Add dropout to model if not present.
        self._add_dropout(model, dropout_rate)
        
    def _add_dropout(self, model: nn.Module, rate: float):
        """Add dropout layers after each major block."""
        for name, module in model.named_children():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Could wrap with dropout here.
                pass
                
    def get_optimizer_params(self) -> List[Dict]:
        """Get parameter groups with proper weight decay."""
        return add_weight_decay(self.model, self.weight_decay)
    
    def compute_loss(self, 
                    predictions: torch.Tensor,
                    targets: torch.Tensor) -> torch.Tensor:
        """Compute loss with all regularization."""
        return self.criterion(predictions, targets)
    
    def on_epoch_start(self, epoch: int, schedule: Optional[Dict[int, int]] = None):
        """Called at start of each epoch for gradual unfreeze."""
        if schedule:
            gradual_unfreeze_step(self.model, epoch, schedule)


def compute_class_weights_from_dataset(
    labels: List[int],
    num_classes: int,
    strategy: str = 'inverse_freq'
) -> torch.Tensor:
    """Compute class weights from dataset labels."""
    counts = Counter(labels)
    class_counts = {i: counts.get(i, 0) for i in range(num_classes)}
    
    loss_fn = ClassWeightedLoss(
        class_counts=class_counts,
        num_classes=num_classes,
        weighting_strategy=strategy
    )
    
    return loss_fn.weights







