"""Per-head loss definitions for MaxSight. Each head supports assistive outputs: what is in the scene, where it is, how urgent, and at what distance—so the system can guide users with low vision or blindness safely."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class ObjectnessLoss(nn.Module):
    """Objectness: BCE + focal loss for binary presence at each location. Ensures the model reliably detects that something is there—foundation for scene awareness for users with low vision."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward: [B, N] logits vs [B, N] binary labels."""
        bce_loss = self.bce(predictions, targets)
        p_t = torch.sigmoid(predictions)
        p_t = torch.where(targets == 1, p_t, 1 - p_t)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        loss = focal_weight * bce_loss
        return loss.mean()


class ClassificationLoss(nn.Module):
    """Classification head loss. Ground Truth: Class indices [0, num_classes-1] per location Loss: Focal loss for class imbalance."""
    
    def __init__(self, num_classes: int, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward: [B, N, C] logits vs [B, N] class indices."""
        ce_loss = F.cross_entropy(
            predictions.reshape(-1, self.num_classes),
            targets.reshape(-1),
            reduction='none'
        )
        
        p = F.softmax(predictions, dim=-1)
        p_t = p.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss
        return loss.mean()


class BoxRegressionLoss(nn.Module):
    """Box regression: smooth L1 on normalized [cx, cy, w, h] boxes. Accurate location is needed to highlight or announce where hazards and objects are in the user’s field of view."""

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward: [B, N, 4] pred vs [B, N, 4] target boxes."""
        diff = predictions - targets
        abs_diff = torch.abs(diff)
        smooth_l1 = torch.where(
            abs_diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            abs_diff - 0.5 * self.beta
        )
        return smooth_l1.mean()


class DistanceZoneLoss(nn.Module):
    """Distance zone: CE over zone indices (0=near, 1=medium, 2=far)."""

    def __init__(self, num_zones: int = 3, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_zones = num_zones
        if class_weights is None:
            class_weights = torch.ones(num_zones)
        self.register_buffer('class_weights', class_weights)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward: [B, N, num_zones] logits vs [B, N] zone indices (-1=ignore). Returns 0 if no valid targets."""
        pred_flat = predictions.reshape(-1, self.num_zones)
        tgt_flat = targets.reshape(-1).long()
        valid = (tgt_flat >= 0) & (tgt_flat < self.num_zones)
        if valid.sum() == 0:
            return torch.zeros((), device=predictions.device, dtype=predictions.dtype)
        
        ce_loss = F.cross_entropy(
            pred_flat[valid],
            tgt_flat[valid],
            weight=self.class_weights,
            reduction='mean'
        )
        if torch.isnan(ce_loss).any() or torch.isinf(ce_loss).any():
            return torch.zeros((), device=predictions.device, dtype=predictions.dtype)
        return ce_loss


class UrgencyLoss(nn.Module):
    """Urgency: focal CE with higher weight for danger levels. Prioritizing hazards (e.g. moving vehicles, drop-offs) is critical for user safety."""

    def __init__(self, num_levels: int = 4, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.num_levels = num_levels
        self.alpha = alpha
        self.gamma = gamma
        # Weight danger higher.
        self.register_buffer('class_weights', torch.tensor([1.0, 1.5, 2.0, 3.0]))
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward: [B, num_levels] logits vs [B] level indices (-1=ignore). Returns 0 if no valid targets."""
        targets = targets.long()
        valid = (targets >= 0) & (targets < self.num_levels)
        if valid.sum() == 0:
            return torch.zeros((), device=predictions.device, dtype=predictions.dtype)
        pred_valid = predictions[valid]
        tgt_valid = targets[valid]
        ce_loss = F.cross_entropy(
            pred_valid,
            tgt_valid,
            weight=self.class_weights,
            reduction='none'
        )
        p = F.softmax(pred_valid, dim=-1)
        p_t = p.gather(1, tgt_valid.unsqueeze(-1)).squeeze(-1)
        focal_weight = self.alpha * (1 - p_t).clamp(min=1e-6) ** self.gamma
        loss = (focal_weight * ce_loss).mean()
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            return torch.zeros((), device=predictions.device, dtype=predictions.dtype)
        return loss


class UncertaintyLoss(nn.Module):
    """Uncertainty: MSE on variance target (self-supervised). Model confidence helps the system know when to trust an alert vs. defer or ask for confirmation."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        prediction_variance: torch.Tensor
    ) -> torch.Tensor:
        """Forward: pred variance vs optional target; uses pred variance as target if None."""
        if targets is None:
            targets = prediction_variance
        
        return self.mse(predictions, targets)


class DepthLoss(nn.Module):
    """Depth: L1 with optional inverse-uncertainty weighting. Depth supports spatial awareness and collision risk for users with reduced depth perception."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='none')
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainty: torch.Tensor
    ) -> torch.Tensor:
        """Args predictions: [B, H, W] predicted depth targets: [B, H, W] ground truth depth uncertainty: [B, H, W] depth uncertainty (for weighting)"""
        l1_loss = self.l1(predictions, targets)
        # Weight by inverse uncertainty (high uncertainty = low weight)
        weights = 1.0 / (uncertainty + 1e-6)
        weighted_loss = weights * l1_loss
        return weighted_loss.mean()


class MotionLoss(nn.Module):
    """Motion: L2 loss on flow vectors."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args predictions: [B, 2, H, W] or [B, H, W] motion predictions targets: [B, 2, H, W] or [B, H, W] ground truth motion."""
        return self.mse(predictions, targets)


class SceneDescriptionLoss(nn.Module):
    """Scene description: CE over token sequence. Verbal scene descriptions support blind and low-vision users who rely on spoken output."""

    def __init__(self, vocab_size: int, ignore_index: int = -100):
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward: [B, seq_len, vocab] logits vs [B, seq_len] token indices."""
        return self.ce(predictions.reshape(-1, self.vocab_size), targets.reshape(-1))


class OCRLoss(nn.Module):
    """OCR head loss (Tier 3+). Ground Truth: Text detections with bounding boxes and text Loss: Detection loss (boxes) + Recognition loss (text)"""
    
    def __init__(self, vocab_size: int):
        super().__init__()
        self.box_loss = BoxRegressionLoss()
        self.text_loss = SceneDescriptionLoss(vocab_size)
    
    def forward(
        self,
        box_predictions: torch.Tensor,
        box_targets: torch.Tensor,
        text_predictions: torch.Tensor,
        text_targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward: box pred/target and text logits/targets."""
        box_loss = self.box_loss(box_predictions, box_targets)
        text_loss = self.text_loss(text_predictions, text_targets)
        return {
            'box_loss': box_loss,
            'text_loss': text_loss,
            'total_loss': box_loss + text_loss
        }


class FatigueLoss(nn.Module):
    """Fatigue head loss (Tier 5+). Ground Truth: Binary fatigue labels (0=not fatigued, 1=fatigued) Loss: Binary cross-entropy."""
    
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward: [B, 1] logits vs [B, 1] binary labels."""
        return self.bce(predictions, targets)


class MultiHeadLoss(nn.Module):
    """Combined multi-head loss with optional per-head weights. Balances detection, safety (urgency/distance), and accessibility (description/OCR) so the assistive system performs well end-to-end."""

    def __init__(
        self,
        loss_functions: Dict[str, nn.Module],
        loss_weights: Optional[Dict[str, float]] = None,
        use_gradnorm: bool = False,
    ):
        super().__init__()
        self.loss_functions = loss_functions
        self.use_gradnorm = use_gradnorm
        if loss_weights is None:
            loss_weights = {
                'objectness': 1.0,
                'classification': 1.0,
                'box': 1.0,
                'distance': 0.5,
                'urgency': 2.0,  # Higher weight for safety.
                'uncertainty': 0.5,
                'depth': 1.0,
                'motion': 0.3,
                'scene_description': 0.2,
                'ocr': 0.2,
                'fatigue': 0.1
            }
        self.register_buffer('loss_weights', torch.tensor([
            loss_weights.get(name, 1.0) for name in loss_functions.keys()
        ]))
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute losses for all heads."""
        losses = {}
        total_loss = 0.0
        
        for i, (head_name, loss_fn) in enumerate(self.loss_functions.items()):
            if head_name in predictions and head_name in targets:
                loss = loss_fn(predictions[head_name], targets[head_name])
                weighted_loss = self.loss_weights[i] * loss
                losses[head_name] = loss
                total_loss += weighted_loss
        
        losses['total_loss'] = total_loss
        return losses


GROUND_TRUTH_SOURCES = {
    'objectness': 'COCO annotations (binary object presence)',
    'classification': 'COCO class labels',
    'box': 'COCO bounding boxes (normalized)',
    'distance': 'Synthetic/calibrated distance zones',
    'urgency': 'Expert-annotated urgency labels',
    'uncertainty': 'Derived from prediction variance (self-supervised)',
    'depth': 'Depth sensors (RGB-D) or stereo vision',
    'motion': 'Optical flow (Lucas-Kanade or deep flow)',
    'scene_description': 'Human-annotated scene descriptions',
    'ocr': 'Text detection datasets (ICDAR, COCO-Text)',
    'fatigue': 'User interaction logs (time-based, interaction patterns)',
}






