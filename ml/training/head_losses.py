"""Unified loss interface for therapy and assistive heads. These losses train outputs that support visual rehabilitation and real-world assistive use (contrast, fatigue, motion, priority, depth, uncertainty)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from abc import ABC, abstractmethod


class HeadLoss(nn.Module, ABC):
    """Base for head losses; subclasses implement forward and return a dict with at least 'loss'."""
    
    @abstractmethod
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute loss for head predictions."""
        raise NotImplementedError


class ContrastLoss(HeadLoss):
    """Loss for contrast head with optional edge-aware weighting."""
    
    def __init__(self, use_edge_aware: bool = True, edge_weight: float = 0.5):
        super().__init__()
        self.use_edge_aware = use_edge_aware
        self.edge_weight = edge_weight
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute contrast loss with optional edge-aware weighting."""
        pred_contrast = predictions.get('contrast_map')
        target_contrast = targets.get('contrast_map')
        
        if pred_contrast is None or target_contrast is None:
            first_tensor = next(iter(predictions.values()))
            return {'loss': torch.zeros((), device=first_tensor.device, dtype=first_tensor.dtype)}
        pixel_loss = self.mse_loss(pred_contrast, target_contrast)
        if self.use_edge_aware and 'edge_map' in predictions:
            edge_map = predictions['edge_map']
            weighted_loss = pixel_loss * (1.0 + self.edge_weight * edge_map)
            loss = weighted_loss.mean()
        else:
            loss = pixel_loss.mean()
        
        return {
            'loss': loss,
            'pixel_loss': pixel_loss.mean(),
        }


class FatigueLoss(HeadLoss):
    """Fatigue head: fatigue score, blink rate, fixation stability; used to modulate rest and demand for user comfort."""
    
    def __init__(self, fatigue_weight: float = 1.0, blink_weight: float = 0.5, fixation_weight: float = 0.5):
        super().__init__()
        self.fatigue_weight = fatigue_weight
        self.blink_weight = blink_weight
        self.fixation_weight = fixation_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss for fatigue head."""
        fatigue_loss = self.mse_loss(
            predictions.get('fatigue_score', torch.tensor(0.0)),
            targets.get('fatigue_score', torch.tensor(0.0))
        )
        
        blink_loss = self.mse_loss(
            predictions.get('blink_rate', torch.tensor(0.0)),
            targets.get('blink_rate', torch.tensor(0.0))
        )
        
        fixation_loss = self.mse_loss(
            predictions.get('fixation_stability', torch.tensor(0.0)),
            targets.get('fixation_stability', torch.tensor(0.0))
        )
        
        total_loss = (
            self.fatigue_weight * fatigue_loss +
            self.blink_weight * blink_loss +
            self.fixation_weight * fixation_loss
        )
        
        return {
            'loss': total_loss,
            'fatigue_loss': fatigue_loss,
            'blink_loss': blink_loss,
            'fixation_loss': fixation_loss,
        }


class MotionLoss(HeadLoss):
    """Motion head: flow + smoothness. Reliable motion estimation supports stability and moving-hazard cues for users with visual impairment."""

    def __init__(
        self,
        flow_weight: float = 1.0,
        smoothness_weight: float = 0.1,
        use_edge_aware: bool = True,
        charbonnier_eps: float = 0.001
    ):
        super().__init__()
        self.flow_weight = flow_weight
        self.smoothness_weight = smoothness_weight
        self.use_edge_aware = use_edge_aware
        self.charbonnier_eps = charbonnier_eps
    
    def _charbonnier_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Charbonnier loss: sqrt(x^2 + eps^2) - eps (robust to outliers)."""
        return torch.sqrt(x ** 2 + self.charbonnier_eps ** 2) - self.charbonnier_eps
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute motion loss with edge-weighted smoothness."""
        pred_flow = predictions.get('flow')
        target_flow = targets.get('flow')
        
        if pred_flow is None or target_flow is None:
            first_tensor = next(iter(predictions.values()))
            return {'loss': torch.zeros((), device=first_tensor.device, dtype=first_tensor.dtype)}
        
        # Flow prediction loss (Charbonnier for robustness)
        flow_error = pred_flow - target_flow
        flow_loss = self._charbonnier_loss(flow_error).mean()
        
        # Smoothness loss (edge-weighted if image provided) Compute flow gradients.
        flow_grad_x = pred_flow[:, :, :, :-1] - pred_flow[:, :, :, 1:]  # [B, 2, H, W-1].
        flow_grad_y = pred_flow[:, :, :-1, :] - pred_flow[:, :, 1:, :]  # [B, 2, H-1, W].
        
        # Edge-aware weighting (if image provided)
        image = targets.get('image')
        if self.use_edge_aware and image is not None:
            # Compute image gradients for edge detection.
            img_grad_x = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]).mean(dim=1, keepdim=True)
            img_grad_y = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]).mean(dim=1, keepdim=True)
            
            # Weight smoothness inversely by image gradient (less penalty at edges)
            edge_weight_x = torch.exp(-img_grad_x * 10.0)  # Scale factor for edge sensitivity.
            edge_weight_y = torch.exp(-img_grad_y * 10.0)
            
            smoothness_x = (self._charbonnier_loss(flow_grad_x) * edge_weight_x).mean()
            smoothness_y = (self._charbonnier_loss(flow_grad_y) * edge_weight_y).mean()
        else:
            # Uniform smoothness (no edge weighting)
            smoothness_x = self._charbonnier_loss(flow_grad_x).mean()
            smoothness_y = self._charbonnier_loss(flow_grad_y).mean()
        
        smoothness_loss = smoothness_x + smoothness_y
        
        total_loss = self.flow_weight * flow_loss + self.smoothness_weight * smoothness_loss
        
        return {
            'loss': total_loss,
            'flow_loss': flow_loss,
            'smoothness_loss': smoothness_loss,
        }


class ROIPriorityLoss(HeadLoss):
    """ROI priority with ranking loss. Ensures the system emphasizes what matters most to the user (e.g. hazards first, then navigation cues) so attention and speech are used effectively."""

    def __init__(self, ranking_margin: float = 0.1, max_rois: int = 100):
        super().__init__()
        self.ranking_margin = ranking_margin
        self.max_rois = max_rois
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute score MSE plus pairwise ranking loss over ROIs."""
        pred_scores = predictions.get('roi_scores')
        target_scores = targets.get('roi_scores')
        target_rankings = targets.get('roi_rankings')
        if pred_scores is None or target_scores is None:
            first_tensor = next(iter(predictions.values()))
            return {'loss': torch.zeros((), device=first_tensor.device, dtype=first_tensor.dtype)}
        if len(pred_scores) > self.max_rois:
            pred_scores = pred_scores[:self.max_rois]
            target_scores = target_scores[:self.max_rois]
            if target_rankings is not None:
                target_rankings = target_rankings[:self.max_rois]
        score_loss = self.mse_loss(pred_scores, target_scores)
        ranking_loss = torch.zeros((), device=pred_scores.device, dtype=pred_scores.dtype)
        if target_rankings is not None and len(pred_scores) > 1:
            N = len(pred_scores)
            pred_diff = pred_scores.unsqueeze(1) - pred_scores.unsqueeze(0)
            rank_diff = target_rankings.unsqueeze(1) - target_rankings.unsqueeze(0)
            valid_mask = (rank_diff != 0) & ~torch.eye(N, dtype=torch.bool, device=pred_scores.device)
            if valid_mask.any():
                expected_sign = torch.sign(rank_diff[valid_mask])
                actual_sign = torch.sign(pred_diff[valid_mask])
                margin_violations = F.relu(
                    self.ranking_margin - pred_diff[valid_mask] * expected_sign
                )
                sign_mismatch = (expected_sign != actual_sign)
                ranking_loss = margin_violations[sign_mismatch].sum()
                num_valid_pairs = valid_mask.sum().float()
                if num_valid_pairs > 0:
                    ranking_loss = ranking_loss / num_valid_pairs
        
        total_loss = score_loss + ranking_loss
        
        return {
            'loss': total_loss,
            'score_loss': score_loss,
            'ranking_loss': ranking_loss,
        }


class DepthLoss(HeadLoss):
    """Uncertainty-weighted depth loss (Kendall & Gal formulation)."""
    
    def __init__(self, zone_weight: float = 0.5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.zone_weight = zone_weight
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute depth L1 with uncertainty weighting plus optional zone classification."""
        first_tensor = next(iter(predictions.values()))
        device = first_tensor.device
        dtype = first_tensor.dtype
        
        pred_depth = predictions.get('depth_map')
        pred_uncertainty = predictions.get('uncertainty')
        pred_zones = predictions.get('zones')
        
        target_depth = targets.get('depth_map')
        target_zones = targets.get('distance_zones')
        
        # Initialize losses with correct dtype.
        depth_loss = torch.zeros((), device=device, dtype=dtype)
        zone_loss = torch.zeros((), device=device, dtype=dtype)
        
        # Uncertainty-weighted depth: L = |d - d_gt|*exp(-u) + u.
        if pred_depth is not None and target_depth is not None:
            depth_error = torch.abs(pred_depth - target_depth)  # [B, H, W].
            
            if pred_uncertainty is not None:
                # Clamp uncertainty away from exactly 0 or 1 for numerical stability.
                uncertainty = torch.clamp(pred_uncertainty, min=1e-6, max=1.0 - 1e-6)
                
                depth_loss = (depth_error * torch.exp(-uncertainty) + uncertainty).mean()
            else:
                # Fallback: standard L1 loss if uncertainty not available.
                depth_loss = depth_error.mean()
        
        # Zone classification loss.
        if pred_zones is not None and target_zones is not None:
            zone_loss = self.ce_loss(pred_zones, target_zones.long()) * self.zone_weight
        
        total_loss = depth_loss + zone_loss
        
        return {
            'loss': total_loss,
            'depth_loss': depth_loss,
            'zone_loss': zone_loss,
        }


class UncertaintyLoss(HeadLoss):
    """Uncertainty head: MSE on predicted vs target uncertainty. Lets the system know when to trust or soften alerts so users are not overloaded with low-confidence cues."""

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute uncertainty prediction loss."""
        pred_uncertainty = predictions.get('uncertainty_score')
        target_uncertainty = targets.get('uncertainty_score')
        
        if pred_uncertainty is None or target_uncertainty is None:
            first_tensor = next(iter(predictions.values()))
            return {'loss': torch.zeros((), device=first_tensor.device, dtype=first_tensor.dtype)}
        
        loss = self.mse_loss(pred_uncertainty, target_uncertainty)
        
        return {
            'loss': loss,
        }


# Registry for head losses.
HEAD_LOSS_REGISTRY = {
    'contrast': ContrastLoss,
    'fatigue': FatigueLoss,
    'motion': MotionLoss,
    'roi_priority': ROIPriorityLoss,
    'depth': DepthLoss,
    'uncertainty': UncertaintyLoss,
}


def create_head_loss(head_type: str, **kwargs) -> HeadLoss:
    """Build a head loss by type (contrast, fatigue, motion, roi_priority, depth, uncertainty) for therapy/assistive training."""
    if head_type not in HEAD_LOSS_REGISTRY:
        available = ', '.join(HEAD_LOSS_REGISTRY.keys())
        raise ValueError(
            f"Unknown head loss type: '{head_type}'. "
            f"Available types: {available}"
        )
    
    return HEAD_LOSS_REGISTRY[head_type](**kwargs)


__all__ = [
    'HeadLoss',
    'ContrastLoss',
    'FatigueLoss',
    'MotionLoss',
    'ROIPriorityLoss',
    'DepthLoss',
    'UncertaintyLoss',
    'HEAD_LOSS_REGISTRY',
    'create_head_loss',
]







