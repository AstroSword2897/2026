"""
Unified loss interface for therapy heads.

Provides a base class for head-specific losses and implementations for each head type.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from abc import ABC, abstractmethod

class HeadLoss(nn.Module, ABC):
    """
    Base class for head-specific losses.
    
    All head losses should inherit from this class and implement the forward method
    that returns a dictionary with at least a 'loss' key.
    """
    
    @abstractmethod
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for head predictions.
        
        Arguments:
            predictions: Dictionary of model predictions
            targets: Dictionary of target values
        
        Returns:
            Dictionary with at least 'loss' key and optional component losses
        """
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
            return {'loss': torch.tensor(0.0, device=next(iter(predictions.values())).device)}
        
        # Pixel-wise MSE
        pixel_loss = self.mse_loss(pred_contrast, target_contrast)
        
        # Edge-aware weighting if enabled
        if self.use_edge_aware and 'edge_map' in predictions:
            edge_map = predictions['edge_map']
            # Weight loss by edge strength (higher weight at edges)
            weighted_loss = pixel_loss * (1.0 + self.edge_weight * edge_map)
            loss = weighted_loss.mean()
        else:
            loss = pixel_loss.mean()
        
        return {
            'loss': loss,
            'pixel_loss': pixel_loss.mean(),
        }

class FatigueLoss(HeadLoss):
    """Loss for fatigue head (fatigue, blink rate, fixation stability)."""
    
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
    """Loss for motion head with smoothness regularization."""
    
    def __init__(
        self,
        flow_weight: float = 1.0,
        smoothness_weight: float = 0.1,
        use_edge_aware: bool = True
    ):
        super().__init__()
        self.flow_weight = flow_weight
        self.smoothness_weight = smoothness_weight
        self.use_edge_aware = use_edge_aware
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute motion loss with smoothness regularization."""
        pred_flow = predictions.get('flow')
        target_flow = targets.get('flow')
        
        if pred_flow is None or target_flow is None:
            return {'loss': torch.tensor(0.0, device=next(iter(predictions.values())).device)}
        
        # Flow prediction loss
        flow_loss = self.mse_loss(pred_flow, target_flow)
        
        # Smoothness loss (edge-aware if image provided)
        image = targets.get('image')
        # Note: This requires the head's compute_smoothness_loss method
        # For now, use simple smoothness
        flow_grad_x = torch.abs(pred_flow[:, :, :, :-1] - pred_flow[:, :, :, 1:])
        flow_grad_y = torch.abs(pred_flow[:, :, :-1, :] - pred_flow[:, :, 1:, :])
        smoothness_loss = flow_grad_x.mean() + flow_grad_y.mean()
        
        total_loss = self.flow_weight * flow_loss + self.smoothness_weight * smoothness_loss
        
        return {
            'loss': total_loss,
            'flow_loss': flow_loss,
            'smoothness_loss': smoothness_loss,
        }

class ROIPriorityLoss(HeadLoss):
    """Loss for ROI priority head with ranking loss."""
    
    def __init__(self, ranking_margin: float = 0.1):
        super().__init__()
        self.ranking_margin = ranking_margin
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute ROI priority loss with ranking."""
        pred_scores = predictions.get('roi_scores')
        target_scores = targets.get('roi_scores')
        target_rankings = targets.get('roi_rankings')
        
        if pred_scores is None or target_scores is None:
            return {'loss': torch.tensor(0.0, device=next(iter(predictions.values())).device)}
        
        # MSE loss for score prediction
        score_loss = self.mse_loss(pred_scores, target_scores)
        
        # Ranking loss if rankings provided
        ranking_loss = torch.tensor(0.0, device=pred_scores.device)
        if target_rankings is not None and len(pred_scores) > 1:
            # Pairwise ranking loss
            for i in range(len(pred_scores)):
                for j in range(i + 1, len(pred_scores)):
                    rank_diff = target_rankings[i] - target_rankings[j]
                    score_diff = pred_scores[i] - pred_scores[j]
                    
                    # Only penalize if ranking order doesn't match score order
                    if rank_diff != 0:  # Exclude ties
                        expected_sign = torch.sign(rank_diff)
                        actual_sign = torch.sign(score_diff)
                        
                        if expected_sign != actual_sign:
                            ranking_loss += F.relu(self.ranking_margin - score_diff * expected_sign)
            
            if len(pred_scores) > 1:
                ranking_loss = ranking_loss / (len(pred_scores) * (len(pred_scores) - 1) / 2)
        
        total_loss = score_loss + ranking_loss
        
        return {
            'loss': total_loss,
            'score_loss': score_loss,
            'ranking_loss': ranking_loss,
        }

class DepthLoss(HeadLoss):
    """Loss for depth head."""
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute depth loss."""
        pred_depth = predictions.get('depth_map')
        target_depth = targets.get('depth_map')
        pred_zones = predictions.get('distance_zones')
        target_zones = targets.get('distance_zones')
        
        depth_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        zone_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        if pred_depth is not None and target_depth is not None:
            depth_loss = self.mse_loss(pred_depth, target_depth)
        
        if pred_zones is not None and target_zones is not None:
            zone_loss = self.ce_loss(pred_zones, target_zones.long())
        
        total_loss = depth_loss + zone_loss
        
        return {
            'loss': total_loss,
            'depth_loss': depth_loss,
            'zone_loss': zone_loss,
        }

class UncertaintyLoss(HeadLoss):
    """Loss for uncertainty head."""
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute uncertainty loss."""
        pred_uncertainty = predictions.get('uncertainty_score')
        target_uncertainty = targets.get('uncertainty_score')
        
        if pred_uncertainty is None or target_uncertainty is None:
            return {'loss': torch.tensor(0.0, device=next(iter(predictions.values())).device)}
        
        loss = self.mse_loss(pred_uncertainty, target_uncertainty)
        
        return {
            'loss': loss,
        }

# Registry for head losses
HEAD_LOSS_REGISTRY = {
    'contrast': ContrastLoss,
    'fatigue': FatigueLoss,
    'motion': MotionLoss,
    'roi_priority': ROIPriorityLoss,
    'depth': DepthLoss,
    'uncertainty': UncertaintyLoss,
}

def create_head_loss(head_type: str, **kwargs) -> HeadLoss:
    """
    Create a head loss by type name.
    
    Arguments:
        head_type: Type of head loss ('contrast', 'fatigue', 'motion', etc.)
        **kwargs: Arguments to pass to loss constructor
    
    Returns:
        HeadLoss instance
    
    Raises:
        ValueError: If head_type is not in registry
    """
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

