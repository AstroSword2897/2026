"""Global Confidence Aggregator for MaxSight 3.0 (v2)"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class GlobalConfidenceAggregator(nn.Module):
    """Global Confidence Aggregator (v2)."""
    
    def __init__(
        self,
        scene_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Linear(scene_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Logit space (NOT sigmoid yet)
        self.confidence_logit = nn.Linear(hidden_dim // 2, 1)
    
    def forward(
        self,
        scene_embedding: torch.Tensor,
        motion_residual: Optional[torch.Tensor] = None,
        ocr_entropy: Optional[torch.Tensor] = None,
        audio_entropy: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with multi-modal uncertainty aggregation."""
        x = self.backbone(scene_embedding)
        
        # Future: additive uncertainty penalties.
        if ocr_entropy is not None:
            x = x - ocr_entropy
        if audio_entropy is not None:
            x = x - audio_entropy
        
        confidence_logit = self.confidence_logit(x)
        confidence = torch.sigmoid(confidence_logit)
        uncertainty = 1.0 - confidence
        
        return {
            "global_confidence": confidence,
            "confidence_logit": confidence_logit,
            "uncertainty_score": uncertainty,  # Backward compatibility.
        }


# Backward compatibility alias.
UncertaintyHead = GlobalConfidenceAggregator






