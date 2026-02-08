"""ROI Priority Head for MaxSight Therapy System."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class ROIPriorityHead(nn.Module):
    """ROI priority head for therapy tasks and attention guidance."""
    
    def __init__(
        self, 
        scene_dim: int = 256, 
        roi_dim: int = 256,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        """Initialize ROI priority head."""
        super().__init__()
        self.scene_dim = scene_dim
        self.roi_dim = roi_dim
        self.use_attention = use_attention
        
        # Optional cross-attention for scene-ROI interaction.
        # WHY CROSS-ATTENTION:.
        # - Enables scene context to influence ROI prioritization.
        # - More expressive than simple concatenation.
        # - Better for understanding which regions are important given scene context.
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=roi_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.norm1 = nn.LayerNorm(roi_dim)  # Residual connection normalization.
        
        # Feature fusion and scoring.
        # WHY THIS ARCHITECTURE:.
        # - Combines scene context with ROI features.
        # - LayerNorm for better training stability.
        # - Dropout for regularization.
        input_dim = scene_dim + roi_dim
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.sigmoid = nn.Sigmoid()  # Output in [0, 1] range.
        
        # Scene context projection for attention.
        if use_attention:
            self.scene_proj = nn.Linear(scene_dim, roi_dim)
    
    def forward(
        self,
        scene_embedding: torch.Tensor,
        roi_features: torch.Tensor,
        roi_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass to generate ROI utility scores."""
        # Validate inputs.
        if scene_embedding.dim() != 2:
            raise ValueError(f"Expected 2D scene_embedding [B, scene_dim], got {scene_embedding.shape}")
        if roi_features.dim() != 3:
            raise ValueError(f"Expected 3D roi_features [B, N, roi_dim], got {roi_features.shape}")
        
        B, N, _ = roi_features.shape
        B_scene = scene_embedding.shape[0]
        if B != B_scene:
            raise ValueError(f"Batch size mismatch: scene {B_scene} vs roi {B}")
        
        if scene_embedding.shape[1] != self.scene_dim:
            raise ValueError(f"Expected scene_dim={self.scene_dim}, got {scene_embedding.shape[1]}")
        if roi_features.shape[2] != self.roi_dim:
            raise ValueError(f"Expected roi_dim={self.roi_dim}, got {roi_features.shape[2]}")
        
        # Apply cross-attention if enabled.
        if self.use_attention:
            # Project scene to query space.
            scene_query = self.scene_proj(scene_embedding).unsqueeze(1)  # [B, 1, roi_dim].
            
            # Attend to ROI features. Key_padding_mask: True = ignore (invalid ROI), False = attend (valid ROI)
            attn_mask = None if roi_mask is None else ~roi_mask  # Invert: True = invalid.
            attended_rois, _ = self.attention(
                scene_query.expand(B, N, -1),  # Query: scene context for each ROI.
                roi_features,  # Key: ROI features.
                roi_features,  # Value: ROI features.
                key_padding_mask=attn_mask
            )
            # Residual connection with normalization.
            roi_features = self.norm1(roi_features + attended_rois)
        
        # Expand scene embedding to match each ROI (efficient broadcasting)
        # Use unsqueeze + expand for memory efficiency (no copy until needed)
        scene_expanded = scene_embedding.unsqueeze(1).expand(B, N, -1)  # [B, N, scene_dim].
        
        # Concatenate scene context with ROI features (vectorized)
        combined = torch.cat([scene_expanded, roi_features], dim=2)  # [B, N, scene_dim + roi_dim].
        
        # Score each ROI (batched, efficient)
        scores = self.sigmoid(self.scorer(combined)).squeeze(-1)  # [B, N].
        
        # Apply mask if provided (set invalid ROIs to 0)
        if roi_mask is not None:
            scores = scores.masked_fill(~roi_mask, 0.0)
        
        # FIXED: Normalize scores per image as attention distribution.
        scores = scores / (scores.sum(dim=1, keepdim=True) + 1e-6)  # [B, N] normalized.
        
        # Validate output.
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            raise RuntimeError(
                "NaN/Inf detected in ROI scores. Check input features and model initialization."
            )
        
        return scores
    
    def compute_ranking_loss(
        self,
        scores: torch.Tensor,
        rankings: torch.Tensor,
        margin: float = 0.1
    ) -> torch.Tensor:
        """Compute pairwise ranking loss with proper tie handling."""
        # Validate inputs.
        if scores.shape != rankings.shape:
            raise ValueError(f"Shape mismatch: scores {scores.shape} vs rankings {rankings.shape}")
        
        B, N = scores.shape
        
        if N < 2:
            # Need at least 2 ROIs for ranking.
            return torch.tensor(0.0, device=scores.device)
        
        # Vectorized pairwise ranking loss. Compute all pairwise differences: score_diff[i,j] = score[i] - score[j].
        score_diff = scores.unsqueeze(2) - scores.unsqueeze(1)  # [B, N, N].
        rank_diff = rankings.unsqueeze(2) - rankings.unsqueeze(1)  # [B, N, N].
        
        # Loss when score order doesn't match rank order. We want: score_diff * sign(rank_diff) > margin.
        loss_matrix = F.relu(margin - score_diff * torch.sign(rank_diff))
        
        # Mask valid pairs (where rankings differ)
        valid_mask = (rank_diff != 0).float()
        valid_count = valid_mask.sum()
        
        if valid_count > 0:
            # Weight by rank difference magnitude.
            weights = torch.abs(rank_diff)
            loss = (loss_matrix * valid_mask * weights).sum() / valid_count
        else:
            loss = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
        
        return loss






