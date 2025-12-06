"""
ROI Priority Head for MaxSight Therapy System

Outputs ROI utility scores for prioritization and attention guidance.

PROJECT PHILOSOPHY & APPROACH:
This module implements ROI (Region of Interest) prioritization as part of MaxSight's
therapy system. Prioritizing regions helps users focus on important objects and
reduces information overload, especially critical for users with CVI or attention
difficulties.

WHY ROI PRIORITIZATION MATTERS:
ROI prioritization enables:

1. Attention guidance: Directs user attention to important objects
2. Information filtering: Reduces cognitive load by prioritizing relevant regions
3. Therapy task generation: Creates exercises that focus on high-priority regions
4. Adaptive assistance: Adjusts priority based on user needs and context

HOW IT CONNECTS TO THE PROBLEM STATEMENT:
The problem emphasizes "Clear Multimodal Communication" and "Environmental Structuring" -
ROI prioritization ensures users receive information about the most important regions
first, reducing cognitive overload while maintaining useful environmental awareness.

RELATIONSHIP TO BARRIER REMOVAL METHODS:
1. ENVIRONMENTAL STRUCTURING: Prioritizes regions for clearer understanding
2. CLEAR MULTIMODAL COMMUNICATION: Reduces information density while maintaining clarity
3. SKILL DEVELOPMENT: Helps users learn to identify important regions
4. ADAPTIVE ASSISTANCE: Adjusts priorities based on user needs and context

TECHNICAL DESIGN DECISIONS:
- Cross-attention: Enables scene-ROI interaction for context-aware prioritization
- LayerNorm + Dropout: Better generalization and training stability
- Pairwise ranking loss: Ensures correct priority ordering
- ROI masking: Supports variable number of regions per image

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class ROIPriorityHead(nn.Module):
    """
    ROI priority head for therapy tasks and attention guidance.
    
    WHY THIS CLASS EXISTS:
    Not all regions in an image are equally important. This head prioritizes regions
    based on scene context and ROI features, enabling the system to:
    - Guide user attention to important objects
    - Reduce information overload by filtering low-priority regions
    - Generate therapy exercises focused on high-priority regions
    - Adapt assistance based on region importance
    
    Architecture:
    - Input: Scene embedding [B, scene_dim] + ROI features [B, N, roi_dim]
    - Cross-attention (optional): Scene-ROI interaction for context-aware prioritization
    - Scorer: Combines scene and ROI features to generate utility scores
    - Output: ROI utility scores [B, N] in [0, 1] range (higher = more important)
    
    Arguments:
        scene_dim: Dimension of scene embedding (default: 256)
        roi_dim: Dimension of ROI features (default: 256)
        hidden_dim: Hidden layer dimension (default: 128)
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout probability (default: 0.1)
        use_attention: Enable cross-attention for scene-ROI interaction (default: True)
    """
    
    def __init__(
        self, 
        scene_dim: int = 256, 
        roi_dim: int = 256,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        """
        Initialize ROI priority head.
        
        Arguments:
            scene_dim: Dimension of scene embedding
            roi_dim: Dimension of ROI features
            hidden_dim: Hidden layer dimension for scorer
            num_heads: Number of attention heads for cross-attention
            dropout: Dropout probability for regularization
            use_attention: Enable cross-attention for context-aware prioritization
        """
        super().__init__()
        self.scene_dim = scene_dim
        self.roi_dim = roi_dim
        self.use_attention = use_attention
        
        # Optional cross-attention for scene-ROI interaction
        # WHY CROSS-ATTENTION:
        # - Enables scene context to influence ROI prioritization
        # - More expressive than simple concatenation
        # - Better for understanding which regions are important given scene context
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=roi_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.norm1 = nn.LayerNorm(roi_dim)  # Residual connection normalization
        
        # Feature fusion and scoring
        # WHY THIS ARCHITECTURE:
        # - Combines scene context with ROI features
        # - LayerNorm for better training stability
        # - Dropout for regularization
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
        
        self.sigmoid = nn.Sigmoid()  # Output in [0, 1] range
        
        # Scene context projection for attention
        if use_attention:
            self.scene_proj = nn.Linear(scene_dim, roi_dim)
    
    def forward(
        self,
        scene_embedding: torch.Tensor,
        roi_features: torch.Tensor,
        roi_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

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
        
        # Apply cross-attention if enabled
        if self.use_attention:
            # Project scene to query space
            scene_query = self.scene_proj(scene_embedding).unsqueeze(1)  # [B, 1, roi_dim]
            
            # Attend to ROI features
            # key_padding_mask: True = ignore (invalid ROI), False = attend (valid ROI)
            attn_mask = None if roi_mask is None else ~roi_mask  # Invert: True = invalid
            attended_rois, _ = self.attention(
                scene_query.expand(B, N, -1),  # Query: scene context for each ROI
                roi_features,  # Key: ROI features
                roi_features,  # Value: ROI features
                key_padding_mask=attn_mask
            )
            # Residual connection with normalization
            roi_features = self.norm1(roi_features + attended_rois)
        
        # Expand scene embedding to match each ROI
        scene_expanded = scene_embedding.unsqueeze(1).expand(B, N, -1)  # [B, N, scene_dim]
        
        # Concatenate scene context with ROI features
        combined = torch.cat([scene_expanded, roi_features], dim=2)  # [B, N, scene_dim + roi_dim]
        
        # Score each ROI
        scores = self.sigmoid(self.scorer(combined)).squeeze(-1)  # [B, N]
        
        # Apply mask if provided (set invalid ROIs to 0)
        if roi_mask is not None:
            scores = scores.masked_fill(~roi_mask, 0.0)
        
        # Validate output
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
        # Validate inputs
        if scores.shape != rankings.shape:
            raise ValueError(f"Shape mismatch: scores {scores.shape} vs rankings {rankings.shape}")
        
        B, N = scores.shape
        
        # Create pairwise comparisons
        # score_diff[i, j] = score[i] - score[j]
        score_diff = scores.unsqueeze(2) - scores.unsqueeze(1)  # [B, N, N]
        # rank_diff[i, j] = rank[i] - rank[j]
        rank_diff = rankings.unsqueeze(2) - rankings.unsqueeze(1)  # [B, N, N]
        
        # Loss when score order doesn't match rank order
        # Should have score_i > score_j when rank_i > rank_j
        # sign(rank_diff) = +1 if rank_i > rank_j, -1 if rank_i < rank_j
        # We want score_diff * sign(rank_diff) > margin
        loss = F.relu(margin - score_diff * torch.sign(rank_diff))
        
        # Only consider valid pairs (where rankings differ)
        valid_pairs = (rank_diff != 0).float()
        if valid_pairs.sum() > 0:
            loss = (loss * valid_pairs).sum() / valid_pairs.sum()
        else:
            loss = torch.tensor(0.0, device=scores.device)
        
        return loss
