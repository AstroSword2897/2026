
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class ROIPriorityHead(nn.Module):
    """
    ROI priority head for therapy tasks and attention guidance.
    
    WHY THIS CLASS EXISTS:
    ----------------------
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
        """
        Forward pass to generate ROI utility scores.
        
        CRITICAL INPUT REQUIREMENTS:
        ----------------------------
        - scene_embedding: Must be [B, scene_dim] from scene embedding
        - roi_features: Must be [B, N, roi_dim] where N is number of ROIs
        - roi_mask: Optional [B, N] boolean mask (True = valid ROI)
        - All inputs must be on same device and have same batch size
        
        Arguments:
            scene_embedding: Scene embedding [B, scene_dim]
            roi_features: ROI features [B, N, roi_dim]
            roi_mask: Optional mask for valid ROIs [B, N] (True = valid, False = invalid)
        
        Returns:
            ROI utility scores [B, N] in [0, 1] range (higher = more important)
        """
        # Validate inputs
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
        
        # Expand scene embedding to match each ROI (efficient broadcasting)
        # Use unsqueeze + expand for memory efficiency (no copy until needed)
        scene_expanded = scene_embedding.unsqueeze(1).expand(B, N, -1)  # [B, N, scene_dim]
        
        # Concatenate scene context with ROI features (vectorized)
        combined = torch.cat([scene_expanded, roi_features], dim=2)  # [B, N, scene_dim + roi_dim]
        
        # Score each ROI (batched, efficient)
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
        """
        Compute pairwise ranking loss with proper tie handling.
        
        WHY RANKING LOSS:
        ----------------
        ROI prioritization is about relative importance, not absolute scores. Ranking
        loss ensures that ROIs with higher ground truth rankings get higher predicted
        scores, which is more appropriate than regression loss for this task.
        
        Arguments:
            scores: Predicted scores [B, N]
            rankings: Ground truth rankings [B, N] (higher = more important)
            margin: Margin for ranking loss (default: 0.1)
        
        Returns:
            Ranking loss scalar
        """
        # Validate inputs
        if scores.shape != rankings.shape:
            raise ValueError(f"Shape mismatch: scores {scores.shape} vs rankings {rankings.shape}")
        
        B, N = scores.shape
        
        if N < 2:
            # Need at least 2 ROIs for ranking
            return torch.tensor(0.0, device=scores.device)
        
        total_loss = torch.tensor(0.0, device=scores.device)
        valid_pairs = 0
        
        # Process each sample in batch
        for b in range(B):
            batch_scores = scores[b]  # [N]
            batch_rankings = rankings[b]  # [N]
            
            # Generate all pairs
            for i in range(N):
                for j in range(i + 1, N):
                    rank_diff = batch_rankings[i] - batch_rankings[j]
                    score_diff = batch_scores[i] - batch_scores[j]
                    
                    # Only consider pairs where rankings differ (exclude ties)
                    if abs(rank_diff) > 1e-6:  # Not a tie
                        valid_pairs += 1
                        
                        # Expected: sign(score_diff) == sign(rank_diff)
                        # Loss when order doesn't match
                        expected_sign = torch.sign(rank_diff)
                        loss = torch.clamp(margin - score_diff * expected_sign, min=0.0)
                        
                        # Weight by rank difference magnitude (larger differences more important)
                        loss = loss * abs(rank_diff)
                        total_loss += loss
        
        # Average over valid pairs
        if valid_pairs > 0:
            return total_loss / valid_pairs
        else:
            return torch.tensor(0.0, device=scores.device)
        
        # Optimized pairwise comparisons (vectorized)
        # score_diff[i, j] = score[i] - score[j]
        score_diff = scores.unsqueeze(2) - scores.unsqueeze(1)  # [B, N, N]
        # rank_diff[i, j] = rank[i] - rank[j]
        rank_diff = rankings.unsqueeze(2) - rankings.unsqueeze(1)  # [B, N, N]
        
        # Loss when score order doesn't match rank order (vectorized)
        # Should have score_i > score_j when rank_i > rank_j
        # sign(rank_diff) = +1 if rank_i > rank_j, -1 if rank_i < rank_j
        # We want score_diff * sign(rank_diff) > margin
        loss_matrix = F.relu(margin - score_diff * torch.sign(rank_diff))
        
        # Only consider valid pairs (where rankings differ) - efficient masking
        valid_pairs = (rank_diff != 0).float()
        valid_count = valid_pairs.sum()
        
        if valid_count > 0:
            # Efficient: sum only valid pairs, divide by count
            loss = (loss_matrix * valid_pairs).sum() / valid_count
        else:
            loss = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
        
        return loss
