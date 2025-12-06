
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class FatigueHead(nn.Module):
    
    def __init__(
        self, 
        eye_dim: int = 4, 
        temporal_dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.eye_dim = eye_dim
        self.temporal_dim = temporal_dim
        self.hidden_dim = hidden_dim
        
        # Shared feature extraction with dropout for regularization
        # WHY SHARED BACKBONE:
        # - Reduces parameters (more efficient)
        # - Encourages learning common features across tasks
        # - Better generalization with shared representations
        self.shared_net = nn.Sequential(
            nn.Linear(eye_dim + temporal_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Better than BatchNorm for variable batch sizes
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),  # Regularization to prevent overfitting
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Task-specific heads with residual connections
        # WHY TASK-SPECIFIC HEADS:
        # - Allows fine-tuning for each output while sharing features
        # - More expressive than single shared head
        head_input_dim = hidden_dim // 2
        self.fatigue_head = self._make_head(head_input_dim)
        self.blink_rate_head = self._make_head(head_input_dim)
        self.fixation_stability_head = self._make_head(head_input_dim)
    
    def _make_head(self, input_dim: int) -> nn.Module:
        """
        Create a task-specific head with additional capacity.
        
        Arguments:
            input_dim: Input dimension for the head
        
        Returns:
            Sequential module for task-specific prediction
        """
        return nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
    
    def forward(
        self,
        eye_features: torch.Tensor,
        temporal_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Validate inputs
        if eye_features.dim() != 2:
            raise ValueError(f"Expected 2D eye_features [B, eye_dim], got {eye_features.shape}")
        if temporal_features.dim() != 2:
            raise ValueError(f"Expected 2D temporal_features [B, temporal_dim], got {temporal_features.shape}")
        
        B_eye = eye_features.shape[0]
        B_temp = temporal_features.shape[0]
        if B_eye != B_temp:
            raise ValueError(f"Batch size mismatch: eye_features {B_eye} vs temporal_features {B_temp}")
        
        if eye_features.shape[1] != self.eye_dim:
            raise ValueError(f"Expected eye_dim={self.eye_dim}, got {eye_features.shape[1]}")
        if temporal_features.shape[1] != self.temporal_dim:
            raise ValueError(f"Expected temporal_dim={self.temporal_dim}, got {temporal_features.shape[1]}")
        
        # Combine and extract shared features
        combined = torch.cat([eye_features, temporal_features], dim=1)
        shared = self.shared_net(combined)
        
        # Validate shared features
        if torch.isnan(shared).any() or torch.isinf(shared).any():
            raise RuntimeError(
                "NaN/Inf detected in shared features. Check input features and model initialization."
            )
        
        # Generate predictions
        outputs = {
            'fatigue_score': self.fatigue_head(shared),
            'blink_rate': self.blink_rate_head(shared),
            'fixation_stability': self.fixation_stability_head(shared),
            'shared_features': shared  # Useful for analysis/visualization
        }
        
        # Validate outputs
        for key, value in outputs.items():
            if torch.isnan(value).any() or torch.isinf(value).any():
                raise RuntimeError(f"NaN/Inf detected in {key}. Check model initialization.")
        
        return outputs
