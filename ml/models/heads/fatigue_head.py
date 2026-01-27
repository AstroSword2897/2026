"""
Fatigue/Gaze Head for MaxSight Therapy System

Outputs fatigue score, blink rate, and fixation stability for adaptive assistance
and therapy task generation.

WHY FATIGUE TRACKING MATTERS:
-----------------------------
Fatigue and attention levels significantly impact how users interact with assistive
technology. This head enables:

1. Adaptive assistance: Adjust verbosity and detail based on fatigue levels
2. Therapy task adaptation: Modify task difficulty when user is fatigued
3. Safety monitoring: Detect when user needs rest or reduced cognitive load
4. Skill development: Track attention patterns to support vision therapy

Phase 2: Therapy Heads
See docs/therapy_system_implementation_plan.md for implementation details.
"""

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
        dropout: float = 0.1,
        use_lstm: bool = True,
        lstm_hidden_size: int = 32,
        lstm_num_layers: int = 2
    ):
        super().__init__()
        self.eye_dim = eye_dim
        self.temporal_dim = temporal_dim
        self.hidden_dim = hidden_dim
        self.use_lstm = use_lstm
        
        # Initial feature extraction (before LSTM)
        self.initial_net = nn.Sequential(
            nn.Linear(eye_dim + temporal_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # LSTM for temporal context (if enabled)
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_num_layers,
                batch_first=True,
                dropout=dropout if lstm_num_layers > 1 else 0.0
            )
            lstm_output_dim = lstm_hidden_size
        else:
            self.lstm = None
            lstm_output_dim = hidden_dim
        
        # Shared feature extraction after LSTM (if used)
        self.shared_net = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Task-specific heads
        head_input_dim = hidden_dim // 2
        self.fatigue_head = self._make_head(head_input_dim)
        self.blink_rate_head = self._make_head(head_input_dim)
        self.fixation_stability_head = self._make_head(head_input_dim)
        
        # LSTM hidden state (for temporal continuity)
        self.lstm_hidden = None
    
    def _make_head(self, input_dim: int) -> nn.Module:
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
        
        # Combine inputs
        combined = torch.cat([eye_features, temporal_features], dim=1)
        
        # Initial feature extraction
        initial_features = self.initial_net(combined)  # [B, hidden_dim]
        
        # Apply LSTM if enabled (adds temporal context)
        if self.use_lstm and self.lstm is not None:
            # Add sequence dimension: [B, hidden_dim] -> [B, 1, hidden_dim]
            initial_seq = initial_features.unsqueeze(1)  # [B, 1, hidden_dim]
            
            # LSTM forward pass
            lstm_out, self.lstm_hidden = self.lstm(initial_seq, self.lstm_hidden)
            
            # Remove sequence dimension: [B, 1, lstm_hidden_size] -> [B, lstm_hidden_size]
            lstm_features = lstm_out.squeeze(1)
        else:
            lstm_features = initial_features
        
        # Final shared feature extraction
        shared = self.shared_net(lstm_features)
        
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
