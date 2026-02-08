"""Sound Event Classification Head for MaxSight 3.0 (v2)"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class SoundEventHead(nn.Module):
    """Sound event classification head (v2)."""
    
    def __init__(
        self,
        freq_bins: int = 128,
        num_classes: int = 15,
        num_directions: int = 4,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_directions = num_directions
        
        # Spectrogram CNN (time preserved)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((None, 16))  # [B, 64, T, 16].
        )
        
        self.cnn_proj = nn.Linear(64 * 16, embed_dim)
        
        # Temporal encoder.
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(embed_dim)
        
        # Learned temporal pooling.
        self.pool_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Heads (logits-first)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.direction_head = nn.Linear(embed_dim, num_directions)
        self.priority_head = nn.Linear(embed_dim, 1)
        
        # Urgency map (risk prior)
        self.register_buffer(
            "urgency_map",
            torch.tensor([
                3, 3, 3,     # Emergency / alarm / siren.
                2, 2,        # Construction / vehicle.
                1, 1, 1,     # Door / bell / footsteps.
                0, 0, 0, 0, 0, 0, 0
            ], dtype=torch.float)
        )
    
    def forward(self, spectrogram: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through sound event head."""
        B, T, freq_bins = spectrogram.shape
        
        # CNN.
        x = spectrogram.unsqueeze(1)  # [B, 1, T, freq_bins].
        x = self.cnn(x)               # [B, 64, T, 16].
        x = x.permute(0, 2, 1, 3).contiguous().reshape(B, T, -1)
        x = self.cnn_proj(x)          # [B, T, embed_dim].
        
        # Temporal attention.
        attn_out, _ = self.temporal_attn(x, x, x)
        x = self.temporal_norm(x + attn_out)
        
        # Attention pooling.
        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.temporal_attn(query, x, x)
        pooled = pooled.squeeze(1)  # [B, embed_dim].
        
        # Predictions (logits-first)
        class_logits = self.classifier(pooled)
        direction_logits = self.direction_head(pooled)
        priority = torch.sigmoid(self.priority_head(pooled))
        
        class_probs = F.softmax(class_logits, dim=1)
        
        # Expected urgency (uncertainty-aware)
        urgency = (class_probs * self.urgency_map).sum(dim=1, keepdim=True)
        
        return {
            "sound_logits": class_logits,
            "sound_probs": class_probs,
            "direction_logits": direction_logits,
            "direction": F.softmax(direction_logits, dim=1),  # For backward compatibility.
            "priority": priority,
            "urgency": urgency,
        }






