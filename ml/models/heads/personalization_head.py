"""Personalization Head for MaxSight 3.0 (v2)"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PersonalizationHead(nn.Module):
    """Personalization head for user-specific adaptation (v2)."""
    
    def __init__(
        self,
        input_dim: int = 512,
        num_users: int = 10_000,
        num_features: int = 10,  # Number of attention features.
        num_alert_types: int = 5,
        embed_dim: int = 256,
        interaction_dim: int = 64,
        crisp: float = 1.0  # Temperature scaling for attention/alert sharpness.
    ):
        super().__init__()
        
        self.num_features = num_features
        self.num_alert_types = num_alert_types
        
        # Crisp parameter: Controls softmax distribution sharpness.
        # Crisp < 1.0: Sharper distribution (more confident, focused on top choice)
        # Example: crisp=0.5 makes [0.1, 0.2, 0.7] → [0.05, 0.15, 0.80].
        # Crisp = 1.0: Standard softmax (default, no scaling)
        # Crisp > 1.0: Softer distribution (less confident, more uniform)
        # Example: crisp=2.0 makes [0.1, 0.2, 0.7] → [0.15, 0.25, 0.60].
        # 
        # Use cases:.
        # - Lower crisp (0.3-0.7): When user wants confident, focused attention/alerts.
        # - Higher crisp (1.5-3.0): When user wants exploratory, distributed behavior.
        # - Can be learned or set per-user based on preferences.
        self.crisp = crisp
        
        # Persistent user representation (scales to real deployments)
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.02)
        
        # Contextual adaptation (gated residual update - stable)
        self.adaptation_gate = nn.Sequential(
            nn.Linear(embed_dim + interaction_dim, embed_dim),
            nn.Sigmoid()
        )
        
        self.adaptation_delta = nn.Sequential(
            nn.Linear(embed_dim + interaction_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Shared fusion trunk.
        fused_dim = input_dim + embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU()
        )
        
        # Heads (logit-first design)
        self.attention_head = nn.Linear(512, num_features)
        self.verbosity_head = nn.Linear(512, 4)  # Logits (0-3)
        self.alert_head = nn.Linear(512, num_alert_types)
        
    def forward(
        self,
        scene_features: torch.Tensor,        # [B, input_dim].
        user_id: torch.LongTensor,            # [B].
        interaction_features: Optional[torch.Tensor] = None  # [B, interaction_dim].
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for personalization (v2)."""
        B = scene_features.size(0)
        
        # Base user embedding.
        user_emb = self.user_embedding(user_id)  # [B, embed_dim].
        
        # Contextual adaptation (optional, gated)
        if interaction_features is not None:
            adapt_input = torch.cat([user_emb, interaction_features], dim=1)
            
            gate = self.adaptation_gate(adapt_input)
            delta = self.adaptation_delta(adapt_input)
            
            user_emb = user_emb + gate * delta  # Stable residual update.
        
        # Fuse scene + user.
        fused = torch.cat([scene_features, user_emb], dim=1)
        fused = self.fusion(fused)
        
        # Outputs (LOGITS FIRST - preferred for training)
        attention_logits = self.attention_head(fused)
        alert_logits = self.alert_head(fused)
        verbosity_logits = self.verbosity_head(fused)
        
        return {
            # Raw logits (preferred for training)
            'attention_logits': attention_logits,
            'alert_priority_logits': alert_logits,
            'verbosity_logits': verbosity_logits,
            
            # Normalized (for inference-time use) Apply crisp scaling: lower = sharper/focused, higher = softer/distributed.
            'attention_weights': torch.softmax(
                attention_logits / self.crisp, dim=1
            ),
            'alert_priority_weights': torch.softmax(
                alert_logits / self.crisp, dim=1
            ),
            
            # Discrete verbosity level.
            'verbosity_level': verbosity_logits.argmax(dim=1),
            
            # Diagnostics.
            'user_embedding': user_emb
        }
    








