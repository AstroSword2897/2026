"""Contrastive loss for personalization (metric learning)."""
import torch
import torch.nn.functional as F
from typing import Tuple


def compute_contrastive_loss(
    user_emb: torch.Tensor,  # [B, 256] normalized.
    object_emb: torch.Tensor,  # [B, K, 256] normalized.
    positive_mask: torch.Tensor,  # [B, K] binary.
    temperature: float = 0.1
) -> torch.Tensor:
    """Corrected InfoNCE contrastive loss for personalization."""
    B, K = object_emb.shape[:2]
    
    # Compute similarities: [B, K].
    similarity = torch.bmm(
        user_emb.unsqueeze(1),  # [B, 1, 256].
        object_emb.transpose(1, 2)  # [B, 256, K].
    ).squeeze(1) / temperature  # [B, K].
    
    # Treat each object independently as positive or negative.
    labels = positive_mask.float()  # [B, K].
    loss = F.binary_cross_entropy_with_logits(similarity, labels, reduction='mean')
    
    return loss







