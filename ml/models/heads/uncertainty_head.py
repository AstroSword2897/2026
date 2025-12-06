import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

# Outputs uncertainty scores for model confidence estimation.


class UncertaintyHead(nn.Module):
    def __init__(self, scene_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.scene_dim = scene_dim

        self.fc1 = nn.Linear(scene_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.norm2 = nn.LayerNorm(hidden_dim // 2, 1)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout - nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, scene_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.relu(self.norm1(self.fc1(scene_embedding)))
        x = self.dropout(x)
        x = self.relu(self.norm2(self.fc2(x)))
        x = self.dropout(x)
        uncertainty = self.sigmoid(self.fc3(x))

        return {
            'uncertainty_score': uncertainty
        }