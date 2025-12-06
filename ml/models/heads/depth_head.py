"""
Depth/Focus Head

Outputs depth map and near/mid/far classification.

Phase 2: Therapy Heads
See docs/therapy_system_implementation_plan.md for implementation details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class DepthHead(nn.Module):
    """
    Depth/focus head for therapy tasks.
    
    Inputs: fused FPN + temporal features
    Output:
    - depth map [B, H, W]
    - near/mid/far classification [B, 3]
    
    Losses:
    - Photometric loss
    - Sparse synthetic depth loss
    """
    
    def __init__(self, in_channels: int = 256 + 128):  # FPN + temporal
        super().__init__()
        self.in_channels = in_channels
        
        # Depth estimation network
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.depth_conv = nn.Conv2d(64, 1, kernel_size=1)  # Depth map
        
        # Distance zone classification
        self.zone_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # near, mid, far
            nn.Softmax(dim=1)
        )
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Arguments:
            features: Fused FPN + temporal features [B, C, H, W]
        
        Returns:
            Dictionary with:
                - 'depth_map': [B, H, W] - Depth map
                - 'zones': [B, 3] - [near_prob, mid_prob, far_prob]
        """
        x = self.relu(self.conv1(features))
        x = self.relu(self.conv2(x))
        
        depth_map = self.sigmoid(self.depth_conv(x)).squeeze(1)  # [B, H, W]
        zones = self.zone_head(x).squeeze(-1).squeeze(-1)  # [B, 3]
        
        return {
            'depth_map': depth_map,
            'zones': zones
        }

