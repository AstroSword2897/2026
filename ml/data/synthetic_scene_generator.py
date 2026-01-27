"""Synthetic Scene Generator for MaxSight 3.0."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class SyntheticSceneGenerator(nn.Module):
    """GAN-based synthetic scene generation for rare scenarios."""
    
    def __init__(self, latent_dim: int = 128, output_size: Tuple[int, int] = (224, 224)):
        super().__init__()
        self.latent_dim = latent_dim
        self._output_size = output_size
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size[0] * output_size[1] * 3),
            nn.Tanh()
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate synthetic scene."""
        return self.generator(z).reshape(-1, 3, *self.output_size)
    
    @property
    def output_size(self):
        """Get output size."""
        return self._output_size

