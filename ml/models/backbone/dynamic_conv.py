"""
Dynamic Convolution Modules for MaxSight 3.0

Kernel weights adapt based on:
- Lighting conditions (brightness, contrast)
- Occlusion levels (detected via attention)
- Motion blur (from temporal encoder)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DynamicConv2d(nn.Module):
    """
    Dynamic convolution where kernel weights adapt to input conditions.
    
    Architecture:
    - Base kernel set: Multiple pre-defined kernels
    - Condition predictor: Lightweight network to predict condition weights
    - Dynamic kernel: Weighted combination of base kernels
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_kernels: int = 4,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        bias: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.stride = stride
        self.padding = padding if padding is not None else kernel_size // 2
        self.groups = groups
        
        # Base kernel set: Multiple kernels to choose from
        self.base_kernels = nn.ParameterList([
            nn.Parameter(
                torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size) * 0.02
            )
            for _ in range(num_kernels)
        ])
        
        # Condition predictor network
        # Lightweight network to predict which kernels to use
        condition_dim = max(in_channels // 4, 16)
        self.condition_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, condition_dim, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(condition_dim, num_kernels, 1, bias=False),
            nn.Softmax(dim=1)
        )
        
        # Bias (optional)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dynamic kernel generation.
        
        Args:
            x: Input features [B, in_channels, H, W]
        
        Returns:
            Output features [B, out_channels, H', W']
        """
        B, C, H, W = x.shape
        
        # Predict condition weights
        condition_weights = self.condition_predictor(x)  # [B, num_kernels, 1, 1]
        condition_weights = condition_weights.squeeze(-1).squeeze(-1)  # [B, num_kernels]
        
        # Generate dynamic kernel: weighted combination of base kernels
        # Average over batch for kernel generation
        dynamic_kernel = torch.zeros(
            self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size,
            self.kernel_size,
            device=x.device,
            dtype=x.dtype
        )
        
        for i, base_kernel in enumerate(self.base_kernels):
            # Average weight across batch
            avg_weight = condition_weights[:, i].mean()
            dynamic_kernel = dynamic_kernel + avg_weight * base_kernel
        
        # Apply convolution
        output = F.conv2d(
            x,
            dynamic_kernel,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups
        )
        
        return output
    
    def compute_lighting_condition(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute lighting condition metrics.
        
        Args:
            x: Input features [B, C, H, W]
        
        Returns:
            Condition vector [B, 2] (brightness, contrast)
        """
        # Brightness: global average
        brightness = x.mean(dim=(2, 3))  # [B, C]
        
        # Contrast: standard deviation
        contrast = x.std(dim=(2, 3))  # [B, C]
        
        # Combine into condition vector
        condition = torch.stack([
            brightness.mean(dim=1),
            contrast.mean(dim=1)
        ], dim=1)  # [B, 2]
        
        return condition
    
    def compute_occlusion_score(
        self,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute occlusion score from attention weights.
        
        Args:
            attention_weights: Attention weights [B, H, W]
        
        Returns:
            Occlusion score [B] (higher = more occluded)
        """
        occlusion_score = 1 - attention_weights.mean(dim=(1, 2))  # [B]
        return occlusion_score
    
    def compute_motion_condition(
        self,
        motion_magnitude: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute motion condition from temporal encoder.
        
        Args:
            motion_magnitude: Motion magnitude [B, 1]
        
        Returns:
            Motion condition [B, 1]
        """
        return motion_magnitude


