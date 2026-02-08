import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

class ContrastMapHead(nn.Module):
    def __init__(self, in_channels: int = 256, motion_dim: int = 256, use_edge_aware: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.motion_dim = motion_dim
        self.use_edge_aware = use_edge_aware
        
        # FIXED: Motion-conditioned contrast estimation. Motion stability indicates reliable contrast regions.
        if motion_dim > 0:
            self.motion_proj = nn.Conv2d(motion_dim, in_channels, kernel_size=1, bias=False)
        
        # Contrast estimation network.
        # WHY THIS ARCHITECTURE:.
        # - Progressive channel reduction: 256 -> 128 -> 64 -> 1 (efficient computation)
        # - 3x3 kernels: Capture local contrast relationships.
        # - 1x1 final layer: Efficiently maps to single contrast value per pixel.
        
        # Edge detection channel (computed from features, used as modulation signal)
        if use_edge_aware:
            # Single 4D edge channel that modulates features during forward pass.
            # Using GroupNorm instead of BatchNorm for robustness to small batches.
            self.edge_conv = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, 32),  # More robust than BatchNorm for small batches.
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1),  # Single edge channel [B, 1, H, W].
                # No sigmoid - clamp after scaling to avoid saturation. Edge map will be clamped to [0, 1] during forward pass.
            )
        
        # Contrast estimation network (now modulated by edges)
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=1)  # Single channel contrast map.
        self.relu = nn.ReLU(inplace=True)
        
        # Initialize weights properly.
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to prevent degenerate outputs."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def get_edge_map(self, features: torch.Tensor) -> torch.Tensor:
        """Get edge map computed during forward pass (for visualization/debugging)."""
        if not self.use_edge_aware:
            return torch.zeros_like(features[:, :1])
        
        return self.edge_conv(features)  # [B, 1, H, W].
    
    def forward(
        self, 
        features: torch.Tensor,
        motion_features: Optional[torch.Tensor] = None  # FIXED: Motion as temporal anchor.
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with edge-aware modulation."""
        # Validate input.
        if features.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], got {features.shape}")
        
        B, C, H, W = features.shape
        if C != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {C}. "
                f"Ensure input features match head configuration."
            )
        
        # FIXED: Motion-conditioned feature extraction. Motion stability indicates reliable contrast regions.
        if motion_features is not None and hasattr(self, 'motion_proj'):
            motion_proj = self.motion_proj(motion_features)
            if motion_proj.shape[2:] != features.shape[2:]:
                motion_proj = F.interpolate(motion_proj, size=features.shape[2:], mode='bilinear', align_corners=False)
            features = features + motion_proj
        
        # Compute edge map as modulation signal (if enabled)
        if self.use_edge_aware:
            edge_logits = self.edge_conv(features)  # [B, 1, H, W] - raw logits.
            # Clamp to [0, 1] to avoid saturation (better than sigmoid for gradients)
            edge_map = torch.clamp(edge_logits, 0, 1)
            # Modulate features with edge information: emphasize edge-relevant features.
            # Edge map acts as attention/importance weighting.
            modulated_features = features * (1.0 + edge_map)  # Boost edge regions.
        else:
            edge_map = None
            modulated_features = features
        
        # Feature extraction (now on edge-modulated features)
        x = self.relu(self.bn1(self.conv1(modulated_features)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # Generate contrast map.
        contrast_map = torch.sigmoid(self.conv3(x))
        contrast_map = contrast_map.squeeze(1)  # [B, H, W].
        
        # Validate output.
        if torch.isnan(contrast_map).any() or torch.isinf(contrast_map).any():
            raise RuntimeError(
                "NaN/Inf detected in contrast map. Check input features and model initialization."
            )
        
        # Return contrast map (and edge map if available, for loss computation)
        if self.use_edge_aware and edge_map is not None:
            return contrast_map, edge_map.squeeze(1)  # Return both for loss.
        return contrast_map, None
    
    def compute_loss(
        self,
        pred_contrast: torch.Tensor,
        target_contrast: torch.Tensor,
        edge_map: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute contrast loss using learned edge map."""
        # Validate inputs.
        if pred_contrast.shape != target_contrast.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred_contrast.shape} vs target {target_contrast.shape}"
            )
        
        # Standard L1 loss.
        l1_loss = F.l1_loss(pred_contrast, target_contrast)
        
        # If edge map provided, use it for weighting (same edges as forward pass)
        if edge_map is not None and self.use_edge_aware:
            # Ensure edge_map is normalized [0, 1] and matches shape.
            if edge_map.shape != pred_contrast.shape:
                # Resize if needed.
                edge_map = F.interpolate(
                    edge_map.unsqueeze(1), 
                    size=pred_contrast.shape[1:], 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(1)
            
            # Clamp edge map to [0, 1] for safe weighting.
            edge_map = edge_map.clamp(0, 1)
            
            # Weight loss by edge strength (detach to prevent double-counting gradients)
            pixel_loss = torch.abs(pred_contrast - target_contrast)
            weighted_loss = pixel_loss * (1.0 + edge_map.detach())
            loss = weighted_loss.mean()
        else:
            # Simple L1 loss if no edge map.
            loss = l1_loss
        
        return {
            'loss': loss,
            'l1_loss': l1_loss,
        }







