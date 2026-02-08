"""Temporal Encoder Module for MaxSight 3.0."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from .conv_lstm import ConvLSTM

# Optional TimeSformer import (only used if use_timesformer=True)
try:
    from .temporal_transformer import TimeSformer
except ImportError:
    TimeSformer = None  # Will be None if module doesn't exist.


class TemporalEncoder(nn.Module):
    """Enhanced temporal encoder for video sequence processing."""
    
    def __init__(
        self,
        in_channels: int = 256,
        num_frames: int = 8,
        hidden_dim: int = 256,
        vit_embed_dim: int = 768,
        use_conv_lstm: bool = True,
        use_timesformer: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        self.use_conv_lstm = use_conv_lstm
        self.use_timesformer = use_timesformer
        
        # ConvLSTM for motion tracking.
        if use_conv_lstm:
            self.conv_lstm = ConvLSTM(
                input_dim=in_channels,
                hidden_dim=hidden_dim,
                kernel_size=3,
                num_layers=2
            )
        
        # TimeSformer for long-range temporal dependencies.
        if use_timesformer:
            if TimeSformer is None:
                raise ImportError("TimeSformer module not found. Set use_timesformer=False or install temporal_transformer module.")
            self.timesformer = TimeSformer(
                embed_dim=vit_embed_dim,
                num_heads=12,
                num_layers=12,
                num_frames=num_frames
            )
        
        # Motion feature head (from ConvLSTM output)
        if use_conv_lstm:
            self.motion_head = nn.Sequential(
                nn.Conv2d(hidden_dim, 2, kernel_size=1),  # U, v motion.
                nn.Tanh()  # Normalize to [-1, 1].
            )
        
        # Temporal consistency head. Flatten before Linear so spatial dimensions are collapsed.
        self.consistency_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),  # [B, C, 1, 1] -> [B, C].
            nn.Linear(hidden_dim if use_conv_lstm else in_channels, 1),
            nn.Sigmoid()
        )
        
        # Flicker detection head.
        self.flicker_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),  # [B, C, 1, 1] -> [B, C].
            nn.Linear(hidden_dim if use_conv_lstm else in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        feature_frames: torch.Tensor,  # RENAMED: frames -> feature_frames for clarity.
        vit_patch_tokens: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through enhanced temporal encoder."""
        B = feature_frames.shape[0]
        
        # Handle different input formats.
        if feature_frames.dim() == 5:
            if feature_frames.shape[1] == self.in_channels:
                # [B, C, T, H, W].
                frames_seq = feature_frames.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W].
            else:
                # [B, T, C, H, W].
                frames_seq = feature_frames
        else:
            raise ValueError(f"Expected 5D input (feature maps), got {feature_frames.dim()}D")
        
        H, W = frames_seq.shape[-2], frames_seq.shape[-1]
        
        outputs = {}
        
        # ConvLSTM for motion tracking.
        motion_features = None  # Initialize for flicker detection.
        if self.use_conv_lstm:
            # For now, assume frames_seq is already feature maps.
            motion_features, (h, c) = self.conv_lstm(frames_seq)  # [B, T, hidden_dim, H, W].
            
            # Use last frame's motion features.
            motion_last = motion_features[:, -1]  # [B, hidden_dim, H, W].
            
            # Motion flow prediction - downsample for efficiency.
            motion = self.motion_head(motion_last)  # [B, 2, H, W].
            # Downsample motion for efficiency (optional but recommended)
            if motion.shape[-1] > 56:  # Only downsample if resolution is high.
                motion = F.interpolate(motion, scale_factor=0.5, mode='bilinear', align_corners=False)
            outputs['motion'] = motion
            outputs['motion_features'] = motion_last  # Also return full features for Stage B.
            
            # Temporal consistency from motion features.
            consistency_feat = motion_last
        else:
            # Fallback: use last frame.
            consistency_feat = frames_seq[:, -1]  # [B, C, H, W].
            outputs['motion'] = torch.zeros(B, 2, H, W, device=feature_frames.device)
        
        # Temporal consistency score.
        consistency = self.consistency_head(consistency_feat).squeeze(-1).squeeze(-1)  # [B, 1].
        outputs['consistency'] = consistency.unsqueeze(1) if consistency.dim() == 1 else consistency
        
        # Flicker detection - CRITICAL FIX: Actually use temporal information.
        # Compare last two frames instead of just using last frame.
        if self.use_conv_lstm and motion_features is not None and motion_features.shape[1] >= 2:
            # Use frame difference for flicker detection.
            flicker_feat = torch.abs(motion_features[:, -1] - motion_features[:, -2])  # [B, hidden_dim, H, W].
        else:
            # Fallback to consistency feature.
            flicker_feat = consistency_feat
        
        flicker = self.flicker_head(flicker_feat).squeeze(-1).squeeze(-1)  # [B, 1].
        outputs['flicker'] = flicker.unsqueeze(1) if flicker.dim() == 1 else flicker
        
        # TimeSformer for long-range temporal context.
        if self.use_timesformer and vit_patch_tokens is not None:
            temporal_context = self.timesformer(vit_patch_tokens)  # [B, embed_dim].
            outputs['temporal_context'] = temporal_context
        
        return outputs


class TemporalBuffer:
    """Buffer for maintaining temporal context across frames. Maintains a sliding window of recent frames for temporal processing."""
    
    def __init__(self, buffer_size: int = 5):
        self.buffer_size = buffer_size
        self.buffer = []
    
    def add_frame(self, frame: torch.Tensor):
        """Add a new frame to the buffer."""
        self.buffer.append(frame)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
    
    def get_sequence(self) -> Optional[torch.Tensor]:
        """Get the current sequence of frames. Returns: Tensor [T, C, H, W] if buffer is full, None otherwise."""
        if len(self.buffer) < self.buffer_size:
            return None
        return torch.stack(self.buffer, dim=0)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer = []







