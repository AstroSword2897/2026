"""Motion/Flow Head for MaxSight Therapy System."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union, List


class MotionHead(nn.Module):
    """Scaled-up motion/flow head for therapy tasks and temporal understanding."""

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 256,  # SCALED: was 64, now 256.
        use_refinement: bool = True,
        num_refinement_stages: int = 3,  # SCALED: was 1, now 3.
        use_temporal_stacking: bool = True,  # NEW: temporal 3D convs.
        temporal_frames: int = 3,  # NEW: T frames for temporal stack.
        use_multi_scale: bool = True,  # NEW: coarse-to-fine.
        use_attention: bool = False  # NEW: optional attention in refinement.
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.use_refinement = use_refinement
        self.num_refinement_stages = num_refinement_stages
        self.use_temporal_stacking = use_temporal_stacking
        self.temporal_frames = temporal_frames
        self.use_multi_scale = use_multi_scale
        self.use_attention = use_attention

        # Coarse_net always takes hidden_channels; use input_proj when needed.
        self.coarse_net = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True)
        )

        # NEW: Temporal stacking with 3D convolutions.
        if use_temporal_stacking:
            # 3D conv for temporal features [B, T, C, H, W].
            self.temporal_conv = nn.Sequential(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=(temporal_frames, 3, 3),
                    padding=(temporal_frames // 2, 1, 1),
                    bias=False
                ),
                nn.BatchNorm3d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=(1, 3, 3),
                    padding=(0, 1, 1),
                    bias=False
                ),
                nn.BatchNorm3d(hidden_channels),
                nn.ReLU(inplace=True)
            )
            # Project temporal output to hidden_channels (for unified coarse_net)
            self.temporal_proj = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
        
        # Always create input_proj for standard 4D input path.
        self.input_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

        # Multi-scale processing (coarse H/2 to fine H).
        if use_multi_scale:
            # Coarse network at H/2 resolution.
            self.coarse_scale_net = nn.Sequential(
                nn.Conv2d(hidden_channels // 2, hidden_channels // 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels // 2),
                nn.ReLU(inplace=True)
            )
            # Upsample and refine.
            self.upsample_refine = nn.Sequential(
                nn.Conv2d(hidden_channels // 2, hidden_channels // 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels // 2),
                nn.ReLU(inplace=True)
            )

        # Initial flow prediction (u, v)
        self.flow_pred = nn.Conv2d(hidden_channels // 2, 2, kernel_size=1)

        # SCALED: Multi-stage refinement (2-3 stages instead of 1)
        if use_refinement:
            self.refinement_stages = nn.ModuleList()
            for stage in range(num_refinement_stages):
                # Each refinement stage: residual connection + conv block.
                refinement_block = nn.Sequential(
                    nn.Conv2d(
                        hidden_channels // 2 + 2,  # Features + previous flow.
                        hidden_channels // 2,
                        kernel_size=3,
                        padding=1,
                        bias=False
                    ),
                    nn.BatchNorm2d(hidden_channels // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_channels // 2, hidden_channels // 2, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(hidden_channels // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_channels // 2, 2, kernel_size=1)  # Flow residual.
                )
                self.refinement_stages.append(refinement_block)

            # NEW: Optional attention in refinement (CBAM-style)
            if use_attention:
                self.attention = ChannelSpatialAttention(hidden_channels // 2)

        # NEW: Multi-resolution supervision heads (for training)
        self.flow_head_full = nn.Conv2d(hidden_channels // 2, 2, kernel_size=1)
        self.flow_head_half = nn.Conv2d(hidden_channels // 2, 2, kernel_size=1)
        self.flow_head_quarter = nn.Conv2d(hidden_channels // 2, 2, kernel_size=1)

        self.tanh = nn.Tanh()  # Normalize flow to [-1, 1].

    def forward(
        self,
        temporal_features: torch.Tensor,
        return_features: bool = False,
        return_multi_scale: bool = False
    ) -> Union[torch.Tensor, Dict[str, Union[torch.Tensor, None]]]:
        """Forward pass to generate motion flow with scaled-up computation."""
        # Unified processing path; no layer skipping or special-case hacks.
        # Process input to always produce [B, hidden_channels, H, W] for coarse_net.
        
        if temporal_features.dim() == 5 and self.use_temporal_stacking:
            # [B, T, C, H, W] -> process with 3D convs.
            B, T, C, H, W = temporal_features.shape
            if C != self.in_channels:
                raise ValueError(f"Expected {self.in_channels} channels, got {C}")
            # Reshape for 3D conv: [B, C, T, H, W].
            temporal_3d = temporal_features.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W].
            # Apply 3D temporal convolution.
            temporal_out = self.temporal_conv(temporal_3d)  # [B, hidden_channels, T', H, W].
            # Average over temporal dimension.
            temporal_out = temporal_out.mean(dim=2)  # [B, hidden_channels, H, W].
            # Project to hidden_channels (for unified coarse_net)
            features_input = self.temporal_proj(temporal_out)  # [B, hidden_channels, H, W].
        elif temporal_features.dim() == 4:
            # [B, C, H, W] - standard case.
            B, C, H, W = temporal_features.shape
            if C != self.in_channels:
                raise ValueError(f"Expected {self.in_channels} channels, got {C}")
            # Project to hidden_channels for unified coarse_net.
            features_input = self.input_proj(temporal_features)  # [B, hidden_channels, H, W].
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got {temporal_features.shape}")

        features = self.coarse_net(features_input)

        # Multi-scale processing (coarse H/2 to fine H).
        multi_scale_flows = {}
        if self.use_multi_scale:
            # Downsample to H/2.
            features_coarse = F.avg_pool2d(features, kernel_size=2, stride=2)  # [B, C, H/2, W/2].
            features_coarse = self.coarse_scale_net(features_coarse)
            
            # Predict coarse flow.
            coarse_flow_half = self.flow_head_half(features_coarse)  # [B, 2, H/2, W/2].
            multi_scale_flows['half'] = coarse_flow_half
            
            # Upsample and refine.
            features_fine = F.interpolate(
                features_coarse,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            features_fine = self.upsample_refine(features_fine)
            # Combine with original features.
            features = features + features_fine  # Residual connection.

        # Initial flow prediction.
        coarse_flow = self.flow_pred(features)  # [B, 2, H, W].

        # SCALED: Multi-stage refinement (2-3 stages)
        motion = coarse_flow
        if self.use_refinement:
            for stage_idx, refinement_stage in enumerate(self.refinement_stages):
                # Concatenate features with current flow estimate.
                refinement_input = torch.cat([features, motion], dim=1)
                
                # Apply attention if enabled.
                if self.use_attention and stage_idx == len(self.refinement_stages) - 1:
                    # Apply attention on last stage.
                    features_attended = self.attention(features)
                    refinement_input = torch.cat([features_attended, motion], dim=1)
                
                # Compute flow residual.
                flow_residual = refinement_stage(refinement_input)  # [B, 2, H, W].
                
                # Residual connection: add to previous flow.
                motion = motion + flow_residual  # [B, 2, H, W].

        # Normalize to [-1, 1].
        motion = self.tanh(motion)

        # Check for NaN/Inf.
        if torch.isnan(motion).any() or torch.isinf(motion).any():
            raise RuntimeError("NaN/Inf detected in motion flow")

        # Motion magnitude.
        flow_magnitude = torch.sqrt(motion[:, 0]**2 + motion[:, 1]**2)

        # NEW: Multi-resolution supervision (for training)
        if return_multi_scale or self.training:
            # Full resolution (already computed)
            multi_scale_flows['full'] = motion
            
            # Half resolution.
            if 'half' not in multi_scale_flows:
                features_half = F.avg_pool2d(features, kernel_size=2, stride=2)
                multi_scale_flows['half'] = self.flow_head_half(features_half)
            
            # Quarter resolution.
            features_quarter = F.avg_pool2d(features, kernel_size=4, stride=4)
            multi_scale_flows['quarter'] = self.flow_head_quarter(features_quarter)

        if return_features or return_multi_scale:
            result = {
                'flow': motion,
                'flow_magnitude': flow_magnitude,
                'features': features,
                'coarse_flow': coarse_flow
            }
            if return_multi_scale:
                result['multi_scale_flows'] = multi_scale_flows
            return result

        return motion

    def compute_smoothness_loss(
        self,
        flow: torch.Tensor,
        image: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Edge-aware smoothness loss."""
        # Flow gradients.
        flow_grad_x = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])
        flow_grad_y = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])

        if image is not None:
            gray = image.mean(dim=1, keepdim=True) if image.shape[1] == 3 else image
            img_grad_x = torch.abs(gray[:, :, :, :-1] - gray[:, :, :, 1:])
            img_grad_y = torch.abs(gray[:, :, :-1, :] - gray[:, :, 1:, :])
            weight_x = torch.exp(-img_grad_x.mean(dim=1, keepdim=True)).expand_as(flow_grad_x)
            weight_y = torch.exp(-img_grad_y.mean(dim=1, keepdim=True)).expand_as(flow_grad_y)
            smoothness = (flow_grad_x * weight_x).mean() + (flow_grad_y * weight_y).mean()
        else:
            smoothness = flow_grad_x.mean() + flow_grad_y.mean()

        return smoothness


class ChannelSpatialAttention(nn.Module):
    """CBAM-style attention for refinement stages. Channel attention + Spatial attention."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # Channel attention.
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention.
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention.
        avg_out = self.channel_attention(self.avg_pool(x))
        max_out = self.channel_attention(self.max_pool(x))
        channel_att = avg_out + max_out
        x = x * channel_att
        
        # Spatial attention.
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att
        
        return x






