"""Unified Therapy State Head for MaxSight 3.0."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class TherapyStateHead(nn.Module):
    """Unified Therapy State Head:."""
    
    def __init__(
        self,
        eye_dim: int = 4,
        motion_dim: int = 256,
        temporal_dim: int = 128,
        hidden_dim: int = 64,
        in_channels_depth: int = 256,
        in_channels_contrast: int = 256,
        dropout: float = 0.1,
        use_lstm: bool = True,
        lstm_hidden_size: int = 32,
        lstm_num_layers: int = 2,
        use_depth_multi_scale: bool = True,
        depth_activation: str = 'sigmoid',
        use_edge_aware: bool = True
    ):
        super().__init__()
        
        # --- Fatigue/Gaze shared backbone ---.
        self.eye_dim = eye_dim
        self.temporal_dim = temporal_dim
        self.hidden_dim = hidden_dim
        self.use_lstm = use_lstm
        
        self.initial_net = nn.Sequential(
            nn.Linear(eye_dim + temporal_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_num_layers,
                batch_first=True,
                dropout=dropout if lstm_num_layers > 1 else 0.0
            )
            lstm_output_dim = lstm_hidden_size
        else:
            self.lstm = None
            lstm_output_dim = hidden_dim
        
        self.shared_net = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        head_input_dim = hidden_dim // 2
        self.fatigue_head = self._make_head(head_input_dim)
        self.blink_rate_head = self._make_head(head_input_dim)
        self.fixation_stability_head = self._make_head(head_input_dim)
        
        self.lstm_hidden = None
        
        # --- Depth/Focus ---.
        self.in_channels_depth = in_channels_depth
        self.motion_dim = motion_dim
        self.use_depth_multi_scale = use_depth_multi_scale
        self.depth_activation = depth_activation
        
        if motion_dim > 0:
            self.motion_proj_depth = nn.Conv2d(motion_dim, in_channels_depth, kernel_size=1, bias=False)
        
        # Depth branch.
        self.depth_branch = nn.Sequential(
            nn.Conv2d(in_channels_depth, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True)
        )
        
        # Uncertainty branch.
        self.uncertainty_branch = nn.Sequential(
            nn.Conv2d(in_channels_depth, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True)
        )
        
        # Depth output.
        if depth_activation == 'sigmoid':
            self.depth_conv = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())
        elif depth_activation == 'softplus':
            self.depth_conv = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Softplus())
        else:
            self.depth_conv = nn.Conv2d(64, 1, 1)
        
        self.uncertainty_conv = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())
        
        if use_depth_multi_scale:
            self.fpn_proj = nn.ModuleDict({
                'p3': nn.Conv2d(256, 64, 1),
                'p4': nn.Conv2d(256, 64, 1)
            })
        
        # Zone head.
        ZONE_DEPTH_FEAT_DIM = 64
        ZONE_DEPTH_STATS_DIM = 3
        ZONE_INPUT_DIM = ZONE_DEPTH_FEAT_DIM + ZONE_DEPTH_STATS_DIM
        self.zone_head = nn.Sequential(
            nn.Linear(ZONE_INPUT_DIM, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 3)
        )
        
        self.in_channels_contrast = in_channels_contrast
        self.use_edge_aware = use_edge_aware
        
        if motion_dim > 0:
            self.motion_proj_contrast = nn.Conv2d(motion_dim, in_channels_contrast, kernel_size=1, bias=False)
        
        # Edge detection (if enabled)
        if use_edge_aware:
            self.edge_conv = nn.Sequential(
                nn.Conv2d(in_channels_contrast, 32, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, 32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1)
            )
        
        # Contrast estimation network.
        self.conv1_contrast = nn.Conv2d(in_channels_contrast, 128, kernel_size=3, padding=1, bias=False)
        self.bn1_contrast = nn.BatchNorm2d(128)
        self.conv2_contrast = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.bn2_contrast = nn.BatchNorm2d(64)
        self.conv3_contrast = nn.Conv2d(64, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Initialize weights.
        self._initialize_weights()
    
    def _make_head(self, input_dim: int) -> nn.Module:
        """Create a task-specific head."""
        return nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def _initialize_weights(self):
        """Initialize weights to prevent degenerate outputs."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        eye_features: torch.Tensor,  # [B, eye_dim].
        motion_features: torch.Tensor,  # [B, motion_dim] or [B, motion_dim, H, W].
        depth_features: torch.Tensor,  # [B, in_channels_depth, H, W].
        contrast_features: torch.Tensor,  # [B, in_channels_contrast, H, W].
        fpn_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for all therapy state outputs."""
        B = eye_features.shape[0]
        device = eye_features.device
        
        # --- Fatigue/Gaze ---. Extract motion features for fatigue (if 2D, pool to 1D)
        if motion_features.dim() == 4:
            # [B, motion_dim, H, W] -> [B, motion_dim].
            motion_1d = F.adaptive_avg_pool2d(motion_features, 1).squeeze(-1).squeeze(-1)
        elif motion_features.dim() == 2:
            motion_1d = motion_features  # [B, motion_dim].
        else:
            raise ValueError(f"Expected motion_features to be 2D [B, D] or 4D [B, D, H, W], got {motion_features.shape}")
        
        # Ensure temporal_dim matches motion_dim for concatenation.
        if motion_1d.shape[1] != self.temporal_dim:
            # Project if needed (create on first use)
            if not hasattr(self, 'motion_proj_fatigue'):
                self.motion_proj_fatigue = nn.Linear(motion_1d.shape[1], self.temporal_dim).to(device)
            motion_1d = self.motion_proj_fatigue(motion_1d)
        
        combined = torch.cat([eye_features, motion_1d], dim=1)
        initial_features = self.initial_net(combined)
        
        if self.use_lstm and self.lstm is not None:
            seq = initial_features.unsqueeze(1)  # [B, 1, hidden_dim].
            lstm_out, self.lstm_hidden = self.lstm(seq, self.lstm_hidden)
            lstm_features = lstm_out.squeeze(1)
        else:
            lstm_features = initial_features
        
        shared_features = self.shared_net(lstm_features)
        
        fatigue_score = self.fatigue_head(shared_features)
        blink_rate = self.blink_rate_head(shared_features)
        fixation_stability = self.fixation_stability_head(shared_features)
        
        # --- Depth/Focus ---. Motion-conditioned depth (if motion is 4D)
        if motion_features.dim() == 4 and hasattr(self, 'motion_proj_depth'):
            motion_proj = self.motion_proj_depth(motion_features)
            if motion_proj.shape[2:] != depth_features.shape[2:]:
                motion_proj = F.interpolate(
                    motion_proj, 
                    size=depth_features.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            depth_features = depth_features + motion_proj
        
        depth_feat = self.depth_branch(depth_features)
        uncertainty_feat = self.uncertainty_branch(depth_features)
        
        if self.use_depth_multi_scale and fpn_features is not None:
            depth_feat_list = [depth_feat]
            for scale_name, proj in self.fpn_proj.items():
                if scale_name in fpn_features:
                    fpn_feat = fpn_features[scale_name]
                    proj_feat = proj(fpn_feat)
                    proj_feat = F.interpolate(
                        proj_feat, 
                        size=depth_feat.shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                    depth_feat_list.append(proj_feat)
            depth_feat = torch.stack(depth_feat_list, dim=0).mean(dim=0)
        
        depth_map = self.depth_conv(depth_feat).squeeze(1)  # [B, H, W].
        uncertainty = self.uncertainty_conv(uncertainty_feat).squeeze(1)  # [B, H, W].
        
        # Zone classification.
        depth_flat = depth_map.contiguous().reshape(B, -1).float()  # Ensure float for quantile.
        p25 = torch.quantile(depth_flat, 0.25, dim=1)
        p50 = torch.quantile(depth_flat, 0.5, dim=1)
        p75 = torch.quantile(depth_flat, 0.75, dim=1)
        depth_stats = torch.stack([p25, p50, p75], dim=1)
        
        depth_pooled = F.adaptive_avg_pool2d(depth_feat, 1).contiguous().reshape(B, -1)
        zone_input = torch.cat([depth_pooled, depth_stats], dim=1)
        zones = self.zone_head(zone_input)
        
        # Motion-conditioned contrast (if motion is 4D)
        if motion_features.dim() == 4 and hasattr(self, 'motion_proj_contrast'):
            motion_proj = self.motion_proj_contrast(motion_features)
            if motion_proj.shape[2:] != contrast_features.shape[2:]:
                motion_proj = F.interpolate(
                    motion_proj, 
                    size=contrast_features.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            contrast_features = contrast_features + motion_proj
        
        # Edge-aware modulation.
        if self.use_edge_aware:
            edge_logits = self.edge_conv(contrast_features)  # [B, 1, H, W].
            edge_map = torch.clamp(edge_logits, 0, 1)
            modulated_features = contrast_features * (1.0 + edge_map)
        else:
            edge_map = None
            modulated_features = contrast_features
        
        # Contrast estimation.
        x = self.relu(self.bn1_contrast(self.conv1_contrast(modulated_features)))
        x = self.relu(self.bn2_contrast(self.conv2_contrast(x)))
        contrast_map = torch.sigmoid(self.conv3_contrast(x)).squeeze(1)  # [B, H, W].
        
        # Build output dictionary.
        outputs = {
            'fatigue_score': fatigue_score,
            'blink_rate': blink_rate,
            'fixation_stability': fixation_stability,
            'shared_features': shared_features,
            'depth_map': depth_map,
            'uncertainty': uncertainty,
            'zones': zones,
            'contrast_map': contrast_map
        }
        
        if edge_map is not None:
            outputs['edge_map'] = edge_map.squeeze(1)  # [B, H, W].
        
        return outputs







