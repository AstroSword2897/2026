"""MaxSight 3.0 Attention Modules - Consolidated Production Version."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# CBAM / SE Attention.

class ChannelAttention(nn.Module):
    """Channel attention with safety checks and GPU optimization."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden_channels = max(1, channels // reduction)  # Safety check.
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # GPU-optimized Conv2d.
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention with adaptive kernel size."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(spatial))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden_channels = max(1, channels // reduction)  # Safety check.
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.avg_pool(x))


# Cross-Modal Attention.

class CrossModalAttention(nn.Module):
    """Cross-modal attention for vision/audio/haptic fusion."""
    
    def __init__(
        self,
        vision_dim: int,
        audio_dim: int,
        haptic_dim: int = 0,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.vision_proj = nn.Linear(vision_dim, embed_dim)
        self.audio_proj = nn.Linear(audio_dim, embed_dim)
        self.haptic_proj = nn.Linear(haptic_dim, embed_dim) if haptic_dim > 0 else None
        
        self.vision_audio_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.audio_vision_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm_vision = nn.LayerNorm(embed_dim)
        self.norm_audio = nn.LayerNorm(embed_dim)
        
        if haptic_dim > 0:
            self.output_proj = nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim),
                nn.Dropout(dropout)
            )
        else:
            self.output_proj = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Dropout(dropout)
            )
    
    def forward(
        self,
        vision_features: torch.Tensor,
        audio_features: torch.Tensor,
        haptic_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vision_proj = self.vision_proj(vision_features)
        audio_proj = self.audio_proj(audio_features)
        
        vision_enhanced, _ = self.vision_audio_attn(query=vision_proj, key=audio_proj, value=audio_proj)
        vision_enhanced = self.norm_vision(vision_proj + vision_enhanced)
        
        audio_enhanced, _ = self.audio_vision_attn(query=audio_proj, key=vision_proj, value=vision_proj)
        audio_enhanced = self.norm_audio(audio_proj + audio_enhanced)
        
        vision_global = vision_enhanced.mean(dim=1)
        audio_global = audio_enhanced.mean(dim=1)
        
        if haptic_features is not None and self.haptic_proj is not None:
            haptic_proj = self.haptic_proj(haptic_features)
            if haptic_proj.dim() == 2:
                fused = torch.cat([vision_global, audio_global, haptic_proj], dim=1)
            else:
                fused = torch.cat([vision_global, audio_global, haptic_proj.mean(dim=1)], dim=1)
        else:
            fused = torch.cat([vision_global, audio_global], dim=1)
        
        fused = self.output_proj(fused)
        return fused, vision_enhanced, audio_enhanced


# Cross-Task Attention.

class CrossTaskAttention(nn.Module):
    """Cross-task attention linking detection/OCR/description."""
    
    def __init__(
        self,
        detection_dim: int,
        ocr_dim: int,
        description_dim: int,
        embed_dim: int = 512,
        num_heads: int = 8
    ):
        super().__init__()
        self.detection_proj = nn.Linear(detection_dim, embed_dim)
        self.ocr_proj = nn.Linear(ocr_dim, embed_dim)
        self.description_proj = nn.Linear(description_dim, embed_dim)
        
        self.ocr_to_detection = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.detection_to_description = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.description_to_ocr = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.norm_detection = nn.LayerNorm(embed_dim)
        self.norm_ocr = nn.LayerNorm(embed_dim)
        self.norm_description = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        detection_features: torch.Tensor,
        ocr_features: torch.Tensor,
        description_context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        det_proj = self.detection_proj(detection_features)
        ocr_proj = self.ocr_proj(ocr_features)
        
        if description_context.dim() == 2:
            desc_proj = self.description_proj(description_context).unsqueeze(1)
        else:
            desc_proj = self.description_proj(description_context)
        
        det_enhanced, _ = self.ocr_to_detection(query=det_proj, key=ocr_proj, value=ocr_proj)
        det_enhanced = self.norm_detection(det_proj + det_enhanced)
        
        desc_enhanced, _ = self.detection_to_description(query=desc_proj, key=det_enhanced, value=det_enhanced)
        desc_enhanced = self.norm_description(desc_proj + desc_enhanced)
        
        N_text = ocr_proj.shape[1]
        if desc_enhanced.shape[1] == 1:
            desc_expanded = desc_enhanced.expand(-1, N_text, -1)
        else:
            desc_expanded = desc_enhanced[:, :N_text, :] if desc_enhanced.shape[1] >= N_text else desc_enhanced.repeat(1, (N_text // desc_enhanced.shape[1]) + 1, 1)[:, :N_text, :]
        
        ocr_enhanced, _ = self.description_to_ocr(query=ocr_proj, key=desc_expanded, value=desc_expanded)
        ocr_enhanced = self.norm_ocr(ocr_proj + ocr_enhanced)
        
        if description_context.dim() == 2:
            desc_enhanced = desc_enhanced.squeeze(1)
        
        return det_enhanced, ocr_enhanced, desc_enhanced







