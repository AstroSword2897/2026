"""Multi-Modal Fusion for MaxSight 3.0 Fuses vision, audio, depth, and haptic modalities using transformer-based fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class EnhancedAudioEncoder(nn.Module):
    """Enhanced audio encoder with spectrogram CNN and temporal attention."""
    
    def __init__(
        self,
        input_dim: int = 128,  # MFCC features or spectrogram bins.
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Spectrogram CNN (if input is spectrogram)
        self.spectrogram_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((None, embed_dim // 4))
        )
        
        # Temporal attention.
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        
        # Projection to final embedding.
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        audio_features: torch.Tensor,  # [B, T, F] or [B, F] (MFCC)
        stereo_channels: Optional[torch.Tensor] = None  # [B, T, 2] for directional.
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through enhanced audio encoder."""
        B = audio_features.shape[0]
        
        # Handle different input formats.
        if audio_features.dim() == 2:
            # [B, F] - single frame, expand to sequence.
            audio_features = audio_features.unsqueeze(1)  # [B, 1, F].
        
        # If input is spectrogram [B, T, F], convert to [B, 1, T, F] for CNN.
        if audio_features.dim() == 3:
            T, F = audio_features.shape[1], audio_features.shape[2]
            # Reshape for CNN: [B, T, F] -> [B, 1, T, F].
            spec = audio_features.unsqueeze(1)
            
            # Apply CNN.
            cnn_out = self.spectrogram_cnn(spec)  # [B, 64, T', embed_dim//4].
            # Flatten: [B, 64, T', embed_dim//4] -> [B, T', embed_dim].
            cnn_out = cnn_out.permute(0, 2, 1, 3).contiguous()
            cnn_out = cnn_out.reshape(B, -1, self.embed_dim)
            
            # Temporal attention.
            attended, _ = self.temporal_attention(cnn_out, cnn_out, cnn_out)
            audio_embed = self.norm(cnn_out + attended)
            audio_embed = self.proj(audio_embed.mean(dim=1))  # [B, embed_dim].
        else:
            # Simple projection for MFCC features.
            audio_embed = self.proj(audio_features.mean(dim=1))  # [B, embed_dim].
        
        # Directional processing (if stereo available)
        spatial_attention = None
        if stereo_channels is not None:
            # Compute direction from stereo channels. Left - Right gives direction.
            direction = stereo_channels[:, :, 0] - stereo_channels[:, :, 1]  # [B, T].
            direction = direction.mean(dim=1)  # [B].
            
            # Create spatial attention map (simplified) In practice, use a more sophisticated fusion strategy.
            spatial_attention = torch.ones(B, 14, 14, device=audio_features.device)
            # Could modulate based on direction here.
        
        return audio_embed, spatial_attention


class MultimodalFusion(nn.Module):
    """Multi-modal transformer fusion. Fuses vision, audio, depth, and haptic modalities using cross-modal attention."""
    
    def __init__(
        self,
        vision_dim: int = 512,
        audio_dim: int = 256,
        depth_dim: int = 128,
        haptic_dim: int = 64,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Modality projections.
        self.vision_proj = nn.Linear(vision_dim, embed_dim)
        self.audio_proj = nn.Linear(audio_dim, embed_dim)
        self.depth_proj = nn.Linear(depth_dim, embed_dim) if depth_dim > 0 else None
        self.haptic_proj = nn.Linear(haptic_dim, embed_dim) if haptic_dim > 0 else None
        
        # Cross-modal transformer layers.
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Modality tokens.
        self.vision_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.audio_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.depth_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02) if depth_dim > 0 else None
        self.haptic_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02) if haptic_dim > 0 else None
        
        # Output projection.
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        vision_features: torch.Tensor,  # [B, N_vision, vision_dim].
        audio_features: torch.Tensor,   # [B, audio_dim].
        depth_features: Optional[torch.Tensor] = None,  # [B, depth_dim].
        haptic_features: Optional[torch.Tensor] = None   # [B, haptic_dim].
    ) -> torch.Tensor:
        """Forward pass through multimodal fusion."""
        B = vision_features.shape[0]
        
        # Project all modalities to common dimension.
        vision_proj = self.vision_proj(vision_features)  # [B, N_vision, embed_dim].
        audio_proj = self.audio_proj(audio_features).unsqueeze(1)  # [B, 1, embed_dim].
        
        # Collect modality tokens.
        tokens = [vision_proj, audio_proj]
        
        if depth_features is not None and self.depth_proj is not None:
            depth_proj = self.depth_proj(depth_features).unsqueeze(1)  # [B, 1, embed_dim].
            tokens.append(depth_proj)
        
        if haptic_features is not None and self.haptic_proj is not None:
            haptic_proj = self.haptic_proj(haptic_features).unsqueeze(1)  # [B, 1, embed_dim].
            tokens.append(haptic_proj)
        
        # Concatenate all tokens.
        multimodal_tokens = torch.cat(tokens, dim=1)  # [B, N_total, embed_dim].
        
        # Apply transformer layers.
        x = multimodal_tokens
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Global pooling.
        fused = x.mean(dim=1)  # [B, embed_dim].
        fused = self.norm(fused)
        fused = self.output_proj(fused)
        
        return fused


class SpatialSoundMapping(nn.Module):
    """Maps 3D audio cues to spatial attention maps for visual CNN."""
    
    def __init__(self, audio_dim: int = 256, attention_size: Tuple[int, int] = (14, 14), num_directions: int = 4):
        super().__init__()
        self.attention_size = attention_size
        self.num_directions = num_directions
        self.direction_estimator = nn.Sequential(
            nn.Linear(audio_dim, 128), nn.ReLU(), nn.Linear(128, num_directions), nn.Softmax(dim=1)
        )
        self.distance_estimator = nn.Sequential(
            nn.Linear(audio_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.attention_generator = nn.Sequential(
            nn.Linear(audio_dim + num_directions + 1, 128), nn.ReLU(),
            nn.Linear(128, attention_size[0] * attention_size[1]), nn.Sigmoid()
        )
    
    def forward(self, audio_features: torch.Tensor, stereo_channels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = audio_features.shape[0]
        H, W = self.attention_size
        if stereo_channels is not None:
            ild = (stereo_channels[:, :, 0] - stereo_channels[:, :, 1]).mean(dim=1)
            direction_raw = torch.stack([
                (ild < -0.1).float(), (ild > 0.1).float(), (torch.abs(ild) < 0.1).float(),
                torch.zeros(B, device=audio_features.device)
            ], dim=1)
            direction = direction_raw / (direction_raw.sum(dim=1, keepdim=True) + 1e-8)
            intensity = stereo_channels.abs().mean(dim=(1, 2))
            intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
            distance = (1 - intensity_norm).unsqueeze(1)
        else:
            direction = self.direction_estimator(audio_features)
            distance = self.distance_estimator(audio_features)
        combined = torch.cat([audio_features, direction, distance], dim=1)
        attention_flat = self.attention_generator(combined)
        attention_map = attention_flat.reshape(B, 1, H, W)
        return attention_map, direction, distance
    
    def apply_audio_attention(self, visual_features: torch.Tensor, attention_map: torch.Tensor) -> torch.Tensor:
        if attention_map.shape[2:] != visual_features.shape[2:]:
            attention_map = F.interpolate(attention_map, size=visual_features.shape[2:], mode='bilinear', align_corners=False)
        return visual_features * attention_map


class HapticEmbedding(nn.Module):
    """Haptic feedback embedding module."""
    
    def __init__(self, haptic_dim: int = 64, embed_dim: int = 128, num_patterns: int = 10):
        super().__init__()
        self.haptic_dim = haptic_dim
        self.embed_dim = embed_dim
        self.num_patterns = num_patterns
        self.pattern_encoder = nn.Sequential(
            nn.Linear(haptic_dim, 128), nn.ReLU(), nn.Linear(128, embed_dim), nn.LayerNorm(embed_dim)
        )
        self.pattern_classifier = nn.Linear(embed_dim, num_patterns)
    
    def forward(self, haptic_pattern: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if haptic_pattern.dim() == 3:
            haptic_pattern = haptic_pattern.mean(dim=1)
        haptic_embedding = self.pattern_encoder(haptic_pattern)
        pattern_logits = self.pattern_classifier(haptic_embedding)
        return haptic_embedding, pattern_logits


class HapticVisualAttention(nn.Module):
    """Cross-modal attention: Haptic to Visual."""
    
    def __init__(self, haptic_embed_dim: int = 128, visual_embed_dim: int = 256, attention_dim: int = 128):
        super().__init__()
        self.haptic_proj = nn.Linear(haptic_embed_dim, attention_dim)
        self.visual_proj = nn.Linear(visual_embed_dim, attention_dim)
        self.attention = nn.MultiheadAttention(attention_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(attention_dim)
    
    def forward(self, haptic_embedding: torch.Tensor, visual_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        haptic_proj = self.haptic_proj(haptic_embedding).unsqueeze(1)
        visual_proj = self.visual_proj(visual_features)
        attended, attn_weights = self.attention(query=haptic_proj, key=visual_proj, value=visual_proj)
        haptic_expanded = haptic_proj.expand(-1, visual_proj.shape[1], -1)
        attended_visual = self.norm(visual_proj + attended.expand(-1, visual_proj.shape[1], -1))
        attention_weights = attn_weights.squeeze(1)
        return attended_visual, attention_weights








