"""
Vision Transformer Backbone for MaxSight 3.0

Provides global context understanding through self-attention mechanisms.
Designed to complement CNN backbone for hybrid architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def create_sinusoidal_pos_embedding(num_positions: int, embed_dim: int) -> torch.Tensor:
    """
    Create sinusoidal positional embeddings (non-learned alternative).
    
    Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
             PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    position = torch.arange(num_positions).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                        -(math.log(10000.0) / embed_dim))
    
    pos_embed = torch.zeros(num_positions, embed_dim)
    pos_embed[:, 0::2] = torch.sin(position * div_term)
    pos_embed[:, 1::2] = torch.cos(position * div_term)
    
    return pos_embed.unsqueeze(0)  # [1, num_positions, embed_dim]


class TransformerBlock(nn.Module):
    """
    Single Transformer encoder block with pre-norm architecture.
    
    Architecture:
    - Pre-norm: LayerNorm before attention/FFN (more stable training)
    - Multi-head self-attention
    - Feed-forward network with GELU activation
    - Residual connections
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0
    ):
        super().__init__()
        
        # Layer norms (pre-norm architecture)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            bias=qkv_bias,
            batch_first=True
        )
        
        # Feed-forward network
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tokens [B, N, embed_dim]
        
        Returns:
            Output tokens [B, N, embed_dim]
        """
        # Pre-norm attention with residual
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attention(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        
        # Pre-norm FFN with residual
        x_norm = self.norm2(x)
        ffn_out = self.mlp(x_norm)
        x = x + ffn_out
        
        return x


class VisionTransformerBackbone(nn.Module):
    """
    Complete Vision Transformer backbone.
    
    Architecture:
    1. Patch embedding: Divide image into patches
    2. CLS token: Learnable classification token
    3. Positional embedding: Learned or sinusoidal
    4. Transformer blocks: Stack of self-attention layers
    5. Output: CLS token (global) + patch tokens (spatial)
    
    Hyperparameters (ViT-Base):
    - embed_dim: 768
    - num_layers: 12
    - num_heads: 12
    - mlp_ratio: 4.0
    - patch_size: 16
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        use_learned_pos: bool = True,
        qkv_bias: bool = True
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding: Conv2d with stride=patch_size
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False  # No bias for patch embedding
        )
        
        # CLS token: Learnable classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Positional embedding
        if use_learned_pos:
            # Learned positional embeddings (recommended)
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
            )
        else:
            # Sinusoidal positional embeddings (non-learned)
            pos_embed = create_sinusoidal_pos_embedding(
                self.num_patches + 1, embed_dim
            )
            self.register_buffer('pos_embed', pos_embed)
        
        # Dropout for embeddings
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
                qkv_bias=qkv_bias
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using ViT initialization strategy."""
        # Patch embedding: Kaiming normal
        nn.init.kaiming_normal_(self.patch_embed.weight, mode='fan_out', nonlinearity='relu')
        
        # CLS token: Normal distribution
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Positional embedding: Normal distribution
        if isinstance(self.pos_embed, nn.Parameter):
            nn.init.normal_(self.pos_embed, std=0.02)
        
        # Transformer blocks: Xavier uniform for linear layers
        for block in self.blocks:
            for name, module in block.named_modules():
                if isinstance(module, nn.Linear):
                    if 'qkv' in name or 'attention' in name:
                        # QKV projection: smaller initialization
                        nn.init.xavier_uniform_(module.weight, gain=1.0 / math.sqrt(2))
                    else:
                        # Standard linear: Xavier uniform
                        nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.constant_(module.bias, 0.0)
                    nn.init.constant_(module.weight, 1.0)
    
    def forward(
        self,
        x: torch.Tensor,
        return_patch_tokens: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through Vision Transformer.
        
        Args:
            x: Input images [B, C, H, W]
            return_patch_tokens: Whether to return patch tokens
        
        Returns:
            cls_token: Global scene representation [B, embed_dim]
            patch_tokens: Spatial features [B, num_patches, embed_dim] (if return_patch_tokens)
        """
        B, C, H, W = x.shape
        
        # Validate input size
        if H != self.img_size or W != self.img_size:
            # Resize to expected size
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        # Handle NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input contains NaN or Inf values")
        
        # Normalize input if needed
        if x.max() > 1.0:
            x = x / 255.0  # Assume input is [0, 255]
        
        # Patch embedding
        # [B, C, H, W] -> [B, embed_dim, H/patch_size, W/patch_size]
        x = self.patch_embed(x)
        
        # Flatten spatial dimensions: [B, embed_dim, H', W'] -> [B, embed_dim, N_patches]
        # Then transpose: [B, N_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, embed_dim]
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Extract CLS token and patch tokens
        cls_token = x[:, 0]  # [B, embed_dim]
        
        if return_patch_tokens:
            patch_tokens = x[:, 1:]  # [B, num_patches, embed_dim]
            return cls_token, patch_tokens
        else:
            return cls_token, None
    
    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int = 4
    ) -> list:
        """
        Get intermediate layer outputs for feature extraction.
        
        Args:
            x: Input images [B, C, H, W]
            n: Number of layers to return (evenly spaced)
        
        Returns:
            List of intermediate outputs
        """
        B, C, H, W = x.shape
        
        # Patch embedding and CLS token
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Collect intermediate outputs
        intermediates = []
        layer_indices = [int(i * (len(self.blocks) - 1) / (n - 1)) for i in range(n)]
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in layer_indices:
                intermediates.append(x)
        
        return intermediates


