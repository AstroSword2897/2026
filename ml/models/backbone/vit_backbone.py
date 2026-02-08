"""Ultra-Optimized Hybrid CNN + Vision Transformer Backbone for MaxSight 3.0."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
import math

# Import xformers for memory-efficient attention when available.
try:
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False


def create_sinusoidal_pos_embedding(num_positions: int, embed_dim: int) -> torch.Tensor:
    """Create sinusoidal positional embeddings (non-learned alternative). Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d_model)) PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))"""
    position = torch.arange(num_positions).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                        -(math.log(10000.0) / embed_dim))
    
    pos_embed = torch.zeros(num_positions, embed_dim)
    pos_embed[:, 0::2] = torch.sin(position * div_term)
    pos_embed[:, 1::2] = torch.cos(position * div_term)
    
    return pos_embed.unsqueeze(0)  # [1, num_positions, embed_dim].


class TransformerBlock(nn.Module):
    """Single Transformer encoder block with pre-norm architecture."""
    
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
        
        # Multi-head self-attention.
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            bias=qkv_bias,
            batch_first=True
        )
        
        # Feed-forward network.
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
        """Forward pass through transformer block. Args: x: Input tokens [B, N, embed_dim] Returns: Output tokens [B, N, embed_dim]."""
        # Pre-norm attention with residual.
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attention(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        
        # Pre-norm FFN with residual.
        x_norm = self.norm2(x)
        ffn_out = self.mlp(x_norm)
        x = x + ffn_out
        
        return x


class VisionTransformerBackbone(nn.Module):
    """Complete Vision Transformer backbone."""
    
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
        qkv_bias: bool = True,
        use_flash_attention: bool = False  # Added for compatibility.
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding: Conv2d with stride=patch_size.
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False  # No bias for patch embedding.
        )
        
        # CLS token: Learnable classification token.
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Positional embedding.
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
        
        # Dropout for embeddings.
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer blocks.
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
        
        # Final layer norm.
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Initialize weights.
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using ViT initialization strategy."""
        # Patch embedding: Kaiming normal.
        nn.init.kaiming_normal_(self.patch_embed.weight, mode='fan_out', nonlinearity='relu')
        
        # CLS token: Normal distribution.
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Positional embedding: Normal distribution.
        if isinstance(self.pos_embed, nn.Parameter):
            nn.init.normal_(self.pos_embed, std=0.02)
        
        # Transformer blocks: Xavier uniform for linear layers.
        for block in self.blocks:
            for name, module in block.named_modules():
                if isinstance(module, nn.Linear):
                    if 'qkv' in name or 'attention' in name:
                        # QKV projection: smaller initialization.
                        nn.init.xavier_uniform_(module.weight, gain=1.0 / math.sqrt(2))
                    else:
                        # Standard linear: Xavier uniform.
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
        """Forward pass through Vision Transformer."""
        B, C, H, W = x.shape
        
        # Validate input size.
        assert H == self.img_size and W == self.img_size, \
            f"Input size {H}x{W} must match img_size {self.img_size}"
        
        # Patch embedding. [B, C, H, W] -> [B, embed_dim, H/patch_size, W/patch_size].
        x = self.patch_embed(x)
        
        # Then transpose: [B, N_patches, embed_dim].
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim].
        
        # Add CLS token.
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim].
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, embed_dim].
        
        # Add positional embedding.
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Apply transformer blocks.
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm.
        x = self.norm(x)
        
        # Extract CLS token and patch tokens.
        cls_token = x[:, 0]  # [B, embed_dim].
        
        if return_patch_tokens:
            patch_tokens = x[:, 1:]  # [B, num_patches, embed_dim].
            return cls_token, patch_tokens
        else:
            return cls_token, None
    
    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int = 4
    ) -> list:
        """Get intermediate layer outputs for feature extraction."""
        B, C, H, W = x.shape
        
        # Patch embedding and CLS token.
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Collect intermediate outputs.
        intermediates = []
        layer_indices = [int(i * (len(self.blocks) - 1) / (n - 1)) for i in range(n)]
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in layer_indices:
                intermediates.append(x)
        
        return intermediates


class MultiHeadEfficientPooling(nn.Module):
    """Multi-head attention pooling with minimal overhead."""
    
    def __init__(self, dim: int, num_heads: int = 2):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Fused QKV projection.
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args x: [B, N, D] Returns: pooled: [B, D]."""
        B, N, D = x.shape
        
        # Mean as query, all patches as keys/values.
        x_with_query = torch.cat([x.mean(dim=1, keepdim=True), x], dim=1)  # [B, N+1, D].
        
        qkv = self.qkv(x_with_query).reshape(B, N+1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N+1, head_dim].
        q, k, v = qkv.unbind(0)
        
        q = q[:, :, :1]  # [B, H, 1, head_dim].
        k = k[:, :, 1:]  # [B, H, N, head_dim].
        v = v[:, :, 1:]  # [B, H, N, head_dim].
        
        # Efficient attention.
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, D)
        
        return self.proj(out)


class ImprovedAdaptiveFusion(nn.Module):
    """Differentiable fusion with better gradient flow."""
    
    def __init__(self, cnn_dim: int, vit_dim: int, fused_dim: int):
        super().__init__()
        self.cnn_dim = cnn_dim
        self.vit_dim = vit_dim
        
        # Separate projections for better control.
        self.cnn_proj = nn.Linear(cnn_dim, fused_dim, bias=False)
        self.vit_proj = nn.Linear(vit_dim, fused_dim, bias=False)
        self.fused_proj = nn.Linear(cnn_dim + vit_dim, fused_dim, bias=False)
        
        self.norm = nn.LayerNorm(fused_dim, eps=1e-6)  # Better for mixed precision.
        
        # Learnable gates per modality (no detach!)
        self.gate_cnn = nn.Parameter(torch.tensor(0.5))
        self.gate_vit = nn.Parameter(torch.tensor(0.5))
        self.gate_fused = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, cnn_feat: torch.Tensor, vit_feat: torch.Tensor) -> torch.Tensor:
        # Three pathways with learnable weights.
        cnn_path = self.cnn_proj(cnn_feat)
        vit_path = self.vit_proj(vit_feat)
        fused_path = self.fused_proj(torch.cat([cnn_feat, vit_feat], dim=-1))
        
        # Normalize gates.
        gates = torch.sigmoid(torch.stack([self.gate_cnn, self.gate_vit, self.gate_fused]))
        gates = gates / gates.sum()
        
        # Weighted combination with full gradient flow.
        out = gates[0] * cnn_path + gates[1] * vit_path + gates[2] * fused_path
        return self.norm(out)


class FusedConvBNActivation(nn.Module):
    """Fused Conv+BN+Activation for maximum efficiency."""
    
    def __init__(self, in_channels, out_channels, kernel_size=1, activation='gelu'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class EfficientCrossModalAttention(nn.Module):
    """Bidirectional cross-attention with Flash Attention and in-place ops."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_flash = XFORMERS_AVAILABLE
        
        if self.use_flash:
            # Xformers memory-efficient attention.
            self.scale = (dim // num_heads) ** -0.5
            self.qkv_cnn = nn.Linear(dim, dim * 3, bias=False)
            self.qkv_vit = nn.Linear(dim, dim * 3, bias=False)
            self.proj_cnn = nn.Linear(dim, dim, bias=False)
            self.proj_vit = nn.Linear(dim, dim, bias=False)
        else:
            # Standard attention.
            self.cnn_to_vit = nn.MultiheadAttention(
                embed_dim=dim, num_heads=num_heads, dropout=dropout, 
                batch_first=True, bias=False
            )
            self.vit_to_cnn = nn.MultiheadAttention(
                embed_dim=dim, num_heads=num_heads, dropout=dropout, 
                batch_first=True, bias=False
            )
        
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        cnn_feat: torch.Tensor, 
        vit_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_flash:
            # Flash attention path.
            B = cnn_feat.shape[0]
            
            # CNN queries ViT.
            qkv_cnn = self.qkv_cnn(cnn_feat).reshape(B, -1, 3, self.num_heads, self.dim // self.num_heads)
            q_cnn, _, _ = qkv_cnn.unbind(2)
            
            qkv_vit = self.qkv_vit(vit_feat).reshape(B, -1, 3, self.num_heads, self.dim // self.num_heads)
            _, k_vit, v_vit = qkv_vit.unbind(2)
            
            cnn_enhanced = xops.memory_efficient_attention(q_cnn, k_vit, v_vit, scale=self.scale)
            cnn_enhanced = cnn_enhanced.reshape(B, -1, self.dim)
            cnn_enhanced = self.proj_cnn(cnn_enhanced)
            
            # ViT queries CNN.
            q_vit, _, _ = qkv_vit.unbind(2)
            _, k_cnn, v_cnn = qkv_cnn.unbind(2)
            
            vit_enhanced = xops.memory_efficient_attention(q_vit, k_cnn, v_cnn, scale=self.scale)
            vit_enhanced = vit_enhanced.reshape(B, -1, self.dim)
            vit_enhanced = self.proj_vit(vit_enhanced)
        else:
            # Standard attention.
            cnn_enhanced, _ = self.cnn_to_vit(cnn_feat, vit_feat, vit_feat)
            vit_enhanced, _ = self.vit_to_cnn(vit_feat, cnn_feat, cnn_feat)
        
        # In-place residual addition.
        cnn_feat = self.norm1(cnn_feat.add_(self.dropout(cnn_enhanced)))
        vit_feat = self.norm2(vit_feat.add_(self.dropout(vit_enhanced)))
        
        return cnn_feat, vit_feat


class FeatureCache:
    """Feature cache for repeated forward passes."""
    
    def __init__(self, max_size: int = 8, use_frame_id: bool = True):
        self.cache = {}
        self.max_size = max_size
        self.keys = []
        self.use_frame_id = use_frame_id
        self.frame_counter = 0
        
    def get(self, key: str):
        return self.cache.get(key, None)
    
    def set(self, key: str, value: torch.Tensor):
        if len(self.keys) >= self.max_size:
            old_key = self.keys.pop(0)
            del self.cache[old_key]
        self.cache[key] = value.detach()
        self.keys.append(key)
    
    def clear(self):
        self.cache.clear()
        self.keys.clear()
        self.frame_counter = 0
    
    def _make_cache_key(self, x: torch.Tensor, frame_id: Optional[int] = None) -> str:
        """FIXED: Use frame ID/timestamp instead of mean hash to prevent collisions."""
        if self.use_frame_id and frame_id is not None:
            # Use frame ID for deterministic, collision-free caching.
            return f"frame_{frame_id}"
        elif self.use_frame_id:
            # Fallback: use counter (not perfect but better than mean)
            key = f"frame_{self.frame_counter}"
            self.frame_counter += 1
            return key
        else:
            # Legacy: mean-based (unsafe, but kept for compatibility)
            return f"cnn_{x.shape}_{x.mean().item():.6f}"


class HybridCNNViTBackbone(nn.Module):
    """Ultra-optimized Hybrid CNN + ViT backbone."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        cnn_out_channels: int = 256,
        vit_embed_dim: int = 768,
        vit_depth: int = 12,
        vit_num_heads: int = 12,
        fused_dim: int = 512,
        fusion_method: str = 'weighted',
        use_cross_layer_connections: bool = True,
        use_bidirectional_attention: bool = True,
        dropout: float = 0.1,
        pretrained_cnn: bool = True,
        use_gradient_checkpointing: bool = False,
        cross_layer_alpha: Optional[float] = None,  # Now learnable if None.
        use_flash_attention: bool = True,
        compile_model: bool = False,
        fpn_levels: List[int] = [3, 4, 5],
        enable_feature_cache: bool = False,
    ):
        super().__init__()
        
        assert img_size % patch_size == 0, \
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_grid_size = img_size // patch_size
        self.cnn_out_channels = cnn_out_channels
        self.vit_embed_dim = vit_embed_dim
        self.fused_dim = fused_dim
        self.fusion_method = fusion_method
        self.use_cross_layer_connections = use_cross_layer_connections
        self.use_bidirectional_attention = use_bidirectional_attention
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.fpn_levels = fpn_levels
        self.enable_feature_cache = enable_feature_cache
        
        if cross_layer_alpha is None:
            # Learnable parameter, will be constrained with sigmoid.
            self.cross_layer_alpha_raw = nn.Parameter(torch.tensor(0.1))
        elif isinstance(cross_layer_alpha, float):
            # Fixed value: convert to parameter and constrain.
            self.cross_layer_alpha_raw = nn.Parameter(torch.tensor(cross_layer_alpha))
        else:
            # Already a parameter.
            self.cross_layer_alpha_raw = cross_layer_alpha
        
        # Feature cache.
        if enable_feature_cache:
            self.cache = FeatureCache()
        
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained_cnn else None
        resnet = resnet50(weights=weights)
        
        self.cnn_stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        
        # Selective layer loading.
        self.cnn_layers = nn.ModuleDict()
        self.fpn_in_channels = []
        
        layer_mapping = {3: ('layer2', 512), 4: ('layer3', 1024), 5: ('layer4', 2048)}
        for level in sorted(fpn_levels):
            name, channels = layer_mapping[level]
            self.cnn_layers[name] = getattr(resnet, name)
            self.fpn_in_channels.append(channels)
        
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.fpn_in_channels,
            out_channels=cnn_out_channels
        )
        
        # Import VisionTransformerBackbone from the module (avoid circular import)
        try:
            # Import from the same module (if defined later in file)
            import sys
            current_module = sys.modules[__name__]
            if hasattr(current_module, 'VisionTransformerBackbone'):
                VisionTransformerBackbone = getattr(current_module, 'VisionTransformerBackbone')
            else:
                # Import from package when not defined in this file.
                from ml.models.backbone import VisionTransformerBackbone
        except (ImportError, AttributeError):
            raise ImportError(
                "VisionTransformerBackbone not found. "
                "It should be defined in ml.models.backbone.vit_backbone or imported from another module."
            )
        
        self.vit = VisionTransformerBackbone(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=vit_embed_dim,
            num_layers=vit_depth,
            num_heads=vit_num_heads,
            dropout=dropout,
            use_flash_attention=use_flash_attention
        )
        
        # === Cross-layer Connections ===.
        if use_cross_layer_connections:
            # Fused projections.
            self.cnn_to_vit_proj = nn.ModuleList([
                FusedConvBNActivation(cnn_out_channels, vit_embed_dim)
                for _ in range(len(fpn_levels))
            ])
            
            # Multi-head pooling.
            self.vit_spatial_pool = MultiHeadEfficientPooling(vit_embed_dim, num_heads=2)
            
            # Fused projection.
            self.vit_to_cnn_proj = FusedConvBNActivation(vit_embed_dim, cnn_out_channels)
            
            # Optional bidirectional attention.
            if use_bidirectional_attention:
                self.cross_modal_attn = EfficientCrossModalAttention(
                    dim=min(cnn_out_channels, vit_embed_dim),
                    num_heads=8,
                    dropout=dropout
                )
                # Alignment projections.
                if cnn_out_channels != vit_embed_dim:
                    self.cnn_align = nn.Linear(cnn_out_channels * len(fpn_levels), 
                                              min(cnn_out_channels, vit_embed_dim), bias=False)
                    self.vit_align = nn.Linear(vit_embed_dim, 
                                              min(cnn_out_channels, vit_embed_dim), bias=False)
        
        total_cnn_dim = cnn_out_channels * len(fpn_levels)
        
        if fusion_method == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(total_cnn_dim + vit_embed_dim, fused_dim, bias=False),
                nn.LayerNorm(fused_dim, eps=1e-6),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        elif fusion_method == 'weighted':
            self.fusion = ImprovedAdaptiveFusion(
                cnn_dim=total_cnn_dim,
                vit_dim=vit_embed_dim,
                fused_dim=fused_dim
            )
        elif fusion_method == 'cross_attention':
            self.cnn_query_proj = nn.Linear(total_cnn_dim, fused_dim, bias=False)
            self.vit_kv_proj = nn.Linear(vit_embed_dim, fused_dim, bias=False)
            
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=fused_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True,
                bias=False
            )
            
            self.fusion_norm = nn.LayerNorm(fused_dim, eps=1e-6)
            self.fusion_ffn = nn.Sequential(
                nn.Linear(fused_dim, fused_dim * 4, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fused_dim * 4, fused_dim, bias=False),
                nn.Dropout(dropout)
            )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize.
        self.apply(self._init_weights)
        
        # Compile if requested.
        if compile_model and hasattr(torch, 'compile'):
            print("Compiling model components with torch.compile...")
            self.cnn_stem = torch.compile(self.cnn_stem, mode='max-autotune')
            for key in self.cnn_layers:
                self.cnn_layers[key] = torch.compile(self.cnn_layers[key], mode='max-autotune')
            self.fpn = torch.compile(self.fpn, mode='max-autotune')
    
    def _init_weights(self, m):
        """Optimized initialization for mixed precision."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def extract_cnn_features(self, x: torch.Tensor, frame_id: Optional[int] = None) -> List[torch.Tensor]:
        """Extract CNN features with optional caching. FIXED: Uses frame_id for cache key instead of mean hash. Caching is experimental and disabled by default in safety-critical paths."""
        
        if self.enable_feature_cache and not self.training:
            # FIXED: Use frame_id-based cache key instead of mean hash.
            cache_key = self.cache._make_cache_key(x, frame_id)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        
        def run_cnn():
            features = []
            x_stem = self.cnn_stem(x)
            x_current = x_stem
            
            for level in sorted(self.fpn_levels):
                layer_name = f'layer{level - 1}'
                x_current = self.cnn_layers[layer_name](x_current)
                features.append(x_current)
            
            return features
        
        if self.use_gradient_checkpointing and self.training:
            features = torch.utils.checkpoint.checkpoint(run_cnn, use_reentrant=False)
        else:
            features = run_cnn()
        
        # FPN.
        fpn_input = {f'feat{i}': feat for i, feat in enumerate(features)}
        fpn_output = self.fpn(fpn_input)
        fpn_features = [fpn_output[f'feat{i}'] for i in range(len(features))]
        
        if self.enable_feature_cache and not self.training:
            # FIXED: Use frame_id-based cache key.
            cache_key = self.cache._make_cache_key(x, frame_id)
            # Cache as tuple to avoid type issues.
            self.cache.set(cache_key, tuple(fpn_features))
        
        return fpn_features

    def cnn_vit_interaction(
        self,
        cnn_features: List[torch.Tensor],
        vit_patches: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Optimized cross-modal interaction with in-place ops."""
        B, N, D = vit_patches.shape
        patch_h = patch_w = int(math.sqrt(N))
        
        vit_spatial = vit_patches.transpose(1, 2).reshape(B, D, patch_h, patch_w)
        vit_projected = self.vit_to_cnn_proj(vit_spatial)
        
        enhanced_cnn = []
        if hasattr(self, 'cross_layer_alpha_raw'):
            alpha = torch.sigmoid(self.cross_layer_alpha_raw)
        else:
            alpha = 0.1  # Default fallback.
        
        for cnn_feat, proj_layer in zip(cnn_features, self.cnn_to_vit_proj):
            # CNN → ViT.
            cnn_proj = proj_layer(cnn_feat)
            cnn_pooled = F.adaptive_avg_pool2d(cnn_proj, (patch_h, patch_w))
            vit_patches = vit_patches.add_(
                cnn_pooled.flatten(2).transpose(1, 2).mul_(alpha)
            )
            
            # ViT → CNN.
            H, W = cnn_feat.shape[2:]
            vit_resized = F.interpolate(vit_projected, (H, W), mode='bilinear', align_corners=False) \
                          if (H, W) != vit_projected.shape[2:] else vit_projected
            
            enhanced_cnn.append(cnn_feat.add(vit_resized.mul(alpha)))
        
        return enhanced_cnn, vit_patches

    def forward(
        self,
        images: torch.Tensor,
        return_all_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Ultra-optimized forward pass."""
        
        B, _, H, W = images.shape
        if H != self.img_size or W != self.img_size:
            raise ValueError(f"Input size mismatch: {H}x{W} vs {self.img_size}x{self.img_size}")
        
        # Extract features.
        fpn_features = self.extract_cnn_features(images)
        
        if self.use_gradient_checkpointing and self.training:
            vit_cls, vit_patches = torch.utils.checkpoint.checkpoint(
                lambda x: self.vit(x, return_patch_tokens=True),
                images,
                use_reentrant=False
            )
        else:
            vit_cls, vit_patches = self.vit(images, return_patch_tokens=True)
        
        # Cross-layer interaction.
        if self.use_cross_layer_connections:
            fpn_features, vit_patches = self.cnn_vit_interaction(fpn_features, vit_patches)
            
            # Optional bidirectional attention.
            if self.use_bidirectional_attention:
                cnn_global = torch.cat([
                    F.adaptive_avg_pool2d(f, 1).flatten(1) for f in fpn_features
                ], dim=1)
                
                if hasattr(self, 'cnn_align'):
                    cnn_aligned = self.cnn_align(cnn_global).unsqueeze(1)
                    vit_aligned = self.vit_align(vit_patches)
                else:
                    cnn_aligned = cnn_global.unsqueeze(1)
                    vit_aligned = vit_patches
                
                cnn_aligned, vit_aligned = self.cross_modal_attn(cnn_aligned, vit_aligned)
                vit_cls = vit_aligned.mean(dim=1)
        
        # Global pooling.
        cnn_global = torch.cat([
            F.adaptive_avg_pool2d(f, 1).flatten(1) for f in fpn_features
        ], dim=1)
        
        # Fusion.
        if self.fusion_method == 'concat':
            fused = self.fusion(torch.cat([cnn_global, vit_cls], dim=1))
        elif self.fusion_method == 'weighted':
            fused = self.fusion(cnn_global, vit_cls)
        elif self.fusion_method == 'cross_attention':
            cnn_q = self.cnn_query_proj(cnn_global).unsqueeze(1)
            vit_kv = self.vit_kv_proj(vit_patches)
            attn_out, _ = self.cross_attn(cnn_q, vit_kv, vit_kv)
            fused = self.fusion_norm(cnn_q + attn_out)
            fused = fused + self.fusion_ffn(fused)
            fused = fused.squeeze(1)
        
        fused = self.dropout(fused)
        
        if return_all_features:
            return fused, {
                'fpn_features': fpn_features,
                'cnn_global': cnn_global,
                'vit_cls': vit_cls,
                'vit_patches': vit_patches,
            }
        
        return fused, None


if __name__ == '__main__':
    model = HybridCNNViTBackbone(
        img_size=224,
        patch_size=16,
        fusion_method='weighted',
        use_bidirectional_attention=True,
        compile_model=True,
        fpn_levels=[4, 5],
        enable_feature_cache=False,
        cross_layer_alpha=None,  # Learnable.
    ).cuda()
    
    print(f"Flash Attention: {'yes' if XFORMERS_AVAILABLE else 'no'}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    x = torch.randn(4, 3, 224, 224).cuda()
    fused, _ = model(x)
    print(f"Output shape: {fused.shape}")
    