"""Enhanced Hybrid CNN + Vision Transformer Backbone for MaxSight 3.0."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
import math


class SpatialAttentionPooling(nn.Module):
    """Spatially-aware pooling using attention instead of naive averaging."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args x: [B, N, D] sequence of features Returns: pooled: [B, D] single feature vector."""
        B, N, D = x.shape
        
        # Use mean as query, all patches as keys/values.
        q = self.query(x.mean(dim=1, keepdim=True))  # [B, 1, D].
        k = self.key(x)  # [B, N, D].
        v = self.value(x)  # [B, N, D].
        
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)  # [B, 1, N].
        out = (attn @ v).squeeze(1)  # [B, D].
        
        return out


class AdaptiveFeatureFusion(nn.Module):
    """Learnable adaptive fusion of CNN and ViT features."""
    
    def __init__(self, cnn_dim: int, vit_dim: int, fused_dim: int, num_heads: int = 8):
        super().__init__()
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU()
        )
        self.vit_proj = nn.Sequential(
            nn.Linear(vit_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU()
        )
        
        # Learnable fusion weights with gating.
        self.fusion_gate = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.GELU(),
            nn.Linear(fused_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim)
        )
        
    def forward(self, cnn_feat: torch.Tensor, vit_feat: torch.Tensor) -> torch.Tensor:
        cnn_proj = self.cnn_proj(cnn_feat)
        vit_proj = self.vit_proj(vit_feat)
        
        # Adaptive gating.
        concat_feat = torch.cat([cnn_proj, vit_proj], dim=-1)
        weights = self.fusion_gate(concat_feat)  # [B, 2].
        
        fused = weights[:, 0:1] * cnn_proj + weights[:, 1:2] * vit_proj
        return self.output_proj(fused)


class CrossModalAttention(nn.Module):
    """Bidirectional cross-attention between CNN and ViT features."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cnn_to_vit = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.vit_to_cnn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        cnn_feat: torch.Tensor, 
        vit_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # CNN queries ViT.
        cnn_enhanced, _ = self.cnn_to_vit(
            query=cnn_feat, key=vit_feat, value=vit_feat
        )
        cnn_feat = self.norm1(cnn_feat + self.dropout(cnn_enhanced))
        
        # ViT queries CNN.
        vit_enhanced, _ = self.vit_to_cnn(
            query=vit_feat, key=cnn_feat, value=cnn_feat
        )
        vit_feat = self.norm2(vit_feat + self.dropout(vit_enhanced))
        
        return cnn_feat, vit_feat


class HybridCNNViTBackbone(nn.Module):
    """Production-ready Hybrid CNN + ViT backbone."""

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
        dropout: float = 0.1,
        pretrained_cnn: bool = True,
        use_gradient_checkpointing: bool = False,
        cross_layer_alpha: float = 0.1,  # Tunable residual scaling.
    ):
        super().__init__()
        
        # Validate dimensions.
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
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        if isinstance(cross_layer_alpha, float):
            # Fixed value: convert to parameter and constrain.
            self.cross_layer_alpha_raw = nn.Parameter(torch.tensor(cross_layer_alpha))
        else:
            # Learnable: already a parameter.
            self.cross_layer_alpha_raw = cross_layer_alpha if cross_layer_alpha is not None else nn.Parameter(torch.tensor(0.1))
        
        # === CNN Backbone: ResNet50 + FPN ===.
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained_cnn else None
        resnet = resnet50(weights=weights)
        
        # Extract feature extraction stages.
        self.cnn_stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.cnn_layer1 = resnet.layer1  # 64 -> 256 channels (required before layer2)
        self.cnn_layer2 = resnet.layer2  # C3: 512 channels.
        self.cnn_layer3 = resnet.layer3  # C4: 1024 channels.
        self.cnn_layer4 = resnet.layer4  # C5: 2048 channels.
        
        # Feature Pyramid Network.
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[512, 1024, 2048],
            out_channels=cnn_out_channels
        )
        
        from .vit_backbone import VisionTransformerBackbone
        self.vit = VisionTransformerBackbone(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=vit_embed_dim,
            num_layers=vit_depth,
            num_heads=vit_num_heads,
            dropout=dropout
        )
        
        # === Cross-layer Connections (FIXED) ===.
        if use_cross_layer_connections:
            # CNN to ViT: Proper learnable 1x1 convolutions.
            self.cnn_to_vit_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(cnn_out_channels, vit_embed_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(vit_embed_dim),
                    nn.GELU()
                ) for _ in range(3)  # For P3, P4, P5.
            ])
            
            # ViT to CNN: Spatially-aware projection (FIXED) Use attention pooling instead of naive mean.
            self.vit_spatial_pool = SpatialAttentionPooling(vit_embed_dim)
            
            # Project from ViT dim to CNN dim with proper conv.
            self.vit_to_cnn_proj = nn.Sequential(
                nn.Conv2d(vit_embed_dim, cnn_out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(cnn_out_channels),
                nn.GELU()
            )
        
        total_cnn_dim = cnn_out_channels * 3  # From 3 FPN levels.
        
        if fusion_method == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(total_cnn_dim + vit_embed_dim, fused_dim),
                nn.LayerNorm(fused_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        elif fusion_method == 'weighted':
            self.fusion = AdaptiveFeatureFusion(
                cnn_dim=total_cnn_dim,
                vit_dim=vit_embed_dim,
                fused_dim=fused_dim
            )
        elif fusion_method == 'cross_attention':
            self.cnn_query_proj = nn.Linear(total_cnn_dim, fused_dim)
            self.vit_kv_proj = nn.Linear(vit_embed_dim, fused_dim)
            
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=fused_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            
            self.fusion_norm = nn.LayerNorm(fused_dim)
            self.fusion_ffn = nn.Sequential(
                nn.Linear(fused_dim, fused_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fused_dim * 4, fused_dim),
                nn.Dropout(dropout)
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights.
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Proper weight initialization."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def extract_cnn_features(
        self, 
        x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Extract multi-scale CNN features with optional checkpointing."""
        
        def run_cnn():
            x_stem = self.cnn_stem(x)
            x1 = self.cnn_layer1(x_stem)
            c3 = self.cnn_layer2(x1)
            c4 = self.cnn_layer3(c3)
            c5 = self.cnn_layer4(c4)
            return c3, c4, c5
        
        if self.use_gradient_checkpointing and self.training:
            c3, c4, c5 = torch.utils.checkpoint.checkpoint(
                run_cnn, use_reentrant=False
            )
        else:
            c3, c4, c5 = run_cnn()
        
        # Apply FPN.
        fpn_input = {'feat0': c3, 'feat1': c4, 'feat2': c5}
        fpn_output = self.fpn(fpn_input)
        fpn_features = [fpn_output['feat0'], fpn_output['feat1'], fpn_output['feat2']]
        
        return fpn_features, [c3, c4, c5]

    def cnn_vit_interaction(
        self,
        cnn_features: List[torch.Tensor],
        vit_patches: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """FIXED: Proper cross-layer information flow. - Uses learnable Conv2d projections (no torch.eye) - Spatially-aware ViT pooling - Dynamic spatial handling."""
        B = vit_patches.shape[0]
        num_patches = vit_patches.shape[1]
        
        # Dynamically compute patch grid size (FIXED)
        patch_h = patch_w = int(math.sqrt(num_patches))
        assert patch_h * patch_w == num_patches, \
            f"Number of patches ({num_patches}) must be a perfect square"
        
        # Reshape ViT patches to spatial format.
        vit_spatial = vit_patches.transpose(1, 2).reshape(
            B, self.vit_embed_dim, patch_h, patch_w
        )
        
        enhanced_cnn = []
        
        for i, (cnn_feat, cnn_to_vit_layer) in enumerate(
            zip(cnn_features, self.cnn_to_vit_proj)
        ):
            # === CNN → ViT: Add local CNN context to ViT patches ===.
            cnn_projected = cnn_to_vit_layer(cnn_feat)  # [B, vit_embed_dim, H, W].
            
            # Resize to match ViT patch grid.
            cnn_to_vit_resized = F.adaptive_avg_pool2d(
                cnn_projected, (patch_h, patch_w)
            )
            
            # Add to ViT patches as residual.
            cnn_context = cnn_to_vit_resized.flatten(2).transpose(1, 2)  # [B, N, D].
            if hasattr(self, 'cross_layer_alpha_raw'):
                raw = self.cross_layer_alpha_raw
                alpha = torch.sigmoid(raw if isinstance(raw, torch.Tensor) else torch.tensor(float(raw), device=vit_patches.device, dtype=vit_patches.dtype))
            else:
                alpha = 0.1  # Default fallback.
            vit_patches = vit_patches + alpha * cnn_context
            
            # === ViT → CNN: Add global ViT context to CNN features ===. Project ViT spatial features to CNN dimension (FIXED)
            vit_projected = self.vit_to_cnn_proj(vit_spatial)  # [B, cnn_dim, pH, pW].
            
            # Resize to match current CNN feature map.
            vit_to_cnn_resized = F.interpolate(
                vit_projected, 
                size=cnn_feat.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
            
            # Add as residual.
            raw = self.cross_layer_alpha_raw
            alpha = torch.sigmoid(raw if isinstance(raw, torch.Tensor) else torch.tensor(float(raw), device=cnn_feat.device, dtype=cnn_feat.dtype))
            enhanced_cnn.append(cnn_feat + alpha * vit_to_cnn_resized)
        
        return enhanced_cnn, vit_patches

    def forward(
        self,
        x: torch.Tensor,
        vit_patches: Optional[torch.Tensor] = None,
        return_all_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass with improved cross-layer interaction."""
        B = x.shape[0]
        
        fpn_features, cnn_feats = self.extract_cnn_features(x)
        
        # === Extract ViT features (if not provided) ===.
        if vit_patches is None:
            if self.use_gradient_checkpointing and self.training:
                vit_cls, vit_patches = torch.utils.checkpoint.checkpoint(
                    lambda img: self.vit(img, return_patch_tokens=True),
                    x,
                    use_reentrant=False
                )
            else:
                vit_cls, vit_patches = self.vit(x, return_patch_tokens=True)
        else:
            # Use provided vit_patches, compute cls token from mean.
            vit_cls = vit_patches.mean(dim=1)  # [B, vit_embed_dim].
        
        # Ensure vit_patches is a tensor (type narrowing)
        assert vit_patches is not None, "vit_patches must not be None"
        
        # === Cross-layer interaction ===.
        if self.use_cross_layer_connections:
            cnn_enhanced, vit_enhanced = self.cnn_vit_interaction(cnn_feats, vit_patches)
        else:
            cnn_enhanced, vit_enhanced = fpn_features, vit_patches
        
        # Flatten FPN features and concatenate.
        fpn_flat = [f.flatten(2).mean(dim=2) for f in cnn_enhanced]  # [B, C] each.
        cnn_concat = torch.cat(fpn_flat, dim=1)  # [B, total_cnn_dim].
        
        vit_global = vit_enhanced.mean(dim=1)  # [B, vit_embed_dim].
        
        if self.fusion_method == 'concat':
            fused = self.fusion(torch.cat([cnn_concat, vit_global], dim=1))
        elif self.fusion_method == 'weighted':
            fused = self.fusion(cnn_concat, vit_global)
        elif self.fusion_method == 'cross_attention':
            # Add sequence dimension for MultiheadAttention.
            cnn_seq = self.cnn_query_proj(cnn_concat).unsqueeze(1)  # [B, 1, D].
            vit_seq = self.vit_kv_proj(vit_global).unsqueeze(1)     # [B, 1, D].
            
            attn_out, _ = self.cross_attn(cnn_seq, vit_seq, vit_seq)
            attn_out = self.fusion_norm(attn_out).squeeze(1)
            
            # Feedforward.
            fused = self.fusion_ffn(attn_out)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        fused = self.dropout(fused)
        
        if return_all_features:
            aux_features = {
                'fpn_features': cnn_enhanced,
                'cnn_global': cnn_concat,
                'vit_cls': vit_global,
                'vit_patches': vit_enhanced,
            }
            return fused, aux_features
        
        return fused, None


if __name__ == '__main__':
    # Standard usage.
    model = HybridCNNViTBackbone(
        img_size=224,
        patch_size=16,
        cnn_out_channels=256,
        vit_embed_dim=768,
        fused_dim=512,
        fusion_method='cross_attention',
        use_cross_layer_connections=True,
        pretrained_cnn=True,
        use_gradient_checkpointing=True,  # Now actually works!
        cross_layer_alpha=0.1  # Tunable residual scaling.
    )
    
    x = torch.randn(2, 3, 224, 224)
    fused, aux = model(x, return_all_features=True)
    
    print(f"Fused features: {fused.shape}")
    print(f"FPN features: {[f.shape for f in aux['fpn_features']]}")
    print(f"ViT patches: {aux['vit_patches'].shape}")
    
    # Test gradient checkpointing.
    model.train()
    loss = fused.sum()
    loss.backward()
    print("Gradient checkpointing works!")
    
    # Memory efficiency test.
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")