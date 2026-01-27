"""
Hybrid CNN + Vision Transformer Backbone for MaxSight 3.0

Combines local CNN features (ResNet50-FPN) with global ViT context.
Uses cross-layer connections for feature fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from .vit_backbone import VisionTransformerBackbone


class HybridCNNViTBackbone(nn.Module):
    """
    Hybrid backbone combining ResNet50-FPN (CNN) and Vision Transformer.
    
    Architecture:
    - CNN Branch: ResNet50-FPN extracts local multi-scale features
    - ViT Branch: Vision Transformer processes global context
    - Cross-Layer Connections: CNN ↔ ViT feature exchange
    - Feature Fusion: Multiple fusion strategies (concat, weighted, cross-attention)
    
    Fusion Methods:
    1. Concatenation: Simple concat + projection
    2. Weighted Sum: Learned weights for CNN/ViT
    3. Cross-Attention: CNN queries ViT features
    """
    
    def __init__(
        self,
        img_size: int = 224,
        cnn_dim: int = 256,  # FPN channels
        vit_dim: int = 768,  # ViT embed_dim
        fused_dim: int = 512,
        fusion_method: str = 'cross_attention',  # 'concat', 'weighted', 'cross_attention'
        use_cross_layer_connections: bool = True
    ):
        super().__init__()
        
        self.cnn_dim = cnn_dim
        self.vit_dim = vit_dim
        self.fused_dim = fused_dim
        self.fusion_method = fusion_method
        self.use_cross_layer_connections = use_cross_layer_connections
        
        # CNN Branch: ResNet50-FPN (will be initialized externally)
        # We'll accept it as a parameter or create it here
        # For now, assume it's passed in or we create a placeholder
        
        # ViT Branch
        self.vit_backbone = VisionTransformerBackbone(
            img_size=img_size,
            embed_dim=vit_dim,
            num_layers=12,
            num_heads=12
        )
        
        # Cross-layer connection projections
        if use_cross_layer_connections:
            # CNN → ViT projection
            self.cnn_to_vit_proj = nn.Conv2d(cnn_dim, vit_dim, 1)
            
            # ViT → CNN projection
            self.vit_to_cnn_proj = nn.Conv2d(vit_dim, cnn_dim, 1)
        
        # Feature fusion based on method
        if fusion_method == 'concat':
            # Concatenation fusion
            self.fusion_proj = nn.Linear(cnn_dim * 4 + vit_dim, fused_dim)
        elif fusion_method == 'weighted':
            # Weighted sum fusion
            self.cnn_proj = nn.Linear(cnn_dim * 4, fused_dim)
            self.vit_proj = nn.Linear(vit_dim, fused_dim)
            self.weight_cnn = nn.Parameter(torch.tensor(0.5))
            self.weight_vit = nn.Parameter(torch.tensor(0.5))
        elif fusion_method == 'cross_attention':
            # Cross-attention fusion
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=fused_dim,
                num_heads=8,
                batch_first=True
            )
            self.cnn_proj = nn.Linear(cnn_dim * 4, fused_dim)
            self.vit_proj = nn.Linear(vit_dim, fused_dim)
            self.norm = nn.LayerNorm(fused_dim)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def forward(
        self,
        images: torch.Tensor,
        cnn_features: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Forward pass through hybrid backbone.
        
        Args:
            images: Input images [B, 3, H, W]
            cnn_features: Optional pre-computed CNN features [P2, P3, P4, P5]
        
        Returns:
            fused_features: Fused global features [B, fused_dim]
            cnn_features: CNN multi-scale features [P2, P3, P4, P5]
            vit_patch_tokens: ViT patch tokens [B, num_patches, vit_dim]
        """
        B = images.shape[0]
        
        # ViT forward pass
        vit_cls, vit_patches = self.vit_backbone(images, return_patch_tokens=True)
        # vit_cls: [B, vit_dim]
        # vit_patches: [B, num_patches, vit_dim]
        
        # If CNN features not provided, we need to compute them
        # For now, assume they're provided or we'll need to integrate ResNet50-FPN
        # This is a placeholder - actual implementation will integrate with existing FPN
        if cnn_features is None:
            # In actual implementation, this would call ResNet50-FPN
            # For now, create placeholder
            H, W = images.shape[2], images.shape[3]
            cnn_features = [
                torch.zeros(B, self.cnn_dim, H//4, W//4, device=images.device),  # P2
                torch.zeros(B, self.cnn_dim, H//8, W//8, device=images.device),  # P3
                torch.zeros(B, self.cnn_dim, H//16, W//16, device=images.device),  # P4
                torch.zeros(B, self.cnn_dim, H//32, W//32, device=images.device),  # P5
            ]
        
        # Cross-layer connections
        if self.use_cross_layer_connections:
            # CNN → ViT: Inject CNN features into ViT
            # Use P4 as representative CNN feature
            p4 = cnn_features[2]  # [B, cnn_dim, H/16, W/16]
            
            # Resize to match ViT patch resolution (14x14 for 224x224 input)
            target_size = (14, 14)
            p4_resized = F.interpolate(p4, size=target_size, mode='bilinear', align_corners=False)
            
            # Project to ViT dimension
            p4_projected = self.cnn_to_vit_proj(p4_resized)  # [B, vit_dim, 14, 14]
            
            # Flatten and add to ViT patch tokens
            p4_flat = p4_projected.flatten(2).transpose(1, 2)  # [B, 196, vit_dim]
            vit_patches = vit_patches + p4_flat
            
            # ViT → CNN: Inject ViT features into CNN
            # Reshape ViT patches to spatial
            vit_spatial = vit_patches.transpose(1, 2).reshape(B, self.vit_dim, 14, 14)
            
            # Project to CNN dimension
            vit_projected = self.vit_to_cnn_proj(vit_spatial)  # [B, cnn_dim, 14, 14]
            
            # Add to CNN features at appropriate scales
            for i, fpn_feat in enumerate(cnn_features):
                vit_resized = F.interpolate(
                    vit_projected,
                    size=fpn_feat.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                cnn_features[i] = cnn_features[i] + vit_resized
        
        # Feature fusion
        # Global pooling on CNN features
        cnn_global = torch.cat([
            F.adaptive_avg_pool2d(fpn, 1).flatten(1) for fpn in cnn_features
        ], dim=1)  # [B, cnn_dim * 4]
        
        if self.fusion_method == 'concat':
            # Concatenation
            fused = torch.cat([cnn_global, vit_cls], dim=1)  # [B, cnn_dim*4 + vit_dim]
            fused = self.fusion_proj(fused)  # [B, fused_dim]
        
        elif self.fusion_method == 'weighted':
            # Weighted sum
            cnn_proj = self.cnn_proj(cnn_global)  # [B, fused_dim]
            vit_proj = self.vit_proj(vit_cls)  # [B, fused_dim]
            
            # Learned weights
            weights = F.softmax(torch.stack([
                self.weight_cnn.expand(B, 1),
                self.weight_vit.expand(B, 1)
            ], dim=1), dim=1)  # [B, 2]
            
            fused = weights[:, 0:1] * cnn_proj + weights[:, 1:2] * vit_proj
        
        elif self.fusion_method == 'cross_attention':
            # Cross-attention: CNN queries ViT
            cnn_proj = self.cnn_proj(cnn_global).unsqueeze(1)  # [B, 1, fused_dim]
            vit_proj = self.vit_proj(vit_patches)  # [B, num_patches, fused_dim]
            
            # Cross-attention
            fused, _ = self.cross_attention(
                query=cnn_proj,
                key=vit_proj,
                value=vit_proj
            )  # [B, 1, fused_dim]
            fused = self.norm(fused.squeeze(1))  # [B, fused_dim]
        
        return fused, cnn_features, vit_patches


