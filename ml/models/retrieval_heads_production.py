"""Production-Ready Multi-Vector Retrieval Heads for MaxSight 3.0."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from ml.retrieval.encoders.global_encoder import GlobalEncoder
from ml.retrieval.encoders.region_extractor import RegionExtractor
from ml.retrieval.encoders.patch_extractor import PatchExtractor
from ml.retrieval.encoders.depth_extractor import DepthExtractor
from ml.retrieval.encoders.ocr_encoder import OCREncoder
from ml.retrieval.encoders.audio_encoder import AudioEncoder
from ml.retrieval.encoders.scene_graph_encoder import SceneGraphRetrievalEncoder


class MultiVectorRetrievalHeads(nn.Module):
    """Production-ready multi-vector retrieval heads."""
    
    def __init__(
        self,
        global_dim: int = 512,
        region_dim: int = 256,
        patch_dim: int = 768,
        depth_dim: int = 256,
        ocr_dim: int = 384,
        audio_dim: int = 256,
        scene_graph_dim: int = 512,
        common_embed_dim: int = 256,  # Common space for all modalities.
        max_regions: int = 8,
        max_patches: int = 25,
        max_texts: int = 10
    ):
        super().__init__()
        
        # Encoders.
        self.global_encoder = GlobalEncoder(embed_dim=global_dim)
        self.region_extractor = RegionExtractor(max_regions=max_regions, region_size=(224, 224))
        self.patch_extractor = PatchExtractor(embed_dim=patch_dim, num_clusters=max_patches)
        self.depth_extractor = DepthExtractor(embed_dim=depth_dim)
        self.ocr_encoder = OCREncoder(embed_dim=ocr_dim, max_texts=max_texts)
        self.audio_encoder = AudioEncoder(embed_dim=audio_dim)
        self.scene_graph_encoder = SceneGraphRetrievalEncoder(embed_dim=scene_graph_dim)
        
        self.global_proj = nn.Linear(global_dim, common_embed_dim)
        self.region_proj = nn.Linear(region_dim, common_embed_dim)
        self.patch_proj = nn.Linear(patch_dim, common_embed_dim)
        self.depth_proj = nn.Linear(depth_dim, common_embed_dim)
        self.ocr_proj = nn.Linear(ocr_dim, common_embed_dim)
        self.audio_proj = nn.Linear(audio_dim, common_embed_dim)
        self.sg_proj = nn.Linear(scene_graph_dim, common_embed_dim)
        
        self.common_embed_dim = common_embed_dim
        self.max_regions = max_regions
        self.max_patches = max_patches
        self.max_texts = max_texts
    
    def forward(
        self,
        images: torch.Tensor,  # [B, 3, H, W].
        audio: Optional[torch.Tensor] = None,  # [B, audio_features].
        text_snippets: Optional[List[List[str]]] = None,  # Variable length per image.
        scene_graph: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """Extract all embedding types in common space."""
        B = images.shape[0]
        device = images.device
        embeddings = {}
        
        # Global embedding.
        global_emb = self.global_encoder(images)  # [B, global_dim].
        embeddings['global'] = self.global_proj(global_emb)  # [B, common_embed_dim].
        
        # Region embeddings (with boxes)
        region_emb, region_boxes = self.region_extractor.extract_regions(images)
        # Preserve region boxes for downstream consumers.
        embeddings['region'] = self.region_proj(region_emb)  # [B, max_regions, common_embed_dim].
        embeddings['region_boxes'] = region_boxes  # [B, max_regions, 4].
        
        # Patch embeddings.
        patch_emb = self.patch_extractor(images)  # [B, max_patches, patch_dim].
        embeddings['patch'] = self.patch_proj(patch_emb)  # [B, max_patches, common_embed_dim].
        
        # Depth embeddings.
        depth_emb = self.depth_extractor(images)  # [B, depth_dim, H, W] or [B, H*W, depth_dim].
        # Pool or flatten to fixed size.
        if depth_emb.dim() == 4:
            # [B, C, H, W] -> pool to [B, C].
            depth_emb = F.adaptive_avg_pool2d(depth_emb, 1).flatten(1)  # [B, depth_dim].
        elif depth_emb.dim() == 3:
            # [B, H*W, C] -> mean pool.
            depth_emb = depth_emb.mean(dim=1)  # [B, depth_dim].
        embeddings['depth'] = self.depth_proj(depth_emb)  # [B, common_embed_dim].
        
        # OCR embeddings.
        if text_snippets is not None:
            ocr_emb, _ = self.ocr_encoder(text_snippets)  # Variable shape.
            # Pad/truncate to max_texts.
            if ocr_emb.dim() == 2:
                # [B, T, ocr_dim] where T may vary.
                B_ocr, T, D = ocr_emb.shape
                if T > self.max_texts:
                    ocr_emb = ocr_emb[:, :self.max_texts]  # Truncate.
                elif T < self.max_texts:
                    # Pad with zeros.
                    padding = torch.zeros(B_ocr, self.max_texts - T, D, device=device)
                    ocr_emb = torch.cat([ocr_emb, padding], dim=1)
            embeddings['ocr'] = self.ocr_proj(ocr_emb)  # [B, max_texts, common_embed_dim].
        else:
            embeddings['ocr'] = torch.zeros(B, self.max_texts, self.common_embed_dim, device=device)
        
        # Audio embeddings (with spatial info)
        if audio is not None:
            audio_emb, spatial = self.audio_encoder(audio)  # Audio_emb: [B, audio_dim] or [B, A, audio_dim].
            if audio_emb.dim() == 2:
                embeddings['audio'] = self.audio_proj(audio_emb)  # [B, common_embed_dim].
            else:
                embeddings['audio'] = self.audio_proj(audio_emb)  # [B, A, common_embed_dim].
            embeddings['audio_spatial'] = spatial  # Preserve spatial info.
        else:
            embeddings['audio'] = torch.zeros(B, self.common_embed_dim, device=device)
            embeddings['audio_spatial'] = None
        
        if scene_graph is not None:
            required_keys = {'node_features', 'edge_index'}
            if not required_keys.issubset(scene_graph.keys()):
                raise ValueError(f"scene_graph must contain {required_keys}")
            
            sg_emb = self.scene_graph_encoder(
                scene_graph['node_features'],
                scene_graph['edge_index'],
                scene_graph.get('edge_attr')
            )  # [scene_graph_dim] or [B, scene_graph_dim].
            
            # Ensure batch dimension.
            if sg_emb.dim() == 1:
                sg_emb = sg_emb.unsqueeze(0).expand(B, -1)  # [B, scene_graph_dim].
            
            embeddings['scene_graph'] = self.sg_proj(sg_emb)  # [B, common_embed_dim].
        else:
            embeddings['scene_graph'] = torch.zeros(B, self.common_embed_dim, device=device)
        
        return embeddings







