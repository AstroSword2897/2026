"""Multi-Vector Retrieval Heads for MaxSight 3.0."""

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
    """Multi-vector retrieval heads combining all embedding types."""
    
    def __init__(
        self,
        global_dim: int = 512,
        region_dim: int = 256,
        patch_dim: int = 768,
        depth_dim: int = 256,
        ocr_dim: int = 384,
        audio_dim: int = 256,
        scene_graph_dim: int = 512
    ):
        super().__init__()
        
        self.global_encoder = GlobalEncoder(embed_dim=global_dim)
        self.region_extractor = RegionExtractor(max_regions=8, region_size=(224, 224))
        self.patch_extractor = PatchExtractor(embed_dim=patch_dim, num_clusters=25)
        self.depth_extractor = DepthExtractor(embed_dim=depth_dim)
        self.ocr_encoder = OCREncoder(embed_dim=ocr_dim, max_texts=10)
        self.audio_encoder = AudioEncoder(embed_dim=audio_dim)
        self.scene_graph_encoder = SceneGraphRetrievalEncoder(embed_dim=scene_graph_dim)
    
    def forward(
        self,
        images: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        text_snippets: Optional[List[List[str]]] = None,
        scene_graph: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """Extract all embedding types."""
        embeddings = {}
        
        # Global embedding.
        embeddings['global'] = self.global_encoder(images)
        
        # Region embeddings.
        region_emb, region_boxes = self.region_extractor.extract_regions(images)
        embeddings['region'] = region_emb
        
        # Patch embeddings.
        embeddings['patch'] = self.patch_extractor(images)
        
        # Depth embeddings.
        embeddings['depth'] = self.depth_extractor(images)
        
        # OCR embeddings.
        if text_snippets is not None:
            ocr_emb, _ = self.ocr_encoder(text_snippets)
            embeddings['ocr'] = ocr_emb
        
        # Audio embeddings.
        if audio is not None:
            audio_emb, spatial = self.audio_encoder(audio)
            embeddings['audio'] = audio_emb
        
        # Scene graph embeddings.
        if scene_graph is not None:
            sg_emb = self.scene_graph_encoder(
                scene_graph['node_features'],
                scene_graph['edge_index'],
                scene_graph.get('edge_attr')
            )
            embeddings['scene_graph'] = sg_emb
        
        return embeddings







