"""
Transformer-Based OCR Head for MaxSight 3.0

Transformer encoder for text detection and decoder for text recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple


class TransformerOCRHead(nn.Module):
    """
    Transformer-based OCR head.
    
    Architecture:
    - Transformer encoder: Text detection
    - Transformer decoder: Text recognition
    - Scene-text contextual embedding: Integration with object detection
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        vocab_size: int = 10000,  # Character vocabulary
        max_text_length: int = 50
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.max_text_length = max_text_length
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_text_length, embed_dim) * 0.02
        )
        
        # Transformer encoder for text detection
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Transformer decoder for text recognition
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Character embedding
        self.char_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        
        # Text region detection head
        self.text_detection_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # Bounding box coordinates
        )
    
    def forward(
        self,
        features: torch.Tensor,  # [B, N_regions, input_dim]
        context_embeddings: Optional[torch.Tensor] = None,  # [B, N_objects, embed_dim]
        target_text: Optional[torch.Tensor] = None  # [B, seq_len] for training
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through OCR head.
        
        Args:
            features: Text region features [B, N_regions, input_dim]
            context_embeddings: Optional object detection context [B, N_objects, embed_dim]
            target_text: Optional target text for training [B, seq_len]
        
        Returns:
            Dictionary with:
                - 'text_logits': [B, N_regions, vocab_size, max_length]
                - 'text_boxes': [B, N_regions, 4]
                - 'text_scores': [B, N_regions]
        """
        B, N_regions, _ = features.shape
        
        # Project input features
        x = self.input_proj(features)  # [B, N_regions, embed_dim]
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :N_regions, :]
        
        # Encode text regions
        encoded = self.encoder(x)  # [B, N_regions, embed_dim]
        
        # Text detection: predict bounding boxes
        text_boxes = self.text_detection_head(encoded)  # [B, N_regions, 4]
        
        # Text recognition: decode text
        if target_text is not None:
            # Training: use target text
            tgt = self.char_embedding(target_text)  # [B, seq_len, embed_dim]
            decoded = self.decoder(tgt, encoded)  # [B, seq_len, embed_dim]
        else:
            # Inference: autoregressive decoding
            # Start with SOS token
            sos_token = torch.zeros(B, 1, self.embed_dim, device=features.device)
            decoded = []
            for _ in range(self.max_text_length):
                tgt = torch.cat([sos_token] + decoded, dim=1) if decoded else sos_token
                out = self.decoder(tgt, encoded)
                decoded.append(out[:, -1:, :])
            
            decoded = torch.cat(decoded, dim=1)  # [B, max_length, embed_dim]
        
        # Output logits
        text_logits = self.output_proj(decoded)  # [B, max_length, vocab_size]
        
        # Text scores (confidence)
        text_scores = F.softmax(text_logits, dim=-1).max(dim=-1)[0].mean(dim=1)  # [B, N_regions]
        
        return {
            'text_logits': text_logits,
            'text_boxes': text_boxes,
            'text_scores': text_scores,
            'encoded_features': encoded
        }


