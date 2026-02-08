"""Transformer-Based OCR Head for MaxSight 3.0."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class TransformerOCRHead(nn.Module):
    """Transformer-based OCR head."""

    def __init__(
        self,
        input_dim: int = 256,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        vocab_size: int = 10000,
        max_text_length: int = 50
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.max_text_length = max_text_length

        # Input projection.
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # Positional encoding for regions.
        self.pos_encoding = nn.Parameter(torch.randn(1, 500, embed_dim) * 0.02)  # Max 500 regions.

        # Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Transformer decoder.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Character embedding.
        self.char_embedding = nn.Embedding(vocab_size, embed_dim)
        self.sos_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # Learned SOS.

        # Output projection.
        self.output_proj = nn.Linear(embed_dim, vocab_size)

        # Text region detection head.
        self.text_detection_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # Bounding box coordinates.
        )

    def forward(
        self,
        features: torch.Tensor,               # [B, N_regions, input_dim].
        context_embeddings: Optional[torch.Tensor] = None,  # [B, N_objects, embed_dim].
        target_text: Optional[torch.Tensor] = None,  # [B, N_regions, seq_len].
        text_likelihood: Optional[torch.Tensor] = None,  # [B, N_regions] - FIXED: Gating signal.
        motion_stability: Optional[torch.Tensor] = None,  # [B, 1] - FIXED: Motion stability gate.
        cognitive_budget: Optional[float] = None  # FIXED: Cognitive budget gate.
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through OCR head."""
        B, N_regions, _ = features.shape

        # FIXED: OCR Gating - don't run unconditionally.
        gated_mask = torch.ones(B, N_regions, device=features.device, dtype=torch.bool)
        
        # Gate 1: Text-likelihood from backbone.
        if text_likelihood is not None:
            text_threshold = 0.3  # Minimum text likelihood to process.
            gated_mask = gated_mask & (text_likelihood > text_threshold)
        
        # Gate 2: Motion stability (don't OCR while moving fast)
        if motion_stability is not None:
            motion_threshold = 0.5  # Minimum motion stability.
            motion_gate = (motion_stability > motion_threshold).float()
            gated_mask = gated_mask & (motion_gate.unsqueeze(1).expand(-1, N_regions) > 0.5)
        
        # Gate 3: Cognitive budget (don't OCR if budget exhausted)
        if cognitive_budget is not None:
            budget_threshold = 0.2  # Minimum cognitive budget remaining.
            if cognitive_budget < budget_threshold:
                gated_mask = torch.zeros_like(gated_mask)
        
        # If no regions pass gates, return empty outputs.
        if not gated_mask.any():
            return {
                'text_logits': torch.zeros(B, N_regions, self.max_text_length, self.vocab_size, device=features.device),
                'text_boxes': torch.zeros(B, N_regions, 4, device=features.device),
                'text_scores': torch.zeros(B, N_regions, device=features.device),
                'encoded_features': torch.zeros(B, N_regions, self.embed_dim, device=features.device),
                'gated': gated_mask
            }
        
        # Apply gating mask to features (zero out gated regions)
        features_gated = features * gated_mask.unsqueeze(-1).float()

        # Project input features (use gated features)
        x = self.input_proj(features_gated)  # [B, N_regions, embed_dim].

        # Add positional encoding.
        x = x + self.pos_encoding[:, :N_regions, :]

        # Integrate optional context embeddings.
        if context_embeddings is not None:
            context_mean = context_embeddings.mean(dim=1, keepdim=True)  # [B,1,embed_dim].
            x = x + context_mean

        # Transformer encoder.
        encoded = self.encoder(x)  # [B, N_regions, embed_dim].

        # Text detection.
        text_boxes = self.text_detection_head(encoded)  # [B, N_regions, 4].

        # Text recognition per region.
        all_logits = []
        for r in range(N_regions):
            region_encoded = encoded[:, r:r+1, :]  # [B,1,embed_dim].

            if target_text is not None:
                # Training: teacher forcing.
                tgt_seq = target_text[:, r, :]  # [B, seq_len].
                tgt_emb = self.char_embedding(tgt_seq)  # [B, seq_len, embed_dim].
                decoded = self.decoder(tgt_emb, region_encoded)
            else:
                # Inference: autoregressive.
                decoded_tokens = []
                sos = self.sos_token.expand(B, -1, -1)  # [B,1,embed_dim].
                for _ in range(self.max_text_length):
                    tgt_input = torch.cat(decoded_tokens, dim=1) if decoded_tokens else sos
                    out = self.decoder(tgt_input, region_encoded)
                    decoded_tokens.append(out[:, -1:, :])  # Append last step.
                decoded = torch.cat(decoded_tokens, dim=1)  # [B, max_length, embed_dim].

            logits = self.output_proj(decoded)  # [B, seq_len/max_length, vocab_size].
            all_logits.append(logits.unsqueeze(1))  # Add region dim.

        text_logits = torch.cat(all_logits, dim=1)  # [B, N_regions, max_length, vocab_size].

        # Text scores: mean confidence per region.
        text_probs = F.softmax(text_logits, dim=-1)
        text_scores = text_probs.max(dim=-1)[0].mean(dim=-1)  # [B, N_regions].

        return {
            'text_logits': text_logits,
            'text_boxes': text_boxes,
            'text_scores': text_scores,
            'encoded_features': encoded,
            'gated': gated_mask  # FIXED: Return gating information.
        }






