"""Temporal Processing Modules for MaxSight 3.0 Includes ConvLSTM for motion tracking and TimeSformer for long-range temporal dependencies."""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell. Processes spatial-temporal information using convolutional operations."""
    
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Convolutional gates: combines input and hidden state.
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,  # I, f, g, o gates.
            kernel_size,
            padding=kernel_size // 2,
            bias=True
        )
    
    def forward(
        self,
        x: torch.Tensor,  # [B, C, H, W].
        hidden: Tuple[torch.Tensor, torch.Tensor]  # (h, c)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through ConvLSTM cell."""
        h_prev, c_prev = hidden
        
        # Concatenate input and hidden state.
        combined = torch.cat([x, h_prev], dim=1)  # [B, C+hidden_dim, H, W].
        
        # Convolutional gates.
        gates = self.conv(combined)  # [B, 4*hidden_dim, H, W].
        
        # Split into gates.
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        
        # Apply activations.
        i = torch.sigmoid(i)  # Input gate.
        f = torch.sigmoid(f)  # Forget gate.
        g = torch.tanh(g)     # Candidate values.
        o = torch.sigmoid(o)  # Output gate.
        
        # Update cell state.
        c_new = f * c_prev + i * g
        
        # Update hidden state.
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new


class ConvLSTM(nn.Module):
    """Multi-layer ConvLSTM for motion tracking. Tracks motion across multiple frames for people, vehicles, and obstacles."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        num_layers: int = 2
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # ConvLSTM cells.
        self.cells = nn.ModuleList([
            ConvLSTMCell(
                input_dim if i == 0 else hidden_dim,
                hidden_dim,
                kernel_size
            )
            for i in range(num_layers)
        ])
    
    def forward(
        self,
        x: torch.Tensor,  # [B, T, C, H, W] - sequence of frames.
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through ConvLSTM."""
        B, T, C, H, W = x.shape
        
        # Initialize hidden state if not provided.
        if hidden_state is None:
            h = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        else:
            h, c = hidden_state
        
        outputs = []
        for t in range(T):
            # Process through each layer - CRITICAL: feed each layer's output to the next.
            cur_input = x[:, t]  # [B, C, H, W].
            for layer_idx, cell in enumerate(self.cells):
                h, c = cell(cur_input, (h, c))
                cur_input = h  # Feed this layer's output to the next layer.
            outputs.append(h)
        
        # Stack outputs: [B, T, hidden_dim, H, W].
        output = torch.stack(outputs, dim=1)
        
        return output, (h, c)


class DividedSpaceTimeAttention(nn.Module):
    """Divided space-time attention for TimeSformer."""
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        x_spatial = x.reshape(B * T, N, D)
        x_spatial, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)
        x_spatial = self.norm1(x.reshape(B * T, N, D) + x_spatial)
        x = x_spatial.reshape(B, T, N, D)
        x_temporal = x.permute(0, 2, 1, 3).contiguous().reshape(B * N, T, D)
        x_temporal, _ = self.temporal_attn(x_temporal, x_temporal, x_temporal)
        x_temporal = self.norm2(x.permute(0, 2, 1, 3).contiguous().reshape(B * N, T, D) + x_temporal)
        x = x_temporal.reshape(B, N, T, D).permute(0, 2, 1, 3).contiguous()
        x_ffn = self.ffn(x.reshape(B * T * N, D)).reshape(B, T, N, D)
        x = self.norm3(x + x_ffn)
        return x


class TimeSformer(nn.Module):
    """TimeSformer: Temporal Transformer for video understanding."""
    
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, num_layers: int = 12, num_frames: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.temporal_embed = nn.Parameter(torch.randn(1, num_frames, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([DividedSpaceTimeAttention(embed_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        # Add temporal embedding: [1, T, 1, D] -> [B, T, N, D].
        temporal_embed = self.temporal_embed.unsqueeze(0).unsqueeze(2)  # [1, T, 1, D].
        x = x + temporal_embed  # Broadcast to [B, T, N, D].
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=(1, 2))  # [B, D].
        return x








