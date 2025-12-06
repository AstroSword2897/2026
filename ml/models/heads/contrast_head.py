import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

class ContrastHead(nn.Module):
    def __init__(self, in_channels: int = 256, use_edge_aware: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.use_edge_aware = use_edge_aware
        
        # Contrast estimation network
        # WHY THIS ARCHITECTURE:
        # - 3 conv layers: Sufficient depth to learn contrast patterns without overfitting
        # - Progressive channel reduction: 256 -> 128 -> 64 -> 1 (efficient computation)
        # - 3x3 kernels: Capture local contrast relationships
        # - 1x1 final layer: Efficiently maps to single contrast value per pixel
        
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=1)  # Single channel contrast map
        self.relu = nn.ReLU(inplace=True)
        
        # Edge detection for edge-aware contrast
        if use_edge_aware:
            # Sobel-like edge detection kernels
            sobel_x_tensor = torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                dtype=torch.float32
            ).view(1, 1, 3, 3)
            
            sobel_y_tensor = torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                dtype=torch.float32
            ).view(1, 1, 3, 3)
            
            self.register_buffer('sobel_x', sobel_x_tensor)
            self.register_buffer('sobel_y', sobel_y_tensor)
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights to prevent degenerate outputs.
        
        WHY PROPER INITIALIZATION:
        Poor initialization can lead to constant or NaN outputs, especially with BatchNorm.
        Proper initialization ensures the head produces meaningful contrast maps from the start.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def compute_edge_map(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute edge map for edge-aware contrast computation.
        
        WHY EDGE-AWARE CONTRAST:
        Contrast at object boundaries (edges) is more perceptually relevant than contrast in
        uniform regions. Edge-aware contrast maps better reflect what users actually perceive
        and are more useful for therapy exercises and navigation assistance.
        
        Arguments:
            features: Input features [B, C, H, W]
        
        Returns:
            Edge map [B, 1, H, W] with edge strength
        """
        if not self.use_edge_aware:
            return torch.zeros_like(features[:, :1])
        
        # Average across channels for edge detection
        gray = features.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Apply Sobel filters
        # Registered buffers are tensors, but type checker needs explicit cast
        if not hasattr(self, 'sobel_x') or not hasattr(self, 'sobel_y'):
            return torch.zeros_like(features[:, :1])
        sobel_x: torch.Tensor = self.sobel_x.to(dtype=gray.dtype)  # type: ignore
        sobel_y: torch.Tensor = self.sobel_y.to(dtype=gray.dtype)  # type: ignore
        edge_x = F.conv2d(gray, sobel_x, padding=1)
        edge_y = F.conv2d(gray, sobel_y, padding=1)
        
        # Compute edge magnitude
        edge_mag = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
        
        # Normalize to [0, 1]
        B = edge_mag.shape[0]
        edge_flat = edge_mag.view(B, -1)
        edge_max = edge_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        edge_mag = edge_mag / (edge_max + 1e-8)
        
        return edge_mag
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:

        # Validate input
        if features.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], got {features.shape}")
        
        B, C, H, W = features.shape
        if C != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {C}. "
                f"Ensure input features match head configuration."
            )
        
        # Feature extraction
        x = self.relu(self.bn1(self.conv1(features)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # Generate contrast map
        contrast_map = torch.sigmoid(self.conv3(x))
        contrast_map = contrast_map.squeeze(1)
        
        # Validate output
        if torch.isnan(contrast_map).any() or torch.isinf(contrast_map).any():
            raise RuntimeError(
                "NaN/Inf detected in contrast map. Check input features and model initialization."
            )
        
        return contrast_map
    
    def compute_loss(
        self,
        pred_contrast: torch.Tensor,
        target_contrast: torch.Tensor,
        use_edge_aware: Optional[bool] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute contrast loss with optional edge-aware weighting.
        
        WHY EDGE-AWARE LOSS:
        Standard L1 loss treats all pixels equally, but contrast at edges is more perceptually
        important. Edge-aware loss emphasizes contrast errors at object boundaries, leading to
        better perceptual quality and more useful therapy feedback.
        
        Arguments:
            pred_contrast: Predicted contrast map [B, H, W] or [B, C, H, W]
            target_contrast: Ground truth contrast map [B, H, W] or [B, C, H, W]
            use_edge_aware: Override instance setting for edge-aware loss
        
        Returns:
            Dictionary with:
                - 'l1_loss': Standard L1 contrast loss
                - 'edge_aware_loss': Edge-weighted L1 loss (if enabled)
                - 'total_loss': Combined loss for training
        """
        use_edge = use_edge_aware if use_edge_aware is not None else self.use_edge_aware
        
        # Validate inputs
        if pred_contrast.shape != target_contrast.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred_contrast.shape} vs target {target_contrast.shape}"
            )
        
        # Standard L1 loss
        l1_loss = F.l1_loss(pred_contrast, target_contrast)
        
        losses = {'l1_loss': l1_loss}
        
        # Edge-aware loss (weighted by edge strength)
        if use_edge and self.use_edge_aware:
            if target_contrast.dim() == 3:
                target_4d = target_contrast.unsqueeze(1)
            else:
                target_4d = target_contrast
    
            if pred_contrast.dim() == 3:
                pred_4d = pred_contrast.unsqueeze(1)
            else:
                pred_4d = pred_contrast

            # Compute edge map from target using filter methods 
            edge_map = self._compute_edge_map(target_4d)
            
            # Weight the pixel-wise loss by edge strength
            pixel_wise_loss = torch.abs(pred_4d - target_4d)
            edge_weighted_loss = pixel_wise_loss * (1.0 + edge_map)
            edge_aware_loss = edge_weighted_loss.mean()
            
            losses['edge_aware_loss'] = edge_aware_loss
            losses['total_loss'] = 0.5 * l1_loss + 0.5 * edge_aware_loss
        else:
            losses['total_loss'] = l1_loss
        
        return losses
    
    def _compute_edge_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute edge map using Sobel filters.
        
        WHY THIS METHOD:
        Edge maps are computed directly from the input tensor using Sobel filters, which detect
        gradients (edges) in the image. This is more flexible than using pre-computed edge maps
        and works with any input tensor shape.
        
        Arguments:
            x: Input tensor of shape [B, C, H, W] or [B, H, W]
        
        Returns:
            Edge magnitude map of shape [B, 1, H, W]
        """
        # Ensure 4D tensor
        original_dim = x.dim()
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        # Safety check BEFORE processing
        if not self.use_edge_aware or not hasattr(self, 'sobel_x'):
            if x.dim() == 4:
                return torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
            else:
                return torch.zeros_like(x)

        # Convert to grayscale if multichannel
        if x.shape[1] > 1:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        else:
            gray = x

        # Now safe to use Sobel filters
        sobel_x_tensor: torch.Tensor = self.sobel_x.to(dtype=x.dtype)  # type: ignore
        sobel_y_tensor: torch.Tensor = self.sobel_y.to(dtype=x.dtype)  # type: ignore
        grad_x = F.conv2d(gray, sobel_x_tensor, padding=1)
        grad_y = F.conv2d(gray, sobel_y_tensor, padding=1)
        # Compute gradient magnitude (vectorized)
        edge_map = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        if edge_map.dim() == 4:
            B, C, H, W = edge_map.shape
            edge_flat = edge_map.view(B, -1)  # [B, H*W]
            edge_min = edge_flat.min(dim=1, keepdim=True)[0]  # [B, 1]
            edge_max = edge_flat.max(dim=1, keepdim=True)[0]  # [B, 1]
            
            # Reshape for broadcasting: [B, 1] -> [B, 1, 1, 1] to match [B, 1, H, W]
            edge_min = edge_min.view(B, 1, 1, 1)
            edge_max = edge_max.view(B, 1, 1, 1)
            
            # Avoid division by zero with efficient masking
            range_mask = (edge_max > edge_min).float()
            edge_map = range_mask * (edge_map - edge_min) / (edge_max - edge_min + 1e-8) + (1 - range_mask) * torch.zeros_like(edge_map)
        else:
                # Fallback for unexpected shapes (shouldn't happen, but safe)
            edge_max_val = edge_map.max()
            if edge_max_val > 0:
                edge_map = edge_map / (edge_max_val + 1e-8)
        
        # Restore original dimensionality if needed
        if original_dim == 2:
            edge_map = edge_map.squeeze(0).squeeze(0)  # [1, 1, H, W] -> [H, W]
        elif original_dim == 3:
            edge_map = edge_map.squeeze(1)  # [B, 1, H, W] -> [B, H, W]
            
        return edge_map

