"""Eye/Face Micro-Model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional
import torchvision.transforms as transforms
from PIL import Image


class EyeImagePreprocessor:
    """Preprocessor for eye/face images before feeding to EyeModel."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (64, 64),
        normalize: bool = True,
        contrast_adjust: bool = True
    ):
        """Initialize eye image preprocessor."""
        self.target_size = target_size
        self.normalize = normalize
        self.contrast_adjust = contrast_adjust
        
        # Build transform pipeline.
        transform_list = [
            transforms.Resize(target_size),
            transforms.ToTensor()  # Converts to [0,1] range automatically.
        ]
        
        if contrast_adjust:
            # Add contrast adjustment (CLAHE-like effect)
            transform_list.insert(-1, transforms.Lambda(self._adjust_contrast))
        
        self.transform = transforms.Compose(transform_list)
    
    def _adjust_contrast(self, image: Image.Image) -> Image.Image:
        """Adjust contrast for better eye detection."""
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.2)  # 20% contrast boost.
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Preprocess eye/face image. Arguments: image: PIL Image of face/eye region Returns: Preprocessed tensor [3, 64, 64] in [0,1] range."""
        result = self.transform(image)
        # Ensure result is a tensor (ToTensor() returns tensor)
        if not isinstance(result, torch.Tensor):
            # Fallback: convert manually if transform didn't work.
            import torchvision.transforms.functional as TF
            result = TF.to_tensor(image)
            result = TF.resize(result.unsqueeze(0), list(self.target_size)).squeeze(0)  # Convert tuple to list.
        return result
    
    def preprocess_tensor(
        self,
        tensor: torch.Tensor,
        ensure_normalized: bool = True
    ) -> torch.Tensor:
        """Preprocess tensor directly (for batch processing)."""
        # Ensure batch dimension.
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Normalize to [0,1] if needed.
        if ensure_normalized:
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            if tensor.min() < 0.0:
                tensor = (tensor + 1.0) / 2.0
        
        # Resize to target size.
        if tensor.shape[2:] != self.target_size:
            tensor = F.interpolate(
                tensor,
                size=list(self.target_size),  # Convert tuple to list for type checker.
                mode='bilinear',
                align_corners=False
            )
        
        # Ensure 3 channels.
        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
        elif tensor.shape[1] != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {tensor.shape[1]}")
        
        if squeeze_output:
            tensor = tensor.squeeze(0)
        
        return tensor


class EyeModel(nn.Module):
    """Tiny CNN for eye/face tracking and fatigue detection."""
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (64, 64),
        dropout: float = 0.15
    ):
        """Initialize eye model."""
        super().__init__()
        if input_size != (64, 64):
            raise ValueError(f"EyeModel requires input_size=(64, 64), got {input_size}")
        
        self.input_size = input_size
        self.dropout = dropout
        
        # Tiny CNN architecture.
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Output heads with dropout to prevent degenerate outputs. Can lead to NaN or constant outputs. Dropout prevents this.
        self.blink_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Blink probability [0, 1].
        )
        
        self.fixation_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 2),  # [fixation_prob, saccade_prob].
            nn.Softmax(dim=1)  # Softmax ensures probabilities sum to 1.
            # Use binary labels (0/1) for fixation vs saccade. If labels are continuous probabilities, use Sigmoid instead.
        )
        
        self.pupil_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Pupil size proxy [0, 1].
        )
        
        # Initialize weights properly to avoid degenerate outputs.
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to prevent degenerate outputs."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, face_region: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through eye model."""
        # Validate input shape.
        if face_region.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B, 3, 64, 64], got {face_region.shape}")
        
        B, C, H, W = face_region.shape
        if C != 3:
            raise ValueError(f"Expected 3 channels, got {C}")
        if (H, W) != self.input_size:
            raise ValueError(
                f"Expected input size {self.input_size}, got {(H, W)}. "
                f"Use EyeImagePreprocessor to resize/crop correctly."
            )
        
        # Validate input range [0,1].
        if face_region.min() < 0.0 or face_region.max() > 1.0:
            # Warn but don't fail - might be intentional. Avoid meaningless conv activations when input is invalid.
            if self.training:
                # In training, normalize on-the-fly.
                face_region = torch.clamp(face_region, 0.0, 1.0)
            else:
                # In inference, warn user.
                import warnings
                warnings.warn(
                    f"Input values outside [0,1] range: min={face_region.min():.3f}, "
                    f"max={face_region.max():.3f}. This may cause meaningless conv activations. "
                    f"Use EyeImagePreprocessor to normalize inputs."
                )
        
        # Feature extraction.
        x = F.relu(self.bn1(self.conv1(face_region)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).flatten(1)  # [B, 32].
        
        # Check for NaN/Inf (can happen with degenerate inputs)
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise RuntimeError(
                "NaN/Inf detected in features. Check input preprocessing - "
                "ensure inputs are normalized to [0,1] and properly resized."
            )
        
        # Output heads.
        blink_prob = self.blink_head(x)
        fixation = self.fixation_head(x)
        pupil_size = self.pupil_head(x)
        
        # Validate outputs.
        if torch.isnan(blink_prob).any() or torch.isnan(fixation).any() or torch.isnan(pupil_size).any():
            raise RuntimeError(
                "NaN detected in outputs. This may be due to: "
                "1. Unnormalized inputs (use EyeImagePreprocessor), "
                "2. Very small batch size, "
                "3. Poor weight initialization. "
                "Check input preprocessing and model initialization."
            )
        
        return {
            'blink_prob': blink_prob,
            'fixation': fixation,  # [B, 2] - softmax probabilities.
            'pupil_size': pupil_size
        }
    
    def get_fixation_label_format(self) -> str:
        """Get expected label format for fixation head."""
        return (
            "Fixation head uses Softmax, so labels should be binary (0/1): "
            "[0, 1] for fixation, [1, 0] for saccade. "
            "If continuous probabilities are needed, modify head to use Sigmoid instead."
        )







