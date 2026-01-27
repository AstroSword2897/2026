"""
Eye/Face Micro-Model

Tiny CNN for eye tracking and fatigue detection:
- Blink probability
- Fixation vs saccade patterns
- Pupil-size proxy

PROJECT PHILOSOPHY & APPROACH:
=============================
This module implements eye/face tracking for fatigue detection and gaze analysis. It's a critical
component for understanding user state and adapting assistance levels accordingly.

WHY EYE TRACKING MATTERS:
Eye tracking provides valuable information about user state:
- Blink rate: Indicates fatigue or attention levels
- Fixation vs saccade: Shows whether user is focused or scanning
- Pupil size: Proxy for cognitive load or lighting conditions

This information enables adaptive assistance - reducing detail when user is fatigued, increasing
support when attention is low, etc.

HOW IT CONNECTS TO THE PROBLEM STATEMENT:
The problem emphasizes "Routine Workflow" and "Skill Development" - understanding user state
enables the system to adapt assistance appropriately, supporting both immediate needs and
long-term skill development.

TECHNICAL DESIGN DECISIONS:
- Small input size (64x64): Fast inference for real-time use
- Normalized inputs [0,1]: Ensures conv layers receive meaningful activations
- Dropout in FC heads: Prevents degenerate outputs with small batches
- Proper initialization: Ensures meaningful outputs from the start

Phase 1: Core ML Backbone & Preprocessing
See docs/therapy_system_implementation_plan.md for implementation details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional
import torchvision.transforms as transforms
from PIL import Image


class EyeImagePreprocessor:
    """
    Preprocessor for eye/face images before feeding to EyeModel.
    
    WHY THIS CLASS EXISTS:
    Eye model requires specific preprocessing:
    1. Normalize to [0,1] range - ensures conv layers receive meaningful activations
    2. Resize/crop to [64, 64] - model expects fixed input size
    3. Optional contrast adjustment - improves detection in varying lighting
    
    Without proper preprocessing, conv layers may output meaningless activations, leading to
    wrong blink/fixation/pupil predictions.
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (64, 64),
        normalize: bool = True,
        contrast_adjust: bool = True
    ):
        """
        Initialize eye image preprocessor.
        
        Arguments:
            target_size: Target image size (height, width) - must be (64, 64) for EyeModel
            normalize: Normalize to [0,1] range (required for meaningful conv activations)
            contrast_adjust: Apply contrast adjustment for better detection
        """
        self.target_size = target_size
        self.normalize = normalize
        self.contrast_adjust = contrast_adjust
        
        # Build transform pipeline
        transform_list = [
            transforms.Resize(target_size),
            transforms.ToTensor()  # Converts to [0,1] range automatically
        ]
        
        if contrast_adjust:
            # Add contrast adjustment (CLAHE-like effect)
            transform_list.insert(-1, transforms.Lambda(self._adjust_contrast))
        
        self.transform = transforms.Compose(transform_list)
    
    def _adjust_contrast(self, image: Image.Image) -> Image.Image:
        """
        Adjust contrast for better eye detection.
        
        WHY CONTRAST ADJUSTMENT:
        Eye images often have low contrast, especially in varying lighting conditions.
        Contrast adjustment ensures conv layers can detect meaningful features (pupil, iris,
        eyelid edges) for accurate blink/fixation/pupil predictions.
        """
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.2)  # 20% contrast boost
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess eye/face image.
        
        Arguments:
            image: PIL Image of face/eye region
        
        Returns:
            Preprocessed tensor [3, 64, 64] in [0,1] range
        """
        result = self.transform(image)
        # Ensure result is a tensor (transform should return tensor due to ToTensor())
        if not isinstance(result, torch.Tensor):
            # Fallback: convert manually if transform didn't work
            import torchvision.transforms.functional as TF
            result = TF.to_tensor(image)
            result = TF.resize(result.unsqueeze(0), list(self.target_size)).squeeze(0)  # Convert tuple to list
        return result
    
    def preprocess_tensor(
        self,
        tensor: torch.Tensor,
        ensure_normalized: bool = True
    ) -> torch.Tensor:
        """
        Preprocess tensor directly (for batch processing).
        
        WHY THIS FUNCTION:
        Sometimes we have tensors already (from video frames). This function ensures they're
        properly normalized and resized before feeding to the model.
        
        Arguments:
            tensor: Input tensor [B, C, H, W] or [C, H, W]
            ensure_normalized: Ensure values are in [0,1] range
        
        Returns:
            Preprocessed tensor [B, 3, 64, 64] or [3, 64, 64] in [0,1] range
        """
        # Ensure batch dimension
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Normalize to [0,1] if needed
        if ensure_normalized:
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            if tensor.min() < 0.0:
                tensor = (tensor + 1.0) / 2.0
        
        # Resize to target size
        if tensor.shape[2:] != self.target_size:
            tensor = F.interpolate(
                tensor,
                size=list(self.target_size),  # Convert tuple to list for type checker
                mode='bilinear',
                align_corners=False
            )
        
        # Ensure 3 channels
        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
        elif tensor.shape[1] != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {tensor.shape[1]}")
        
        if squeeze_output:
            tensor = tensor.squeeze(0)
        
        return tensor


class EyeModel(nn.Module):
    """
    Tiny CNN for eye/face tracking and fatigue detection.
    
    WHY THIS CLASS EXISTS:
    This model provides eye tracking capabilities for understanding user state (fatigue, attention,
    cognitive load). This information enables adaptive assistance - the system can adjust
    verbosity, frequency, and detail levels based on user state.
    
    Architecture:
    Conv -> ReLU -> Conv -> ReLU -> FC -> outputs:
    - Blink probability (0–1)
    - Fixation vs saccade pattern (softmax over 2 classes)
    - Pupil-size proxy (0–1)
    
    Input: [B, 3, 64, 64] - Face/eye region (64x64 for speed)
    Output: Dict with blink_prob, fixation_pattern, pupil_size
    
    CRITICAL REQUIREMENTS:
    1. Input must be normalized to [0,1] range - otherwise conv layers output meaningless activations
    2. Input must be exactly [B, 3, 64, 64] - model assumes this shape
    3. Training labels for fixation must match softmax format (binary 0/1 is fine)
    4. Dropout prevents degenerate FC outputs with small batches
    """
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (64, 64),
        dropout: float = 0.15
    ):
        """
        Initialize eye model.
        
        WHY THESE PARAMETERS:
        - input_size: Must be (64, 64) - model architecture assumes this
        - dropout: Prevents degenerate FC outputs with small batches or poor initialization
        
        Arguments:
            input_size: Input image size (height, width) - must be (64, 64)
            dropout: Dropout probability for FC heads (0.15 = 15% dropout)
        """
        super().__init__()
        if input_size != (64, 64):
            raise ValueError(f"EyeModel requires input_size=(64, 64), got {input_size}")
        
        self.input_size = input_size
        self.dropout = dropout
        
        # Tiny CNN architecture
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Output heads with dropout to prevent degenerate outputs
        # WHY DROPOUT: With only 32 features into FC heads, small batches or poor initialization
        #              can lead to NaN or constant outputs. Dropout prevents this.
        self.blink_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Blink probability [0, 1]
        )
        
        self.fixation_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 2),  # [fixation_prob, saccade_prob]
            nn.Softmax(dim=1)  # Softmax ensures probabilities sum to 1
            # NOTE: Training labels should be binary (0/1) for fixation vs saccade
            # If labels are continuous probabilities, use Sigmoid instead
        )
        
        self.pupil_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Pupil size proxy [0, 1]
        )
        
        # Initialize weights properly to avoid degenerate outputs
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize weights to prevent degenerate outputs.
        
        WHY PROPER INITIALIZATION:
        With only 32 features into FC heads, poor initialization can lead to NaN or constant
        outputs. Proper initialization ensures heads produce meaningful outputs from the start.
        """
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
        """
        Forward pass through eye model.
        
        CRITICAL INPUT REQUIREMENTS:
        1. Input must be normalized to [0,1] range
        2. Input shape must be [B, 3, 64, 64]
        3. Use EyeImagePreprocessor to ensure proper preprocessing
        
        Arguments:
            face_region: Face/eye region [B, 3, 64, 64] in [0,1] range
        
        Returns:
            Dictionary with:
                - 'blink_prob': [B, 1] - Blink probability [0, 1]
                - 'fixation': [B, 2] - [fixation_prob, saccade_prob] (softmax probabilities)
                - 'pupil_size': [B, 1] - Pupil size proxy [0, 1]
        """
        # Validate input shape
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
        
        # Validate input range (should be [0,1])
        if face_region.min() < 0.0 or face_region.max() > 1.0:
            # Warn but don't fail - might be intentional
            # However, this can lead to meaningless conv activations
            if self.training:
                # In training, normalize on-the-fly
                face_region = torch.clamp(face_region, 0.0, 1.0)
            else:
                # In inference, warn user
                import warnings
                warnings.warn(
                    f"Input values outside [0,1] range: min={face_region.min():.3f}, "
                    f"max={face_region.max():.3f}. This may cause meaningless conv activations. "
                    f"Use EyeImagePreprocessor to normalize inputs."
                )
        
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(face_region)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).flatten(1)  # [B, 32]
        
        # Check for NaN/Inf (can happen with degenerate inputs)
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise RuntimeError(
                "NaN/Inf detected in features. Check input preprocessing - "
                "ensure inputs are normalized to [0,1] and properly resized."
            )
        
        # Output heads
        blink_prob = self.blink_head(x)
        fixation = self.fixation_head(x)
        pupil_size = self.pupil_head(x)
        
        # Validate outputs
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
            'fixation': fixation,  # [B, 2] - softmax probabilities
            'pupil_size': pupil_size
        }
    
    def get_fixation_label_format(self) -> str:
        """
        Get expected label format for fixation head.
        
        WHY THIS FUNCTION:
        Clarifies that fixation head uses softmax, so labels should be binary (0/1) for
        fixation vs saccade. If continuous probabilities are needed, use Sigmoid instead.
        
        Returns:
            Description of expected label format
        """
        return (
            "Fixation head uses Softmax, so labels should be binary (0/1): "
            "[0, 1] for fixation, [1, 0] for saccade. "
            "If continuous probabilities are needed, modify head to use Sigmoid instead."
        )

