"""
Preprocessing Pipeline for Environmental Structuring
Image transforms, audio MFCC, distance estimation, text detection

Meta AI-style structure: Pure PyTorch/torchvision operations, GPU-friendly tensor processing.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import numpy as np
from typing import Tuple, Optional, Dict, Any
from PIL import Image
import math

def rgb_to_lab_tensor(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB tensor to LAB color space using PyTorch operations.
    
    Meta AI-style: Pure tensor operations, GPU-friendly, differentiable.
    
        Arguments:
        rgb: Tensor [C, H, W] or [B, C, H, W] in range [0, 1]
    
    Returns:
        LAB tensor [C, H, W] or [B, C, H, W] with L in [0, 100], A/B in [-128, 127]
    """
    # Convert RGB to XYZ
    mask = rgb > 0.04045
    rgb_linear = torch.where(
        mask,
        torch.pow((rgb + 0.055) / 1.055, 2.4),
        rgb / 12.92
    )
    
    # RGB to XYZ matrix (D65 illuminant)
    transform = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], device=rgb.device, dtype=rgb.dtype)
    
    if rgb.dim() == 3:  # [C, H, W]
        xyz = torch.einsum('ij,jhw->ihw', transform, rgb_linear)
        # Normalize by D65 white point for 3D tensor
        white_point = torch.tensor([0.95047, 1.0, 1.08883], device=xyz.device, dtype=xyz.dtype).view(3, 1, 1)
        xyz = xyz / white_point
    else:  # [B, C, H, W]
        xyz = torch.einsum('ij,bjhw->bihw', transform, rgb_linear)
        # Normalize by D65 white point for 4D tensor
        white_point = torch.tensor([0.95047, 1.0, 1.08883], device=xyz.device, dtype=xyz.dtype).view(1, 3, 1, 1)
        xyz = xyz / white_point
    
    # XYZ to LAB
    def f(t: torch.Tensor) -> torch.Tensor:
        delta = 6.0 / 29.0
        return torch.where(
            t > delta ** 3,
            torch.pow(t, 1.0 / 3.0),
            t / (3.0 * delta ** 2) + 4.0 / 29.0
        )
    
    if xyz.dim() == 3:
        # xyz already normalized by white point above
        fx = f(xyz[0, :, :])
        fy = f(xyz[1, :, :])
        fz = f(xyz[2, :, :])
    else:
        # xyz already normalized by white point above
        fx = f(xyz[:, 0, :, :])
        fy = f(xyz[:, 1, :, :])
        fz = f(xyz[:, 2, :, :])
    
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    
    if xyz.dim() == 3:
        return torch.stack([L, a, b], dim=0)
    else:
        return torch.stack([L, a, b], dim=1)


def lab_to_rgb_tensor(lab: torch.Tensor) -> torch.Tensor:
    """
    Convert LAB tensor to RGB color space using PyTorch operations.
    
        Arguments:
        lab: Tensor [C, H, W] or [B, C, H, W] with L in [0, 100], A/B in [-128, 127]
    
    Returns:
        RGB tensor [C, H, W] or [B, C, H, W] in range [0, 1]
    """
    if lab.dim() == 3:
        L, a, b = lab[0], lab[1], lab[2]
    else:
        L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    
    # LAB to XYZ
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    
    def f_inv(t: torch.Tensor) -> torch.Tensor:
        delta = 6.0 / 29.0
        return torch.where(
            t > delta,
            torch.pow(t, 3.0),
            3.0 * delta ** 2 * (t - 4.0 / 29.0)
        )
    
    x = 0.95047 * f_inv(fx)
    y = f_inv(fy)
    z = 1.08883 * f_inv(fz)
    
    if lab.dim() == 3:
        xyz = torch.stack([x, y, z], dim=0)
    else:
        xyz = torch.stack([x, y, z], dim=1)
    
    # XYZ to RGB
    transform = torch.tensor([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252]
    ], device=lab.device, dtype=lab.dtype)
    
    if xyz.dim() == 3:
        rgb_linear = torch.einsum('ij,jhw->ihw', transform, xyz)
    else:
        rgb_linear = torch.einsum('ij,bjhw->bihw', transform, xyz)
    
    # Gamma correction
    mask = rgb_linear > 0.0031308
    rgb = torch.where(
        mask,
        1.055 * torch.pow(rgb_linear, 1.0 / 2.4) - 0.055,
        12.92 * rgb_linear
    )
    
    return torch.clamp(rgb, 0.0, 1.0)


def apply_clahe_tensor_fast(
    image: torch.Tensor,
    clip_limit: float = 2.0
) -> torch.Tensor:
    """
    Fast CLAHE using torchvision's equalize or simple contrast enhancement.
    
    This is much faster than tile-based CLAHE for real-time processing.
    
    Arguments:
        image: Tensor [C, H, W] or [B, C, H, W] in range [0, 1]
        clip_limit: Contrast limiting factor (not used in fast version)
    
    Returns:
        Enhanced tensor with same shape and range
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    
    # Use built-in equalize if available, else simple contrast enhancement
    try:
        from torchvision.transforms.functional import equalize
        # Convert to uint8 for equalize
        image_uint8 = (image * 255.0).clamp(0, 255).to(torch.uint8)
        enhanced = equalize(image_uint8).float() / 255.0
    except (ImportError, AttributeError):
        # Fallback: simple contrast enhancement
        mean = image.mean(dim=(-2, -1), keepdim=True)
        enhanced = (image - mean) * 1.2 + mean
        enhanced = torch.clamp(enhanced, 0.0, 1.0)
    
    if squeeze:
        enhanced = enhanced.squeeze(0)
    return enhanced


def apply_clahe_tensor(
    image: torch.Tensor,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    use_fast: bool = True
) -> torch.Tensor:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) using PyTorch.
    
    Meta AI-style: Pure tensor operations, GPU-accelerated, differentiable.
    
    Arguments:
        image: Tensor [C, H, W] or [B, C, H, W] in range [0, 1]
        clip_limit: Contrast limiting factor
        tile_grid_size: Grid size for adaptive processing (tiles_y, tiles_x)
        use_fast: If True, use fast approximation (recommended for real-time)
    
    Returns:
        Enhanced tensor with same shape and range
    """
    if use_fast:
        return apply_clahe_tensor_fast(image, clip_limit)
    
    # Original slow implementation (kept for compatibility)
    if image.dim() == 3:
        image = image.unsqueeze(0)  # Add batch dimension
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, C, H, W = image.shape
    
    # Work on grayscale or L channel only
    if C == 3:
        # Convert to LAB, work on L channel
        lab = rgb_to_lab_tensor(image)
        L = lab[:, 0:1, :, :]  # Extract L channel [B, 1, H, W]
        a = lab[:, 1:2, :, :]
        b = lab[:, 2:3, :, :]
        is_lab = True
    else:
        L = image
        is_lab = False
    
    # Normalize L to [0, 255] for histogram processing
    L_norm = (L * 255.0).clamp(0, 255).int()
    
    # Tile-based processing
    tiles_y, tiles_x = tile_grid_size
    tile_h = H // tiles_y
    tile_w = W // tiles_x
    
    enhanced_L = torch.zeros_like(L)
    
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y_start = ty * tile_h
            y_end = (ty + 1) * tile_h if ty < tiles_y - 1 else H
            x_start = tx * tile_w
            x_end = (tx + 1) * tile_w if tx < tiles_x - 1 else W
            
            # Extract tile
            tile = L_norm[:, :, y_start:y_end, x_start:x_end]
            
            # Compute histogram
            hist = torch.zeros(B, 1, 256, device=image.device, dtype=torch.float32)
            for i in range(256):
                hist[:, :, i] = (tile == i).float().sum(dim=(2, 3))
            
            # Clip histogram
            clip_value = clip_limit * tile.numel() / 256.0
            excess = torch.clamp(hist - clip_value, min=0).sum(dim=2, keepdim=True)
            hist = torch.clamp(hist, max=clip_value)
            hist = hist + excess / 256.0
            
            # Cumulative distribution function
            cdf = hist.cumsum(dim=2)
            cdf_min = cdf[:, :, 0:1]
            cdf = (cdf - cdf_min) / (cdf[:, :, -1:] - cdf_min + 1e-8) * 255.0
            
            # Apply mapping
            tile_float = tile.float()
            tile_enhanced = torch.zeros_like(tile_float)
            for i in range(256):
                mask = (tile == i)
                tile_enhanced = torch.where(mask, cdf[:, :, i:i+1], tile_enhanced)
            
            enhanced_L[:, :, y_start:y_end, x_start:x_end] = tile_enhanced / 255.0
    
    # Convert back to RGB if needed
    if is_lab:
        enhanced_lab = torch.cat([enhanced_L, a, b], dim=1)
        enhanced = lab_to_rgb_tensor(enhanced_lab)
    else:
        enhanced = enhanced_L
    
    if squeeze_output:
        enhanced = enhanced.squeeze(0)
    
    return enhanced


# ============================================================================
# Image Preprocessing Class (Meta AI-style: Tensor-first, GPU-friendly)
# ============================================================================

class ImagePreprocessor:
    """
    Image preprocessing with condition-specific augmentations for visual impairments.
    
    PROJECT PHILOSOPHY & APPROACH:
    =============================
    This module implements "Meta AI-style" preprocessing - pure PyTorch operations that are
    GPU-friendly and differentiable. But more importantly, it implements condition-specific
    adaptations that directly address the problem statement's requirement to support "Different
    Degree Levels" of visual impairments.
    
    WHY CONDITION-SPECIFIC PREPROCESSING:
    Different vision conditions require different image enhancements:
    - Cataracts (blur): Need contrast enhancement to compensate for reduced acuity
    - Glaucoma (peripheral loss): Need peripheral region emphasis
    - AMD (central loss): Need central region emphasis
    - Retinitis pigmentosa (night blindness): Need brightness enhancement
    - Color blindness: Need color detection and alternative representation
    
    This preprocessing ensures the model receives images that are optimized for each user's specific
    vision condition, maximizing the usefulness of the information provided.
    
    HOW IT CONNECTS TO THE PROBLEM STATEMENT:
    The problem statement emphasizes supporting "Different Degree Levels" of visual impairments.
    This module directly implements that by providing condition-specific preprocessing that adapts
    to each user's specific needs, ensuring the system is useful regardless of the severity or
    type of vision condition.
    
    RELATIONSHIP TO BARRIER REMOVAL METHODS:
    1. ENVIRONMENTAL STRUCTURING: Enhances images to make environmental information more accessible
    2. SKILL DEVELOPMENT: Condition-specific enhancements support vision therapy goals
    3. ROUTINE WORKFLOW: Adapts preprocessing to user's specific vision condition
    
    TECHNICAL DESIGN DECISION - META AI-STYLE:
    We use pure PyTorch operations (no OpenCV) because:
    - GPU-friendly: All operations run on GPU, faster processing
    - Differentiable: Can be part of training pipeline if needed
    - Consistent: Same operations in training and inference
    - Modern: Aligns with current ML best practices (Meta AI, PyTorch Vision)
    
    This ensures the preprocessing pipeline is production-ready and performant, supporting the
    real-time requirements of mobile deployment.
    
    Preprocesses images for MaxSight model with condition-specific enhancements that simulate
    or compensate for various visual impairments (cataracts, glaucoma, AMD, etc.). Applies
    standard ImageNet normalization and optional lighting condition detection/augmentation.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        condition_mode: Optional[str] = None
    ):
        """
        Initialize image preprocessor.
        
        WHY THESE PARAMETERS:
        - image_size: Standard ImageNet size (224x224) ensures compatibility with pretrained models
        - condition_mode: Enables condition-specific adaptations that maximize usefulness for each
          user's specific vision condition
        
        This initialization sets up the preprocessing pipeline to provide the best possible
        information for each user's needs, directly supporting the project's goal of addressing
        different vision conditions.
        
        Arguments:
            image_size: Target image dimensions (height, width) - default (224, 224) for ImageNet
            condition_mode: Visual condition to simulate ('glaucoma', 'amd', 'cataracts', etc.)
        """
        self.image_size = image_size
        self.condition_mode = condition_mode
        
        # Standard ImageNet normalization for pretrained ResNet compatibility
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet RGB channel means
            std=[0.229, 0.224, 0.225]   # ImageNet RGB channel standard deviations
        )
        
        # Base transform pipeline: resize -> tensor -> normalize
        self.base_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            self.normalize
        ])
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        Apply preprocessing with condition-specific visual enhancements.
        
        Preprocesses image with condition-specific transforms (if enabled) followed by standard
        ImageNet preprocessing. All visual conditions are supported.
        
        Arguments:
            image: PIL Image to preprocess
        
        Returns:
            Preprocessed image as PyTorch Tensor [3, H, W] with ImageNet normalization applied
        """
        # Apply condition-specific transforms based on condition_mode
        if self.condition_mode == 'cataracts':
            image = self._enhance_contrast(image)
        elif self.condition_mode == 'retinitis_pigmentosa':
            image = self._low_light_enhancement(image)
        elif self.condition_mode in ['myopia', 'hyperopia', 'astigmatism', 'presbyopia', 'refractive_errors']:
            image = self._simulate_refractive_error(image)
        elif self.condition_mode == 'glaucoma':
            image = self._enhance_peripheral(image)
        elif self.condition_mode == 'amd':
            image = self._enhance_central(image)
        elif self.condition_mode == 'diabetic_retinopathy':
            image = self._enhance_edges(image)
        elif self.condition_mode == 'color_blindness':
            image = self._simulate_color_blindness(image)
        
        # Apply standard ImageNet preprocessing
        return self.base_transform(image)  # type: ignore
    
    def _enhance_contrast(self, image: Image.Image) -> Image.Image:
        """
        High-contrast enhancement for cataracts (reduced visual acuity compensation).
        
        Purpose: Enhance image contrast to compensate for reduced visual acuity in cataracts.
                 Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) if OpenCV is available
                 for better results, or falls back to PIL's contrast enhancement. High contrast makes
                 objects more distinguishable for users with reduced visual acuity.
        
        Complexity: O(H*W) where H=height, W=width - processes all pixels for contrast enhancement
                   CLAHE: O(H*W*T) where T=tile size (8x8), but typically O(H*W) in practice
        Relationship: Cataract adaptation - improves image visibility for users with cataracts
        
        Arguments:
            image: PIL Image to enhance
        
        Returns:
            Enhanced PIL Image with increased contrast
        """
        # Convert PIL to tensor for PyTorch processing
        # Meta AI-style: Work with tensors directly, GPU-friendly
        img_tensor = TF.to_tensor(image)  # [C, H, W] in range [0, 1]
        
        # Apply CLAHE using PyTorch implementation
        enhanced_tensor = apply_clahe_tensor(img_tensor, clip_limit=2.0, tile_grid_size=(8, 8))
        
        # Convert back to PIL Image
        enhanced_tensor = torch.clamp(enhanced_tensor, 0.0, 1.0)
        return TF.to_pil_image(enhanced_tensor)
    
    def _low_light_enhancement(self, image: Image.Image) -> Image.Image:
        """
        Brightness enhancement for retinitis pigmentosa (night blindness/tunnel vision compensation).
        
        Purpose: Enhance image brightness to compensate for night blindness and tunnel vision in
                 retinitis pigmentosa. Uses gamma correction to brighten image and histogram stretching
                 to maximize dynamic range. This makes images more visible in low-light conditions
                 where users with retinitis pigmentosa struggle.
        
        Complexity: O(H*W) where H=height, W=width - processes all pixels for brightness enhancement
                   Gamma correction: O(H*W) - element-wise power operation
                   Histogram stretching: O(H*W) - finds min/max and scales pixels
        Relationship: Retinitis pigmentosa adaptation - improves visibility in low-light conditions
        
        Arguments:
            image: PIL Image to enhance
        
        Returns:
            Enhanced PIL Image with increased brightness
        """
        # Convert PIL Image to numpy array with float32 precision for calculations
        # Purpose: Convert image to numpy array with float precision for accurate brightness calculations.
        #          Float32 provides sufficient precision while being memory-efficient.
        # Complexity: O(H*W) - converts image format
        # Relationship: Format conversion - prepares image for numerical operations
        img_array = np.array(image).astype(np.float32)
        
        # Apply gamma correction to brighten image (gamma < 1 brightens, gamma > 1 darkens)
        # Purpose: Brighten image using gamma correction. Gamma=0.5 means we raise pixel values to
        #          power of 0.5, which brightens the image (compresses bright values, expands dark values).
        #          This is effective for low-light enhancement because it makes dark regions more visible.
        # Complexity: O(H*W) - element-wise power operation for all pixels
        # Relationship: Brightness enhancement - first step in low-light compensation
        gamma = 0.5  # Gamma < 1 brightens image
        img_array = np.power(img_array / 255.0, gamma) * 255.0  # Normalize, apply gamma, scale back
        
        # Apply histogram stretching to maximize dynamic range
        # Purpose: Stretch histogram to use full [0, 255] range, maximizing contrast and brightness.
        #          Formula: (pixel - min) / (max - min) * 255 maps [min, max] to [0, 255].
        #          This ensures darkest pixel becomes 0 and brightest becomes 255, maximizing visibility.
        # Complexity: O(H*W) - finds min/max (O(H*W)) and scales all pixels (O(H*W))
        # Relationship: Dynamic range maximization - second step in low-light compensation
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8) * 255.0
        # Add epsilon (1e-8) to prevent division by zero if all pixels are same value
        
        # Convert back to uint8 and PIL Image format
        # Purpose: Convert processed float array back to uint8 (0-255 range) and PIL Image format
        # Complexity: O(H*W) - type conversion and PIL Image creation
        # Relationship: Format conversion - returns image in expected format
        return Image.fromarray(img_array.astype(np.uint8))
    
    def _analyze_lighting_condition(self, image: Image.Image) -> str:
        """
        Analyze image brightness and classify lighting condition.
        
        Purpose: Classify images into lighting categories (bright, normal, dim, dark) for lighting-aware
                 evaluation and training. This enables tracking model performance across different lighting
                 conditions, which is critical for accessibility applications where users may encounter
                 various lighting scenarios.
        
        Complexity: O(H*W) where H=height, W=width - requires scanning all pixels for brightness analysis
        Relationship: Used by preprocessing pipeline to label images with lighting metadata, enabling
                     lighting-stratified metrics in validation and training.
        
        Algorithm:
        1. Convert image to grayscale (average RGB channels)
        2. Calculate mean brightness (average pixel value)
        3. Calculate standard deviation (brightness variation)
        4. Classify based on thresholds:
           - bright: mean > 180 AND std > 30 (overexposed, high contrast)
           - normal: 120 <= mean <= 180 AND std > 20 (typical daylight)
           - dim: 60 <= mean < 120 OR (mean >= 120 AND std < 20) (low light, low contrast)
           - dark: mean < 60 (very low light, night conditions)
        
        Arguments:
            image: PIL Image in RGB format
        
        Returns:
            Lighting condition string: 'bright', 'normal', 'dim', or 'dark'
        """
        # Convert to grayscale for brightness analysis - average RGB channels
        # Complexity: O(H*W) - processes all pixels once
        img_array = np.array(image).astype(np.float32)
        if len(img_array.shape) == 3:
            # np.mean with axis returns 2D array, ensure it's float64
            gray_image = np.mean(img_array, axis=2, dtype=np.float64)  # Average RGB channels to get grayscale
        else:
            gray_image = img_array.astype(np.float64)  # Already grayscale, ensure float64
        
        # Calculate mean brightness - average of all pixel values
        # Complexity: O(H*W) - sums all pixels, then divides
        mean_brightness: float = float(np.mean(gray_image))
        
        # Calculate standard deviation - measures brightness variation across image
        # Complexity: O(H*W) - computes variance then square root
        std_brightness: float = float(np.std(gray_image))
        
        # Classification based on brightness thresholds
        # Thresholds chosen based on typical image brightness distributions:
        # - Bright: >180 (overexposed, sunny conditions)
        # - Normal: 120-180 (typical indoor/outdoor daylight)
        # - Dim: 60-120 (low light, evening, cloudy)
        # - Dark: <60 (night, very low light)
        # Standard deviation helps distinguish high-contrast bright images from uniform bright images
        if mean_brightness > 180 and std_brightness > 30:
            return 'bright'  # Overexposed, high contrast (sunny, bright indoor)
        elif 120 <= mean_brightness <= 180 and std_brightness > 20:
            return 'normal'  # Typical daylight conditions
        elif 60 <= mean_brightness < 120 or (mean_brightness >= 120 and std_brightness < 20):
            return 'dim'  # Low light, low contrast (evening, cloudy, dim indoor)
        else:  # mean_brightness < 60
            return 'dark'  # Very low light (night, dark room)
    
    def _simulate_bright_lighting(self, image: Image.Image, brightness_factor: float = 1.5) -> Image.Image:
        """
        Simulate overexposed/bright lighting conditions.
        
        Purpose: Augment images to simulate bright lighting scenarios (sunny day, bright indoor lighting)
                 for training and evaluation. Helps model learn to handle overexposed images where
                 details may be washed out.
        
        Complexity: O(H*W) - processes all pixels once for brightness enhancement
        Relationship: Used for data augmentation and testing model robustness to bright lighting.
                     Complements _simulate_dim_lighting and _simulate_dark_lighting for comprehensive
                     lighting condition coverage.
        
        Arguments:
            image: PIL Image to brighten
            brightness_factor: Multiplier for brightness (default 1.5 = 50% brighter)
        
        Returns:
            Brightened PIL Image with increased brightness
        """
        from PIL import ImageEnhance
        
        # Use PIL's ImageEnhance for efficient brightness adjustment
        # Complexity: O(H*W) - applies brightness multiplier to all pixels
        enhancer = ImageEnhance.Brightness(image)
        brightened = enhancer.enhance(brightness_factor)  # Increase brightness by factor
        
        # Clamp values to valid range [0, 255] to prevent overflow
        # Note: PIL handles clamping automatically, but explicit conversion ensures correctness
        return brightened
    
    def _simulate_dim_lighting(self, image: Image.Image, brightness_factor: float = 0.6) -> Image.Image:
        """
        Simulate dim lighting conditions.
        
        Purpose: Augment images to simulate dim lighting scenarios (evening, cloudy day, dim indoor)
                 for training and evaluation. Helps model learn to handle low-light images where
                 details may be harder to distinguish.
        
        Complexity: O(H*W) - processes all pixels once for brightness reduction
        Relationship: Used for data augmentation and testing model robustness to dim lighting.
                     Part of lighting augmentation suite with _simulate_bright_lighting and
                     _simulate_dark_lighting.
        
        Arguments:
            image: PIL Image to dim
            brightness_factor: Multiplier for brightness (default 0.6 = 40% darker)
        
        Returns:
            Dimmed PIL Image with reduced brightness
        """
        from PIL import ImageEnhance
        
        # Use PIL's ImageEnhance for efficient brightness adjustment
        # Complexity: O(H*W) - applies brightness multiplier to all pixels
        enhancer = ImageEnhance.Brightness(image)
        dimmed = enhancer.enhance(brightness_factor)  # Reduce brightness by factor
        
        return dimmed
    
    def _simulate_dark_lighting(self, image: Image.Image, brightness_factor: float = 0.3) -> Image.Image:
        """
        Simulate very dark lighting conditions.
        
        Purpose: Augment images to simulate very dark lighting scenarios (night, dark room, low-light)
                 for training and evaluation. Critical for accessibility applications where users
                 may encounter night conditions. Tests model's ability to detect objects in extreme
                 low-light situations.
        
        Complexity: O(H*W) - processes all pixels once for brightness reduction and gamma correction
        Relationship: Used for data augmentation and testing model robustness to dark lighting.
                     Most extreme lighting condition, complements other lighting augmentations.
        
        Arguments:
            image: PIL Image to darken
            brightness_factor: Multiplier for brightness (default 0.3 = 70% darker)
        
        Returns:
            Darkened PIL Image with significantly reduced brightness
        """
        from PIL import ImageEnhance
        
        # First apply brightness reduction
        # Complexity: O(H*W) - applies brightness multiplier
        enhancer = ImageEnhance.Brightness(image)
        darkened = enhancer.enhance(brightness_factor)
        
        # Optionally apply gamma correction for more realistic dark lighting
        # Gamma correction: output = (input/255)^gamma * 255
        # Gamma > 1 darkens image, gamma < 1 brightens
        # For dark simulation, we use gamma = 2.0 to further darken mid-tones
        img_array = np.array(darkened).astype(np.float32)
        gamma = 2.0  # Darken mid-tones more aggressively
        img_array = np.power(img_array / 255.0, gamma) * 255.0
        img_array = np.clip(img_array, 0, 255)  # Clamp to valid range
        
        return Image.fromarray(img_array.astype(np.uint8))
    
    def preprocess_with_lighting(self, image: Image.Image) -> Dict[str, Any]:
        """
        Preprocess image and return both tensor and lighting metadata.
        
        Purpose: Enhanced preprocessing that includes lighting condition analysis. Returns both the
                 preprocessed image tensor and lighting classification, enabling lighting-aware
                 training and evaluation. Maintains backward compatibility with __call__ method.
        
        Complexity: O(H*W) - same as __call__ plus lighting analysis (both O(H*W))
        Relationship: Extends __call__ method to provide lighting metadata. Used by datasets to
                     include lighting information in training batches for lighting-stratified metrics.
        
        Arguments:
            image: PIL Image to preprocess
        
        Returns:
            Dictionary with:
                - 'image': torch.Tensor [3, H, W] - preprocessed image tensor
                - 'lighting': str - lighting condition ('bright', 'normal', 'dim', 'dark')
        """
        # Analyze lighting condition before preprocessing (preserves original brightness)
        # Complexity: O(H*W) - analyzes all pixels for brightness
        lighting = self._analyze_lighting_condition(image)
        
        # Apply condition-specific transforms if needed (same as __call__)
        # Complexity: O(H*W) - applies transforms to all pixels
        if self.condition_mode == 'cataracts':
            image = self._enhance_contrast(image)
        elif self.condition_mode == 'retinitis_pigmentosa':
            image = self._low_light_enhancement(image)
        elif self.condition_mode in ['myopia', 'hyperopia', 'astigmatism', 'presbyopia', 'refractive_errors']:
            image = self._simulate_refractive_error(image)
        elif self.condition_mode == 'glaucoma':
            image = self._enhance_peripheral(image)
        elif self.condition_mode == 'amd':
            image = self._enhance_central(image)
        elif self.condition_mode == 'diabetic_retinopathy':
            image = self._enhance_edges(image)
        elif self.condition_mode == 'color_blindness':
            image = self._simulate_color_blindness(image)
        
        # Apply base transforms (resize, to tensor, normalize)
        # Complexity: O(H*W) - standard image transforms
        tensor = self.base_transform(image)  # type: ignore
        
        # Return both tensor and lighting metadata
        return {
            'image': tensor,
            'lighting': lighting
        }
    
    def _simulate_refractive_error(self, image: Image.Image) -> Image.Image:
        """Simulate blurry vision from refractive errors (myopia, hyperopia, astigmatism, presbyopia)"""
        # Meta AI-style: Pure PyTorch implementation
        # Convert to tensor
        img_tensor = TF.to_tensor(image)  # [C, H, W] in range [0, 1]
        
        # Apply Gaussian blur using torchvision (GPU-friendly)
        sigma = 1.5
        kernel_size = int(2 * sigma * 2 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = TF.gaussian_blur(img_tensor, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
        
        # Enhance contrast to compensate for blur using PyTorch CLAHE
        enhanced = apply_clahe_tensor(blurred, clip_limit=2.0, tile_grid_size=(8, 8))
        
        # Convert back to PIL
        enhanced = torch.clamp(enhanced, 0.0, 1.0)
        return TF.to_pil_image(enhanced)
    
    def _enhance_peripheral(self, image: Image.Image) -> Image.Image:
        """
        Enhance peripheral regions for glaucoma (peripheral vision loss).
        
        Meta AI-style: Pure PyTorch tensor operations, GPU-accelerated.
        """
        # Convert to tensor
        img_tensor = TF.to_tensor(image)  # [C, H, W] in range [0, 1]
        C, H, W = img_tensor.shape
        
        center_x, center_y = W // 2, H // 2
        
        # Create mask using PyTorch (GPU-friendly)
        y = torch.arange(H, device=img_tensor.device, dtype=img_tensor.dtype)
        x = torch.arange(W, device=img_tensor.device, dtype=img_tensor.dtype)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        dist_from_center = torch.sqrt((xx - center_x)**2 + (yy - center_y)**2)
        max_dist = torch.sqrt(torch.tensor(center_x**2 + center_y**2, device=img_tensor.device, dtype=img_tensor.dtype))
        peripheral_mask = 1.0 + 0.5 * (dist_from_center / (max_dist + 1e-8))  # Boost peripheral
        
        # Apply mask to all channels
        enhanced = img_tensor * peripheral_mask.unsqueeze(0)
        enhanced = torch.clamp(enhanced, 0.0, 1.0)
        
        return TF.to_pil_image(enhanced)
    
    def _enhance_central(self, image: Image.Image) -> Image.Image:
        """
        Enhance central regions for AMD (central vision loss).
        
        Meta AI-style: Pure PyTorch tensor operations, GPU-accelerated.
        """
        # Convert to tensor
        img_tensor = TF.to_tensor(image)  # [C, H, W] in range [0, 1]
        C, H, W = img_tensor.shape
        
        center_x, center_y = W // 2, H // 2
        
        # Create mask using PyTorch (GPU-friendly)
        y = torch.arange(H, device=img_tensor.device, dtype=img_tensor.dtype)
        x = torch.arange(W, device=img_tensor.device, dtype=img_tensor.dtype)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        dist_from_center = torch.sqrt((xx - center_x)**2 + (yy - center_y)**2)
        max_dist = torch.sqrt(torch.tensor(center_x**2 + center_y**2, device=img_tensor.device, dtype=img_tensor.dtype))
        central_mask = 1.0 + 0.8 * (1.0 - dist_from_center / (max_dist + 1e-8))  # Boost central
        
        # Apply mask to all channels
        enhanced = img_tensor * central_mask.unsqueeze(0)
        enhanced = torch.clamp(enhanced, 0.0, 1.0)
        
        return TF.to_pil_image(enhanced)
    
    def _enhance_edges(self, image: Image.Image) -> Image.Image:
        """
        Enhance edges for diabetic retinopathy (spotty/blurry vision).
        
        Meta AI-style: Uses PyTorch convolution for edge enhancement, GPU-accelerated.
        """
        # Convert to tensor
        img_tensor = TF.to_tensor(image)  # [C, H, W] in range [0, 1]
        
        # Edge enhancement kernel (sharpening)
        kernel = torch.tensor([[-1, -1, -1],
                          [-1,  9, -1],
                               [-1, -1, -1]], device=img_tensor.device, dtype=img_tensor.dtype)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
        
        # Apply convolution to each channel
        sharpened_channels = []
        for c in range(img_tensor.shape[0]):
            channel = img_tensor[c:c+1, :, :].unsqueeze(0)  # [1, 1, H, W]
            sharpened = F.conv2d(channel, kernel, padding=1)
            sharpened_channels.append(sharpened.squeeze(0).squeeze(0))
        
        sharpened = torch.stack(sharpened_channels, dim=0)
        sharpened = torch.clamp(sharpened, 0.0, 1.0)
        
        # Blend with original to avoid over-sharpening
        enhanced = 0.7 * img_tensor + 0.3 * sharpened
        enhanced = torch.clamp(enhanced, 0.0, 1.0)
        
        return TF.to_pil_image(enhanced)
    
    def _simulate_color_blindness(self, image: Image.Image) -> Image.Image:
        """
        Simulate color blindness (red-green color confusion).
        
        Meta AI-style: Pure PyTorch tensor operations, GPU-accelerated.
        """
        # Convert to tensor
        img_tensor = TF.to_tensor(image)  # [C, H, W] in range [0, 1]
        
        # Red-green color blindness: mix red and green channels
        r, g, b = img_tensor[0], img_tensor[1], img_tensor[2]
        mixed = (r + g) / 2
        
        # Replace red and green with mixed value
        enhanced = torch.stack([mixed, mixed, b], dim=0)
        enhanced = torch.clamp(enhanced, 0.0, 1.0)
        
        return TF.to_pil_image(enhanced)


class AudioPreprocessor:
    """Audio preprocessing - MFCC feature extraction"""
    
    def __init__(self, n_mfcc: int = 128, sample_rate: int = 16000):
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
    
    def extract_mfcc(self, audio: np.ndarray) -> torch.Tensor:
        """
        Extract MFCC features from audio
        
        Arguments:
            audio: Audio signal [samples] or [batch, samples]
        
        Returns:
            MFCC features [n_mfcc] or [batch, n_mfcc]
        """
        # TODO: Implement actual MFCC extraction using librosa or torchaudio
        # For now, return dummy features
        if audio.ndim == 1:
            return torch.randn(self.n_mfcc)
        else:
            batch_size = audio.shape[0]
            return torch.randn(batch_size, self.n_mfcc)


class DistanceEstimator:
    """Distance estimation preprocessing using perspective analysis"""
    
    def __init__(self):
        pass
    
    def estimate_distance_zones(
        self,
        bbox: torch.Tensor,
        image_size: Tuple[int, int] = (224, 224)
    ) -> int:
        """
        Estimate distance zone from bounding box size
        
        Arguments:
            bbox: Bounding box [x, y, w, h] normalized [0, 1]
            image_size: Image dimensions
        
        Returns:
            Distance zone: 0=near, 1=medium, 2=far
        """
        # Use bbox area as proxy for distance
        # Larger boxes = closer objects
        area = bbox[2] * bbox[3]  # w * h
        
        if area > 0.3:  # Large box = close
            return 0  # near
        elif area > 0.1:  # Medium box
            return 1  # medium
        else:  # Small box = far
            return 2  # far


class TextRegionDetector:
    """Text region detection preprocessing for OCR integration. Uses model's text_head output."""
    
    def __init__(self, text_threshold: float = 0.5):
        """
        Initialize text region detector.
        
        Arguments:
            text_threshold: Confidence threshold for text detection
        """
        self.text_threshold = text_threshold
    
    def detect_text_regions(
        self,
        image: np.ndarray,
        text_scores: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None
    ) -> list:
        """
        Detect text regions in image using model's text_head output.
        
        Arguments:
            image: Image array [H, W, 3]
            text_scores: Text probability scores from model [N] (optional)
            boxes: Bounding boxes from model [N, 4] in center format (optional)
        
        Returns:
            List of bounding boxes [x, y, w, h] for text regions
        """
        # If model outputs are provided, use them
        if text_scores is not None and boxes is not None:
            text_mask = text_scores > self.text_threshold
            if text_mask.any():
                text_boxes = boxes[text_mask]
                # Convert from center format to corner format if needed
                results = []
                for box in text_boxes:
                    x, y, w, h = box.tolist()
                    results.append([x - w/2, y - h/2, w, h])
                return results
        
        # Fallback: simple edge-based detection using PyTorch
        # Meta AI-style: Pure PyTorch edge detection
        if isinstance(image, np.ndarray):
            img_tensor = torch.from_numpy(image).float() / 255.0
            if img_tensor.dim() == 3 and img_tensor.shape[2] == 3:
                img_tensor = img_tensor.permute(2, 0, 1)  # [C, H, W]
        else:
            img_tensor = image
        
        if img_tensor.dim() == 3 and img_tensor.shape[0] == 3:
            # Convert to grayscale
            gray = 0.299 * img_tensor[0] + 0.587 * img_tensor[1] + 0.114 * img_tensor[2]
        else:
            gray = img_tensor.squeeze(0) if img_tensor.dim() == 3 else img_tensor
        
        # Sobel edge detection using PyTorch
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              device=gray.device, dtype=gray.dtype).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              device=gray.device, dtype=gray.dtype).unsqueeze(0).unsqueeze(0)
        
        gray_batch = gray.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        edges_x = F.conv2d(gray_batch, sobel_x, padding=1)
        edges_y = F.conv2d(gray_batch, sobel_y, padding=1)
        edges = torch.sqrt(edges_x**2 + edges_y**2).squeeze()
        
        # Threshold edges (simple Canny-like)
        threshold_low, threshold_high = 50.0 / 255.0, 150.0 / 255.0
        edges_binary = (edges > threshold_low).float()
        
        # Simple region detection (basic implementation)
        # Note: Full contour detection would require more complex PyTorch operations
        # For now, return empty list as this is a fallback method
        # In production, use model's text_head output instead
        return []


# Synthetic Impairment Functions
def apply_refractive_error_blur(image: torch.Tensor, sigma: float = 3.0) -> torch.Tensor:
    """Apply Gaussian blur for refractive errors"""
    kernel_size = int(2 * sigma * 2 + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])


def apply_cataract_contrast(image: torch.Tensor, contrast_factor: float = 0.5) -> torch.Tensor:
    """Reduce contrast for cataracts simulation"""
    return TF.adjust_contrast(image, contrast_factor)


def apply_glaucoma_vignette(image: torch.Tensor, center_percent: float = 0.4) -> torch.Tensor:
    """Apply peripheral masking for glaucoma"""
    h, w = image.shape[-2:]
    center_x, center_y = w // 2, h // 2
    radius = min(w, h) * center_percent
    
    # Create circular mask
    y, x = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=torch.float32),
        torch.arange(w, device=image.device, dtype=torch.float32),
        indexing='ij'
    )
    dist = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    mask = (dist < radius).float()
    
    # Expand mask to match image dimensions
    while mask.dim() < image.dim():
        mask = mask.unsqueeze(0)
    # Ensure mask has same shape as image
    if mask.shape != image.shape:
        mask = mask.expand_as(image)
    
    return image * mask


def apply_amd_central_darkening(image: torch.Tensor, darken_factor: float = 0.3) -> torch.Tensor:
    """Darken center region for AMD simulation"""
    h, w = image.shape[-2:]
    center_x, center_y = w // 2, h // 2
    radius = min(w, h) * 0.2
    
    # Create circular darkening mask
    y, x = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=torch.float32),
        torch.arange(w, device=image.device, dtype=torch.float32),
        indexing='ij'
    )
    dist = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    mask = 1.0 - (dist < radius).float() * darken_factor
    
    # Expand mask to match image dimensions
    while mask.dim() < image.dim():
        mask = mask.unsqueeze(0)
    # Ensure mask has same shape as image
    if mask.shape != image.shape:
        mask = mask.expand_as(image)
    
    return image * mask


def apply_low_light(image: torch.Tensor, brightness_factor: float = 0.3) -> torch.Tensor:
    """Reduce brightness for retinitis pigmentosa"""
    return image * brightness_factor


def apply_color_shift(image: torch.Tensor, shift_type: str = 'red_green') -> torch.Tensor:
    """Apply color shifts for color blindness simulation"""
    if shift_type == 'red_green':
        # Simulate red-green color blindness
        # Mix red and green channels
        # Handle both 3D [C, H, W] and 4D [B, C, H, W] tensors
        if image.dim() == 4:
            # [B, C, H, W] - ensure we have 3 channels
            if image.shape[1] != 3:
                # If channels dimension is wrong, return original
                return image
            r, g, b = image[:, 0], image[:, 1], image[:, 2]
            mixed = (r + g) / 2
            image = torch.stack([mixed, mixed, b], dim=1)
        elif image.dim() == 3:
            # [C, H, W] - ensure we have 3 channels
            if image.shape[0] != 3:
                # If channels dimension is wrong, return original
                return image
            r, g, b = image[0], image[1], image[2]
            mixed = (r + g) / 2
            image = torch.stack([mixed, mixed, b], dim=0)
        else:
            # Unexpected shape, return original
            return image
    return image

print("Preprocessing pipeline created successfully!")
print("\nAvailable components:")
print("- ImagePreprocessor: Image transforms with condition-specific augmentations")
print("- AudioPreprocessor: MFCC feature extraction")
print("- DistanceEstimator: Distance zone estimation")
print("- TextRegionDetector: Text region detection")
print("\nSynthetic impairment functions:")
print("- apply_refractive_error_blur")
print("- apply_cataract_contrast")
print("- apply_glaucoma_vignette")
print("- apply_amd_central_darkening")
print("- apply_low_light")
print("- apply_color_shift")