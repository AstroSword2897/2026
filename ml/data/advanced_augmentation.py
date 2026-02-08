"""Advanced Data Augmentation for Real-World Robustness."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import Dict, List, Tuple, Optional, Any, Callable
import random
import math
from dataclasses import dataclass, field


@dataclass
class AugmentationConfig:
    """Configuration for augmentation pipeline."""
    # Geometric.
    rotation_range: Tuple[float, float] = (-30, 30)
    scale_range: Tuple[float, float] = (0.8, 1.2)
    translate_range: Tuple[float, float] = (-0.1, 0.1)
    perspective_strength: float = 0.2
    flip_horizontal_prob: float = 0.5
    flip_vertical_prob: float = 0.1
    
    # Photometric.
    brightness_range: Tuple[float, float] = (0.7, 1.3)
    contrast_range: Tuple[float, float] = (0.7, 1.3)
    saturation_range: Tuple[float, float] = (0.7, 1.3)
    hue_range: Tuple[float, float] = (-0.1, 0.1)
    gamma_range: Tuple[float, float] = (0.8, 1.2)
    
    # Noise.
    gaussian_noise_std: float = 0.05
    salt_pepper_prob: float = 0.02
    motion_blur_kernel: int = 7
    motion_blur_prob: float = 0.3
    
    # Occlusion.
    random_erasing_prob: float = 0.3
    random_erasing_scale: Tuple[float, float] = (0.02, 0.2)
    cutout_prob: float = 0.2
    cutout_size: int = 32
    
    # Weather.
    fog_prob: float = 0.1
    rain_prob: float = 0.1
    snow_prob: float = 0.05
    
    # Camera artifacts.
    lens_distortion_prob: float = 0.1
    jpeg_compression_prob: float = 0.2
    jpeg_quality_range: Tuple[int, int] = (50, 95)
    
    # Edge cases.
    extreme_lighting_prob: float = 0.15
    partial_occlusion_prob: float = 0.25


class AdvancedAugmentation:
    """Comprehensive augmentation pipeline for real-world robustness."""
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        
    def __call__(self, image: torch.Tensor, 
                 targets: Optional[Dict] = None) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Apply augmentation pipeline."""
        # Randomly select augmentations.
        augmentations = self._select_augmentations()
        
        for aug_fn in augmentations:
            image, targets = aug_fn(image, targets)
            
        return image, targets
    
    def _select_augmentations(self) -> List[Callable]:
        """Select random subset of augmentations."""
        augmentations = []
        
        # Always apply some geometric.
        if random.random() < self.config.flip_horizontal_prob:
            augmentations.append(self.horizontal_flip)
        if random.random() < self.config.flip_vertical_prob:
            augmentations.append(self.vertical_flip)
        
        # Rotation and scale.
        augmentations.append(self.random_affine)
        
        # Photometric (always apply some)
        augmentations.append(self.color_jitter)
        
        # Noise (selective)
        if random.random() < self.config.motion_blur_prob:
            augmentations.append(self.motion_blur)
        if random.random() < 0.3:
            augmentations.append(self.gaussian_noise)
            
        # Occlusion (selective)
        if random.random() < self.config.random_erasing_prob:
            augmentations.append(self.random_erasing)
        if random.random() < self.config.cutout_prob:
            augmentations.append(self.cutout)
            
        # Weather (rare)
        if random.random() < self.config.fog_prob:
            augmentations.append(self.fog_effect)
        if random.random() < self.config.rain_prob:
            augmentations.append(self.rain_effect)
            
        # Edge cases.
        if random.random() < self.config.extreme_lighting_prob:
            augmentations.append(self.extreme_lighting)
        if random.random() < self.config.partial_occlusion_prob:
            augmentations.append(self.partial_occlusion)
            
        # Camera artifacts.
        if random.random() < self.config.jpeg_compression_prob:
            augmentations.append(self.jpeg_compression)
            
        return augmentations
    
    
    def horizontal_flip(self, image: torch.Tensor, 
                       targets: Optional[Dict] = None) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Horizontal flip with bbox adjustment."""
        image = torch.flip(image, dims=[-1])
        
        if targets and 'boxes' in targets:
            boxes = targets['boxes'].clone()
            if len(boxes) > 0:
                # Flip x coordinates.
                width = image.shape[-1]
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                targets['boxes'] = boxes
                
        return image, targets
    
    def vertical_flip(self, image: torch.Tensor,
                     targets: Optional[Dict] = None) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Vertical flip with bbox adjustment."""
        image = torch.flip(image, dims=[-2])
        
        if targets and 'boxes' in targets:
            boxes = targets['boxes'].clone()
            if len(boxes) > 0:
                height = image.shape[-2]
                boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
                targets['boxes'] = boxes
                
        return image, targets
    
    def random_affine(self, image: torch.Tensor,
                     targets: Optional[Dict] = None) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Random affine transformation."""
        angle = random.uniform(*self.config.rotation_range)
        scale = random.uniform(*self.config.scale_range)
        tx = random.uniform(*self.config.translate_range)
        ty = random.uniform(*self.config.translate_range)
        
        # Build affine matrix.
        theta = torch.tensor([
            [scale * math.cos(math.radians(angle)), 
             -scale * math.sin(math.radians(angle)), tx],
            [scale * math.sin(math.radians(angle)), 
             scale * math.cos(math.radians(angle)), ty]
        ], dtype=image.dtype, device=image.device)
        
        # Ensure batch dimension.
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
            
        grid = F.affine_grid(theta.unsqueeze(0), image.size(), align_corners=False)
        image = F.grid_sample(image, grid, align_corners=False, mode='bilinear', 
                             padding_mode='reflection')
        
        if squeeze:
            image = image.squeeze(0)
            
        # Note: bbox adjustment for rotation is complex, skip for now.
        return image, targets
    
    
    def color_jitter(self, image: torch.Tensor,
                    targets: Optional[Dict] = None) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Random color jittering."""
        # Brightness.
        brightness = random.uniform(*self.config.brightness_range)
        image = image * brightness
        
        # Contrast.
        contrast = random.uniform(*self.config.contrast_range)
        mean = image.mean(dim=(-2, -1), keepdim=True)
        image = (image - mean) * contrast + mean
        
        # Saturation (for RGB)
        if image.shape[-3] == 3:
            saturation = random.uniform(*self.config.saturation_range)
            gray = image.mean(dim=-3, keepdim=True)
            image = (image - gray) * saturation + gray
            
        # Clamp to valid range.
        image = image.clamp(0, 1)
        
        return image, targets
    
    def extreme_lighting(self, image: torch.Tensor,
                        targets: Optional[Dict] = None) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Simulate extreme lighting conditions."""
        condition = random.choice(['overexposed', 'underexposed', 'harsh_shadows'])
        
        if condition == 'overexposed':
            # Simulate overexposure.
            image = image * random.uniform(1.5, 2.5)
            image = image.clamp(0, 1)
            
        elif condition == 'underexposed':
            # Simulate underexposure.
            image = image * random.uniform(0.2, 0.5)
            
        elif condition == 'harsh_shadows':
            # Add shadow gradient.
            h, w = image.shape[-2:]
            shadow = torch.linspace(0.3, 1.0, w, device=image.device)
            shadow = shadow.reshape(1, 1, 1, w).expand_as(image)
            image = image * shadow
            
        return image, targets
    
    
    def gaussian_noise(self, image: torch.Tensor,
                      targets: Optional[Dict] = None) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Add Gaussian noise."""
        noise = torch.randn_like(image) * self.config.gaussian_noise_std
        image = (image + noise).clamp(0, 1)
        return image, targets
    
    def salt_pepper_noise(self, image: torch.Tensor,
                         targets: Optional[Dict] = None) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Add salt and pepper noise."""
        mask = torch.rand_like(image)
        salt_mask = mask < self.config.salt_pepper_prob / 2
        pepper_mask = mask > (1 - self.config.salt_pepper_prob / 2)
        
        image = image.clone()
        image[salt_mask] = 1.0
        image[pepper_mask] = 0.0
        
        return image, targets
    
    def motion_blur(self, image: torch.Tensor,
                   targets: Optional[Dict] = None) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Simulate motion blur."""
        kernel_size = self.config.motion_blur_kernel
        angle = random.uniform(0, 360)
        
        # Create motion blur kernel.
        kernel = torch.zeros(kernel_size, kernel_size, device=image.device)
        center = kernel_size // 2
        
        # Draw line at angle.
        for i in range(kernel_size):
            offset = i - center
            x = int(center + offset * math.cos(math.radians(angle)))
            y = int(center + offset * math.sin(math.radians(angle)))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0
                
        kernel = kernel / kernel.sum()
        kernel = kernel.reshape(1, 1, kernel_size, kernel_size)
        
        # Apply convolution per channel.
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
            
        channels = image.shape[1]
        kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
        padding = kernel_size // 2
        
        image = F.conv2d(image, kernel, padding=padding, groups=channels)
        
        if squeeze:
            image = image.squeeze(0)
            
        return image, targets
    
    
    def random_erasing(self, image: torch.Tensor,
                      targets: Optional[Dict] = None) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Random erasing augmentation."""
        h, w = image.shape[-2:]
        area = h * w
        
        for _ in range(random.randint(1, 3)):
            target_area = random.uniform(*self.config.random_erasing_scale) * area
            aspect_ratio = random.uniform(0.3, 3.3)
            
            eh = int(round(math.sqrt(target_area * aspect_ratio)))
            ew = int(round(math.sqrt(target_area / aspect_ratio)))
            
            if eh < h and ew < w:
                x = random.randint(0, w - ew)
                y = random.randint(0, h - eh)
                
                # Fill with random noise or mean.
                if random.random() < 0.5:
                    image[..., y:y+eh, x:x+ew] = torch.rand_like(
                        image[..., y:y+eh, x:x+ew])
                else:
                    image[..., y:y+eh, x:x+ew] = image.mean()
                    
        return image, targets
    
    def cutout(self, image: torch.Tensor,
              targets: Optional[Dict] = None) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Cutout augmentation."""
        h, w = image.shape[-2:]
        size = self.config.cutout_size
        
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        
        x1 = max(0, x - size // 2)
        x2 = min(w, x + size // 2)
        y1 = max(0, y - size // 2)
        y2 = min(h, y + size // 2)
        
        image = image.clone()
        image[..., y1:y2, x1:x2] = 0
        
        return image, targets
    
    def partial_occlusion(self, image: torch.Tensor,
                         targets: Optional[Dict] = None) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Simulate partial occlusion by another object."""
        h, w = image.shape[-2:]
        
        # Random polygon occlusion.
        num_points = random.randint(3, 6)
        center_x = random.randint(w//4, 3*w//4)
        center_y = random.randint(h//4, 3*h//4)
        radius = random.randint(h//8, h//3)
        
        # Create mask.
        mask = torch.ones_like(image)
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=image.device),
            torch.arange(w, device=image.device),
            indexing='ij'
        )
        
        dist = torch.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        occlusion_mask = dist < radius
        
        # Apply occlusion.
        image = image.clone()
        occlusion_color = random.uniform(0.1, 0.3)
        image[..., occlusion_mask] = occlusion_color
        
        return image, targets
    
    
    def fog_effect(self, image: torch.Tensor,
                  targets: Optional[Dict] = None) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Simulate fog/haze."""
        fog_intensity = random.uniform(0.3, 0.7)
        fog_color = torch.ones_like(image) * 0.8
        
        image = image * (1 - fog_intensity) + fog_color * fog_intensity
        return image, targets
    
    def rain_effect(self, image: torch.Tensor,
                   targets: Optional[Dict] = None) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Simulate rain drops."""
        h, w = image.shape[-2:]
        num_drops = random.randint(50, 200)
        
        rain_layer = torch.zeros_like(image)
        
        for _ in range(num_drops):
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            length = random.randint(3, 10)
            
            # Draw streak.
            for i in range(length):
                ny = min(y + i, h - 1)
                if 0 <= ny < h and 0 <= x < w:
                    rain_layer[..., ny, x] = 0.7
                    
        image = image * 0.8 + rain_layer * 0.2
        return image.clamp(0, 1), targets
    
    
    def jpeg_compression(self, image: torch.Tensor,
                        targets: Optional[Dict] = None) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Simulate JPEG compression artifacts."""
        quality = random.randint(*self.config.jpeg_quality_range)
        
        # Convert to PIL, compress, convert back. For efficiency, simulate with block artifacts.
        block_size = 8
        h, w = image.shape[-2:]
        
        # Round to block boundaries.
        new_h = (h // block_size) * block_size
        new_w = (w // block_size) * block_size
        
        if new_h > 0 and new_w > 0:
            # Average within blocks.
            image_reshaped = image[..., :new_h, :new_w]
            image_reshaped = image_reshaped.reshape(
                *image.shape[:-2], 
                new_h // block_size, block_size,
                new_w // block_size, block_size
            )
            block_means = image_reshaped.mean(dim=(-3, -1), keepdim=True)
            
            # Add some variation based on quality.
            noise_scale = (100 - quality) / 500
            noise = torch.randn_like(block_means) * noise_scale
            block_means = (block_means + noise).clamp(0, 1)
            
            # Expand back.
            image_reshaped = block_means.expand_as(image_reshaped)
            image[..., :new_h, :new_w] = image_reshaped.reshape(
                *image.shape[:-2], new_h, new_w)
            
        return image, targets


class StressTestAugmentation(AdvancedAugmentation):
    """Stress-test augmentation for edge case robustness testing. Applies more aggressive transforms to find model weaknesses."""
    
    def __init__(self):
        config = AugmentationConfig(
            rotation_range=(-45, 45),
            scale_range=(0.5, 1.5),
            brightness_range=(0.3, 2.0),
            contrast_range=(0.3, 2.0),
            gaussian_noise_std=0.15,
            motion_blur_prob=0.5,
            random_erasing_prob=0.5,
            random_erasing_scale=(0.1, 0.4),
            fog_prob=0.3,
            rain_prob=0.2,
            extreme_lighting_prob=0.4,
            partial_occlusion_prob=0.4,
            jpeg_compression_prob=0.4,
            jpeg_quality_range=(20, 70)
        )
        super().__init__(config)


class MixUp:
    """MixUp augmentation for regularization."""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        
    def __call__(self, images: torch.Tensor, 
                 labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply MixUp. Returns: Mixed images, labels_a, labels_b, lambda."""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a = labels
        labels_b = labels[index]
        
        return mixed_images, labels_a, labels_b, lam


class CutMix:
    """CutMix augmentation for regularization."""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        
    def __call__(self, images: torch.Tensor,
                 labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply CutMix."""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        batch_size = images.size(0)
        index = torch.randperm(batch_size, device=images.device)
        
        h, w = images.shape[-2:]
        
        # Get cut region.
        cut_rat = np.sqrt(1 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        x1 = np.clip(cx - cut_w // 2, 0, w)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        
        mixed_images = images.clone()
        mixed_images[..., y1:y2, x1:x2] = images[index, ..., y1:y2, x1:x2]
        
        # Adjust lambda to actual proportion.
        lam = 1 - ((x2 - x1) * (y2 - y1) / (h * w))
        
        return mixed_images, labels, labels[index], lam


def create_augmentation_pipeline(mode: str = 'train') -> AdvancedAugmentation:
    """Create augmentation pipeline based on mode. Args: mode: 'train', 'val', 'test', or 'stress_test'"""
    if mode == 'train':
        return AdvancedAugmentation()
    elif mode == 'stress_test':
        return StressTestAugmentation()
    else:
        # Minimal augmentation for val/test.
        return AdvancedAugmentation(AugmentationConfig(
            rotation_range=(0, 0),
            scale_range=(1, 1),
            flip_horizontal_prob=0,
            flip_vertical_prob=0,
            motion_blur_prob=0,
            random_erasing_prob=0,
            fog_prob=0,
            rain_prob=0,
            extreme_lighting_prob=0,
            partial_occlusion_prob=0
        ))







