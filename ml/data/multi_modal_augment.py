"""Multi-Modal Augmentation for MaxSight 3.0."""

import torch
import torchvision.transforms as T
import torchaudio.transforms as AT
from typing import Tuple, Optional


class MultiModalAugmentation:
    """Augmentations for vision and audio."""
    
    def __init__(self):
        self.vision_aug = T.Compose([
            T.RandomAdjustSharpness(sharpness_factor=2),
            T.RandomAutocontrast(),
            T.RandomEqualize(),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.3, contrast=0.3),
        ])
        self.audio_aug = AT.TimeMasking(time_mask_param=10)
    
    def __call__(self, image: torch.Tensor, audio: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply augmentations."""
        aug_image = self.vision_aug(image)
        aug_audio = self.audio_aug(audio) if audio is not None else None
        return aug_image, aug_audio

