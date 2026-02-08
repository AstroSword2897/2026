#!/usr/bin/env python3
"""MaxSight Accessibility Dataset (Production Version)"""

import json
import random
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
from PIL import Image
from scipy import ndimage  # type: ignore
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# Core Dataset (Pure Loader - No On-the-Fly Augmentation)

class AccessibilityDataset(Dataset):
    """Pure dataset loader for accessibility features."""
    
    def __init__(
        self,
        image_dir: Path,
        label_file: Path,
        target_size: Tuple[int, int] = (224, 224)
    ):
        self.image_dir = Path(image_dir)
        self.target_size = target_size
        
        # Load labels.
        if not label_file.exists():
            raise FileNotFoundError(f"Label file not found: {label_file}")
        
        with open(label_file, 'r') as f:
            self.labels = json.load(f)
        
        # Sort images with labels only (enforce label-image matching)
        self.image_files = [
            p for p in sorted(self.image_dir.glob("*.jpg")) + sorted(self.image_dir.glob("*.png"))
            if p.stem in self.labels
        ]
        
        if not self.image_files:
            raise ValueError(f"No images found in {image_dir} with matching labels in {label_file}")
        
        # Standard ImageNet transforms.
        self.to_tensor = T.ToTensor()
        self.norm = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item."""
        path = self.image_files[idx]
        
        # Load and resize image.
        img = Image.open(path).convert("RGB").resize(self.target_size)
        
        # Convert to tensor and normalize.
        tensor = self.norm(self.to_tensor(img))
        
        # Look up labels for the current image path.
        labels = self.labels[path.stem]
        
        # Validate and format labels.
        formatted_labels = {
            "contrast_sensitivity": float(labels.get("contrast_sensitivity", 0.5)),
            "glare_risk_level": int(labels.get("glare_risk_level", 0)),
            "object_findability": float(labels.get("object_findability", 0.5)),
            "navigation_difficulty": float(labels.get("navigation_difficulty", 0.5)),
        }
        
        # Validate ranges.
        formatted_labels["contrast_sensitivity"] = np.clip(formatted_labels["contrast_sensitivity"], 0.0, 1.0)
        formatted_labels["glare_risk_level"] = np.clip(formatted_labels["glare_risk_level"], 0, 3)
        formatted_labels["object_findability"] = np.clip(formatted_labels["object_findability"], 0.0, 1.0)
        formatted_labels["navigation_difficulty"] = np.clip(formatted_labels["navigation_difficulty"], 0.0, 1.0)
        
        return {
            "image": tensor,
            "image_path": str(path),
            "labels": formatted_labels
        }


# Synthetic Augmentation Engine (Medically-Aligned)

class SyntheticImpairmentEngine:
    """Engine for generating synthetic impairments more realistically."""
    
    @staticmethod
    def apply_contrast_loss(img: Image.Image, level: float) -> Image.Image:
        """Apply contrast loss (medically-grounded)."""
        arr = np.array(img).astype(np.float32)
        mean = arr.mean(axis=(0, 1), keepdims=True)
        
        # Mean-preserving contrast reduction. Level 1.0 = full contrast, level 0.0 = gray (no contrast)
        contrast_factor = level
        arr = (arr - mean) * contrast_factor + mean
        
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    
    @staticmethod
    def apply_glare(img: Image.Image, intensity: float) -> Image.Image:
        """Apply veiling glare (medically-grounded)."""
        arr = np.array(img).astype(np.float32)
        h, w = arr.shape[:2]
        
        # Place glare source (prefer upper regions, like sunlight)
        gx = random.randint(int(w * 0.2), int(w * 0.8))
        gy = random.randint(0, int(h * 0.4))  # Upper region.
        
        # Create Gaussian glare mask (high-frequency like real lens flare)
        y, x = np.ogrid[:h, :w]
        distance_sq = (x - gx)**2 + (y - gy)**2
        sigma_sq = (min(w, h) / 6.0) ** 2
        mask = np.exp(-distance_sq / (2 * sigma_sq))
        
        # Apply veiling glare (additive, not multiplicative)
        if len(arr.shape) == 3:
            mask = mask[:, :, np.newaxis]
        
        glare = mask * intensity * 200  # Scale glare intensity.
        arr = np.clip(arr + glare, 0, 255)
        
        return Image.fromarray(arr.astype(np.uint8))
    
    @staticmethod
    def apply_peripheral_blur(img: Image.Image, amount: float) -> Image.Image:
        """Apply peripheral blur (medically-grounded)."""
        arr = np.array(img).astype(np.float32)
        
        # Apply Gaussian blur (handles both grayscale and RGB) Pre-allocate output to avoid memory leaks from temporary arrays.
        if len(arr.shape) == 3:
            # Multi-channel: apply filter to each channel with pre-allocated output.
            blurred = np.empty_like(arr)
            for i in range(arr.shape[2]):
                ndimage.gaussian_filter(arr[:, :, i], sigma=amount, output=blurred[:, :, i])
        else:
            # Grayscale: single channel.
            blurred = np.empty_like(arr)
            ndimage.gaussian_filter(arr, sigma=amount, output=blurred)
        
        # Create radial mask (center clear, periphery blurred)
        h, w = arr.shape[:2]
        mask = SyntheticImpairmentEngine._radial_mask((h, w))
        
        # Blend: center = original, periphery = blurred.
        if len(arr.shape) == 3:
            mask = mask[:, :, np.newaxis]
        
        blended = arr * mask + blurred * (1 - mask)
        
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    
    @staticmethod
    def apply_depth_flattening(img: Image.Image, strength: float) -> Image.Image:
        """Apply depth flattening (medically-grounded)."""
        arr = np.array(img).astype(np.float32)
        
        # Reduce local contrast (depth cues rely on contrast) Apply subtle blur to depth-separating edges.
        if strength > 0:
            blurred = ndimage.gaussian_filter(arr, sigma=strength * 2.0)
            arr = arr * (1 - strength * 0.3) + blurred * (strength * 0.3)
        
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    
    @staticmethod
    def apply_halo_effect(img: Image.Image, intensity: float) -> Image.Image:
        """Apply halo effect (medically-grounded)."""
        arr = np.array(img).astype(np.float32)
        
        # Detect edges (high contrast regions)
        gray = np.mean(arr, axis=2) if len(arr.shape) == 3 else arr
        edges = ndimage.gaussian_gradient_magnitude(gray, sigma=1.0)
        edges = edges / (edges.max() + 1e-8)
        
        # Create halo (bright ring around edges)
        halo_mask = np.clip(edges * intensity * 0.5, 0, 1)
        if len(arr.shape) == 3:
            halo_mask = halo_mask[:, :, np.newaxis]
        
        arr = np.clip(arr + halo_mask * intensity * 100, 0, 255)
        
        return Image.fromarray(arr.astype(np.uint8))
    
    @staticmethod
    def apply_low_resolution(img: Image.Image, acuity_drop: float) -> Image.Image:
        """Apply low resolution (visual acuity drop)."""
        if acuity_drop <= 0:
            return img
        
        # Reduce resolution then upscale (simulates low acuity)
        scale_factor = 1.0 - acuity_drop * 0.7  # Max 70% reduction.
        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
        
        # Downsample then upsample (creates pixelation/blur)
        low_res = img.resize(new_size, Image.LANCZOS)
        upscaled = low_res.resize(img.size, Image.LANCZOS)
        
        return upscaled
    
    @staticmethod
    def _radial_mask(shape: Tuple[int, int]) -> np.ndarray:
        """Create radial mask (center = 1.0, edges = 0.0). Arguments: shape: (height, width) Returns: Radial mask [H, W] with values [0, 1]."""
        h, w = shape
        y, x = np.ogrid[-h/2:h/2, -w/2:w/2]
        r = np.sqrt(x*x + y*y)
        r = r / (r.max() + 1e-8)  # Normalize to [0, 1].
        
        # Invert: center = 1.0, edges = 0.0.
        mask = 1.0 - r
        
        # Apply smooth falloff.
        mask = np.power(mask, 2.0)
        
        return mask


# Synthetic Dataset Generator.

def generate_synthetic_dataset(
    source: Path,
    output: Path,
    n_per_image: int = 8,
    augmentation_types: Optional[List[str]] = None
) -> Dict[str, int]:
    """Generate synthetic dataset with pre-computed augmentations."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    
    if augmentation_types is None:
        augmentation_types = ['contrast', 'glare', 'peripheral_blur']
    
    synthetic_labels = {}
    engine = SyntheticImpairmentEngine()
    
    # Get source images.
    images = sorted(list(source.glob("*.jpg")) + list(source.glob("*.png")))
    
    if not images:
        raise ValueError(f"No images found in {source}")
    
    print(f"Generating synthetic dataset from {len(images)} source images...")
    
    for img_path in images:
        try:
            original = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
            continue
        
        for i in range(n_per_image):
            # Select augmentation type.
            mode = random.choice(augmentation_types)
            aug_image = original.copy()
            labels = {}
            
            if mode == "contrast":
                level = random.uniform(0.3, 0.9)
                aug_image = engine.apply_contrast_loss(original, level)
                labels = {
                    "contrast_sensitivity": level,
                    "glare_risk_level": 0,
                    "object_findability": max(0.1, level * 0.9),  # Lower contrast = harder to find.
                    "navigation_difficulty": 1.0 - level,  # Lower contrast = harder navigation.
                }
            
            elif mode == "glare":
                intensity = random.uniform(0.3, 1.0)
                aug_image = engine.apply_glare(original, intensity)
                labels = {
                    "contrast_sensitivity": 0.6,  # Glare reduces effective contrast.
                    "glare_risk_level": min(3, int(intensity * 3)),
                    "object_findability": max(0.2, 0.6 - intensity * 0.3),  # Glare obscures objects.
                    "navigation_difficulty": 0.5 + intensity * 0.3,  # Glare increases difficulty.
                }
            
            elif mode == "peripheral_blur":
                sigma = random.uniform(2.0, 5.0)
                aug_image = engine.apply_peripheral_blur(original, sigma)
                labels = {
                    "contrast_sensitivity": 0.8,  # Center still clear.
                    "glare_risk_level": 0,
                    "object_findability": max(0.3, 1.0 - sigma / 5.0),  # Peripheral objects harder to find.
                    "navigation_difficulty": min(1.0, sigma / 5.0),  # Peripheral blur increases difficulty.
                }
            
            elif mode == "depth_flattening":
                strength = random.uniform(0.3, 0.8)
                aug_image = engine.apply_depth_flattening(original, strength)
                labels = {
                    "contrast_sensitivity": 0.7,
                    "glare_risk_level": 0,
                    "object_findability": 0.6,  # Depth cues help findability.
                    "navigation_difficulty": 0.4 + strength * 0.3,  # Depth loss increases difficulty.
                }
            
            elif mode == "halo":
                intensity = random.uniform(0.4, 0.9)
                aug_image = engine.apply_halo_effect(original, intensity)
                labels = {
                    "contrast_sensitivity": 0.7,
                    "glare_risk_level": min(3, int(intensity * 2)),  # Halos are a form of glare.
                    "object_findability": max(0.3, 0.7 - intensity * 0.2),
                    "navigation_difficulty": 0.4 + intensity * 0.2,
                }
            
            elif mode == "low_resolution":
                acuity_drop = random.uniform(0.3, 0.7)
                aug_image = engine.apply_low_resolution(original, acuity_drop)
                labels = {
                    "contrast_sensitivity": 0.6,
                    "glare_risk_level": 0,
                    "object_findability": max(0.2, 1.0 - acuity_drop * 0.8),
                    "navigation_difficulty": min(1.0, acuity_drop * 0.9),
                }
            
            # Save augmented image.
            aug_id = f"{img_path.stem}_aug{i:03d}"
            aug_file = output / f"{aug_id}.jpg"
            aug_image.save(aug_file, quality=95)
            
            # Store labels.
            synthetic_labels[aug_id] = labels
    
    # Save annotations.
    annotations_file = output / "annotations.json"
    with open(annotations_file, "w") as f:
        json.dump(synthetic_labels, f, indent=2)
    
    stats = {
        'total_samples': len(synthetic_labels),
        'source_images': len(images),
        'augmentations_per_image': n_per_image,
        'output_dir': str(output)
    }
    
    print(f"OK Generated {len(synthetic_labels)} synthetic samples -> {output}")
    print(f"   Annotations saved to: {annotations_file}")
    
    return stats


# Labeling Template.

def create_label_template(path: Path):
    """Create labeling template for user annotation. Arguments: path: Path to save template JSON file."""
    template = {
        "example_image_id": {
            "contrast_sensitivity": 0.0,  # 0-1: How well can user detect contrast?
            "glare_risk_level": 0,  # 0-3: 0=none, 1=low, 2=medium, 3=high.
            "object_findability": 0.0,  # 0-1: How easy to find objects? (therapy-relevant)
            "navigation_difficulty": 0.0,  # 0-1: How difficult to navigate? (therapy-relevant)
            "notes": "Optional notes about the scene, lighting, or user experience"
        }
    }
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"OK Labeling template saved -> {path}")


# Dataset Registry & Utilities.

def combine_datasets(
    real_dir: Path,
    real_labels: Path,
    synthetic_dir: Path,
    synthetic_labels: Path,
    output_dir: Path
) -> Dict[str, Any]:
    """Combine real and synthetic datasets into unified structure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load labels.
    with open(real_labels, 'r') as f:
        real_labels_dict = json.load(f)
    
    with open(synthetic_labels, 'r') as f:
        synthetic_labels_dict = json.load(f)
    
    # Copy images and combine labels.
    combined_labels = {}
    
    # Copy real images.
    real_images = list(Path(real_dir).glob("*.jpg")) + list(Path(real_dir).glob("*.png"))
    for img_path in real_images:
        if img_path.stem in real_labels_dict:
            # Copy image.
            import shutil
            shutil.copy2(img_path, output_dir / img_path.name)
            combined_labels[img_path.stem] = real_labels_dict[img_path.stem]
    
    # Copy synthetic images.
    synthetic_images = list(Path(synthetic_dir).glob("*.jpg")) + list(Path(synthetic_dir).glob("*.png"))
    for img_path in synthetic_images:
        if img_path.stem in synthetic_labels_dict:
            # Copy image.
            import shutil
            shutil.copy2(img_path, output_dir / img_path.name)
            combined_labels[img_path.stem] = synthetic_labels_dict[img_path.stem]
    
    # Save combined labels.
    combined_labels_file = output_dir / "annotations.json"
    with open(combined_labels_file, 'w') as f:
        json.dump(combined_labels, f, indent=2)
    
    stats = {
        'real_samples': len([k for k in combined_labels.keys() if not k.endswith('_aug')]),
        'synthetic_samples': len([k for k in combined_labels.keys() if k.endswith('_aug')]),
        'total_samples': len(combined_labels),
        'output_dir': str(output_dir)
    }
    
    print(f"OK Combined dataset: {stats['real_samples']} real + {stats['synthetic_samples']} synthetic = {stats['total_samples']} total")
    
    return stats


# CLI Interface.

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MaxSight Accessibility Dataset Generator (Production Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:..."""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Generate synthetic dataset.
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic dataset')
    gen_parser.add_argument('--source', type=Path, required=True, help='Source image directory')
    gen_parser.add_argument('--output', type=Path, required=True, help='Output directory')
    gen_parser.add_argument('--n_per_image', type=int, default=8, help='Number of augmentations per image')
    gen_parser.add_argument('--augmentations', nargs='+', 
                           choices=['contrast', 'glare', 'peripheral_blur', 'depth_flattening', 'halo', 'low_resolution'],
                           default=['contrast', 'glare', 'peripheral_blur'],
                           help='Augmentation types to use')
    
    # Create template.
    template_parser = subparsers.add_parser('template', help='Create labeling template')
    template_parser.add_argument('--output', type=Path, required=True, help='Output template file path')
    
    # Combine datasets.
    combine_parser = subparsers.add_parser('combine', help='Combine real and synthetic datasets')
    combine_parser.add_argument('--real_dir', type=Path, required=True, help='Real images directory')
    combine_parser.add_argument('--real_labels', type=Path, required=True, help='Real labels JSON file')
    combine_parser.add_argument('--synthetic_dir', type=Path, required=True, help='Synthetic images directory')
    combine_parser.add_argument('--synthetic_labels', type=Path, required=True, help='Synthetic labels JSON file')
    combine_parser.add_argument('--output', type=Path, required=True, help='Output combined dataset directory')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        stats = generate_synthetic_dataset(
            source=args.source,
            output=args.output,
            n_per_image=args.n_per_image,
            augmentation_types=args.augmentations
        )
        print(f"\nGeneration Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    elif args.command == 'template':
        create_label_template(args.output)
    
    elif args.command == 'combine':
        stats = combine_datasets(
            real_dir=args.real_dir,
            real_labels=args.real_labels,
            synthetic_dir=args.synthetic_dir,
            synthetic_labels=args.synthetic_labels,
            output_dir=args.output
        )
        print(f"\nCombination Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
    
    else:
        parser.print_help()






