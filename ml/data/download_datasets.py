"""Dataset download helpers for COCO, Open Images, Objects365, Visual Genome, LVIS, AudioSet."""

import os
import requests
import zipfile
import json
from pathlib import Path
from typing import List, Dict, Optional

from ml.models.maxsight_cnn import COCO_CLASSES
ENVIRONMENTAL_CLASSES = COCO_CLASSES

# 15 Environmental Sound Classes
SOUND_CLASSES = [
    'fire alarm', 'smoke detector', 'doorbell',
    'siren', 'car horn', 'breaking glass',
    'footsteps', 'door closing', 'water running',
    'human voice', 'dog bark', 'cat meow',
    'phone ringing', 'alarm clock', 'vehicle engine'
]


def verify_coco_dataset(data_dir: Path = Path("datasets/coco")) -> Dict[str, bool]:
    """
    Verify COCO dataset is properly downloaded and structured.
    
    Returns:
        Dictionary with verification status for each component
    """
    status = {
        'train_images': False,
        'val_images': False,
        'annotations': False,
        'train_annotations': False,
        'val_annotations': False
    }
    
    # Check directories
    train_img_dir = data_dir / "train2017"
    val_img_dir = data_dir / "val2017"
    ann_dir = data_dir / "annotations"
    
    # Check image directories
    if train_img_dir.exists():
        img_count = len(list(train_img_dir.glob("*.jpg")))
        status['train_images'] = img_count > 100000  # Should have ~118K images
        if status['train_images']:
            print(f"Train images: {img_count} images found")
        else:
            print(f"⚠ Train images: Only {img_count} images found (expected ~118K)")
    
    if val_img_dir.exists():
        img_count = len(list(val_img_dir.glob("*.jpg")))
        status['val_images'] = img_count > 4000  # Should have ~5K images
        if status['val_images']:
            print(f"Val images: {img_count} images found")
        else:
            print(f"⚠ Val images: Only {img_count} images found (expected ~5K)")
    
    # Check annotations
    if ann_dir.exists():
        status['annotations'] = True
        train_ann = ann_dir / "instances_train2017.json"
        val_ann = ann_dir / "instances_val2017.json"
        
        if train_ann.exists():
            try:
                with open(train_ann, 'r') as f:
                    data = json.load(f)
                    status['train_annotations'] = len(data.get('images', [])) > 100000
                    if status['train_annotations']:
                        print(f"Train annotations: {len(data.get('images', []))} images")
            except Exception:
                status['train_annotations'] = False
        
        if val_ann.exists():
            try:
                with open(val_ann, 'r') as f:
                    data = json.load(f)
                    status['val_annotations'] = len(data.get('images', [])) > 4000
                    if status['val_annotations']:
                        print(f"Val annotations: {len(data.get('images', []))} images")
            except Exception:
                status['val_annotations'] = False
    
    return status


def download_coco_dataset(data_dir: Path = Path("datasets/coco"), auto_download: bool = False):
    """
    Download COCO dataset
    
    Args:
        data_dir: Directory to save COCO dataset
        auto_download: If True, attempt automatic download (requires wget/curl)
    
    Note: COCO dataset is large (~20GB). Manual download recommended.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if auto_download:
        print("Attempting automatic COCO dataset download...")
        try:
            import subprocess
            urls = {
                'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
                'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
                'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
            }
            for name, url in urls.items():
                print(f"Downloading {name}...")
                subprocess.run(['wget', '-c', url, '-P', str(data_dir)], check=False)
            print("Download complete. Please extract zip files manually.")
        except Exception as e:
            print(f"Automatic download failed: {e}")
            print("Falling back to manual download instructions...")
            auto_download = False
    
    if not auto_download:
        print("COCO Dataset Download Instructions:")
        print("1. Visit: https://cocodataset.org/#download")
        print("2. Download: 2017 Train images (18GB)")
        print("3. Download: 2017 Val images (1GB)")
        print("4. Download: 2017 Train/Val annotations (241MB)")
        print("5. Extract to: datasets/coco/")
        print("\nOr use COCO API:")
        print("  from pycocotools.coco import COCO")
        print("  coco = COCO('annotations/instances_train2017.json')")
    
    print(f"\nDataset directory: {data_dir}")
    
    # Verify if dataset already exists
    print("\nVerifying existing dataset...")
    status = verify_coco_dataset(data_dir)
    if all(status.values()):
        print("COCO dataset fully verified!")
    else:
        print("COCO dataset incomplete. Please download missing components.")


def download_open_images(data_dir: Path = Path("datasets/open_images")):
    """Download Open Images V7 dataset (9M+ images, 36M+ instances)."""
    print("Open Images V7 Dataset Download Instructions:")
    print("1. Visit: https://storage.googleapis.com/openimages/web/index.html")
    print("2. Download: Open Images V7 Train (9M images, ~500GB)")
    print("3. Download: Open Images V7 Validation (41K images, ~2GB)")
    print("4. Download: Open Images V7 Test (125K images, ~6GB)")
    print("5. Download: Bounding box annotations (CSV format)")
    print("6. Extract to: datasets/open_images/")
    print("\nOr use Kaggle API:")
    print("  kaggle datasets download -d googleai/open-images-v7")
    
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDataset directory created: {data_dir}")
    print("Note: Open Images provides 9M+ images for maximum training data")


def download_objects365(data_dir: Path = Path("datasets/objects365")):
    """Download Objects365 dataset (2M+ images, 30M+ instances)."""
    print("Objects365 Dataset Download Instructions:")
    print("1. Visit: https://www.objects365.org/download.html")
    print("2. Register and request access")
    print("3. Download: Objects365 V2 Train (2M images, ~500GB)")
    print("4. Download: Objects365 V2 Val (80K images, ~20GB)")
    print("5. Download: Annotations (JSON format)")
    print("6. Extract to: datasets/objects365/")
    
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDataset directory created: {data_dir}")
    print("Note: Objects365 provides 2M+ images, 30M+ instances")


def download_visual_genome(data_dir: Path = Path("datasets/visual_genome")):
    """Download Visual Genome dataset (108K images, 3.8M+ instances)."""
    print("Visual Genome Dataset Download Instructions:")
    print("1. Visit: https://visualgenome.org/api/v0/api_home.html")
    print("2. Download: Visual Genome 1.4 images (108K images, ~20GB)")
    print("3. Download: Scene graphs and annotations (JSON)")
    print("4. Extract to: datasets/visual_genome/")
    
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDataset directory created: {data_dir}")
    print("Note: Visual Genome provides rich scene understanding data")


def download_lvis(data_dir: Path = Path("datasets/lvis")):
    """
    Download LVIS dataset - 164K images, 2.2M+ instances, 1203 classes
    
    Long-tail distribution dataset - good for rare object detection
    """
    print("LVIS Dataset Download Instructions:")
    print("1. Visit: https://www.lvisdataset.org/dataset")
    print("2. Download: LVIS V1.0 Train (100K images, ~20GB)")
    print("3. Download: LVIS V1.0 Val (20K images, ~4GB)")
    print("4. Download: LVIS V1.0 Test (44K images, ~9GB)")
    print("5. Download: Annotations (JSON format)")
    print("6. Extract to: datasets/lvis/")
    
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDataset directory created: {data_dir}")
    print("Note: LVIS provides long-tail distribution for rare objects")


def download_audioset(data_dir: Path = Path("datasets/audioset"), auto_download: bool = False):
    """
    Download AudioSet dataset - 2M+ audio clips, 632 classes
    
    Args:
        data_dir: Directory to save AudioSet dataset
        auto_download: If True, attempt automatic download (requires youtube-dl)
    
    Large-scale audio dataset for audio-visual fusion
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if auto_download:
        print("Attempting automatic AudioSet download...")
        print("Note: AudioSet requires YouTube-DL and API access")
        try:
            import subprocess
            # AudioSet provides CSV files with YouTube video IDs
            # Actual download requires youtube-dl and API keys
            print("AudioSet automatic download requires:")
            print("  1. AudioSet CSV files (from Google Research)")
            print("  2. youtube-dl or yt-dlp installed")
            print("  3. YouTube API key (optional, for faster downloads)")
            print("\nFor now, please use manual download instructions.")
        except Exception as e:
            print(f"Automatic download setup failed: {e}")
            auto_download = False
    
    if not auto_download:
        print("AudioSet Dataset Download Instructions:")
        print("1. Visit: https://research.google.com/audioset/")
        print("2. Request access to AudioSet")
        print("3. Download: Balanced train/val/eval sets (2M+ clips, ~1TB)")
        print("4. Download: CSV files with YouTube video IDs")
        print("5. Use youtube-dl to download audio clips from YouTube")
        print("6. Extract to: datasets/audioset/")
        print("\nExample with youtube-dl:")
        print("  youtube-dl -x --audio-format wav <youtube_url>")
    
    print(f"\nDataset directory: {data_dir}")
    print("Note: AudioSet provides 2M+ audio clips for audio-visual fusion")
    print("Note: Focus on environmental sound classes matching SOUND_CLASSES")


def create_synthetic_impairments():
    """
    Create functions for synthetic impairment simulations.
    These will be applied during training data loading.
    
    Returns:
        Dictionary of impairment functions keyed by condition name
    """
    import numpy as np
    from scipy import ndimage  # type: ignore
    
    def apply_blur(image: np.ndarray, sigma: float = 3.0) -> np.ndarray:
        """Apply Gaussian blur for refractive errors (myopia, hyperopia, astigmatism)."""
        if len(image.shape) == 3:
            return np.stack([ndimage.gaussian_filter(image[:, :, i], sigma=sigma) 
                           for i in range(image.shape[2])], axis=2)
        return ndimage.gaussian_filter(image, sigma=sigma)
    
    def apply_contrast_reduction(image: np.ndarray, factor: float = 0.5) -> np.ndarray:
        """Reduce contrast for cataracts."""
        mean = image.mean()
        return (image - mean) * factor + mean
    
    def apply_peripheral_mask(image: np.ndarray, mask_radius: float = 0.7) -> np.ndarray:
        """Apply peripheral masking for glaucoma (tunnel vision)."""
        h, w = image.shape[:2]
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        mask = np.sqrt((x - center_x)**2 + (y - center_y)**2) / max(center_x, center_y) < mask_radius
        if len(image.shape) == 3:
            mask = mask[:, :, np.newaxis]
        return image * mask
    
    def apply_central_darkening(image: np.ndarray, darken_radius: float = 0.3) -> np.ndarray:
        """Apply central darkening for AMD."""
        h, w = image.shape[:2]
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2) / max(center_x, center_y)
        darken_mask = np.clip(dist / darken_radius, 0, 1)
        if len(image.shape) == 3:
            darken_mask = darken_mask[:, :, np.newaxis]
        return image * (1 - darken_mask * 0.7)
    
    def apply_low_light(image: np.ndarray, brightness: float = 0.3) -> np.ndarray:
        """Reduce brightness for retinitis pigmentosa."""
        return np.clip(image * brightness, 0, 255).astype(image.dtype)
    
    def apply_color_shift(image: np.ndarray, shift_type: str = 'protanopia') -> np.ndarray:
        """Apply color shifts for color blindness."""
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image
        if shift_type == 'protanopia':
            # Red-green color blindness (protanopia)
            matrix = np.array([[0.567, 0.433, 0.0],
                             [0.558, 0.442, 0.0],
                             [0.0, 0.242, 0.758]])
        elif shift_type == 'deuteranopia':
            # Red-green color blindness (deuteranopia)
            matrix = np.array([[0.625, 0.375, 0.0],
                             [0.7, 0.3, 0.0],
                             [0.0, 0.3, 0.7]])
        else:  # tritanopia
            # Blue-yellow color blindness
            matrix = np.array([[0.95, 0.05, 0.0],
                             [0.0, 0.433, 0.567],
                             [0.0, 0.475, 0.525]])
        return np.dot(image.reshape(-1, 3), matrix.T).reshape(image.shape)
    
    impairments = {
        'myopia': lambda img: apply_blur(img, sigma=4.0),
        'hyperopia': lambda img: apply_blur(img, sigma=3.0),
        'astigmatism': lambda img: apply_blur(img, sigma=3.5),
        'cataracts': lambda img: apply_contrast_reduction(apply_blur(img, sigma=2.0), factor=0.5),
        'glaucoma': lambda img: apply_peripheral_mask(img, mask_radius=0.6),
        'amd': lambda img: apply_central_darkening(img, darken_radius=0.3),
        'diabetic_retinopathy': lambda img: apply_contrast_reduction(img, factor=0.6),
        'retinitis_pigmentosa': lambda img: apply_low_light(img, brightness=0.3),
        'color_blindness': lambda img: apply_color_shift(img, shift_type='protanopia'),
        'amblyopia': lambda img: apply_blur(img, sigma=2.0)  # Mild blur for lazy eye
    }
    
    print("\nSynthetic Impairment Functions Created:")
    print(f"  - {len(impairments)} impairment functions available")
    print("  - Functions can be applied during data loading")
    print("  - Supports: myopia, hyperopia, astigmatism, cataracts, glaucoma, AMD,")
    print("             diabetic_retinopathy, retinitis_pigmentosa, color_blindness, amblyopia")
    
    return impairments


def save_class_mappings(data_dir: Path = Path("datasets")):
    """Save class mappings to files"""
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save environmental classes
    with open(data_dir / "environmental_classes.txt", "w") as f:
        for i, cls in enumerate(ENVIRONMENTAL_CLASSES):
            f.write(f"{i}: {cls}\n")
    
    # Save sound classes
    with open(data_dir / "sound_classes.txt", "w") as f:
        for i, cls in enumerate(SOUND_CLASSES):
            f.write(f"{i}: {cls}\n")
    
    print(f"\nClass mappings saved to {data_dir}/")


def get_all_datasets_info() -> Dict[str, Dict]:
    """Get information about all supported datasets"""
    return {
        'coco': {
            'name': 'COCO 2017',
            'images': '200K+',
            'instances': '1.5M+',
            'classes': '80',
            'size': '~25GB',
            'url': 'https://cocodataset.org/'
        },
        'open_images': {
            'name': 'Open Images V7',
            'images': '9M+',
            'instances': '36M+',
            'classes': '600',
            'size': '~500GB',
            'url': 'https://storage.googleapis.com/openimages/web/index.html'
        },
        'objects365': {
            'name': 'Objects365 V2',
            'images': '2M+',
            'instances': '30M+',
            'classes': '365',
            'size': '~500GB',
            'url': 'https://www.objects365.org/'
        },
        'visual_genome': {
            'name': 'Visual Genome',
            'images': '108K',
            'instances': '3.8M+',
            'classes': '80K+',
            'size': '~20GB',
            'url': 'https://visualgenome.org/'
        },
        'lvis': {
            'name': 'LVIS V1.0',
            'images': '164K',
            'instances': '2.2M+',
            'classes': '1203',
            'size': '~35GB',
            'url': 'https://www.lvisdataset.org/'
        },
        'audioset': {
            'name': 'AudioSet',
            'clips': '2M+',
            'classes': '632',
            'size': '~1TB',
            'url': 'https://research.google.com/audioset/'
        }
    }


if __name__ == "__main__":
    print("MaxSight Comprehensive Dataset Acquisition")
    print("Maximum Data for 347-Class Training")
    
    # Create dataset directories
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    # Show all available datasets
    print("\nAvailable Datasets for Maximum Training Data:")
    datasets_info = get_all_datasets_info()
    for key, info in datasets_info.items():
        print(f"\n{info['name']}:")
        if 'images' in info:
            print(f"  Images: {info['images']}")
            print(f"  Instances: {info['instances']}")
        if 'clips' in info:
            print(f"  Audio Clips: {info['clips']}")
        print(f"  Classes: {info['classes']}")
        print(f"  Size: {info['size']}")
        print(f"  URL: {info['url']}")
    
    # Download instructions for all datasets
    print("\nDataset Download Instructions:")
    
    download_coco_dataset()
    download_open_images()
    download_objects365()
    download_visual_genome()
    download_lvis()
    download_audioset()
    
    # Create synthetic impairment info
    create_synthetic_impairments()
    
    # Save class mappings
    save_class_mappings()
    
    # Final verification summary
    print("\nDataset Verification Summary:")
    coco_status = verify_coco_dataset()
    if any(coco_status.values()):
        print("\nCOCO Dataset Status:")
        for key, status in coco_status.items():
            print(f"  {key}: {'PASS' if status else 'FAIL'}")
    else:
        print("\n⚠ No datasets verified. Please download datasets first.")
    
    print("\nTotal Available Training Data:")
    print("  - COCO: 200K+ images, 1.5M+ instances")
    print("  - Open Images: 9M+ images, 36M+ instances")
    print("  - Objects365: 2M+ images, 30M+ instances")
    print("  - Visual Genome: 108K images, 3.8M+ instances")
    print("  - LVIS: 164K images, 2.2M+ instances")
    print("  - AudioSet: 2M+ audio clips")
    print("\n  TOTAL: 11M+ images, 70M+ instances for maximum training data")

