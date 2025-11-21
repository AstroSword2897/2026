"""Dataset download helpers for COCO, Open Images, Objects365, Visual Genome, LVIS, AudioSet."""

import os
import requests
import zipfile
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


def download_coco_dataset(data_dir: Path = Path("datasets/coco")):
    """
    Download COCO dataset
    
    Note: COCO dataset is large (~20GB). This script provides instructions.
    For actual download, use official COCO API or manual download.
    """
    print("COCO Dataset Download Instructions:")
    print("1. Visit: https://cocodataset.org/#download")
    print("2. Download: 2017 Train images (18GB)")
    print("3. Download: 2017 Val images (1GB)")
    print("4. Download: 2017 Train/Val annotations (241MB)")
    print("5. Extract to: datasets/coco/")
    print("\nOr use COCO API:")
    print("  from pycocotools.coco import COCO")
    print("  coco = COCO('annotations/instances_train2017.json')")
    
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDataset directory created: {data_dir}")


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


def download_audioset(data_dir: Path = Path("datasets/audioset")):
    """
    Download AudioSet dataset - 2M+ audio clips, 632 classes
    
    Large-scale audio dataset for audio-visual fusion
    """
    print("AudioSet Dataset Download Instructions:")
    print("1. Visit: https://research.google.com/audioset/")
    print("2. Request access to AudioSet")
    print("3. Download: Balanced train/val/eval sets (2M+ clips, ~1TB)")
    print("4. Extract to: datasets/audioset/")
    
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDataset directory created: {data_dir}")
    print("Note: AudioSet provides 2M+ audio clips for audio-visual fusion")


def create_synthetic_impairments():
    """
    Create functions for synthetic impairment simulations
    These will be applied during training data loading
    """
    print("\nSynthetic Impairment Functions:")
    print("These will be implemented in preprocessing pipeline")
    print("- Blur (refractive errors): Gaussian blur σ=2-5")
    print("- Contrast reduction (cataracts): 0.3-0.7")
    print("- Peripheral masking (glaucoma): Vignette")
    print("- Central darkening (AMD): Center region darkening")
    print("- Low-light (retinitis pigmentosa): Brightness reduction")
    print("- Color shifts (color blindness): Color channel manipulation")


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
    print("Dataset Download Instructions:")
    
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
    
    print("Total Available Training Data:")
    print("  - COCO: 200K+ images, 1.5M+ instances")
    print("  - Open Images: 9M+ images, 36M+ instances")
    print("  - Objects365: 2M+ images, 30M+ instances")
    print("  - Visual Genome: 108K images, 3.8M+ instances")
    print("  - LVIS: 164K images, 2.2M+ instances")
    print("  - AudioSet: 2M+ audio clips")
    print("\n  TOTAL: 11M+ images, 70M+ instances for maximum training data")

