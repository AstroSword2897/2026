"""Dataset loader with environmental context, audio, and condition-specific augmentations."""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np
from PIL import Image
import torchaudio

from ml.models.maxsight_cnn import COCO_CLASSES
from ml.utils.preprocessing import ImagePreprocessor


class MaxSightDataset(Dataset):
    """Dataset for MaxSight: images, audio, annotations with condition-specific augmentations."""
    
    def __init__(
        self,
        data_dir: Path,
        annotation_file: Optional[Path] = None,
        image_dir: Optional[Path] = None,
        audio_dir: Optional[Path] = None,
        condition_mode: Optional[str] = None,
        apply_lighting_augmentation: bool = True,
        max_objects: int = 10
    ):
        """
        Initialize MaxSight dataset.
        
        Arguments:
            data_dir: Root directory containing dataset
            annotation_file: Path to JSON annotation file (COCO format or custom)
            image_dir: Directory containing images (default: data_dir/images)
            audio_dir: Directory containing audio clips (default: data_dir/audio, optional)
            condition_mode: Visual condition to simulate ('glaucoma', 'amd', 'cataracts', etc.)
            apply_lighting_augmentation: Whether to apply lighting condition augmentation
            max_objects: Maximum number of objects per image (for padding)
        """
        self.data_dir = Path(data_dir)
        self.annotation_file = Path(annotation_file) if annotation_file else None
        self.image_dir = Path(image_dir) if image_dir else self.data_dir / 'images'
        self.audio_dir = Path(audio_dir) if audio_dir else (self.data_dir / 'audio' if (self.data_dir / 'audio').exists() else None)
        self.condition_mode = condition_mode
        self.apply_lighting_augmentation = apply_lighting_augmentation
        self.max_objects = max_objects
        
        # Initialize preprocessor with condition-specific transforms
        self.preprocessor = ImagePreprocessor(condition_mode=condition_mode)
        
        # Class name to index mapping (must be defined before _load_annotations)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(COCO_CLASSES)}
        self.idx_to_class = {idx: cls_name for idx, cls_name in enumerate(COCO_CLASSES)}
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Create image/annotation mapping
        self.image_ids = list(self.annotations.keys()) if self.annotations else []
    
    def _load_annotations(self) -> Dict[str, Any]:
        """Load annotations from JSON (COCO or custom format)."""
        if not self.annotation_file or not self.annotation_file.exists():
            return {}
        
        with open(self.annotation_file, 'r') as f:
            data = json.load(f)
        
        annotations = {}
        
        # Detect format: COCO has 'images'/'annotations', custom format is simpler
        if 'images' in data and 'annotations' in data:
            # COCO format: parse standard structure with separate images/annotations/categories sections
            image_map = {img['id']: img for img in data['images']}
            category_map = {cat['id']: cat['name'] for cat in data.get('categories', [])}
            
            # Group annotations by image_id for efficient per-image processing
            for ann in data['annotations']:
                image_id = ann['image_id']
                if image_id not in annotations:
                    img_info = image_map.get(image_id, {})
                    annotations[image_id] = {
                        'image_path': self.image_dir / img_info.get('file_name', f'{image_id}.jpg'),
                        'objects': [],
                        'urgency': 0,  # Default urgency (will be computed from objects)
                        'lighting': 'normal',  # Default lighting (will be detected)
                        'audio_path': None
                    }
                
                # Extract bounding box from COCO format [x, y, width, height] in pixels
                bbox = ann['bbox']
                img_info = image_map[image_id]
                img_width = img_info.get('width', 224)
                img_height = img_info.get('height', 224)
                
                # Convert to center format and normalize to [0, 1]
                cx = (bbox[0] + bbox[2] / 2) / img_width
                cy = (bbox[1] + bbox[3] / 2) / img_height
                w = bbox[2] / img_width
                h = bbox[3] / img_height
                
                # Map COCO category to MaxSight class index
                category_name = category_map.get(ann['category_id'], 'unknown')
                class_idx = self.class_to_idx.get(category_name, 0)
                
                # Estimate distance zone from box area (larger = closer)
                box_area = w * h
                if box_area > 0.1:
                    distance_zone = 0  # Near
                elif box_area > 0.05:
                    distance_zone = 1  # Medium
                else:
                    distance_zone = 2  # Far
                
                # Estimate urgency from category (vehicles/hazards = high urgency)
                urgency_keywords = ['car', 'truck', 'bus', 'vehicle', 'fire', 'hazard', 'stop', 'traffic']
                urgency = 3 if any(kw in category_name.lower() for kw in urgency_keywords) else 0
                
                annotations[image_id]['objects'].append({
                    'box': [cx, cy, w, h],  # Center format, normalized
                    'class': class_idx,
                    'category': category_name,
                    'distance': distance_zone,
                    'urgency': urgency
                })
                
                # Update scene urgency (max of all object urgencies)
                annotations[image_id]['urgency'] = max(annotations[image_id]['urgency'], urgency)
        else:
            # Custom format: assume list of annotations
            for ann in data:
                image_id = ann.get('image_id', ann.get('id', len(annotations)))
                annotations[image_id] = {
                    'image_path': self.image_dir / ann.get('image_path', f'{image_id}.jpg'),
                    'objects': ann.get('objects', []),
                    'urgency': ann.get('urgency', 0),
                    'lighting': ann.get('lighting', 'normal'),
                    'audio_path': ann.get('audio_path')
                }
        
        return annotations
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_id = self.image_ids[idx]
        ann = self.annotations[image_id]
        
        # Load image from file
        image_path = ann['image_path']
        if isinstance(image_path, str):
            image_path = Path(image_path)
        
        # Handle missing/corrupted files with fallback
        if not image_path.exists():
            image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        else:
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception:
                image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Apply preprocessing with optional lighting augmentation
        if self.apply_lighting_augmentation:
            # Use preprocessing with lighting detection
            preprocessed = self.preprocessor.preprocess_with_lighting(image)
            image_tensor = preprocessed['image']
            lighting = preprocessed.get('lighting', ann.get('lighting', 'normal'))
        else:
            # Standard preprocessing without lighting augmentation
            image_tensor = self.preprocessor(image)
            lighting = ann.get('lighting', 'normal')
        
        # Load audio if available
        audio_tensor = None
        if self.audio_dir and ann.get('audio_path'):
            audio_path = self.audio_dir / ann['audio_path']
            if audio_path.exists():
                try:
                    # Load audio and extract features (MFCC or raw waveform)
                    waveform, sample_rate = torchaudio.load(str(audio_path))
                    # Use first channel if stereo, resample to 16kHz if needed
                    if waveform.shape[0] > 1:
                        waveform = waveform[0:1]  # Take first channel
                    if sample_rate != 16000:
                        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                        waveform = resampler(waveform)
                    # Extract MFCC features (13 coefficients, standard for audio)
                    mfcc_transform = torchaudio.transforms.MFCC(
                        sample_rate=16000,
                        n_mfcc=13,
                        melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 23}
                    )
                    audio_tensor = mfcc_transform(waveform)  # [1, 13, T]
                except Exception:
                    # Fallback on error
                    audio_tensor = None
        
        # Extract objects and format targets
        objects = ann.get('objects', [])
        num_objs = min(len(objects), self.max_objects)
        
        # Initialize padded arrays
        labels = torch.zeros(self.max_objects, dtype=torch.long)
        boxes = torch.zeros(self.max_objects, 4, dtype=torch.float32)
        distance = torch.zeros(self.max_objects, dtype=torch.long)
        
        # Fill with actual objects
        for i in range(num_objs):
            obj = objects[i]
            labels[i] = obj.get('class', 0)
            boxes[i] = torch.tensor(obj.get('box', [0.5, 0.5, 0.1, 0.1]), dtype=torch.float32)
            distance[i] = obj.get('distance', 1)
        
        # Get scene urgency (max of all object urgencies)
        urgency = ann.get('urgency', 0)
        if objects:
            urgency = max(urgency, max(obj.get('urgency', 0) for obj in objects))
        
        # Build return dictionary
        result = {
            'images': image_tensor,  # [3, H, W] preprocessed image
            'labels': labels,  # [max_objects] class labels (padded)
            'boxes': boxes,  # [max_objects, 4] bounding boxes in center format (cx, cy, w, h)
            'urgency': torch.tensor(urgency, dtype=torch.long),  # Scene urgency (0-3)
            'distance': distance,  # [max_objects] distance zones (0-2)
            'num_objects': torch.tensor(num_objs, dtype=torch.long),  # Valid object count
            'lighting': lighting  # Lighting condition string
        }
        
        # Add optional fields
        if audio_tensor is not None:
            result['audio'] = audio_tensor  # [1, 13, T] MFCC features
        
        if self.condition_mode:
            result['condition_mode'] = self.condition_mode
        
        return result

