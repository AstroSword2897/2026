import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Optional, Any
import json
import numpy as np
from PIL import Image
import torchaudio

from ml.models.maxsight_cnn import COCO_CLASSES, NUM_CLASSES
from ml.utils.preprocessing import ImagePreprocessor

class MaxSightDataset(Dataset):
    def __init__(
            self,
            data_dir: Path,
            annotation_file: Optional[Path] = None,
            image_dir: Optional[Path] = None,
            audio_dir: Optional[Path] = None,
            condition_mode: Optional[str] = None,
            apply_lighting_augumentation: bool = True,
            max_objects: int = 10
    ):
        self.data_dir = Path(data_dir)
        self.annotation_file = Path(annotation_file) if annotation_file else None
        self.image_dir = Path(image_dir) if image_dir else self.data_dir / 'images'
        self.audio_dir = Path(audio_dir) if audio_dir else (self.data_dir / 'audio' if (self.data_dir / 'audio').exists() else None)
        self.condition_mode = condition_mode
        self.apply_lightning_augmentation = apply_lighting_augumentation
        self.max_objects = max_objects
        self.preprocessor = ImagePreprocessor(condition_mode=condition_mode)

        self.anotations = self._load_annotations()

        self.image_ids = list(self.annotations.key()) if self.annotations else []

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate (COCO_CLASSES)}
        self.idx_to_class = {idx: cls_name for idx, cls_name in enumerate (COCO_CLASSES)}

        def _load_annotations(self) -> Dict[str, Any]:
            if not self.annotation_file or not self.annotation_file.exists():
                return {}
            with open(self.annotation_file, 'r') as f:
                data = json.load(f)
            annotations = {}
            if 'images' in data and 'annotations' in data:
                image_map = {img['id']: img for img in data['images']}
                category_map = {cat['id']: cat['name'] for cat in data.get('categories', [])}
                for ann in data['annotations']:
                    image_id = ann['image_id']
                    if image_id not in annotations:
                        img_info = image_map.get(image_id, {})
                        annotations[image_id] = {
                            'image path': self.image_dir / img_info.get('file_name', f'{image_id}.jpg'),
                            'objects': [],
                            'urgency': 0,
                            'lighting': 'normal',
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
                        
                        category_name = category_map.get(ann['category_id'], 'unknown')
                        class_idx = self.class_to_idx.get(category_name, 0)
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
                            'box': [cx, cy, w, h],
                            'class': class_idx,
                            'category': category_name,
                            'distance': distance_zone,
                            'urgency': urgency
                        })

                        annotations[image_id]['urgency'] = max(annotations[image_id]['urgency'], urgency)
                else:
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
        """Return dataset size - O(1) complexity"""
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
       #Load sammple with index
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
        
        if self.apply_lighting_augmentation:
            preprocessed = self.preprocessor.preprocess_with_lighting(image)
            image_tensor = preprocessed['image']
            lighting = preprocessed.get('lighting', ann.get('lighting', 'normal'))
        else:
            image_tensor = self.preprocessor(image)
            lighting = ann.get('lighting', 'normal')
        
        #Audio features
        audio_tensor = None
        if self.audio_dir and ann.get('audio_path'):
            audio_path = self.audio_dir / ann['audio_path']
            if audio_path.exists():
                try:
                    waveform, sample_rate = torchaudio.load(str(audio_path))
                    
                    if waveform.shape[0] > 1:
                        waveform = waveform[0:1]  # Take first channel
                    if sample_rate != 16000:
                        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                        waveform = resampler(waveform)
                    #Extraction
                    mfcc_transform = torchaudio.transforms.MFCC(
                        sample_rate=16000,
                        n_mfcc=13,
                        melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 23}
                    )
                    audio_tensor = mfcc_transform(waveform)  # [1, 13, T]
                except Exception:
                    audio_tensor = None
        
        objects = ann.get('objects', [])
        num_objs = min(len(objects), self.max_objects)
        
        labels = torch.zeros(self.max_objects, dtype=torch.long)
        boxes = torch.zeros(self.max_objects, 4, dtype=torch.float32)
        distance = torch.zeros(self.max_objects, dtype=torch.long)
        
        for i in range(num_objs):
            obj = objects[i]
            labels[i] = obj.get('class', 0)
            boxes[i] = torch.tensor(obj.get('box', [0.5, 0.5, 0.1, 0.1]), dtype=torch.float32)
            distance[i] = obj.get('distance', 1)
        
        urgency = ann.get('urgency', 0)
        if objects:
            urgency = max(urgency, max(obj.get('urgency', 0) for obj in objects))
        
        # Build return dictionary
        result = {
            'images': image_tensor,  # preprocessed image
            'labels': labels,  # class labels 
            'boxes': boxes,  # boxes in format
            'urgency': torch.tensor(urgency, dtype=torch.long),  # Scene urgency (0-3)
            'distance': distance,  # distance zones
            'num_objects': torch.tensor(num_objs, dtype=torch.long),  # Valid object count
            'lighting': lighting  # Lighting condition string
        }
        
        if audio_tensor is not None:
            result['audio'] = audio_tensor
        
        if self.condition_mode:
            result['condition_mode'] = self.condition_mode
        
        return result

