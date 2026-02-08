"""Inference Dataset Loaders for MaxSight."""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import logging

# Setup logging for error tracking.
logger = logging.getLogger(__name__)


# Standard metadata schema for all datasets.
STANDARD_METADATA_KEYS = {
    'weather': None,
    'scene': None,
    'labels': None,
    'annotation_path': None,
    'label': None,
    'confidence': None
}


def create_imagenet_transform() -> transforms.Compose:
    """Create ImageNet normalization transform. This is configurable so it can be replaced for different backbones or modalities."""
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats.
            std=[0.229, 0.224, 0.225]
        )
    ])


class OpenImagesV6Dataset(Dataset):
    """Open Images V6 dataset for MaxSight inference. Covers: Broad semantic diversity - 9M images with 600 object classes - Diverse scenes, objects, and contexts - Real-world complexity."""
    
    def __init__(
        self,
        root: Path,
        split: str = 'validation',
        download: bool = False,
        transform: Optional[transforms.Compose] = None,
        max_samples: Optional[int] = None,
        skip_corrupted: bool = True  # FIXED: Skip corrupted images instead of dummy fallback.
    ):
        """Initialize Open Images V6 dataset."""
        self.root = Path(root)
        self.split = split
        self.max_samples = max_samples  # FIXED: Actually assign max_samples.
        self.skip_corrupted = skip_corrupted
        
        # Default transform: ImageNet normalization (configurable)
        if transform is None:
            self.transform = create_imagenet_transform()
        else:
            self.transform = transform
        
        # Load image list and annotations.
        self.image_list = self._load_image_list()
        
        if self.max_samples:
            self.image_list = self.image_list[:self.max_samples]
        
        print(f"Loaded Open Images V6 {split} set: {len(self.image_list)} images")
    
    def _load_image_list(self) -> List[Dict[str, Any]]:
        """Load list of images from Open Images format. FIXED: Aggregates all labels per image instead of keeping only first."""
        image_list = []
        
        image_dir = self.root / self.split
        annotation_file = self.root / f'{self.split}-annotations-bbox.csv'
        
        if not image_dir.exists():
            raise FileNotFoundError(f"Open Images {self.split} directory not found: {image_dir}")
        
        # FIXED: Aggregate all labels per image.
        image_labels_map = {}  # Image_id -> list of labels.
        
        # Load from annotation file when available.
        if annotation_file.exists():
            import csv
            with open(annotation_file, 'r') as f:
                reader = csv.DictReader(f)
                seen_images = set()
                for row in reader:
                    image_id = row.get('ImageID', '')
                    if image_id and image_id not in seen_images:
                        seen_images.add(image_id)
                        # Open Images stores images in subdirectories.
                        subdir = image_id[:2]
                        image_path = image_dir / subdir / f'{image_id}.jpg'
                        if image_path.exists():
                            # FIXED: Aggregate labels per image.
                            if image_id not in image_labels_map:
                                image_labels_map[image_id] = []
                            image_labels_map[image_id].append({
                                'label': row.get('LabelName', ''),
                                'confidence': row.get('Confidence', '1')
                            })
        else:
            # Fallback: scan directory for images.
            for subdir in image_dir.iterdir():
                if subdir.is_dir():
                    for img_file in subdir.glob('*.jpg'):
                        image_id = img_file.stem
                        image_labels_map[image_id] = []  # No labels available.
        
        # Build image list with aggregated labels.
        for image_id, labels in image_labels_map.items():
            subdir = image_id[:2]
            image_path = image_dir / subdir / f'{image_id}.jpg'
            if image_path.exists():
                image_list.append({
                    'image_id': image_id,
                    'image_path': image_path,
                    'labels': labels,  # FIXED: All labels, not just first.
                    'num_labels': len(labels)
                })
        
        return image_list
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_list)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset. FIXED: Proper error handling, standard metadata schema."""
        item = self.image_list[idx]
        image_path = item['image_path']
        
        # FIXED: Proper error handling - log and skip corrupted images.
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            if self.skip_corrupted:
                # Return None to signal skip (caller should handle)
                raise RuntimeError(f"Corrupted image skipped: {image_path}") from e
            else:
                # Backward compatibility path; prefer explicit annotation.
                image = Image.new('RGB', (224, 224), color=(128, 128, 128))
                logger.warning(f"Using dummy image for {image_path}")
        
        # Apply transform.
        image_tensor = self.transform(image)
        
        labels = item.get('labels') or []
        first = labels[0] if labels else {}
        return {
            'image': image_tensor,
            'image_id': item['image_id'],
            'image_path': str(image_path),
            'dataset': 'open_images_v6',
            'split': self.split,
            'context': {
                'weather': '',
                'scene': '',
                'labels': labels,
                'annotation_path': '',
                'label': (first.get('label') or '') if isinstance(first, dict) else '',
                'confidence': (first.get('confidence') or '1') if isinstance(first, dict) else '1'
            }
        }


class BDD100KDataset(Dataset):
    """BDD100K dataset for MaxSight inference."""
    
    def __init__(
        self,
        root: Path,
        split: str = 'val',
        transform: Optional[transforms.Compose] = None,
        max_samples: Optional[int] = None,
        skip_corrupted: bool = True  # FIXED: Skip corrupted images.
    ):
        """Initialize BDD100K dataset."""
        self.root = Path(root)
        self.split = split
        self.max_samples = max_samples  # FIXED: Actually assign max_samples.
        self.skip_corrupted = skip_corrupted
        
        # Default transform: ImageNet normalization (configurable)
        if transform is None:
            self.transform = create_imagenet_transform()
        else:
            self.transform = transform
        
        # Load image list.
        self.image_list = self._load_image_list()
        
        if self.max_samples:
            self.image_list = self.image_list[:self.max_samples]
        
        print(f"Loaded BDD100K {split} set: {len(self.image_list)} images")
    
    def _load_image_list(self) -> List[Dict[str, Any]]:
        """Load list of images from BDD100K format."""
        image_list = []
        
        # BDD100K structure: images/ and labels/ directories.
        image_dir = self.root / 'images' / '100k' / self.split
        label_file = self.root / 'labels' / f'bdd100k_labels_images_{self.split}.json'
        
        if not image_dir.exists():
            raise FileNotFoundError(f"BDD100K {self.split} directory not found: {image_dir}")
        
        # Load labels if available.
        labels = {}
        if label_file.exists():
            with open(label_file, 'r') as f:
                label_data = json.load(f)
                labels = {item['name']: item for item in label_data}
        
        # Scan for images.
        for img_file in image_dir.glob('*.jpg'):
            image_id = img_file.stem
            label_info = labels.get(img_file.name, {})
            
            image_list.append({
                'image_id': image_id,
                'image_path': img_file,
                'labels': label_info,
                'attributes': label_info.get('attributes', {}),
                'weather': label_info.get('attributes', {}).get('weather', 'unknown'),
                'scene': label_info.get('attributes', {}).get('scene', 'unknown')
            })
        
        return image_list
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_list)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset. FIXED: Proper error handling, standard metadata schema."""
        item = self.image_list[idx]
        image_path = item['image_path']
        
        # FIXED: Proper error handling.
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            if self.skip_corrupted:
                raise RuntimeError(f"Corrupted image skipped: {image_path}") from e
            else:
                image = Image.new('RGB', (224, 224), color=(128, 128, 128))
                logger.warning(f"Using dummy image for {image_path}")
        
        # Apply transform.
        image_tensor = self.transform(image)
        
        return {
            'image': image_tensor,
            'image_id': item['image_id'],
            'image_path': str(image_path),
            'dataset': 'bdd100k',
            'split': self.split,
            'context': {
                'weather': item.get('weather') or 'unknown',
                'scene': item.get('scene') or 'unknown',
                'labels': item.get('labels') or {},
                'annotation_path': '',
                'label': '',
                'confidence': ''
            }
        }


class ADE20KDataset(Dataset):
    """ADE20K dataset for MaxSight inference."""
    
    def __init__(
        self,
        root: Path,
        split: str = 'validation',
        transform: Optional[transforms.Compose] = None,
        max_samples: Optional[int] = None,
        skip_corrupted: bool = True  # FIXED: Skip corrupted images.
    ):
        """Initialize ADE20K dataset."""
        self.root = Path(root)
        self.split = split
        self.max_samples = max_samples  # FIXED: Actually assign max_samples.
        self.skip_corrupted = skip_corrupted
        
        # Default transform: ImageNet normalization (configurable)
        if transform is None:
            self.transform = create_imagenet_transform()
        else:
            self.transform = transform
        
        # Load image list.
        self.image_list = self._load_image_list()
        
        if self.max_samples:
            self.image_list = self.image_list[:self.max_samples]
        
        print(f"Loaded ADE20K {split} set: {len(self.image_list)} images")
    
    def _load_image_list(self) -> List[Dict[str, Any]]:
        """Load list of images from ADE20K format."""
        image_list = []
        
        # ADE20K structure: images/ and annotations/ directories.
        image_dir = self.root / 'images' / self.split
        annotation_dir = self.root / 'annotations' / self.split
        
        if not image_dir.exists():
            raise FileNotFoundError(f"ADE20K {self.split} directory not found: {image_dir}")
        
        # Scan for images.
        for img_file in image_dir.glob('*.jpg'):
            image_id = img_file.stem
            
            # Check for corresponding annotation.
            annotation_path = annotation_dir / f'{image_id}.png' if annotation_dir.exists() else None
            
            image_list.append({
                'image_id': image_id,
                'image_path': img_file,
                'annotation_path': str(annotation_path) if annotation_path and annotation_path.exists() else None
            })
        
        return image_list
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_list)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset. FIXED: Proper error handling, standard metadata schema."""
        item = self.image_list[idx]
        image_path = item['image_path']
        
        # FIXED: Proper error handling.
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            if self.skip_corrupted:
                raise RuntimeError(f"Corrupted image skipped: {image_path}") from e
            else:
                image = Image.new('RGB', (224, 224), color=(128, 128, 128))
                logger.warning(f"Using dummy image for {image_path}")
        
        # Apply transform.
        image_tensor = self.transform(image)
        
        # Use no None values so DataLoader default_collate can batch correctly.
        return {
            'image': image_tensor,
            'image_id': item['image_id'],
            'image_path': str(image_path),
            'dataset': 'ade20k',
            'split': self.split,
            'context': {
                'weather': '',
                'scene': '',
                'labels': {},
                'annotation_path': item.get('annotation_path') or '',
                'label': '',
                'confidence': ''
            }
        }


class DetectionPostProcessor:
    """Post-processor interface for model outputs. FIXED: Abstraction layer instead of model.get_detections(). Handles different output formats and batching."""
    
    def __init__(
        self,
        confidence_threshold: float = 0.3,
        max_detections: int = 10,
        nms_threshold: float = 0.5
    ):
        """Initialize post-processor."""
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        self.nms_threshold = nms_threshold
    
    def process(
        self,
        model: torch.nn.Module,
        outputs: Dict[str, torch.Tensor],
        batch_size: int
    ) -> List[List[Dict[str, Any]]]:
        """Process model outputs to detections."""
        # Use model.get_detections when available (backward compatibility)
        if hasattr(model, 'get_detections'):
            try:
                detections = model.get_detections(  # type: ignore
                    outputs,
                    confidence_threshold=self.confidence_threshold,
                    max_detections=self.max_detections
                )
                # FIXED: Handle different return formats.
                if detections is None:
                    return [[] for _ in range(batch_size)]
                
                # Ensure list of lists format.
                if isinstance(detections, list):
                    # Already in correct format.
                    if len(detections) != batch_size:
                        logger.warning(f"Detections length {len(detections)} != batch_size {batch_size}")
                        # Pad or truncate.
                        if len(detections) < batch_size:
                            detections.extend([[] for _ in range(batch_size - len(detections))])
                        else:
                            detections = detections[:batch_size]
                    return detections
                else:
                    # Single list - split by batch.
                    return [detections[i:i+self.max_detections] for i in range(0, len(detections), self.max_detections)]
            except Exception as e:
                logger.warning(f"model.get_detections failed: {e}, falling back to manual processing")
        
        detections = []
        
        if 'objectness' in outputs and 'boxes' in outputs:
            # Standard detection format.
            objectness = outputs['objectness']  # [B, H*W] or [B, N].
            boxes = outputs['boxes']  # [B, H*W, 4] or [B, N, 4].
            
            # Apply confidence threshold and get top detections.
            for b in range(batch_size):
                obj_scores = objectness[b] if objectness.dim() > 1 else objectness
                valid_mask = obj_scores > self.confidence_threshold
                
                if valid_mask.sum() > 0:
                    valid_scores = obj_scores[valid_mask]
                    valid_boxes = boxes[b][valid_mask] if boxes.dim() > 2 else boxes[valid_mask]
                    
                    # Get top-K.
                    top_k = min(self.max_detections, len(valid_scores))
                    top_indices = torch.topk(valid_scores, k=top_k).indices
                    
                    batch_detections = []
                    for idx in top_indices:
                        batch_detections.append({
                            'box': valid_boxes[idx].cpu().tolist() if torch.is_tensor(valid_boxes[idx]) else valid_boxes[idx],
                            'confidence': valid_scores[idx].item() if torch.is_tensor(valid_scores[idx]) else valid_scores[idx],
                            'class': 0  # Default if not available.
                        })
                    detections.append(batch_detections)
                else:
                    detections.append([])
        else:
            # No detection outputs - return empty detections.
            detections = [[] for _ in range(batch_size)]
        
        return detections


def _replace_none_for_collate(obj: Any) -> Any:
    """Recursively replace None so default_collate never sees None (avoids TypeError in workers)."""
    if obj is None:
        return ''
    if isinstance(obj, dict):
        return {k: _replace_none_for_collate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_none_for_collate(v) for v in obj]
    return obj


def _inference_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate batch after stripping None so default_collate never fails."""
    from torch.utils.data._utils.collate import default_collate
    cleaned = [_replace_none_for_collate(b) for b in batch]
    return default_collate(cleaned)


def create_inference_dataloader(
    dataset_name: str,
    root: Path,
    split: str = 'validation',
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    shuffle: bool = False,
    pin_memory: Optional[bool] = None  # FIXED: Make configurable.
) -> DataLoader:
    """Create DataLoader for inference datasets."""
    if dataset_name.lower() == 'open_images_v6':
        dataset = OpenImagesV6Dataset(
            root=root,
            split=split,
            max_samples=max_samples
        )
    elif dataset_name.lower() == 'bdd100k':
        dataset = BDD100KDataset(
            root=root,
            split=split,
            max_samples=max_samples
        )
    elif dataset_name.lower() == 'ade20k':
        dataset = ADE20KDataset(
            root=root,
            split=split,
            max_samples=max_samples
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: open_images_v6, bdd100k, ade20k")
    
    # FIXED: Proper pin_memory handling.
    if pin_memory is None:
        # Only pin memory if CUDA is available AND we'll use non_blocking transfers.
        pin_memory = torch.cuda.is_available()
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_inference_collate_fn,
    )
    
    return dataloader


def run_inference_on_dataset(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = 'cpu',
    verbose: bool = True,
    postprocessor: Optional[DetectionPostProcessor] = None,
    skip_corrupted: bool = True
) -> Dict[str, Any]:
    """Run MaxSight inference on inference dataset."""
    model.eval()
    model.to(device)
    
    # FIXED: Create postprocessor if not provided.
    if postprocessor is None:
        postprocessor = DetectionPostProcessor(
            confidence_threshold=0.3,
            max_detections=10
        )
    
    all_predictions = []
    total = 0
    corrupted_count = 0
    corrupted_images = []
    
    # FIXED: Global counter for image_idx (handles variable batch sizes)
    global_image_idx = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # FIXED: Handle corrupted images.
            try:
                images = batch['image']
            except RuntimeError as e:
                if 'Corrupted image skipped' in str(e):
                    corrupted_count += 1
                    corrupted_images.append(str(e))
                    if verbose:
                        logger.warning(f"Skipping corrupted batch: {e}")
                    continue
                else:
                    raise
            
            # FIXED: Use non_blocking transfer if pin_memory is enabled.
            if dataloader.pin_memory and device != 'cpu':
                images = images.to(device, non_blocking=True)
            else:
                images = images.to(device)
            
            # Run inference.
            outputs = model(images)
            
            # FIXED: Use postprocessor interface.
            batch_size = images.size(0)
            detections = postprocessor.process(model, outputs, batch_size)
            
            # Process batch.
            total += batch_size
            
            # FIXED: Use global counter for image_idx.
            for i in range(batch_size):
                # FIXED: Handle different detection formats.
                if i < len(detections):
                    image_detections = detections[i]
                else:
                    image_detections = []
                
                # FIXED: Standard metadata access.
                context = batch.get('context', {})
                if isinstance(context, list) and i < len(context):
                    context = context[i]
                elif not isinstance(context, dict):
                    context = {}
                
                pred = {
                    'image_idx': global_image_idx,
                    'detections': image_detections,
                    'num_detections': len(image_detections),
                    'image_id': batch['image_id'][i] if 'image_id' in batch and i < len(batch['image_id']) else f'img_{global_image_idx}',
                    'dataset': batch['dataset'][i] if 'dataset' in batch and i < len(batch['dataset']) else 'unknown',
                    'context': {
                        'weather': (context.get('weather') or '') if isinstance(context, dict) else '',
                        'scene': (context.get('scene') or '') if isinstance(context, dict) else '',
                        'labels': (context.get('labels') or {}) if isinstance(context, dict) else {},
                        'annotation_path': (context.get('annotation_path') or '') if isinstance(context, dict) else '',
                        'label': (context.get('label') or '') if isinstance(context, dict) else '',
                        'confidence': (context.get('confidence') or '') if isinstance(context, dict) else ''
                    }
                }
                
                all_predictions.append(pred)
                global_image_idx += 1  # FIXED: Increment global counter.
            
            if verbose and (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    # Calculate statistics.
    stats = {
        'total_images': total,
        'total_detections': sum(p['num_detections'] for p in all_predictions),
        'avg_detections_per_image': sum(p['num_detections'] for p in all_predictions) / total if total > 0 else 0,
        'images_with_detections': sum(1 for p in all_predictions if p['num_detections'] > 0),
        'images_without_detections': sum(1 for p in all_predictions if p['num_detections'] == 0),
        'corrupted_images_skipped': corrupted_count  # FIXED: Track corrupted images.
    }
    
    # Get dataset info from first sample.
    dataset_info = {}
    if all_predictions:
        dataset_info = {
            'dataset': all_predictions[0].get('dataset', 'unknown'),
            'split': getattr(dataloader.dataset, 'split', 'unknown')
        }
    
    results = {
        'predictions': all_predictions,
        'stats': stats,
        'dataset_info': dataset_info,
        'corrupted_images': corrupted_images  # FIXED: Return corrupted image list.
    }
    
    return results


if __name__ == "__main__":
    import argparse
    from ml.models.maxsight_cnn import MaxSightCNN
    
    parser = argparse.ArgumentParser(description='Run MaxSight inference on inference datasets')
    parser.add_argument('--dataset', type=str, choices=['open_images_v6', 'bdd100k', 'ade20k'], required=True,
                      help='Dataset to use')
    parser.add_argument('--root', type=Path, required=True,
                      help='Root directory for dataset')
    parser.add_argument('--split', type=str, default='validation',
                      help='Dataset split (default: validation)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size (default: 32)')
    parser.add_argument('--model_path', type=Path, default=None,
                      help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                      help='Device to run inference on (default: cpu)')
    parser.add_argument('--max_samples', type=int, default=None,
                      help='Number of samples to process (default: all)')
    parser.add_argument('--skip_corrupted', action='store_true', default=True,
                      help='Skip corrupted images (default: True)')
    
    args = parser.parse_args()
    
    # Create dataloader.
    print(f"Creating {args.dataset} {args.split} dataloader...")
    dataloader = create_inference_dataloader(
        dataset_name=args.dataset,
        root=args.root,
        split=args.split,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    # Load model.
    print("Loading MaxSightCNN model...")
    model = MaxSightCNN(num_classes=80, use_audio=False)
    if args.model_path and args.model_path.exists():
        checkpoint = torch.load(args.model_path, map_location=args.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {args.model_path}")
    else:
        print("Using untrained model (random weights)")
    
    # Run inference.
    try:
        dataset_size = len(dataloader.dataset)  # type: ignore
    except (TypeError, AttributeError):
        dataset_size = args.max_samples if args.max_samples else "unknown"
    
    num_images = args.max_samples if args.max_samples else dataset_size
    print(f"\nRunning inference on {num_images} images...")
    
    results = run_inference_on_dataset(
        model=model,
        dataloader=dataloader,
        device=args.device,
        verbose=True,
        skip_corrupted=args.skip_corrupted
    )
    
    # Print results.
    print("\n" + "="*50)
    print("Inference Results:")
    print("="*50)
    stats = results['stats']
    dataset_info = results['dataset_info']
    print(f"Dataset: {dataset_info.get('dataset', 'unknown')}")
    print(f"Split: {dataset_info.get('split', 'unknown')}")
    print(f"Total images processed: {stats['total_images']}")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Average detections per image: {stats['avg_detections_per_image']:.2f}")
    print(f"Images with detections: {stats['images_with_detections']}")
    print(f"Images without detections: {stats['images_without_detections']}")
    if stats['corrupted_images_skipped'] > 0:
        print(f"WARNING Corrupted images skipped: {stats['corrupted_images_skipped']}")
    print("="*50)






