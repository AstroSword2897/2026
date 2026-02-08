"""COCO Dataset Splitter for MaxSight Creates train/test/validation splits from COCO dataset. Handles both COCO 2017 format and custom MaxSight annotations."""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np


def split_coco_dataset(
    coco_annotation_file: Path,
    output_dir: Path,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    min_objects_per_image: int = 1
) -> Tuple[Path, Path, Path]:
    """Split COCO dataset into train/val/test splits."""
    # Validate splits.
    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        raise ValueError(f"Splits must sum to 1.0, got {train_split + val_split + test_split}")
    
    # Set random seed.
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directory.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load COCO annotations.
    print(f"Loading COCO annotations from {coco_annotation_file}...")
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # Validate COCO format.
    if 'images' not in coco_data or 'annotations' not in coco_data:
        raise ValueError("Invalid COCO format: missing 'images' or 'annotations'")
    
    # Build mappings.
    image_map = {img['id']: img for img in coco_data['images']}
    category_map = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
    
    # Group annotations by image_id.
    image_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        image_annotations[ann['image_id']].append(ann)
    
    # Filter images with minimum objects.
    valid_image_ids = [
        img_id for img_id in image_map.keys()
        if len(image_annotations.get(img_id, [])) >= min_objects_per_image
    ]
    
    print(f"Found {len(valid_image_ids)} valid images (with >= {min_objects_per_image} objects)")
    
    valid_image_ids = list(np.random.permutation(valid_image_ids))
    
    # Calculate split indices.
    total = len(valid_image_ids)
    if total == 0:
        raise ValueError("No valid images found after filtering.")
    train_end = int(total * train_split)
    val_end = train_end + int(total * val_split)
    
    train_ids = valid_image_ids[:train_end]
    val_ids = valid_image_ids[train_end:val_end]
    test_ids = valid_image_ids[val_end:]
    
    print(f"Split sizes:")
    print(f"  Train: {len(train_ids)} ({len(train_ids)/total*100:.1f}%)")
    print(f"  Val:   {len(val_ids)} ({len(val_ids)/total*100:.1f}%)")
    print(f"  Test:  {len(test_ids)} ({len(test_ids)/total*100:.1f}%)")
    
    # Create split annotation files.
    def create_split_file(image_ids: List[int], split_name: str) -> Path:
        """Create annotation file for a split."""
        split_images = [image_map[img_id] for img_id in image_ids]
        split_annotations = []
        split_category_ids = set()
        
        for img_id in image_ids:
            for ann in image_annotations[img_id]:
                split_annotations.append(ann)
                split_category_ids.add(ann['category_id'])
        
        # Restrict categories to those present in the current split.
        split_categories = [cat for cat in coco_data.get('categories', [])
                            if cat['id'] in split_category_ids]
            
        split_data = {
            'info': coco_data.get('info', {}),
            'licenses': coco_data.get('licenses', []),
            'images': split_images,
            'annotations': split_annotations,
            'categories': split_categories
        }
        output_file = output_dir / f'instances_{split_name}2017.json'
        with open(output_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        print(f"Created {split_name} split: {output_file} ({len(split_images)} images, {len(split_annotations)} annotations)")
        return output_file
    
    # Create all splits.
    train_file = create_split_file(train_ids, 'train')
    val_file = create_split_file(val_ids, 'val')
    test_file = create_split_file(test_ids, 'test')
    
    return train_file, val_file, test_file


def create_maxsight_splits_from_coco(
    coco_annotation_file: Path,
    image_dir: Optional[Path],
    output_dir: Path,
    train_split: Optional[float] = None,
    val_split: Optional[float] = None,
    test_split: Optional[float] = None,
    train_samples: Optional[int] = None,
    val_samples: Optional[int] = None,
    seed: int = 42,
    num_samples: Optional[int] = None,
    min_objects_per_image: int = 1
) -> Tuple[Path, Path, Path]:
    """Create MaxSight-format train/val/test splits from COCO dataset."""
    from ml.data.generate_annotations import (
        map_coco_to_environmental,
        assign_urgency_score,
        estimate_distance_zone,
        generate_scene_description
    )
    
    # Validate: use either ratios OR absolute counts, not both.
    if train_samples is not None and val_samples is not None:
        # Using absolute counts - validate they're positive.
        if train_samples <= 0 or val_samples <= 0:
            raise ValueError(f"Sample counts must be positive: train={train_samples}, val={val_samples}")
        use_absolute_counts = True
    elif train_split is not None and val_split is not None:
        # Using ratios - validate they sum to 1.0.
        test_split = test_split or (1.0 - train_split - val_split)
        if abs(train_split + val_split + test_split - 1.0) > 1e-6:
            raise ValueError(f"Splits must sum to 1.0, got {train_split + val_split + test_split}")
        use_absolute_counts = False
    else:
        # Default to ratios if nothing specified.
        train_split = train_split or 0.7
        val_split = val_split or 0.15
        test_split = test_split or 0.15
        use_absolute_counts = False
    
    # Set random seed.
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directory.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load COCO annotations.
    print(f"Loading COCO annotations from {coco_annotation_file}...")
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    image_map = {img['id']: img for img in coco_data['images']}
    category_map = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
    
    image_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        image_annotations[ann['image_id']].append(ann)
    
    # Filter images with minimum objects.
    valid_image_ids = [
        img_id for img_id in image_annotations.keys()
        if len(image_annotations.get(img_id, [])) >= min_objects_per_image
    ]
    
    print(f"Found {len(valid_image_ids)} valid images (with >= {min_objects_per_image} objects)")
    
    # Limit samples if specified.
    if num_samples and len(valid_image_ids) > num_samples:
        valid_image_ids = random.sample(valid_image_ids, num_samples)
        print(f"Limited to {num_samples} samples")
    
    image_ids = valid_image_ids
    print(f"Processing {len(image_ids)} images...")
    
    # Convert to MaxSight format.
    maxsight_annotations = []
    for image_id in image_ids:
        img_info = image_map.get(image_id)
        if img_info is None:
            continue
        
        anns = image_annotations[image_id]
        
        objects = []
        scene_urgency = 0
        
        for ann in anns:
            category_id = ann['category_id']
            category_name = category_map.get(category_id, 'unknown')
            env_category = map_coco_to_environmental(category_name)
            
            bbox = ann['bbox']  # [x, y, w, h] in pixels.
            img_width = max(1, img_info.get('width', 224))
            img_height = max(1, img_info.get('height', 224))

            # Convert to center format and normalize.
            cx = (bbox[0] + bbox[2] / 2) / img_width
            cy = (bbox[1] + bbox[3] / 2) / img_height
            w = max(0.01, bbox[2] / img_width)
            h = max(0.01, bbox[3] / img_height)
            box_size = w * h
            
            urgency = assign_urgency_score(env_category, box_size)
            distance_zone = estimate_distance_zone(box_size)
            scene_urgency = max(scene_urgency, urgency)
            
            objects.append({
                'box': [cx, cy, w, h],
                'category': env_category,
                'urgency': urgency,
                'distance': distance_zone,
                'confidence': ann.get('score', 1.0)
            })
        
        # Skip images with zero objects.
        if not objects:
            continue
        
        # Generate scene description.
        scene_desc = generate_scene_description(objects)
        
        # Handle image path (check if image_dir is None)
        if image_dir is not None:
            image_path = str(image_dir / img_info.get('file_name', f'{image_id}.jpg'))
        else:
            image_path = img_info.get('file_name', f'{image_id}.jpg')
        
        # Create MaxSight annotation.
        maxsight_ann = {
            'image_id': image_id,
            'image_path': image_path,
            'objects': objects,
            'urgency': scene_urgency,
            'lighting': 'normal',  # Will be detected during training.
            'description': scene_desc,
            'audio_path': None  # No audio in COCO.
        }
        
        maxsight_annotations.append(maxsight_ann)
    
    maxsight_annotations = list(np.random.permutation(maxsight_annotations))
    
    # Calculate split indices.
    total = len(maxsight_annotations)
    
    if use_absolute_counts:
        # Use absolute sample counts.
        train_end = min(train_samples, total)
        val_end = min(train_end + val_samples, total)
        
        train_annotations = maxsight_annotations[:train_end]
        val_annotations = maxsight_annotations[train_end:val_end]
        test_annotations = maxsight_annotations[val_end:]
        
        print(f"\nSplit sizes (absolute counts):")
        print(f"  Train: {len(train_annotations):,} samples (requested: {train_samples:,})")
        print(f"  Val:   {len(val_annotations):,} samples (requested: {val_samples:,})")
        print(f"  Test:  {len(test_annotations):,} samples (remaining)")
    else:
        # Use ratios.
        train_end = int(total * train_split)
        val_end = train_end + int(total * val_split)
        
        train_annotations = maxsight_annotations[:train_end]
        val_annotations = maxsight_annotations[train_end:val_end]
        test_annotations = maxsight_annotations[val_end:]
        
        print(f"\nSplit sizes (ratios):")
        print(f"  Train: {len(train_annotations):,} ({len(train_annotations)/total*100:.1f}%)")
        print(f"  Val:   {len(val_annotations):,} ({len(val_annotations)/total*100:.1f}%)")
        print(f"  Test:  {len(test_annotations):,} ({len(test_annotations)/total*100:.1f}%)")
    
    # Save split files.
    def save_split(annotations: List[Dict], split_name: str) -> Path:
        output_file = output_dir / f'maxsight_{split_name}.json'
        with open(output_file, 'w') as f:
            json.dump(annotations, f, indent=2)
        print(f"Saved {split_name} split: {output_file}")
        return output_file
    
    train_file = save_split(train_annotations, 'train')
    val_file = save_split(val_annotations, 'val')
    test_file = save_split(test_annotations, 'test')
    
    return train_file, val_file, test_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Split COCO dataset into train/val/test')
    parser.add_argument('--coco_annotations', type=Path, required=True,
                      help='Path to COCO annotation JSON file')
    parser.add_argument('--image_dir', type=Path, required=False,
                      help='Directory containing COCO images for the input')
    parser.add_argument('--output_dir', type=Path, required=True,
                      help='Output directory for split annotation files')
    parser.add_argument('--train_split', type=float, default=0.7,
                      help='Fraction for training (default: 0.7)')
    parser.add_argument('--val_split', type=float, default=0.15,
                      help='Fraction for validation (default: 0.15)')
    parser.add_argument('--test_split', type=float, default=0.15,
                      help='Fraction for testing (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed (default: 42)')
    parser.add_argument('--num_samples', type=int, default=None,
                      help='Optional limit on total samples')
    parser.add_argument('--format', type=str, choices=['coco', 'maxsight'], default='maxsight',
                      help='Output format: coco (original) or maxsight (converted)')
    
    args = parser.parse_args()
    
    # Validate image_dir requirement for MaxSight format.
    if args.format == 'maxsight' and args.image_dir is None:
        parser.error("--image_dir is required for MaxSight format")
    
    if args.format == 'coco':
        train_file, val_file, test_file = split_coco_dataset(
            coco_annotation_file=args.coco_annotations,
            output_dir=args.output_dir,
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=args.test_split,
            seed=args.seed
        )
    else:
        train_file, val_file, test_file = create_maxsight_splits_from_coco(
            coco_annotation_file=args.coco_annotations,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            train_split=args.train_split if args.train_samples is None else None,
            val_split=args.val_split if args.val_samples is None else None,
            test_split=args.test_split if args.train_samples is None else None,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            seed=args.seed,
            num_samples=args.num_samples,
            min_objects_per_image=1  # Default: require at least 1 object per image.
        )
    
    print("\nOK Dataset splitting complete!")
    print(f"Train: {train_file}")
    print(f"Val:   {val_file}")
    print(f"Test:  {test_file}")







