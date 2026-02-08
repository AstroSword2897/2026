"""Generate MaxSight annotations from COCO: map categories, assign urgency, estimate distance."""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import random
from collections import defaultdict

from ml.models.maxsight_cnn import COCO_CLASSES, COCO_BASE_CLASSES


def map_coco_to_environmental(coco_category_name: str) -> str:
    """Map COCO category to environmental category in comprehensive class list."""
    normalized = coco_category_name.lower().strip()
    
    # Find exact match in comprehensive class list.
    for cls in COCO_CLASSES:
        if cls.lower() == normalized:
            return cls
    
    # Fallback if no match found.
    return COCO_CLASSES[0] if COCO_CLASSES else 'person'


def assign_urgency_score(category_name: str, box_size: float) -> int:
    """Assign urgency score based on object category and size."""
    category_lower = category_name.lower()
    
    # Use sets for O(1) lookup instead of O(K) keyword matching.
    danger_categories = {
        'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'vehicle', 'train', 'airplane',
        'fire', 'hazard', 'stop', 'traffic', 'traffic light', 'fire hydrant',
        'emergency', 'siren', 'alarm'
    }
    warning_categories = {
        'person', 'dog', 'cat', 'horse', 'bird', 'cow', 'sheep', 'elephant',
        'zebra', 'giraffe', 'bear'
    }
    caution_categories = {
        'stairs', 'staircase', 'escalator', 'elevator', 'door', 'window',
        'obstacle', 'barrier', 'fence'
    }
    
    # Direct category match (most efficient)
    if category_lower in danger_categories:
        return 3 if box_size > 0.1 else 2
    elif category_lower in warning_categories:
        return 2 if box_size > 0.15 else 1
    elif category_lower in caution_categories:
        return 1 if box_size > 0.1 else 0
    else:
        if any(kw in category_lower for kw in ['vehicle', 'motor', 'fire', 'hazard', 'emergency']):
            return 3 if box_size > 0.1 else 2
        elif any(kw in category_lower for kw in ['person', 'animal', 'pet']):
            return 2 if box_size > 0.15 else 1
        else:
            return 0  # Safe - no immediate threat.


def estimate_distance_zone(box_size: float, image_size: Tuple[int, int] = (224, 224)) -> int:
    """Estimate distance zone from bounding box size."""
    # Complexity: O(1) - three simple comparisons (if/elif/else)
    if box_size > 0.1:  # Large box = near (close to camera, occupies >10% of image)
        # Complexity: O(1) - simple return.
        return 0  # Near.
    elif box_size > 0.05:
        return 1  # Medium.
    else:
        return 2  # Far.


def generate_scene_description(objects: List[Dict]) -> str:
    """Generate scene description from detected objects with urgency and distance context. Enhanced to include urgency information and prioritize important objects."""
    if not objects:
        return "Empty scene"
    
    sorted_objects = sorted(objects, key=lambda x: x.get('urgency', 0), reverse=True)
    
    # Get top 5 most urgent objects for description.
    top_objects = sorted_objects[:5]
    categories = [obj.get('category', 'object') for obj in top_objects]
    unique_categories = list(dict.fromkeys(categories))  # Preserve order while removing duplicates.
    
    # Count objects by category for richer descriptions.
    category_counts = {}
    for obj in top_objects:
        cat = obj.get('category', 'object')
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    if len(unique_categories) == 1:
        cat = unique_categories[0]
        count = category_counts.get(cat, 1)
        if count > 1:
            return f"Scene with {count} {cat}s"
        return f"Scene with {cat}"
    elif len(unique_categories) <= 3:
        # Build description with counts.
        desc_parts = []
        for cat in unique_categories:
            count = category_counts.get(cat, 1)
            if count > 1:
                desc_parts.append(f"{count} {cat}s")
            else:
                desc_parts.append(cat)
        return f"Scene with {', '.join(desc_parts)}"
    else:
        # Many object types - summarize with count and mention most urgent.
        most_urgent = sorted_objects[0].get('category', 'objects') if sorted_objects else 'objects'
        return f"Scene with {len(unique_categories)} different objects including {most_urgent}"


def generate_annotations_from_coco(
    coco_annotation_file: Path,
    image_dir: Path,
    output_file: Path,
    num_samples: int = 6000,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15
) -> Tuple[Path, Path, Path]:
    """Generate MaxSight annotations from COCO dataset."""
    print(f"Loading COCO annotations from {coco_annotation_file}...")
    
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    image_map = {img['id']: img for img in coco_data['images']}
    category_map = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
    
    image_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        image_annotations[ann['image_id']].append(ann)
    
    maxsight_annotations = []
    image_ids = list(image_annotations.keys())
    
    if len(image_ids) > num_samples:
        image_ids = random.sample(image_ids, num_samples)
    
    print(f"Processing {len(image_ids)} images...")
    
    for image_id in image_ids:
        img_info = image_map[image_id]
        anns = image_annotations[image_id]
        
        objects = []
        scene_urgency = 0
        
        for ann in anns:
            category_id = ann['category_id']
            category_name = category_map.get(category_id, 'unknown')
            env_category = map_coco_to_environmental(category_name)
            
            bbox = ann['bbox']  # [x, y, w, h] in pixels.
            img_width = img_info.get('width', 224)
            img_height = img_info.get('height', 224)
            
            # Convert to center format and normalize.
            cx = (bbox[0] + bbox[2] / 2) / img_width
            cy = (bbox[1] + bbox[3] / 2) / img_height
            w = bbox[2] / img_width
            h = bbox[3] / img_height
            box_size = w * h
            
            urgency = assign_urgency_score(env_category, box_size)
            distance_zone = estimate_distance_zone(box_size)
            scene_urgency = max(scene_urgency, urgency)
            
            objects.append({
                'box': [cx, cy, w, h],  # Center format, normalized.
                'category': env_category,
                'urgency': urgency,
                'distance': distance_zone
            })
        
        # Generate scene description.
        scene_desc = generate_scene_description(objects)
        
        # Create MaxSight annotation.
        maxsight_ann = {
            'image_id': image_id,
            'image_path': img_info.get('file_name', f'{image_id}.jpg'),
            'objects': objects,
            'urgency': scene_urgency,
            'lighting': 'normal',  # Will be detected during training.
            'description': scene_desc,
            'audio_path': None  # No audio in COCO.
        }
        
        maxsight_annotations.append(maxsight_ann)
    
    random.shuffle(maxsight_annotations)
    
    # Validate splits sum to 1.0.
    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        raise ValueError(f"Splits must sum to 1.0, got {train_split + val_split + test_split}")
    
    # Calculate split indices.
    total = len(maxsight_annotations)
    train_end = int(total * train_split)
    val_end = train_end + int(total * val_split)
    
    train_annotations = maxsight_annotations[:train_end]
    val_annotations = maxsight_annotations[train_end:val_end]
    test_annotations = maxsight_annotations[val_end:]
    
    train_file = output_file.parent / f'{output_file.stem}_train.json'
    val_file = output_file.parent / f'{output_file.stem}_val.json'
    test_file = output_file.parent / f'{output_file.stem}_test.json'
    
    with open(train_file, 'w') as f:
        json.dump(train_annotations, f, indent=2)
    
    with open(val_file, 'w') as f:
        json.dump(val_annotations, f, indent=2)
    
    with open(test_file, 'w') as f:
        json.dump(test_annotations, f, indent=2)
    
    print(f"Generated {len(train_annotations)} training annotations ({len(train_annotations)/total*100:.1f}%)")
    print(f"Generated {len(val_annotations)} validation annotations ({len(val_annotations)/total*100:.1f}%)")
    print(f"Generated {len(test_annotations)} test annotations ({len(test_annotations)/total*100:.1f}%)")
    print(f"Saved to {train_file}, {val_file}, and {test_file}")
    
    return train_file, val_file, test_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate MaxSight annotations from COCO dataset')
    parser.add_argument('--coco_annotations', type=Path, required=True,
                      help='Path to COCO annotation JSON file')
    parser.add_argument('--image_dir', type=Path, required=True,
                      help='Directory containing COCO images')
    parser.add_argument('--output', type=Path, default=Path('annotations/maxsight_annotations.json'),
                      help='Output annotation file path (will create train/val versions)')
    parser.add_argument('--num_samples', type=int, default=6000,
                      help='Total number of samples to generate (default: 6000)')
    parser.add_argument('--train_split', type=float, default=0.83,
                      help='Fraction of samples for training (default: 0.83 for 5000/1000 split)')
    
    args = parser.parse_args()
    
    # Create output directory.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate annotations.
    train_file, val_file = generate_annotations_from_coco(
        coco_annotation_file=args.coco_annotations,
        image_dir=args.image_dir,
        output_file=args.output,
        num_samples=args.num_samples,
        train_split=args.train_split
    )
    
    print("\nAnnotation generation complete!")
    print(f"Training annotations: {train_file}")
    print(f"Validation annotations: {val_file}")







