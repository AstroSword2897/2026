# Generate MaxSight annotations from COCO: map categories, assign urgency, estimate distance.

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import random
from collections import defaultdict

from ml.models.maxsight_cnn import COCO_CLASSES, COCO_BASE_CLASSES


def map_coco_to_environmental(coco_category_name: str) -> str:
    normalized = coco_category_name.lower().strip()
    
    # Find exact match in comprehensive class list
    for cls in COCO_CLASSES:
        if cls.lower() == normalized:
            return cls
    
    # Fallback if no match found
    return COCO_CLASSES[0] if COCO_CLASSES else 'person'


def assign_urgency_score(category_name: str, box_size: float) -> int:
    category_lower = category_name.lower()
    
    danger_keywords = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'vehicle',
                      'fire', 'hazard', 'stop', 'traffic', 'train']
    warning_keywords = ['person', 'dog', 'cat', 'horse', 'bird']
    
    if any(kw in category_lower for kw in danger_keywords):
        if box_size > 0.1:
            return 3  # Danger
        else:
            return 2  # Warning
    elif any(kw in category_lower for kw in warning_keywords):
        if box_size > 0.15:
            return 2  # Warning
        else:
            return 1  # Caution - moving object farther away
    else:
        return 0  # Safe - no immediate threat


def estimate_distance_zone(box_size: float, image_size: Tuple[int, int] = (224, 224)) -> int:
    # Distance estimation based on normalized box area (larger boxes = closer objects)
    # Purpose: Estimate distance zone from bounding box size using heuristic: larger boxes indicate
    #          objects closer to camera, smaller boxes indicate objects farther away. This is a
    #          simple but effective heuristic for accessibility applications where precise distance
    #          isn't critical, but relative proximity is important for navigation guidance.
    # Complexity: O(1) - three simple comparisons (if/elif/else)
    # Relationship: Core distance estimation logic - used during annotation generation to assign
    #              distance zones (0=near, 1=medium, 2=far) to objects for spatial awareness features
    if box_size > 0.1:
        # Purpose: Identify objects very close to camera - these require immediate attention
        return 0  # Near
    elif box_size > 0.05:
        return 1  # Medium
    else:
        return 2  # Far


def generate_scene_description(objects: List[Dict]) -> str:
    # Generate simple scene description from detected objects.
    if not objects:
        return "Empty scene"
    
    categories = [obj.get('category', 'object') for obj in objects[:5]]
    unique_categories = list(set(categories))
    if len(unique_categories) == 1:
        return f"Scene with {unique_categories[0]}"
    elif len(unique_categories) <= 3:
        return f"Scene with {', '.join(unique_categories)}"
    else:
        return f"Scene with {len(unique_categories)} different objects"


def generate_annotations_from_coco(
    coco_annotation_file: Path,
    image_dir: Path,
    output_file: Path,
    num_samples: int = 6000,
    train_split: float = 0.83  # 5000/6000 = 0.83
) -> Tuple[Path, Path]:
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
            
            bbox = ann['bbox']  # [x, y, w, h] in pixels
            img_width = img_info.get('width', 224)
            img_height = img_info.get('height', 224)
            
            # Convert to center format and normalize
            cx = (bbox[0] + bbox[2] / 2) / img_width
            cy = (bbox[1] + bbox[3] / 2) / img_height
            w = bbox[2] / img_width
            h = bbox[3] / img_height
            box_size = w * h
            
            urgency = assign_urgency_score(env_category, box_size)
            distance_zone = estimate_distance_zone(box_size)
            scene_urgency = max(scene_urgency, urgency)
            
            objects.append({
                'box': [cx, cy, w, h],  # Center format, normalized
                'category': env_category,
                'urgency': urgency,
                'distance': distance_zone
            })
        
        # Generate scene description
        scene_desc = generate_scene_description(objects)
        
        # Create MaxSight annotation
        maxsight_ann = {
            'image_id': image_id,
            'image_path': img_info.get('file_name', f'{image_id}.jpg'),
            'objects': objects,
            'urgency': scene_urgency,
            'lighting': 'normal',  # Will be detected during training
            'description': scene_desc,
            'audio_path': None  # No audio in COCO
        }
        
        maxsight_annotations.append(maxsight_ann)
    
    random.shuffle(maxsight_annotations)
    split_idx = int(len(maxsight_annotations) * train_split)
    train_annotations = maxsight_annotations[:split_idx]
    val_annotations = maxsight_annotations[split_idx:]
    
    train_file = output_file.parent / f'{output_file.stem}_train.json'
    val_file = output_file.parent / f'{output_file.stem}_val.json'
    
    with open(train_file, 'w') as f:
        json.dump(train_annotations, f, indent=2)
    
    with open(val_file, 'w') as f:
        json.dump(val_annotations, f, indent=2)
    
    print(f"Generated {len(train_annotations)} training annotations")
    print(f"Generated {len(val_annotations)} validation annotations")
    print(f"Saved to {train_file} and {val_file}")
    
    return train_file, val_file


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
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate annotations
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

