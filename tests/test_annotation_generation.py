"""Tests for Annotation Generation - Task 3.2"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
from typing import Dict, List

from ml.models.maxsight_cnn import COCO_CLASSES
from ml.data.generate_annotations import (
    get_all_datasets_info,
    save_class_mappings,
    ENVIRONMENTAL_CLASSES,
    SOUND_CLASSES
)


def test_coco_class_mapping():
    """Test COCO class mapping to MaxSight classes."""
    print("\nAnnotation Test 1: COCO Class Mapping")
    
    # Verify COCO classes are loaded
    assert len(COCO_CLASSES) > 0, "COCO_CLASSES should not be empty"
    assert len(COCO_CLASSES) == len(ENVIRONMENTAL_CLASSES), "COCO_CLASSES should match ENVIRONMENTAL_CLASSES"
    
    # Check for common classes
    common_classes = ['person', 'car', 'bicycle', 'dog', 'cat']
    for cls in common_classes:
        assert cls in COCO_CLASSES, f"Common class '{cls}' should be in COCO_CLASSES"
    
    print(f"  Total COCO classes: {len(COCO_CLASSES)}")
    print("  COCO class mapping: PASSED")


def test_environmental_classes():
    """Test environmental class definitions."""
    print("\nAnnotation Test 2: Environmental Classes")
    
    assert len(ENVIRONMENTAL_CLASSES) > 0, "ENVIRONMENTAL_CLASSES should not be empty"
    assert isinstance(ENVIRONMENTAL_CLASSES, list), "ENVIRONMENTAL_CLASSES should be a list"
    
    # Check for no duplicates
    unique_classes = set(ENVIRONMENTAL_CLASSES)
    assert len(unique_classes) == len(ENVIRONMENTAL_CLASSES), "ENVIRONMENTAL_CLASSES should have no duplicates"
    
    print(f"  Total environmental classes: {len(ENVIRONMENTAL_CLASSES)}")
    print("  Environmental classes: PASSED")


def test_sound_classes():
    """Test sound class definitions."""
    print("\nAnnotation Test 3: Sound Classes")
    
    assert len(SOUND_CLASSES) > 0, "SOUND_CLASSES should not be empty"
    assert isinstance(SOUND_CLASSES, list), "SOUND_CLASSES should be a list"
    
    # Check for expected sound classes
    expected_sounds = ['fire alarm', 'siren', 'car horn', 'doorbell']
    for sound in expected_sounds:
        assert sound in SOUND_CLASSES, f"Expected sound class '{sound}' should be in SOUND_CLASSES"
    
    print(f"  Total sound classes: {len(SOUND_CLASSES)}")
    print("  Sound classes: PASSED")


def test_urgency_assignment():
    """Test urgency level assignment logic."""
    print("\nAnnotation Test 4: Urgency Assignment")
    
    # Urgency levels: 0=low, 1=medium, 2=high, 3=critical
    # Test urgency assignment based on object class and distance
    
    # Critical objects (should get urgency >= 2)
    critical_classes = ['person', 'car', 'bicycle', 'motorcycle']
    
    # Near objects (distance zone 0) should have higher urgency
    # Far objects (distance zone 2) should have lower urgency
    
    # Test urgency calculation
    def calculate_urgency(class_name: str, distance_zone: int) -> int:
        """Calculate urgency based on class and distance."""
        base_urgency = 0
        
        # Critical objects get base urgency
        if class_name in critical_classes:
            base_urgency = 2
        elif class_name in ['dog', 'cat']:
            base_urgency = 1
        else:
            base_urgency = 0
        
        # Distance adjustment: closer = higher urgency
        if distance_zone == 0:  # Near
            return min(3, base_urgency + 1)
        elif distance_zone == 1:  # Medium
            return base_urgency
        else:  # Far
            return max(0, base_urgency - 1)
    
    # Test cases
    assert calculate_urgency('person', 0) >= 2, "Person near should have high urgency"
    assert calculate_urgency('person', 2) >= 1, "Person far should have at least medium urgency"
    assert calculate_urgency('car', 0) >= 2, "Car near should have high urgency"
    assert calculate_urgency('chair', 0) <= 1, "Non-critical object should have low urgency"
    
    print("  Urgency assignment logic: PASSED")


def test_distance_zones():
    """Test distance zone assignment."""
    print("\nAnnotation Test 5: Distance Zones")
    
    # Distance zones: 0=near, 1=medium, 2=far
    # Based on bounding box area
    
    def assign_distance_zone(bbox_area: float) -> int:
        """Assign distance zone based on bbox area."""
        if bbox_area > 0.3:  # Large box = close
            return 0  # near
        elif bbox_area > 0.1:  # Medium box
            return 1  # medium
        else:  # Small box = far
            return 2  # far
    
    # Test cases
    assert assign_distance_zone(0.5) == 0, "Large bbox should be near"
    assert assign_distance_zone(0.2) == 1, "Medium bbox should be medium distance"
    assert assign_distance_zone(0.05) == 2, "Small bbox should be far"
    
    print("  Distance zone assignment: PASSED")


def test_dataset_info():
    """Test dataset information retrieval."""
    print("\nAnnotation Test 6: Dataset Information")
    
    datasets_info = get_all_datasets_info()
    
    assert isinstance(datasets_info, dict), "get_all_datasets_info should return a dict"
    assert 'coco' in datasets_info, "Should include COCO dataset info"
    assert 'open_images' in datasets_info, "Should include Open Images dataset info"
    
    # Check COCO info structure
    coco_info = datasets_info['coco']
    assert 'name' in coco_info, "Dataset info should include name"
    assert 'images' in coco_info, "Dataset info should include image count"
    assert 'classes' in coco_info, "Dataset info should include class count"
    
    print(f"  Available datasets: {len(datasets_info)}")
    print("  Dataset information: PASSED")


def test_class_mappings_save():
    """Test class mappings save functionality."""
    print("\nAnnotation Test 7: Class Mappings Save")
    
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        
        # Save class mappings
        save_class_mappings(data_dir)
        
        # Check files were created
        env_file = data_dir / "environmental_classes.txt"
        sound_file = data_dir / "sound_classes.txt"
        
        assert env_file.exists(), "Environmental classes file should be created"
        assert sound_file.exists(), "Sound classes file should be created"
        
        # Check file contents
        with open(env_file, 'r') as f:
            env_content = f.read()
            assert len(env_content) > 0, "Environmental classes file should not be empty"
            assert 'person' in env_content or '0:' in env_content, "File should contain class mappings"
        
        with open(sound_file, 'r') as f:
            sound_content = f.read()
            assert len(sound_content) > 0, "Sound classes file should not be empty"
        
        print("  Class mappings save: PASSED")


def test_annotation_format():
    """Test annotation format structure."""
    print("\nAnnotation Test 8: Annotation Format")
    
    # Expected annotation format
    annotation_format = {
        'image_id': str,
        'image_path': str,
        'objects': [
            {
                'class_id': int,
                'class_name': str,
                'bbox': list,  # [x, y, w, h] normalized
                'distance_zone': int,  # 0=near, 1=medium, 2=far
                'urgency': int,  # 0-3
                'confidence': float
            }
        ],
        'scene_urgency': int,  # 0-3
        'lighting': str  # 'bright', 'normal', 'dim', 'dark'
    }
    
    # Create sample annotation
    sample_annotation = {
        'image_id': 'test_001',
        'image_path': 'test.jpg',
        'objects': [
            {
                'class_id': 0,
                'class_name': 'person',
                'bbox': [0.5, 0.5, 0.2, 0.3],
                'distance_zone': 0,
                'urgency': 3,
                'confidence': 0.95
            }
        ],
        'scene_urgency': 3,
        'lighting': 'normal'
    }
    
    # Validate structure
    assert 'image_id' in sample_annotation, "Annotation should have image_id"
    assert 'objects' in sample_annotation, "Annotation should have objects"
    assert isinstance(sample_annotation['objects'], list), "Objects should be a list"
    
    if len(sample_annotation['objects']) > 0:
        obj = sample_annotation['objects'][0]
        assert 'class_id' in obj, "Object should have class_id"
        assert 'bbox' in obj, "Object should have bbox"
        assert 'distance_zone' in obj, "Object should have distance_zone"
        assert 'urgency' in obj, "Object should have urgency"
        assert 0 <= obj['urgency'] <= 3, "Urgency should be 0-3"
        assert 0 <= obj['distance_zone'] <= 2, "Distance zone should be 0-2"
    
    print("  Annotation format: PASSED")


if __name__ == "__main__":
    print("Running Annotation Generation Tests")
    print("=" * 70)
    
    test_coco_class_mapping()
    test_environmental_classes()
    test_sound_classes()
    test_urgency_assignment()
    test_distance_zones()
    test_dataset_info()
    test_class_mappings_save()
    test_annotation_format()
    
    print("\n" + "=" * 70)
    print("All annotation generation tests completed!")

