"""
Dataset Image Testing Script
Processes images from datasets through the MaxSight simulator and runs validation tests.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import time
from PIL import Image
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.models.maxsight_cnn import create_model
from ml.utils.preprocessing import ImagePreprocessor
from ml.utils.output_scheduler import CrossModalScheduler, OutputConfig
from ml.utils.ocr_integration import OCRIntegration
from ml.utils.description_generator import DescriptionGenerator
from ml.utils.spatial_memory import SpatialMemory
from ml.utils.path_planning import PathPlanner
from ml.therapy.session_manager import SessionManager
from ml.therapy.task_generator import TaskGenerator
from ml.therapy.therapy_integration import TherapyTaskIntegrator
from app.overlays.overlay_engine import OverlayEngine
from app.ui.voice_feedback import VoiceFeedback
from app.ui.haptic_feedback import HapticFeedback


class DatasetImageTester:
    """
    Tests dataset images through the complete MaxSight pipeline.
    """
    
    def __init__(self, device: Optional[str] = None):
        print("Initializing Dataset Image Tester")
        
        # Device setup
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Device: {self.device}")
        
        # Initialize model
        print("Loading model...")
        self.model = create_model()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize all components
        print("Initializing components...")
        self.preprocessor = ImagePreprocessor()
        self.scheduler = CrossModalScheduler(OutputConfig())
        self.ocr = OCRIntegration()
        self.description_gen = DescriptionGenerator()
        self.spatial_memory = SpatialMemory()
        self.path_planner = PathPlanner()
        self.session_manager = SessionManager()
        self.task_generator = TaskGenerator()
        self.therapy = TherapyTaskIntegrator()
        self.overlay_engine = OverlayEngine()
        self.voice_feedback = VoiceFeedback()
        self.haptic_feedback = HapticFeedback()
        
        # Test results
        self.test_results = []
        
        print("✅ Tester initialized!\n")
    
    def find_dataset_images(
        self,
        dataset_dirs: List[Path],
        max_images: int = 50
    ) -> List[Path]:
        """
        Find images in dataset directories.
        
        Arguments:
            dataset_dirs: List of dataset directory paths
            max_images: Maximum number of images to process
        
        Returns:
            List of image file paths
        """
        image_paths = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        for dataset_dir in dataset_dirs:
            if not dataset_dir.exists():
                print(f"⚠️  Dataset directory not found: {dataset_dir}")
                continue
            
            # Search for images
            for ext in image_extensions:
                # Check common COCO structure
                for subdir in ['train2017', 'val2017', 'test2017', 'images']:
                    img_dir = dataset_dir / subdir
                    if img_dir.exists():
                        found = list(img_dir.glob(f"*{ext}"))[:max_images]
                        image_paths.extend(found)
                        if found:
                            print(f"  Found {len(found)} images in {img_dir}")
            
            # Also check root directory
            for ext in image_extensions:
                found = list(dataset_dir.glob(f"*{ext}"))[:max_images]
                image_paths.extend(found)
        
        # Remove duplicates and limit
        image_paths = list(set(image_paths))[:max_images]
        print(f"\n✅ Found {len(image_paths)} unique images to test\n")
        
        return image_paths
    
    def process_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Process a single image through the complete MaxSight pipeline.
        
        Arguments:
            image_path: Path to image file
        
        Returns:
            Dictionary with processing results and metrics
        """
        start_time = time.perf_counter()
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            
            # Preprocess (ImagePreprocessor.__call__ returns a tensor)
            image_tensor = self.preprocessor(image).unsqueeze(0).to(self.device)
            
            # Model inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
            
            # Extract detections
            detections = self._extract_detections(outputs, confidence_threshold=0.3)
            
            # OCR processing (needs text_scores and boxes from model)
            text_scores = outputs.get('text_regions', torch.zeros(1, 196))
            boxes = outputs.get('boxes', torch.zeros(1, 196, 4))
            ocr_results = self.ocr.process_image_for_ocr(
                image=image,
                text_scores=text_scores[0],  # [196] tensor
                boxes=boxes[0]  # [196, 4] tensor
            )
            
            # Description generation
            urgency_score = outputs.get('urgency_scores', torch.zeros(1, 4))
            urgency_level = urgency_score.argmax(dim=1).item() if urgency_score.numel() > 0 else 0
            
            # Convert detections to format expected by generate_scene_description
            scene_detections: List[Dict[str, Any]] = []
            for det in detections:
                scene_detections.append({
                    'class_name': f"class_{det.get('class_idx', 0)}",
                    'box': torch.tensor(det.get('box', [0.5, 0.5, 0.1, 0.1]), dtype=torch.float32),
                    'distance': 1,  # Default to medium
                    'urgency': urgency_level,
                    'priority': float(det.get('confidence', 0.5) * 100)
                })
            
            description = self.description_gen.generate_scene_description(
                detections=scene_detections,
                urgency_score=urgency_level
            )
            
            # Add OCR text to description if available
            if ocr_results:
                ocr_texts = [r.get('text', '') for r in ocr_results if r.get('text')]
                if ocr_texts:
                    description += f" Text detected: {', '.join(ocr_texts[:3])}"
            
            # Spatial memory update
            self.spatial_memory.update(detections)
            spatial_context = self.spatial_memory.get_spatial_summary()
            
            # Path planning
            path_info = self.path_planner.plan_path(
                detections=detections,
                target_direction='forward'
            )
            
            # Output scheduling
            model_outputs = {
                'urgency_scores': outputs.get('urgency_scores', None),
                'uncertainty': outputs.get('uncertainty', None)
            }
            scheduled_outputs = self.scheduler.schedule_outputs(
                detections=detections,
                model_outputs=model_outputs,
                timestamp=time.time()
            )
            
            # Therapy integration (create a task from detections)
            if detections:
                target_objects = [d.get('class_idx', 0) for d in detections[:3]]  # First 3 detections
                therapy_feedback = self.therapy.create_attention_task(
                    scene_description=description or "Scene with objects",
                    target_objects=[str(obj) for obj in target_objects],
                    difficulty=0.5
                )
            else:
                therapy_feedback = None
            
            # Overlay generation
            overlay_info = self.overlay_engine.add_halo(
                center=(0.5, 0.5),
                radius=50.0,
                intensity=0.3
            )
            
            # Voice feedback
            if description:
                self.voice_feedback.speak_custom(description)
            
            # Haptic feedback
            if scheduled_outputs:
                self.haptic_feedback.trigger(
                    self.haptic_feedback.HapticPattern.MICRO_PULSE,
                    intensity=0.5
                )
            
            # Calculate metrics
            processing_time = (time.perf_counter() - start_time) * 1000  # ms
            num_detections = len(detections) if detections else 0
            num_ocr_texts = len(ocr_results) if ocr_results else 0
            
            result = {
                'image_path': str(image_path),
                'success': True,
                'processing_time_ms': processing_time,
                'num_detections': num_detections,
                'num_ocr_texts': num_ocr_texts,
                'has_description': bool(description),
                'has_spatial_context': bool(spatial_context),
                'has_path_info': bool(path_info),
                'num_scheduled_outputs': len(scheduled_outputs) if scheduled_outputs else 0,
                'urgency_level': outputs.get('urgency_scores', torch.zeros(1, 4)).argmax(dim=1).item() if 'urgency_scores' in outputs else 0,
                'uncertainty': outputs.get('uncertainty', torch.zeros(1, 1)).mean().item() if 'uncertainty' in outputs else 0.0,
                'original_size': original_size,
                'error': None
            }
            
            return result
            
        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            return {
                'image_path': str(image_path),
                'success': False,
                'processing_time_ms': processing_time,
                'error': str(e),
                'num_detections': 0,
                'num_ocr_texts': 0,
                'has_description': False,
                'has_spatial_context': False,
                'has_path_info': False,
                'num_scheduled_outputs': 0
            }
    
    def _extract_detections(
        self,
        outputs: Dict[str, torch.Tensor],
        confidence_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Extract detections from model outputs."""
        detections = []
        
        if 'classifications' not in outputs or 'boxes' not in outputs or 'objectness' not in outputs:
            return detections
        
        classifications = outputs['classifications']  # [B, 196, 80]
        boxes = outputs['boxes']  # [B, 196, 4]
        objectness = outputs['objectness']  # [B, 196]
        
        # Get batch 0
        cls_logits = classifications[0]  # [196, 80]
        boxes_batch = boxes[0]  # [196, 4]
        obj_scores = objectness[0]  # [196]
        
        # Apply softmax to classifications
        cls_probs = torch.softmax(cls_logits, dim=1)  # [196, 80]
        
        # Find detections above threshold
        for i in range(cls_probs.shape[0]):
            obj_score = obj_scores[i].item()
            if obj_score < confidence_threshold:
                continue
            
            # Get class with highest probability
            class_probs = cls_probs[i]
            class_idx = int(class_probs.argmax().item())
            class_prob = float(class_probs[class_idx].item())
            
            if class_prob < confidence_threshold:
                continue
            
            # Get box coordinates
            box = boxes_batch[i].cpu().numpy()
            
            detections.append({
                'class_idx': class_idx,
                'class_prob': class_prob,
                'objectness': obj_score,
                'box': box.tolist(),
                'confidence': obj_score * class_prob
            })
        
        return detections
    
    def run_tests(
        self,
        image_paths: List[Path],
        conditions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run tests on multiple images.
        
        Arguments:
            image_paths: List of image paths to test
            conditions: Optional list of visual conditions to test
        
        Returns:
            Test summary dictionary
        """
        # Convert conditions to List[Optional[str]]
        if conditions is not None:
            test_conditions: List[Optional[str]] = list(conditions)  # List[str] -> List[Optional[str]]
        else:
            test_conditions = [None]
        
        print("Running Dataset Image Tests")
        print(f"Images to test: {len(image_paths)}")
        print(f"Conditions to test: {len(test_conditions)}\n")
        
        all_results = []
        
        for condition in test_conditions:
            if condition:
                print(f"\nTesting with condition: {condition}")
                self.preprocessor = ImagePreprocessor(condition_mode=condition)
            
            for i, image_path in enumerate(image_paths, 1):
                print(f"[{i}/{len(image_paths)}] Processing: {image_path.name}...", end=" ")
                
                result = self.process_image(image_path)
                result['condition'] = condition
                all_results.append(result)
                
                if result['success']:
                    print(f"✅ ({result['processing_time_ms']:.1f}ms, {result['num_detections']} detections)")
                else:
                    print(f"❌ Error: {result.get('error', 'Unknown')}")
        
        # Calculate statistics
        successful = [r for r in all_results if r['success']]
        failed = [r for r in all_results if not r['success']]
        
        if successful:
            avg_time = np.mean([r['processing_time_ms'] for r in successful])
            avg_detections = np.mean([r['num_detections'] for r in successful])
            avg_ocr = np.mean([r['num_ocr_texts'] for r in successful])
            total_detections = sum([r['num_detections'] for r in successful])
        else:
            avg_time = 0.0
            avg_detections = 0.0
            avg_ocr = 0.0
            total_detections = 0
        
        summary = {
            'total_images': len(image_paths),
            'total_tests': len(all_results),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(all_results) if all_results else 0.0,
            'avg_processing_time_ms': avg_time,
            'avg_detections_per_image': avg_detections,
            'avg_ocr_texts_per_image': avg_ocr,
            'total_detections': total_detections,
            'results': all_results
        }
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        print("Test Summary")
        print(f"Total Images Tested: {summary['total_images']}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful: {summary['successful']} ({summary['success_rate']*100:.1f}%)")
        print(f"Failed: {summary['failed']}")
        print(f"\nPerformance Metrics:")
        print(f"  Average Processing Time: {summary['avg_processing_time_ms']:.2f} ms")
        print(f"  Average Detections per Image: {summary['avg_detections_per_image']:.2f}")
        print(f"  Average OCR Texts per Image: {summary['avg_ocr_texts_per_image']:.2f}")
        print(f"  Total Detections: {summary['total_detections']}")
    
    def save_results(self, summary: Dict[str, Any], output_file: Path):
        """Save test results to JSON file."""
        # Remove full results for file size (keep summary)
        summary_to_save = {k: v for k, v in summary.items() if k != 'results'}
        summary_to_save['sample_results'] = summary['results'][:10]  # Save first 10 as samples
        
        with open(output_file, 'w') as f:
            json.dump(summary_to_save, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")


def create_test_images(output_dir: Path, num_images: int = 10) -> List[Path]:
    """Create synthetic test images if no dataset images are available."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    
    print(f"Creating {num_images} synthetic test images...")
    for i in range(num_images):
        # Create a simple test image
        img = Image.new('RGB', (224, 224), color=(i * 25 % 255, (i * 50) % 255, (i * 75) % 255))
        # Add some random noise/patterns
        import random
        pixels = img.load()
        if pixels is not None:  # Type guard
            for x in range(224):
                for y in range(224):
                    if random.random() < 0.1:  # 10% noise
                        pixels[x, y] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        img_path = output_dir / f"test_image_{i:03d}.png"
        img.save(img_path)
        image_paths.append(img_path)
    
    print(f" Created {len(image_paths)} test images in {output_dir}\n")
    return image_paths


def main():
    """Main function to run dataset image tests."""
    # Default dataset directories to check
    project_root = Path(__file__).parent.parent.parent
    dataset_dirs = [
        project_root / 'datasets' / 'coco',
        project_root / 'datasets',
        project_root / 'datasets' / 'open_images',
        project_root / 'datasets' / 'objects365',
    ]
    
    # Initialize tester
    tester = DatasetImageTester()
    
    # Find images
    image_paths = tester.find_dataset_images(dataset_dirs, max_images=50)
    
    # If no dataset images found, create synthetic test images
    if not image_paths:
        print("Creating synthetic test images for testing...\n")
        test_images_dir = project_root / 'test_images'
        image_paths = create_test_images(test_images_dir, num_images=20)
    
    # Run tests
    summary = tester.run_tests(image_paths)
    
    # Print summary
    tester.print_summary(summary)
    
    # Save results
    output_file = project_root / 'test_results' / 'dataset_image_tests.json'
    output_file.parent.mkdir(exist_ok=True)
    tester.save_results(summary, output_file)

