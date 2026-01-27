"""
MaxSight Web-Based Product Simulator
Complete end-to-end simulation of the MaxSight product on a local web server.

This simulator integrates ALL components:
- Model inference (MaxSightCNN)
- Preprocessing (condition-specific)
- OCR integration
- Output scheduling
- Therapy system
- Description generation
- Spatial memory
- Path planning
- Voice feedback
- Haptic feedback
- Visual overlays
- Session management
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import time
import json
import base64
from io import BytesIO
from PIL import Image
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Flask for web server
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

# Import ALL MaxSight components
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


app = Flask(__name__, 
            template_folder=Path(__file__).parent / 'templates',
            static_folder=Path(__file__).parent / 'static')
CORS(app)


class MaxSightSimulator:
    """
    Complete MaxSight product simulator integrating all components.
    """
    
    def __init__(self, device: Optional[str] = None):
        """Initialize all MaxSight components."""
        print("Initializing MaxSight Simulator...")
        
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
        
        print(f"  Device: {self.device}")
        
        # Initialize model
        print("  Loading model...")
        self.model = create_model()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize all components
        print("  Initializing components...")
        self.preprocessor = None  # Will be set per user condition
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
        
        # User state
        self.current_condition = None
        self.current_scenario = None
        self.session_active = False
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'avg_latency_ms': 0.0,
            'total_inference_time': 0.0
        }
        
    
    def set_user_condition(self, condition: str):
        """Set user's visual condition."""
        self.current_condition = condition
        self.preprocessor = ImagePreprocessor(condition_mode=condition)
        print(f"  Condition set to: {condition}")
    
    def process_frame(
        self,
        image: Image.Image,
        audio_features: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Process a single frame through the complete MaxSight pipeline.
        
        This integrates ALL components:
        1. Preprocessing (condition-specific)
        2. Model inference
        3. OCR text detection
        4. Description generation
        5. Spatial memory update
        6. Path planning
        7. Output scheduling
        8. Therapy integration
        9. Overlay generation
        10. Voice/haptic feedback
        """
        start_time = time.perf_counter()
        
        # 1. Preprocessing (condition-specific)
        import torchvision.transforms as T
        if self.preprocessor:
            preprocessed_tensor = self.preprocessor(image)  # ImagePreprocessor.__call__ returns tensor
            image_tensor = preprocessed_tensor.unsqueeze(0).to(self.device)
        else:
            # Convert PIL Image to tensor
            to_tensor = T.ToTensor()
            image_tensor = to_tensor(image).unsqueeze(0).to(self.device)
        
        # 2. Model inference
        inference_start = time.perf_counter()
        with torch.no_grad():
            if audio_features is not None:
                audio_tensor = torch.from_numpy(audio_features).unsqueeze(0).to(self.device)
                outputs = self.model(image_tensor, audio_tensor)
            else:
                outputs = self.model(image_tensor)
        inference_time = time.perf_counter() - inference_start
        
        # 3. Post-process detections
        detections = self.model.get_detections(outputs, confidence_threshold=0.3)
        detections_list = detections[0] if detections else []
        
        # 4. OCR text detection
        ocr_results = []
        try:
            # Get text scores and boxes from model outputs
            text_scores = outputs.get('text_regions', torch.zeros(1, 196))
            boxes = outputs.get('boxes', torch.zeros(1, 196, 4))
            ocr_results = self.ocr.process_image_for_ocr(
                image=image,
                text_scores=text_scores[0],
                boxes=boxes[0]
            )
        except Exception as e:
            print(f"  OCR error: {e}")
            ocr_results = []
        
        # 5. Description generation
        urgency_score = outputs.get('urgency_scores', torch.zeros(1, 4))
        urgency_level = int(urgency_score.argmax(dim=1).item()) if urgency_score.numel() > 0 else 0
        
        # Convert detections to format expected by generate_scene_description
        scene_detections = []
        for det in detections_list:
            if 'bbox' in det and 'class_name' in det:
                scene_detections.append({
                    'class_name': det.get('class_name', 'object'),
                    'box': torch.tensor(det.get('bbox', [0.5, 0.5, 0.1, 0.1]), dtype=torch.float32),
                    'distance': det.get('distance', 1),
                    'urgency': det.get('urgency', urgency_level),
                    'priority': det.get('confidence', 0.5) * 100
                })
        
        scene_description = self.description_gen.generate_scene_description(
            detections=scene_detections,
            urgency_score=urgency_level
        )
        
        # Add OCR text to description if available
        if ocr_results:
            ocr_texts = [r.get('text', '') for r in ocr_results if r.get('text')]
            if ocr_texts:
                scene_description += f" Text detected: {', '.join(ocr_texts[:3])}"
        
        # 6. Spatial memory update
        spatial_detections = []
        for det in detections_list:
            if 'bbox' in det and 'class_name' in det:
                spatial_detections.append({
                    'class_name': det['class_name'],
                    'bbox': det['bbox'],
                    'confidence': det.get('confidence', 0.0),
                    'distance': det.get('distance', 1)
                })
        if spatial_detections:
            self.spatial_memory.update(
                detections=spatial_detections,
                timestamp=time.time()
            )
        
        # 7. Path planning (if navigation scenario)
        path_info = None
        if self.current_scenario == 'navigation':
            path_info = self.path_planner.plan_path(
                detections=detections_list,
                target_direction='forward'
            )
        
        # 8. Output scheduling
        model_outputs = {
            'urgency_scores': outputs.get('urgency_scores', None),
            'uncertainty': outputs.get('uncertainty', None)
        }
        scheduled_outputs = self.scheduler.schedule_outputs(
            detections=detections_list,
            model_outputs=model_outputs,
            timestamp=time.time()
        )
        
        # 9. Therapy integration
        therapy_feedback = None
        if self.session_active and detections_list:
            # Create therapy task from detections
            target_objects = [det.get('class_name', 'object') for det in detections_list[:3]]
            therapy_feedback = self.therapy.create_attention_task(
                scene_description=scene_description or "Scene with objects",
                target_objects=target_objects,
                difficulty=0.5
            )
        
        # 10. Generate overlays (placeholder - return original image with overlay info)
        # In production, this would draw bounding boxes, labels, etc.
        # Store overlay info in result dict instead
        
        # 11. Generate voice feedback
        voice_announcements = []
        if scene_description:
            # Use speak_custom for scene description
            self.voice_feedback.speak_custom(scene_description, priority=0)
            voice_announcements.append(scene_description)
        
        # Add urgent alerts
        urgency_scores = outputs.get('urgency_scores', torch.zeros(1, 4))
        if urgency_scores.numel() > 0:
            urgency_level = int(urgency_scores.argmax(dim=1).item())
            if urgency_level >= 2:  # Warning or danger
                self.voice_feedback.speak_custom(f"Warning: High urgency detected", priority=urgency_level)
                voice_announcements.append(f"Warning: High urgency detected")
        
        # 12. Generate haptic feedback
        haptic_patterns = []
        urgency_scores = outputs.get('urgency_scores', torch.zeros(1, 4))
        if urgency_scores.numel() > 0:
            urgency_level = int(urgency_scores.argmax(dim=1).item())
            if urgency_level >= 2:  # Warning or danger
                self.haptic_feedback.trigger(
                    self.haptic_feedback.HapticPattern.LONG_PULSE,
                    intensity=0.7
                )
                haptic_patterns.append({'pattern': 'long_pulse', 'intensity': 0.7})
            elif len(detections_list) > 0:
                self.haptic_feedback.trigger(
                    self.haptic_feedback.HapticPattern.MICRO_PULSE,
                    intensity=0.3
                )
                haptic_patterns.append({'pattern': 'micro_pulse', 'intensity': 0.3})
        
        # Update statistics
        self.stats['frames_processed'] += 1
        self.stats['total_inference_time'] += inference_time
        self.stats['total_detections'] += len(detections_list)
        self.stats['avg_latency_ms'] = (self.stats['total_inference_time'] / 
                                        self.stats['frames_processed'] * 1000)
        
        total_time = time.perf_counter() - start_time
        
        # Compile complete result
        result = {
            'frame_number': self.stats['frames_processed'],
            'timestamp': time.time(),
            'processing_time_ms': total_time * 1000,
            'inference_time_ms': inference_time * 1000,
            
            # Model outputs
            'detections': detections_list,
            'num_detections': len(detections_list),
            'urgency_scores': outputs['urgency_scores'][0].cpu().tolist(),
            'distance_zones': outputs['distance_zones'][0].cpu().tolist(),
            'scene_embedding': outputs['scene_embedding'][0].cpu().tolist(),
            
            # OCR results
            'text_regions': ocr_results,
            'num_text_regions': len(ocr_results),
            
            # Generated content
            'scene_description': scene_description,
            'scheduled_outputs': scheduled_outputs,
            'voice_announcements': voice_announcements,
            'haptic_patterns': haptic_patterns,
            'path_info': path_info,
            'therapy_feedback': therapy_feedback,
            
            # Statistics
            'stats': self.stats.copy()
        }
        
        return result


# Global simulator instance
simulator = None


def init_simulator():
    """Initialize simulator on first use."""
    global simulator
    if simulator is None:
        simulator = MaxSightSimulator()
    return simulator



@app.route('/')
def index():
    """Main simulator interface."""
    return render_template('simulator.html')


@app.route('/api/init', methods=['POST'])
def api_init():
    """Initialize simulator with user settings."""
    data = request.json
    condition = data.get('condition', 'normal')
    scenario = data.get('scenario', 'general')
    
    sim = init_simulator()
    sim.set_user_condition(condition)
    sim.current_scenario = scenario
    sim.session_active = data.get('start_session', False)
    
    if sim.session_active:
        sim.session_manager.start_session()
    
    return jsonify({
        'status': 'initialized',
        'condition': condition,
        'scenario': scenario,
        'session_active': sim.session_active
    })


@app.route('/api/process', methods=['POST'])
def api_process():
    """Process image through complete pipeline."""
    sim = init_simulator()
    
    # Get image from request
    if 'image' in request.files:
        image_file = request.files['image']
        image = Image.open(image_file.stream).convert('RGB')
    elif 'image_data' in request.json:
        # Base64 encoded image
        image_data = request.json['image_data']
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
    else:
        return jsonify({'error': 'No image provided'}), 400
    
    # Get audio features if provided
    audio_features = None
    if 'audio_features' in request.json:
        audio_features = np.array(request.json['audio_features'])
    
    # Process frame
    result = sim.process_frame(image, audio_features)
    
    # Convert original image to base64 for display
    image_buffer = BytesIO()
    image.save(image_buffer, format='PNG')
    image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
    result['overlay_image'] = f"data:image/png;base64,{image_base64}"
    
    return jsonify(result)


@app.route('/api/scenarios', methods=['GET'])
def api_scenarios():
    """Get available test scenarios."""
    scenarios = [
        {
            'id': 'general',
            'name': 'General Environment',
            'description': 'Standard object detection and scene understanding'
        },
        {
            'id': 'navigation',
            'name': 'Navigation Assistance',
            'description': 'Path planning and obstacle avoidance'
        },
        {
            'id': 'text_reading',
            'name': 'Text Reading',
            'description': 'OCR and text-to-speech focus'
        },
        {
            'id': 'therapy',
            'name': 'Vision Therapy',
            'description': 'Therapy session with task generation'
        },
        {
            'id': 'safety',
            'name': 'Safety Alerts',
            'description': 'Urgency detection and hazard warnings'
        },
        {
            'id': 'accessibility',
            'name': 'Accessibility Features',
            'description': 'Condition-specific adaptations'
        }
    ]
    return jsonify({'scenarios': scenarios})


@app.route('/api/conditions', methods=['GET'])
def api_conditions():
    """Get available visual conditions."""
    conditions = [
        {'id': 'normal', 'name': 'Normal Vision'},
        {'id': 'myopia', 'name': 'Myopia'},
        {'id': 'hyperopia', 'name': 'Hyperopia'},
        {'id': 'astigmatism', 'name': 'Astigmatism'},
        {'id': 'cataracts', 'name': 'Cataracts'},
        {'id': 'glaucoma', 'name': 'Glaucoma'},
        {'id': 'amd', 'name': 'AMD (Age-Related Macular Degeneration)'},
        {'id': 'diabetic_retinopathy', 'name': 'Diabetic Retinopathy'},
        {'id': 'retinitis_pigmentosa', 'name': 'Retinitis Pigmentosa'},
        {'id': 'color_blindness', 'name': 'Color Blindness'},
        {'id': 'cvi', 'name': 'CVI (Cortical Visual Impairment)'},
        {'id': 'amblyopia', 'name': 'Amblyopia'},
        {'id': 'strabismus', 'name': 'Strabismus'}
    ]
    return jsonify({'conditions': conditions})


@app.route('/api/stats', methods=['GET'])
def api_stats():
    """Get current statistics."""
    sim = init_simulator()
    return jsonify(sim.stats)


@app.route('/api/session/start', methods=['POST'])
def api_session_start():
    """Start therapy session."""
    sim = init_simulator()
    sim.session_active = True
    sim.session_manager.start_session()
    return jsonify({'status': 'session_started'})


@app.route('/api/session/stop', methods=['POST'])
def api_session_stop():
    """Stop therapy session."""
    sim = init_simulator()
    sim.session_active = False
    session_summary = sim.session_manager.end_session()
    return jsonify({
        'status': 'session_stopped',
        'summary': session_summary
    })


@app.route('/api/session/status', methods=['GET'])
def api_session_status():
    """Get session status."""
    sim = init_simulator()
    return jsonify({
        'active': sim.session_active,
        'stats': sim.stats
    })


if __name__ == '__main__':
    print("MaxSight Product Simulator")
    print("\nStarting web server...")
    print("Access the simulator at: http://localhost:5001")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
