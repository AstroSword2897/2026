"""Simulation Harness."""

from typing import Dict, List, Optional, Any
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class TherapySimulator:
    """End-to-end simulation harness for therapy system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, model=None):
        """Initialize simulator. Arguments: config: Optional configuration dictionary model: Optional MaxSightCNN model instance (if None, will need to be set later)"""
        self.config = config or {}
        self.model = model
        self.is_running = False
        self.frame_count = 0
        self.logs = []
        self.processing_times = []  # Track processing times for summary.
    
    def start_simulation(self, video_source: Optional[str] = None):
        """Start simulation."""
        self.is_running = True
        self.frame_count = 0
        self.logs = []
        self.processing_times = []
        
        if video_source:
            logger.warning(
                "Video source parameter provided but not implemented. "
                "Use ComprehensiveSimulator for video support."
            )
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a single frame."""
        if not self.is_running:
            logger.warning("Simulation not started. Call start_simulation() first.")
            return {}
        
        start_time = time.perf_counter()
        self.frame_count += 1
        
        # Run model inference if model is available.
        model_output = {}
        if self.model is not None:
            try:
                import torch
                from PIL import Image
                
                # Convert numpy array to PIL Image.
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                pil_image = Image.fromarray(frame)
                
                # Convert to tensor (simplified; proper preprocessing not used here)
                import torchvision.transforms as T
                transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                image_tensor = transform(pil_image).unsqueeze(0)
                
                # Run inference.
                with torch.no_grad():
                    outputs = self.model(image_tensor)
                    model_output = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v 
                                   for k, v in outputs.items()}
            except Exception as e:
                logger.error(f"Model inference failed: {e}")
                model_output = {'error': str(e)}
        else:
            logger.debug("No model available - skipping inference")
        
        # Generate overlays (simplified; overlay_engine not used here)
        overlays = []
        if model_output and 'boxes' in model_output:
            # Basic overlay generation.
            overlays = [{'type': 'detection', 'data': model_output.get('boxes', [])}]
        
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        self.processing_times.append(processing_time_ms)
        
        result = {
            'frame_number': self.frame_count,
            'timestamp': self.frame_count / 30.0,  # Assuming 30 FPS.
            'model_output': model_output,
            'overlays': overlays,
            'user_input': None,  # User input handling not implemented.
            'processing_time_ms': processing_time_ms
        }
        
        self.logs.append(result)
        return result
    
    def stop_simulation(self) -> Dict[str, Any]:
        """Stop simulation and return summary."""
        self.is_running = False
        
        return {
            'total_frames': self.frame_count,
            'logs': self.logs,
            'summary': self._generate_summary()
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate simulation summary with statistics."""
        errors = []
        for log in self.logs:
            if 'error' in log.get('model_output', {}):
                errors.append({
                    'frame': log['frame_number'],
                    'error': log['model_output']['error']
                })
        
        avg_time = 0.0
        min_time = 0.0
        max_time = 0.0
        
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            min_time = min(self.processing_times)
            max_time = max(self.processing_times)
        
        return {
            'frames_processed': self.frame_count,
            'avg_processing_time_ms': avg_time,
            'min_processing_time_ms': min_time,
            'max_processing_time_ms': max_time,
            'errors': errors
        }
    
    def save_logs(self, filepath: str):
        """Save simulation logs to file. Arguments: filepath: Path to save JSON log file."""
        import json
        with open(filepath, 'w') as f:
            json.dump({
                'config': self.config,
                'logs': self.logs,
                'summary': self._generate_summary()
            }, f, indent=2)







