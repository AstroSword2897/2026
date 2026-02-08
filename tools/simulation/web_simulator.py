"""MaxSight web simulator - local web server for end-to-end testing. Run: python tools/simulation/web_simulator.py Access: http://localhost:8002 Note: Development mode only - not production-hardened."""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import time
import json
import base64
from io import BytesIO
import asyncio
from queue import Queue
import threading
from PIL import Image
import sys
import os
import logging
import uuid
from collections import defaultdict

# Add parent directory to path.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Flask for web server.
from flask import Flask, render_template, request, jsonify, send_from_directory  # type: ignore
from flask_cors import CORS  # type: ignore

# Import ALL MaxSight components.
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
from app.ui.haptic_feedback import HapticFeedback, HapticPattern
from ml.utils.logging_config import setup_logging
from ml.utils.output_scheduler import (
    OutputMode, Severity, RuntimeOutput, 
    create_patient_output, create_clinician_output, create_dev_output
)

from .config import config
from .exceptions import (
    MaxSightSimulatorError, SessionError, SessionNotFoundError,
    SessionExpiredError, ImageProcessingError, InvalidImageError,
    ImageTooLargeError, RateLimitError, ValidationError
)
RateLimitExceededError = RateLimitError
from .rate_limiter import RateLimiter, GlobalRateLimiter
from .validators import (
    validate_session_id, validate_condition, validate_scenario,
    validate_output_mode, validate_image_file, validate_image_data,
    validate_init_request
)
from .structured_logging import setup_structured_logging, get_component_logger

from ml.middleware.security_headers import add_security_headers
from ml.middleware.error_sanitizer import sanitize_error, log_error
from ml.security.validation import decode_and_validate_image
from ml.auth.token import make_token, verify_token
from .metrics import metrics, get_health_status
from .priority_queue import PriorityQueue, MessagePriority
from .degraded_modes import DegradedState, DegradedMode
from .output_hierarchy import OutputAuthorityManager, OutputAuthority, OutputRequest
from .utils import get_device, preprocess_image, postprocess_outputs, run_inference, extract_urgency_level, prepare_scene_detections
from ml.utils.priority_filter import PriorityBudgetFilter
from ml.utils.alert_cooldown import AlertCooldownFilter

# Setup structured logging.
logger = setup_structured_logging(config.log_level)
api_logger = get_component_logger('api')
session_logger = get_component_logger('session')
core_logger = get_component_logger('core')

app = Flask(__name__, 
            template_folder=Path(__file__).parent / 'templates',
            static_folder=Path(__file__).parent / 'static')

cors_origins = os.getenv('MAXSIGHT_CORS_ORIGINS', 'http://localhost:8002,http://127.0.0.1:8002').split(',')
CORS(app, origins=cors_origins, supports_credentials=True)

# Add security headers to all responses.
@app.after_request
def add_security_headers_after_request(response):
    """Add security headers to all Flask responses."""
    add_security_headers(response)
    return response

# Rate limiters.
session_rate_limiter = RateLimiter(config.rate_limit_per_session)
global_rate_limiter = GlobalRateLimiter(config.rate_limit_global)

# Serialize inference to avoid GPU memory leaks and keep model access thread-safe.
INFERENCE_SEMAPHORE = threading.Semaphore(value=1)


# Pipeline latency tracker (per-stage timing for bottleneck analysis)
class PipelineLatencyTracker:
    """Track per-stage pipeline timing: preprocess, gpu_transfer, model, postprocess, overlay, audio."""

    STAGES = ("preprocess", "gpu_transfer", "model", "postprocess", "overlay", "audio")

    def __init__(self):
        self._stage_start: Optional[float] = None
        self._current_stage: Optional[str] = None
        self._times_ms: Dict[str, float] = {s: 0.0 for s in self.STAGES}

    def start_stage(self, name: str) -> None:
        if self._current_stage is not None:
            self.end_stage()
        self._current_stage = name
        self._stage_start = time.perf_counter()

    def end_stage(self) -> None:
        if self._current_stage is not None and self._stage_start is not None:
            elapsed_ms = (time.perf_counter() - self._stage_start) * 1000
            self._times_ms[self._current_stage] = elapsed_ms
        self._current_stage = None
        self._stage_start = None

    def get_breakdown(self) -> Dict[str, float]:
        if self._current_stage is not None:
            self.end_stage()
        total = sum(self._times_ms.values())
        out = dict(self._times_ms)
        out["total_ms"] = total
        return out


class MaxSightCore:
    """Shared resources across all sessions."""
    
    _instance: Optional['MaxSightCore'] = None
    _lock = threading.Lock()
    
    def __init__(self, device: Optional[str] = None):
        """Initialize shared resources (model, schedulers, etc.)."""
        core_logger.info("Initializing MaxSightCore (shared resources)")
        
        # Device setup (using utility function)
        self.device = get_device(device)
        core_logger.info(f"Core device: {self.device}")
        
        # Initialize model (shared, thread-safe under no_grad)
        core_logger.info("Loading model")
        try:
            self.model = create_model()
            checkpoint_path = getattr(config, "model_checkpoint_path", None)
            if not checkpoint_path:
                default_glaucoma = Path("/content/drive/MyDrive/MaxSight/checkpoints_glaucoma/best_model.pt")
                if default_glaucoma.exists():
                    checkpoint_path = str(default_glaucoma)
                    core_logger.info("Using default glaucoma checkpoint: %s", checkpoint_path)
            if checkpoint_path and Path(checkpoint_path).exists():
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                self.model.load_state_dict(state, strict=False)
                core_logger.info("Loaded checkpoint: %s", checkpoint_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            core_logger.info("Model loaded successfully")
        except Exception as e:
            core_logger.exception("Error loading model", exc_info=True)
            raise RuntimeError(f"Failed to initialize model: {str(e)}") from e
        
        self.scheduler = CrossModalScheduler(OutputConfig())
        self.ocr = OCRIntegration()
        self.description_gen = DescriptionGenerator()
        self.overlay_engine = OverlayEngine()
        
        core_logger.info("MaxSightCore initialized")
    
    @classmethod
    def get_instance(cls, device: Optional[str] = None) -> 'MaxSightCore':
        """Get singleton instance (thread-safe). Uses proper locking pattern (no double-checked locking)."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(device)
        return cls._instance


class MaxSightSession:
    """Per-user session state."""
    
    def __init__(self, session_id: str, core: MaxSightCore, output_mode: Optional[OutputMode] = None):
        self.session_id = session_id
        self.core = core
        self.output_mode = output_mode or config.default_output_mode
        
        # Per-session lock for frame processing.
        self.lock = threading.Lock()
        
        # Per-session components.
        self.preprocessor: Optional[ImagePreprocessor] = None
        self.spatial_memory = SpatialMemory()
        self.path_planner = PathPlanner()
        self.session_manager = SessionManager()
        self.task_generator = TaskGenerator()
        self.therapy = TherapyTaskIntegrator()
        self.voice_feedback = VoiceFeedback()
        self.haptic_feedback = HapticFeedback()
        self.current_condition: Optional[str] = None
        self.current_scenario: Optional[str] = None
        self.session_active = False
        self.stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'avg_latency_ms': 0.0,
            'total_inference_time': 0.0
        }
        self.baseline_output_path = Path(__file__).parent / f'baseline_output_{session_id}.json'
        self.voice_queue = PriorityQueue(maxsize=config.voice_queue_maxsize)
        self.haptic_queue = PriorityQueue(maxsize=config.haptic_queue_maxsize)
        self._voice_worker_running = False
        self._haptic_worker_running = False
        self._aborted = False
        self.last_processed_frame_id = -1
        self._frame_id_lock = threading.Lock()
        self.degraded_state = DegradedState()
        self.output_authority = OutputAuthorityManager()
        self._memory_usage_mb = 0.0
        self._spatial_memory_count = 0
        self.created_at = time.time()
        self.last_activity = time.time()
        
        session_logger.info("Session initialized", session_id=session_id)
        self._start_async_workers()
    
    def set_user_condition(self, condition: str):
        with self.lock:
            self.current_condition = condition
            self.preprocessor = ImagePreprocessor(condition_mode=condition)
            logger.info(f"Session {self.session_id}: Condition set to {condition}")
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()
    
    def is_expired(self, timeout_seconds: Optional[int] = None) -> bool:
        """Check if session has expired."""
        if timeout_seconds is None:
            timeout_seconds = config.session_timeout_seconds
        assert timeout_seconds is not None  # Type narrowing for linter.
        return (time.time() - self.last_activity) > timeout_seconds
    
    def abort(self):
        """Hard kill switch - immediately stop all outputs."""
        session_logger.warning("Session aborted", session_id=self.session_id)
        with self.lock:
            self._aborted = True
            while not self.voice_queue.empty():
                try:
                    self.voice_queue.get_nowait()
                except Exception as e:
                    session_logger.warning(f"Error flushing voice queue during abort: {str(e)}", 
                                          session_id=self.session_id)
            while not self.haptic_queue.empty():
                try:
                    self.haptic_queue.get_nowait()
                except Exception as e:
                    session_logger.warning(f"Error flushing haptic queue during abort: {str(e)}", 
                                          session_id=self.session_id)
            # Clear output authority.
            self.output_authority.reset()
            # Stop voice/haptic immediately.
            try:
                self.voice_feedback.stop()
            except Exception as e:
                session_logger.warning(f"Error stopping voice feedback during abort: {str(e)}", 
                                     session_id=self.session_id)
            try:
                self.haptic_feedback.stop()
            except Exception as e:
                session_logger.warning(f"Error stopping haptic feedback during abort: {str(e)}", 
                                     session_id=self.session_id)
    
    def shutdown(self):
        session_logger.info("Shutting down session", session_id=self.session_id)
        self._voice_worker_running = False
        self._haptic_worker_running = False
        try:
            self.voice_queue.put((None, MessagePriority.LOW), block=False)
        except Exception:
            pass  # Queue may be full, workers will stop on timeout.
        try:
            self.haptic_queue.put((None, MessagePriority.LOW), block=False)
        except Exception:
            pass
        
        if hasattr(self, 'voice_thread'):
            self.voice_thread.join(timeout=1.0)
        if hasattr(self, 'haptic_thread'):
            self.haptic_thread.join(timeout=1.0)
    
    def _start_async_workers(self) -> None:
        """Start background workers for async voice and haptic processing."""
        def voice_worker():
            self._voice_worker_running = True
            consecutive_failures = 0
            max_failures = 5
            backoff_seconds = 0.1
            
            while self._voice_worker_running and not self._aborted:
                try:
                    priority, message = self.voice_queue.get(timeout=0.1)
                    if message is None:
                        break
                    if self._aborted:
                        break
                    if not self.degraded_state.is_degraded(DegradedMode.AUDIO_UNAVAILABLE):
                        try:
                            self.voice_feedback.speak_custom(message, priority=priority)
                            consecutive_failures = 0  # Reset on success.
                            backoff_seconds = 0.1
                        except (OSError, IOError) as e:
                            # Hardware/system errors.
                            consecutive_failures += 1
                            session_logger.error(f"Voice hardware error: {str(e)}", 
                                                session_id=self.session_id)
                            if consecutive_failures >= max_failures:
                                self.degraded_state.set_degraded(DegradedMode.AUDIO_UNAVAILABLE, 
                                                                f"Hardware failure: {str(e)}")
                                break
                            time.sleep(backoff_seconds)
                            backoff_seconds = min(backoff_seconds * 2, 5.0)
                        except Exception as e:
                            session_logger.error(f"Voice processing error: {str(e)}", 
                                                session_id=self.session_id)
                            consecutive_failures += 1
                            if consecutive_failures >= max_failures:
                                self.degraded_state.set_degraded(DegradedMode.AUDIO_UNAVAILABLE, str(e))
                                break
                    else:
                        session_logger.warning("Voice output suppressed - audio degraded", 
                                              session_id=self.session_id)
                except Exception as e:
                    # Queue errors - log but continue. Empty exception is expected on timeout, don't log as error.
                    from queue import Empty
                    if not isinstance(e, (Empty, TimeoutError, AttributeError)):
                        session_logger.error(f"Voice queue error: {str(e)}", 
                                            session_id=self.session_id)
                    time.sleep(0.1)  # Brief pause on error.
        
        def haptic_worker():
            self._haptic_worker_running = True
            consecutive_failures = 0
            max_failures = 5
            backoff_seconds = 0.1
            
            while self._haptic_worker_running and not self._aborted:
                try:
                    priority, message = self.haptic_queue.get(timeout=0.1)
                    if message is None:
                        break
                    if self._aborted:
                        break
                    if not self.degraded_state.is_degraded(DegradedMode.HAPTIC_UNAVAILABLE):
                        try:
                            if isinstance(message, tuple):
                                pattern, intensity = message
                                self.haptic_feedback.trigger(pattern, intensity=intensity)
                            else:
                                self.haptic_feedback.trigger(message)
                            consecutive_failures = 0  # Reset on success.
                            backoff_seconds = 0.1
                        except (OSError, IOError) as e:
                            # Hardware/system errors.
                            consecutive_failures += 1
                            session_logger.error(f"Haptic hardware error: {str(e)}", 
                                                session_id=self.session_id)
                            if consecutive_failures >= max_failures:
                                self.degraded_state.set_degraded(DegradedMode.HAPTIC_UNAVAILABLE, 
                                                                f"Hardware failure: {str(e)}")
                                break
                            time.sleep(backoff_seconds)
                            backoff_seconds = min(backoff_seconds * 2, 5.0)
                        except Exception as e:
                            session_logger.error(f"Haptic processing error: {str(e)}", 
                                                session_id=self.session_id)
                            consecutive_failures += 1
                            if consecutive_failures >= max_failures:
                                self.degraded_state.set_degraded(DegradedMode.HAPTIC_UNAVAILABLE, str(e))
                                break
                    else:
                        session_logger.warning("Haptic output suppressed - haptic degraded", 
                                            session_id=self.session_id)
                except Exception as e:
                    # Queue errors - log but continue. Empty exception is expected on timeout, don't log as error.
                    from queue import Empty
                    if not isinstance(e, (Empty, TimeoutError, AttributeError)):
                        session_logger.error(f"Haptic queue error: {str(e)}", 
                                            session_id=self.session_id)
                    time.sleep(0.1)  # Brief pause on error.
        
        self.voice_thread = threading.Thread(target=voice_worker, daemon=True, name=f"Voice-{self.session_id}")
        self.haptic_thread = threading.Thread(target=haptic_worker, daemon=True, name=f"Haptic-{self.session_id}")
        self.voice_thread.start()
        self.haptic_thread.start()
    
    # Forward all the processing methods from MaxSightSimulator. (We'll move them in the next step)
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input (using utility function)."""
        return preprocess_image(image, self.preprocessor, self.core.device)
    
    def _postprocess_outputs(self, outputs: Dict[str, Any], confidence_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Post-process model outputs to extract detections (using utility function)."""
        return postprocess_outputs(self.core.model, outputs, confidence_threshold)
    
    def _run_inference(self, image_tensor: torch.Tensor, audio_features: Optional[np.ndarray] = None) -> Tuple[Dict[str, Any], float]:
        """Run model inference with thread safety guarantees. Uses semaphore to serialize inference and torch.no_grad() to prevent memory leaks."""
        # Acquire inference semaphore (serializes model access)
        with INFERENCE_SEMAPHORE:
            inference_start = time.perf_counter()
            with torch.no_grad():  # Prevents graph construction and memory leaks.
                if audio_features is not None:
                    audio_tensor = torch.from_numpy(audio_features).unsqueeze(0).to(self.core.device)
                    outputs = self.core.model(image_tensor, audio_tensor)
                else:
                    outputs = self.core.model(image_tensor)
            inference_time = time.perf_counter() - inference_start
        return outputs, inference_time
    
    def _run_ocr(self, image: Image.Image, outputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run OCR text detection."""
        ocr_results = []
        try:
            text_scores = outputs.get('text_regions', torch.zeros(1, 196))
            boxes = outputs.get('boxes', torch.zeros(1, 196, 4))
            ocr_results = self.core.ocr.process_image_for_ocr(
                image=image,
                text_scores=text_scores[0],
                boxes=boxes[0]
            )
        except Exception as e:
            session_logger.error(f"OCR processing failed: {str(e)}", 
                                session_id=self.session_id)
            self.degraded_state.set_degraded(DegradedMode.TEXT_DETECTION_OFFLINE, str(e))
            ocr_results = []
        return ocr_results
    
    def _generate_description(self, detections_list: List[Dict[str, Any]], outputs: Dict[str, Any], ocr_results: List[Dict[str, Any]]) -> str:
        """Generate scene description from detections and OCR (using utility functions)."""
        urgency_level = extract_urgency_level(outputs)
        scene_detections = prepare_scene_detections(detections_list, urgency_level)
        
        scene_description = self.core.description_gen.generate_scene_description(
            detections=scene_detections,
            urgency_score=urgency_level
        )
        
        if ocr_results:
            ocr_texts = [r.get('text', '') for r in ocr_results if r.get('text')]
            if ocr_texts:
                max_texts = config.max_ocr_texts_in_description
                scene_description += f" Text detected: {', '.join(ocr_texts[:max_texts])}"
        
        return scene_description
    
    def _update_memory(self, detections_list: List[Dict[str, Any]]) -> None:
        """Update spatial memory with current detections, enforcing resource caps."""
        # Check resource cap before updating.
        if self._spatial_memory_count >= config.max_spatial_memory_entries:
            # Prune oldest entries if at capacity.
            try:
                # Get memory size and prune if needed.
                memory_size = len(self.spatial_memory.memory) if hasattr(self.spatial_memory, 'memory') else 0
                if memory_size >= config.max_spatial_memory_entries:
                    # Prune oldest 10% of entries.
                    prune_count = max(1, memory_size // 10)
                    if hasattr(self.spatial_memory, 'prune_oldest'):
                        self.spatial_memory.prune_oldest(prune_count)
                    self.degraded_state.set_degraded(DegradedMode.MEMORY_FULL, 
                                                    f"Memory at capacity, pruned {prune_count} entries")
            except Exception as e:
                session_logger.error(f"Error pruning spatial memory: {str(e)}", 
                                    session_id=self.session_id)
        
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
            try:
                self.spatial_memory.update(
                    detections=spatial_detections,
                    timestamp=time.time()
                )
                # Update resource tracking.
                self._spatial_memory_count = len(self.spatial_memory.memory) if hasattr(self.spatial_memory, 'memory') else self._spatial_memory_count + len(spatial_detections)
            except Exception as e:
                session_logger.error(f"Error updating spatial memory: {str(e)}", 
                                    session_id=self.session_id)
                self.degraded_state.set_degraded(DegradedMode.MEMORY_FULL, str(e))
    
    def _plan_path(self, detections_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Plan navigation path if navigation scenario."""
        path_info = None
        if self.current_scenario == 'navigation':
            path_result = self.path_planner.plan_path(
                detections=detections_list,
                target_direction='forward'
            )
            if path_result is not None:
                if hasattr(path_result, '__dict__'):
                    path_info = path_result.__dict__
                else:
                    path_info = {'path': str(path_result)}
        return path_info
    
    def _schedule_outputs(self, detections_list: List[Dict[str, Any]], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule cross-modal outputs (voice, haptic, visual)."""
        model_outputs: Dict[str, Any] = {}
        urgency_scores = outputs.get('urgency_scores')
        uncertainty = outputs.get('uncertainty')
        if urgency_scores is not None:
            model_outputs['urgency_scores'] = urgency_scores
        if uncertainty is not None:
            model_outputs['uncertainty'] = uncertainty
        
        scheduled_outputs = self.core.scheduler.schedule_outputs(
            detections=detections_list,
            model_outputs=model_outputs,
            timestamp=time.time()
        )
        if isinstance(scheduled_outputs, list):
            return {'outputs': scheduled_outputs, 'count': len(scheduled_outputs)}
        return scheduled_outputs if isinstance(scheduled_outputs, dict) else {'outputs': scheduled_outputs}
    
    def _render_overlay(self, image: Image.Image, detections_list: List[Dict[str, Any]], ocr_results: List[Dict[str, Any]], path_info: Optional[Dict[str, Any]]) -> Optional[str]:
        """Render visual overlays on image."""
        try:
            urgency_scores = None
            if detections_list and 'urgency' in detections_list[0]:
                urgency_scores = np.array([det.get('urgency', 0) for det in detections_list])
            
            overlay_image = self.core.overlay_engine.create_overlay(
                base_image=image,
                detections=detections_list,
                urgency_scores=urgency_scores,
                text_regions=ocr_results
            )
            
            buffered = BytesIO()
            overlay_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            logger.warning(f"Overlay rendering error in session {self.session_id}: {e}")
            return None
    
    def _queue_outputs(self, scene_description: str, outputs: Dict[str, Any], detections_list: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Queue voice and haptic outputs asynchronously with confidence gating and authority hierarchy."""
        voice_announcements = []
        haptic_patterns = []
        
        # Compute confidence metrics.
        objectness = outputs.get('objectness', torch.zeros(1, 196))
        max_objectness = float(objectness.max().item()) if objectness.numel() > 0 else 0.0
        
        urgency_scores = outputs.get('urgency_scores', torch.zeros(1, 4))
        urgency_level = 0
        urgency_confidence = 0.0
        if urgency_scores.numel() > 0:
            urgency_level = int(urgency_scores.argmax(dim=1).item())
            urgency_probs = torch.softmax(urgency_scores, dim=-1)[0]
            urgency_confidence = float(urgency_probs.max().item())
        
        overall_confidence = max(max_objectness, urgency_confidence)
        
        # Confidence gating for patient output.
        min_confidence = config.min_confidence_for_patient_output
        if self.output_mode == OutputMode.PATIENT and overall_confidence < min_confidence:
            # Suppress low-confidence output for patients.
            neutral_message = "Unable to confirm objects in view"
            request = OutputRequest(
                authority=OutputAuthority.DESCRIPTIVE_NARRATION,
                content=neutral_message,
                priority=0
            )
            if self.output_authority.request_output(request):
                # Queue format: (message, priority) - PriorityQueue.put expects this.
                self.voice_queue.put((neutral_message, MessagePriority.LOW))
                voice_announcements.append(neutral_message)
            return voice_announcements, haptic_patterns
        
        # Descriptive narration (lowest authority)
        if scene_description and overall_confidence >= min_confidence:
            request = OutputRequest(
                authority=OutputAuthority.DESCRIPTIVE_NARRATION,
                content=scene_description,
                priority=0
            )
            if self.output_authority.request_output(request):
                # Queue format: (message, priority)
                self.voice_queue.put((scene_description, MessagePriority.NORMAL))
                voice_announcements.append(scene_description)
        
        # Safety alerts (highest authority)
        warning_threshold = config.urgency_warning_threshold
        min_critical_confidence = config.min_confidence_for_critical_alert
        
        if urgency_level >= warning_threshold and urgency_confidence >= min_critical_confidence:
            warning_message = f"Warning: High urgency detected"
            request = OutputRequest(
                authority=OutputAuthority.SAFETY_ALERTS,
                content=warning_message,
                priority=urgency_level
            )
            if self.output_authority.request_output(request):
                # Queue format: (message, priority)
                self.voice_queue.put((warning_message, MessagePriority.CRITICAL))
                voice_announcements.append(warning_message)
                # Haptic feedback for critical alerts - format: (message, priority)
                haptic_msg = (HapticPattern.LONG_PULSE, config.haptic_intensity_high)
                self.haptic_queue.put((haptic_msg, MessagePriority.CRITICAL))
                haptic_patterns.append({'pattern': 'long_pulse', 'intensity': config.haptic_intensity_high})
        elif len(detections_list) > 0 and overall_confidence >= min_confidence:
            # Normal detection feedback - format: (message, priority)
            haptic_msg = (HapticPattern.MICRO_PULSE, config.haptic_intensity_low)
            self.haptic_queue.put((haptic_msg, MessagePriority.NORMAL))
            haptic_patterns.append({'pattern': 'micro_pulse', 'intensity': config.haptic_intensity_low})
        
        return voice_announcements, haptic_patterns
    
    def process_frame(
        self,
        image: Image.Image,
        audio_features: Optional[np.ndarray] = None,
        frame_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process a single frame through the complete MaxSight pipeline."""
        with self.lock:  # Per-session locking.
            # Check if aborted.
            if self._aborted:
                session_logger.warning("Frame processing rejected - session aborted", 
                                      session_id=self.session_id)
                return {'error': 'Session aborted', 'session_id': self.session_id}
            
            start_time = time.perf_counter()
            self.update_activity()
            
            # Frame ordering check (deterministic processing)
            if frame_id is not None:
                with self._frame_id_lock:
                    if frame_id <= self.last_processed_frame_id:
                        session_logger.warning("Frame rejected - out of order", 
                                              session_id=self.session_id,
                                              frame_id=frame_id,
                                              last_processed=self.last_processed_frame_id)
                        return {
                            'error': 'Frame out of order',
                            'frame_id': frame_id,
                            'last_processed': self.last_processed_frame_id
                        }
                    self.last_processed_frame_id = frame_id
            
            # Resource cap check.
            if self._spatial_memory_count >= config.max_spatial_memory_entries:
                self.degraded_state.set_degraded(DegradedMode.MEMORY_FULL, 
                                                f"Memory at capacity: {self._spatial_memory_count}")
                session_logger.warning("Spatial memory at capacity", 
                                      session_id=self.session_id,
                                      count=self._spatial_memory_count)
            
            # Pipeline latency tracker (per-stage timing)
            tracker = PipelineLatencyTracker()
            
            # 1. Preprocessing.
            tracker.start_stage('preprocess')
            try:
                image_tensor = self._preprocess_image(image)
            except Exception as e:
                tracker.end_stage()
                session_logger.error(f"Image preprocessing failed: {str(e)}", 
                                    session_id=self.session_id)
                self.degraded_state.set_degraded(DegradedMode.VISION_UNSTABLE, str(e))
                return {'error': 'Image preprocessing failed', 'degraded_mode': 'vision_unstable'}
            tracker.end_stage()
            
            # Check abort before expensive operations.
            if self._aborted:
                return {'error': 'Session aborted', 'session_id': self.session_id}
            
            # 2. Model inference (thread-safe under no_grad)
            tracker.start_stage('model')
            try:
                outputs, inference_time = self._run_inference(image_tensor, audio_features)
            except Exception as e:
                tracker.end_stage()
                session_logger.error(f"Model inference failed: {str(e)}", 
                                    session_id=self.session_id)
                self.degraded_state.set_degraded(DegradedMode.VISION_UNSTABLE, str(e))
                return {'error': 'Model inference failed', 'degraded_mode': 'vision_unstable'}
            tracker.end_stage()
            
            # Check abort after inference.
            if self._aborted:
                return {'error': 'Session aborted', 'session_id': self.session_id}
            
            # 3. Post-process detections + priority budget + alert cooldown.
            tracker.start_stage('postprocess')
            detections_list = self._postprocess_outputs(outputs, confidence_threshold=config.confidence_threshold)
            if not hasattr(self, 'priority_filter'):
                self.priority_filter = PriorityBudgetFilter(max_alerts_per_frame=config.max_alerts_per_frame)
                self.alert_cooldown = AlertCooldownFilter(cooldown_frames=config.alert_cooldown_frames)
            detections_list = self.priority_filter.filter_alerts(detections_list)
            detections_list = self.alert_cooldown.filter_alerts(detections_list, frame_id=frame_id)
            tracker.end_stage()
            
            # 4. OCR text detection.
            ocr_results = self._run_ocr(image, outputs)
            
            # 5. Description generation.
            scene_description = self._generate_description(detections_list, outputs, ocr_results)
            
            # 6. Spatial memory update.
            self._update_memory(detections_list)
            
            # 7. Path planning.
            path_info = self._plan_path(detections_list)
            
            # 8. Output scheduling.
            scheduled_outputs = self._schedule_outputs(detections_list, outputs)
            
            # 9. Therapy integration.
            therapy_feedback = None
            if self.session_active and detections_list and not self._aborted:
                target_objects = [det.get('class_name', 'object') for det in detections_list[:3]]
                therapy_feedback = self.therapy.create_attention_task(
                    scene_description=scene_description or "Scene with objects",
                    target_objects=target_objects,
                    difficulty=config.therapy_difficulty
                )
            
            # Check abort before generating outputs.
            if self._aborted:
                return {'error': 'Session aborted', 'session_id': self.session_id}
            
            # Pipeline breakdown so far (for adaptive skip)
            breakdown_so_far = tracker.get_breakdown()
            total_so_far_ms = float(breakdown_so_far.get('total_ms', 0.0))
            skip_non_critical = total_so_far_ms > 200.0
            try:
                import psutil
                if psutil.cpu_percent(interval=None) > 80.0:
                    skip_non_critical = True
            except Exception:
                pass
            if skip_non_critical:
                session_logger.debug(
                    "Adaptive skip: pipeline %.0fms > 200ms or CPU > 80%%, skipping overlay/audio",
                    total_so_far_ms,
                    session_id=self.session_id,
                )
            
            # 10. Generate overlays (skip if adaptive skip)
            if not skip_non_critical:
                tracker.start_stage('overlay')
                overlay_image_b64 = self._render_overlay(image, detections_list, ocr_results, path_info)
                tracker.end_stage()
            else:
                overlay_image_b64 = None
            
            # 11. Queue outputs (voice and haptic) - only if not aborted and not skipped.
            if not self._aborted and not skip_non_critical:
                tracker.start_stage('audio')
                voice_announcements, haptic_patterns = self._queue_outputs(scene_description, outputs, detections_list)
                tracker.end_stage()
            else:
                voice_announcements, haptic_patterns = [], []
            
            pipeline_breakdown = tracker.get_breakdown()
            if session_logger.isEnabledFor(logging.DEBUG):
                session_logger.debug(
                    "Pipeline: total=%.1fms model=%.1fms",
                    pipeline_breakdown.get('total_ms', 0),
                    pipeline_breakdown.get('model', 0),
                    session_id=self.session_id,
                )
            
            # Update statistics.
            self.stats['frames_processed'] += 1
            self.stats['total_inference_time'] += inference_time
            self.stats['total_detections'] += len(detections_list)
            self.stats['avg_latency_ms'] = (self.stats['total_inference_time'] / 
                                            self.stats['frames_processed'] * 1000)
            
            total_time = time.perf_counter() - start_time
            
            # Extract 3 perspectives.
            reasoning_trace = self._extract_reasoning_trace(outputs, detections_list)
            final_judgment = self._extract_final_judgment(outputs, detections_list)
            
            # Shape response based on output mode.
            result = self._shape_response(
                detections_list=detections_list,
                outputs=outputs,
                scene_description=scene_description,
                ocr_results=ocr_results,
                voice_announcements=voice_announcements,
                haptic_patterns=haptic_patterns,
                path_info=path_info,
                therapy_feedback=therapy_feedback,
                overlay_image_b64=overlay_image_b64,
                inference_time_ms=inference_time * 1000,
                total_time_ms=total_time * 1000,
                reasoning_trace=reasoning_trace,
                final_judgment=final_judgment,
                scheduled_outputs=scheduled_outputs
            )
            result['pipeline_breakdown'] = pipeline_breakdown
            
            # Save baseline output for regression testing (first frame only)
            self._save_baseline_output(result)
            
            return result
    
    def _extract_reasoning_trace(self, outputs: Dict[str, Any], detections_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract model reasoning trace showing how predictions are made."""
        trace = {
            'feature_extraction': {},
            'attention_weights': {},
            'confidence_scores': {},
            'decision_path': []
        }
        
        scene_emb = outputs.get('scene_embedding')
        if scene_emb is not None:
            trace['feature_extraction'] = {
                'scene_embedding_dim': scene_emb.shape[-1] if scene_emb.numel() > 0 else 0,
                'embedding_norm': float(scene_emb.norm().item()) if scene_emb.numel() > 0 else 0.0
            }
        
        objectness = outputs.get('objectness', torch.zeros(1, 196))
        if objectness.numel() > 0:
            trace['confidence_scores'] = {
                'max_objectness': float(objectness.max().item()),
                'mean_objectness': float(objectness.mean().item()),
                'num_high_confidence': int((objectness > 0.5).sum().item())
            }
        
        urgency_scores = outputs.get('urgency_scores', torch.zeros(1, 4))
        if urgency_scores.numel() > 0:
            urgency_probs = torch.softmax(urgency_scores, dim=-1)[0]
            trace['attention_weights'] = {
                'urgency_levels': urgency_probs.cpu().tolist(),
                'predicted_urgency': int(urgency_scores.argmax(dim=1).item())
            }
        
        if detections_list:
            top_detections = sorted(detections_list, key=lambda x: x.get('confidence', 0), reverse=True)[:5]
            trace['decision_path'] = [
                {
                    'step': i+1,
                    'object': det.get('class_name', 'unknown'),
                    'confidence': round(det.get('confidence', 0), 3),
                    'urgency': det.get('urgency', 0),
                    'distance': det.get('distance', 0)
                }
                for i, det in enumerate(top_detections)
            ]
        
        return trace
    
    def _extract_final_judgment(self, outputs: Dict[str, Any], detections_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract final weighted judgment with confidence scores."""
        urgency_scores = outputs.get('urgency_scores', torch.zeros(1, 4))
        urgency_level = int(urgency_scores.argmax(dim=1).item()) if urgency_scores.numel() > 0 else 0
        urgency_confidence = float(torch.softmax(urgency_scores, dim=-1)[0].max().item()) if urgency_scores.numel() > 0 else 0.0
        
        objectness = outputs.get('objectness', torch.zeros(1, 196))
        max_objectness = float(objectness.max().item()) if objectness.numel() > 0 else 0.0
        
        final_score = (urgency_confidence * 0.6 + max_objectness * 0.4)
        
        top_detections = sorted(detections_list, key=lambda x: x.get('confidence', 0), reverse=True)[:3]
        weighted_detections = [
            {
                'object': det.get('class_name', 'unknown'),
                'confidence': round(det.get('confidence', 0), 3),
                'weight': round(det.get('confidence', 0) * (1 + det.get('urgency', 0) * 0.2), 3),
                'urgency': det.get('urgency', 0)
            }
            for det in top_detections
        ]
        
        return {
            'final_score': round(final_score, 3),
            'urgency_level': urgency_level,
            'urgency_confidence': round(urgency_confidence, 3),
            'objectness_confidence': round(max_objectness, 3),
            'num_detections': len(detections_list),
            'weighted_detections': weighted_detections,
            'decision': 'high_alert' if urgency_level >= 3 else 'moderate_alert' if urgency_level >= 2 else 'normal'
        }
    
    def _shape_response(
        self,
        detections_list: List[Dict[str, Any]],
        outputs: Dict[str, Any],
        scene_description: str,
        ocr_results: List[Dict[str, Any]],
        voice_announcements: List[str],
        haptic_patterns: List[Dict[str, Any]],
        path_info: Optional[Dict[str, Any]],
        therapy_feedback: Optional[Dict[str, Any]],
        overlay_image_b64: Optional[str],
        inference_time_ms: float,
        total_time_ms: float,
        reasoning_trace: Dict[str, Any],
        final_judgment: Dict[str, Any],
        scheduled_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Shape response based on output mode."""
        urgency_scores = outputs.get('urgency_scores', torch.zeros(1, 4))
        urgency_level = int(urgency_scores.argmax(dim=1).item()) if urgency_scores.numel() > 0 else 0
        
        if urgency_level >= 3:
            severity = Severity.CRITICAL
        elif urgency_level >= 2:
            severity = Severity.HAZARD
        elif urgency_level >= 1:
            severity = Severity.WARNING
        else:
            severity = Severity.INFO
        
        objectness = outputs.get('objectness', torch.zeros(1, 196))
        avg_confidence = objectness.max().item() if objectness.numel() > 0 else 0.0
        
        if self.output_mode == OutputMode.PATIENT:
            top_hazards = [d for d in detections_list if d.get('urgency', 0) >= 2]
            if top_hazards:
                message = f"{top_hazards[0]['class_name']} detected"
            elif scene_description:
                message = scene_description.split('.')[0] + '.'
            else:
                message = "Scene clear"
            
            return {
                'mode': 'patient',
                'severity': severity.value,
                'message': message,
                'confidence': round(avg_confidence, 2),
                'cooldown_applied': False,
                'overlay_image': overlay_image_b64,
                'detections': detections_list,
                'num_detections': len(detections_list),
                'scene_description': scene_description,
                'voice_announcements': voice_announcements,
                'haptic_patterns': haptic_patterns,
                'text_regions': ocr_results,
                'therapy_feedback': therapy_feedback,
                'stats': {
                    'frames_processed': self.stats['frames_processed'],
                    'avg_latency_ms': self.stats['avg_latency_ms'],
                    'total_detections': len(detections_list)
                },
                'perspectives': {
                    'user_view': {
                        'overlay_image': overlay_image_b64,
                        'scene_description': scene_description,
                        'voice_announcements': voice_announcements
                    },
                    'model_reasoning': reasoning_trace,
                    'final_judgment': final_judgment
                }
            }
        
        elif self.output_mode == OutputMode.CLINICIAN:
            return {
                'mode': 'clinician',
                'severity': severity.value,
                'message': scene_description or "No description",
                'confidence': round(avg_confidence, 2),
                'cooldown_applied': False,
                'latency_ms': round(inference_time_ms, 1),
                'total_time_ms': round(total_time_ms, 1),
                'inference_time_ms': round(inference_time_ms, 1),
                'detections': detections_list,
                'num_detections': len(detections_list),
                'num_hazards': len([d for d in detections_list if d.get('urgency', 0) >= 2]),
                'ocr_texts': [r.get('text', '') for r in ocr_results],
                'text_regions': ocr_results,
                'voice_announcements': voice_announcements,
                'haptic_patterns': haptic_patterns,
                'therapy_feedback': therapy_feedback,
                'component_breakdown': {
                    'detections': len(detections_list),
                    'ocr': len(ocr_results),
                    'voice': len(voice_announcements),
                    'haptic': len(haptic_patterns)
                },
                'overlay_image': overlay_image_b64,
                'stats': {
                    'frames_processed': self.stats['frames_processed'],
                    'avg_latency_ms': self.stats['avg_latency_ms'],
                    'total_detections': self.stats['total_detections']
                },
                'perspectives': {
                    'user_view': {
                        'overlay_image': overlay_image_b64,
                        'scene_description': scene_description,
                        'voice_announcements': voice_announcements,
                        'haptic_patterns': haptic_patterns
                    },
                    'model_reasoning': reasoning_trace,
                    'final_judgment': final_judgment
                }
            }
        
        else:  # DEV mode.
            return {
                'mode': 'dev',
                'severity': severity.value,
                'message': scene_description or "No description",
                'confidence': round(avg_confidence, 2),
                'cooldown_applied': False,
                'frame_number': self.stats['frames_processed'],
                'timestamp': time.time(),
                'processing_time_ms': total_time_ms,
                'inference_time_ms': inference_time_ms,
                'detections': detections_list,
                'num_detections': len(detections_list),
                'urgency_scores': urgency_scores[0].cpu().tolist(),
                'distance_zones': outputs['distance_zones'][0].cpu().tolist(),
                'scene_embedding': outputs['scene_embedding'][0].cpu().tolist(),
                'text_regions': ocr_results,
                'num_text_regions': len(ocr_results),
                'scene_description': scene_description,
                'scheduled_outputs': scheduled_outputs,
                'voice_announcements': voice_announcements,
                'haptic_patterns': haptic_patterns,
                'path_info': path_info,
                'therapy_feedback': therapy_feedback,
                'overlay_image': overlay_image_b64,
                'stats': self.stats.copy(),
                'debug_info': {
                    'condition': self.current_condition,
                    'scenario': self.current_scenario,
                    'session_active': self.session_active,
                    'session_id': self.session_id
                },
                'perspectives': {
                    'user_view': {
                        'overlay_image': overlay_image_b64,
                        'scene_description': scene_description,
                        'voice_announcements': voice_announcements,
                        'haptic_patterns': haptic_patterns,
                        'scheduled_outputs': scheduled_outputs
                    },
                    'model_reasoning': reasoning_trace,
                    'final_judgment': final_judgment
                }
            }
    
    def _save_baseline_output(self, result: Dict[str, Any]) -> None:
        """Save baseline output for regression testing."""
        if self.stats['frames_processed'] == config.baseline_save_frame:
            try:
                baseline = {
                    'frame_number': result.get('frame_number', 0),
                    'num_detections': result.get('num_detections', 0),
                    'num_text_regions': result.get('num_text_regions', 0),
                    'processing_time_ms': result.get('processing_time_ms', 0),
                    'inference_time_ms': result.get('inference_time_ms', 0),
                    'scene_description': result.get('scene_description', ''),
                    'urgency_scores': result.get('urgency_scores', []),
                    'stats': result.get('stats', {}),
                    'session_id': self.session_id
                }
                with open(self.baseline_output_path, 'w') as f:
                    json.dump(baseline, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not save baseline output for session {self.session_id}: {e}")


class SessionRegistry:
    """Thread-safe registry for managing user sessions."""
    
    def __init__(self):
        self.sessions: Dict[str, MaxSightSession] = {}
        self.lock = threading.Lock()
        self.core: Optional[MaxSightCore] = None
        self._janitor_running = False
        self._start_janitor()
    
    def _start_janitor(self):
        """Start background thread to clean up expired sessions."""
        def janitor_worker():
            self._janitor_running = True
            while self._janitor_running:
                try:
                    time.sleep(60)  # Check every minute.
                    self.cleanup_expired_sessions()
                except Exception as e:
                    logger.error(f"Janitor error: {e}")
        
        janitor_thread = threading.Thread(target=janitor_worker, daemon=True, name="SessionJanitor")
        janitor_thread.start()
    
    def get_core(self, device: Optional[str] = None) -> MaxSightCore:
        """Get or create shared core instance."""
        if self.core is None:
            self.core = MaxSightCore.get_instance(device)
        return self.core
    
    def create_session(self, output_mode: OutputMode = OutputMode.PATIENT, device: Optional[str] = None) -> str:
        """Create a new session and return session ID."""
        session_id = str(uuid.uuid4())
        core = self.get_core(device)
        
        with self.lock:
            session = MaxSightSession(session_id, core, output_mode)
            self.sessions[session_id] = session
            logger.info(f"Created session {session_id}")
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[MaxSightSession]:
        """Get session by ID (thread-safe)."""
        with self.lock:
            session = self.sessions.get(session_id)
            if session:
                session.update_activity()
            return session
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session and clean up resources."""
        with self.lock:
            session = self.sessions.pop(session_id, None)
            if session:
                session.shutdown()
                logger.info(f"Deleted session {session_id}")
                return True
            return False
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions."""
        expired = []
        with self.lock:
            for session_id, session in list(self.sessions.items()):
                if session.is_expired():
                    expired.append(session_id)
        
        for session_id in expired:
            logger.info(f"Cleaning up expired session {session_id}")
            self.delete_session(session_id)
    
    def shutdown(self):
        """Shutdown all sessions and janitor."""
        self._janitor_running = False
        with self.lock:
            for session in list(self.sessions.values()):
                session.shutdown()
            self.sessions.clear()


# Global registry (replaces singleton simulator)
registry = SessionRegistry()


# Helper Functions for Flask Routes.

def get_session_id() -> Optional[str]:
    """Get session ID from request (header or JSON)."""
    # Checks header first.
    session_id = request.headers.get('X-Session-ID')
    if session_id:
        return session_id
    
    # Checks JSON body.
    if request.is_json and request.json:
        session_id = request.json.get('session_id')
        if session_id:
            return session_id
    
    return None


def require_session() -> MaxSightSession:
    """Get session from request, create if needed (single-user mode fallback)."""
    session_id = get_session_id()
    
    if not config.multi_user_enabled:
        # Single-user demo mode: use a default session.
        session_id = session_id or 'default'
        session = registry.get_session(session_id)
        if not session:
            session_id = registry.create_session()
            session = registry.get_session(session_id)
            if not session:
                raise RuntimeError("Failed to create default session")
        return session
    
    # Multi-user mode: session ID is required.
    if not session_id:
        raise ValueError("Session ID required. Send X-Session-ID header or session_id in JSON.")
    
    session = registry.get_session(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found. Initialize session first with /api/init")
    
    return session


# Legacy MaxSightSimulator (DEPRECATED - Use MaxSightSession instead)
# MaxSightSession with SessionRegistry provides multi-user support.

class MaxSightSimulator:
    """DEPRECATED: Legacy single-user simulator."""
    
    # Configuration constants (DEPRECATED - use config module instead)
    _CONFIG = {
        'confidence_threshold': 0.3,
        'max_ocr_texts_in_description': 3,
        'therapy_difficulty': 0.5,
        'urgency_warning_threshold': 2,
        'haptic_intensity_high': 0.7,
        'haptic_intensity_low': 0.3,
        'baseline_save_frame': 1
    }
    
    def __init__(self, device: Optional[str] = None, output_mode: OutputMode = OutputMode.PATIENT):
        """Initialize all MaxSight components (DEPRECATED)."""
        import warnings
        warnings.warn(
            "MaxSightSimulator is deprecated. Use MaxSightSession with SessionRegistry instead.",
            DeprecationWarning,
            stacklevel=2
        )
        logger.warning("Using deprecated MaxSightSimulator class. Consider migrating to MaxSightSession.")
        
        # Output mode.
        self.output_mode = output_mode
        
        # Device setup (using utility function)
        self.device = get_device(device)
        logger.info(f"Device: {self.device}")
        
        # Initialize model.
        logger.info("Loading model...")
        try:
            self.model = create_model()
            checkpoint_path = getattr(config, "model_checkpoint_path", None)
            if checkpoint_path and Path(checkpoint_path).exists():
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                self.model.load_state_dict(state, strict=False)
                logger.info("Loaded checkpoint: %s", checkpoint_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize model: {str(e)}") from e
        
        # Initialize all components.
        logger.info("Initializing components...")
        self.preprocessor = None  # Will be set per user condition.
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
        
        # User state.
        self.current_condition = None
        self.current_scenario = None
        self.session_active = False
        
        # Statistics.
        self.stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'avg_latency_ms': 0.0,
            'total_inference_time': 0.0
        }
        
        # Baseline output path for regression testing.
        self.baseline_output_path = Path(__file__).parent / 'baseline_output.json'
        
        # Initialize async queues for voice and haptic feedback (thread safety)
        self.voice_queue: Queue = Queue()
        self.haptic_queue: Queue = Queue()
        self._voice_worker_running = False
        self._haptic_worker_running = False
        
        logger.info("Simulator initialized")
        self._start_async_workers()
    
    def set_user_condition(self, condition: str):
        """Set user's visual condition."""
        self.current_condition = condition
        self.preprocessor = ImagePreprocessor(condition_mode=condition)
        logger.info(f"Condition set to: {condition}")
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input (using utility function)."""
        return preprocess_image(image, self.preprocessor, self.device)
    
    def _postprocess_outputs(self, outputs: Dict[str, Any], confidence_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Post-process model outputs to extract detections (using utility function)."""
        return postprocess_outputs(self.model, outputs, confidence_threshold)
    
    def _run_inference(self, image_tensor: torch.Tensor, audio_features: Optional[np.ndarray] = None) -> Tuple[Dict[str, Any], float]:
        """Run model inference with thread safety guarantees (legacy class). Uses semaphore to serialize inference and torch.no_grad() to prevent memory leaks."""
        # Acquire inference semaphore (serializes model access)
        with INFERENCE_SEMAPHORE:
            inference_start = time.perf_counter()
            with torch.no_grad():  # Prevents graph construction and memory leaks.
                if audio_features is not None:
                    audio_tensor = torch.from_numpy(audio_features).unsqueeze(0).to(self.device)
                    outputs = self.model(image_tensor, audio_tensor)
                else:
                    outputs = self.model(image_tensor)
            inference_time = time.perf_counter() - inference_start
        return outputs, inference_time
    
    def _run_ocr(self, image: Image.Image, outputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run OCR text detection. Args: image: Original PIL image outputs: Model outputs dictionary Returns: List of OCR result dictionaries."""
        ocr_results = []
        try:
            text_scores = outputs.get('text_regions', torch.zeros(1, 196))
            boxes = outputs.get('boxes', torch.zeros(1, 196, 4))
            ocr_results = self.ocr.process_image_for_ocr(
                image=image,
                text_scores=text_scores[0],
                boxes=boxes[0]
            )
        except Exception as e:
            logger.warning(f"OCR error: {e}")
            ocr_results = []
        return ocr_results
    
    def _generate_description(self, detections_list: List[Dict[str, Any]], outputs: Dict[str, Any], ocr_results: List[Dict[str, Any]]) -> str:
        """Generate scene description from detections and OCR (using utility functions)."""
        urgency_level = extract_urgency_level(outputs)
        scene_detections = prepare_scene_detections(detections_list, urgency_level)
        
        scene_description = self.description_gen.generate_scene_description(
            detections=scene_detections,
            urgency_score=urgency_level
        )
        
        if ocr_results:
            ocr_texts = [r.get('text', '') for r in ocr_results if r.get('text')]
            if ocr_texts:
                max_texts = config.max_ocr_texts_in_description
                scene_description += f" Text detected: {', '.join(ocr_texts[:max_texts])}"
        
        return scene_description
    
    def _update_memory(self, detections_list: List[Dict[str, Any]]) -> None:
        """Update spatial memory with current detections. Args: detections_list: List of detection dictionaries."""
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
    
    def _plan_path(self, detections_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Plan navigation path if navigation scenario. Args: detections_list: List of detection dictionaries Returns: Path info dictionary or None."""
        path_info = None
        if self.current_scenario == 'navigation':
            path_result = self.path_planner.plan_path(
                detections=detections_list,
                target_direction='forward'
            )
            if path_result is not None:
                # Convert PathSuggestion to dict if needed.
                if hasattr(path_result, '__dict__'):
                    path_info = path_result.__dict__
                else:
                    path_info = {'path': str(path_result)}
        return path_info
    
    def _schedule_outputs(self, detections_list: List[Dict[str, Any]], outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule cross-modal outputs (voice, haptic, visual)."""
        model_outputs: Dict[str, Any] = {}
        urgency_scores = outputs.get('urgency_scores')
        uncertainty = outputs.get('uncertainty')
        if urgency_scores is not None:
            model_outputs['urgency_scores'] = urgency_scores
        if uncertainty is not None:
            model_outputs['uncertainty'] = uncertainty
        
        scheduled_outputs = self.scheduler.schedule_outputs(
            detections=detections_list,
            model_outputs=model_outputs,
            timestamp=time.time()
        )
        # Convert list to dict for consistency.
        if isinstance(scheduled_outputs, list):
            return {'outputs': scheduled_outputs, 'count': len(scheduled_outputs)}
        return scheduled_outputs if isinstance(scheduled_outputs, dict) else {'outputs': scheduled_outputs}
    
    def _render_overlay(self, image: Image.Image, detections_list: List[Dict[str, Any]], ocr_results: List[Dict[str, Any]], path_info: Optional[Dict[str, Any]]) -> Optional[str]:
        """Render visual overlays on image."""
        try:
            urgency_scores = None
            if detections_list and 'urgency' in detections_list[0]:
                urgency_scores = np.array([det.get('urgency', 0) for det in detections_list])
            
            overlay_image = self.overlay_engine.create_overlay(
                base_image=image,
                detections=detections_list,
                urgency_scores=urgency_scores,
                text_regions=ocr_results
            )
            
            # Convert to base64 for web display.
            buffered = BytesIO()
            overlay_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            logger.warning(f"Overlay rendering error: {e}")
            return None
    
    def _start_async_workers(self) -> None:
        """Start background workers for async voice and haptic processing."""
        def voice_worker():
            self._voice_worker_running = True
            consecutive_failures = 0
            max_failures = 5
            
            while self._voice_worker_running:
                try:
                    priority, message = self.voice_queue.get(timeout=0.1)
                    if message is None:
                        break
                    try:
                        self.voice_feedback.speak_custom(message, priority=priority)
                        consecutive_failures = 0
                    except (OSError, IOError) as e:
                        consecutive_failures += 1
                        logger.error(f"Voice hardware error: {e}")
                        if consecutive_failures >= max_failures:
                            break
                    except Exception as e:
                        logger.error(f"Voice processing error: {e}")
                        consecutive_failures += 1
                        if consecutive_failures >= max_failures:
                            break
                    self.voice_queue.task_done()
                except Exception:
                    pass  # Queue timeout or other non-critical errors.
        
        def haptic_worker():
            self._haptic_worker_running = True
            consecutive_failures = 0
            max_failures = 5
            
            while self._haptic_worker_running:
                try:
                    priority, message = self.haptic_queue.get(timeout=0.1)
                    if message is None:
                        break
                    try:
                        # Message format: (pattern, intensity) tuple.
                        if isinstance(message, tuple):
                            pattern, intensity = message
                            self.haptic_feedback.trigger(pattern, intensity=intensity)
                        else:
                            self.haptic_feedback.trigger(message)
                        consecutive_failures = 0
                    except (OSError, IOError) as e:
                        consecutive_failures += 1
                        logger.error(f"Haptic hardware error: {e}")
                        if consecutive_failures >= max_failures:
                            break
                    except Exception as e:
                        logger.error(f"Haptic processing error: {e}")
                        consecutive_failures += 1
                        if consecutive_failures >= max_failures:
                            break
                    self.haptic_queue.task_done()
                except Exception:
                    pass  # Queue timeout or other non-critical errors.
        
        # Start worker threads (daemon=True ensures they stop when main thread exits)
        self.voice_thread = threading.Thread(target=voice_worker, daemon=True)
        self.haptic_thread = threading.Thread(target=haptic_worker, daemon=True)
        self.voice_thread.start()
        self.haptic_thread.start()
    
    def shutdown(self) -> None:
        """Gracefully shutdown async workers."""
        self._voice_worker_running = False
        self._haptic_worker_running = False
        # Signal workers to stop - format: (message, priority)
        try:
            self.voice_queue.put((None, MessagePriority.LOW), block=False)
        except Exception:
            pass
        try:
            self.haptic_queue.put((None, MessagePriority.LOW), block=False)
        except Exception:
            pass
        # Wait for threads to finish (with timeout)
        if hasattr(self, 'voice_thread'):
            self.voice_thread.join(timeout=1.0)
        if hasattr(self, 'haptic_thread'):
            self.haptic_thread.join(timeout=1.0)
    
    def _queue_outputs(self, scene_description: str, outputs: Dict[str, Any], detections_list: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Queue voice and haptic outputs asynchronously."""
        voice_announcements = []
        haptic_patterns = []
        
        if scene_description:
            # Queue format: (message, priority) for PriorityQueue.
            self.voice_queue.put((scene_description, MessagePriority.NORMAL))
            voice_announcements.append(scene_description)
        
        urgency_scores = outputs.get('urgency_scores', torch.zeros(1, 4))
        if urgency_scores.numel() > 0:
            urgency_level = int(urgency_scores.argmax(dim=1).item())
            warning_threshold = config.urgency_warning_threshold
            if urgency_level >= warning_threshold:
                # Queue format: (message, priority)
                self.voice_queue.put((f"Warning: High urgency detected", MessagePriority.CRITICAL))
                voice_announcements.append(f"Warning: High urgency detected")
                # Haptic format: (message, priority) where message is (pattern, intensity) tuple.
                haptic_msg = (HapticPattern.LONG_PULSE, config.haptic_intensity_high)
                self.haptic_queue.put((haptic_msg, MessagePriority.CRITICAL))
                haptic_patterns.append({'pattern': 'long_pulse', 'intensity': config.haptic_intensity_high})
            elif len(detections_list) > 0:
                # Queue format: (message, priority)
                haptic_msg = (HapticPattern.MICRO_PULSE, config.haptic_intensity_low)
                self.haptic_queue.put((haptic_msg, MessagePriority.NORMAL))
                haptic_patterns.append({'pattern': 'micro_pulse', 'intensity': config.haptic_intensity_low})
        
        return voice_announcements, haptic_patterns
    
    def process_frame(
        self,
        image: Image.Image,
        audio_features: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Process a single frame through the complete MaxSight pipeline."""
        start_time = time.perf_counter()
        tracker = PipelineLatencyTracker()
        
        # 1. Preprocessing.
        tracker.start_stage('preprocess')
        image_tensor = self._preprocess_image(image)
        tracker.end_stage()
        
        # 2. Model inference.
        tracker.start_stage('model')
        outputs, inference_time = self._run_inference(image_tensor, audio_features)
        tracker.end_stage()
        
        # 3. Post-process detections + priority budget + alert cooldown.
        tracker.start_stage('postprocess')
        detections_list = self._postprocess_outputs(outputs, confidence_threshold=config.confidence_threshold)
        if not hasattr(self, 'priority_filter'):
            self.priority_filter = PriorityBudgetFilter(max_alerts_per_frame=config.max_alerts_per_frame)
            self.alert_cooldown = AlertCooldownFilter(cooldown_frames=config.alert_cooldown_frames)
        detections_list = self.priority_filter.filter_alerts(detections_list)
        detections_list = self.alert_cooldown.filter_alerts(detections_list, frame_id=None)
        tracker.end_stage()
        
        # 4. OCR text detection.
        ocr_results = self._run_ocr(image, outputs)
        
        # 5. Description generation.
        scene_description = self._generate_description(detections_list, outputs, ocr_results)
        
        # 6. Spatial memory update.
        self._update_memory(detections_list)
        
        # 7. Path planning.
        path_info = self._plan_path(detections_list)
        
        # 8. Output scheduling.
        scheduled_outputs = self._schedule_outputs(detections_list, outputs)
        
        # 9. Therapy integration.
        therapy_feedback = None
        if self.session_active and detections_list:
            target_objects = [det.get('class_name', 'object') for det in detections_list[:3]]
            therapy_feedback = self.therapy.create_attention_task(
                scene_description=scene_description or "Scene with objects",
                target_objects=target_objects,
                    difficulty=config.therapy_difficulty
            )
        
        breakdown_so_far = tracker.get_breakdown()
        total_so_far_ms = float(breakdown_so_far.get('total_ms', 0.0))
        skip_non_critical = total_so_far_ms > 200.0
        try:
            import psutil
            if psutil.cpu_percent(interval=None) > 80.0:
                skip_non_critical = True
        except Exception:
            pass
        
        # 10. Generate overlays.
        if not skip_non_critical:
            tracker.start_stage('overlay')
            overlay_image_b64 = self._render_overlay(image, detections_list, ocr_results, path_info)
            tracker.end_stage()
        else:
            overlay_image_b64 = None
        
        # 11. Queue outputs (voice and haptic)
        if not skip_non_critical:
            tracker.start_stage('audio')
            voice_announcements, haptic_patterns = self._queue_outputs(scene_description, outputs, detections_list)
            tracker.end_stage()
        else:
            voice_announcements, haptic_patterns = [], []
        
        pipeline_breakdown = tracker.get_breakdown()
        
        # Update statistics.
        self.stats['frames_processed'] += 1
        self.stats['total_inference_time'] += inference_time
        self.stats['total_detections'] += len(detections_list)
        self.stats['avg_latency_ms'] = (self.stats['total_inference_time'] / 
                                        self.stats['frames_processed'] * 1000)
        
        total_time = time.perf_counter() - start_time
        
        # Extract 3 perspectives.
        reasoning_trace = self._extract_reasoning_trace(outputs, detections_list)
        final_judgment = self._extract_final_judgment(outputs, detections_list)
        
        # Shape response based on output mode.
        result = self._shape_response(
            detections_list=detections_list,
            outputs=outputs,
            scene_description=scene_description,
            ocr_results=ocr_results,
            voice_announcements=voice_announcements,
            haptic_patterns=haptic_patterns,
            path_info=path_info,
            therapy_feedback=therapy_feedback,
            overlay_image_b64=overlay_image_b64,
            inference_time_ms=inference_time * 1000,
            total_time_ms=total_time * 1000,
            reasoning_trace=reasoning_trace,
            final_judgment=final_judgment,
            scheduled_outputs=scheduled_outputs
        )
        result['pipeline_breakdown'] = pipeline_breakdown
        
        # Save baseline output for regression testing (first frame only)
        self._save_baseline_output(result)
        
        return result
    
    def _extract_reasoning_trace(self, outputs: Dict[str, Any], detections_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract model reasoning trace showing how predictions are made. Returns: Dictionary with reasoning trace information."""
        trace = {
            'feature_extraction': {},
            'attention_weights': {},
            'confidence_scores': {},
            'decision_path': []
        }
        
        # Extract feature statistics.
        scene_emb = outputs.get('scene_embedding')
        if scene_emb is not None:
            trace['feature_extraction'] = {
                'scene_embedding_dim': scene_emb.shape[-1] if scene_emb.numel() > 0 else 0,
                'embedding_norm': float(scene_emb.norm().item()) if scene_emb.numel() > 0 else 0.0
            }
        
        # Extract attention/confidence information.
        objectness = outputs.get('objectness', torch.zeros(1, 196))
        if objectness.numel() > 0:
            trace['confidence_scores'] = {
                'max_objectness': float(objectness.max().item()),
                'mean_objectness': float(objectness.mean().item()),
                'num_high_confidence': int((objectness > 0.5).sum().item())
            }
        
        # Extract urgency reasoning.
        urgency_scores = outputs.get('urgency_scores', torch.zeros(1, 4))
        if urgency_scores.numel() > 0:
            urgency_probs = torch.softmax(urgency_scores, dim=-1)[0]
            trace['attention_weights'] = {
                'urgency_levels': urgency_probs.cpu().tolist(),
                'predicted_urgency': int(urgency_scores.argmax(dim=1).item())
            }
        
        # Build decision path.
        if detections_list:
            top_detections = sorted(detections_list, key=lambda x: x.get('confidence', 0), reverse=True)[:5]
            trace['decision_path'] = [
                {
                    'step': i+1,
                    'object': det.get('class_name', 'unknown'),
                    'confidence': round(det.get('confidence', 0), 3),
                    'urgency': det.get('urgency', 0),
                    'distance': det.get('distance', 0)
                }
                for i, det in enumerate(top_detections)
            ]
        
        return trace
    
    def _extract_final_judgment(self, outputs: Dict[str, Any], detections_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract final weighted judgment with confidence scores. Returns: Dictionary with final judgment information."""
        # Compute weighted scores.
        urgency_scores = outputs.get('urgency_scores', torch.zeros(1, 4))
        urgency_level = int(urgency_scores.argmax(dim=1).item()) if urgency_scores.numel() > 0 else 0
        urgency_confidence = float(torch.softmax(urgency_scores, dim=-1)[0].max().item()) if urgency_scores.numel() > 0 else 0.0
        
        objectness = outputs.get('objectness', torch.zeros(1, 196))
        max_objectness = float(objectness.max().item()) if objectness.numel() > 0 else 0.0
        
        # Weighted final score.
        final_score = (urgency_confidence * 0.6 + max_objectness * 0.4)
        
        # Top detections with weights.
        top_detections = sorted(detections_list, key=lambda x: x.get('confidence', 0), reverse=True)[:3]
        weighted_detections = [
            {
                'object': det.get('class_name', 'unknown'),
                'confidence': round(det.get('confidence', 0), 3),
                'weight': round(det.get('confidence', 0) * (1 + det.get('urgency', 0) * 0.2), 3),
                'urgency': det.get('urgency', 0)
            }
            for det in top_detections
        ]
        
        return {
            'final_score': round(final_score, 3),
            'urgency_level': urgency_level,
            'urgency_confidence': round(urgency_confidence, 3),
            'objectness_confidence': round(max_objectness, 3),
            'num_detections': len(detections_list),
            'weighted_detections': weighted_detections,
            'decision': 'high_alert' if urgency_level >= 3 else 'moderate_alert' if urgency_level >= 2 else 'normal'
        }
    
    def _shape_response(
        self,
        detections_list: List[Dict[str, Any]],
        outputs: Dict[str, Any],
        scene_description: str,
        ocr_results: List[Dict[str, Any]],
        voice_announcements: List[str],
        haptic_patterns: List[Dict[str, Any]],
        path_info: Optional[Dict[str, Any]],
        therapy_feedback: Optional[Dict[str, Any]],
        overlay_image_b64: Optional[str],
        inference_time_ms: float,
        total_time_ms: float,
        reasoning_trace: Dict[str, Any],
        final_judgment: Dict[str, Any],
        scheduled_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Shape response based on output mode. Patient mode: minimal, actionable only Clinician mode: adds metrics and component breakdown Dev mode: full debug information."""
        # Extract urgency for patient safety.
        urgency_scores = outputs.get('urgency_scores', torch.zeros(1, 4))
        urgency_level = int(urgency_scores.argmax(dim=1).item()) if urgency_scores.numel() > 0 else 0
        
        # Determine severity.
        if urgency_level >= 3:
            severity = Severity.CRITICAL
        elif urgency_level >= 2:
            severity = Severity.HAZARD
        elif urgency_level >= 1:
            severity = Severity.WARNING
        else:
            severity = Severity.INFO
        
        # Compute confidence.
        objectness = outputs.get('objectness', torch.zeros(1, 196))
        avg_confidence = objectness.max().item() if objectness.numel() > 0 else 0.0
        
        # Patient mode: minimal, calm, actionable.
        if self.output_mode == OutputMode.PATIENT:
            # Only top hazard + one instruction.
            top_hazards = [d for d in detections_list if d.get('urgency', 0) >= 2]
            if top_hazards:
                message = f"{top_hazards[0]['class_name']} detected"
            elif scene_description:
                # Truncate to first sentence.
                message = scene_description.split('.')[0] + '.'
            else:
                message = "Scene clear"
            
            return {
                'mode': 'patient',
                'severity': severity.value,
                'message': message,
                'confidence': round(avg_confidence, 2),
                'cooldown_applied': False,
                'overlay_image': overlay_image_b64,
                'stats': {
                    'frames_processed': self.stats['frames_processed'],
                    'avg_latency_ms': self.stats['avg_latency_ms'],
                    'total_detections': len(detections_list)
                },
                # 3 Perspectives.
                'perspectives': {
                    'user_view': {
                        'overlay_image': overlay_image_b64,
                        'scene_description': scene_description,
                        'voice_announcements': voice_announcements
                    },
                    'model_reasoning': reasoning_trace,
                    'final_judgment': final_judgment
                }
            }
        
        # Clinician mode: adds metrics and breakdown.
        elif self.output_mode == OutputMode.CLINICIAN:
            return {
                'mode': 'clinician',
                'severity': severity.value,
                'message': scene_description or "No description",
                'confidence': round(avg_confidence, 2),
                'cooldown_applied': False,
                
                # Clinician-specific fields.
                'latency_ms': round(inference_time_ms, 1),
                'total_time_ms': round(total_time_ms, 1),
                'inference_time_ms': round(inference_time_ms, 1),
                'num_detections': len(detections_list),
                'num_hazards': len([d for d in detections_list if d.get('urgency', 0) >= 2]),
                'ocr_texts': [r.get('text', '') for r in ocr_results],
                'component_breakdown': {
                    'detections': len(detections_list),
                    'ocr': len(ocr_results),
                    'voice': len(voice_announcements),
                    'haptic': len(haptic_patterns)
                },
                'overlay_image': overlay_image_b64,
                'stats': {
                    'frames_processed': self.stats['frames_processed'],
                    'avg_latency_ms': self.stats['avg_latency_ms'],
                    'total_detections': self.stats['total_detections']
                },
                # 3 Perspectives.
                'perspectives': {
                    'user_view': {
                        'overlay_image': overlay_image_b64,
                        'scene_description': scene_description,
                        'voice_announcements': voice_announcements,
                        'haptic_patterns': haptic_patterns
                    },
                    'model_reasoning': reasoning_trace,
                    'final_judgment': final_judgment
                }
            }
        
        # Dev mode: full information.
        else:
            return {
                'mode': 'dev',
                'severity': severity.value,
                'message': scene_description or "No description",
                'confidence': round(avg_confidence, 2),
                'cooldown_applied': False,
                
                # Dev-specific fields.
                'frame_number': self.stats['frames_processed'],
                'timestamp': time.time(),
                'processing_time_ms': total_time_ms,
                'inference_time_ms': inference_time_ms,
                
                # Full model outputs.
                'detections': detections_list,
                'num_detections': len(detections_list),
                'urgency_scores': urgency_scores[0].cpu().tolist(),
                'distance_zones': outputs['distance_zones'][0].cpu().tolist(),
                'scene_embedding': outputs['scene_embedding'][0].cpu().tolist(),
                
                # OCR results.
                'text_regions': ocr_results,
                'num_text_regions': len(ocr_results),
                
                # Generated content.
                'scene_description': scene_description,
                'scheduled_outputs': scheduled_outputs,
                'voice_announcements': voice_announcements,
                'haptic_patterns': haptic_patterns,
                'path_info': path_info,
                'therapy_feedback': therapy_feedback,
                'overlay_image': overlay_image_b64,
                
                # Statistics.
                'stats': self.stats.copy(),
                
                # Debug info.
                'debug_info': {
                    'condition': self.current_condition,
                    'scenario': self.current_scenario,
                    'session_active': self.session_active
                },
                # 3 Perspectives.
                'perspectives': {
                    'user_view': {
                        'overlay_image': overlay_image_b64,
                        'scene_description': scene_description,
                        'voice_announcements': voice_announcements,
                        'haptic_patterns': haptic_patterns,
                        'scheduled_outputs': scheduled_outputs
                    },
                    'model_reasoning': reasoning_trace,
                    'final_judgment': final_judgment
                }
            }
    
    def _save_baseline_output(self, result: Dict[str, Any]) -> None:
        """Save baseline output for regression testing. Only saves first frame output to establish baseline."""
        if self.stats['frames_processed'] == config.baseline_save_frame:
            try:
                stats = result.get('stats', {})
                baseline = {
                    'frame_number': result.get('frame_number', stats.get('frames_processed', 0)),
                    'num_detections': result.get('num_detections', stats.get('total_detections', 0)),
                    'num_text_regions': result.get('num_text_regions', 0),
                    'processing_time_ms': result.get('processing_time_ms', result.get('total_time_ms', 0)),
                    'inference_time_ms': result.get('inference_time_ms', stats.get('avg_latency_ms', 0)),
                    'scene_description': result.get('scene_description', result.get('message', '')),
                    'urgency_scores': result.get('urgency_scores', []),
                    'stats': stats
                }
                with open(self.baseline_output_path, 'w') as f:
                    json.dump(baseline, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not save baseline output: {e}")




# Web Routes.

@app.route('/')
def index():
    """Main simulator interface."""
    return render_template('simulator.html')


@app.route('/api/init', methods=['POST'])
def api_init():
    """Initialize/create a new session with user settings."""
    try:
        # Apply global rate limiting.
        client_ip = request.remote_addr or 'unknown'
        global_rate_limiter.check_rate_limit(client_ip)
        
        # Validate input.
        data = request.json or {}
        validated_data = validate_init_request(data)
        
        condition = validated_data.get('condition', 'glaucoma')
        scenario = validated_data['scenario']
        output_mode_str = validated_data['output_mode']
        
        # Parse output mode.
        if output_mode_str == 'clinician':
            output_mode = OutputMode.CLINICIAN
        elif output_mode_str == 'dev':
            output_mode = OutputMode.DEV
        else:
            output_mode = OutputMode.PATIENT
        
        # Create new session.
        session_id = registry.create_session(output_mode=output_mode)
        session = registry.get_session(session_id)
        
        if not session:
            return jsonify({'error': 'Failed to create session'}), 500
        
        # Configure session.
        session.set_user_condition(condition)
        session.current_scenario = scenario
        session.session_active = validated_data.get('start_session', False)
        
        if session.session_active:
            session.session_manager.start_session()
        
        return jsonify({
            'status': 'initialized',
            'session_id': session_id,
            'condition': condition,
            'scenario': scenario,
            'output_mode': output_mode_str,
            'session_active': session.session_active,
            'multi_user_enabled': config.multi_user_enabled
        })
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except RateLimitExceededError as e:
        return jsonify({'error': str(e)}), 429
    except Exception as e:
        api_logger.error(f"Error in api_init: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/process', methods=['POST'])
def api_process():
    """Process image through complete pipeline."""
    try:
        # Apply global rate limiting.
        client_ip = request.remote_addr or 'unknown'
        global_rate_limiter.check_rate_limit(client_ip)
        
        # Get session (creates default if single-user mode)
        try:
            session = require_session()
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        
        # Apply per-session rate limiting.
        session_rate_limiter.check_rate_limit(session.session_id, client_ip)
        
        # Get and validate image from request.
        image = None
        if 'image' in request.files:
            image_file = request.files['image']
            
            # Check filename for format hints.
            filename = image_file.filename or ''
            if filename.lower().endswith(('.heic', '.heif')):
                return jsonify({
                    'error': 'HEIC/HEIF format is not supported. Please convert to JPEG or PNG. '
                             'On Mac: Open in Preview > File > Export > Format: JPEG'
                }), 400
            
            # Ensure file pointer is at the beginning.
            image_file.seek(0)
            
            # Read all bytes - ensure we get actual bytes.
            try:
                image_bytes = image_file.read()
            except Exception as e:
                return jsonify({
                    'error': f'Failed to read image file: {str(e)}'
                }), 400
            
            # Ensure we have actual bytes.
            if not image_bytes:
                return jsonify({'error': 'Empty image file received'}), 400
            
            # Ensure it's bytes, not a BytesIO object.
            if hasattr(image_bytes, 'read'):
                # It's a file-like object, read from it.
                image_bytes.seek(0)
                image_bytes = image_bytes.read()
            
            if not isinstance(image_bytes, bytes):
                return jsonify({
                    'error': f'Invalid image data type: {type(image_bytes).__name__}. Expected bytes.'
                }), 400
            
            # Validate and load image.
            try:
                image = validate_image_file(image_bytes)
            except InvalidImageError as e:
                return jsonify({'error': str(e)}), 400
            # Convert to RGB if necessary.
            if image.mode != 'RGB':
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'RGBA':
                    rgb_image.paste(image, mask=image.split()[3])
                else:
                    rgb_image.paste(image)
                image = rgb_image
            else:
                image = image.convert('RGB')
        elif request.is_json and 'image_data' in request.json:
            image_data = request.json['image_data']
            image = validate_image_data(image_data)
            # Convert to RGB if necessary.
            if image.mode != 'RGB':
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'RGBA':
                    rgb_image.paste(image, mask=image.split()[3])
                else:
                    rgb_image.paste(image)
                image = rgb_image
            else:
                image = image.convert('RGB')
        else:
            return jsonify({'error': 'No image provided. Send image file or image_data in JSON.'}), 400
        
        # Get audio features if provided.
        audio_features = None
        if request.is_json and 'audio_features' in request.json:
            try:
                audio_features = np.array(request.json['audio_features'])
            except Exception as e:
                logger.warning(f"Error parsing audio features: {e}")
        
        # Get frame_id for deterministic ordering.
        frame_id = None
        if request.is_json and 'frame_id' in request.json:
            try:
                frame_id = int(request.json['frame_id'])
            except (ValueError, TypeError):
                pass
        elif 'frame_id' in request.form:
            try:
                frame_id = int(request.form['frame_id'])
            except (ValueError, TypeError):
                pass
        
        # Process frame (session.process_frame handles locking internally)
        try:
            result = session.process_frame(image, audio_features, frame_id=frame_id)
            
            # Add degraded mode status to response.
            degraded_status = session.degraded_state.get_status()
            result['degraded_status'] = degraded_status
            
            # Add resource usage info.
            result['resource_usage'] = {
                'spatial_memory_count': session._spatial_memory_count,
                'memory_usage_mb': session._memory_usage_mb,
                'queue_dropped_voice': session.voice_queue.get_dropped_count(),
                'queue_dropped_haptic': session.haptic_queue.get_dropped_count()
            }
        except ValidationError as e:
            return jsonify({'error': str(e)}), 400
        except RateLimitExceededError as e:
            return jsonify({'error': str(e)}), 429
        except Exception as e:
            api_logger.error(f"Error processing frame in session {session.session_id}: {e}", exc_info=True)
            return jsonify({
                'error': f'Error processing image: {str(e)}',
                'error_type': type(e).__name__
            }), 500
        
        # Overlay image is already in result from process_frame. If overlay_image is None, fallback to original image.
        if not result.get('overlay_image'):
            try:
                image_buffer = BytesIO()
                image.save(image_buffer, format='PNG')
                image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
                result['overlay_image'] = f"data:image/png;base64,{image_base64}"
            except Exception as e:
                logger.warning(f"Error creating fallback overlay: {e}")
        
        # Add original image for comparison.
        try:
            original_buffer = BytesIO()
            image.save(original_buffer, format='PNG')
            original_base64 = base64.b64encode(original_buffer.getvalue()).decode('utf-8')
            result['original_image'] = f"data:image/png;base64,{original_base64}"
        except Exception as e:
            logger.warning(f"Error creating original image: {e}")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Unexpected error in api_process: {e}", exc_info=True)
        return jsonify({
            'error': f'Unexpected error: {str(e)}',
            'error_type': type(e).__name__
        }), 500


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
    """Get current statistics for session."""
    try:
        session = require_session()
        return jsonify(session.stats)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/session/start', methods=['POST'])
def api_session_start():
    """Start therapy session."""
    try:
        session = require_session()
        with session.lock:
            session.session_active = True
            session.session_manager.start_session()
        return jsonify({
            'status': 'session_started',
            'session_id': session.session_id
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/session/stop', methods=['POST'])
def api_session_stop():
    """Stop therapy session."""
    try:
        session = require_session()
        with session.lock:
            session.session_active = False
            session_summary = session.session_manager.end_session()
        return jsonify({
            'status': 'session_stopped',
            'session_id': session.session_id,
            'summary': session_summary
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/session/status', methods=['GET'])
def api_session_status():
    """Get session status."""
    try:
        session = require_session()
        return jsonify({
            'session_id': session.session_id,
            'active': session.session_active,
            'stats': session.stats,
            'output_mode': session.output_mode.value,
            'condition': session.current_condition,
            'scenario': session.current_scenario,
            'degraded_status': session.degraded_state.get_status(),
            'aborted': session._aborted
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/session/abort', methods=['POST'])
def api_session_abort():
    """Hard kill switch - immediately stop all outputs for a session. Used for patient panic, clinician override, or system malfunction."""
    try:
        session = require_session()
        session.abort()
        logger.warning(f"Session {session.session_id} aborted via API")
        return jsonify({
            'session_id': session.session_id,
            'status': 'aborted',
            'message': 'All outputs stopped, queues flushed'
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error aborting session: {e}", exc_info=True)
        return jsonify({'error': f'Error aborting session: {str(e)}'}), 500


@app.route('/api/mode', methods=['POST'])
def api_set_mode():
    """Set output mode (patient/clinician/dev)."""
    try:
        session = require_session()
        data = request.json or {}
        mode_str = data.get('mode', 'patient')
        
        if mode_str == 'clinician':
            mode = OutputMode.CLINICIAN
        elif mode_str == 'dev':
            mode = OutputMode.DEV
        else:
            mode = OutputMode.PATIENT
        
        with session.lock:
            session.output_mode = mode
        
        return jsonify({
            'status': 'mode_updated',
            'session_id': session.session_id,
            'output_mode': mode_str
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check endpoint. Returns system health status including model, sessions, and degraded modes."""
    try:
        health_status = get_health_status()
        
        # Add session registry status.
        with registry.lock:
            active_sessions = len(registry.sessions)
            expired_sessions = sum(1 for s in registry.sessions.values() if s.is_expired())
        
        health_status['sessions'] = {
            'active': active_sessions,
            'expired': expired_sessions,
            'total': active_sessions + expired_sessions
        }
        
        # Add core status.
        core = registry.get_core()
        health_status['core'] = {
            'device': str(core.device),
            'model_loaded': core.model is not None,
            'model_mode': 'eval' if core.model.training == False else 'train'
        }
        
        # Determine overall status.
        overall_status = 'healthy'
        if expired_sessions > active_sessions * 0.5:
            overall_status = 'degraded'
        if core.model is None:
            overall_status = 'unhealthy'
        
        health_status['status'] = overall_status
        
        status_code = 200 if overall_status == 'healthy' else (503 if overall_status == 'unhealthy' else 200)
        
        return jsonify(health_status), status_code
        
    except Exception as e:
        api_logger.error(f"Health check error: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': time.time()
        }), 500


@app.route('/api/sample', methods=['POST', 'GET'])
def api_sample():
    """Process a sample image from the dataset."""
    try:
        # Get dataset directories.
        project_root = Path(__file__).parent.parent.parent
        dataset_dirs = {
            'test': project_root / 'datasets' / 'test' / 'images',
            'val': project_root / 'datasets' / 'val' / 'images',
            'train': project_root / 'datasets' / 'train' / 'images'
        }
        
        if request.method == 'GET':
            # Return list of available sample images.
            available_samples = {}
            for dataset_name, dataset_dir in dataset_dirs.items():
                if dataset_dir.exists():
                    images = sorted(list(dataset_dir.glob('*.jpg')))[:20]  # First 20.
                    available_samples[dataset_name] = [img.name for img in images]
            
            return jsonify({
                'available_samples': available_samples,
                'total_datasets': len([d for d in dataset_dirs.values() if d.exists()])
            })
        
        # POST: Process a sample image. Get session.
        try:
            session = require_session()
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        
        # Get parameters.
        if request.is_json:
            data = request.json
            image_name = data.get('image_name', 'test_000001.jpg')
            dataset = data.get('dataset', 'test')
        else:
            image_name = request.form.get('image_name', 'test_000001.jpg')
            dataset = request.form.get('dataset', 'test')
        
        # Validate dataset.
        if dataset not in dataset_dirs:
            return jsonify({'error': f'Invalid dataset: {dataset}. Must be one of: {list(dataset_dirs.keys())}'}), 400
        
        # Load image from dataset.
        image_path = dataset_dirs[dataset] / image_name
        if not image_path.exists():
            # Find any image in the dataset.
            available = list(dataset_dirs[dataset].glob('*.jpg'))
            if not available:
                return jsonify({
                    'error': f'No images found in {dataset} dataset',
                    'dataset_path': str(dataset_dirs[dataset])
                }), 404
            
            # Use first available image.
            image_path = available[0]
            image_name = image_path.name
            api_logger.info(f"Image {image_name} not found, using {image_name} instead")
        
        # Read and validate image.
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            image = validate_image_file(image_bytes)
            # Convert to RGB.
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            return jsonify({
                'error': f'Failed to load sample image: {str(e)}',
                'image_path': str(image_path)
            }), 400
        
        # Process frame.
        try:
            result = session.process_frame(image, audio_features=None, frame_id=None)
            
            # Add degraded mode status.
            degraded_status = session.degraded_state.get_status()
            result['degraded_status'] = degraded_status
            
            # Add resource usage info.
            result['resource_usage'] = {
                'spatial_memory_count': session._spatial_memory_count,
                'memory_usage_mb': session._memory_usage_mb,
                'queue_dropped_voice': session.voice_queue.get_dropped_count(),
                'queue_dropped_haptic': session.haptic_queue.get_dropped_count()
            }
            
            # Add sample image info.
            result['sample_info'] = {
                'image_name': image_name,
                'dataset': dataset,
                'image_path': str(image_path.relative_to(project_root))
            }
            
        except Exception as e:
            api_logger.error(f"Error processing sample image in session {session.session_id}: {e}", exc_info=True)
            return jsonify({
                'error': f'Error processing sample image: {str(e)}',
                'error_type': type(e).__name__
            }), 500
        
        # Ensure overlay image exists.
        if not result.get('overlay_image'):
            try:
                image_buffer = BytesIO()
                image.save(image_buffer, format='PNG')
                image_base64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
                result['overlay_image'] = f"data:image/png;base64,{image_base64}"
            except Exception as e:
                logger.warning(f"Error creating fallback overlay: {e}")
        
        # Add original image for comparison.
        try:
            original_buffer = BytesIO()
            image.save(original_buffer, format='PNG')
            original_base64 = base64.b64encode(original_buffer.getvalue()).decode('utf-8')
            result['original_image'] = f"data:image/png;base64,{original_base64}"
        except Exception as e:
            logger.warning(f"Error creating original image: {e}")
        
        return jsonify(result)
        
    except Exception as e:
        api_logger.error(f"Sample endpoint error: {e}", exc_info=True)
        return jsonify({
            'error': f'Error in sample endpoint: {str(e)}',
            'error_type': type(e).__name__
        }), 500


@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    """Metrics endpoint. Returns system metrics including performance, usage, and error rates."""
    try:
        metrics_data = metrics.get_metrics()
        
        # Add session-level metrics.
        with registry.lock:
            session_metrics = {}
            for session_id, session in registry.sessions.items():
                session_metrics[session_id] = {
                    'frames_processed': session.stats.get('frames_processed', 0),
                    'total_detections': session.stats.get('total_detections', 0),
                    'avg_latency_ms': session.stats.get('avg_latency_ms', 0.0),
                    'degraded_modes': session.degraded_state.get_status(),
                    'queue_sizes': {
                        'voice': session.voice_queue.qsize(),
                        'haptic': session.haptic_queue.qsize()
                    }
                }
        
        return jsonify({
            'system_metrics': metrics_data,
            'session_metrics': session_metrics,
            'timestamp': time.time()
        }), 200
        
    except Exception as e:
        api_logger.error(f"Metrics error: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'timestamp': time.time()
        }), 500


# Global error handlers for consistent error responses.
@app.errorhandler(400)
def bad_request(error):
    """Handle 400 Bad Request errors."""
    return jsonify({
        'error': 'Bad Request',
        'message': str(error) if hasattr(error, 'description') and error.description else 'Invalid request',
        'status_code': 400
    }), 400


@app.errorhandler(404)
def not_found(error):
    """Handle 404 Not Found errors."""
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested resource was not found',
        'status_code': 404
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 Internal Server errors."""
    # Log detailed error server-side.
    log_error(error, {'endpoint': request.path, 'method': request.method})
    # Return sanitized error to client.
    error_response = sanitize_error(error, debug=config.debug)
    error_response['status_code'] = 500
    return jsonify(error_response), 500


@app.errorhandler(RateLimitExceededError)
def rate_limit_error(error):
    """Handle rate limit errors."""
    return jsonify({
        'error': 'Rate Limit Exceeded',
        'message': str(error),
        'status_code': 429
    }), 429


@app.errorhandler(ValidationError)
def validation_error(error):
    """Handle validation errors."""
    return jsonify({
        'error': 'Validation Error',
        'message': str(error),
        'status_code': 400
    }), 400


@app.errorhandler(SessionNotFoundError)
def session_not_found_error(error):
    """Handle session not found errors."""
    return jsonify({
        'error': 'Session Not Found',
        'message': str(error),
        'status_code': 404
    }), 404


if __name__ == '__main__':
    logger.info("MaxSight Product Simulator")
    logger.info(f"Multi-user mode: {'ENABLED' if config.multi_user_enabled else 'DISABLED (single-user demo)'}")
    logger.info(f"Session timeout: {config.session_timeout_seconds // 60} minutes")
    logger.info("Starting web server...")
    logger.info("Access the simulator at: http://localhost:8002")
    logger.info("Press Ctrl+C to stop")
    
    # WARNING: Flask dev server is NOT production-safe for multi-user scenarios.
    # For production, use Gunicorn with 1 worker and proper WSGI configuration.
    if config.multi_user_enabled:
        logger.warning(
            "WARNING: Running multi-user mode with Flask dev server is unsafe. "
            "Use Gunicorn for production: gunicorn -w 1 -t 120 tools.simulation.web_simulator:app"
        )
    
    try:
        app.run(host=config.host, port=config.port, debug=config.debug, threaded=True)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        registry.shutdown()
        logger.info("Shutdown complete")







