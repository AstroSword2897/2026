"""Shared utility functions for MaxSight simulator. Extracted from duplicated code to reduce duplication."""
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import torchvision.transforms as T


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the best available device for PyTorch. Args: device: Optional device string ('cuda', 'mps', 'cpu') Returns: torch.device instance."""
    if device is None:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device)


def preprocess_image(
    image: Image.Image,
    preprocessor: Optional[Any],
    device: torch.device
) -> torch.Tensor:
    """Preprocess image for model input."""
    if preprocessor:
        preprocessed_tensor = preprocessor(image)
        image_tensor = preprocessed_tensor.unsqueeze(0).to(device)
    else:
        to_tensor = T.ToTensor()
        image_tensor = to_tensor(image).unsqueeze(0).to(device)
    return image_tensor


def postprocess_outputs(
    model: Any,
    outputs: Dict[str, Any],
    confidence_threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """Post-process model outputs to extract detections."""
    detections = model.get_detections(outputs, confidence_threshold=confidence_threshold)
    detections_list: List[Dict[str, Any]] = detections[0] if detections else []
    return detections_list


def run_inference(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    audio_features: Optional[np.ndarray] = None,
    device: torch.device = None
) -> Tuple[Dict[str, Any], float]:
    """Run model inference (thread-safe under no_grad)."""
    import time
    inference_start = time.perf_counter()
    with torch.no_grad():
        if audio_features is not None:
            if device is None:
                device = image_tensor.device
            audio_tensor = torch.from_numpy(audio_features).unsqueeze(0).to(device)
            outputs = model(image_tensor, audio_tensor)
        else:
            outputs = model(image_tensor)
    inference_time = time.perf_counter() - inference_start
    return outputs, inference_time


def extract_urgency_level(outputs: Dict[str, Any]) -> int:
    """Extract urgency level from model outputs. Args: outputs: Model outputs dictionary Returns: Urgency level (0-3)"""
    urgency_score = outputs.get('urgency_scores', torch.zeros(1, 4))
    if urgency_score.numel() > 0:
        return int(urgency_score.argmax(dim=1).item())
    return 0


def prepare_scene_detections(
    detections_list: List[Dict[str, Any]],
    urgency_level: int
) -> List[Dict[str, Any]]:
    """Prepare detection list for scene description generation."""
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
    return scene_detections


