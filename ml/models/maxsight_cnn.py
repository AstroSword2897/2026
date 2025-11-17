import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Optional, List

COCO_BASE_CLASSES = [
    'person', 'bicycle', 'car', ',motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign' 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'hair drier', 'toothbrush'
]

ACCESIBILITY_CLASSES = [
    'door', 'door_open', 'door_closed', 'door_handle', 'doorknob', 'doorlock'
    'sliding_door', 'sliding_door_open', 'sliding_door_closed', 'revolving_door',
    'automatic_door',
]
   
def _get_unique_classes(base: List[str], additional: List[str]) -> List[str]:
    seen = set(base)
    result = list(base)
    for cls in additional:
        if cls not in seen:
            result.append(cls)
            seen.add(cls)
    return result
COCO_CLASSES = _get_unique_classes(COCO_BASE_CLASSES, ACCESIBILITY_CLASSES)

UURGENCY_LEVELS = ['safe', 'caution', 'warning', 'danger']
DISTANCE_ZONES = ['near', 'medium', 'far']