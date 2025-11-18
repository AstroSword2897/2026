import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Optional, List

COCO_BASE_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'hair drier', 'toothbrush', 
]

ACCESSIBILITY_CLASSES = [
    'door', 'door_open', 'door_closed', 'door_handle', 'doorknob', 'doorlock',
    'sliding_door', 'sliding_door_open', 'sliding_door_closed', 'revolving_door',
    'automatic_door',

    'stairs', 'staircase', 'stairway', 'stairs_up', 'stairs_down', 'stair_step', 'ramp', 'wheelchair_ramp',
    'access_ramp',
]
   
def _get_unique_classes(base: List[str], additional: List[str]) -> List[str]:
    seen = set(base)
    result = list(base)
    for cls in additional:
        if cls not in seen:
            result.append(cls)
            seen.add(cls)
    return result
COCO_CLASSES = _get_unique_classes(COCO_BASE_CLASSES, ACCESSIBILITY_CLASSES)

URGENCY_LEVELS = ['safe', 'caution', 'warning', 'danger']
DISTANCE_ZONES = ['near', 'medium', 'far']

class SimplifiedFPN(nn.Module):
    def __init__(self, in_channels_list=[256, 512, 1024, 2024], out_channels=256):
            super().__init__()
            self.out_channels = out_channels
            
            self.lateral_convs = nn.ModuleList([
                nn.Sequential(
                     nn.Conv2d(in_ch, out_channels, 1, bias=False),
                     nn.BatchNorm2d(out_channels),
                     nn.ReLU(inplace=True)
                )for in_ch in in_channels_list 
            ])

            self.fpn_convs = nn.ModuleList([
                 nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)  
                 ) for _ in range(len(in_channels_list))
            ])

            def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
                 laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
                 
                 fpn_features = []
                 prev = None
                 for i in range(len(laterals) -1, -1, -1):
                      if prev is not None:
                           prev = F.interpolate(prev, size=laterals[i].shape[2:],
                                                mode='nearest')
                           laterals[i] = laterals[i] + prev

                        fpn_out = self.fpn_convs[i](laterals[i])
                        fpn_features.insert(0, fpn_out)
                        prev = laterals[i]
        
            return fpn_features
    
class MaxSightCNN(nn.Module):
     def __init__(
         self,
         num_classes: int = len(COCO_CLASSES),
         num_urgency_levels: int = 4,
         num_distance_zones: int = 3,
         use_audio: bool = True,
         condition_mode: Optional[str] = None,
         fpn_channels: int = 256,
         detection_threshold: float = 0.5,      
     ):
          super().__init__()

          self.num_classes = num_classes
          self.num_urgency_levels = num_urgency_levels
          self.num_distance_zones = num_distance_zones
          self.use_audio = use_audio
          self.condition_mode = condition_mode
          self.fpn_channels = fpn_channels
          self.detection_threshold = detection_threshold
