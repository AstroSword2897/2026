"""
Fixed Loss Functions for MaxSight CNN
Loss functions revisision help to improve the trainability of the MaxSight model

Improvements made and why are:
- Focal Loss
    The focal loss implementation allows weights to work as confidence score for each class checking on accuracy of CNN
- IoU Loss for bounding boxes
    The shape of the object is optimized off of mAP metric and seeing how much loss on overlap can indicate better box regression
- Proper target assignment to anchors
    A simplified matching strategy is used to assign ground truth boxes to anchor locations based on center inclusion.
    This is less computationally expensive than Hungarian matching and works well for dense prediction tasks.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in object detection.
    
    Downweights easy examples and focuses learning on hard examples, which is important
    for object detection where most locations are background. Especially useful for
    MaxSight's 400+ classes where many classes are rare.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Arguments:
            alpha: Weighting factor for rare classes (0.25 means rare classes get 4x weight)
            gamma: Focusing parameter (2.0 = more focus on hard examples)
            reduction: How to aggregate loss ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Computes focal loss: alpha * (1-pt)^gamma * ce_loss, where pt is the probability
        of the true class. This downweights easy examples (high confidence) and focuses
        on hard examples (low confidence).
        
        Arguments:
            inputs: [*, num_classes] logits from model
            targets: [*] class indices
        
        Returns:
            Focal loss value (scalar if reduction='mean'/'sum', tensor if reduction='none')
        """
        # Get base cross-entropy loss for each sample
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Convert to probability: pt = exp(-ce_loss)
        # High pt = confident (easy), low pt = uncertain (hard)
        pt = torch.exp(-ce_loss)
        
        # Apply focal weighting: downweight easy examples, focus on hard ones
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Aggregate loss across samples
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute Intersection over Union (IoU) between two sets of bounding boxes.
    
    IoU = intersection_area / union_area, ranges from 0 (no overlap) to 1 (perfect overlap).
    Used for box regression loss, target assignment, and evaluation metrics.
    
    Arguments:
        boxes1: [N, 4] boxes in center format (cx, cy, w, h), normalized [0, 1]
        boxes2: [N, 4] boxes in center format (cx, cy, w, h), normalized [0, 1]
    
    Returns:
        IoU scores [N] - one per box pair, range [0, 1]
    """
    # Convert center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)
    boxes1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
    boxes1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
    boxes1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
    boxes1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2
    
    boxes2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
    boxes2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
    boxes2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
    boxes2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2
    
    # Find intersection rectangle
    inter_x1 = torch.max(boxes1_x1, boxes2_x1)
    inter_y1 = torch.max(boxes1_y1, boxes2_y1)
    inter_x2 = torch.min(boxes1_x2, boxes2_x2)
    inter_y2 = torch.min(boxes1_y2, boxes2_y2)
    
    # Compute intersection area (clamp negative to 0 for non-overlapping boxes)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                 torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Compute union area
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = boxes1_area + boxes2_area - inter_area
    
    # IoU = intersection / union
    iou = inter_area / (union_area + 1e-8)
    return iou


def assign_targets_to_anchors(
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    anchor_points: torch.Tensor,
    num_classes: int,
    pos_iou_thresh: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Assign ground truth to anchor locations (simplified matching)
    
    Arguments:
        gt_boxes: [M, 4] ground truth boxes (x, y, w, h)
        gt_labels: [M] ground truth class labels
        anchor_points: [N, 2] anchor center points (normalized x, y)
        num_classes: Number of classes
        pos_iou_thresh: IoU threshold for positive assignment
    
    Returns:
        assigned_labels: [N] class for each location (-1=ignore, 0=background, >0=object)
        assigned_boxes: [N, 4] box targets
        assigned_mask: [N] binary mask (1=positive, 0=negative/ignore)
    """
    N = anchor_points.shape[0]
    M = gt_boxes.shape[0]
    
    assigned_labels = torch.zeros(N, dtype=torch.long, device=gt_boxes.device)
    assigned_boxes = torch.zeros(N, 4, device=gt_boxes.device)
    assigned_mask = torch.zeros(N, dtype=torch.float32, device=gt_boxes.device)
    
    if M == 0:
        return assigned_labels, assigned_boxes, assigned_mask
    
    # For each ground truth box, find closest anchor points
    # Simple strategy: assign GT to anchors whose centers are inside the box
    for i in range(M):
        gt_box = gt_boxes[i]  # [4]: (x, y, w, h)
        gt_label = gt_labels[i]
        
        # Box boundaries
        x1 = gt_box[0] - gt_box[2] / 2
        y1 = gt_box[1] - gt_box[3] / 2
        x2 = gt_box[0] + gt_box[2] / 2
        y2 = gt_box[1] + gt_box[3] / 2
        
        # Find anchors inside this box
        inside_mask = (anchor_points[:, 0] >= x1) & (anchor_points[:, 0] <= x2) & \
                      (anchor_points[:, 1] >= y1) & (anchor_points[:, 1] <= y2)
        
        if inside_mask.sum() > 0:
            assigned_labels[inside_mask] = gt_label + 1  # +1 because 0 is background
            assigned_boxes[inside_mask] = gt_box
            assigned_mask[inside_mask] = 1.0
    
    return assigned_labels, assigned_boxes, assigned_mask


class MaxSightLoss(nn.Module):
    """
    Multi-task loss for MaxSight CNN
    
    Improvements:
    - Focal loss for classification (handles imbalance)
    - IoU loss for boxes (simpler and effective)
    - Binary cross-entropy for objectness
    - Proper target assignment
    """
    
    def __init__(
        self,
        num_classes: int = 48,
        classification_weight: float = 1.0,
        localization_weight: float = 5.0,
        objectness_weight: float = 1.0,
        urgency_weight: float = 0.5,
        distance_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.classification_weight = classification_weight
        self.localization_weight = localization_weight
        self.objectness_weight = objectness_weight
        self.urgency_weight = urgency_weight
        self.distance_weight = distance_weight
        
        # Loss functions
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Arguments:
            outputs: Model outputs
                - classifications: [B, N, num_classes]
                - boxes: [B, N, 4]
                - objectness: [B, N]
                - urgency_scores: [B, num_urgency_levels]
                - distance_zones: [B, N, num_distance_zones]
            targets: Ground truth
                - labels: [B, M] class indices (M = max objects per image)
                - boxes: [B, M, 4]
                - urgency: [B] urgency level index
                - distance: [B, M] distance zone indices
                - num_objects: [B] number of valid objects per image
        
        Returns:
            Dictionary with individual and total losses
        """
        batch_size = outputs['classifications'].size(0)
        num_locations = outputs['num_locations']
        device = outputs['classifications'].device
        
        # Generate anchor points (grid centers)
        grid_size = int(num_locations ** 0.5)  # Assume square grid
        y_coords = torch.linspace(0, 1, grid_size, device=device)
        x_coords = torch.linspace(0, 1, grid_size, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        anchor_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # [N, 2]
        
        # Initialize losses (as tensors to ensure type consistency)
        total_cls_loss = torch.tensor(0.0, device=device)
        total_box_loss = torch.tensor(0.0, device=device)
        total_obj_loss = torch.tensor(0.0, device=device)
        num_positives = 0
        
        # Per-image loss computation
        for b in range(batch_size):
            # Get predictions
            cls_logits = outputs['classifications'][b]  # [N, num_classes]
            box_preds = outputs['boxes'][b]  # [N, 4]
            obj_preds = outputs['objectness'][b]  # [N]
            
            # Get targets
            gt_labels = targets['labels'][b]  # [M]
            gt_boxes = targets['boxes'][b]  # [M, 4]
            num_objs = targets.get('num_objects', torch.tensor([len(gt_labels)]))[b]
            
            # Filter valid objects (remove padding)
            gt_labels = gt_labels[:num_objs]
            gt_boxes = gt_boxes[:num_objs]
            
            # Assign targets to anchors
            assigned_labels, assigned_boxes, assigned_mask = assign_targets_to_anchors(
                gt_boxes, gt_labels, anchor_points, self.num_classes
            )
            
            # Classification loss (only on assigned locations)
            if assigned_mask.sum() > 0:
                pos_mask = assigned_mask > 0
                cls_targets = assigned_labels[pos_mask] - 1  # Remove background offset
                cls_targets = torch.clamp(cls_targets, 0, self.num_classes - 1)
                
                cls_loss = self.focal_loss(cls_logits[pos_mask], cls_targets)
                total_cls_loss += cls_loss
                
                # Box loss (only on positive locations)
                box_targets = assigned_boxes[pos_mask]
                box_pred = box_preds[pos_mask]
                
                # IoU loss
                iou = compute_iou(box_pred, box_targets)
                box_loss = (1 - iou).mean()
                total_box_loss += box_loss
                
                num_positives += pos_mask.sum().item()
            
            # Objectness loss (all locations)
            obj_targets = assigned_mask  # 1 for objects, 0 for background
            obj_loss = self.bce_loss(obj_preds, obj_targets)
            total_obj_loss += obj_loss
        
        # Average over batch (all are already tensors)
        if num_positives > 0:
            cls_loss = total_cls_loss / batch_size
            box_loss = total_box_loss / batch_size
        else:
            cls_loss = torch.tensor(0.0, device=device)
            box_loss = torch.tensor(0.0, device=device)
        
        obj_loss = total_obj_loss / batch_size
        
        # Urgency loss (scene-level)
        if 'urgency' in targets:
            urgency_targets = targets['urgency']  # [B]
            urgency_loss = self.ce_loss(outputs['urgency_scores'], urgency_targets)
        else:
            urgency_loss = torch.tensor(0.0, device=device)
        
        # Distance loss (simplified: average over detections)
        if 'distance' in targets and num_positives > 0:
            # Use only positive detections for distance
            distance_loss = torch.tensor(0.0, device=device)
            for b in range(batch_size):
                dist_preds = outputs['distance_zones'][b]  # [N, 3]
                gt_distances = targets['distance'][b]  # [M]
                num_objs = targets.get('num_objects', torch.tensor([len(gt_distances)]))[b]
                gt_distances = gt_distances[:num_objs]
                
                if len(gt_distances) > 0:
                    # Simple: take top-K detections by objectness
                    num_gt = len(gt_distances)  # len() always returns int
                    K = min(num_gt, int(num_locations))
                    obj_scores = outputs['objectness'][b]
                    top_k_indices = torch.topk(obj_scores, K).indices
                    
                    dist_pred_k = dist_preds[top_k_indices]  # [K, 3]
                    dist_targets_k = gt_distances[:K]
                    
                    distance_loss += self.ce_loss(dist_pred_k, dist_targets_k)
            
            distance_loss = distance_loss / batch_size
        else:
            distance_loss = torch.tensor(0.0, device=device)
        
        # Combined loss
        total_loss = (
            self.classification_weight * cls_loss +
            self.localization_weight * box_loss +
            self.objectness_weight * obj_loss +
            self.urgency_weight * urgency_loss +
            self.distance_weight * distance_loss
        )
        
        return {
            'total_loss': total_loss,
            'classification_loss': cls_loss,
            'localization_loss': box_loss,
            'objectness_loss': obj_loss,
            'urgency_loss': urgency_loss,
            'distance_loss': distance_loss,
            'num_positives': torch.tensor(num_positives, device=device, dtype=torch.long)
        }


if __name__ == "__main__":
    print("Testing MaxSight Loss...")
    
    # Dummy data
    batch_size = 2
    num_locations = 196  # 14x14
    num_classes = 48
    
    outputs = {
        'classifications': torch.randn(batch_size, num_locations, num_classes),
        'boxes': torch.rand(batch_size, num_locations, 4) * 0.5 + 0.25,
        'objectness': torch.rand(batch_size, num_locations),
        'urgency_scores': torch.randn(batch_size, 4),
        'distance_zones': torch.randn(batch_size, num_locations, 3),
        'num_locations': num_locations
    }
    
    targets = {
        'labels': torch.randint(0, num_classes, (batch_size, 5)),
        'boxes': torch.rand(batch_size, 5, 4) * 0.5 + 0.25,
        'urgency': torch.randint(0, 4, (batch_size,)),
        'distance': torch.randint(0, 3, (batch_size, 5)),
        'num_objects': torch.tensor([3, 2])
    }
    
    criterion = MaxSightLoss(num_classes=num_classes)
    losses = criterion(outputs, targets)
    
    print("\nLoss values:")
    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item():.4f}")
        else:
            print(f"  {k}: {v}")
    
