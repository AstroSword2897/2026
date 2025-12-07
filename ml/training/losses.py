import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class FocalLoss(nn.Module):
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Use torch.tensor(1.0) instead of literal 1 for proper type inference
        focal_loss = self.alpha * (torch.tensor(1.0, device=pt.device, dtype=pt.dtype) - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class IoULoss(nn.Module):
    
    def __init__(self, loss_type: str = 'iou', reduction: str = 'mean'):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.reduction = reduction
        
        assert self.loss_type in ['iou', 'giou', 'diou', 'ciou'], \
            f"Invalid loss_type: {loss_type}"
    
    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        iou = self._compute_iou(pred_boxes, target_boxes)
        
        if self.loss_type == 'iou':
            loss = torch.tensor(1.0, device=iou.device, dtype=iou.dtype) - iou
        elif self.loss_type == 'giou':
            giou = self._compute_giou(pred_boxes, target_boxes, iou)
            loss = torch.tensor(1.0, device=giou.device, dtype=giou.dtype) - giou
        elif self.loss_type in ['diou', 'ciou']:
            diou_ciou = self._compute_diou_ciou(pred_boxes, target_boxes, iou)
            loss = torch.tensor(1.0, device=diou_ciou.device, dtype=diou_ciou.dtype) - diou_ciou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
    
    def _compute_iou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor
    ) -> torch.Tensor:
        b1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
        b1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
        b1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
        b1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2
        
        b2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
        b2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
        b2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
        b2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2
        
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        
        inter_area = (
            (inter_x2 - inter_x1).clamp(min=0) *
            (inter_y2 - inter_y1).clamp(min=0)
        )
        
        b1_area = boxes1[:, 2] * boxes1[:, 3]
        b2_area = boxes2[:, 2] * boxes2[:, 3]
        union_area = b1_area + b2_area - inter_area
        
        return inter_area / (union_area + 1e-8)
    
    def _compute_giou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
        iou: torch.Tensor
    ) -> torch.Tensor:
        b1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
        b1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
        b1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
        b1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2
        
        b2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
        b2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
        b2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
        b2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2
        
        c_x1 = torch.min(b1_x1, b2_x1)
        c_y1 = torch.min(b1_y1, b2_y1)
        c_x2 = torch.max(b1_x2, b2_x2)
        c_y2 = torch.max(b1_y2, b2_y2)
        
        c_area = (c_x2 - c_x1) * (c_y2 - c_y1)
        
        b1_area = boxes1[:, 2] * boxes1[:, 3]
        b2_area = boxes2[:, 2] * boxes2[:, 3]
        union_area = b1_area + b2_area - iou * (b1_area + b2_area) / (1 + iou)
        
        giou = iou - (c_area - union_area) / (c_area + 1e-8)
        return giou
    
    def _compute_diou_ciou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
        iou: torch.Tensor
    ) -> torch.Tensor:
        center_dist_sq = (
            (boxes1[:, 0] - boxes2[:, 0]) ** 2 +
            (boxes1[:, 1] - boxes2[:, 1]) ** 2
        )
        
        b1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
        b1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
        b1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
        b1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2
        
        b2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
        b2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
        b2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
        b2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2
        
        c_x1 = torch.min(b1_x1, b2_x1)
        c_y1 = torch.min(b1_y1, b2_y1)
        c_x2 = torch.max(b1_x2, b2_x2)
        c_y2 = torch.max(b1_y2, b2_y2)
        
        diagonal_sq = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2
        
        diou = iou - center_dist_sq / (diagonal_sq + 1e-8)
        
        if self.loss_type == 'diou':
            return diou
        
        v = (4 / (torch.pi ** 2)) * torch.pow(
            torch.atan(boxes2[:, 2] / (boxes2[:, 3] + 1e-8)) -
            torch.atan(boxes1[:, 2] / (boxes1[:, 3] + 1e-8)),
            2
        )
        
        alpha = v / (torch.tensor(1.0, device=iou.device, dtype=iou.dtype) - iou + v + 1e-8)
        ciou = diou - alpha * v
        
        return ciou


def assign_targets_to_anchors(
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    anchor_points: torch.Tensor,
    num_classes: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N = anchor_points.shape[0]
    M = gt_boxes.shape[0]
    device = gt_boxes.device
    
    assigned_labels = torch.zeros(N, dtype=torch.long, device=device)
    assigned_boxes = torch.zeros(N, 4, device=device)
    assigned_mask = torch.zeros(N, dtype=torch.float32, device=device)
    
    if M == 0:
        return assigned_labels, assigned_boxes, assigned_mask
    
    for i in range(M):
        gt_box = gt_boxes[i]
        gt_label = gt_labels[i]
        
        x1 = gt_box[0] - gt_box[2] / 2
        y1 = gt_box[1] - gt_box[3] / 2
        x2 = gt_box[0] + gt_box[2] / 2
        y2 = gt_box[1] + gt_box[3] / 2
        
        inside = (
            (anchor_points[:, 0] >= x1) &
            (anchor_points[:, 0] <= x2) &
            (anchor_points[:, 1] >= y1) &
            (anchor_points[:, 1] <= y2)
        )
        
        if inside.any():
            assigned_labels[inside] = gt_label + 1
            assigned_boxes[inside] = gt_box
            assigned_mask[inside] = 1.0
    
    return assigned_labels, assigned_boxes, assigned_mask


class DetectionLoss(nn.Module):
    
    def __init__(
        self,
        num_classes: int,
        classification_weight: float = 1.0,
        localization_weight: float = 5.0,
        objectness_weight: float = 1.0,
        urgency_weight: float = 0.5,
        distance_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        iou_loss_type: str = 'iou'
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.classification_weight = classification_weight
        self.localization_weight = localization_weight
        self.objectness_weight = objectness_weight
        self.urgency_weight = urgency_weight
        self.distance_weight = distance_weight
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.iou_loss = IoULoss(loss_type=iou_loss_type)
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        batch_size = predictions['classifications'].size(0)
        num_locations = predictions['classifications'].size(1)
        device = predictions['classifications'].device
        
        grid_size = int(num_locations ** 0.5)
        y_coords = torch.linspace(0, 1, grid_size, device=device)
        x_coords = torch.linspace(0, 1, grid_size, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        anchor_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        
        cls_loss = torch.tensor(0.0, device=device)
        box_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)
        num_positives = 0
        
        for b in range(batch_size):
            cls_logits = predictions['classifications'][b]
            box_preds = predictions['boxes'][b]
            obj_preds = predictions['objectness'][b]
            
            gt_labels = targets['labels'][b]
            gt_boxes = targets['boxes'][b]
            num_objs = targets.get('num_objects', torch.tensor([len(gt_labels)]))[b]
            
            gt_labels = gt_labels[:num_objs]
            gt_boxes = gt_boxes[:num_objs]
            
            assigned_labels, assigned_boxes, assigned_mask = assign_targets_to_anchors(
                gt_boxes, gt_labels, anchor_points, self.num_classes
            )
            
            pos_mask = assigned_mask > 0
            if pos_mask.any():
                pos_labels = (assigned_labels[pos_mask] - 1).clamp(0, self.num_classes - 1)
                cls_loss += self.focal_loss(cls_logits[pos_mask], pos_labels)
                
                box_loss += self.iou_loss(box_preds[pos_mask], assigned_boxes[pos_mask])
                
                num_positives += pos_mask.sum().item()
            
            obj_loss += self.bce_loss(obj_preds, assigned_mask)
        
        cls_loss = cls_loss / batch_size if num_positives > 0 else cls_loss
        box_loss = box_loss / batch_size if num_positives > 0 else box_loss
        obj_loss = obj_loss / batch_size
        
        urgency_loss = torch.tensor(0.0, device=device)
        if 'urgency' in targets and 'urgency_scores' in predictions:
            urgency_loss = self.ce_loss(
                predictions['urgency_scores'],
                targets['urgency']
            )
        
        distance_loss = torch.tensor(0.0, device=device)
        if 'distance' in targets and 'distance_zones' in predictions and num_positives > 0:
            for b in range(batch_size):
                dist_preds = predictions['distance_zones'][b]
                gt_distances = targets['distance'][b]
                num_objs = targets.get('num_objects', torch.tensor([len(gt_distances)]))[b]
                gt_distances = gt_distances[:num_objs]
                
                if len(gt_distances) > 0:
                    K = min(len(gt_distances), num_locations)
                    top_k_indices = torch.topk(predictions['objectness'][b], K).indices
                    
                    distance_loss += self.ce_loss(
                        dist_preds[top_k_indices],
                        gt_distances[:K]
                    )
            
            distance_loss = distance_loss / batch_size
        
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


MaxSightLoss = DetectionLoss
