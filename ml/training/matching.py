#Hungarian Matching for bipartite matching between predictions and ground truth


import torch
from typing import Tuple, List


def compute_cost_matrix(
    pred_boxes: torch.Tensor,
    pred_logits: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    cost_class: float = 1.0,
    cost_bbox: float = 5.0,
    cost_giou: float = 2.0
) -> torch.Tensor:
    #Compute cost matrix for Hungarian matching
    num_queries = pred_boxes.shape[0]
    num_gt = gt_boxes.shape[0]
    
    # Classification cost: negative log probability for each GT label
    pred_probs = pred_logits.softmax(dim=-1)
    
    # For each GT label, get the negative log probability from each query
    class_cost = -pred_probs[:, gt_labels].log() 
    
    # Bbox L1 cost
    pred_boxes_expanded = pred_boxes.unsqueeze(1) 
    gt_boxes_expanded = gt_boxes.unsqueeze(0) 
    bbox_cost = torch.cdist(pred_boxes_expanded, gt_boxes_expanded, p=1) 
    
    # GIoU cost
    giou_cost = compute_giou_cost(pred_boxes, gt_boxes)  # [num_queries, num_gt]
    
    # Combined cost
    cost_matrix = (
        cost_class * class_cost +
        cost_bbox * bbox_cost +
        cost_giou * giou_cost
    )
    
    return cost_matrix


def compute_giou_cost(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor
) -> torch.Tensor:
    """
    Compute GIoU cost (1 - GIoU) between two sets of boxes
    
    Arguments:
        boxes1: [N, 4] (x, y, w, h) normalized
        boxes2: [M, 4] (x, y, w, h) normalized
    
    Returns:
        [N, M] cost matrix (1 - GIoU)
    """
    # Convert to (x1, y1, x2, y2)
    boxes1_x1 = boxes1[:, 0]
    boxes1_y1 = boxes1[:, 1]
    boxes1_x2 = boxes1[:, 0] + boxes1[:, 2]
    boxes1_y2 = boxes1[:, 1] + boxes1[:, 3]
    
    boxes2_x1 = boxes2[:, 0]
    boxes2_y1 = boxes2[:, 1]
    boxes2_x2 = boxes2[:, 0] + boxes2[:, 2]
    boxes2_y2 = boxes2[:, 1] + boxes2[:, 3]
    
    # Expand for pairwise computation
    boxes1_x1 = boxes1_x1.unsqueeze(1)  # [N, 1]
    boxes1_y1 = boxes1_y1.unsqueeze(1)
    boxes1_x2 = boxes1_x2.unsqueeze(1)
    boxes1_y2 = boxes1_y2.unsqueeze(1)
    
    boxes2_x1 = boxes2_x1.unsqueeze(0)  # [1, M]
    boxes2_y1 = boxes2_y1.unsqueeze(0)
    boxes2_x2 = boxes2_x2.unsqueeze(0)
    boxes2_y2 = boxes2_y2.unsqueeze(0)
    
    # Intersection
    inter_x1 = torch.max(boxes1_x1, boxes2_x1)
    inter_y1 = torch.max(boxes1_y1, boxes2_y1)
    inter_x2 = torch.min(boxes1_x2, boxes2_x2)
    inter_y2 = torch.min(boxes1_y2, boxes2_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Union
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]  # [N]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]  # [M]
    boxes1_area = boxes1_area.unsqueeze(1)  # [N, 1]
    boxes2_area = boxes2_area.unsqueeze(0)  # [1, M]
    union_area = boxes1_area + boxes2_area - inter_area
    
    # IoU
    #Small epsilon for computaion purposes no (n / 0)
    iou = inter_area / (union_area + 1e-8) 
    
    # Enclosing box
    c_x1 = torch.min(boxes1_x1, boxes2_x1)
    c_y1 = torch.min(boxes1_y1, boxes2_y1)
    c_x2 = torch.max(boxes1_x2, boxes2_x2)
    c_y2 = torch.max(boxes1_y2, boxes2_y2)
    c_area = (c_x2 - c_x1) * (c_y2 - c_y1)
    
    # GIoU
    giou = iou - (c_area - union_area) / (c_area + 1e-8)
    
    # Cost = 1 - GIoU
    return 1.0 - giou


def hungarian_matching(
    pred_boxes: torch.Tensor,
    pred_logits: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    cost_class: float = 1.0,
    cost_bbox: float = 5.0,
    cost_giou: float = 2.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform Hungarian matching between predictions and ground truth
    
    Arguments:
        pred_boxes: [num_queries, 4] predicted boxes
        pred_logits: [num_queries, num_classes] classification logits
        gt_boxes: [num_gt, 4] ground truth boxes
        gt_labels: [num_gt] ground truth class indices
        cost_class: Weight for classification cost
        cost_bbox: Weight for bbox L1 cost
        cost_giou: Weight for GIoU cost
    
    Returns:
        indices: [2, num_matched] (query_idx, gt_idx) pairs
        matched_costs: [num_matched] costs for matched pairs
    """
    # Compute cost matrix
    cost_matrix = compute_cost_matrix(
        pred_boxes, pred_logits, gt_boxes, gt_labels,
        cost_class, cost_bbox, cost_giou
    )
    
    cost_matrix_np = cost_matrix.detach().cpu().numpy()

    query_indices_torch = torch.from_numpy(query_indices).to(pred_boxes.device)
    gt_indices_torch = torch.from_numpy(gt_indices).to(pred_boxes.device)
    
    # Stack indices into [2, num_matched] tensor format
    # Complexity: O(M) - stacks two tensors
    # Relationship: Index formatting - creates standard format for matched pairs
    indices = torch.stack([query_indices_torch, gt_indices_torch])
    
    # Extract matched costs using torch tensor indices
    # Purpose: Get the cost value for each matched pair using the indices. This provides
    #          the matching cost (classification + bbox + GIoU) for each matched prediction.
    # Complexity: O(M) - indexes into cost matrix for each matched pair
    # Relationship: Cost extraction - provides matching costs for analysis/debugging
    matched_costs = cost_matrix[query_indices_torch, gt_indices_torch]
    
    return indices, matched_costs


def batch_hungarian_matching(
    pred_boxes: torch.Tensor,  # [batch, num_queries, 4]
    pred_logits: torch.Tensor,  # [batch, num_queries, num_classes]
    gt_boxes: torch.Tensor,  # [batch, num_gt, 4]
    gt_labels: torch.Tensor,  # [batch, num_gt]
    cost_class: float = 1.0,
    cost_bbox: float = 5.0,
    cost_giou: float = 2.0
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    batch_size = pred_boxes.shape[0]
    indices_list = []
    matched_costs_list = []
    
    for b in range(batch_size):
        # Filter out invalid GT boxes (width == 0)
        valid_mask = gt_boxes[b, :, 2] > 0
        if valid_mask.sum() == 0:
            # No valid GT boxes, skip matching
            indices_list.append(torch.empty((2, 0), dtype=torch.long, device=pred_boxes.device))
            matched_costs_list.append(torch.empty((0,), device=pred_boxes.device))
            continue
        
        valid_gt_boxes = gt_boxes[b][valid_mask]
        valid_gt_labels = gt_labels[b][valid_mask]
        
        indices, costs = hungarian_matching(
            pred_boxes[b],
            pred_logits[b],
            valid_gt_boxes,
            valid_gt_labels,
            cost_class, cost_bbox, cost_giou
        )
        
        indices_list.append(indices)
        matched_costs_list.append(costs)
    
    return indices_list, matched_costs_list

