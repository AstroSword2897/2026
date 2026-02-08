"""Hungarian matching for multi-object detection. Aligns predictions to ground truth so we train reliable what/where/urgency outputs for assistive scene understanding and safety cues."""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

_EPS = 1e-8
_MIN_BOX_DIM = 1e-5


def compute_giou_cost(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
    """Cost = 1 - GIoU for box overlap. Accurate box matching ensures hazards and objects are localized correctly for the user."""
    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    gt_x1 = gt_boxes[:, 0] - gt_boxes[:, 2] / 2
    gt_y1 = gt_boxes[:, 1] - gt_boxes[:, 3] / 2
    gt_x2 = gt_boxes[:, 0] + gt_boxes[:, 2] / 2
    gt_y2 = gt_boxes[:, 1] + gt_boxes[:, 3] / 2

    p_x1 = pred_x1.unsqueeze(1)
    p_y1 = pred_y1.unsqueeze(1)
    p_x2 = pred_x2.unsqueeze(1)
    p_y2 = pred_y2.unsqueeze(1)
    g_x1 = gt_x1.unsqueeze(0)
    g_y1 = gt_y1.unsqueeze(0)
    g_x2 = gt_x2.unsqueeze(0)
    g_y2 = gt_y2.unsqueeze(0)

    inter_x1 = torch.max(p_x1, g_x1)
    inter_y1 = torch.max(p_y1, g_y1)
    inter_x2 = torch.min(p_x2, g_x2)
    inter_y2 = torch.min(p_y2, g_y2)
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h

    pred_area = (pred_boxes[:, 2] * pred_boxes[:, 3]).unsqueeze(1)
    gt_area = (gt_boxes[:, 2] * gt_boxes[:, 3]).unsqueeze(0)
    union_area = pred_area + gt_area - inter_area
    iou = inter_area / (union_area + _EPS)

    enclose_x1 = torch.min(p_x1, g_x1)
    enclose_y1 = torch.min(p_y1, g_y1)
    enclose_x2 = torch.max(p_x2, g_x2)
    enclose_y2 = torch.max(p_y2, g_y2)
    enclose_area = (enclose_x2 - enclose_x1).clamp(min=0) * (enclose_y2 - enclose_y1).clamp(min=0)
    giou = iou - (enclose_area - union_area) / (enclose_area + _EPS)

    return 1.0 - giou


def _assert_finite(name: str, t: torch.Tensor) -> None:
    if torch.isnan(t).any() or torch.isinf(t).any():
        n_nan = torch.isnan(t).sum().item()
        n_inf = torch.isinf(t).sum().item()
        raise ValueError(f"Invalid {name}: NaN={n_nan}, Inf={n_inf}")


def _clamp_box_dims(boxes: torch.Tensor, min_dim: float = _MIN_BOX_DIM) -> torch.Tensor:
    """Ensure width/height >= min_dim (non-inplace to preserve graph)."""
    if (boxes[:, 2] >= min_dim).all() and (boxes[:, 3] >= min_dim).all():
        return boxes
    out = boxes.clone()
    out[:, 2] = torch.clamp(out[:, 2], min=min_dim)
    out[:, 3] = torch.clamp(out[:, 3], min=min_dim)
    return out


def compute_matching_cost(
    pred_boxes: torch.Tensor,
    pred_logits: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    lambda_class: float = 1.0,
    lambda_bbox: float = 5.0,
    lambda_giou: float = 2.0,
) -> torch.Tensor:
    """Build cost matrix [num_pred, num_gt] for Hungarian assignment. Cell [i,j] = cost of assigning prediction i to ground truth j."""
    pred_boxes = pred_boxes.clone().float()
    pred_logits = pred_logits.clone().float()
    gt_boxes = gt_boxes.clone().float()

    _assert_finite("pred_boxes", pred_boxes)
    _assert_finite("gt_boxes", gt_boxes)
    _assert_finite("pred_logits", pred_logits)

    gt_boxes = _clamp_box_dims(gt_boxes)
    pred_boxes = _clamp_box_dims(pred_boxes)

    probs = torch.softmax(pred_logits, dim=-1)
    class_cost = -torch.log(probs[:, gt_labels] + _EPS)

    bbox_cost = torch.cdist(
        pred_boxes.unsqueeze(1),
        gt_boxes.unsqueeze(0),
        p=1,
    ).squeeze()

    giou_cost = compute_giou_cost(pred_boxes, gt_boxes)
    total_cost = lambda_class * class_cost + lambda_bbox * bbox_cost + lambda_giou * giou_cost

    _assert_finite("cost matrix", total_cost)
    return total_cost


def _empty_matches(device: torch.device, dtype: torch.dtype = torch.long) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty((2, 0), dtype=dtype, device=device),
        torch.empty((0,), device=device),
    )


def _sanitize_boxes_and_logits(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    pred_logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace NaN/Inf with zeros; clamp width/height and coords to [0,1]."""
    pred_boxes = torch.where(torch.isfinite(pred_boxes), pred_boxes, torch.zeros_like(pred_boxes)).float()
    gt_boxes = torch.where(torch.isfinite(gt_boxes), gt_boxes, torch.zeros_like(gt_boxes)).float()
    pred_logits = torch.where(torch.isfinite(pred_logits), pred_logits, torch.zeros_like(pred_logits)).float()

    pred_boxes = pred_boxes.clone()
    pred_boxes[:, 2] = torch.clamp(pred_boxes[:, 2], min=1e-4)
    pred_boxes[:, 3] = torch.clamp(pred_boxes[:, 3], min=1e-4)
    gt_boxes = gt_boxes.clone()
    gt_boxes[:, 2] = torch.clamp(gt_boxes[:, 2], min=1e-4)
    gt_boxes[:, 3] = torch.clamp(gt_boxes[:, 3], min=1e-4)
    pred_boxes = torch.clamp(pred_boxes, 0.0, 1.0)
    gt_boxes = torch.clamp(gt_boxes, 0.0, 1.0)
    return pred_boxes, gt_boxes, pred_logits


def _greedy_assignment(cost: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Assign each gt to its lowest-cost unused prediction."""
    num_gt = cost.shape[1]
    pred_idx_list: List[int] = []
    gt_idx_list: List[int] = []
    matched_costs_list: List[torch.Tensor] = []
    used_preds: Set[int] = set()

    for gt_i in range(num_gt):
        gt_costs = cost[:, gt_i]
        for rank in gt_costs.argsort():
            pred_i = int(rank.item())
            if pred_i not in used_preds:
                pred_idx_list.append(pred_i)
                gt_idx_list.append(gt_i)
                matched_costs_list.append(gt_costs[pred_i])
                used_preds.add(pred_i)
                break

    if not pred_idx_list:
        return _empty_matches(device)

    pred_idx = torch.tensor(pred_idx_list, dtype=torch.long, device=device)
    gt_idx = torch.tensor(gt_idx_list, dtype=torch.long, device=device)
    matched_costs = torch.stack(matched_costs_list)
    return torch.stack([pred_idx, gt_idx]), matched_costs


def match_predictions_to_gt(
    pred_boxes: torch.Tensor,
    pred_logits: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    lambda_class: float = 1.0,
    lambda_bbox: float = 5.0,
    lambda_giou: float = 2.0,
    use_hungarian: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimal assignment pred<->gt. Returns (indices [2, K], costs [K])."""
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return _empty_matches(pred_boxes.device)

    pred_boxes, gt_boxes, pred_logits = _sanitize_boxes_and_logits(pred_boxes, gt_boxes, pred_logits)

    try:
        cost = compute_matching_cost(
            pred_boxes, pred_logits, gt_boxes, gt_labels,
            lambda_class, lambda_bbox, lambda_giou,
        )
    except ValueError as e:
        logger.debug("Cost computation failed after sanitization: %s. Returning empty matches.", e)
        return _empty_matches(pred_boxes.device)

    if use_hungarian:
        try:
            from scipy.optimize import linear_sum_assignment
            cost_np = cost.detach().cpu().numpy()
            if np.isfinite(cost_np).all():
                pred_indices, gt_indices = linear_sum_assignment(cost_np)
                pred_idx = torch.tensor(pred_indices, dtype=torch.long, device=pred_boxes.device)
                gt_idx = torch.tensor(gt_indices, dtype=torch.long, device=pred_boxes.device)
                matched_costs = cost[pred_idx, gt_idx]
                return torch.stack([pred_idx, gt_idx]), matched_costs
            logger.warning("Cost matrix has non-finite values, falling back to greedy matching")
        except (ImportError, ValueError) as e:
            logger.warning("Hungarian matching failed (%s), falling back to greedy matching", e)

    return _greedy_assignment(cost, pred_boxes.device)


def _sanitize_sample_predictions(
    pred_boxes_i: torch.Tensor,
    pred_logits_i: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    sample_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replace NaN/Inf in one sample's predictions (clone then column updates for GradNorm)."""
    if torch.isnan(pred_boxes_i).any() or torch.isinf(pred_boxes_i).any():
        logger.warning("Sample %d has NaN/Inf in pred_boxes, sanitizing", sample_idx)
        pred_boxes_i = pred_boxes_i.clone()
        nan_mask = torch.isnan(pred_boxes_i) | torch.isinf(pred_boxes_i)
        default_box = torch.tensor([0.5, 0.5, 0.1, 0.1], device=device, dtype=dtype)
        for j in range(4):
            pred_boxes_i[:, j] = pred_boxes_i[:, j].masked_fill(nan_mask[:, j], default_box[j])
    if torch.isnan(pred_logits_i).any() or torch.isinf(pred_logits_i).any():
        logit_nan_mask = torch.isnan(pred_logits_i) | torch.isinf(pred_logits_i)
        pred_logits_i = pred_logits_i.masked_fill(logit_nan_mask, 0.0)
    return pred_boxes_i, pred_logits_i


def match_batch(
    pred_boxes: torch.Tensor,
    pred_logits: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    lambda_class: float = 1.0,
    lambda_bbox: float = 5.0,
    lambda_giou: float = 2.0,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """One-to-one matching per batch item. Produces consistent training targets across the batch for assistive detection quality."""
    batch_size = pred_boxes.shape[0]
    device = pred_boxes.device
    empty_indices = torch.empty((2, 0), dtype=torch.long, device=device)
    empty_costs = torch.empty((0,), device=device)

    indices_list: List[torch.Tensor] = []
    costs_list: List[torch.Tensor] = []

    for i in range(batch_size):
        valid_gt = (gt_boxes[i, :, 2] > 0) & (gt_boxes[i, :, 3] > 0)

        if torch.isnan(gt_boxes[i]).any() or torch.isinf(gt_boxes[i]).any():
            logger.warning("Sample %d has NaN/Inf in gt_boxes, skipping", i)
            indices_list.append(empty_indices)
            costs_list.append(empty_costs)
            continue

        pred_boxes_i = pred_boxes[i].clone()
        pred_logits_i = pred_logits[i].clone()
        pred_boxes_i, pred_logits_i = _sanitize_sample_predictions(
            pred_boxes_i, pred_logits_i, device, pred_boxes.dtype, i
        )

        if valid_gt.sum() == 0:
            indices_list.append(empty_indices)
            costs_list.append(empty_costs)
            continue

        gt_boxes_valid = gt_boxes[i][valid_gt]
        gt_labels_valid = gt_labels[i][valid_gt]
        indices, costs = match_predictions_to_gt(
            pred_boxes_i, pred_logits_i, gt_boxes_valid, gt_labels_valid,
            lambda_class, lambda_bbox, lambda_giou,
        )
        indices_list.append(indices)
        costs_list.append(costs)

    return indices_list, costs_list


# Build aligned pred/target dicts for loss (GradNorm / MultiHeadLoss)


def _interpret_indices(
    idx: torch.Tensor,
    num_locations: int,
    num_gt: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """From [2, K] indices (pred row, gt row, order may swap), return (pred_idx, gt_idx) or (None, None)."""
    if idx[0].max().item() < num_locations and idx[1].max().item() < num_gt:
        return idx[0], idx[1]
    if idx[1].max().item() < num_locations and idx[0].max().item() < num_gt:
        return idx[1], idx[0]
    return None, None


def build_matched_pred_targets(
    outputs: Dict[str, Any],
    targets: Dict[str, Any],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Run Hungarian matching and return aligned pred/target dicts for loss. Aligns model outputs to ground truth so all assistive heads (objectness, class, box, distance, urgency) get correct supervision."""
    batch_size = outputs["boxes"].size(0)
    num_locations = outputs["boxes"].size(1)
    device = outputs["boxes"].device
    pred_boxes = outputs["boxes"]
    pred_logits = outputs["classifications"]
    gt_boxes = targets["boxes"]
    gt_labels = targets["labels"].float().clamp(min=0).long()

    indices_list, _ = match_batch(pred_boxes, pred_logits, gt_boxes, gt_labels)

    target_objectness = torch.zeros(batch_size, num_locations, device=device, dtype=pred_boxes.dtype)
    matched_pred_cls: List[torch.Tensor] = []
    matched_gt_cls: List[torch.Tensor] = []
    matched_pred_box: List[torch.Tensor] = []
    matched_gt_box: List[torch.Tensor] = []
    matched_pred_dist: List[torch.Tensor] = []
    matched_gt_dist: List[torch.Tensor] = []

    for i in range(batch_size):
        idx = indices_list[i]
        if idx.size(1) == 0:
            continue

        valid_gt = gt_boxes[i, :, 2] > 0
        gt_boxes_valid = gt_boxes[i][valid_gt]
        gt_labels_valid = gt_labels[i][valid_gt]
        num_gt_valid = gt_labels_valid.size(0)

        pred_idx, gt_idx = _interpret_indices(idx, num_locations, num_gt_valid)
        if pred_idx is None:
            continue

        target_objectness[i, pred_idx] = 1.0
        gt_dist_valid = targets["distance"][i][valid_gt].long().clamp(0, 2)

        matched_pred_cls.append(pred_logits[i][pred_idx])
        matched_gt_cls.append(gt_labels_valid[gt_idx])
        matched_pred_box.append(pred_boxes[i][pred_idx])
        matched_gt_box.append(gt_boxes_valid[gt_idx])
        if outputs.get("distance_zones") is not None:
            matched_pred_dist.append(outputs["distance_zones"][i][pred_idx])
            matched_gt_dist.append(gt_dist_valid[gt_idx])

    num_classes = pred_logits.size(-1)
    aligned_pred = {
        "objectness": outputs["objectness"],
        "classification": torch.cat(matched_pred_cls, dim=0) if matched_pred_cls else torch.empty(0, num_classes, device=device),
        "box": torch.cat(matched_pred_box, dim=0) if matched_pred_box else torch.empty(0, 4, device=device),
        "distance": torch.cat(matched_pred_dist, dim=0) if matched_pred_dist else torch.empty(0, 3, device=device),
        "urgency_scores": outputs.get("urgency_scores"),
    }
    aligned_target = {
        "objectness": target_objectness,
        "labels": torch.cat(matched_gt_cls, dim=0) if matched_gt_cls else torch.empty(0, dtype=torch.long, device=device),
        "boxes": torch.cat(matched_gt_box, dim=0) if matched_gt_box else torch.empty(0, 4, device=device),
        "distance": torch.cat(matched_gt_dist, dim=0) if matched_gt_dist else torch.empty(0, dtype=torch.long, device=device),
        "urgency": targets.get("urgency"),
    }
    return aligned_pred, aligned_target






