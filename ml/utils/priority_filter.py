"""Per-frame priority budget filter for MaxSight. Caps alerts per frame to avoid user overload in crowded scenes."""
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Distance ordinal for priority: near=0 -> higher priority, far=2 -> lower.
DISTANCE_ORDINAL = {"near": 0, "medium": 1, "far": 2}


def _distance_ordinal(d: Dict[str, Any]) -> int:
    dist = d.get("distance", "medium")
    if isinstance(dist, str):
        return DISTANCE_ORDINAL.get(dist.lower(), 1)
    return 1


def _priority_score(det: Dict[str, Any]) -> float:
    """Priority = urgency * confidence * (1 / (distance_ordinal + 1))."""
    urgency = int(det.get("urgency", 0)) + 1  # 0-3 -> 1-4.
    confidence = float(det.get("confidence", 0.5))
    do = _distance_ordinal(det) + 1
    return urgency * confidence * (1.0 / do)


class PriorityBudgetFilter:
    """Filter detections to top N by priority score (urgency * confidence * distance)."""

    def __init__(self, max_alerts_per_frame: int = 5):
        self.max_alerts_per_frame = max_alerts_per_frame

    def filter_alerts(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return top max_alerts_per_frame detections by priority score. Handles empty list and lists shorter than N."""
        if not detections:
            return []
        if len(detections) <= self.max_alerts_per_frame:
            return detections
        # Score and sort descending.
        scored = [(det, _priority_score(det)) for det in detections]
        scored.sort(key=lambda x: x[1], reverse=True)
        result = [det for det, _ in scored[: self.max_alerts_per_frame]]
        dropped = len(detections) - len(result)
        if dropped > 0:
            logger.debug(
                "Priority filter dropped %d detections (max_per_frame=%d)",
                dropped,
                self.max_alerts_per_frame,
            )
        return result






