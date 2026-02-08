"""Tiered alert cooldown for MaxSight. Prevents repeated alerts for the same object across frames."""
from typing import List, Dict, Any, Optional
import hashlib


def _object_id(det: Dict[str, Any]) -> str:
    """Stable key: class + bbox hash."""
    cls = det.get("class_name", "unknown")
    box = det.get("box")
    if box is None:
        return cls
    if hasattr(box, "tolist"):
        box = box.tolist()
    # Round to reduce jitter.
    if len(box) >= 4:
        key = (cls, round(box[0], 2), round(box[1], 2), round(box[2], 2), round(box[3], 2))
    else:
        key = (cls,)
    return hashlib.md5(str(key).encode()).hexdigest()[:12]


class AlertCooldownFilter:
    """Filter out detections that were alerted in the last cooldown_frames frames."""

    def __init__(self, cooldown_frames: int = 5):
        self.cooldown_frames = cooldown_frames
        self._last_alert_frame: Dict[str, int] = {}

    def filter_alerts(
        self,
        detections: List[Dict[str, Any]],
        frame_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return only detections that are not in cooldown. frame_id: current frame index; if None, use internal counter."""
        if not hasattr(self, "_frame_counter"):
            self._frame_counter = 0
        if frame_id is not None:
            self._frame_counter = frame_id
        else:
            self._frame_counter += 1
        fid = self._frame_counter

        result: List[Dict[str, Any]] = []
        for det in detections:
            oid = _object_id(det)
            last = self._last_alert_frame.get(oid, -9999)
            if fid - last >= self.cooldown_frames:
                result.append(det)
                self._last_alert_frame[oid] = fid
        return result






