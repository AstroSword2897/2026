"""Stage A temporal smoother for MaxSight. EMA smoothing of box and confidence across frames to reduce flicker."""
from typing import List, Dict, Any, Optional
import math


def _get_object_id(det: Dict[str, Any]) -> str:
    """Stable ID from class + approximate position (grid)."""
    cls = det.get("class_name", "unknown")
    box = det.get("box")
    if box is None:
        return cls
    if hasattr(box, "tolist"):
        box = box.tolist()
    # Cx, cy, w, h or x1, y1, x2, y2.
    if len(box) >= 4:
        cx = (box[0] + box[2]) / 2.0 if len(box) == 4 else box[0]
        cy = (box[1] + box[3]) / 2.0 if len(box) == 4 else box[1]
        grid = (int(math.floor(cx * 10)), int(math.floor(cy * 10)))
        return f"{cls}_{grid[0]}_{grid[1]}"
    return cls


def _box_to_list(box: Any) -> List[float]:
    if hasattr(box, "tolist"):
        return box.tolist()
    if isinstance(box, (list, tuple)):
        return list(box)
    return [0.0, 0.0, 0.01, 0.01]


class StageATemporalSmoother:
    """EMA smoothing of detections across frames. Tracks objects by id; smooths box and confidence; drops if not seen for max_age frames."""

    def __init__(self, alpha: float = 0.7, max_age: int = 5):
        self.alpha = alpha
        self.max_age = max_age
        self._history: Dict[str, Dict[str, Any]] = {}
        self._last_seen: Dict[str, int] = {}
        self._frame_count = 0

    def smooth_detections(
        self, detections: List[Dict[str, Any]], frame_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Smooth detection boxes and confidence with EMA. Uses frame_id if provided, else internal frame counter."""
        if frame_id is not None:
            self._frame_count = frame_id
        else:
            self._frame_count += 1
        fid = self._frame_count

        result: List[Dict[str, Any]] = []
        for det in detections:
            oid = _get_object_id(det)
            self._last_seen[oid] = fid

            if oid in self._history:
                prev = self._history[oid]
                # EMA box.
                prev_box = _box_to_list(prev.get("box", [0, 0, 0.01, 0.01]))
                cur_box = _box_to_list(det.get("box", prev_box))
                new_box = [
                    self.alpha * prev_box[i] + (1 - self.alpha) * cur_box[i]
                    for i in range(min(len(prev_box), len(cur_box), 4))
                ]
                if len(new_box) < 4:
                    new_box = new_box + [0.01] * (4 - len(new_box))
                # EMA confidence.
                prev_conf = float(prev.get("confidence", 0.5))
                cur_conf = float(det.get("confidence", 0.5))
                new_conf = self.alpha * prev_conf + (1 - self.alpha) * cur_conf
                out = dict(det)
                out["box"] = new_box
                out["confidence"] = new_conf
                self._history[oid] = out
                result.append(out)
            else:
                self._history[oid] = dict(det)
                result.append(dict(det))

        # Drop stale.
        cutoff = fid - self.max_age
        for oid in list(self._last_seen.keys()):
            if self._last_seen[oid] < cutoff:
                del self._history[oid]
                del self._last_seen[oid]

        return result






