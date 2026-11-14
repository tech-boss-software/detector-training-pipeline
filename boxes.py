"""Dependency-free box arithmetic and YOLO-format label parsing."""
from __future__ import annotations

import math


def validate_box(box) -> list[float]:
    """Return a finite, positive-area [xmin, ymin, xmax, ymax] list or raise ValueError."""
    if len(box) != 4:
        raise ValueError("box must be [xmin, ymin, xmax, ymax]")
    box = [float(v) for v in box]
    if any(not math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("box must have finite coordinates and positive area")
    return box


def clip_box(box, width, height):
    """Clip to the image; returns None when nothing visible remains."""
    box = validate_box(box)
    clipped = [max(0.0, min(float(width), box[0])), max(0.0, min(float(height), box[1])),
               max(0.0, min(float(width), box[2])), max(0.0, min(float(height), box[3]))]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def iou(a, b) -> float:
    a, b = validate_box(a), validate_box(b)
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union


def parse_yolo_line(line: str, width: int, height: int, class_count: int):
    """Parse one 'class cx cy w h' line with normalised coordinates into (class_id, xyxy)."""
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"expected 5 fields, got {len(parts)}: {line!r}")
    try:
        class_id = int(parts[0])
        cx, cy, w, h = (float(v) for v in parts[1:])
    except ValueError as error:
        raise ValueError(f"non-numeric label fields: {line!r}") from error
    if not 0 <= class_id < class_count:
        raise ValueError(f"class id {class_id} outside the {class_count} declared classes")
    if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in (cx, cy, w, h)) or w <= 0 or h <= 0:
        raise ValueError(f"normalised coordinates must lie in [0, 1] with positive size: {line!r}")
    box = [(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height]
    # Rounding in annotation tools can push a box a fraction of a pixel outside; clip, never fail.
    clipped = clip_box(box, width, height)
    if clipped is None:
        raise ValueError(f"box has no visible area inside the image: {line!r}")
    return class_id, clipped


def to_yolo_line(class_id: int, box, width: int, height: int) -> str:
    box = validate_box(box)
    cx = (box[0] + box[2]) / 2 / width
    cy = (box[1] + box[3]) / 2 / height
    return f"{int(class_id)} {cx:.6f} {cy:.6f} {(box[2] - box[0]) / width:.6f} {(box[3] - box[1]) / height:.6f}"
