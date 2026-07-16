"""Small, dependency-light helpers shared across the pipeline."""
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:
    from detector import StickerBox


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed, and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 arbitrary points as top-left, top-right, bottom-right, bottom-left.

    ``cv2.boxPoints`` does not guarantee a consistent winding order, which
    would otherwise flip/mirror the perspective-warped crop.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def expand_quad(rect: np.ndarray, margin: float, image_shape: Tuple[int, int]) -> np.ndarray:
    """Push each corner of an ordered quad outward from its centroid by ``margin``
    pixels (negative values shrink it), clamped to stay inside ``image_shape``.

    Moving relative to the centroid -- rather than adding/subtracting a fixed
    offset per axis -- keeps the margin correct even when the quad is rotated.
    """
    h_img, w_img = image_shape
    centroid = rect.mean(axis=0)
    expanded = np.zeros_like(rect)
    for i, point in enumerate(rect):
        direction = point - centroid
        norm = np.linalg.norm(direction)
        unit = direction / norm if norm > 1e-6 else np.zeros_like(direction)
        moved = point + unit * margin
        moved[0] = np.clip(moved[0], 0, w_img - 1)
        moved[1] = np.clip(moved[1], 0, h_img - 1)
        expanded[i] = moved
    return expanded


def draw_debug_overlay(image: np.ndarray, boxes: List["StickerBox"]) -> np.ndarray:
    """Render detected contours, bounding boxes, and sticker numbering for QA."""
    overlay = image.copy()
    for index, box in enumerate(boxes, start=1):
        pts = box.box_points.astype(int)
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

        cx, cy = int(box.center[0]), int(box.center[1])
        cv2.circle(overlay, (cx, cy), 6, (255, 0, 0), -1)

        label = str(index)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.putText(
            overlay, label, (cx - text_w // 2, cy + text_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA,
        )
    return overlay
