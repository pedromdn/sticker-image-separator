"""Deskewing and cropping of individual detected stickers."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import Config
from detector import StickerBox
from utils import expand_quad, order_points


def crop_sticker(image: np.ndarray, box: StickerBox, config: Config) -> np.ndarray:
    """Extract a single sticker from ``image`` as an upright, deskewed crop.

    A perspective warp (rather than a naive axis-aligned crop) is used so
    that any slight rotation is corrected in the same step. The quad is
    first shrunk by ``SAFETY_INSET`` to drop any residual edge bleed from a
    neighboring cell, then re-expanded by ``MARGIN`` so the sticker's own
    border is never clipped.
    """
    ordered = order_points(box.box_points)
    image_shape = image.shape[:2]

    inset = expand_quad(ordered, -config.SAFETY_INSET, image_shape)
    final_quad = expand_quad(inset, config.MARGIN + config.SAFETY_INSET, image_shape)

    tl, tr, br, bl = final_quad
    out_w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    out_h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    out_w, out_h = max(out_w, 1), max(out_h, 1)

    destination = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(final_quad.astype("float32"), destination)
    return cv2.warpPerspective(
        image, matrix, (out_w, out_h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )


def save_sticker(sticker_bgr: np.ndarray, output_path: Path, config: Config) -> None:
    """Persist a cropped sticker (BGR) to disk in ``config.OUTPUT_FORMAT``."""
    rgb = cv2.cvtColor(sticker_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(output_path, format=config.OUTPUT_FORMAT.upper())
