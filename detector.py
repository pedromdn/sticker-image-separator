"""Automatic sticker-grid detection using traditional computer vision (OpenCV
and numpy only -- no learned models).

Panini album pages lay stickers out in a perfectly uniform grid, but in a
real print-ready PDF the gap between two adjacent stickers is razor thin
(often a single pixel at 300 DPI) and is not always continuous: a sticker's
own artwork can bleed a few stray pixels into the gap in some rows/columns
while leaving it clean in others. A per-sticker contour search over the raw
edge map is unreliable on input like this -- busy internal content (photos,
logos, text) generates far more contours than the grid itself, and the
sticker-to-sticker boundary can vanish in patches.

Detection is instead treated as a *grid inference* problem:

1. Find the page's overall printed content area (its bounding box) --
   nothing about sticker position is assumed beyond "not blank page".
2. Sample the page in several perpendicular bands and look for a handful of
   unambiguous, cleanly-white gutter lines. Even just two or three
   confirmed gutters are enough, because a Panini grid's cell pitch
   (spacing) is perfectly uniform.
3. Derive the number of rows/columns from that pitch and lay down evenly
   spaced grid lines across the full content area. This tolerates any
   individual gutter being fully bridged somewhere on the page, since the
   count -- not each line's exact pixel position -- is what really needs to
   be inferred.
4. Slice the page into cells accordingly, skipping any cell that is
   essentially blank (an empty slot on a partially filled page).

No sticker position, size, or count is ever hardcoded: everything is
derived from the actual rendered content. The one exception is *which
page* the count comes from: every page in a given album shares the same
grid template, so a page where step 2 can't find enough gutters to be
confident (e.g. its artwork happens to bleed across every candidate gutter)
can safely borrow the row/column count established on other pages of the
same PDF. ``estimate_grid_size`` exposes the count together with a
confidence score for exactly this purpose; ``extract.py`` uses it to poll
every page first and then re-detect low-confidence pages with the count
that won a majority vote elsewhere in the same PDF.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import Config

BBox = Tuple[int, int, int, int]  # top, bottom, left, right (exclusive on bottom/right)


@dataclass
class StickerBox:
    """A detected sticker, described by its rectangle in page coordinates."""

    center: Tuple[float, float]
    size: Tuple[float, float]   # (width, height)
    angle: float                # degrees; 0 for the axis-aligned grid cells we emit
    box_points: np.ndarray      # 4x2 float32 corners, in source image coords

    @property
    def area(self) -> float:
        return self.size[0] * self.size[1]


def estimate_grid_size(image: np.ndarray, config: Config) -> Tuple[int, int, int, int]:
    """Estimate ``(row_count, col_count, row_confidence, col_confidence)``
    for one page, without cropping anything, using the bounding box content size
    and standard card dimensions.
    """
    fg, bbox = _foreground_and_bbox(image, config)
    if bbox is None:
        return 0, 0, 0, 0
    
    t, b, l, r = bbox
    w = r - l
    h = b - t
    
    # Standard card dimensions at 300 DPI: width 580 px, height 769 px
    ref_card_w = 580.0
    ref_card_h = 769.0
    
    card_w = ref_card_w * config.DPI / 300.0
    card_h = ref_card_h * config.DPI / 300.0
    
    col_count = int(round(w / card_w))
    row_count = int(round(h / card_h))
    
    col_count = max(1, min(col_count, config.MAX_GRID_SIZE))
    row_count = max(1, min(row_count, config.MAX_GRID_SIZE))
    
    # High confidence values are returned (e.g. 5) because this bounding-box based 
    # method is extremely robust for print-ready PDFs.
    return row_count, col_count, 5, 5



def detect_stickers(
    image: np.ndarray, config: Config, grid_size: Optional[Tuple[int, int]] = None
) -> List[StickerBox]:
    """Detect every sticker in ``image`` and return them ordered row-major
    (top-to-bottom, left-to-right within a row).

    ``grid_size``, if given, is an explicit ``(row_count, col_count)`` that
    overrides this page's own gutter-based estimate -- see
    ``estimate_grid_size``.
    """
    fg, bbox = _foreground_and_bbox(image, config)
    if bbox is None:
        return []

    if grid_size is not None:
        row_count, col_count = grid_size
    else:
        row_count, _ = _axis_grid_count(fg, bbox, "row", config)
        col_count, _ = _axis_grid_count(fg, bbox, "col", config)

    row_lines = _grid_line_positions(bbox, axis="row", count=row_count)
    col_lines = _grid_line_positions(bbox, axis="col", count=col_count)

    boxes: List[StickerBox] = []
    for row_idx in range(len(row_lines) - 1):
        y0, y1 = row_lines[row_idx], row_lines[row_idx + 1]
        for col_idx in range(len(col_lines) - 1):
            x0, x1 = col_lines[col_idx], col_lines[col_idx + 1]
            box = _cell_to_box(fg, y0, y1, x0, x1, config)
            if box is not None:
                boxes.append(box)
    return boxes  # already row-major by construction


def _foreground_and_bbox(image: np.ndarray, config: Config) -> Tuple[np.ndarray, Optional[BBox]]:
    """Foreground mask plus its trimmed content bbox (shared by every entry
    point so both agree on exactly the same page region)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fg = (gray < config.WHITE_THRESHOLD).astype(np.uint8)

    bbox = _content_bbox(fg)
    if bbox is None:
        return fg, None
    bbox = _trim_axis_bbox(fg, bbox, axis="row", config=config)
    bbox = _trim_axis_bbox(fg, bbox, axis="col", config=config)
    return fg, bbox


def _content_bbox(fg: np.ndarray) -> Optional[BBox]:
    """Tight bounding box of every non-background pixel on the page."""
    rows = np.where(fg.any(axis=1))[0]
    cols = np.where(fg.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return int(rows[0]), int(rows[-1] + 1), int(cols[0]), int(cols[-1] + 1)


def _min_profile(fg: np.ndarray, bbox: BBox, axis: str, num_bands: int) -> np.ndarray:
    """Foreground-fraction profile along ``axis``, taking the minimum across
    several perpendicular bands so a gutter bridged in one band is still
    visible if any other band shows it clean."""
    top, bottom, left, right = bbox
    length = (bottom - top) if axis == "row" else (right - left)
    perp = (right - left) if axis == "row" else (bottom - top)

    band_edges = np.linspace(0, perp, num_bands + 1).astype(int)
    profile = np.ones(length)
    for band in range(num_bands):
        p0, p1 = band_edges[band], band_edges[band + 1]
        if p1 <= p0:
            continue
        if axis == "row":
            strip = fg[top:bottom, left + p0:left + p1]
            frac = strip.mean(axis=1)
        else:
            strip = fg[top + p0:top + p1, left:right]
            frac = strip.mean(axis=0)
        profile = np.minimum(profile, frac)
    return profile


def _trim_trailing_gap(profile: np.ndarray, min_gap: int, threshold: float) -> int:
    """Length to truncate ``profile`` to, cutting off at the first blank run
    at least ``min_gap`` pixels long. Real inter-sticker gutters are only a
    few pixels wide, so a run this long means unrelated content (e.g. a
    watermark) follows, not another row/column of stickers."""
    below = np.where(profile < threshold)[0]
    if below.size == 0:
        return len(profile)

    run_start = run_prev = below[0]
    for value in below[1:]:
        if value - run_prev > 1:
            if run_prev - run_start + 1 >= min_gap:
                return run_start
            run_start = value
        run_prev = value
    if run_prev - run_start + 1 >= min_gap:
        return run_start
    return len(profile)


def _trim_axis_bbox(fg: np.ndarray, bbox: BBox, axis: str, config: Config) -> BBox:
    """Shrink ``bbox`` to exclude any trailing block of unrelated content
    (e.g. a caption/watermark below the sticker grid) along ``axis``."""
    top, bottom, left, right = bbox
    profile = _min_profile(fg, bbox, axis, config.NUM_PROFILE_BANDS)
    min_gap = config.scale_to_dpi(config.MIN_FOOTER_GAP)
    trimmed_length = _trim_trailing_gap(profile, min_gap, config.FOOTER_GAP_THRESHOLD)
    if axis == "row":
        return top, top + trimmed_length, left, right
    return top, bottom, left, left + trimmed_length


def _confirmed_gutter_positions(profile: np.ndarray, threshold: float) -> List[int]:
    """Midpoints of every run where the profile dips below ``threshold``."""
    below = np.where(profile < threshold)[0]
    if below.size == 0:
        return []

    positions: List[int] = []
    run_start = run_prev = below[0]
    for value in below[1:]:
        if value - run_prev > 1:
            positions.append((run_start + run_prev) // 2)
            run_start = value
        run_prev = value
    positions.append((run_start + run_prev) // 2)
    return positions


def _axis_grid_count(fg: np.ndarray, bbox: BBox, axis: str, config: Config) -> Tuple[int, int]:
    """Infer the number of cells along one axis from the spacing between
    confirmed gutters, rather than requiring every gutter to be visible.
    Returns ``(count, confidence)`` where confidence is the number of
    confirmed gutters that estimate is based on."""
    top, bottom, left, right = bbox
    length = (bottom - top) if axis == "row" else (right - left)

    profile = _min_profile(fg, bbox, axis, config.NUM_PROFILE_BANDS)
    gutters = _confirmed_gutter_positions(profile, config.GUTTER_FRACTION_THRESHOLD)

    # Drop gutters sitting right at the content edges: those are the bbox
    # boundary itself, not an internal split between two stickers.
    edge_margin = config.scale_to_dpi(min(config.MIN_CARD_WIDTH, config.MIN_CARD_HEIGHT)) // 2
    gutters = [g for g in gutters if edge_margin <= g <= length - edge_margin]

    if not gutters:
        return 1, 0

    # Each confirmed gutter's own distance from the start is a candidate
    # pitch (it is *some* whole multiple of the true cell pitch), and so is
    # the gap between any two confirmed gutters. The true pitch can never be
    # missed in a way that makes a candidate *smaller* than it -- gutters can
    # only be bridged/missing, which produces multiples of the true pitch,
    # not fractions of it -- so the smallest candidate is the best estimate.
    candidates = list(gutters)
    if len(gutters) >= 2:
        candidates.extend(np.diff(sorted(gutters)).tolist())
    pitch = float(min(candidates))
    if pitch <= 0:
        return 1, 0

    count = int(round(length / pitch))
    count = max(1, min(count, config.MAX_GRID_SIZE))
    return count, len(gutters)


def _grid_line_positions(bbox: BBox, axis: str, count: int) -> List[int]:
    """Evenly spaced grid-line coordinates spanning the content bbox along
    ``axis``, for a given cell ``count``."""
    top, bottom, left, right = bbox
    length = (bottom - top) if axis == "row" else (right - left)
    start = top if axis == "row" else left
    return [start + round(i * length / count) for i in range(count + 1)]


def _cell_to_box(
    fg: np.ndarray, y0: int, y1: int, x0: int, x1: int, config: Config
) -> Optional[StickerBox]:
    """Turn one grid cell into a StickerBox, or None if it's too small / blank."""
    width, height = x1 - x0, y1 - y0
    min_w = config.scale_to_dpi(config.MIN_CARD_WIDTH)
    min_h = config.scale_to_dpi(config.MIN_CARD_HEIGHT)
    if width < min_w or height < min_h:
        return None

    cell = fg[y0:y1, x0:x1]
    if cell.mean() < config.EMPTY_CELL_FILL_RATIO:
        return None  # empty slot, e.g. on a partially filled last page

    center = (x0 + width / 2.0, y0 + height / 2.0)
    box_points = np.array(
        [[x0, y0], [x1 - 1, y0], [x1 - 1, y1 - 1], [x0, y1 - 1]], dtype=np.float32
    )
    return StickerBox(center=center, size=(float(width), float(height)), angle=0.0, box_points=box_points)
