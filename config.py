"""Central configuration for the Panini sticker extractor.

All tunable parameters live here so the detection/cropping pipeline never has
magic numbers scattered across modules. Every field has a sensible default;
override individual fields via the CLI flags in ``extract.py`` or by
constructing ``Config(...)`` directly when using the package as a library.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Tunable parameters for PDF rendering, grid detection, and cropping."""

    # --- PDF -> image rendering -------------------------------------------------
    DPI: int = 300
    REFERENCE_DPI: int = 300  # DPI the size thresholds below were tuned at

    # --- Sticker size sanity filters (pixels, at REFERENCE_DPI; auto-scaled) ---
    # A detected grid cell smaller than this in either dimension is discarded
    # as noise rather than treated as a sticker.
    MIN_CARD_WIDTH: int = 150
    MIN_CARD_HEIGHT: int = 150

    # --- Foreground / background separation -------------------------------------
    # Grayscale value at/above which a pixel is considered blank page
    # background. Real album pages print stickers edge-to-edge with only a
    # hairline (often 1px) white gap between them, so this threshold -- not a
    # fixed page margin -- is what actually separates one sticker from
    # another.
    WHITE_THRESHOLD: int = 245

    # --- Grid line inference -----------------------------------------------------
    # The gap between two stickers can be bridged by a few stray dark pixels
    # (print bleed, antialiasing) in some rows/columns but not others, so a
    # single full-width/height profile is not reliable. Instead the page is
    # sampled in several perpendicular bands and a gutter is confirmed if ANY
    # band shows a clean gap there.
    NUM_PROFILE_BANDS: int = 6
    # Max fraction of foreground pixels, within a band, for a line to count
    # as a confirmed gutter.
    GUTTER_FRACTION_THRESHOLD: float = 0.05
    # Sanity cap on inferred rows/columns per page, guards against a
    # pathological pitch estimate on unexpected input.
    MAX_GRID_SIZE: int = 12
    # A grid cell whose foreground fill ratio is below this is treated as an
    # empty slot (e.g. a partially filled last page) and skipped.
    EMPTY_CELL_FILL_RATIO: float = 0.03

    # --- Trailing content trimming ------------------------------------------------
    # Some pages carry a watermark/caption below the sticker grid. Sticker
    # artwork (even a mostly-white badge/crest) almost always has some dark
    # pixel within any short span, so this reuses the strict gutter
    # threshold; what distinguishes a footer gap from a normal inter-sticker
    # gutter is purely how long the blank run is.
    MIN_FOOTER_GAP: int = 30
    FOOTER_GAP_THRESHOLD: float = 0.05

    # --- Cropping ----------------------------------------------------------------
    MARGIN: int = 6          # pixels kept around each detected sticker edge
    SAFETY_INSET: int = 2    # shave this many px off the raw edge first, to
                             # compensate for edge bleed, before re-expanding
                             # by MARGIN

    # --- Output ----------------------------------------------------------------------
    OUTPUT_FORMAT: str = "png"
    FILENAME_DIGITS: int = 4

    # --- Debug -------------------------------------------------------------------------
    DEBUG: bool = False
    DEBUG_DIR_NAME: str = "debug"

    def scale_to_dpi(self, value: int) -> int:
        """Scale a REFERENCE_DPI-tuned pixel threshold to the active DPI."""
        return int(round(value * self.DPI / self.REFERENCE_DPI))
