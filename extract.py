#!/usr/bin/env python3
"""CLI entry point: extract every sticker from a Panini album PDF into PNGs.

Usage:
    python extract.py album.pdf output/
    python extract.py album.pdf output/ --dpi 600 --margin 10 --debug
"""
import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from config import Config
from cropper import crop_sticker, save_sticker
from detector import detect_stickers, estimate_grid_size
from pdf_loader import PageImage, load_pdf_pages
from utils import draw_debug_overlay, ensure_dir

# Gutters required for a page's own grid estimate to count as a "vote" when
# establishing the album's shared template (see _majority_grid_size).
MIN_GRID_CONFIDENCE = 2


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract stickers from a Panini album PDF.")
    parser.add_argument("pdf_path", type=Path, help="Path to the source PDF")
    parser.add_argument(
        "output_dir", type=Path, nargs="?", default=Path("output"),
        help="Directory where PNGs are written (default: output/)",
    )
    parser.add_argument("--dpi", type=int, default=None, help="Override rendering DPI")
    parser.add_argument("--margin", type=int, default=None, help="Override crop margin, in px")
    parser.add_argument("--debug", action="store_true", help="Write per-page debug overlays")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    overrides = {}
    if args.dpi is not None:
        overrides["DPI"] = args.dpi
    if args.margin is not None:
        overrides["MARGIN"] = args.margin
    if args.debug:
        overrides["DEBUG"] = True
    return Config(**overrides)


def _majority_grid_size(pages: List[PageImage], config: Config) -> Optional[Tuple[int, int]]:
    """Poll every page's own grid estimate and return the (row_count,
    col_count) that wins a majority vote among high-confidence pages.

    A Panini album uses one fixed grid template throughout, so a handful of
    pages where the local pitch estimate is ambiguous (busy artwork bleeding
    across every candidate gutter) can safely reuse the count the rest of
    the album agrees on, rather than falling back to a wrong per-page guess.
    """
    row_votes: Counter = Counter()
    col_votes: Counter = Counter()
    for page in pages:
        row_count, col_count, row_confidence, col_confidence = estimate_grid_size(page.image, config)
        if row_confidence >= MIN_GRID_CONFIDENCE:
            row_votes[row_count] += 1
        if col_confidence >= MIN_GRID_CONFIDENCE:
            col_votes[col_count] += 1

    if not row_votes or not col_votes:
        return None
    return row_votes.most_common(1)[0][0], col_votes.most_common(1)[0][0]


def run(pdf_path: Path, output_dir: Path, config: Config) -> int:
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    ensure_dir(output_dir)
    debug_dir = ensure_dir(output_dir / config.DEBUG_DIR_NAME) if config.DEBUG else None

    pages = load_pdf_pages(pdf_path, config)
    print(f"Loaded {len(pages)} page(s) at {config.DPI} DPI from '{pdf_path.name}'")

    print("Scanning pages to establish the album's grid template...")
    majority_grid = _majority_grid_size(pages, config)
    if majority_grid:
        print(f"  Album majority grid template: {majority_grid[0]}x{majority_grid[1]}")
    else:
        print("  No consistent grid found; falling back to per-page detection")

    sticker_index = 0
    for page in pages:
        # Detect grid size locally for this page
        row_count, col_count, _, _ = estimate_grid_size(page.image, config)
        if row_count == 0 or col_count == 0:
            row_count, col_count = majority_grid if majority_grid else (4, 4)

        boxes = detect_stickers(page.image, config, grid_size=(row_count, col_count))
        print(f"  Page {page.page_number}: using {row_count}x{col_count} grid (detected {len(boxes)} sticker(s))")


        if debug_dir is not None:
            overlay = draw_debug_overlay(page.image, boxes)
            cv2.imwrite(str(debug_dir / f"page_{page.page_number:03d}.png"), overlay)

        for box in boxes:
            sticker_index += 1
            cropped = crop_sticker(page.image, box, config)
            filename = f"{sticker_index:0{config.FILENAME_DIGITS}d}.{config.OUTPUT_FORMAT}"
            save_sticker(cropped, output_dir / filename, config)

    print(f"Done. Extracted {sticker_index} sticker(s) into '{output_dir}'")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    return run(args.pdf_path, args.output_dir, config)


if __name__ == "__main__":
    sys.exit(main())
