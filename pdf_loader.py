"""Converts PDF pages into in-memory raster images ready for OpenCV.

Uses ``pdf2image``, which shells out to Poppler's ``pdftoppm`` under the
hood. See README.md for Poppler installation instructions on Windows/macOS/
Linux.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from pdf2image import convert_from_path

from config import Config


@dataclass
class PageImage:
    """A single rendered PDF page."""

    page_number: int   # 1-indexed, matches the PDF's page order
    image: np.ndarray  # BGR uint8, as expected by OpenCV


def load_pdf_pages(pdf_path: Path, config: Config) -> List[PageImage]:
    """Render every page of ``pdf_path`` to a BGR numpy array at ``config.DPI``."""
    pil_pages = convert_from_path(str(pdf_path), dpi=config.DPI)

    pages: List[PageImage] = []
    for page_number, pil_image in enumerate(pil_pages, start=1):
        rgb = np.array(pil_image.convert("RGB"))
        bgr = rgb[:, :, ::-1].copy()  # RGB -> BGR for OpenCV
        pages.append(PageImage(page_number=page_number, image=bgr))
    return pages
