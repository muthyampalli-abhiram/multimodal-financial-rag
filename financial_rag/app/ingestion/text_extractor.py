"""Block-level text extraction from financial PDFs."""

import logging
from typing import List, Optional, Sequence
import pymupdf as fitz

from app.core.schemas import DocumentElement, ElementType

logger = logging.getLogger(__name__)


def is_contained_in_table(
    block_rect: fitz.Rect,
    table_rects: Sequence[fitz.Rect],
    overlap_threshold: float = 0.5,
) -> bool:
    """Check if a text block's bounding box significantly overlaps with any detected table.

    Args:
        block_rect: PyMuPDF Rect of the text block.
        table_rects: Sequence of PyMuPDF Rects for tables on the same page.
        overlap_threshold: Fraction of block area that must intersect a table to consider it tabular.

    Returns:
        bool: True if the text block is substantially inside a detected table bounding box.
    """
    block_area = block_rect.get_area()
    if block_area <= 0:
        return False

    for tbl_rect in table_rects:
        intersection = block_rect & tbl_rect
        if not intersection.is_empty:
            intersection_ratio = intersection.get_area() / block_area
            if intersection_ratio >= overlap_threshold:
                return True

    return False


def extract_text_elements(
    doc: fitz.Document,
    doc_name: str,
    min_chars: int = 15,
    skip_table_overlap: bool = True,
) -> List[DocumentElement]:
    """Extract block-level textual elements from a PDF document.

    Iterates over pages in the document and extracts text blocks using PyMuPDF's
    block-level extraction. Skips trivial/empty blocks and blocks that fall within
    detected table bounding boxes.

    Args:
        doc: Opened fitz.Document.
        doc_name: Name of the source document (e.g., 'AAPL_2023_10K.pdf').
        min_chars: Minimum character length threshold to consider a block meaningful.
        skip_table_overlap: If True, uses table detection to skip text blocks overlapping tables.

    Returns:
        List[DocumentElement]: Extracted narrative text blocks with page numbers and bboxes.
    """
    text_elements: List[DocumentElement] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_number = page_idx + 1

        # Detect table bounding boxes on the page to prevent duplicate table text
        table_rects: List[fitz.Rect] = []
        if skip_table_overlap:
            try:
                tables = page.find_tables()
                if tables and tables.tables:
                    table_rects = [fitz.Rect(t.bbox) for t in tables]
            except Exception as e:
                logger.debug(f"Table detection skipped on page {page_number}: {e}")

        # Block extraction: returns (x0, y0, x1, y1, text, block_no, block_type)
        # block_type: 0 for text, 1 for image
        blocks = page.get_text("blocks")

        for block in blocks:
            if len(block) < 7:
                continue

            x0, y0, x1, y1, text, block_no, block_type = block[:7]

            # Process text blocks only
            if block_type != 0:
                continue

            clean_text = text.strip()

            # Skip near-empty or short artifact blocks
            if len(clean_text) < min_chars:
                continue

            block_rect = fitz.Rect(x0, y0, x1, y1)

            # Skip if overlapping with a structured table
            if table_rects and is_contained_in_table(block_rect, table_rects):
                logger.debug(
                    f"Skipping text block {block_no} on page {page_number} (overlaps table)"
                )
                continue

            # Standardize bounding box as (x0, y0, x1, y1)
            bbox = (round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2))

            element = DocumentElement(
                doc_name=doc_name,
                page_number=page_number,
                element_type=ElementType.TEXT,
                content=clean_text,
                bbox=bbox,
                metadata={
                    "block_number": block_no,
                    "char_count": len(clean_text),
                },
            )
            text_elements.append(element)

    logger.info(
        f"Extracted {len(text_elements)} text blocks from '{doc_name}' ({len(doc)} pages)."
    )
    return text_elements
