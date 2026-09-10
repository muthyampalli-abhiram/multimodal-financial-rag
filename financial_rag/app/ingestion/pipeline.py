"""Orchestrator pipeline for PDF ingestion and multimodal element extraction."""

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional, Union

from app.core.schemas import DocumentElement, ElementType
from app.ingestion.image_extractor import extract_image_elements
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.table_extractor import extract_table_elements
from app.ingestion.text_extractor import extract_text_elements

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def ingest_pdf(
    file_path: Union[str, Path],
    output_image_dir: Union[str, Path] = "data/processed/images",
) -> List[DocumentElement]:
    """Ingest a financial PDF and extract all text, table, chart, and image elements.

    Orchestrates the extraction steps:
    1. Loads and validates the PDF document.
    2. Extracts structured tables first (using PyMuPDF table detection).
    3. Extracts embedded images and candidate charts, saving them to disk.
    4. Extracts text blocks, filtering out text that overlaps detected table regions.
    5. Assigns sequential element IDs per element type ({doc_name}_{element_type}_{index}).
    6. Combines and sorts all elements by page number and reading order.

    Args:
        file_path: Path to the PDF file.
        output_image_dir: Directory where extracted images will be stored.

    Returns:
        List[DocumentElement]: Sorted list of all extracted document elements.
    """
    path = Path(file_path)
    doc = load_pdf(path)

    # Derive document name from file stem (without extension)
    doc_name = path.stem

    try:
        # Step 1: Extract tables first
        logger.info(f"Extracting tables from '{doc_name}'...")
        table_elements = extract_table_elements(doc=doc, doc_name=doc_name)

        # Step 2: Extract images and candidate charts
        logger.info(f"Extracting images and charts from '{doc_name}'...")
        image_elements = extract_image_elements(
            doc=doc,
            doc_name=doc_name,
            output_dir=output_image_dir,
        )

        # Step 3: Extract narrative text blocks (skipping table overlap)
        logger.info(f"Extracting text blocks from '{doc_name}'...")
        text_elements = extract_text_elements(
            doc=doc,
            doc_name=doc_name,
            skip_table_overlap=True,
        )

        # Step 4: Group and assign sequential element IDs per element type
        # Format: {doc_name}_{element_type}_{index} (1-indexed)
        type_counters = {
            ElementType.TEXT.value: 0,
            ElementType.TABLE.value: 0,
            ElementType.CHART.value: 0,
            ElementType.IMAGE.value: 0,
        }

        all_raw_elements = table_elements + image_elements + text_elements

        # Sort elements by page_number, then top-to-bottom bbox coordinate (y0)
        def sort_key(elem: DocumentElement):
            y0 = 0.0
            x0 = 0.0
            if elem.bbox is not None:
                if isinstance(elem.bbox, (tuple, list)) and len(elem.bbox) >= 2:
                    x0 = elem.bbox[0]
                    y0 = elem.bbox[1]
                elif hasattr(elem.bbox, "y0"):
                    x0 = elem.bbox.x0
                    y0 = elem.bbox.y0
            return (elem.page_number, y0, x0)

        sorted_elements = sorted(all_raw_elements, key=sort_key)

        # Re-assign sequential IDs in page-sorted order
        for el in sorted_elements:
            type_str = el.element_type.value if hasattr(el.element_type, "value") else str(el.element_type)
            type_counters[type_str] = type_counters.get(type_str, 0) + 1
            idx = type_counters[type_str]
            el.element_id = f"{doc_name}_{type_str}_{idx}"

        logger.info(
            f"Successfully ingested '{doc_name}': {len(sorted_elements)} total elements extracted."
        )
        return sorted_elements

    finally:
        doc.close()


def main():
    """CLI entrypoint for running the PDF ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Ingest a financial PDF and extract text, tables, charts, and images."
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the PDF file to ingest",
    )
    parser.add_argument(
        "--output-images",
        type=str,
        default="data/processed/images",
        help="Directory to save extracted images/charts (default: data/processed/images)",
    )

    args = parser.parse_args()
    pdf_path = Path(args.pdf_path)

    if not pdf_path.exists():
        print(f"Error: File not found at '{pdf_path}'", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Starting Ingestion Pipeline for: {pdf_path.name}")
    print(f"{'='*60}\n")

    start_time = time.perf_counter()
    try:
        elements = ingest_pdf(file_path=pdf_path, output_image_dir=args.output_images)
    except Exception as e:
        print(f"\nPipeline failed with error: {e}", file=sys.stderr)
        sys.exit(1)

    elapsed_time = time.perf_counter() - start_time

    # Compute type counts
    counts = Counter(
        el.element_type.value if hasattr(el.element_type, "value") else str(el.element_type)
        for el in elements
    )

    print(f"\n{'-'*60}")
    print("INGESTION SUMMARY")
    print(f"{'-'*60}")
    print(f"Document:                {pdf_path.stem}")
    print(f"Total Elements Extracted: {len(elements)}")
    print(f"  - Text blocks:         {counts.get(ElementType.TEXT.value, 0)}")
    print(f"  - Tables:              {counts.get(ElementType.TABLE.value, 0)}")
    print(f"  - Charts (candidates): {counts.get(ElementType.CHART.value, 0)}")
    print(f"  - Images:              {counts.get(ElementType.IMAGE.value, 0)}")
    print(f"Total Processing Time:   {elapsed_time:.2f} seconds")
    print(f"{'-'*60}\n")

    # Table and Chart Previews
    tables_and_charts = [
        el for el in elements if el.element_type in (ElementType.TABLE, ElementType.CHART)
    ]

    if tables_and_charts:
        print(f"STRUCTURED & VISUAL ELEMENTS PREVIEW ({len(tables_and_charts)} found):")
        print(f"{'-'*60}")
        for el in tables_and_charts:
            type_label = el.element_type.value.upper()
            # Clean single line preview (first ~100 characters)
            preview_clean = " ".join(el.content.split())
            preview_snippet = (
                preview_clean[:97] + "..." if len(preview_clean) > 100 else preview_clean
            )
            print(f"[{type_label}] ID: {el.element_id} | Page {el.page_number}")
            print(f"  Preview: {preview_snippet}")
            if el.bbox:
                print(f"  BBox: {el.bbox}")
            print()
    else:
        print("No tables or chart elements detected in document.")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
