"""Table extraction from financial PDFs using PyMuPDF table detection."""

import logging
from typing import List, Optional, Tuple
import pymupdf as fitz
import pandas as pd

from app.core.schemas import DocumentElement, ElementType

logger = logging.getLogger(__name__)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert a Pandas DataFrame into a clean Markdown table string.

    Provides a robust fallback if tabulate/external dependencies are unavailable.
    """
    try:
        md = df.to_markdown(index=False)
        if md:
            return md
    except Exception:
        pass

    # Fallback Markdown generator
    headers = [str(c) if c is not None else "" for c in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"

    rows = []
    for _, row in df.iterrows():
        row_vals = [str(val).strip() if pd.notna(val) and val is not None else "" for val in row]
        rows.append("| " + " | ".join(row_vals) + " |")

    return "\n".join([header_line, separator_line] + rows)


def extract_table_elements(
    doc: fitz.Document,
    doc_name: str,
    min_rows: int = 1,
    min_cols: int = 1,
) -> List[DocumentElement]:
    """Extract structured tabular elements from a PDF document using PyMuPDF table detection.

    Locates tables on each page, parses them into Pandas DataFrames, serializes
    them to Markdown formatted strings, and captures raw record data in metadata.

    Args:
        doc: Opened fitz.Document.
        doc_name: Name of the source document (e.g., 'AAPL_2023_10K.pdf').
        min_rows: Minimum number of rows required to retain a detected table.
        min_cols: Minimum number of columns required to retain a detected table.

    Returns:
        List[DocumentElement]: Extracted table elements with markdown content and metadata.
    """
    table_elements: List[DocumentElement] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_number = page_idx + 1

        try:
            detected_tables = page.find_tables()
        except Exception as e:
            logger.warning(
                f"Error detecting tables on page {page_number} of '{doc_name}': {e}"
            )
            continue

        if not detected_tables or not detected_tables.tables:
            continue

        for tbl_idx, table in enumerate(detected_tables):
            try:
                # Convert PyMuPDF table to pandas DataFrame
                df: pd.DataFrame = table.to_pandas()
            except Exception as e:
                logger.warning(
                    f"Failed to convert table {tbl_idx + 1} on page {page_number} to DataFrame: {e}"
                )
                continue

            if df.empty or len(df) < min_rows or len(df.columns) < min_cols:
                continue

            # Clean and sanitize column names and values
            df = df.fillna("")
            df.columns = [str(c).strip() if str(c).strip() else f"Col_{i}" for i, c in enumerate(df.columns)]

            # Convert to Markdown string representation
            markdown_table = dataframe_to_markdown(df)

            # Extract raw record format for downstream structured querying
            records_data = df.to_dict(orient="records")

            # Extract bounding box (x0, y0, x1, y1)
            raw_bbox = table.bbox
            bbox: Tuple[float, float, float, float] = (
                round(raw_bbox[0], 2),
                round(raw_bbox[1], 2),
                round(raw_bbox[2], 2),
                round(raw_bbox[3], 2),
            )

            element = DocumentElement(
                doc_name=doc_name,
                page_number=page_number,
                element_type=ElementType.TABLE,
                content=markdown_table,
                bbox=bbox,
                metadata={
                    "table_index": tbl_idx + 1,
                    "row_count": len(df),
                    "col_count": len(df.columns),
                    "columns": list(df.columns),
                    "table_data": records_data,
                },
            )
            table_elements.append(element)

    logger.info(
        f"Extracted {len(table_elements)} tables from '{doc_name}' ({len(doc)} pages)."
    )
    return table_elements
