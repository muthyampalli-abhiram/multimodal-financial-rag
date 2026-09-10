#!/usr/bin/env python3
"""
Standalone utility script to extract a subset of pages from a PDF into a new smaller PDF.

Usage:
    python scripts/extract_pages.py --input path/to/source.pdf --pages "1,3,5-8" --output path/to/output.pdf
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        print(
            "Error: PyMuPDF is not installed. Please install it using: pip install pymupdf",
            file=sys.stderr,
        )
        sys.exit(1)


def format_file_size(size_in_bytes: int) -> str:
    """Format bytes into human-readable string with byte count."""
    if size_in_bytes < 1024:
        return f"{size_in_bytes} B"
    elif size_in_bytes < 1024 * 1024:
        kb = size_in_bytes / 1024
        return f"{kb:.2f} KB ({size_in_bytes:,} bytes)"
    else:
        mb = size_in_bytes / (1024 * 1024)
        return f"{mb:.2f} MB ({size_in_bytes:,} bytes)"


def parse_page_spec(pages_str: str, total_pages: int) -> list[int]:
    """
    Parse a comma-separated 1-indexed page specification into a sorted list of unique 0-indexed page numbers.

    Supports:
        - Individual pages: "1, 3, 5"
        - Page ranges: "10-15", "80-82"
        - Mixed: "1, 3, 5-8, 12, 20-25"

    Args:
        pages_str: The page specification string.
        total_pages: Total number of pages in the source document.

    Returns:
        Sorted list of unique 0-indexed page integers.

    Raises:
        ValueError: If the page specification is invalid or page numbers are out of bounds.
    """
    if not pages_str or not pages_str.strip():
        raise ValueError("Page specification cannot be empty.")

    page_numbers_1indexed = set()
    parts = [p.strip() for p in pages_str.split(",") if p.strip()]

    if not parts:
        raise ValueError("No valid page numbers found in specification.")

    for part in parts:
        if "-" in part:
            range_parts = [r.strip() for r in part.split("-")]
            if len(range_parts) != 2 or not range_parts[0] or not range_parts[1]:
                raise ValueError(
                    f"Invalid range format '{part}'. Expected format 'start-end' (e.g. '80-82')."
                )

            try:
                start = int(range_parts[0])
                end = int(range_parts[1])
            except ValueError:
                raise ValueError(
                    f"Invalid non-integer value in range '{part}'. Page numbers must be integers."
                )

            if start < 1 or end < 1:
                raise ValueError(
                    f"Invalid page numbers in range '{part}'. Page numbers must be >= 1."
                )

            if start > end:
                raise ValueError(
                    f"Invalid range '{part}': start page ({start}) cannot be greater than end page ({end})."
                )

            for page_num in range(start, end + 1):
                page_numbers_1indexed.add(page_num)
        else:
            try:
                page_num = int(part)
            except ValueError:
                raise ValueError(
                    f"Invalid page number '{part}'. Expected an integer or range (e.g. '5' or '10-15')."
                )

            if page_num < 1:
                raise ValueError(
                    f"Invalid page number '{part}'. Page numbers must be >= 1."
                )

            page_numbers_1indexed.add(page_num)

    # Validate against document page count
    out_of_bounds = [p for p in sorted(page_numbers_1indexed) if p > total_pages]
    if out_of_bounds:
        raise ValueError(
            f"Page number(s) {out_of_bounds} are out of bounds. "
            f"The source document has {total_pages} page(s) (valid range: 1-{total_pages})."
        )

    # Convert 1-indexed to 0-indexed, deduplicate, and sort
    sorted_0indexed = sorted([p - 1 for p in page_numbers_1indexed])
    return sorted_0indexed


def extract_pages(input_path: str, pages_spec: str, output_path: str) -> None:
    """
    Extract specified pages from source PDF and save to a new PDF.

    Args:
        input_path: Path to source PDF.
        pages_spec: Comma-separated 1-indexed page specification (e.g. "1,3,5-8").
        output_path: Path for output PDF.
    """
    input_file = Path(input_path).resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Source PDF file not found: {input_path}")
    if not input_file.is_file():
        raise ValueError(f"Source path is not a file: {input_path}")

    try:
        doc = fitz.open(str(input_file))
    except Exception as e:
        raise RuntimeError(f"Failed to open source PDF '{input_path}': {e}") from e

    total_pages = len(doc)
    if total_pages == 0:
        doc.close()
        raise ValueError(f"Source PDF '{input_path}' contains 0 pages.")

    try:
        selected_pages_0indexed = parse_page_spec(pages_spec, total_pages)
    except Exception:
        doc.close()
        raise

    selected_count = len(selected_pages_0indexed)
    selected_1indexed = [p + 1 for p in selected_pages_0indexed]

    # Use doc.select() to reduce document to the specified pages in-place
    # preserving embedded images, charts, annotations, and layouts.
    try:
        doc.select(selected_pages_0indexed)
    except Exception as e:
        doc.close()
        raise RuntimeError(f"Failed to select pages from document: {e}") from e

    output_file = Path(output_path).resolve()
    # Create parent directories if they don't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        doc.save(str(output_file), garbage=3, deflate=True)
    except Exception as e:
        doc.close()
        raise RuntimeError(f"Failed to save output PDF to '{output_path}': {e}") from e
    finally:
        doc.close()

    output_size = output_file.stat().st_size

    print("\n" + "=" * 60)
    print("PDF Page Extraction Successful")
    print("=" * 60)
    print(f"Source PDF:     {input_file} ({total_pages} total pages)")
    print(f"Pages Extracted: {selected_count} page(s) -> {selected_1indexed}")
    print(f"Output PDF:     {output_file}")
    print(f"Output Size:    {format_file_size(output_size)}")
    print("=" * 60 + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Extract a subset of pages from a PDF into a new smaller PDF for testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/extract_pages.py --input data/raw_pdfs/report.pdf --pages "1,3,5" --output test_sample.pdf
  python scripts/extract_pages.py --input data/raw_pdfs/report.pdf --pages "45,46,47,80-82" --output test_sample.pdf
  python scripts/extract_pages.py --input report.pdf --pages "1-10, 25, 30-35" --output extracted.pdf
        """,
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the source PDF file",
    )
    parser.add_argument(
        "-p",
        "--pages",
        required=True,
        help='Comma-separated 1-indexed page numbers or ranges (e.g. "45,46,47,80-82")',
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path for the new smaller PDF output file",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        extract_pages(
            input_path=args.input,
            pages_spec=args.pages,
            output_path=args.output,
        )
    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
