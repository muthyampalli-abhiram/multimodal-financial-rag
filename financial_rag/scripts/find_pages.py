#!/usr/bin/env python3
"""
Standalone utility script to search a PDF for keyword matches per page.

Usage:
    python scripts/find_pages.py --input path/to/source.pdf --keywords "Revenue by,Operating margin,Geographic,Segment"
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

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


def parse_keywords(keywords_str: str) -> List[str]:
    """Parse comma-separated keywords into a list of cleaned, non-empty search terms.

    Args:
        keywords_str: Comma-separated keyword string.

    Returns:
        List[str]: List of non-empty search terms.
    """
    if not keywords_str or not keywords_str.strip():
        raise ValueError("Keywords string cannot be empty.")

    keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
    if not keywords:
        raise ValueError("No valid keywords found in input.")

    return keywords


def find_pages(input_path: str, keywords_str: str) -> Dict[int, List[str]]:
    """Search each page of a PDF for case-insensitive occurrences of specified keywords.

    Args:
        input_path: Path to the source PDF.
        keywords_str: Comma-separated keywords or phrases.

    Returns:
        Dict[int, List[str]]: Map of 1-indexed page number -> list of matched keywords.
    """
    input_file = Path(input_path).resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Source PDF file not found: {input_path}")
    if not input_file.is_file():
        raise ValueError(f"Source path is not a file: {input_path}")

    keywords = parse_keywords(keywords_str)

    try:
        doc = fitz.open(str(input_file))
    except Exception as e:
        raise RuntimeError(f"Failed to open source PDF '{input_path}': {e}") from e

    total_pages = len(doc)
    if total_pages == 0:
        doc.close()
        raise ValueError(f"Source PDF '{input_path}' contains 0 pages.")

    matched_pages: Dict[int, List[str]] = {}

    try:
        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_text = page.get_text("text").lower()

            page_matches: List[str] = []
            for kw in keywords:
                if kw.lower() in page_text:
                    page_matches.append(kw)

            if page_matches:
                page_num_1indexed = page_idx + 1
                matched_pages[page_num_1indexed] = page_matches
    finally:
        doc.close()

    print("\n" + "=" * 65)
    print("PDF Keyword Page Search")
    print("=" * 65)
    print(f"Source PDF:     {input_file.name} ({total_pages} pages)")
    print(f"Keywords:       {', '.join(keywords)}")
    print("-" * 65)

    if matched_pages:
        print("MATCHING PAGES & KEYWORDS:")
        for page_num in sorted(matched_pages.keys()):
            kws = ", ".join(matched_pages[page_num])
            print(f"  Page {page_num}: {kws}")

        suggested_pages = ",".join(str(p) for p in sorted(matched_pages.keys()))
        print("-" * 65)
        print(f"Total Matches:  {len(matched_pages)} matching page(s) found.")
        print(f"Suggested --pages value: {suggested_pages}")
    else:
        print("No matching pages found for the specified keywords.")

    print("=" * 65 + "\n")
    return matched_pages


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Search a PDF for keyword matches per page and generate suggested page lists for extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/find_pages.py --input data/raw_pdfs/nvidia_2023_annual_report.pdf --keywords "Revenue by,Operating margin,Geographic,Segment"
  python scripts/find_pages.py -i report.pdf -k "Total Shareholder Return,Capital Allocation,Balance Sheet"
        """,
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the source PDF file",
    )
    parser.add_argument(
        "-k",
        "--keywords",
        required=True,
        help='Comma-separated list of keywords or phrases to search for (e.g. "Revenue by,Operating margin,Segment")',
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        find_pages(
            input_path=args.input,
            keywords_str=args.keywords,
        )
    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
