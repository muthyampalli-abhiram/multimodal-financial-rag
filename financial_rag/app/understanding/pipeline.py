"""Pipeline module for multimodal enrichment of extracted document elements."""

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional, Union

from app.core.config import settings
from app.core.schemas import DocumentElement, ElementType
from app.ingestion.pipeline import ingest_pdf
from app.understanding.chart_analyzer import analyze_chart

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def save_elements_to_json(
    elements: List[DocumentElement],
    output_path: Union[str, Path],
) -> None:
    """Serialize and save a list of DocumentElements to a pretty-printed JSON file.

    Args:
        elements: List of DocumentElement instances.
        output_path: Destination JSON file path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [el.model_dump(mode="json") for el in elements]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(elements)} elements to '{path.resolve()}'")


def load_elements_from_json(
    input_path: Union[str, Path],
) -> List[DocumentElement]:
    """Load a list of DocumentElements from a JSON file.

    Args:
        input_path: Source JSON file path.

    Returns:
        List[DocumentElement]: Deserialized list of DocumentElements.
    """
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [DocumentElement.model_validate(item) for item in data]


def enrich_elements_with_understanding(
    elements: List[DocumentElement],
    existing_elements: Optional[List[DocumentElement]] = None,
    max_calls: Optional[int] = None,
    delay_seconds: float = 2.0,
) -> List[DocumentElement]:
    """Enrich extracted document elements with multimodal LLM understanding.

    Supports resumable and quota-aware processing by reusing valid previous analyses
    and capping API calls per run.

    Args:
        elements: List of DocumentElements from the ingestion stage.
        existing_elements: Optional list of previously processed DocumentElements.
                           Elements with valid (non-error) analyses are reused from cache.
        max_calls: Optional maximum number of new Gemini API calls to make in this run.
        delay_seconds: Seconds to wait between new Gemini API calls to prevent burst 429s.

    Returns:
        List[DocumentElement]: Enriched list with multimodal chart descriptions.
    """
    # Build lookup map for existing valid analyses: element_id -> DocumentElement
    cached_map: dict[str, DocumentElement] = {}
    if existing_elements:
        for ex in existing_elements:
            gemini_analysis = ex.metadata.get("gemini_analysis")
            if (
                isinstance(gemini_analysis, dict)
                and "error" not in gemini_analysis
                and gemini_analysis.get("summary")
            ):
                cached_map[ex.element_id] = ex

    # Build page-level text context map to supply surrounding narrative to Gemini
    page_context_map: dict[int, str] = {}
    for el in elements:
        if el.element_type == ElementType.TEXT:
            page_context_map[el.page_number] = (
                page_context_map.get(el.page_number, "") + "\n" + el.content
            )

    visual_types = {ElementType.CHART, ElementType.IMAGE, "chart", "image"}
    visual_elements = [el for el in elements if el.element_type in visual_types]
    total_visual = len(visual_elements)

    calls_made = 0
    visual_idx = 0
    stopped_early = False

    for el in elements:
        if el.element_type in visual_types:
            visual_idx += 1

            # Check if this element already has a valid cached analysis
            if el.element_id in cached_map:
                cached_el = cached_map[el.element_id]
                el.metadata["gemini_analysis"] = cached_el.metadata["gemini_analysis"]
                el.content = cached_el.content
                print(
                    f"  -> [CACHE REUSED] Visual element {visual_idx}/{total_visual} "
                    f"(ID: {el.element_id} | Page {el.page_number})",
                    flush=True,
                )
                logger.info(
                    f"Reused cached analysis for visual element {visual_idx}/{total_visual} ({el.element_id})"
                )
                continue

            # Check if max_calls limit has been reached
            if max_calls is not None and calls_made >= max_calls:
                if not stopped_early:
                    stopped_early = True
                    remaining = total_visual - (visual_idx - 1)
                    print(
                        f"\nReached max_calls limit ({max_calls}). Stopping early — "
                        f"{remaining} elements remain unprocessed. Re-run this command later to continue.\n",
                        flush=True,
                    )
                    logger.warning(
                        f"Reached max_calls limit ({max_calls}). Stopping early with {remaining} elements unprocessed."
                    )
                continue

            image_path = (
                el.metadata.get("image_path")
                or el.metadata.get("file_path")
                or el.metadata.get("relative_path")
            )

            if not image_path:
                logger.warning(
                    f"Element '{el.element_id}' tagged as visual but has no image_path in metadata."
                )
                continue

            print(
                f"  -> Analyzing visual element {visual_idx}/{total_visual} "
                f"(ID: {el.element_id} | Page {el.page_number})...",
                flush=True,
            )

            page_ctx = page_context_map.get(el.page_number, "").strip()

            analysis = analyze_chart(
                image_path=image_path,
                page_context=page_ctx[:1000] if page_ctx else "",
            )
            calls_made += 1

            # Store full structured analysis in element metadata
            el.metadata["gemini_analysis"] = analysis

            # Update content field to plain-English summary if available
            summary = analysis.get("summary")
            if summary and not analysis.get("error"):
                title = analysis.get("title", "Financial Visual")
                chart_type = analysis.get("chart_type", "visual")
                el.content = f"[{chart_type.upper()}: {title}] {summary}"
            elif analysis.get("error"):
                logger.warning(
                    f"Understanding returned error for {el.element_id}: {analysis.get('error')}"
                )

            # Add delay between API calls to prevent burst 429 quota exhaustion
            if delay_seconds > 0 and (max_calls is None or calls_made < max_calls):
                time.sleep(delay_seconds)

    return elements


def main():
    """CLI entrypoint to test ingestion + multimodal understanding end-to-end."""
    parser = argparse.ArgumentParser(
        description="Ingest a financial PDF and enrich charts/images with Gemini Multimodal Understanding."
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the financial PDF file",
    )
    parser.add_argument(
        "--output-images",
        type=str,
        default="data/processed/images",
        help="Directory to save extracted images (default: data/processed/images)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save enriched JSON (default: data/processed/{doc_name}_enriched.json)",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="Maximum number of new Gemini API calls to make in this run (default: None, unlimited)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds delay between Gemini API calls (default: 2.0s)",
    )

    args = parser.parse_args()
    pdf_path = Path(args.pdf_path)

    if not pdf_path.exists():
        print(f"Error: PDF file not found at '{pdf_path}'", file=sys.stderr)
        sys.exit(1)

    # Validate Gemini API Key before running
    api_key = settings.GEMINI_API_KEY
    if not api_key or api_key.strip() in ("", "your_gemini_api_key_here"):
        print(
            "\n" + "=" * 60 + "\n"
            "ERROR: GEMINI_API_KEY is not configured.\n"
            "Please add your Google Gemini API key to the .env file:\n"
            "  GEMINI_API_KEY=your_actual_key_here\n"
            + "=" * 60 + "\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve output path
    doc_slug = pdf_path.stem
    output_json_path = (
        Path(args.output_json)
        if args.output_json
        else Path("data/processed") / f"{doc_slug}_enriched.json"
    )

    # Check for existing enriched JSON to enable resume from cache
    existing_elements: Optional[List[DocumentElement]] = None
    if output_json_path.is_file():
        try:
            existing_elements = load_elements_from_json(output_json_path)
            cached_valid = sum(
                1
                for el in existing_elements
                if isinstance(el.metadata.get("gemini_analysis"), dict)
                and "error" not in el.metadata["gemini_analysis"]
            )
            print(
                f"Found existing progress in '{output_json_path.name}' "
                f"({cached_valid} valid analyses cached). Resuming with cache...\n"
            )
        except Exception as e:
            logger.warning(f"Could not load existing file '{output_json_path}': {e}")

    print(f"\n{'='*65}")
    print(f"Starting Multimodal Understanding Pipeline: {pdf_path.name}")
    print(f"{'='*65}\n")

    enriched_elements: List[DocumentElement] = []
    try:
        # Step 1: Raw Ingestion
        print("[1/2] Running PDF Ingestion & Extraction...")
        raw_elements = ingest_pdf(file_path=pdf_path, output_image_dir=args.output_images)

        # Step 2: Gemini Multimodal Enrichment (resumable)
        print(f"\n[2/2] Running Gemini Multimodal Analysis on extracted elements...")
        enriched_elements = enrich_elements_with_understanding(
            elements=raw_elements,
            existing_elements=existing_elements,
            max_calls=args.max_calls,
            delay_seconds=args.delay,
        )
    except Exception as e:
        print(f"\nPipeline failed: {e}", file=sys.stderr)
        if enriched_elements:
            save_elements_to_json(enriched_elements, output_json_path)
        sys.exit(1)
    finally:
        # Save progress so partial progress is never lost
        if enriched_elements:
            save_elements_to_json(enriched_elements, output_json_path)

    # Compute execution breakdown counts
    counts = Counter(
        el.element_type.value if hasattr(el.element_type, "value") else str(el.element_type)
        for el in enriched_elements
    )
    all_visuals = [
        el for el in enriched_elements
        if el.element_type in (ElementType.CHART, ElementType.IMAGE)
    ]

    analyzed_this_run = 0
    reused_from_cache = 0
    pending_count = 0
    financial_visuals = []

    for el in all_visuals:
        analysis = el.metadata.get("gemini_analysis")
        if isinstance(analysis, dict) and "error" not in analysis:
            if analysis.get("is_financial") is True:
                financial_visuals.append(el)
            if existing_elements and any(
                ex.element_id == el.element_id
                and isinstance(ex.metadata.get("gemini_analysis"), dict)
                and "error" not in ex.metadata["gemini_analysis"]
                for ex in existing_elements
            ):
                reused_from_cache += 1
            else:
                analyzed_this_run += 1
        else:
            pending_count += 1

    print(f"\n{'-'*65}")
    print("INGESTION & UNDERSTANDING SUMMARY")
    print(f"{'-'*65}")
    print(f"Document:                     {pdf_path.stem}")
    print(f"Total Elements Extracted:     {len(enriched_elements)}")
    print(f"  - Text blocks:              {counts.get(ElementType.TEXT.value, 0)}")
    print(f"  - Tables:                   {counts.get(ElementType.TABLE.value, 0)}")
    print(f"  - Visuals (Charts/Images):  {len(all_visuals)}")
    print(f"    |-- Analyzed this run:    {analyzed_this_run}")
    print(f"    |-- Reused from cache:    {reused_from_cache}")
    print(f"    |-- Still pending/error:  {pending_count}")
    print(f"    |-- Financial charts:     {len(financial_visuals)}")
    print(f"{'-'*65}\n")

    # Filtered view: ONLY financial charts / visuals
    if financial_visuals:
        print(f"RELEVANT FINANCIAL CHARTS & VISUALS ({len(financial_visuals)} found):")
        print(f"{'-'*65}")
        for el in financial_visuals:
            analysis = el.metadata.get("gemini_analysis", {})
            chart_type = analysis.get("chart_type", "Unknown")
            title = analysis.get("title", "No Title")
            summary = analysis.get("summary", el.content)
            data_points = analysis.get("data_points", [])
            time_period = analysis.get("time_period", "N/A")

            print(f"[{chart_type.upper()}] ID: {el.element_id} | Page {el.page_number}")
            print(f"  Title:        {title}")
            if time_period and time_period != "N/A":
                print(f"  Time Period:  {time_period}")
            print(f"  Summary:      {summary}")

            if data_points:
                print(f"  Data Points ({len(data_points)}):")
                for dp in data_points:
                    label = dp.get("label", "")
                    val = dp.get("value", "")
                    print(f"    - {label}: {val}")
            print()
    else:
        print("No financial charts/figures identified for analyst review.\n")

    # Print full output path of saved JSON
    print(f"{'='*65}")
    print("Enriched dataset successfully saved to:")
    print(f"  {output_json_path.resolve()}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
