"""Two-stage retrieval pipeline with query intent routing and broad search fallback."""

import argparse
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

from app.core.config import settings
from app.indexing.vector_store import get_or_create_collection
from app.retrieval.query_analyzer import detect_query_intent
from app.retrieval.retriever import retrieve_relevant_elements

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def retrieve_with_fallback(
    query: str,
    collection: Optional[Any] = None,
    n_results: int = 8,
) -> Tuple[List[Dict[str, Any]], str]:
    """Retrieve elements with intent-based filtering and automatic broad fallback.

    Args:
        query: User query string.
        collection: Optional ChromaDB collection instance or collection name.
        n_results: Maximum number of results to retrieve (default: 8).

    Returns:
        Tuple[List[Dict[str, Any]], str]:
            - List of retrieved element dictionaries
            - path_taken: 'financial_filtered', 'broad_fallback', or 'broad_default'
    """
    if not query or not query.strip():
        return [], "empty_query"

    # Step a: Detect query intent
    intent = detect_query_intent(query)
    wants_financial = intent.get("wants_financial_data", False)

    logger.info(
        f"Query intent: wants_financial={wants_financial}, "
        f"wants_comparison={intent.get('wants_comparison')}, "
        f"years={intent.get('mentioned_years')}"
    )

    if wants_financial:
        # Step b: Try retrieving with financial filter first
        results = retrieve_relevant_elements(
            query=query,
            collection=collection,
            n_results=n_results,
            filter_financial_only=True,
        )

        # Step c: Check if results are sufficient
        if len(results) >= 3:
            return results, "financial_filtered"

        logger.info(
            f"Financial filter returned {len(results)} results (< 3). Triggering broad search fallback..."
        )
        fallback_results = retrieve_relevant_elements(
            query=query,
            collection=collection,
            n_results=n_results,
            filter_financial_only=False,
        )
        return fallback_results, "broad_fallback"

    # Default broad retrieval path
    results = retrieve_relevant_elements(
        query=query,
        collection=collection,
        n_results=n_results,
        filter_financial_only=False,
    )
    return results, "broad_default"


def retrieve_for_query(
    query: str,
    collection_name: str = "financial_rag",
    n_results: int = 8,
) -> List[Dict[str, Any]]:
    """Retrieve relevant document elements for a given query from the specified collection.

    Resolves the ChromaDB collection and executes intent routing with fallback.

    Args:
        query: User query text.
        collection_name: Name of the ChromaDB collection (default: 'financial_rag').
        n_results: Maximum number of elements to retrieve (default: 8).

    Returns:
        List[Dict[str, Any]]: List of retrieved element dicts.
    """
    collection = get_or_create_collection(name=collection_name)
    results, _ = retrieve_with_fallback(
        query=query,
        collection=collection,
        n_results=n_results,
    )
    return results


def main():
    """CLI entrypoint for running intent-aware hybrid retrieval with fallback."""
    parser = argparse.ArgumentParser(
        description="Retrieve relevant financial document elements using intent routing with fallback."
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Analyst query or financial question (required)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="financial_rag",
        help="ChromaDB collection name (default: financial_rag)",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=8,
        help="Maximum results to return (default: 8)",
    )

    args = parser.parse_args()

    api_key = settings.GEMINI_API_KEY
    if not api_key or api_key.strip() in ("", "your_gemini_api_key_here"):
        print(
            "\n" + "=" * 65 + "\n"
            "ERROR: GEMINI_API_KEY is not configured.\n"
            "Please set GEMINI_API_KEY in your .env file.\n"
            + "=" * 65 + "\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"Executing Retrieval Pipeline: '{args.query}'")
    print(f"{'='*65}\n")

    # Step 1: Detect Intent
    intent = detect_query_intent(args.query)
    print("Query Intent Analysis:")
    print(f"  - Wants Financial Data: {intent['wants_financial_data']}")
    print(f"  - Wants Comparison:     {intent['wants_comparison']}")
    print(f"  - Mentioned Years:      {intent['mentioned_years'] if intent['mentioned_years'] else 'None'}\n")

    # Step 2: Retrieve with Fallback
    collection = get_or_create_collection(name=args.collection)
    results, path_taken = retrieve_with_fallback(
        query=args.query,
        collection=collection,
        n_results=args.n_results,
    )

    print(f"Retrieval Routing Path: [{path_taken.upper()}]")
    print(f"Total Results Found:     {len(results)}\n")

    if not results:
        print("No matching elements found in collection.\n")
        return

    print("-" * 65)
    print("RANKED RETRIEVAL RESULTS")
    print("-" * 65)

    for rank, res in enumerate(results, start=1):
        meta = res.get("metadata", {})
        elem_id = res.get("element_id", "N/A")
        elem_type = meta.get("element_type", "unknown").upper()
        page = meta.get("page_number", "N/A")
        doc = meta.get("doc_name", "N/A")
        sim = res.get("similarity")
        sim_str = f"{sim:.4f}" if sim is not None else "N/A"

        content = res.get("content", "").strip()
        # Clean newlines for preview
        clean_preview = " ".join(content.split())
        preview = clean_preview[:150] + ("..." if len(clean_preview) > 150 else "")

        print(f"Rank {rank:02d} | Sim: {sim_str} | ID: {elem_id} | Type: {elem_type} | Page: {page} | Doc: {doc}")
        print(f"  Preview: \"{preview}\"")
        if meta.get("chart_title"):
            print(f"  Chart Title: {meta.get('chart_title')}")
        print()

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
