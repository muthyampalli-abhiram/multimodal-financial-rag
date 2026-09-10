"""End-to-end RAG generation pipeline from retrieval to grounded answer synthesis."""

import argparse
import logging
import sys
from typing import Optional

from app.core.config import settings
from app.core.schemas import QueryResult
from app.generation.answer_generator import generate_answer
from app.retrieval.pipeline import retrieve_for_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def answer_query(
    query: str,
    collection_name: str = "financial_rag",
    n_results: int = 8,
) -> QueryResult:
    """Execute end-to-end question answering with hybrid retrieval and grounded synthesis.

    Args:
        query: Analyst query or financial question.
        collection_name: ChromaDB collection name (default: 'financial_rag').
        n_results: Maximum candidate elements to retrieve for context (default: 8).

    Returns:
        QueryResult: Synthesized answer with citations and source elements.
    """
    logger.info(f"Executing answer pipeline for query: '{query}'")

    # Step a: Retrieve candidate elements with intent routing and fallback
    retrieved_results = retrieve_for_query(
        query=query,
        collection_name=collection_name,
        n_results=n_results,
    )

    # Step b & c: Synthesize answer and extract citations
    query_result = generate_answer(
        query=query,
        retrieved_results=retrieved_results,
    )

    return query_result


def main():
    """CLI entrypoint for interactive financial question answering."""
    parser = argparse.ArgumentParser(
        description="Ask financial queries against indexed filings and receive grounded answers with citations."
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Financial question or analysis prompt to ask (required)",
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
        help="Number of candidate elements to retrieve (default: 8)",
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
    print(f"Question: '{args.query}'")
    print(f"{'='*65}\n")

    result = answer_query(
        query=args.query,
        collection_name=args.collection,
        n_results=args.n_results,
    )

    print("ANALYST ANSWER:")
    print("-" * 65)
    print(result.answer)
    print("-" * 65 + "\n")

    if result.citations:
        print(f"SOURCES & GROUNDED CITATIONS ({len(result.citations)}):")
        print("-" * 65)
        for idx, cit in enumerate(result.citations, start=1):
            elem_type = (
                cit.element_type.value
                if hasattr(cit.element_type, "value")
                else str(cit.element_type or "unknown").upper()
            )
            print(f"[{idx}] Source ID:    {cit.element_id}")
            print(f"    Document:     {cit.doc_name} (Page {cit.page_number})")
            print(f"    Element Type: {elem_type.upper()}")
            if cit.snippet:
                clean_snippet = " ".join(cit.snippet.split())
                preview = clean_snippet[:180] + ("..." if len(clean_snippet) > 180 else "")
                print(f"    Excerpt:      \"{preview}\"")
            print()
    else:
        print("No citations referenced in response.\n")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
