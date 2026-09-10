"""Hybrid and semantic retrieval module using ChromaDB and Gemini query embeddings."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.core.config import settings
from app.core.schemas import DocumentElement, ElementType
from app.indexing.embedder import get_embedding
from app.indexing.vector_store import (
    get_chroma_client,
    get_or_create_collection,
)

logger = logging.getLogger(__name__)


def build_where_filter(
    filter_financial_only: bool = False,
    element_types: Optional[List[Union[str, ElementType]]] = None,
    custom_where: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Build a ChromaDB-compliant metadata filter dictionary.

    Args:
        filter_financial_only: If True, filters elements where is_financial == True.
        element_types: Optional list of element type strings (e.g. ['chart', 'table']).
        custom_where: Optional custom ChromaDB where filter to combine with.

    Returns:
        Optional[Dict[str, Any]]: Valid ChromaDB where filter dict or None if no filters.
    """
    conditions: List[Dict[str, Any]] = []

    if filter_financial_only:
        conditions.append({
            "$or": [
                {"element_type": {"$in": ["text", "table"]}},
                {"is_financial": {"$eq": True}},
            ]
        })

    if element_types:
        normalized_types = [
            et.value if hasattr(et, "value") else str(et).lower()
            for et in element_types
            if et
        ]
        if len(normalized_types) == 1:
            conditions.append({"element_type": {"$eq": normalized_types[0]}})
        elif len(normalized_types) > 1:
            conditions.append({"element_type": {"$in": normalized_types}})

    if custom_where:
        conditions.append(custom_where)

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def retrieve_relevant_elements(
    query: str,
    collection: Optional[Any] = None,
    n_results: int = 8,
    filter_financial_only: bool = False,
    element_types: Optional[List[Union[str, ElementType]]] = None,
    custom_where: Optional[Dict[str, Any]] = None,
    persist_directory: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve relevant document elements from the vector store for a given query.

    Args:
        query: User or analyst financial query text.
        collection: ChromaDB Collection instance or collection name string (or None for default).
        n_results: Maximum number of candidate elements to return (default: 8).
        filter_financial_only: Whether to restrict to elements tagged as financial.
        element_types: Optional list of element types to filter by (e.g. ['table', 'chart']).
        custom_where: Optional additional ChromaDB where filter.
        persist_directory: Path to vector store directory (if collection instance is not provided).

    Returns:
        List[Dict[str, Any]]: List of retrieved element dicts with keys:
            - element_id: Unique ID of the document element
            - content: Text payload or chart description
            - metadata: Stored element metadata
            - distance: Cosine distance from query embedding
            - similarity: Semantic similarity score (1.0 - distance)
    """
    if not query or not query.strip():
        logger.warning("Empty query passed to retrieve_relevant_elements.")
        return []

    # Step a: Embed query with task_type="retrieval_query"
    query_embedding = get_embedding(
        text=query.strip(),
        task_type="retrieval_query",
    )

    if not query_embedding or len(query_embedding) == 0:
        logger.error(f"Failed to generate query embedding for: '{query}'")
        return []

    # Resolve collection instance
    target_collection = collection
    if target_collection is None or isinstance(target_collection, str):
        coll_name = target_collection if isinstance(target_collection, str) else "financial_rag"
        target_collection = get_or_create_collection(
            name=coll_name,
            persist_directory=persist_directory,
        )

    # Check if collection is empty
    try:
        count = target_collection.count()
        if count == 0:
            logger.warning(f"Chroma collection '{target_collection.name}' is empty. Returning no results.")
            return []
        effective_n_results = min(n_results, count)
    except Exception:
        effective_n_results = n_results

    # Step b: Build ChromaDB where metadata filter
    where_filter = build_where_filter(
        filter_financial_only=filter_financial_only,
        element_types=element_types,
        custom_where=custom_where,
    )

    # Step c: Query collection
    query_params: Dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": effective_n_results,
    }
    if where_filter:
        query_params["where"] = where_filter

    try:
        raw_results = target_collection.query(**query_params)
    except Exception as e:
        logger.error(f"Error querying ChromaDB collection '{target_collection.name}': {e}")
        return []

    # Step d: Normalize and package returned records
    retrieved_elements: List[Dict[str, Any]] = []

    ids = raw_results.get("ids", [[]])[0] if raw_results.get("ids") else []
    docs = raw_results.get("documents", [[]])[0] if raw_results.get("documents") else []
    metas = raw_results.get("metadatas", [[]])[0] if raw_results.get("metadatas") else []
    dists = raw_results.get("distances", [[]])[0] if raw_results.get("distances") else []

    for i in range(len(ids)):
        elem_id = ids[i]
        doc_content = docs[i] if i < len(docs) else ""
        meta = metas[i] if i < len(metas) else {}
        dist = dists[i] if i < len(dists) else None

        # Compute cosine similarity score (clamped between 0.0 and 1.0)
        similarity = max(0.0, 1.0 - dist) if dist is not None else None

        retrieved_elements.append({
            "element_id": elem_id,
            "content": doc_content,
            "metadata": meta,
            "distance": dist,
            "similarity": similarity,
        })

    logger.info(
        f"Retrieved {len(retrieved_elements)} elements for query: '{query}' "
        f"(filter_financial={filter_financial_only}, types={element_types})"
    )
    return retrieved_elements


def elements_from_retrieval(retrieval_results: List[Dict[str, Any]]) -> List[DocumentElement]:
    """Convert raw retrieval result dictionaries into Pydantic DocumentElement models.

    Args:
        retrieval_results: List of dicts returned by retrieve_relevant_elements.

    Returns:
        List[DocumentElement]: Deserialized DocumentElement instances.
    """
    elements: List[DocumentElement] = []
    for item in retrieval_results:
        meta = item.get("metadata", {})
        doc_name = meta.get("doc_name", "unknown_doc")
        page_num = int(meta.get("page_number", 1))
        elem_type_raw = meta.get("element_type", "text")

        try:
            elem_type = ElementType(elem_type_raw.lower())
        except ValueError:
            elem_type = ElementType.TEXT

        # Parse original metadata if stored in metadata_json
        parsed_metadata = {}
        if "metadata_json" in meta and meta["metadata_json"]:
            try:
                parsed_metadata = json.loads(meta["metadata_json"])
            except Exception:
                parsed_metadata = dict(meta)
        else:
            parsed_metadata = dict(meta)

        # Parse bbox if present
        bbox = None
        if "bbox" in meta and meta["bbox"]:
            try:
                bbox_val = json.loads(meta["bbox"]) if isinstance(meta["bbox"], str) else meta["bbox"]
                if isinstance(bbox_val, (list, tuple)) and len(bbox_val) == 4:
                    bbox = tuple(float(x) for x in bbox_val)
            except Exception:
                pass

        element = DocumentElement(
            element_id=item.get("element_id"),
            doc_name=doc_name,
            page_number=page_num,
            element_type=elem_type,
            content=item.get("content", ""),
            metadata=parsed_metadata,
            bbox=bbox,
        )
        elements.append(element)

    return elements


def format_retrieved_context(
    retrieval_results: List[Dict[str, Any]],
    max_length_per_element: Optional[int] = 1200,
) -> str:
    """Format retrieved elements into a clean textual context string for generation prompts.

    Args:
        retrieval_results: List of dicts returned by retrieve_relevant_elements.
        max_length_per_element: Optional character cutoff for individual element contents.

    Returns:
        str: Structured markdown context formatted with source references.
    """
    if not retrieval_results:
        return "No relevant context found in indexed financial documents."

    formatted_blocks: List[str] = []
    for idx, item in enumerate(retrieval_results, start=1):
        elem_id = item.get("element_id", "N/A")
        meta = item.get("metadata", {})
        doc_name = meta.get("doc_name", "Document")
        page_num = meta.get("page_number", "N/A")
        elem_type = meta.get("element_type", "text").upper()
        sim = item.get("similarity")
        sim_str = f" | Similarity: {sim:.3f}" if sim is not None else ""

        content = item.get("content", "").strip()
        if max_length_per_element and len(content) > max_length_per_element:
            content = content[:max_length_per_element] + " ...[truncated]"

        block = (
            f"--- [Element {idx} | ID: {elem_id} | {elem_type} | Source: {doc_name}, Page: {page_num}{sim_str}] ---\n"
            f"{content}\n"
        )
        formatted_blocks.append(block)

    return "\n".join(formatted_blocks)


def main():
    """CLI entrypoint to test vector retrieval directly from command line."""
    parser = argparse.ArgumentParser(
        description="Retrieve relevant financial document elements using ChromaDB vector search."
    )
    parser.add_argument(
        "query",
        type=str,
        help="Query text or financial question to search for",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="financial_rag",
        help="ChromaDB collection name (default: financial_rag)",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=None,
        help=f"Path to ChromaDB storage directory (default: {settings.VECTOR_STORE_PATH})",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=5,
        help="Number of top results to retrieve (default: 5)",
    )
    parser.add_argument(
        "--financial-only",
        action="store_true",
        help="Filter results to only those tagged as financial",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=None,
        help="Filter by specific element types (e.g. text table chart image)",
    )

    args = parser.parse_args()

    api_key = settings.GEMINI_API_KEY
    if not api_key or api_key.strip() in ("", "your_gemini_api_key_here"):
        print(
            "\n" + "=" * 65 + "\n"
            "ERROR: GEMINI_API_KEY is not configured.\n"
            "Please add your Google Gemini API key to the .env file:\n"
            "  GEMINI_API_KEY=your_actual_key_here\n"
            + "=" * 65 + "\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"Executing Retrieval Query: '{args.query}'")
    print(f"{'='*65}\n")

    results = retrieve_relevant_elements(
        query=args.query,
        collection=args.collection,
        n_results=args.n_results,
        filter_financial_only=args.financial_only,
        element_types=args.types,
        persist_directory=args.persist_dir,
    )

    if not results:
        print("No matching elements found.\n")
        return

    print(f"Found {len(results)} matching elements:\n")
    for i, res in enumerate(results, start=1):
        meta = res.get("metadata", {})
        elem_type = meta.get("element_type", "unknown").upper()
        doc = meta.get("doc_name", "N/A")
        page = meta.get("page_number", "N/A")
        sim = res.get("similarity")
        sim_text = f"{sim:.4f}" if sim is not None else "N/A"

        print(f"[{i}] ID: {res['element_id']} | Type: {elem_type} | Page: {page} | Doc: {doc} | Sim: {sim_text}")
        print(f"    Content: {res['content'][:200]}...")
        if meta.get("chart_title"):
            print(f"    Chart Title: {meta.get('chart_title')}")
        print()


if __name__ == "__main__":
    main()
