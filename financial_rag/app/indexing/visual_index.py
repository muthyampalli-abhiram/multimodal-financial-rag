"""Multimodal visual vector index for image and chart elements using ChromaDB and CLIP embeddings.

Note: This visual collection is currently a standalone proof-of-concept — merging its
results into the main generation pipeline (app/generation/) would be a future enhancement,
not part of this step.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chromadb

from app.core.config import settings
from app.core.schemas import DocumentElement, ElementType
from app.indexing.clip_embedder import embed_image, embed_text_clip
from app.indexing.vector_store import format_metadata_for_chroma, get_chroma_client

logger = logging.getLogger(__name__)

VISUAL_COLLECTION_NAME = "financial_rag_visual"


def get_visual_collection(
    persist_directory: Optional[Union[str, Path]] = None,
) -> chromadb.Collection:
    """Retrieve or create the dedicated multimodal visual ChromaDB collection.

    Creates/gets a separate collection named 'financial_rag_visual' using the CLIP vector space.
    This collection is intentionally kept separate from the text-based 'financial_rag' collection.

    Args:
        persist_directory: Path to vector store directory.

    Returns:
        chromadb.Collection: ChromaDB collection for visual elements.
    """
    client = get_chroma_client(persist_directory=persist_directory)
    return client.get_or_create_collection(
        name=VISUAL_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_visual_elements(
    elements: List[DocumentElement],
    collection: Optional[chromadb.Collection] = None,
    persist_directory: Optional[Union[str, Path]] = None,
) -> Dict[str, int]:
    """Index chart and image elements into the dedicated visual vector collection using CLIP embeddings.

    Args:
        elements: List of DocumentElement objects extracted from financial files.
        collection: Target visual ChromaDB collection. If None, retrieves 'financial_rag_visual'.
        persist_directory: Path to vector store directory (if collection is None).

    Returns:
        Dict[str, int]: Summary dictionary with keys 'indexed', 'skipped', and 'errors'.
    """
    if collection is None:
        collection = get_visual_collection(persist_directory=persist_directory)

    indexed = 0
    skipped = 0
    errors = 0

    if not elements:
        logger.warning("No elements provided to index_visual_elements.")
        return {"indexed": 0, "skipped": 0, "errors": 0}

    for el in elements:
        # Determine element type string
        elem_type_str = (
            el.element_type.value
            if hasattr(el.element_type, "value")
            else str(el.element_type)
        ).lower()

        # Step a: Filter to only elements with element_type in ["chart", "image"]
        if elem_type_str not in ("chart", "image"):
            continue

        # Step b: Get image file path from metadata
        img_path_raw = el.metadata.get("image_path") or el.metadata.get("relative_path")
        if not img_path_raw:
            logger.debug(f"Skipping element '{el.element_id}': no image path found in metadata.")
            skipped += 1
            continue

        img_path = Path(img_path_raw)
        if not img_path.exists() or not img_path.is_file():
            logger.warning(f"Skipping element '{el.element_id}': image file does not exist at '{img_path}'.")
            skipped += 1
            continue

        # Step c: Call embed_image() on image path
        try:
            emb = embed_image(str(img_path))
            if not emb or len(emb) == 0:
                logger.error(f"Failed to generate CLIP embedding for image '{img_path}'.")
                errors += 1
                continue
        except Exception as e:
            logger.error(f"Error embedding image for element '{el.element_id}': {e}")
            errors += 1
            continue

        # Step d: Format metadata and upsert into visual collection
        chroma_meta = format_metadata_for_chroma(el)
        # Ensure image_path string is stored
        chroma_meta["image_path"] = str(img_path.resolve())

        try:
            collection.upsert(
                ids=[el.element_id],
                embeddings=[emb],
                documents=[el.content],
                metadatas=[chroma_meta],
            )
            indexed += 1
        except Exception as e:
            logger.error(f"Failed to upsert element '{el.element_id}' into ChromaDB: {e}")
            errors += 1

    summary = {"indexed": indexed, "skipped": skipped, "errors": errors}
    logger.info(f"Visual indexing completed for collection '{collection.name}': {summary}")
    return summary


def visual_search(
    query_text: str,
    collection: Optional[chromadb.Collection] = None,
    n_results: int = 5,
    persist_directory: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Perform cross-modal visual similarity search using CLIP text-to-image embeddings.

    Embeds query_text via embed_text_clip() and queries the visual ChromaDB collection.

    Args:
        query_text: Plain text search query (e.g., 'bar chart of quarterly revenue').
        collection: Target visual ChromaDB collection (defaults to 'financial_rag_visual').
        n_results: Maximum number of top matching visual elements to return.
        persist_directory: Optional vector store directory path.

    Returns:
        List[Dict[str, Any]]: Result list containing dicts matching retriever.py shape:
            - element_id: Unique ID of document element
            - content: Summary/description of the visual element
            - metadata: Flattened Chroma metadata dictionary
            - distance: Cosine distance score
            - similarity: Semantic similarity score (1.0 - distance)
    """
    if not query_text or not query_text.strip():
        logger.warning("Empty query_text passed to visual_search.")
        return []

    if collection is None:
        collection = get_visual_collection(persist_directory=persist_directory)

    try:
        count = collection.count()
        if count == 0:
            logger.warning(f"Visual collection '{collection.name}' is empty.")
            return []
        effective_n_results = min(n_results, count)
    except Exception:
        effective_n_results = n_results

    query_emb = embed_text_clip(query_text)
    if not query_emb or len(query_emb) == 0:
        logger.error(f"Could not generate CLIP query embedding for text: '{query_text}'")
        return []

    try:
        raw_results = collection.query(
            query_embeddings=[query_emb],
            n_results=effective_n_results,
        )
    except Exception as e:
        logger.error(f"Error querying visual collection '{collection.name}': {e}")
        return []

    results: List[Dict[str, Any]] = []

    ids = raw_results.get("ids", [[]])[0] if raw_results.get("ids") else []
    docs = raw_results.get("documents", [[]])[0] if raw_results.get("documents") else []
    metas = raw_results.get("metadatas", [[]])[0] if raw_results.get("metadatas") else []
    dists = raw_results.get("distances", [[]])[0] if raw_results.get("distances") else []

    for i in range(len(ids)):
        elem_id = ids[i]
        doc_content = docs[i] if i < len(docs) else ""
        meta = metas[i] if i < len(metas) else {}
        dist = dists[i] if i < len(dists) else None

        similarity = max(0.0, 1.0 - dist) if dist is not None else None

        results.append({
            "element_id": elem_id,
            "content": doc_content,
            "metadata": meta,
            "distance": dist,
            "similarity": similarity,
        })

    logger.info(f"Visual search for '{query_text}' returned {len(results)} matches.")
    return results


def _load_elements_from_file(json_path: Path) -> List[DocumentElement]:
    """Helper function to load DocumentElements from an enriched JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    elements: List[DocumentElement] = []
    for item in data:
        if isinstance(item, dict):
            elements.append(DocumentElement(**item))
    return elements


def main():
    """CLI entrypoint to demonstrate visual indexing and text-to-image search."""
    parser = argparse.ArgumentParser(
        description="Index image/chart elements into ChromaDB visual collection and run sample visual search."
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default="data/processed/nvidia_financial_highlights_enriched.json",
        help="Path to an enriched JSON file containing chart/image elements.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="financial chart revenue and growth trends",
        help="Text query to search against visual index.",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=None,
        help=f"Path to ChromaDB vector store directory (default: {settings.VECTOR_STORE_PATH}).",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=5,
        help="Number of visual search results to display.",
    )

    args = parser.parse_args()
    json_path = Path(args.json_path)

    print(f"\n{'='*65}")
    print("Multimodal Visual Vector Indexing & Search Test")
    print(f"{'='*65}\n")

    collection = get_visual_collection(persist_directory=args.persist_dir)

    if json_path.exists():
        print(f"1. Loading elements from: '{json_path}'")
        elements = _load_elements_from_file(json_path)
        print(f"   Loaded {len(elements)} total document elements.")

        print("\n2. Indexing chart and image elements into visual collection 'financial_rag_visual'...")
        summary = index_visual_elements(
            elements=elements,
            collection=collection,
        )
        print(f"   Indexing Summary: {summary}")
    else:
        print(f"File not found at '{json_path}'. Skipping visual indexing step.")

    print(f"\n3. Executing Visual Search Query: '{args.query}'")
    results = visual_search(
        query_text=args.query,
        collection=collection,
        n_results=args.n_results,
    )

    if not results:
        print("   No matching visual elements found.")
    else:
        print(f"   Retrieved {len(results)} visual elements:\n")
        for idx, res in enumerate(results, start=1):
            meta = res.get("metadata", {})
            chart_type = meta.get("chart_type") or meta.get("element_type", "N/A")
            sim = res.get("similarity")
            sim_str = f"{sim:.4f}" if sim is not None else "N/A"
            content_preview = res.get("content", "")[:120].replace("\n", " ")

            print(f"   [{idx}] Element ID: {res['element_id']}")
            print(f"       Chart/Elem Type: {chart_type}")
            print(f"       Similarity Score: {sim_str}")
            print(f"       Doc Name:        {meta.get('doc_name', 'N/A')} (Page {meta.get('page_number', 'N/A')})")
            print(f"       Content Preview: {content_preview}...")
            if meta.get("image_path"):
                print(f"       Image Path:      {meta.get('image_path')}")
            print()


if __name__ == "__main__":
    main()
