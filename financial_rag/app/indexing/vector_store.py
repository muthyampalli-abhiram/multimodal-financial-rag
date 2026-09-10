"""ChromaDB vector store initialization, metadata formatting, and upsert operations."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.core.schemas import BoundingBox, DocumentElement
from app.indexing.embedder import get_embedding, get_embeddings_batch

logger = logging.getLogger(__name__)


def get_chroma_client(
    persist_directory: Optional[Union[str, Path]] = None,
) -> chromadb.PersistentClient:
    """Initialize and return a persistent ChromaDB client.

    Args:
        persist_directory: Destination directory for vector storage
                           (defaults to settings.VECTOR_STORE_PATH).

    Returns:
        chromadb.PersistentClient: Persistent ChromaDB client instance.
    """
    path = Path(persist_directory or settings.VECTOR_STORE_PATH)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path.resolve()))


def get_or_create_collection(
    name: str = "financial_rag",
    persist_directory: Optional[Union[str, Path]] = None,
    distance_metric: str = "cosine",
) -> chromadb.Collection:
    """Retrieve or create a ChromaDB collection configured with the specified distance metric.

    Args:
        name: Name of the collection.
        persist_directory: Path to vector store directory.
        distance_metric: Distance metric ('cosine', 'l2', or 'ip'). Default is 'cosine'.

    Returns:
        chromadb.Collection: Chroma collection instance.
    """
    client = get_chroma_client(persist_directory=persist_directory)
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": distance_metric},
    )
    return collection


def format_metadata_for_chroma(element: DocumentElement) -> Dict[str, Union[str, int, float, bool]]:
    """Format and sanitize DocumentElement metadata for ChromaDB storage.

    ChromaDB strictly requires metadata values to be primitive types: str, int, float, or bool.
    Nested dictionaries, lists, and None values are flattened or serialized to JSON strings.

    Args:
        element: Source DocumentElement instance.

    Returns:
        Dict[str, Union[str, int, float, bool]]: Chroma-compatible metadata dictionary.
    """
    elem_type_str = (
        element.element_type.value
        if hasattr(element.element_type, "value")
        else str(element.element_type)
    )

    chroma_meta: Dict[str, Union[str, int, float, bool]] = {
        "doc_name": str(element.doc_name),
        "page_number": int(element.page_number),
        "element_type": elem_type_str,
    }

    # Bounding box serialization
    if element.bbox is not None:
        if isinstance(element.bbox, BoundingBox):
            chroma_meta["bbox"] = json.dumps(list(element.bbox.to_tuple()))
        elif isinstance(element.bbox, (list, tuple)):
            chroma_meta["bbox"] = json.dumps(list(element.bbox))
        else:
            chroma_meta["bbox"] = str(element.bbox)
    else:
        chroma_meta["bbox"] = ""

    # Flatten structured multimodal analysis if present
    gemini_analysis = element.metadata.get("gemini_analysis")
    if isinstance(gemini_analysis, dict):
        if "chart_type" in gemini_analysis:
            chroma_meta["chart_type"] = str(gemini_analysis.get("chart_type") or "")
        if "title" in gemini_analysis:
            chroma_meta["chart_title"] = str(gemini_analysis.get("title") or "")
        if "is_financial" in gemini_analysis:
            chroma_meta["is_financial"] = bool(gemini_analysis.get("is_financial", False))
        if "time_period" in gemini_analysis:
            chroma_meta["time_period"] = str(gemini_analysis.get("time_period") or "")

    # Flatten other metadata fields safely
    for key, value in element.metadata.items():
        if key in ("gemini_analysis", "doc_name", "page_number", "element_type", "bbox"):
            continue
        if isinstance(value, (str, int, float, bool)):
            chroma_meta[key] = value
        elif value is None:
            chroma_meta[key] = ""
        else:
            try:
                chroma_meta[key] = json.dumps(value, ensure_ascii=False)
            except Exception:
                chroma_meta[key] = str(value)

    # Lossless JSON snapshot of the full metadata
    try:
        chroma_meta["metadata_json"] = json.dumps(element.metadata, ensure_ascii=False)
    except Exception:
        chroma_meta["metadata_json"] = "{}"

    return chroma_meta


def add_elements_to_collection(
    collection: chromadb.Collection,
    elements: List[DocumentElement],
    embeddings: Optional[List[List[float]]] = None,
    batch_size: int = 100,
    delay_seconds: float = 0.0,
) -> int:
    """Index and upsert a list of DocumentElements into a ChromaDB collection.

    Generates dense embeddings if not provided, formats metadata into Chroma-compatible
    primitives, and batches upsert operations.

    Args:
        collection: Target ChromaDB collection.
        elements: List of DocumentElement instances to index.
        embeddings: Optional precomputed embeddings matching elements 1-to-1.
        batch_size: Number of records to upsert per Chroma batch call (default: 100).
        delay_seconds: Optional delay passed to embedding generator between batches.

    Returns:
        int: Number of elements successfully added/updated in the collection.
    """
    if not elements:
        logger.warning("No elements provided to add_elements_to_collection.")
        return 0

    # Compute embeddings if not precomputed
    if embeddings is None:
        texts_to_embed = [el.content for el in elements]
        logger.info(f"Generating embeddings for {len(texts_to_embed)} elements...")
        embeddings = get_embeddings_batch(
            texts=texts_to_embed,
            task_type="retrieval_document",
            delay_seconds=delay_seconds,
        )

    # Filter out elements with failed/empty embeddings
    valid_ids: List[str] = []
    valid_docs: List[str] = []
    valid_metas: List[Dict[str, Union[str, int, float, bool]]] = []
    valid_embs: List[List[float]] = []

    for el, emb in zip(elements, embeddings):
        if not emb or len(emb) == 0:
            logger.warning(
                f"Skipping element '{el.element_id}' (page {el.page_number}) due to empty embedding."
            )
            continue

        valid_ids.append(el.element_id)
        valid_docs.append(el.content)
        valid_metas.append(format_metadata_for_chroma(el))
        valid_embs.append(emb)

    if not valid_ids:
        logger.error("No valid elements with embeddings to upsert into ChromaDB.")
        return 0

    # Upsert in batches
    total_indexed = 0
    for i in range(0, len(valid_ids), batch_size):
        b_ids = valid_ids[i : i + batch_size]
        b_docs = valid_docs[i : i + batch_size]
        b_metas = valid_metas[i : i + batch_size]
        b_embs = valid_embs[i : i + batch_size]

        collection.upsert(
            ids=b_ids,
            documents=b_docs,
            metadatas=b_metas,
            embeddings=b_embs,
        )
        total_indexed += len(b_ids)

    logger.info(f"Successfully upserted {total_indexed} elements into collection '{collection.name}'.")
    return total_indexed


def query_collection(
    collection: chromadb.Collection,
    query_text: Optional[str] = None,
    query_embedding: Optional[List[float]] = None,
    n_results: int = 5,
    where: Optional[Dict[str, Any]] = None,
    where_document: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Query a ChromaDB collection using vector embedding or query text.

    Args:
        collection: Target ChromaDB collection.
        query_text: Plain text query (embedded via get_embedding if query_embedding not provided).
        query_embedding: Optional precomputed query embedding vector.
        n_results: Number of top results to retrieve.
        where: Optional metadata filter dict.
        where_document: Optional document text filter dict.

    Returns:
        Dict[str, Any]: ChromaDB query result dictionary containing ids, documents,
                        metadatas, and distances.
    """
    if query_embedding is None and query_text is not None:
        query_embedding = get_embedding(query_text, task_type="retrieval_query")

    if query_embedding is None or len(query_embedding) == 0:
        raise ValueError("Must provide either a non-empty query_embedding or query_text.")

    query_kwargs: Dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
    }
    if where:
        query_kwargs["where"] = where
    if where_document:
        query_kwargs["where_document"] = where_document

    return collection.query(**query_kwargs)
