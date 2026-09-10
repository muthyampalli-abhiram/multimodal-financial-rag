"""Embeddings generation and vector store indexing module."""

from app.indexing.embedder import (
    configure_genai,
    get_embedding,
    get_embeddings_batch,
)
from app.indexing.vector_store import (
    add_elements_to_collection,
    format_metadata_for_chroma,
    get_chroma_client,
    get_or_create_collection,
    query_collection,
)


def __getattr__(name: str):
    if name in ("index_document_elements", "index_enriched_json"):
        import app.indexing.pipeline as pipeline_mod

        return getattr(pipeline_mod, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "configure_genai",
    "get_embedding",
    "get_embeddings_batch",
    "get_chroma_client",
    "get_or_create_collection",
    "format_metadata_for_chroma",
    "add_elements_to_collection",
    "query_collection",
    "index_document_elements",
    "index_enriched_json",
]
