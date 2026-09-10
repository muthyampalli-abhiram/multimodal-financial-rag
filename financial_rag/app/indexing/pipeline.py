"""Pipeline and CLI entrypoint for embedding generation and vector store indexing."""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Union

from app.core.config import settings
from app.core.schemas import DocumentElement, ElementType
from app.indexing.embedder import get_embeddings_batch
from app.indexing.vector_store import (
    add_elements_to_collection,
    get_chroma_client,
    get_or_create_collection,
)
from app.understanding.pipeline import load_elements_from_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def index_document_elements(
    elements: List[DocumentElement],
    collection_name: str = "financial_rag",
    persist_directory: Optional[Union[str, Path]] = None,
    batch_size: int = 32,
    delay_seconds: float = 0.0,
) -> int:
    """Index a list of DocumentElements into the persistent ChromaDB vector store.

    Args:
        elements: List of DocumentElements to index.
        collection_name: Target Chroma collection name (default: 'financial_rag').
        persist_directory: Path to vector store directory.
        batch_size: Batch size for embedding and vector store upsert.
        delay_seconds: Delay in seconds between embedding batches.

    Returns:
        int: Number of elements successfully indexed.
    """
    if not elements:
        logger.warning("No elements provided for indexing.")
        return 0

    collection = get_or_create_collection(
        name=collection_name,
        persist_directory=persist_directory,
    )

    logger.info(
        f"Starting indexing of {len(elements)} elements into collection '{collection_name}'..."
    )

    texts_to_embed = [el.content for el in elements]
    embeddings = get_embeddings_batch(
        texts=texts_to_embed,
        task_type="retrieval_document",
        batch_size=batch_size,
        delay_seconds=delay_seconds,
    )

    indexed_count = add_elements_to_collection(
        collection=collection,
        elements=elements,
        embeddings=embeddings,
        batch_size=batch_size,
    )

    return indexed_count


def index_enriched_json(
    json_path: Union[str, Path],
    collection_name: str = "financial_rag",
    persist_directory: Optional[Union[str, Path]] = None,
    batch_size: int = 32,
    delay_seconds: float = 0.0,
) -> int:
    """Load enriched document elements from a JSON file and index them into ChromaDB.

    Args:
        json_path: Path to enriched JSON file.
        collection_name: Target Chroma collection name.
        persist_directory: Path to vector store directory.
        batch_size: Batch size for embedding and vector store upsert.
        delay_seconds: Delay in seconds between embedding batches.

    Returns:
        int: Number of elements indexed.
    """
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"Enriched JSON file not found at: {path.resolve()}")

    logger.info(f"Loading enriched elements from '{path.resolve()}'...")
    elements = load_elements_from_json(path)
    return index_document_elements(
        elements=elements,
        collection_name=collection_name,
        persist_directory=persist_directory,
        batch_size=batch_size,
        delay_seconds=delay_seconds,
    )


def main():
    """CLI entrypoint for vector indexing of enriched financial documents."""
    parser = argparse.ArgumentParser(
        description="Index enriched financial document elements into ChromaDB using Gemini embeddings."
    )
    parser.add_argument(
        "json_path",
        type=str,
        help="Path to the enriched document JSON file (e.g. data/processed/nvidia_test_subset_enriched.json)",
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
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation and upsert (default: 32)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay in seconds between embedding batch calls (default: 0.0)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset/delete the existing collection before indexing",
    )

    args = parser.parse_args()
    json_path = Path(args.json_path)

    if not json_path.exists():
        print(f"Error: File not found at '{json_path}'", file=sys.stderr)
        sys.exit(1)

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

    persist_dir = args.persist_dir or settings.VECTOR_STORE_PATH

    print(f"\n{'='*65}")
    print(f"Starting Vector Indexing Pipeline: {json_path.name}")
    print(f"{'='*65}\n")

    try:
        elements = load_elements_from_json(json_path)
        print(f"Loaded {len(elements)} elements from '{json_path.name}'.")

        client = get_chroma_client(persist_directory=persist_dir)
        if args.reset:
            try:
                client.delete_collection(name=args.collection)
                print(f"Reset: Deleted existing collection '{args.collection}'.")
            except Exception:
                pass

        collection = get_or_create_collection(
            name=args.collection,
            persist_directory=persist_dir,
        )

        counts = Counter(
            el.element_type.value if hasattr(el.element_type, "value") else str(el.element_type)
            for el in elements
        )

        print(f"\nIndexing {len(elements)} elements using embedding model '{settings.EMBEDDING_MODEL_NAME}'...")
        indexed_count = index_document_elements(
            elements=elements,
            collection_name=args.collection,
            persist_directory=persist_dir,
            batch_size=args.batch_size,
            delay_seconds=args.delay,
        )

        total_in_collection = collection.count()

        print(f"\n{'-'*65}")
        print("VECTOR INDEXING SUMMARY")
        print(f"{'-'*65}")
        print(f"Source File:               {json_path.name}")
        print(f"Collection Name:           {args.collection}")
        print(f"Vector Store Directory:    {Path(persist_dir).resolve()}")
        print(f"Embedding Model:           {settings.EMBEDDING_MODEL_NAME}")
        print(f"Total Elements Loaded:     {len(elements)}")
        print(f"  - Text blocks:           {counts.get(ElementType.TEXT.value, 0)}")
        print(f"  - Tables:                {counts.get(ElementType.TABLE.value, 0)}")
        print(f"  - Charts:                {counts.get(ElementType.CHART.value, 0)}")
        print(f"  - Images:                {counts.get(ElementType.IMAGE.value, 0)}")
        print(f"Elements Successfully Indexed: {indexed_count}")
        print(f"Total Records in Collection:   {total_in_collection}")
        print(f"{'-'*65}\n")

    except Exception as e:
        print(f"\nVector indexing failed: {e}", file=sys.stderr)
        logger.exception("Vector indexing pipeline failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
