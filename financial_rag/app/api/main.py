"""FastAPI application exposing the Multimodal Financial Intelligence RAG system."""

from collections import Counter
import logging
from pathlib import Path
import shutil
import time
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api.schemas import (
    CollectionStatsResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
)
from app.core.config import settings
from app.core.schemas import QueryResult
from app.generation.pipeline import answer_query
from app.indexing.pipeline import index_document_elements
from app.indexing.vector_store import get_or_create_collection
from app.ingestion.pipeline import ingest_pdf
from app.understanding.pipeline import (
    enrich_elements_with_understanding,
    save_elements_to_json,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multimodal Financial Intelligence RAG API",
    description=(
        "Production-ready REST API for ingesting SEC filings / financial reports, "
        "extracting multimodal charts/tables, generating dense Gemini embeddings, "
        "and querying grounded financial intelligence backed by verifiable citations."
    ),
    version="1.0.0",
)

# Enable CORS for flexible integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
)
def health_check() -> HealthResponse:
    """Return system operational and health status."""
    return HealthResponse(status="ok")


@app.post(
    "/query",
    response_model=QueryResult,
    tags=["Financial Q&A"],
    summary="Ask financial queries",
)
def query_financial_rag(request: QueryRequest) -> QueryResult:
    """Synthesize grounded analyst answers with citations from indexed financial disclosures.

    Executes intent routing, vector similarity retrieval with fallback, and Gemini LLM reasoning.
    """
    try:
        logger.info(f"Received query request: '{request.query}' (collection='{request.collection_name}')")
        result = answer_query(
            query=request.query,
            collection_name=request.collection_name,
            n_results=request.n_results,
        )
        return result
    except Exception as e:
        logger.exception(f"Failed to process query '{request.query}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}",
        ) from e


@app.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["Ingestion & Indexing"],
    summary="Upload and process a financial PDF",
)
def ingest_document(
    file: UploadFile = File(..., description="Financial filing PDF file to upload and process"),
    max_calls: Optional[int] = Query(
        default=None,
        description="Optional cap on new Gemini Vision multimodal API calls for chart analysis",
    ),
    collection_name: str = Query(
        default="financial_rag",
        description="Target ChromaDB collection name for vector indexing",
    ),
) -> IngestResponse:
    """Upload a financial PDF and execute the end-to-end ingestion and indexing pipeline.

    Workflow:
    1. Saves uploaded PDF to data/raw_pdfs/
    2. Ingests PDF text, financial tables, and visual chart crops
    3. Enriches visual elements with Gemini Multimodal Vision analysis
    4. Persists intermediate enriched JSON artifact
    5. Generates dense Gemini vector embeddings and indexes into ChromaDB

    Note:
    This endpoint executes the full multimodal extraction and indexing pipeline synchronously.
    Processing multi-page filings with multiple charts can take several seconds to minutes.
    (Background job queueing / async workers can be integrated in future phases).
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported for ingestion.")

    start_time = time.perf_counter()
    raw_pdf_dir = Path("data/raw_pdfs")
    raw_pdf_dir.mkdir(parents=True, exist_ok=True)
    saved_pdf_path = raw_pdf_dir / file.filename

    try:
        logger.info(f"Saving uploaded PDF to '{saved_pdf_path.resolve()}'...")
        with open(saved_pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Step 1: Raw PDF Ingestion
        logger.info(f"Extracting elements from '{file.filename}'...")
        raw_elements = ingest_pdf(
            file_path=saved_pdf_path,
            output_image_dir="data/processed/images",
        )

        # Step 2: Gemini Multimodal Vision Enrichment
        logger.info(f"Enriching {len(raw_elements)} elements with multimodal understanding...")
        enriched_elements = enrich_elements_with_understanding(
            elements=raw_elements,
            max_calls=max_calls,
        )

        # Step 3: Persist enriched JSON artifact
        doc_slug = saved_pdf_path.stem
        output_json_path = Path("data/processed") / f"{doc_slug}_enriched.json"
        save_elements_to_json(enriched_elements, output_json_path)

        # Step 4: Vector Embedding & ChromaDB Indexing
        logger.info(f"Indexing elements into ChromaDB collection '{collection_name}'...")
        indexed_count = index_document_elements(
            elements=enriched_elements,
            collection_name=collection_name,
        )

        elapsed = time.perf_counter() - start_time

        # Calculate breakdown counts by element type
        breakdown = Counter(
            el.element_type.value if hasattr(el.element_type, "value") else str(el.element_type)
            for el in enriched_elements
        )

        logger.info(
            f"Completed ingestion of '{file.filename}' in {elapsed:.2f}s "
            f"({indexed_count}/{len(enriched_elements)} elements indexed)."
        )

        return IngestResponse(
            filename=file.filename,
            total_elements=len(enriched_elements),
            elements_indexed=indexed_count,
            processing_time_seconds=round(elapsed, 2),
            breakdown=dict(breakdown),
        )

    except Exception as e:
        logger.exception(f"Ingestion pipeline failed for '{file.filename}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion pipeline failed for '{file.filename}': {str(e)}",
        ) from e


@app.get(
    "/collections/{collection_name}/stats",
    response_model=CollectionStatsResponse,
    tags=["Vector Store"],
    summary="Get collection statistics",
)
def get_collection_stats(collection_name: str) -> CollectionStatsResponse:
    """Retrieve metadata and indexed record counts for a given ChromaDB collection."""
    try:
        collection = get_or_create_collection(name=collection_name)
        count = collection.count()
        return CollectionStatsResponse(
            collection_name=collection_name,
            total_documents=count,
        )
    except Exception as e:
        logger.exception(f"Failed to fetch stats for collection '{collection_name}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve statistics for collection '{collection_name}': {str(e)}",
        ) from e


if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )
