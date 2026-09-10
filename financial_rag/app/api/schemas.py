"""Pydantic data models for FastAPI request and response payloads."""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(default="ok", description="Service operational status")


class QueryRequest(BaseModel):
    """Financial Q&A query request schema."""

    query: str = Field(
        ...,
        description="Analyst question or query regarding financial disclosures",
        examples=["What was NVIDIA's revenue and gross margin trend in FY24?"],
    )
    collection_name: str = Field(
        default="financial_rag",
        description="Name of the ChromaDB collection to retrieve from",
    )
    n_results: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Maximum number of candidate elements to retrieve for context",
    )


class IngestResponse(BaseModel):
    """Response schema for PDF ingestion and multimodal indexing."""

    filename: str = Field(..., description="Filename of the ingested PDF document")
    total_elements: int = Field(..., description="Total document elements extracted (text, tables, charts)")
    elements_indexed: int = Field(..., description="Total elements successfully indexed into vector store")
    processing_time_seconds: float = Field(..., description="Total processing wall-clock duration in seconds")
    breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Breakdown count of elements by element type (text, table, chart, image)",
    )


class CollectionStatsResponse(BaseModel):
    """Response schema for ChromaDB collection statistics."""

    collection_name: str = Field(..., description="Name of the ChromaDB collection")
    total_documents: int = Field(..., description="Total number of elements indexed in the collection")
