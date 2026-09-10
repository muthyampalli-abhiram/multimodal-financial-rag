"""Unit and integration tests for FastAPI endpoints."""

from io import BytesIO
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.core.schemas import Citation, DocumentElement, ElementType, QueryResult

client = TestClient(app)


def test_health_endpoint():
    """Test GET /health returns 200 and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_endpoint_success():
    """Test POST /query returns 200 with valid QueryResult schema."""
    fake_result = QueryResult(
        answer="NVIDIA revenue grew to $60.9B in FY24 [Source: elem_1].",
        citations=[
            Citation(
                element_id="elem_1",
                doc_name="NVDA_FY24.pdf",
                page_number=1,
                element_type=ElementType.TEXT,
                snippet="Total revenue was $60.9B.",
            )
        ],
        retrieved_elements=[
            DocumentElement(
                element_id="elem_1",
                doc_name="NVDA_FY24.pdf",
                page_number=1,
                element_type=ElementType.TEXT,
                content="Total revenue was $60.9B.",
            )
        ],
    )

    with patch("app.api.main.answer_query", return_value=fake_result) as mock_answer:
        payload = {
            "query": "What was NVIDIA revenue?",
            "collection_name": "financial_rag",
            "n_results": 5,
        }
        response = client.post("/query", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "NVIDIA revenue grew" in data["answer"]
        assert len(data["citations"]) == 1
        assert data["citations"][0]["element_id"] == "elem_1"
        assert len(data["retrieved_elements"]) == 1

        mock_answer.assert_called_once_with(
            query="What was NVIDIA revenue?",
            collection_name="financial_rag",
            n_results=5,
        )


def test_query_endpoint_error_handling():
    """Test POST /query returns 500 when pipeline raises an exception."""
    with patch("app.api.main.answer_query", side_effect=RuntimeError("Vector database timeout")):
        payload = {"query": "What is the operating income?"}
        response = client.post("/query", json=payload)

        assert response.status_code == 500
        assert "Query processing failed" in response.json()["detail"]


def test_ingest_endpoint_success(tmp_path):
    """Test POST /ingest handles PDF upload and executes ingestion pipeline."""
    sample_pdf_bytes = b"%PDF-1.4 \n%mock pdf header\n%%EOF"
    sample_elements = [
        DocumentElement(
            element_id="el_1",
            doc_name="sample_filing.pdf",
            page_number=1,
            element_type=ElementType.TEXT,
            content="Sample text from filing.",
        ),
        DocumentElement(
            element_id="el_2",
            doc_name="sample_filing.pdf",
            page_number=2,
            element_type=ElementType.CHART,
            content="[BAR: Revenue Trend] 20% growth.",
        ),
    ]

    with patch("app.api.main.ingest_pdf", return_value=sample_elements):
        with patch("app.api.main.enrich_elements_with_understanding", return_value=sample_elements):
            with patch("app.api.main.save_elements_to_json"):
                with patch("app.api.main.index_document_elements", return_value=2):
                    files = {
                        "file": ("sample_filing.pdf", BytesIO(sample_pdf_bytes), "application/pdf")
                    }
                    response = client.post(
                        "/ingest",
                        files=files,
                        params={"max_calls": 5, "collection_name": "test_collection"},
                    )

                    assert response.status_code == 200
                    data = response.json()
                    assert data["filename"] == "sample_filing.pdf"
                    assert data["total_elements"] == 2
                    assert data["elements_indexed"] == 2
                    assert "processing_time_seconds" in data
                    assert data["breakdown"]["text"] == 1
                    assert data["breakdown"]["chart"] == 1


def test_ingest_endpoint_non_pdf():
    """Test POST /ingest rejects non-PDF file uploads with 400 Bad Request."""
    files = {"file": ("notes.txt", BytesIO(b"plain text file content"), "text/plain")}
    response = client.post("/ingest", files=files)

    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]


def test_collection_stats_endpoint():
    """Test GET /collections/{collection_name}/stats returns document count."""
    mock_coll = MagicMock()
    mock_coll.count.return_value = 88

    with patch("app.api.main.get_or_create_collection", return_value=mock_coll):
        response = client.get("/collections/financial_rag/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["collection_name"] == "financial_rag"
        assert data["total_documents"] == 88
