"""Unit and integration tests for embeddings generation and ChromaDB vector indexing."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from app.core.schemas import BoundingBox, DocumentElement, ElementType
from app.indexing.embedder import (
    configure_genai,
    get_embedding,
    get_embeddings_batch,
)
from app.indexing.pipeline import (
    index_document_elements,
    index_enriched_json,
)
from app.indexing.vector_store import (
    add_elements_to_collection,
    format_metadata_for_chroma,
    get_chroma_client,
    get_or_create_collection,
    query_collection,
)
from app.understanding.pipeline import save_elements_to_json


def test_configure_genai_missing_key():
    """Test configure_genai raises ValueError when GEMINI_API_KEY is empty."""
    with patch("app.core.config.settings.GEMINI_API_KEY", ""):
        with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
            configure_genai(api_key="")


def test_get_embedding_empty_text():
    """Test get_embedding returns empty list for empty or whitespace strings."""
    assert get_embedding("") == []
    assert get_embedding("   \n\t  ") == []


def test_get_embedding_success():
    """Test get_embedding successfully calls Gemini API and returns embedding vector."""
    fake_vector = [0.12, -0.34, 0.56, 0.78]

    with patch("app.indexing.embedder.configure_genai"):
        with patch("google.generativeai.embed_content") as mock_embed:
            mock_embed.return_value = {"embedding": fake_vector}

            res = get_embedding(
                text="Financial analysis of NVIDIA Q4 earnings",
                task_type="retrieval_document",
                title="NVIDIA 10-K",
                model_name="models/text-embedding-004",
            )

            assert res == fake_vector
            mock_embed.assert_called_once_with(
                model="models/text-embedding-004",
                content="Financial analysis of NVIDIA Q4 earnings",
                task_type="retrieval_document",
                title="NVIDIA 10-K",
            )


def test_get_embedding_retry_on_failure():
    """Test get_embedding retries upon transient exception and returns vector on second attempt."""
    fake_vector = [0.5, 0.6, 0.7]

    with patch("app.indexing.embedder.configure_genai"):
        with patch("google.generativeai.embed_content") as mock_embed:
            # First attempt raises Exception, second succeeds
            mock_embed.side_effect = [
                RuntimeError("Transient 429 Quota / Rate limit error"),
                {"embedding": fake_vector},
            ]

            res = get_embedding(
                text="Operating margin trend",
                max_retries=2,
                retry_delay=0.01,
            )

            assert res == fake_vector
            assert mock_embed.call_count == 2


def test_get_embedding_all_retries_fail_returns_empty_list():
    """Test get_embedding returns empty list without crashing when all retries fail."""
    with patch("app.indexing.embedder.configure_genai"):
        with patch("google.generativeai.embed_content") as mock_embed:
            mock_embed.side_effect = RuntimeError("Persistent API error")

            res = get_embedding(
                text="Failing text embedding",
                max_retries=2,
                retry_delay=0.01,
            )

            assert res == []
            assert mock_embed.call_count == 3


def test_get_embeddings_batch_success():
    """Test get_embeddings_batch returns list of vectors matching input texts."""
    texts = [
        "Revenue increased 122% to $60.9 billion.",
        "Data Center revenue was a record $47.5 billion.",
        "Gross margin expanded to 72.7%.",
    ]
    fake_embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]

    with patch("app.indexing.embedder.configure_genai"):
        with patch("google.generativeai.embed_content") as mock_embed:
            mock_embed.return_value = {"embedding": fake_embeddings}

            results = get_embeddings_batch(texts=texts, batch_size=32)

            assert len(results) == 3
            assert results == fake_embeddings
            mock_embed.assert_called_once()


def test_get_embeddings_batch_empty():
    """Test get_embeddings_batch returns empty list for empty input."""
    assert get_embeddings_batch([]) == []


def test_get_embeddings_batch_fallback_on_batch_error():
    """Test get_embeddings_batch falls back to per-item embedding when batch fails."""
    texts = ["Text item 1", "Text item 2"]
    fallback_vec1 = [0.1, 0.1]
    fallback_vec2 = [0.2, 0.2]

    with patch("app.indexing.embedder.configure_genai"):
        with patch("google.generativeai.embed_content") as mock_embed:
            # Batch call fails, subsequent individual calls succeed
            mock_embed.side_effect = [
                RuntimeError("Batch payload too large"),  # attempt 1
                RuntimeError("Batch payload too large"),  # attempt 2
                {"embedding": fallback_vec1},  # individual 1
                {"embedding": fallback_vec2},  # individual 2
            ]

            results = get_embeddings_batch(
                texts=texts,
                batch_size=2,
                max_retries=1,
                retry_delay=0.01,
            )

            assert len(results) == 2
            assert results[0] == fallback_vec1
            assert results[1] == fallback_vec2


def test_format_metadata_for_chroma():
    """Test format_metadata_for_chroma flattens and sanitizes metadata to primitive types."""
    element = DocumentElement(
        element_id="test_el_1",
        doc_name="NVDA_2023_10K.pdf",
        page_number=3,
        element_type=ElementType.CHART,
        content="[BAR: Revenue Trend] Strong growth in Data Center segment.",
        bbox=BoundingBox(x0=50.0, y0=100.0, x1=500.0, y1=400.0),
        metadata={
            "gemini_analysis": {
                "chart_type": "bar",
                "title": "Revenue by Segment",
                "summary": "Data center surged.",
                "data_points": [{"label": "Data Center", "value": "$47.5B"}],
                "time_period": "FY24",
                "is_financial": True,
            },
            "ticker": "NVDA",
            "fiscal_year": 2024,
            "tags": ["revenue", "growth"],
        },
    )

    meta = format_metadata_for_chroma(element)

    assert meta["doc_name"] == "NVDA_2023_10K.pdf"
    assert meta["page_number"] == 3
    assert meta["element_type"] == "chart"
    assert meta["chart_type"] == "bar"
    assert meta["chart_title"] == "Revenue by Segment"
    assert meta["is_financial"] is True
    assert meta["time_period"] == "FY24"
    assert meta["ticker"] == "NVDA"
    assert meta["fiscal_year"] == 2024
    assert isinstance(meta["bbox"], str)
    assert json.loads(meta["bbox"]) == [50.0, 100.0, 500.0, 400.0]

    # Verify all values in metadata dictionary are ChromaDB compatible primitives
    for k, v in meta.items():
        assert isinstance(v, (str, int, float, bool)), f"Non-primitive field {k}: {type(v)}"


def test_chroma_vector_store_crud(tmp_path: Path):
    """Test ChromaDB client creation, collection lifecycle, upsert, and query."""
    persist_dir = tmp_path / "test_vector_store"
    client = get_chroma_client(persist_directory=persist_dir)
    assert client is not None

    collection = get_or_create_collection(
        name="test_financial_collection",
        persist_directory=persist_dir,
    )
    assert collection.name == "test_financial_collection"

    elements = [
        DocumentElement(
            element_id="elem_1",
            doc_name="AAPL_10K.pdf",
            page_number=1,
            element_type=ElementType.TEXT,
            content="Total revenue was $383 billion in 2023.",
        ),
        DocumentElement(
            element_id="elem_2",
            doc_name="MSFT_10K.pdf",
            page_number=2,
            element_type=ElementType.TEXT,
            content="Microsoft Cloud revenue grew 27% to $110 billion.",
        ),
    ]

    mock_embeddings = [
        [0.9, 0.1, 0.0],
        [0.0, 0.1, 0.9],
    ]

    indexed_count = add_elements_to_collection(
        collection=collection,
        elements=elements,
        embeddings=mock_embeddings,
    )

    assert indexed_count == 2
    assert collection.count() == 2

    # Query collection with mock vector
    query_vec = [0.85, 0.1, 0.0]
    query_res = query_collection(
        collection=collection,
        query_embedding=query_vec,
        n_results=1,
    )

    assert len(query_res["ids"][0]) == 1
    assert query_res["ids"][0][0] == "elem_1"
    assert "Total revenue was $383 billion" in query_res["documents"][0][0]


def test_index_document_elements_pipeline(tmp_path: Path):
    """Test index_document_elements orchestration pipeline."""
    persist_dir = tmp_path / "pipeline_vector_store"

    elements = [
        DocumentElement(
            element_id="pipe_1",
            doc_name="NVIDIA_Report.pdf",
            page_number=1,
            element_type=ElementType.TEXT,
            content="Compute & Networking revenue grew 217%.",
        ),
        DocumentElement(
            element_id="pipe_2",
            doc_name="NVIDIA_Report.pdf",
            page_number=2,
            element_type=ElementType.CHART,
            content="[BAR: Data Center] Data center revenue quarterly breakdown.",
            metadata={"gemini_analysis": {"chart_type": "bar", "is_financial": True}},
        ),
    ]

    fake_embs = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    with patch("app.indexing.pipeline.get_embeddings_batch", return_value=fake_embs):
        indexed = index_document_elements(
            elements=elements,
            collection_name="test_pipeline_coll",
            persist_directory=persist_dir,
        )

        assert indexed == 2

        collection = get_or_create_collection(
            name="test_pipeline_coll",
            persist_directory=persist_dir,
        )
        assert collection.count() == 2


def test_index_enriched_json_pipeline(tmp_path: Path):
    """Test index_enriched_json loading from file and persisting to ChromaDB."""
    json_path = tmp_path / "enriched_test.json"
    persist_dir = tmp_path / "json_vector_store"

    elements = [
        DocumentElement(
            element_id="json_el_1",
            doc_name="Sample_Doc.pdf",
            page_number=1,
            element_type=ElementType.TABLE,
            content="| Revenue | $100M |\n| Net Income | $20M |",
        )
    ]
    save_elements_to_json(elements, json_path)

    fake_embs = [[0.1, 0.3, 0.5]]

    with patch("app.indexing.pipeline.get_embeddings_batch", return_value=fake_embs):
        indexed = index_enriched_json(
            json_path=json_path,
            collection_name="json_test_coll",
            persist_directory=persist_dir,
        )

        assert indexed == 1

        collection = get_or_create_collection(
            name="json_test_coll",
            persist_directory=persist_dir,
        )
        assert collection.count() == 1
