"""Unit and integration tests for hybrid and vector retrieval."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from app.core.schemas import DocumentElement, ElementType
from app.indexing.vector_store import (
    add_elements_to_collection,
    get_chroma_client,
    get_or_create_collection,
)
from app.retrieval.pipeline import (
    retrieve_for_query,
    retrieve_with_fallback,
)
from app.retrieval.query_analyzer import detect_query_intent
from app.retrieval.retriever import (
    build_where_filter,
    elements_from_retrieval,
    format_retrieved_context,
    retrieve_relevant_elements,
)


# =====================================================================
# Query Analyzer Tests
# =====================================================================


def test_detect_query_intent_financial_and_years():
    """Test detect_query_intent with multi-year financial comparison."""
    query = "What was NVIDIA's revenue and gross margin in FY23 vs FY24?"
    intent = detect_query_intent(query)

    assert intent["wants_financial_data"] is True
    assert intent["wants_comparison"] is True
    assert len(intent["mentioned_years"]) == 2
    assert "FY23" in intent["mentioned_years"]
    assert "FY24" in intent["mentioned_years"]


def test_detect_query_intent_comparison_keywords():
    """Test detect_query_intent detecting comparison keywords like 'over time'."""
    query = "Compare segment growth and TSR trends over time"
    intent = detect_query_intent(query)

    assert intent["wants_financial_data"] is True
    assert intent["wants_comparison"] is True


def test_detect_query_intent_non_financial():
    """Test detect_query_intent for general non-financial queries."""
    query = "Who is the corporate secretary and where is the principal executive office located?"
    intent = detect_query_intent(query)

    assert intent["wants_financial_data"] is False
    assert intent["wants_comparison"] is False
    assert intent["mentioned_years"] == []


def test_detect_query_intent_calendar_years():
    """Test detect_query_intent extracting 4-digit calendar years."""
    query = "Operating income for 2022, 2023, and 2024"
    intent = detect_query_intent(query)

    assert intent["wants_financial_data"] is True
    assert intent["wants_comparison"] is True
    assert intent["mentioned_years"] == ["2022", "2023", "2024"]


def test_detect_query_intent_empty():
    """Test detect_query_intent on empty string."""
    intent = detect_query_intent("   ")
    assert intent["wants_financial_data"] is False
    assert intent["wants_comparison"] is False
    assert intent["mentioned_years"] == []


# =====================================================================
# Filter & Retrieval Tests
# =====================================================================


def test_build_where_filter_empty():
    """Test build_where_filter returns None when no filters are set."""
    assert build_where_filter() is None
    assert build_where_filter(filter_financial_only=False, element_types=[]) is None


def test_build_where_filter_financial_only():
    """Test build_where_filter with filter_financial_only=True."""
    f = build_where_filter(filter_financial_only=True)
    assert f == {
        "$or": [
            {"element_type": {"$in": ["text", "table"]}},
            {"is_financial": {"$eq": True}},
        ]
    }


def test_build_where_filter_single_element_type():
    """Test build_where_filter with a single element type string or enum."""
    f1 = build_where_filter(element_types=["chart"])
    assert f1 == {"element_type": {"$eq": "chart"}}

    f2 = build_where_filter(element_types=[ElementType.TABLE])
    assert f2 == {"element_type": {"$eq": "table"}}


def test_build_where_filter_multiple_element_types():
    """Test build_where_filter with multiple element types."""
    f = build_where_filter(element_types=["chart", "table", "text"])
    assert f == {"element_type": {"$in": ["chart", "table", "text"]}}


def test_build_where_filter_combined():
    """Test build_where_filter combining financial flag, element types, and custom where."""
    f = build_where_filter(
        filter_financial_only=True,
        element_types=["chart", "table"],
        custom_where={"doc_name": {"$eq": "NVIDIA_10K.pdf"}},
    )
    assert f == {
        "$and": [
            {
                "$or": [
                    {"element_type": {"$in": ["text", "table"]}},
                    {"is_financial": {"$eq": True}},
                ]
            },
            {"element_type": {"$in": ["chart", "table"]}},
            {"doc_name": {"$eq": "NVIDIA_10K.pdf"}},
        ]
    }


def test_retrieve_relevant_elements_empty_query():
    """Test retrieve_relevant_elements returns empty list for empty query."""
    assert retrieve_relevant_elements("") == []
    assert retrieve_relevant_elements("   ") == []


def test_retrieve_relevant_elements_embedding_task_type(tmp_path: Path):
    """Test retrieve_relevant_elements uses task_type='retrieval_query'."""
    persist_dir = tmp_path / "test_store"
    coll = get_or_create_collection("test_coll", persist_directory=persist_dir)

    with patch("app.retrieval.retriever.get_embedding") as mock_embed:
        mock_embed.return_value = [0.1, 0.2, 0.3]
        retrieve_relevant_elements("What is NVIDIA revenue?", collection=coll)

        mock_embed.assert_called_once_with(
            text="What is NVIDIA revenue?",
            task_type="retrieval_query",
        )


def test_retrieve_relevant_elements_end_to_end(tmp_path: Path):
    """Test end-to-end retrieval with indexed elements and similarity scoring."""
    persist_dir = tmp_path / "e2e_vector_store"
    collection = get_or_create_collection("e2e_coll", persist_directory=persist_dir)

    elements = [
        DocumentElement(
            element_id="elem_rev",
            doc_name="NVDA_FY24.pdf",
            page_number=1,
            element_type=ElementType.TEXT,
            content="Total revenue for fiscal year 2024 was $60.9 billion, up 122%.",
        ),
        DocumentElement(
            element_id="elem_chart",
            doc_name="NVDA_FY24.pdf",
            page_number=3,
            element_type=ElementType.CHART,
            content="[BAR: Segment Breakdown] Data center compute revenue surged to $47.5 billion.",
            metadata={"gemini_analysis": {"chart_type": "bar", "is_financial": True, "title": "Segment Breakdown"}},
        ),
        DocumentElement(
            element_id="elem_unrelated",
            doc_name="NVDA_FY24.pdf",
            page_number=10,
            element_type=ElementType.TEXT,
            content="Company headquarters is located in Santa Clara, California.",
        ),
    ]

    mock_embeddings = [
        [0.9, 0.1, 0.0],
        [0.85, 0.15, 0.0],
        [0.0, 0.1, 0.9],
    ]

    add_elements_to_collection(
        collection=collection,
        elements=elements,
        embeddings=mock_embeddings,
    )

    query_vec = [0.95, 0.05, 0.0]

    with patch("app.retrieval.retriever.get_embedding", return_value=query_vec):
        results = retrieve_relevant_elements(
            query="Tell me about annual revenue",
            collection=collection,
            n_results=2,
        )

        assert len(results) == 2
        assert results[0]["element_id"] == "elem_rev"
        assert "Total revenue for fiscal year 2024" in results[0]["content"]
        assert results[0]["similarity"] is not None
        assert results[0]["similarity"] > 0.8


def test_retrieve_relevant_elements_with_financial_filter(tmp_path: Path):
    """Test retrieve_relevant_elements applying filter_financial_only=True.

    Verifies that:
    - Text elements (no is_financial field) are included
    - Table elements (no is_financial field) are included
    - Charts with is_financial=True are included
    - Visuals with is_financial=False are excluded
    """
    persist_dir = tmp_path / "filter_vector_store"
    collection = get_or_create_collection("filter_coll", persist_directory=persist_dir)

    elements = [
        DocumentElement(
            element_id="text_elem",
            doc_name="Doc.pdf",
            page_number=1,
            element_type=ElementType.TEXT,
            content="Revenue for FY24 reached $60.9 billion.",
            metadata={"doc_name": "Doc.pdf", "page_number": 1},
        ),
        DocumentElement(
            element_id="table_elem",
            doc_name="Doc.pdf",
            page_number=2,
            element_type=ElementType.TABLE,
            content="| Revenue | $60.9B |\n| Gross Margin | 72.7% |",
            metadata={"doc_name": "Doc.pdf", "page_number": 2},
        ),
        DocumentElement(
            element_id="chart_financial",
            doc_name="Doc.pdf",
            page_number=3,
            element_type=ElementType.CHART,
            content="Revenue growth by region chart.",
            metadata={"gemini_analysis": {"chart_type": "bar", "is_financial": True}},
        ),
        DocumentElement(
            element_id="chart_decorative",
            doc_name="Doc.pdf",
            page_number=5,
            element_type=ElementType.IMAGE,
            content="Decorative corporate logo.",
            metadata={"gemini_analysis": {"chart_type": "logo", "is_financial": False}},
        ),
    ]

    add_elements_to_collection(
        collection=collection,
        elements=elements,
        embeddings=[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
    )

    with patch("app.retrieval.retriever.get_embedding", return_value=[0.5, 0.5]):
        results = retrieve_relevant_elements(
            query="Financial statements and revenue",
            collection=collection,
            filter_financial_only=True,
            n_results=10,
        )

        matched_ids = [r["element_id"] for r in results]

        # Text, Table, and Financial chart must be included
        assert "text_elem" in matched_ids
        assert "table_elem" in matched_ids
        assert "chart_financial" in matched_ids

        # Decorative logo with is_financial=False must be excluded
        assert "chart_decorative" not in matched_ids
        assert len(results) == 3


def test_elements_from_retrieval():
    """Test converting raw retrieval dictionaries back to DocumentElement objects."""
    raw_results = [
        {
            "element_id": "item_123",
            "content": "Operating margin improved by 500 bps.",
            "metadata": {
                "doc_name": "Report.pdf",
                "page_number": 4,
                "element_type": "text",
                "bbox": "[10.0, 20.0, 300.0, 150.0]",
                "ticker": "AAPL",
            },
            "distance": 0.1,
            "similarity": 0.9,
        }
    ]

    elements = elements_from_retrieval(raw_results)
    assert len(elements) == 1
    el = elements[0]
    assert isinstance(el, DocumentElement)
    assert el.element_id == "item_123"
    assert el.doc_name == "Report.pdf"
    assert el.page_number == 4
    assert el.element_type == ElementType.TEXT
    assert el.bbox == (10.0, 20.0, 300.0, 150.0)
    assert el.metadata.get("ticker") == "AAPL"


def test_format_retrieved_context():
    """Test formatting retrieved elements into a structured prompt context block."""
    raw_results = [
        {
            "element_id": "el_1",
            "content": "Revenue was $100M.",
            "metadata": {"doc_name": "Doc1.pdf", "page_number": 2, "element_type": "text"},
            "similarity": 0.92,
        },
        {
            "element_id": "el_2",
            "content": "[BAR: Growth] Grew 20%.",
            "metadata": {"doc_name": "Doc1.pdf", "page_number": 3, "element_type": "chart"},
            "similarity": 0.88,
        },
    ]

    formatted = format_retrieved_context(raw_results)
    assert "[Element 1 | ID: el_1 | TEXT | Source: Doc1.pdf, Page: 2 | Similarity: 0.920]" in formatted
    assert "Revenue was $100M." in formatted
    assert "[Element 2 | ID: el_2 | CHART | Source: Doc1.pdf, Page: 3 | Similarity: 0.880]" in formatted
    assert "[BAR: Growth] Grew 20%." in formatted


# =====================================================================
# Retrieval Pipeline Fallback Tests
# =====================================================================


def test_retrieve_with_fallback_financial_filtered():
    """Test retrieve_with_fallback uses financial_filtered path when >= 3 results returned."""
    mock_results = [{"element_id": f"id_{i}", "content": f"text {i}"} for i in range(4)]

    with patch("app.retrieval.pipeline.retrieve_relevant_elements", return_value=mock_results) as mock_retrieve:
        results, path_taken = retrieve_with_fallback("Revenue growth in 2023", collection="mock_coll")

        assert path_taken == "financial_filtered"
        assert len(results) == 4
        # Should have called retrieve_relevant_elements once with filter_financial_only=True
        mock_retrieve.assert_called_once_with(
            query="Revenue growth in 2023",
            collection="mock_coll",
            n_results=8,
            filter_financial_only=True,
        )


def test_retrieve_with_fallback_triggers_broad_fallback():
    """Test retrieve_with_fallback triggers broad fallback when financial filter yields < 3 items."""
    few_financial = [{"element_id": "fin_1", "content": "Only 1 chart"}]
    broad_results = [
        {"element_id": "fin_1", "content": "Only 1 chart"},
        {"element_id": "text_1", "content": "Text explanation"},
        {"element_id": "text_2", "content": "Table breakdown"},
    ]

    with patch(
        "app.retrieval.pipeline.retrieve_relevant_elements",
        side_effect=[few_financial, broad_results],
    ) as mock_retrieve:
        results, path_taken = retrieve_with_fallback("Capital allocation breakdown", collection="mock_coll")

        assert path_taken == "broad_fallback"
        assert len(results) == 3
        # Should have been called twice: first financial_only=True, then financial_only=False
        assert mock_retrieve.call_count == 2
        assert mock_retrieve.call_args_list[0].kwargs["filter_financial_only"] is True
        assert mock_retrieve.call_args_list[1].kwargs["filter_financial_only"] is False


def test_retrieve_with_fallback_broad_default():
    """Test retrieve_with_fallback uses broad_default for non-financial queries."""
    mock_results = [{"element_id": "gov_1", "content": "Board governance policies."}]

    with patch("app.retrieval.pipeline.retrieve_relevant_elements", return_value=mock_results) as mock_retrieve:
        results, path_taken = retrieve_with_fallback(
            "Who are the independent directors?",
            collection="mock_coll",
        )

        assert path_taken == "broad_default"
        assert len(results) == 1
        mock_retrieve.assert_called_once_with(
            query="Who are the independent directors?",
            collection="mock_coll",
            n_results=8,
            filter_financial_only=False,
        )


def test_retrieve_for_query():
    """Test retrieve_for_query resolves collection and returns results list."""
    mock_coll = MagicMock()
    mock_res = [{"element_id": "el_1", "content": "Content"}]

    with patch("app.retrieval.pipeline.get_or_create_collection", return_value=mock_coll):
        with patch("app.retrieval.pipeline.retrieve_with_fallback", return_value=(mock_res, "financial_filtered")):
            results = retrieve_for_query(
                query="Operating income margin",
                collection_name="custom_rag_coll",
                n_results=5,
            )

            assert results == mock_res
