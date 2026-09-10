"""Unit and integration tests for grounded answer generation and citation synthesis."""

from unittest.mock import MagicMock, patch
import pytest

from app.core.schemas import Citation, DocumentElement, ElementType, QueryResult
from app.generation.answer_generator import (
    build_generation_prompt,
    extract_citations_from_text,
    generate_answer,
)
from app.generation.pipeline import answer_query


def test_build_generation_prompt():
    """Test build_generation_prompt includes query, retrieved context, and citation format."""
    query = "What was the operating margin in 2023?"
    context = "--- [Element 1 | ID: elem_1 | TEXT] ---\nOperating margin reached 45%."

    prompt = build_generation_prompt(query=query, retrieved_context=context)

    assert query in prompt
    assert context in prompt
    assert "[Source: <element_id>]" in prompt
    assert "Strict Grounding" in prompt


def test_extract_citations_from_text():
    """Test extract_citations_from_text correctly extracts and maps citations."""
    sample_answer = (
        "NVIDIA's Total Shareholder Return over 5 years significantly outperformed the market [Source: chart_tsr]. "
        "Additionally, $10.4 billion was returned to shareholders via dividends and repurchases [Source: chart_capital, text_summary]. "
        "Reiterating TSR performance [Source: chart_tsr]."
    )

    retrieved_results = [
        {
            "element_id": "chart_tsr",
            "content": "[BAR: TSR Performance] Total Shareholder return 5-year comparison.",
            "metadata": {
                "doc_name": "NVDA_FY24.pdf",
                "page_number": 2,
                "element_type": "chart",
                "bbox": "[10.0, 20.0, 400.0, 300.0]",
            },
        },
        {
            "element_id": "chart_capital",
            "content": "[BAR: Capital Returned] Total Capital Returned to Shareholders was $10.4B.",
            "metadata": {
                "doc_name": "NVDA_FY24.pdf",
                "page_number": 2,
                "element_type": "chart",
            },
        },
        {
            "element_id": "text_summary",
            "content": "Narrative summarizing shareholder returns.",
            "metadata": {
                "doc_name": "NVDA_FY24.pdf",
                "page_number": 1,
                "element_type": "text",
            },
        },
    ]

    citations = extract_citations_from_text(sample_answer, retrieved_results)

    # Should have extracted 3 unique citations in order of appearance
    assert len(citations) == 3
    assert citations[0].element_id == "chart_tsr"
    assert citations[0].doc_name == "NVDA_FY24.pdf"
    assert citations[0].page_number == 2
    assert citations[0].element_type == ElementType.CHART
    assert citations[0].bbox == (10.0, 20.0, 400.0, 300.0)

    assert citations[1].element_id == "chart_capital"
    assert citations[2].element_id == "text_summary"
    assert citations[2].element_type == ElementType.TEXT


def test_generate_answer_empty_results():
    """Test generate_answer returns empty QueryResult when no elements are retrieved."""
    res = generate_answer("What is the revenue?", retrieved_results=[])
    assert isinstance(res, QueryResult)
    assert "No relevant document elements" in res.answer
    assert res.citations == []
    assert res.retrieved_elements == []


def test_generate_answer_success():
    """Test generate_answer successfully calls Gemini model and returns grounded QueryResult."""
    retrieved_results = [
        {
            "element_id": "nvidia_chart_1",
            "content": "[BAR: Segment Revenue] Data Center revenue was $47.5B in FY24.",
            "metadata": {
                "doc_name": "NVDA_Report.pdf",
                "page_number": 3,
                "element_type": "chart",
            },
            "similarity": 0.85,
        }
    ]

    mock_response = MagicMock()
    mock_response.text = "In FY24, Data Center revenue reached $47.5 billion [Source: nvidia_chart_1]."

    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response

    with patch("app.generation.answer_generator.get_gemini_model", return_value=mock_model):
        query_result = generate_answer(
            query="What was Data Center revenue in FY24?",
            retrieved_results=retrieved_results,
        )

        assert isinstance(query_result, QueryResult)
        assert "Data Center revenue reached $47.5 billion" in query_result.answer
        assert len(query_result.citations) == 1
        assert query_result.citations[0].element_id == "nvidia_chart_1"
        assert query_result.citations[0].element_type == ElementType.CHART
        assert len(query_result.retrieved_elements) == 1
        assert isinstance(query_result.retrieved_elements[0], DocumentElement)


def test_generate_answer_retry_on_failure():
    """Test generate_answer retries upon transient API error and succeeds."""
    retrieved_results = [
        {
            "element_id": "elem_text_1",
            "content": "Gross margin was 72.7%.",
            "metadata": {"doc_name": "Doc.pdf", "page_number": 1, "element_type": "text"},
        }
    ]

    mock_response = MagicMock()
    mock_response.text = "Gross margin expanded to 72.7% [Source: elem_text_1]."

    mock_model = MagicMock()
    mock_model.generate_content.side_effect = [
        RuntimeError("Transient 503 Service Unavailable"),
        mock_response,
    ]

    with patch("app.generation.answer_generator.get_gemini_model", return_value=mock_model):
        res = generate_answer(
            query="What was the gross margin?",
            retrieved_results=retrieved_results,
            max_retries=2,
            retry_delay=0.01,
        )

        assert "Gross margin expanded to 72.7%" in res.answer
        assert len(res.citations) == 1
        assert mock_model.generate_content.call_count == 2


def test_generate_answer_all_retries_fail_returns_error_result():
    """Test generate_answer returns QueryResult with error message when all retries fail."""
    retrieved_results = [
        {
            "element_id": "elem_1",
            "content": "Content text.",
            "metadata": {"doc_name": "Doc.pdf", "page_number": 1, "element_type": "text"},
        }
    ]

    mock_model = MagicMock()
    mock_model.generate_content.side_effect = RuntimeError("Persistent API Quota Exhausted")

    with patch("app.generation.answer_generator.get_gemini_model", return_value=mock_model):
        res = generate_answer(
            query="Query",
            retrieved_results=retrieved_results,
            max_retries=1,
            retry_delay=0.01,
        )

        assert isinstance(res, QueryResult)
        assert "Error generating answer" in res.answer
        assert res.citations == []
        assert len(res.retrieved_elements) == 1


def test_answer_query_pipeline():
    """Test answer_query end-to-end orchestration."""
    mock_retrieved = [
        {
            "element_id": "pipe_chart_1",
            "content": "[BAR: Capital Return] Returned $10.4B.",
            "metadata": {"doc_name": "Report.pdf", "page_number": 2, "element_type": "chart"},
        }
    ]

    mock_query_result = QueryResult(
        answer="NVIDIA returned $10.4B [Source: pipe_chart_1].",
        citations=[
            Citation(
                element_id="pipe_chart_1",
                doc_name="Report.pdf",
                page_number=2,
                element_type=ElementType.CHART,
            )
        ],
        retrieved_elements=[
            DocumentElement(
                element_id="pipe_chart_1",
                doc_name="Report.pdf",
                page_number=2,
                element_type=ElementType.CHART,
                content="[BAR: Capital Return] Returned $10.4B.",
            )
        ],
    )

    with patch("app.generation.pipeline.retrieve_for_query", return_value=mock_retrieved) as mock_retrieve:
        with patch("app.generation.pipeline.generate_answer", return_value=mock_query_result) as mock_gen:
            res = answer_query(query="How much capital was returned?", collection_name="test_coll", n_results=5)

            mock_retrieve.assert_called_once_with(
                query="How much capital was returned?",
                collection_name="test_coll",
                n_results=5,
            )
            mock_gen.assert_called_once_with(
                query="How much capital was returned?",
                retrieved_results=mock_retrieved,
            )
            assert res == mock_query_result
