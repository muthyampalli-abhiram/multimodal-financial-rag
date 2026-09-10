"""Unit tests for Gemini multimodal chart and image understanding."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image

from app.core.schemas import DocumentElement, ElementType
from app.understanding.chart_analyzer import analyze_chart, clean_json_response
from app.understanding.gemini_client import get_gemini_model
from app.understanding.pipeline import enrich_elements_with_understanding


def test_gemini_client_missing_key():
    """Test get_gemini_model raises ValueError when API key is missing."""
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
        get_gemini_model(api_key="")


def test_gemini_client_configured():
    """Test get_gemini_model succeeds with a mock API key."""
    with patch("google.generativeai.configure") as mock_configure:
        with patch("google.generativeai.GenerativeModel") as mock_model:
            model = get_gemini_model(api_key="mock_test_key_123")
            mock_configure.assert_called_once_with(api_key="mock_test_key_123")
            assert model is not None


def test_clean_json_response_raw_json():
    """Test clean_json_response parses valid raw JSON."""
    raw = '{"chart_type": "bar", "title": "Revenue Growth", "summary": "Revenues grew 15%", "data_points": [{"label": "FY23", "value": "$100M"}], "time_period": "2023", "is_financial": true}'
    result = clean_json_response(raw)
    assert result["chart_type"] == "bar"
    assert result["title"] == "Revenue Growth"
    assert len(result["data_points"]) == 1


def test_clean_json_response_markdown_fences():
    """Test clean_json_response parses JSON wrapped in markdown code fences."""
    fenced = """```json
{
  "chart_type": "line",
  "title": "Margin Trend",
  "summary": "Operating margin improved.",
  "data_points": [],
  "time_period": "2021-2023",
  "is_financial": true
}
```"""
    result = clean_json_response(fenced)
    assert result["chart_type"] == "line"
    assert result["title"] == "Margin Trend"
    assert result["is_financial"] is True


def test_clean_json_response_fallback_on_error():
    """Test clean_json_response produces fallback dict on invalid JSON."""
    invalid = "This is not JSON text from model"
    result = clean_json_response(invalid)
    assert "error" in result
    assert result["chart_type"] == "other"
    assert result["summary"] == "This is not JSON text from model"


def test_analyze_chart_missing_file():
    """Test analyze_chart returns error dictionary when image file does not exist."""
    res = analyze_chart("non_existent_chart_image.png")
    assert "error" in res
    assert "not found" in res["error"]


def test_analyze_chart_with_mock_gemini(tmp_path: Path):
    """Test analyze_chart invoking mocked Gemini model."""
    img_path = tmp_path / "test_chart.png"
    img = Image.new("RGB", (200, 100), color=(20, 40, 60))
    img.save(img_path)

    mock_response = MagicMock()
    mock_response.text = '{"chart_type": "bar", "title": "Operating Income", "summary": "Income expanded 20%.", "data_points": [{"label": "Q4", "value": "$50M"}], "time_period": "Q4 2023", "is_financial": true}'

    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response

    with patch("app.understanding.chart_analyzer.get_gemini_model", return_value=mock_model):
        res = analyze_chart(img_path)
        assert res["chart_type"] == "bar"
        assert res["title"] == "Operating Income"
        assert len(res["data_points"]) == 1
        assert res["is_financial"] is True


def test_enrich_elements_with_understanding(tmp_path: Path):
    """Test enrich_elements_with_understanding updating content and metadata."""
    img_path = tmp_path / "financial_chart.png"
    img = Image.new("RGB", (200, 100), color=(100, 50, 50))
    img.save(img_path)

    text_el = DocumentElement(
        doc_name="report",
        page_number=1,
        element_type=ElementType.TEXT,
        content="Narrative discussing FY23 results.",
    )
    chart_el = DocumentElement(
        doc_name="report",
        page_number=1,
        element_type=ElementType.CHART,
        content="[Candidate Chart placeholder]",
        metadata={"image_path": str(img_path)},
    )

    fake_analysis = {
        "chart_type": "bar",
        "title": "Segment Breakdown",
        "summary": "Segment revenue grew strongly across all regions.",
        "data_points": [{"label": "Americas", "value": "$500M"}],
        "time_period": "FY2023",
        "is_financial": True,
    }

    with patch("app.understanding.pipeline.analyze_chart", return_value=fake_analysis):
        enriched = enrich_elements_with_understanding([text_el, chart_el])

        # Text element should remain unchanged
        assert enriched[0].content == "Narrative discussing FY23 results."

        # Chart element content and metadata should be updated
        assert "gemini_analysis" in enriched[1].metadata
        assert enriched[1].metadata["gemini_analysis"]["chart_type"] == "bar"
        assert "Segment revenue grew strongly" in enriched[1].content


def test_save_and_load_elements_to_json(tmp_path: Path):
    """Test serializing and deserializing DocumentElement objects."""
    from app.understanding.pipeline import load_elements_from_json, save_elements_to_json

    out_file = tmp_path / "enriched_test.json"
    elements = [
        DocumentElement(
            doc_name="acme_2023",
            page_number=1,
            element_type=ElementType.TEXT,
            content="Sample narrative.",
            bbox=(50.0, 50.0, 500.0, 100.0),
        ),
        DocumentElement(
            doc_name="acme_2023",
            page_number=2,
            element_type=ElementType.CHART,
            content="[BAR: Revenue] Revenue rose 15%.",
            metadata={"gemini_analysis": {"is_financial": True, "chart_type": "bar"}},
        ),
    ]

    save_elements_to_json(elements, out_file)
    assert out_file.is_file()

    loaded = load_elements_from_json(out_file)
    assert len(loaded) == 2
    assert loaded[0].element_id == elements[0].element_id
    assert loaded[0].content == "Sample narrative."
    assert loaded[1].metadata["gemini_analysis"]["is_financial"] is True


def test_enrich_elements_with_cache_reuse(tmp_path: Path):
    """Test that existing valid analyses are reused from cache without calling Gemini."""
    img_path = tmp_path / "chart1.png"
    Image.new("RGB", (100, 100)).save(img_path)

    cached_chart = DocumentElement(
        element_id="report_chart_1",
        doc_name="report",
        page_number=1,
        element_type=ElementType.CHART,
        content="[BAR: Cached Chart] Cached summary.",
        metadata={
            "image_path": str(img_path),
            "gemini_analysis": {
                "chart_type": "bar",
                "title": "Cached Chart",
                "summary": "Cached summary.",
                "data_points": [],
                "is_financial": True,
            },
        },
    )

    new_chart = DocumentElement(
        element_id="report_chart_1",
        doc_name="report",
        page_number=1,
        element_type=ElementType.CHART,
        content="[Pending understanding]",
        metadata={"image_path": str(img_path)},
    )

    with patch("app.understanding.pipeline.analyze_chart") as mock_analyze:
        enriched = enrich_elements_with_understanding(
            elements=[new_chart],
            existing_elements=[cached_chart],
            delay_seconds=0,
        )

        # analyze_chart should NOT have been called because it was cached
        mock_analyze.assert_not_called()
        assert enriched[0].content == "[BAR: Cached Chart] Cached summary."
        assert enriched[0].metadata["gemini_analysis"]["title"] == "Cached Chart"


def test_enrich_elements_with_max_calls(tmp_path: Path):
    """Test that max_calls limits the number of new Gemini API invocations."""
    img_path1 = tmp_path / "chart1.png"
    img_path2 = tmp_path / "chart2.png"
    Image.new("RGB", (100, 100)).save(img_path1)
    Image.new("RGB", (100, 100)).save(img_path2)

    chart1 = DocumentElement(
        element_id="report_chart_1",
        doc_name="report",
        page_number=1,
        element_type=ElementType.CHART,
        content="[Pending 1]",
        metadata={"image_path": str(img_path1)},
    )
    chart2 = DocumentElement(
        element_id="report_chart_2",
        doc_name="report",
        page_number=2,
        element_type=ElementType.CHART,
        content="[Pending 2]",
        metadata={"image_path": str(img_path2)},
    )

    fake_analysis = {
        "chart_type": "line",
        "title": "Analyzed Chart 1",
        "summary": "First chart analyzed successfully.",
        "data_points": [],
        "is_financial": True,
    }

    with patch("app.understanding.pipeline.analyze_chart", return_value=fake_analysis) as mock_analyze:
        enriched = enrich_elements_with_understanding(
            elements=[chart1, chart2],
            max_calls=1,
            delay_seconds=0,
        )

        # Should only call analyze_chart once
        assert mock_analyze.call_count == 1
        assert "gemini_analysis" in enriched[0].metadata
        assert "gemini_analysis" not in enriched[1].metadata
        assert enriched[1].content == "[Pending 2]"


