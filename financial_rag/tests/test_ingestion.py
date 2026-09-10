"""Unit tests for PDF ingestion and raw element extraction."""

import os
import shutil
import tempfile
from pathlib import Path
import pymupdf as fitz
from PIL import Image, ImageDraw
import pytest

from app.core.schemas import ElementType
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.text_extractor import extract_text_elements
from app.ingestion.image_extractor import extract_image_elements
from app.ingestion.table_extractor import extract_table_elements


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> str:
    """Generate a realistic test PDF with text narrative, a table, and an image."""
    pdf_path = tmp_path / "sample_financial_report.pdf"
    doc = fitz.open()

    # Page 1: Text & Image
    page1 = doc.new_page(width=595, height=842)
    page1.insert_text(
        (50, 72),
        "Alphabet Inc. Q4 Financial Highlights and Operating Results Analysis.",
        fontsize=14,
    )
    page1.insert_text(
        (50, 120),
        "Total consolidated revenues reached $86.3 billion, representing a 13% year-over-year increase.",
        fontsize=10,
    )
    page1.insert_text(
        (50, 160),
        "x",  # Very short string to test min_chars filtering
        fontsize=10,
    )

    # Create a dummy chart image and insert it
    img = Image.new("RGB", (300, 200), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    d.rectangle([(20, 20), (280, 180)], outline=(255, 255, 255), width=2)
    img_temp_path = tmp_path / "temp_chart.png"
    img.save(img_temp_path)

    img_rect = fitz.Rect(50, 200, 350, 400)
    page1.insert_image(img_rect, filename=str(img_temp_path))

    # Page 2: Table
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text(
        (50, 72),
        "Consolidated Statements of Income (in millions):",
        fontsize=12,
    )

    # Draw table borders so find_tables() can detect it
    table_rect = fitz.Rect(50, 100, 500, 220)
    # Draw outer rectangle
    page2.draw_rect(table_rect, color=(0, 0, 0), width=1)
    # Draw horizontal row dividers
    page2.draw_line((50, 140), (500, 140), color=(0, 0, 0), width=1)
    page2.draw_line((50, 180), (500, 180), color=(0, 0, 0), width=1)
    # Draw vertical column dividers
    page2.draw_line((200, 100), (200, 220), color=(0, 0, 0), width=1)
    page2.draw_line((350, 100), (350, 220), color=(0, 0, 0), width=1)

    # Insert cell texts
    page2.insert_text((55, 125), "Segment", fontsize=10)
    page2.insert_text((205, 125), "FY 2023", fontsize=10)
    page2.insert_text((355, 125), "FY 2022", fontsize=10)

    page2.insert_text((55, 165), "Google Services", fontsize=10)
    page2.insert_text((205, 165), "$77,200", fontsize=10)
    page2.insert_text((355, 165), "$69,000", fontsize=10)

    page2.insert_text((55, 205), "Google Cloud", fontsize=10)
    page2.insert_text((205, 205), "$9,100", fontsize=10)
    page2.insert_text((355, 205), "$7,300", fontsize=10)

    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


def test_load_pdf_success(sample_pdf_path: str):
    """Test successful loading of valid PDF."""
    doc = load_pdf(sample_pdf_path)
    assert doc is not None
    assert len(doc) == 2
    doc.close()


def test_load_pdf_missing_file():
    """Test load_pdf raises FileNotFoundError for non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_pdf("non_existent_file.pdf")


def test_load_pdf_invalid_file(tmp_path: Path):
    """Test load_pdf raises ValueError for corrupt/invalid file."""
    corrupt_file = tmp_path / "corrupt.pdf"
    corrupt_file.write_text("Not a valid PDF content")
    with pytest.raises(ValueError):
        load_pdf(str(corrupt_file))


def test_extract_text_elements(sample_pdf_path: str):
    """Test text block extraction and filtering."""
    doc = load_pdf(sample_pdf_path)
    elements = extract_text_elements(doc, "sample_financial_report.pdf", min_chars=15)
    doc.close()

    assert len(elements) > 0
    for el in elements:
        assert el.element_type == ElementType.TEXT
        assert el.doc_name == "sample_financial_report.pdf"
        assert el.page_number in (1, 2)
        assert len(el.content) >= 15
        assert el.bbox is not None
        assert "block_number" in el.metadata

    # Check that the short single character "x" was excluded
    contents = [el.content for el in elements]
    assert not any(c == "x" for c in contents)


def test_extract_image_elements(sample_pdf_path: str, tmp_path: Path):
    """Test image extraction, saving to disk, and chart tagging."""
    output_dir = tmp_path / "processed_images"
    doc = load_pdf(sample_pdf_path)
    elements = extract_image_elements(
        doc,
        "sample_financial_report.pdf",
        output_dir=output_dir,
        min_chart_width=200,
        min_chart_height=150,
    )
    doc.close()

    assert len(elements) >= 1
    chart_or_img = elements[0]
    assert chart_or_img.element_type in (ElementType.CHART, ElementType.IMAGE)
    assert chart_or_img.page_number == 1
    assert "image_path" in chart_or_img.metadata
    assert Path(chart_or_img.metadata["image_path"]).is_file()


def test_extract_table_elements(sample_pdf_path: str):
    """Test table detection, markdown generation, and record serialization."""
    doc = load_pdf(sample_pdf_path)
    tables = extract_table_elements(doc, "sample_financial_report.pdf")
    doc.close()

    assert len(tables) >= 1
    tbl = tables[0]
    assert tbl.element_type == ElementType.TABLE
    assert tbl.page_number == 2
    assert "|" in tbl.content
    assert "table_data" in tbl.metadata
    assert isinstance(tbl.metadata["table_data"], list)
    assert tbl.metadata["row_count"] >= 1
    assert tbl.metadata["col_count"] >= 1


def test_ingest_pdf_pipeline(sample_pdf_path: str, tmp_path: Path):
    """Test full ingestion pipeline orchestrator."""
    from app.ingestion.pipeline import ingest_pdf

    output_dir = tmp_path / "pipeline_images"
    elements = ingest_pdf(sample_pdf_path, output_image_dir=output_dir)

    assert len(elements) >= 3
    # Check sequential IDs
    ids = [el.element_id for el in elements]
    assert all(el.doc_name == "sample_financial_report" for el in elements)
    assert any("text" in eid for eid in ids)

    # Check page sorting
    page_numbers = [el.page_number for el in elements]
    assert page_numbers == sorted(page_numbers)

