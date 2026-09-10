"""PDF parsing, text/table/image extraction module."""

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.text_extractor import extract_text_elements
from app.ingestion.image_extractor import extract_image_elements
from app.ingestion.table_extractor import extract_table_elements


def __getattr__(name: str):
    if name == "ingest_pdf":
        from app.ingestion.pipeline import ingest_pdf
        return ingest_pdf
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "load_pdf",
    "extract_text_elements",
    "extract_image_elements",
    "extract_table_elements",
    "ingest_pdf",
]

