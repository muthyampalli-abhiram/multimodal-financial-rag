"""PDF loading and validation utility for financial documents."""

import logging
from pathlib import Path
from typing import Union
import pymupdf as fitz

logger = logging.getLogger(__name__)


def load_pdf(file_path: Union[str, Path]) -> fitz.Document:
    """Open and validate a PDF file using PyMuPDF (fitz).

    Args:
        file_path: Path to the PDF file (string or Path object).

    Returns:
        fitz.Document: Opened PyMuPDF document instance.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is not a valid PDF or is empty/corrupt.
        RuntimeError: If the document is password-protected or cannot be parsed.
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path.resolve()}")

    try:
        doc = fitz.open(str(path))
    except (fitz.FileDataError, fitz.EmptyFileError) as e:
        logger.error(f"Failed to open corrupt or invalid PDF: {path.name}. Error: {e}")
        raise ValueError(f"Corrupt or invalid PDF file '{path.name}': {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error opening PDF '{path.name}': {e}")
        raise RuntimeError(f"Unable to open PDF '{path.name}': {e}") from e

    if doc.is_encrypted:
        logger.warning(f"PDF '{path.name}' is encrypted.")
        # Attempt blank password decryption
        if not doc.authenticate(""):
            doc.close()
            raise RuntimeError(f"PDF '{path.name}' is encrypted and requires a password.")

    if len(doc) == 0:
        doc.close()
        raise ValueError(f"PDF '{path.name}' contains 0 pages.")

    logger.info(f"Successfully loaded '{path.name}' ({len(doc)} pages).")
    return doc
