"""Image and chart extraction from financial PDFs."""

import io
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
import pymupdf as fitz
from PIL import Image

from app.core.schemas import DocumentElement, ElementType

logger = logging.getLogger(__name__)

# Default pixel dimension thresholds
DEFAULT_MIN_IMAGE_WIDTH = 60
DEFAULT_MIN_IMAGE_HEIGHT = 60
DEFAULT_MIN_IMAGE_AREA = 3600

# Thresholds to flag candidate charts (e.g., substantial diagrams, figures, or plots)
DEFAULT_MIN_CHART_WIDTH = 250
DEFAULT_MIN_CHART_HEIGHT = 180
DEFAULT_MIN_CHART_AREA = 50000


def is_candidate_chart(
    width: int,
    height: int,
    min_chart_width: int = DEFAULT_MIN_CHART_WIDTH,
    min_chart_height: int = DEFAULT_MIN_CHART_HEIGHT,
    min_chart_area: int = DEFAULT_MIN_CHART_AREA,
) -> bool:
    """Determine whether an extracted image meets dimensions of a candidate financial chart/plot."""
    area = width * height
    return (width >= min_chart_width and height >= min_chart_height) or (area >= min_chart_area)


def extract_image_elements(
    doc: fitz.Document,
    doc_name: str,
    output_dir: Union[str, Path] = "data/processed/images",
    min_width: int = DEFAULT_MIN_IMAGE_WIDTH,
    min_height: int = DEFAULT_MIN_IMAGE_HEIGHT,
    min_chart_width: int = DEFAULT_MIN_CHART_WIDTH,
    min_chart_height: int = DEFAULT_MIN_CHART_HEIGHT,
    min_chart_area: int = DEFAULT_MIN_CHART_AREA,
) -> List[DocumentElement]:
    """Extract embedded raster images and candidate charts from a PDF document.

    Saves extracted images to disk under {output_dir}/{doc_name}/ and produces
    DocumentElements tagged with 'chart' or 'image' based on size thresholds.

    Args:
        doc: Opened fitz.Document.
        doc_name: Name of the source document (e.g., 'AAPL_2023_10K.pdf').
        output_dir: Base directory to save extracted image files.
        min_width: Minimum pixel width to ignore tiny icons/logos.
        min_height: Minimum pixel height to ignore tiny icons/logos.
        min_chart_width: Width threshold to classify an image as a candidate chart.
        min_chart_height: Height threshold to classify an image as a candidate chart.
        min_chart_area: Pixel area threshold to classify an image as a candidate chart.

    Returns:
        List[DocumentElement]: Extracted image and candidate chart elements.
    """
    doc_slug = Path(doc_name).stem
    save_dir = Path(output_dir) / doc_slug
    save_dir.mkdir(parents=True, exist_ok=True)

    image_elements: List[DocumentElement] = []
    seen_xrefs = set()

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_number = page_idx + 1

        # page.get_images() returns list of (xref, smask, width, height, bpc, colorspace, ...)
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                base_image = doc.extract_image(xref)
            except Exception as e:
                logger.warning(
                    f"Failed to extract image xref {xref} on page {page_number}: {e}"
                )
                continue

            if not base_image:
                continue

            width = base_image.get("width", 0)
            height = base_image.get("height", 0)
            image_bytes = base_image.get("image")
            image_ext = base_image.get("ext", "png")

            # Filter out tiny logos, bullets, decorative lines
            if width < min_width or height < min_height or (width * height) < DEFAULT_MIN_IMAGE_AREA:
                logger.debug(
                    f"Skipping small image {xref} ({width}x{height}) on page {page_number}"
                )
                continue

            # Resolve bounding box on page
            bbox: Optional[Tuple[float, float, float, float]] = None
            try:
                rects = page.get_image_rects(xref)
                if rects:
                    r = rects[0]
                    bbox = (round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2))
            except Exception as e:
                logger.debug(f"Could not calculate rect for image {xref}: {e}")

            # Save image as PNG using PIL for consistency and format normalization
            img_filename = f"page_{page_number}_img_{img_idx + 1}_xref_{xref}.png"
            img_file_path = save_dir / img_filename

            try:
                pil_img = Image.open(io.BytesIO(image_bytes))
                # Convert CMYK/RGBA if needed to RGB/PNG
                if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
                    pil_img.save(img_file_path, format="PNG")
                elif pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                    pil_img.save(img_file_path, format="PNG")
                else:
                    pil_img.save(img_file_path, format="PNG")
            except Exception as e:
                # Fallback: write raw image bytes
                logger.warning(f"PIL save error for xref {xref}, saving raw bytes: {e}")
                img_file_path = save_dir / f"page_{page_number}_img_{img_idx + 1}_xref_{xref}.{image_ext}"
                with open(img_file_path, "wb") as f:
                    f.write(image_bytes)

            # Classify as candidate chart vs. standard image
            if is_candidate_chart(
                width=width,
                height=height,
                min_chart_width=min_chart_width,
                min_chart_height=min_chart_height,
                min_chart_area=min_chart_area,
            ):
                element_type = ElementType.CHART
                content_desc = (
                    f"[Candidate Chart on Page {page_number}: {img_filename} "
                    f"({width}x{height}px) - Pending Gemini Multimodal Understanding]"
                )
            else:
                element_type = ElementType.IMAGE
                content_desc = (
                    f"[Embedded Image on Page {page_number}: {img_filename} "
                    f"({width}x{height}px)]"
                )

            element = DocumentElement(
                doc_name=doc_name,
                page_number=page_number,
                element_type=element_type,
                content=content_desc,
                bbox=bbox,
                metadata={
                    "image_path": str(img_file_path.resolve()),
                    "relative_path": str(img_file_path).replace("\\", "/"),
                    "image_filename": img_filename,
                    "width": width,
                    "height": height,
                    "pixel_area": width * height,
                    "xref": xref,
                    "original_ext": image_ext,
                },
            )
            image_elements.append(element)

    logger.info(
        f"Extracted {len(image_elements)} images/charts from '{doc_name}' saved to '{save_dir}'."
    )
    return image_elements
