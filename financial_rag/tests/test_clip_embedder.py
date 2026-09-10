"""Unit tests for multimodal CLIP image/text embeddings and visual index module.

Note: This visual collection is currently a standalone proof-of-concept — merging its
results into the main generation pipeline (app/generation/) would be a future enhancement,
not part of this step.
"""

from pathlib import Path
import pytest
from PIL import Image

from app.core.schemas import DocumentElement, ElementType
from app.indexing.clip_embedder import (
    embed_image,
    embed_text_clip,
    get_clip_model,
)
from app.indexing.visual_index import (
    get_visual_collection,
    index_visual_elements,
    visual_search,
)


@pytest.fixture
def sample_image_path(tmp_path: Path) -> Path:
    """Create a temporary real image file (RGB PNG) for visual embedding tests."""
    img_path = tmp_path / "test_chart.png"
    img = Image.new("RGB", (100, 100), color="blue")
    img.save(img_path)
    return img_path


def test_get_clip_model_singleton():
    """Test get_clip_model returns a valid model instance and reuses it (singleton pattern)."""
    model1 = get_clip_model()
    assert model1 is not None
    model2 = get_clip_model()
    assert model1 is model2


def test_embed_image_returns_vector(sample_image_path: Path):
    """Test embed_image returns a non-empty list of floats for a real image file."""
    vector = embed_image(sample_image_path)
    assert isinstance(vector, list)
    assert len(vector) > 0
    assert all(isinstance(x, float) for x in vector)


def test_embed_text_clip_returns_matching_dimensionality(sample_image_path: Path):
    """Test embed_text_clip returns a vector with the exact same dimensionality as embed_image.

    Proves that text and image embeddings inhabit the identical joint vector space (e.g. 512 dimensions).
    """
    image_vector = embed_image(sample_image_path)
    text_vector = embed_text_clip("bar chart showing revenue growth")

    assert isinstance(text_vector, list)
    assert len(text_vector) > 0
    assert len(text_vector) == len(image_vector)
    assert len(text_vector) == 512  # CLIP ViT-B-32 produces 512-dimensional vector representations


def test_embed_image_missing_file_error_handling(tmp_path: Path):
    """Test error handling when a non-existent or corrupt image path is passed."""
    non_existent_path = tmp_path / "does_not_exist.png"
    vector = embed_image(non_existent_path)
    assert vector == []


def test_embed_text_clip_empty_string():
    """Test embed_text_clip returns empty list when given empty string or whitespace."""
    assert embed_text_clip("") == []
    assert embed_text_clip("   ") == []


def test_visual_index_and_search_flow(tmp_path: Path, sample_image_path: Path):
    """Test creating a visual collection, indexing elements, and running cross-modal visual_search."""
    persist_dir = tmp_path / "visual_store"
    collection = get_visual_collection(persist_directory=persist_dir)
    assert collection.name == "financial_rag_visual"

    elements = [
        DocumentElement(
            element_id="visual_elem_1",
            doc_name="NVDA_2023.pdf",
            page_number=4,
            element_type=ElementType.CHART,
            content="[BAR CHART] Data Center Revenue Breakdown FY24",
            metadata={
                "image_path": str(sample_image_path.resolve()),
                "chart_type": "bar",
                "is_financial": True,
            },
        ),
        DocumentElement(
            element_id="text_elem_2",
            doc_name="NVDA_2023.pdf",
            page_number=5,
            element_type=ElementType.TEXT,
            content="Plain narrative text block explaining gross margins.",
            metadata={"image_path": str(sample_image_path.resolve())},
        ),
    ]

    summary = index_visual_elements(
        elements=elements,
        collection=collection,
    )

    # Should index 1 chart element, skip non-chart/image element
    assert summary["indexed"] == 1
    assert summary["errors"] == 0
    assert collection.count() == 1

    results = visual_search(
        query_text="Data Center revenue chart",
        collection=collection,
        n_results=5,
    )

    assert len(results) == 1
    res = results[0]
    assert res["element_id"] == "visual_elem_1"
    assert "Data Center Revenue Breakdown" in res["content"]
    assert "similarity" in res
    assert res["similarity"] is not None
    assert 0.0 <= res["similarity"] <= 1.0
