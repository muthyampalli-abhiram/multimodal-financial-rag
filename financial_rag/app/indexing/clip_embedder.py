"""CLIP-based multimodal image and text embedding module using sentence-transformers."""

import logging
from pathlib import Path
from typing import List, Optional, Union

from PIL import Image
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Module-level singleton variable for caching the loaded CLIP model instance
_CLIP_MODEL_INSTANCE: Optional[SentenceTransformer] = None
CLIP_MODEL_NAME = "clip-ViT-B-32"


def get_clip_model(model_name: str = CLIP_MODEL_NAME) -> SentenceTransformer:
    """Load and cache the sentence-transformers CLIP model instance.

    Uses a module-level singleton pattern to ensure the model is loaded once
    and reused across subsequent embedding calls.

    Args:
        model_name: HuggingFace/SentenceTransformers model identifier.
                    Defaults to 'clip-ViT-B-32'.

    Returns:
        SentenceTransformer: Cached CLIP model instance.
    """
    global _CLIP_MODEL_INSTANCE
    if _CLIP_MODEL_INSTANCE is None:
        logger.info(f"Loading cached CLIP embedding model: '{model_name}'...")
        _CLIP_MODEL_INSTANCE = SentenceTransformer(model_name)
        logger.info(f"CLIP model '{model_name}' loaded successfully.")
    return _CLIP_MODEL_INSTANCE


def embed_image(image_path: Union[str, Path]) -> List[float]:
    """Generate a CLIP visual embedding vector for an image file.

    Loads the image via PIL, converts it to RGB mode, and processes it through
    the cached CLIP model visual encoder.

    Args:
        image_path: Path to the target image file.

    Returns:
        List[float]: Vector embedding as a list of floating-point numbers.
                     Returns an empty list if image loading or embedding fails.
    """
    if not image_path:
        logger.warning("Empty image_path provided to embed_image.")
        return []

    path = Path(image_path)
    if not path.exists() or not path.is_file():
        logger.error(f"Image file does not exist or is invalid: '{image_path}'")
        return []

    try:
        model = get_clip_model()
        with Image.open(path) as img:
            img_rgb = img.convert("RGB")
            embedding = model.encode(img_rgb)

        if hasattr(embedding, "tolist"):
            embedding_list = embedding.tolist()
        else:
            embedding_list = list(embedding)

        return [float(val) for val in embedding_list]
    except Exception as e:
        logger.error(f"Failed to generate CLIP image embedding for '{image_path}': {e}", exc_info=True)
        return []


def embed_text_clip(text: str) -> List[float]:
    """Generate a CLIP text embedding vector for a plain text query.

    Encodes text into the shared CLIP vector space using the CLIP text encoder.
    Enables cross-modal text-to-image similarity search against visual embeddings.

    Args:
        text: Plain text string to embed.

    Returns:
        List[float]: Vector embedding as a list of floating-point numbers.
                     Returns an empty list if text is empty or embedding fails.
    """
    if not text or not text.strip():
        logger.warning("Empty text string provided to embed_text_clip.")
        return []

    try:
        model = get_clip_model()
        embedding = model.encode(text.strip())

        if hasattr(embedding, "tolist"):
            embedding_list = embedding.tolist()
        else:
            embedding_list = list(embedding)

        return [float(val) for val in embedding_list]
    except Exception as e:
        logger.error(f"Failed to generate CLIP text embedding for text snippet: {e}", exc_info=True)
        return []
