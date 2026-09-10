"""Gemini text and multimodal element embedding generation module."""

import logging
import time
import warnings
from typing import List, Optional

warnings.filterwarnings("ignore", category=FutureWarning)

import google.generativeai as genai

from app.core.config import settings

logger = logging.getLogger(__name__)


def configure_genai(api_key: Optional[str] = None) -> None:
    """Configure the Google Generative AI client with an API key.

    Args:
        api_key: Optional Gemini API key (defaults to settings.GEMINI_API_KEY).

    Raises:
        ValueError: If the Gemini API key is missing or not configured.
    """
    effective_api_key = settings.GEMINI_API_KEY if api_key is None else api_key
    if not effective_api_key or effective_api_key.strip() in ("", "your_gemini_api_key_here"):
        raise ValueError(
            "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in your .env file "
            "or export it as an environment variable."
        )
    genai.configure(api_key=effective_api_key)


def get_embedding(
    text: str,
    task_type: str = "retrieval_document",
    title: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    max_retries: int = 2,
    retry_delay: float = 1.0,
) -> List[float]:
    """Generate a dense vector embedding for a single text using Gemini Embedding API.

    Args:
        text: Input string to embed.
        task_type: Embedding task type ('retrieval_document', 'retrieval_query',
                   'semantic_similarity', 'classification', 'clustering').
        title: Optional document title (used when task_type is 'retrieval_document').
        model_name: Gemini embedding model name (defaults to settings.EMBEDDING_MODEL_NAME).
        api_key: Optional Gemini API key.
        max_retries: Maximum number of retries for transient API/rate-limit errors.
        retry_delay: Base delay in seconds between retries (uses exponential backoff).

    Returns:
        List[float]: Vector embedding as a list of floats, or empty list on failure.
    """
    if not text or not text.strip():
        logger.warning("Empty or whitespace text passed to get_embedding; returning empty list.")
        return []

    configure_genai(api_key=api_key)

    effective_model = model_name or settings.EMBEDDING_MODEL_NAME
    # Ensure model has "models/" prefix if needed by genai
    if not effective_model.startswith("models/"):
        effective_model = f"models/{effective_model}"

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            kwargs = {
                "model": effective_model,
                "content": text.strip(),
                "task_type": task_type,
            }
            if title and task_type == "retrieval_document":
                kwargs["title"] = title

            result = genai.embed_content(**kwargs)

            if isinstance(result, dict) and "embedding" in result:
                embedding = result["embedding"]
                # If single vector returned
                if embedding and isinstance(embedding, list) and isinstance(embedding[0], (int, float)):
                    return [float(x) for x in embedding]
                elif embedding and isinstance(embedding, list) and isinstance(embedding[0], list):
                    return [float(x) for x in embedding[0]]

            raise ValueError(f"Unexpected response format from Gemini embed_content: {type(result)}")

        except Exception as e:
            last_error = e
            logger.warning(
                f"Embedding generation attempt {attempt + 1}/{max_retries + 1} failed "
                f"for model '{effective_model}': {e}"
            )
            if attempt < max_retries:
                sleep_time = retry_delay * (2**attempt)
                logger.info(f"Retrying embedding call in {sleep_time:.1f}s...")
                time.sleep(sleep_time)

    logger.error(
        f"All {max_retries + 1} embedding attempts failed for model '{effective_model}': {last_error}"
    )
    return []


def get_embeddings_batch(
    texts: List[str],
    task_type: str = "retrieval_document",
    batch_size: int = 32,
    delay_seconds: float = 0.0,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    max_retries: int = 2,
    retry_delay: float = 1.0,
) -> List[List[float]]:
    """Generate dense vector embeddings for a batch of texts using Gemini Embedding API.

    Processes inputs in chunks and gracefully falls back to individual calls if a batch fails,
    ensuring that rate-limit errors or one bad payload do not crash the entire batch.

    Args:
        texts: List of input strings to embed.
        task_type: Embedding task type ('retrieval_document', 'retrieval_query', etc.).
        batch_size: Number of texts per batch request (default: 32).
        delay_seconds: Delay in seconds between batch calls to prevent rate limit spikes.
        model_name: Gemini embedding model name (defaults to settings.EMBEDDING_MODEL_NAME).
        api_key: Optional Gemini API key.
        max_retries: Maximum number of retries for transient API/rate-limit errors.
        retry_delay: Base delay in seconds between retries.

    Returns:
        List[List[float]]: List of vector embeddings matching input texts 1-to-1.
    """
    if not texts:
        return []

    configure_genai(api_key=api_key)

    effective_model = model_name or settings.EMBEDDING_MODEL_NAME
    if not effective_model.startswith("models/"):
        effective_model = f"models/{effective_model}"

    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        # Clean / normalize strings in chunk, replacing empty strings with fallback space
        cleaned_chunk = [t.strip() if t and t.strip() else " " for t in chunk]

        chunk_embeddings: Optional[List[List[float]]] = None
        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                result = genai.embed_content(
                    model=effective_model,
                    content=cleaned_chunk,
                    task_type=task_type,
                )

                if isinstance(result, dict) and "embedding" in result:
                    raw_embs = result["embedding"]
                    if isinstance(raw_embs, list):
                        chunk_embeddings = [[float(val) for val in emb] for emb in raw_embs]
                        break

                raise ValueError(f"Unexpected batch response format: {type(result)}")

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Batch embedding attempt {attempt + 1}/{max_retries + 1} failed "
                    f"for {len(chunk)} items: {e}"
                )
                if attempt < max_retries:
                    sleep_time = retry_delay * (2**attempt)
                    time.sleep(sleep_time)

        # If batch call succeeded, append
        if chunk_embeddings is not None and len(chunk_embeddings) == len(chunk):
            all_embeddings.extend(chunk_embeddings)
        else:
            # Fallback: Process each text individually so single error doesn't drop the entire batch
            logger.warning(
                f"Batch embedding failed ({last_error}). Falling back to per-item embedding for {len(chunk)} items."
            )
            for single_text in chunk:
                single_emb = get_embedding(
                    text=single_text,
                    task_type=task_type,
                    model_name=effective_model,
                    api_key=api_key,
                    max_retries=1,
                    retry_delay=retry_delay,
                )
                all_embeddings.append(single_emb)

        if delay_seconds > 0 and (i + batch_size) < len(texts):
            time.sleep(delay_seconds)

    return all_embeddings
