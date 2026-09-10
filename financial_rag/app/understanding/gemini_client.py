"""Gemini API client wrapper for multimodal chart and document understanding."""

import logging
import warnings
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)

import google.generativeai as genai

from app.core.config import settings

logger = logging.getLogger(__name__)


def get_gemini_model(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.1,
) -> genai.GenerativeModel:
    """Initialize and configure a Google Gemini multimodal model instance.

    Args:
        model_name: Name of the Gemini model (defaults to settings.GEMINI_MODEL_NAME).
        api_key: Google Gemini API key (defaults to settings.GEMINI_API_KEY).
        temperature: Sampling temperature for deterministic structural output.

    Returns:
        genai.GenerativeModel: Configured Gemini GenerativeModel instance.

    Raises:
        ValueError: If the Gemini API key is missing or empty.
        RuntimeError: If configuration of the Gemini client fails.
    """
    effective_api_key = settings.GEMINI_API_KEY if api_key is None else api_key
    if not effective_api_key or effective_api_key.strip() in ("", "your_gemini_api_key_here"):

        raise ValueError(
            "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in your .env file "
            "or export it as an environment variable."
        )

    effective_model_name = model_name or settings.GEMINI_MODEL_NAME

    try:
        genai.configure(api_key=effective_api_key)
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            top_p=0.95,
        )
        model = genai.GenerativeModel(
            model_name=effective_model_name,
            generation_config=generation_config,
        )
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Gemini model '{effective_model_name}': {e}")
        raise RuntimeError(f"Could not initialize Gemini model '{effective_model_name}': {e}") from e
