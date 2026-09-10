"""Multimodal chart and image analysis using the Gemini Vision API."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
from PIL import Image

from app.understanding.gemini_client import get_gemini_model

logger = logging.getLogger(__name__)

CHART_ANALYSIS_PROMPT = """You are a senior financial analyst and vision intelligence expert.
Analyze the provided image or chart extracted from a financial document (e.g. 10-K, annual report, investor deck).

Extract structured information and return ONLY a valid, raw JSON object matching the schema below.
Do NOT include markdown backticks (like ```json), no intro text, and no explanation.

Schema:
{{
  "chart_type": "<one of: bar, line, pie, area, scatter, table_image, map, infographic, flowchart, logo, photo, other>",
  "title": "<inferred title or concise description of the chart subject>",
  "summary": "<2-4 sentences plain-English description explaining what the visual shows, key trends, and main analytical takeaway>",
  "data_points": [
    {{
      "label": "<category or series label, e.g. FY23 Revenue, Operating Margin, Cloud Segment>",
      "value": "<numeric value with unit, e.g. $120.5M, 24.5%, 1500 units>"
    }}
  ],
  "time_period": "<date or fiscal period range covered, e.g. 2021-2023, Q4 FY24, or null if none>",
  "is_financial": <true if the image contains financial/business/operational metrics, false if decorative/icon>
}}

Optional Page Context:
{context}
"""


def clean_json_response(raw_text: str) -> Dict[str, Any]:
    """Clean and defensively parse JSON string from model response."""
    text = raw_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            # Guarantee required keys
            parsed.setdefault("chart_type", "other")
            parsed.setdefault("title", "Financial Visual")
            parsed.setdefault("summary", "")
            parsed.setdefault("data_points", [])
            parsed.setdefault("time_period", None)
            parsed.setdefault("is_financial", True)
            return parsed
    except json.JSONDecodeError:
        pass

    # Substring search fallback for '{' ... '}'
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            parsed = json.loads(text[start_idx : end_idx + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    logger.warning("Failed to decode JSON from Gemini response; creating fallback dict.")
    return {
        "error": "Failed to parse JSON response from Gemini",
        "raw_response": raw_text,
        "chart_type": "other",
        "title": "Unparsed Document Visual",
        "summary": raw_text.strip() or "Visual element extracted from document.",
        "data_points": [],
        "time_period": None,
        "is_financial": False,
    }


def analyze_chart(
    image_path: Union[str, Path],
    page_context: str = "",
    max_retries: int = 2,
    retry_delay: float = 1.5,
) -> Dict[str, Any]:
    """Analyze a chart or image using the Gemini Multimodal API.

    Loads the image, formats a structured prompt, invokes Gemini,
    and returns a normalized dictionary of structured financial insights.

    Args:
        image_path: Path to the image file.
        page_context: Optional surrounding narrative text for context.
        max_retries: Maximum retry attempts for transient API errors.
        retry_delay: Delay in seconds between retries.

    Returns:
        Dict[str, Any]: Structured dictionary containing chart_type, title,
                        summary, data_points, time_period, and is_financial.
    """
    path = Path(image_path)
    if not path.is_file():
        logger.error(f"Image file not found at: {path}")
        return {
            "error": f"Image file not found: {path}",
            "chart_type": "other",
            "title": path.stem,
            "summary": f"Image file not found at {path}",
            "data_points": [],
            "time_period": None,
            "is_financial": False,
        }

    try:
        pil_img = Image.open(path)
        # Ensure image is loaded in memory
        pil_img.load()
    except Exception as e:
        logger.error(f"Failed to open image file '{path}': {e}")
        return {
            "error": f"Failed to load image: {e}",
            "chart_type": "other",
            "title": path.stem,
            "summary": "Corrupt or unreadable image file.",
            "data_points": [],
            "time_period": None,
            "is_financial": False,
        }

    prompt = CHART_ANALYSIS_PROMPT.format(context=page_context or "None provided.")

    model = get_gemini_model()

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            logger.info(
                f"Analyzing image '{path.name}' with Gemini (attempt {attempt + 1}/{max_retries + 1})..."
            )
            response = model.generate_content([prompt, pil_img])

            if not response or not response.text:
                raise ValueError("Received empty text response from Gemini API.")

            result = clean_json_response(response.text)
            logger.info(
                f"Successfully analyzed '{path.name}' -> [{result.get('chart_type')}] {result.get('title')}"
            )
            return result

        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1} failed for image '{path.name}': {e}")
            if attempt < max_retries:
                time.sleep(retry_delay * (2**attempt))

    logger.error(f"All {max_retries + 1} attempts failed for image '{path.name}': {last_error}")
    return {
        "error": str(last_error),
        "chart_type": "other",
        "title": path.stem,
        "summary": f"Error analyzing visual with Gemini: {last_error}",
        "data_points": [],
        "time_period": None,
        "is_financial": False,
    }
