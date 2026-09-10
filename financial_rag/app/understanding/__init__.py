"""Multimodal understanding module for financial charts and images."""

from app.understanding.gemini_client import get_gemini_model
from app.understanding.chart_analyzer import analyze_chart


def __getattr__(name: str):
    if name in ("enrich_elements_with_understanding", "save_elements_to_json", "load_elements_from_json"):
        import app.understanding.pipeline as pipeline_mod
        return getattr(pipeline_mod, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "get_gemini_model",
    "analyze_chart",
    "enrich_elements_with_understanding",
    "save_elements_to_json",
    "load_elements_from_json",
]
