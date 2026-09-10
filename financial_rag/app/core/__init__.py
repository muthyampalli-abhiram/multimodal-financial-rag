"""Core configuration and schemas package."""

from app.core.config import settings
from app.core.schemas import (
    BoundingBox,
    Citation,
    DocumentElement,
    ElementType,
    QueryResult,
)

__all__ = [
    "settings",
    "BoundingBox",
    "Citation",
    "DocumentElement",
    "ElementType",
    "QueryResult",
]
