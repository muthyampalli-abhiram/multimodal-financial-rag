"""Grounded answer generation with traceable citations using Gemini LLM reasoning."""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Set

from app.core.schemas import (
    BoundingBox,
    Citation,
    DocumentElement,
    ElementType,
    QueryResult,
)
from app.retrieval.retriever import (
    elements_from_retrieval,
    format_retrieved_context,
)
from app.understanding.gemini_client import get_gemini_model

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are a senior financial analyst and intelligence expert analyzing corporate disclosures (e.g., SEC filings, 10-Ks, annual reports, earnings presentations).

Your task is to answer the analyst's question using ONLY the provided document context below.

Instructions:
1. Strict Grounding: Base your answer exclusively on the facts, numbers, and takeaways directly stated in the context. Do not invent, speculate, or bring in outside information.
2. Verifiable Citations: For every key factual claim, data point, or takeaway, include an inline citation in the exact format [Source: <element_id>] matching the Element ID from the context (for example: "[Source: nvidia_test_subset_chart_6]").
3. Insufficient Context: If the provided context does not contain sufficient details to answer the question, state explicitly and concisely what information is missing rather than guessing.
4. Professional Tone: Maintain a concise, objective, analytical financial analyst tone.

---
Retrieved Document Context:
{context}
---

User Query: {query}

Analyst Answer:"""

CITATION_PATTERN = re.compile(r"\[Source:\s*([^\]]+)\]", re.IGNORECASE)


def build_generation_prompt(query: str, retrieved_context: str) -> str:
    """Construct a grounded generation prompt for Gemini.

    Args:
        query: User or analyst financial query.
        retrieved_context: Structured textual context containing candidate elements.

    Returns:
        str: Fully formatted prompt string.
    """
    clean_context = retrieved_context.strip() if retrieved_context else "No context provided."
    return SYSTEM_PROMPT_TEMPLATE.format(
        context=clean_context,
        query=query.strip(),
    )


def extract_citations_from_text(
    answer_text: str,
    retrieved_results: List[Dict[str, Any]],
) -> List[Citation]:
    """Parse inline [Source: <element_id>] citations from generated text and construct Citation models.

    Args:
        answer_text: Model-generated response text containing citations.
        retrieved_results: List of candidate element dictionaries from retrieval stage.

    Returns:
        List[Citation]: Deduplicated list of Citation objects linked to source elements.
    """
    if not answer_text or not retrieved_results:
        return []

    # Map element_id -> retrieved item dict for fast lookup
    lookup: Dict[str, Dict[str, Any]] = {
        str(item.get("element_id")): item for item in retrieved_results if item.get("element_id")
    }

    raw_matches = CITATION_PATTERN.findall(answer_text)
    citations: List[Citation] = []
    seen_ids: Set[str] = set()

    for match in raw_matches:
        # Match can be single ID or comma/semicolon-separated IDs (e.g. "id1, id2")
        split_ids = re.split(r"[,;]\s*", match)
        for raw_id in split_ids:
            elem_id = raw_id.strip()
            if not elem_id or elem_id in seen_ids:
                continue

            if elem_id in lookup:
                item = lookup[elem_id]
                meta = item.get("metadata", {})
                doc_name = meta.get("doc_name", "Unknown Document")
                page_num = int(meta.get("page_number", 1))
                elem_type_str = meta.get("element_type", "text").lower()

                try:
                    elem_type = ElementType(elem_type_str)
                except ValueError:
                    elem_type = ElementType.TEXT

                # Parse visual bounding box if available
                bbox = None
                if "bbox" in meta and meta["bbox"]:
                    try:
                        bbox_val = (
                            json.loads(meta["bbox"])
                            if isinstance(meta["bbox"], str)
                            else meta["bbox"]
                        )
                        if isinstance(bbox_val, (list, tuple)) and len(bbox_val) == 4:
                            bbox = tuple(float(x) for x in bbox_val)
                    except Exception:
                        pass

                # Extract concise snippet from content
                raw_content = item.get("content", "").strip()
                snippet = raw_content[:300] if raw_content else None

                citation = Citation(
                    element_id=elem_id,
                    doc_name=doc_name,
                    page_number=page_num,
                    element_type=elem_type,
                    snippet=snippet,
                    bbox=bbox,
                )
                citations.append(citation)
                seen_ids.add(elem_id)
            else:
                logger.debug(f"Citation referenced element_id '{elem_id}' not found in retrieved results.")

    return citations


def generate_answer(
    query: str,
    retrieved_results: List[Dict[str, Any]],
    max_retries: int = 2,
    retry_delay: float = 1.0,
) -> QueryResult:
    """Generate a synthesized financial answer with traceable citations from retrieved results.

    Args:
        query: Analyst query text.
        retrieved_results: List of candidate element dictionaries from retrieval.
        max_retries: Maximum attempts for transient model API errors.
        retry_delay: Base delay in seconds between retries (uses exponential backoff).

    Returns:
        QueryResult: Result object containing answer text, grounded citations,
                     and deserialized DocumentElements.
    """
    if not retrieved_results:
        logger.warning(f"No retrieved elements provided to generate_answer for query: '{query}'")
        return QueryResult(
            answer="No relevant document elements were found in the database to answer your query.",
            citations=[],
            retrieved_elements=[],
        )

    # Step a: Format retrieved context
    context_str = format_retrieved_context(retrieved_results)

    # Step b: Build generation prompt
    prompt = build_generation_prompt(query=query, retrieved_context=context_str)

    # Deserialized DocumentElements for QueryResult
    elements = elements_from_retrieval(retrieved_results)

    # Step c: Invoke Gemini model with retry logic
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            model = get_gemini_model()
            logger.info(
                f"Generating answer for query with {len(retrieved_results)} elements "
                f"(attempt {attempt + 1}/{max_retries + 1})..."
            )
            response = model.generate_content(prompt)

            if response and response.text:
                answer_text = response.text.strip()

                # Step d & e: Parse citations
                citations = extract_citations_from_text(
                    answer_text=answer_text,
                    retrieved_results=retrieved_results,
                )

                logger.info(
                    f"Successfully generated answer ({len(answer_text)} chars, {len(citations)} citations)."
                )

                # Step f: Return completed QueryResult
                return QueryResult(
                    answer=answer_text,
                    citations=citations,
                    retrieved_elements=elements,
                )

            raise ValueError("Received empty response from Gemini generation API.")

        except Exception as e:
            last_error = e
            logger.warning(
                f"Generation attempt {attempt + 1}/{max_retries + 1} failed: {e}"
            )
            if attempt < max_retries:
                sleep_time = retry_delay * (2**attempt)
                time.sleep(sleep_time)

    # Step g: Fallback QueryResult on failure
    logger.error(f"Answer generation failed after {max_retries + 1} attempts: {last_error}")
    return QueryResult(
        answer=f"Error generating answer: {last_error}",
        citations=[],
        retrieved_elements=elements,
    )
