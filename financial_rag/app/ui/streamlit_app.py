import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path when executed via `streamlit run`
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.core.schemas import ElementType, QueryResult
from app.generation.pipeline import answer_query
from app.indexing.pipeline import index_document_elements
from app.ingestion.pipeline import ingest_pdf
from app.understanding.pipeline import (
    enrich_elements_with_understanding,
    load_elements_from_json,
    save_elements_to_json,
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Intelligence Assistant",
    page_icon="📊",
    layout="wide",
)

# Initialize Session State
st.session_state.setdefault("query_input", "")
st.session_state.setdefault("last_result", None)
st.session_state.setdefault("last_query", "")


# -----------------------------------------------------------------------------
# 2. Sidebar: System Status & Example Queries
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("📊 Financial RAG Status")

    st.markdown("---")
    st.subheader("💡 Example Queries")
    st.caption("Click any query below to pre-fill the search bar:")

    example_queries = [
        "What was NVIDIA's total shareholder return and capital allocation?",
        "How did NVIDIA's segment revenue change between Data Center and Gaming?",
        "What does the board composition and diversity look like?",
        "How did gross margins change and what were the inventory charges?",
        "What was the total capital returned to shareholders via stock repurchases?",
    ]

    for ex in example_queries:
        if st.button(ex, key=f"btn_{ex}", use_container_width=True):
            st.session_state["query_input"] = ex
            st.rerun()

    st.markdown("---")
    st.caption(
        "Powered by **Google Gemini** multimodal vision & embeddings with **ChromaDB** vector storage."
    )


# -----------------------------------------------------------------------------
# 3. Main Header & Navigation Tabs
# -----------------------------------------------------------------------------
st.title("📊 Financial Intelligence Assistant")
st.markdown(
    "Analyze financial disclosures — ingest documents, extract multimodal elements, and "
    "get grounded answers backed by verifiable citations from text narrative, tables, and charts."
)

tab_query, tab_upload = st.tabs(["🔍 Ask Questions", "📤 Upload New Document"])

def escape_markdown_dollars(text: str) -> str:
    """Escape dollar signs so Streamlit's markdown parser does not treat them as LaTeX math delimiters."""
    if not text:
        return ""
    return text.replace("$", "\\$")


# =============================================================================
# TAB 1: Ask Questions
# =============================================================================
with tab_query:
    with st.container():
        query_text = st.text_input(
            label="Financial Analyst Question",
            placeholder="e.g. What was NVIDIA's revenue growth, operating margin, and TSR performance?",
            key="query_input",
        )

        col1, col2 = st.columns([1, 5])
        with col1:
            submit_btn = st.button("Analyze", type="primary", use_container_width=True)

    if submit_btn and query_text.strip():
        with st.spinner("Analyzing financial documents and synthesizing grounded response..."):
            try:
                result: QueryResult = answer_query(
                    query=query_text.strip(),
                    collection_name="financial_rag",
                    n_results=8,
                )
                st.session_state["last_result"] = result
                st.session_state["last_query"] = query_text.strip()
            except Exception as e:
                logger.exception(f"Error during query execution: {e}")
                st.session_state["last_result"] = None
                st.error(
                    "⚠️ I couldn't process that query due to a system error. "
                    "Please verify your API key and connection, or try rephrasing your question."
                )

    if st.session_state.get("last_result") is not None:
        res: QueryResult = st.session_state["last_result"]
        current_q = st.session_state.get("last_query", "")

        st.markdown(f"### 💡 Analyst Synthesis: *\"{escape_markdown_dollars(current_q)}\"*")
        st.markdown(escape_markdown_dollars(res.answer))
        st.markdown("---")

        # Sources & Citations Display
        st.subheader(f"🔍 Grounded Sources & Citations ({len(res.citations)})")

        if res.citations:
            st.caption("Expand any source below to inspect the original excerpt and page provenance:")

            # Badge mapping helper
            def get_type_badge(elem_type: Optional[ElementType]) -> str:
                type_str = (
                    elem_type.value if hasattr(elem_type, "value") else str(elem_type or "").lower()
                )
                if type_str == "chart":
                    return "📈 [CHART]"
                elif type_str == "table":
                    return "📊 [TABLE]"
                elif type_str == "image":
                    return "🖼️ [IMAGE]"
                return "📄 [TEXT]"

            for idx, cit in enumerate(res.citations, start=1):
                badge = get_type_badge(cit.element_type)
                expander_title = (
                    f"{badge} Source {idx}: {cit.doc_name} (Page {cit.page_number}) "
                    f"— ID: `{cit.element_id}`"
                )

                with st.expander(expander_title, expanded=False):
                    if cit.snippet:
                        st.markdown("**Grounded Excerpt:**")
                        st.info(escape_markdown_dollars(cit.snippet))
                    else:
                        st.caption("No excerpt available.")

                    if cit.bbox:
                        st.caption(f"Bounding Box Coordinates (x0, y0, x1, y1): `{cit.bbox}`")
        else:
            st.info("No specific source elements were cited for this query.")

        # Inspect All Retrieved Context Elements
        if res.retrieved_elements:
            with st.expander(
                f"📚 Inspect Full Retrieved Context ({len(res.retrieved_elements)} Elements)",
                expanded=False,
            ):
                for i, elem in enumerate(res.retrieved_elements, start=1):
                    elem_type = (
                        elem.element_type.value
                        if hasattr(elem.element_type, "value")
                        else str(elem.element_type).upper()
                    )
                    st.markdown(
                        f"**Element {i}:** `{elem.element_id}` | Type: `{elem_type.upper()}` | "
                        f"Document: `{elem.doc_name}` (Page {elem.page_number})"
                    )
                    st.text(elem.content[:500] + ("..." if len(elem.content) > 500 else ""))
                    st.divider()


# =============================================================================
# TAB 2: Upload New Document
# =============================================================================
with tab_upload:
    st.subheader("📤 Ingest & Process Financial PDF")
    st.markdown(
        "Upload a new financial disclosure (SEC 10-K, annual report, earnings presentation) "
        "to extract text, tables, and visual charts, generate Gemini embeddings, and index into RAG."
    )

    uploaded_file = st.file_uploader("Select a financial PDF file", type=["pdf"])

    max_calls = st.number_input(
        "Max chart analysis calls",
        min_value=0,
        value=18,
        step=1,
        help=(
            "Chart/image understanding uses Google's free Gemini API tier, limited to ~20 requests/day. "
            "Set this below your remaining daily quota. Large documents may need multiple days — "
            "re-upload the same file later to resume and analyze the rest."
        ),
    )
    st.caption(
        "Chart/image understanding uses Google's free Gemini API tier, limited to ~20 requests/day. "
        "Set this below your remaining daily quota. Large documents may need multiple days — "
        "re-upload the same file later to resume and analyze the rest."
    )

    process_btn = st.button("Process Document", type="primary", disabled=(uploaded_file is None))

    if process_btn and uploaded_file is not None:
        raw_pdfs_dir = Path("data/raw_pdfs")
        raw_pdfs_dir.mkdir(parents=True, exist_ok=True)
        saved_path = raw_pdfs_dir / uploaded_file.name

        # Save uploaded file to disk
        with open(saved_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        doc_name = saved_path.stem
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        enriched_json_path = processed_dir / f"{doc_name}_enriched.json"

        status = st.status("Starting document processing pipeline...", expanded=True)

        raw_elements = []
        enriched_elements = []
        pipeline_error = None

        try:
            # Stage 1: Ingestion
            status.update(label="Extracting text, tables, and images...", state="running")
            st.write(f"📄 Ingesting `{uploaded_file.name}` (text blocks, financial tables, chart images)...")
            raw_elements = ingest_pdf(file_path=saved_path)
            st.write(f"✓ Extracted **{len(raw_elements)}** document elements.")

            # Stage 2: Resume Check & Multimodal Understanding
            existing_elements = None
            if enriched_json_path.is_file():
                try:
                    existing_elements = load_elements_from_json(enriched_json_path)
                    st.write(f"ℹ️ Found existing cached elements in `{enriched_json_path.name}` — resuming with cache.")
                except Exception as ex_load:
                    logger.warning(f"Could not load existing elements from {enriched_json_path}: {ex_load}")

            status.update(label="Analyzing charts with Gemini...", state="running")
            st.write(f"🤖 Running Gemini Multimodal vision analysis on charts (max calls: `{max_calls}`)...")
            enriched_elements = enrich_elements_with_understanding(
                elements=raw_elements,
                existing_elements=existing_elements,
                max_calls=int(max_calls),
            )

            # Persist progress
            save_elements_to_json(enriched_elements, enriched_json_path)
            st.write(f"✓ Saved enriched elements to `{enriched_json_path.name}`.")

            # Stage 3: Vector Indexing
            status.update(label="Indexing into vector database...", state="running")
            st.write("⚡ Generating Gemini embeddings and indexing elements into ChromaDB vector store...")
            indexed_count = index_document_elements(elements=enriched_elements)
            st.write(f"✓ Successfully indexed **{indexed_count}** elements into ChromaDB collection.")

            status.update(label="Document processing complete!", state="complete", expanded=False)

        except Exception as e:
            pipeline_error = e
            logger.exception(f"Error processing PDF document '{uploaded_file.name}': {e}")
            status.update(label="Document processing halted due to an error.", state="error", expanded=True)
            st.error(
                f"⚠️ **Processing Error:** {e}\n\n"
                "Attempting to save and index any document elements that were successfully processed..."
            )

            # Attempt recovery of partial progress
            if enriched_elements:
                try:
                    save_elements_to_json(enriched_elements, enriched_json_path)
                    index_document_elements(elements=enriched_elements)
                except Exception as save_err:
                    logger.exception(f"Secondary error during partial progress recovery: {save_err}")

        # Post-Processing Summary
        if enriched_elements:
            counts = Counter(
                el.element_type.value if hasattr(el.element_type, "value") else str(el.element_type)
                for el in enriched_elements
            )
            visual_elements = [
                el for el in enriched_elements
                if (el.element_type.value if hasattr(el.element_type, "value") else str(el.element_type)) in ("chart", "image")
            ]
            analyzed_visuals = sum(
                1 for el in visual_elements
                if isinstance(el.metadata.get("gemini_analysis"), dict)
                and "error" not in el.metadata["gemini_analysis"]
            )
            pending_visuals = len(visual_elements) - analyzed_visuals

            st.success("🎉 Document Ingestion & Processing Summary")
            st.markdown(
                f"- **Total Extracted Elements:** `{len(enriched_elements)}`\n"
                f"- **Element Breakdown:** Text blocks: `{counts.get(ElementType.TEXT.value, 0)}` | "
                f"Tables: `{counts.get(ElementType.TABLE.value, 0)}` | "
                f"Charts & Images: `{len(visual_elements)}`\n"
                f"- **Gemini Multimodal Analysis:** `{analyzed_visuals}` analyzed | `{pending_visuals}` pending/quota-limited\n"
                f"- **RAG Status:** Document is fully indexed and searchable in the **Ask Questions** tab!"
            )
            if pipeline_error:
                st.info(
                    "💡 *Resume Note:* Processing stopped early (e.g., API rate limit). "
                    "All completed chart analyses have been saved. Re-upload this file later to resume processing!"
                )

