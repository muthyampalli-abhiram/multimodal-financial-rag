"""Hybrid and vector retrieval module."""


def __getattr__(name: str):
    if name in (
        "retrieve_relevant_elements",
        "build_where_filter",
        "elements_from_retrieval",
        "format_retrieved_context",
    ):
        import app.retrieval.retriever as retriever_mod

        return getattr(retriever_mod, name)

    if name == "detect_query_intent":
        import app.retrieval.query_analyzer as qa_mod

        return getattr(qa_mod, name)

    if name in ("retrieve_with_fallback", "retrieve_for_query"):
        import app.retrieval.pipeline as pipe_mod

        return getattr(pipe_mod, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "retrieve_relevant_elements",
    "build_where_filter",
    "elements_from_retrieval",
    "format_retrieved_context",
    "detect_query_intent",
    "retrieve_with_fallback",
    "retrieve_for_query",
]
