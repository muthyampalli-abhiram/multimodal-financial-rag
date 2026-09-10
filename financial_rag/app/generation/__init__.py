"""Answer synthesis and citation logic module."""


def __getattr__(name: str):
    if name in ("build_generation_prompt", "extract_citations_from_text", "generate_answer"):
        import app.generation.answer_generator as gen_mod

        return getattr(gen_mod, name)

    if name == "answer_query":
        import app.generation.pipeline as pipe_mod

        return getattr(pipe_mod, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "build_generation_prompt",
    "extract_citations_from_text",
    "generate_answer",
    "answer_query",
]
