"""One-shot legal Q&A grounded in retrieved Labour Code articles.

Pipeline: question -> retrieve -> relevance guardrail -> LLM ->
citation guardrail -> answer. Every legal-facing output ends with the
centralized disclaimer, and every failure path returns the fixed
Azerbaijani fallback instead of an invented answer.

Usage from the repo root::

    python -m src.generation.qa_prompts "Mən haqsız işdən çıxarılmışam"
"""

import sys
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from src.config import settings
from src.guardrails import (
    add_legal_disclaimer,
    is_relevant,
    validate_article_citations,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

NOT_FOUND_TOKEN = "NOT_FOUND"

SYSTEM_PROMPT = (
    "Sən Azərbaycan Respublikasının Əmək Məcəlləsi üzrə hüquqi "
    "məlumat köməkçisisən. Yalnız aşağıda verilən kontekst "
    "maddələrinə əsaslanaraq cavab ver. Kontekstdə olmayan maddə "
    "nömrəsi, müddət, məbləğ və ya hüquqi iddia yazma. Hər hüquqi "
    "iddiaya 'Maddə N' formatında istinad et. Kontekst sualı "
    "cavablandırmaq üçün kifayət deyilsə, yalnız bu sözü yaz: "
    + NOT_FOUND_TOKEN + ". Cavabı Azərbaycan dilində, qısa və aydın "
    "yaz."
)

USER_TEMPLATE = "Kontekst:\n{context}\n\nSual: {question}"

LLMFn = Callable[[str, str], str]
RetrieveFn = Callable[..., List[Any]]


@dataclass
class QAResult:
    """Outcome of one QA call.

    Attributes:
        answer: Final text shown to the user, disclaimer included.
        grounded: True only if an LLM answer passed all guardrails.
        sources: Retrieved chunks used as context (empty on fallback).
        flagged_articles: Article numbers the LLM cited that were not
            present in the retrieved context.
    """

    answer: str
    grounded: bool
    sources: List[Any] = field(default_factory=list)
    flagged_articles: List[str] = field(default_factory=list)


def format_context(chunks: List[Any]) -> str:
    """Renders retrieved chunks as a labelled context block.

    Args:
        chunks: RetrievedChunk objects with ``article_number``,
            ``title`` and ``text`` attributes.

    Returns:
        Text with one ``[Maddə N]`` block per chunk.
    """
    blocks = []
    for chunk in chunks:
        title = f"[Maddə {chunk.article_number}]"
        if chunk.title:
            title += f" ({chunk.title})"
        blocks.append(f"{title}\n{chunk.text}")
    return "\n\n".join(blocks)


def build_prompts(question: str, chunks: List[Any]) -> Tuple[str, str]:
    """Builds the (system, user) prompt pair for a question.

    Args:
        question: The user's legal question.
        chunks: Retrieved article chunks.

    Returns:
        Tuple of system prompt and user prompt strings.
    """
    user = USER_TEMPLATE.format(
        context=format_context(chunks), question=question.strip()
    )
    return SYSTEM_PROMPT, user


def call_llm(system: str, user: str) -> str:
    """Calls the configured Gemini model and returns its text.
 
    Uses ``settings.GEMINI_API_KEY`` and ``settings.LLM_MODEL_NAME``.
 
    Args:
        system: System prompt.
        user: User prompt.
 
    Returns:
        The model's text output (empty string if nothing was
        returned, e.g. a blocked or truncated response).
 
    Raises:
        RuntimeError: If no Gemini API key is configured.
    """
    from google import genai
    from google.genai import types
 
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set (.env).")
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=settings.LLM_MODEL_NAME,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            # Headroom: reasoning models count thinking tokens here.
            max_output_tokens=2048,
        ),
    )
    return response.text or ""

def _fallback(flagged: Optional[List[str]] = None) -> QAResult:
    """Builds an ungrounded result carrying the fixed fallback text."""
    return QAResult(
        answer=add_legal_disclaimer(settings.NO_ARTICLE_FALLBACK_AZ),
        grounded=False,
        flagged_articles=flagged or [],
    )


def answer_question(
    question: str,
    top_k: Optional[int] = None,
    llm_fn: Optional[LLMFn] = None,
    retrieve_fn: Optional[RetrieveFn] = None,
) -> QAResult:
    """Answers one legal question, grounded strictly in retrieved text.

    Args:
        question: The user's question in Azerbaijani.
        top_k: Number of chunks to retrieve (config default if None).
        llm_fn: Callable ``(system, user) -> str``; defaults to
            :func:`call_llm`. Injectable for tests.
        retrieve_fn: Retrieval callable; defaults to Phase 3
            ``retrieve``. Injectable for tests.

    Returns:
        A QAResult. On weak retrieval, LLM refusal or ungrounded
        article citations the fixed fallback is returned.
    """
    llm_fn = llm_fn or call_llm
    if retrieve_fn is None:
        # Imported lazily so tests do not load embedding models.
        from src.retrieval import retrieve as retrieve_fn

    chunks = retrieve_fn(question, top_k=top_k)
    if not chunks or not is_relevant(question, chunks):
        logger.info("Retrieval below threshold; returning fallback.")
        return _fallback()

    system, user = build_prompts(question, chunks)
    raw = (llm_fn(system, user) or "").strip()

    if not raw or NOT_FOUND_TOKEN in raw:
        logger.info("LLM reported insufficient context.")
        return _fallback()

    check = validate_article_citations(raw, chunks)
    if not check.is_valid:
        logger.warning(
            "Ungrounded articles cited: %s", check.unsupported_articles
        )
        return _fallback(check.unsupported_articles)

    return QAResult(add_legal_disclaimer(raw), True, chunks)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entrypoint: prints the answer and cited sources."""
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print("Usage: python -m src.generation.qa_prompts <sual>")
        return
    result = answer_question(" ".join(args))
    print(result.answer)
    if result.sources:
        print("\nMənbələr:")
        for chunk in result.sources:
            print(f"  - Maddə {chunk.article_number} "
                  f"(score={chunk.score:.3f})")


if __name__ == "__main__":
    main()
