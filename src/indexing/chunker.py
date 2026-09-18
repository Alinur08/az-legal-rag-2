"""Article chunking module for the Azerbaijani Labour Code.

Splits structured legal articles into retrieval chunks while preserving article
numbers, headings, and chapter context across all generated chunks for citation
traceability.
"""

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from src.ingestion.article_parser import ArticleRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ArticleChunk:
    """Represents a discrete text chunk indexed for retrieval.

    Attributes:
        chunk_id: Unique identifier, e.g. 'art_70_chunk_0'.
        article_number: Number of the parent article, e.g. '70'.
        title: Title of the parent article.
        section: Section (Bölmə) heading.
        chapter: Chapter (Fəsil) heading.
        text: Full chunk text including article header for grounding.
        chunk_index: 0-indexed position within parent article.
        total_chunks: Total chunks generated from the parent article.
        page_start: First page of parent article in source PDF.
        page_end: Last page of parent article in source PDF.
    """
    chunk_id: str
    article_number: str
    title: str
    section: str
    chapter: str
    text: str
    chunk_index: int
    total_chunks: int
    page_start: int
    page_end: int

    def to_dict(self) -> Dict[str, Any]:
        """Converts chunk instance to dictionary representation."""
        return asdict(self)


def chunk_article(
    article: ArticleRecord,
    max_chunk_chars: int = 1200,
    overlap_chars: int = 150,
) -> List[ArticleChunk]:
    """Splits a single legal article into one or more contextualized chunks.

    For articles shorter than max_chunk_chars, creates a single chunk.
    For longer articles, splits by sub-clauses or paragraphs while prepending
    the article heading so semantic identity is never lost.

    Args:
        article: Structured ArticleRecord from ingestion.
        max_chunk_chars: Maximum character target for a chunk.
        overlap_chars: Overlap character count when splitting long bodies.

    Returns:
        List of ArticleChunk instances.
    """
    header_prefix = f"Maddə {article.article_number}. {article.title}\n\n"
    content = article.content.strip()

    # Short article: return single chunk
    if len(content) <= max_chunk_chars:
        full_text = f"{header_prefix}{content}".strip()
        return [
            ArticleChunk(
                chunk_id=f"art_{article.article_number}_chunk_0",
                article_number=article.article_number,
                title=article.title,
                section=article.section,
                chapter=article.chapter,
                text=full_text,
                chunk_index=0,
                total_chunks=1,
                page_start=article.page_start,
                page_end=article.page_end,
            )
        ]

    # Split by sub-clauses (e.g. numbered items '1.', '2.', or paragraphs)
    clause_delimiter = re.compile(
        r"(?=(?:^\d+\.\s+|^[a-zçəğıöşü]\)\s+|\n\d+\.\s+|\n[a-zçəğıöşü]\)\s+))",
        re.MULTILINE
    )
    raw_parts = [
        p.strip() for p in clause_delimiter.split(content) if p.strip()
    ]

    if not raw_parts:
        raw_parts = [content]

    # Group parts into chunk bodies respecting max_chunk_chars
    grouped_bodies: List[str] = []
    current_body = ""

    for part in raw_parts:
        if not current_body:
            current_body = part
        elif len(current_body) + len(part) + 1 <= max_chunk_chars:
            current_body += "\n" + part
        else:
            grouped_bodies.append(current_body)
            current_body = part

    if current_body:
        grouped_bodies.append(current_body)

    total_chunks = len(grouped_bodies)
    chunks: List[ArticleChunk] = []

    for idx, body in enumerate(grouped_bodies):
        chunk_text = (
            f"Maddə {article.article_number}. {article.title} "
            f"(Hissə {idx + 1}/{total_chunks})\n\n{body}"
        ).strip()

        chunks.append(
            ArticleChunk(
                chunk_id=f"art_{article.article_number}_chunk_{idx}",
                article_number=article.article_number,
                title=article.title,
                section=article.section,
                chapter=article.chapter,
                text=chunk_text,
                chunk_index=idx,
                total_chunks=total_chunks,
                page_start=article.page_start,
                page_end=article.page_end,
            )
        )

    return chunks


def chunk_all_articles(
    articles: List[ArticleRecord],
    max_chunk_chars: int = 1200,
) -> List[ArticleChunk]:
    """Chunks a list of ArticleRecord instances.

    Args:
        articles: List of parsed ArticleRecord objects.
        max_chunk_chars: Max character length per chunk.

    Returns:
        List of ArticleChunk instances across all articles.
    """
    all_chunks: List[ArticleChunk] = []
    for art in articles:
        all_chunks.extend(
            chunk_article(art, max_chunk_chars=max_chunk_chars)
        )
    logger.info(
        f"Chunked {len(articles)} articles into {len(all_chunks)} chunks."
    )
    return all_chunks
