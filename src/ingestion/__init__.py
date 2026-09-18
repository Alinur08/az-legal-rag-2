"""Ingestion package initialization."""

from src.ingestion.pdf_extractor import PDFExtractor, PageText
from src.ingestion.text_cleaner import (
    is_suspicious_text,
    normalize_azerbaijani_text,
)
from src.ingestion.article_parser import (
    ArticleParser,
    ArticleRecord,
    run_ingestion,
)

__all__ = [
    "PDFExtractor",
    "PageText",
    "is_suspicious_text",
    "normalize_azerbaijani_text",
    "ArticleParser",
    "ArticleRecord",
    "run_ingestion",
]
