"""Retrieval package initialization."""

from src.retrieval.retriever import (
    LegalRetriever,
    RetrievedChunk,
    retrieve,
)

__all__ = [
    "LegalRetriever",
    "RetrievedChunk",
    "retrieve",
]
