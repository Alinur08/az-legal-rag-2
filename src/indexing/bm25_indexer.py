"""BM25 lexical search indexer for the Azerbaijani Labour Code.

Implements tokenization tuned for Azerbaijani legal vocabulary and utilizes
rank-bm25 (BM25Okapi) for exact keyword, article number, and phrase matching.
"""

import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from rank_bm25 import BM25Okapi

from src.config import settings
from src.indexing.chunker import ArticleChunk
from src.utils.logger import get_logger

logger = get_logger(__name__)


def tokenize_azerbaijani(text: str) -> List[str]:
    """Tokenizes Azerbaijani text for BM25 lexical indexing.

    Converts to lowercase and extracts alphabetic words (including Azerbaijani
    characters ə, ğ, ı, ö, ş, ü, ç) and digits.

    Args:
        text: Input string.

    Returns:
        List of cleaned token strings.
    """
    if not text:
        return []
    # Lowercase with Azerbaijani-aware character replacements
    text_lower = text.lower().replace("İ", "i").replace("I", "ı")
    tokens = re.findall(r"[a-zçəğıöşü0-9]+", text_lower)
    return tokens


class BM25Manager:
    """Manages building, saving, loading, and querying the BM25 index."""

    def __init__(self):
        """Initializes empty BM25Manager."""
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[ArticleChunk] = []

    def build_index(self, chunks: List[ArticleChunk]) -> None:
        """Builds BM25 index from a list of ArticleChunk objects.

        Args:
            chunks: List of ArticleChunk instances to index.
        """
        if not chunks:
            logger.warning("No chunks provided to build BM25 index.")
            return

        self.chunks = chunks
        tokenized_corpus = [
            tokenize_azerbaijani(chunk.text) for chunk in chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Apply a positive floor on IDF values to prevent zero/negative weights
        # when a term appears in exactly half or more of small test collections
        for term, idf_val in self.bm25.idf.items():
            if idf_val <= 0.0:
                self.bm25.idf[term] = 0.25

        logger.info(
            f"Built BM25 index over {len(chunks)} chunks "
            f"(vocab size: {len(self.bm25.idf)} terms)."
        )

    def save(self, path: Optional[Path] = None) -> None:
        """Serializes the BM25 index and associated chunks to disk.

        Args:
            path: Destination file path. Defaults to settings.BM25_INDEX_PATH.
        """
        dest_path = path or settings.BM25_INDEX_PATH
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)
        logger.info(f"Saved BM25 index to {dest_path}")

    def load(self, path: Optional[Path] = None) -> None:
        """Loads serialized BM25 index and chunk objects from disk.

        Args:
            path: Source file path. Defaults to settings.BM25_INDEX_PATH.

        Raises:
            FileNotFoundError: If index file does not exist.
        """
        src_path = path or settings.BM25_INDEX_PATH
        if not src_path.is_file():
            raise FileNotFoundError(f"BM25 index not found at: {src_path}")

        with open(src_path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.chunks = data["chunks"]
        logger.info(
            f"Loaded BM25 index from {src_path} ({len(self.chunks)} chunks)."
        )

    def query(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Searches BM25 index and returns scored top-k chunk records.

        Args:
            query_text: Search query string.
            top_k: Maximum number of ranked results.

        Returns:
            List of matching records with chunk metadata and normalized scores.
        """
        if self.bm25 is None or not self.chunks:
            logger.warning("BM25 index is not initialized.")
            return []

        tokens = tokenize_azerbaijani(query_text)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        max_score = float(max(scores)) if len(scores) > 0 else 0.0

        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        matches: List[Dict[str, Any]] = []
        for idx in ranked_indices:
            raw_score = float(scores[idx])
            if raw_score <= 0.0:
                continue

            norm_score = raw_score / max_score if max_score > 0 else 0.0
            chunk = self.chunks[idx]

            matches.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "score": norm_score,
                "raw_bm25_score": raw_score,
                "article_number": chunk.article_number,
                "title": chunk.title,
                "section": chunk.section,
                "chapter": chunk.chapter,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
            })

        return matches
