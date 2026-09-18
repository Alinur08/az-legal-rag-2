"""CLI entrypoint and manager for building and rebuilding all search indices.

Orchestrates chunking, ChromaDB vector database population (bge-m3), and
BM25 lexical index creation from processed Azerbaijani Labour Code articles.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from src.config import settings
from src.indexing.bm25_indexer import BM25Manager
from src.indexing.chunker import ArticleChunk, chunk_all_articles
from src.indexing.vector_store import EmbeddingManager, VectorStoreManager
from src.ingestion.article_parser import ArticleRecord, run_ingestion
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_articles(json_path: Optional[Path] = None) -> List[ArticleRecord]:
    """Loads structured articles from JSON or initiates ingestion.

    Args:
        json_path: Path to articles.json. Defaults to settings.ARTICLES_JSON.

    Returns:
        List of ArticleRecord instances.
    """
    path = json_path or settings.ARTICLES_JSON
    if not path.is_file():
        logger.warning(
            f"Articles JSON not found at {path}. "
            "Running ingestion from data/raw/..."
        )
        return run_ingestion()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    articles = [ArticleRecord(**item) for item in data]
    logger.info(f"Loaded {len(articles)} articles from {path}")
    return articles


def build_indices(
    articles_path: Optional[Path] = None,
    rebuild_vector: bool = True,
    embedding_model_name: Optional[str] = None,
) -> List[ArticleChunk]:
    """Builds both dense vector and BM25 indices from processed articles.

    Args:
        articles_path: Optional path to articles.json.
        rebuild_vector: Whether to recreate the ChromaDB collection.
        embedding_model_name: Optional override for embedding model.

    Returns:
        List of ArticleChunk instances indexed.
    """
    # 1. Load articles
    articles = load_articles(articles_path)
    if not articles:
        raise ValueError("No articles found to index.")

    # 2. Chunk articles
    chunks = chunk_all_articles(articles)
    if not chunks:
        raise ValueError("Chunking produced 0 chunks.")

    # 3. Build & populate ChromaDB Vector Store
    logger.info("Initializing vector store manager...")
    embedding_mgr = EmbeddingManager(model_name=embedding_model_name)
    vector_mgr = VectorStoreManager(embedding_manager=embedding_mgr)

    if rebuild_vector:
        logger.info("Rebuilding vector store collection from scratch...")
        vector_mgr.reset_collection()

    vector_mgr.add_chunks(chunks)

    # 4. Build & save BM25 Index
    logger.info("Building and saving BM25 lexical index...")
    bm25_mgr = BM25Manager()
    bm25_mgr.build_index(chunks)
    bm25_mgr.save()

    logger.info("=" * 60)
    logger.info("INDEXING COMPLETE SUMMARY:")
    logger.info(f"  - Total Articles Processed: {len(articles)}")
    logger.info(f"  - Total Chunks Indexed:     {len(chunks)}")
    logger.info(f"  - Vector DB Persist Dir:    {settings.CHROMA_PERSIST_DIR}")
    logger.info(f"  - BM25 Index File:          {settings.BM25_INDEX_PATH}")
    logger.info(f"  - Embedding Model:          {embedding_mgr.model_name}")
    logger.info("=" * 60)

    return chunks


def main():
    """CLI entrypoint to trigger index creation or rebuild."""
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Build or rebuild search indices for az-legal-rag."
    )
    parser.add_argument(
        "--articles",
        type=Path,
        default=None,
        help="Path to articles.json (default: data/processed/articles.json)"
    )
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Do not wipe existing Chroma collection before adding chunks."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace embedding model name (default: from config)"
    )

    args = parser.parse_args()
    build_indices(
        articles_path=args.articles,
        rebuild_vector=not args.no_rebuild,
        embedding_model_name=args.model,
    )


if __name__ == "__main__":
    main()
