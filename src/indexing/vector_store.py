"""Vector store and embedding management module.

Encapsulates dense embedding generation via sentence-transformers (BAAI/bge-m3)
and persistence in ChromaDB vector database with cosine similarity index.
"""

from typing import Any, Dict, List, Optional
import chromadb
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.indexing.chunker import ArticleChunk
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingManager:
    """Manages loading and inference for the dense sentence transformer."""

    _instance: Optional["EmbeddingManager"] = None

    def __init__(self, model_name: Optional[str] = None):
        """Initializes the embedding model.

        Args:
            model_name: HuggingFace model identifier. Defaults to settings.
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazily loads and returns the SentenceTransformer model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[List[float]]:
        """Generates normalized dense embeddings for a list of strings.

        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for model inference.

        Returns:
            List of float vectors (embeddings).
        """
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embeds a single query string.

        Args:
            query: Query text string.

        Returns:
            List of floats representing query embedding.
        """
        vectors = self.embed_texts([query])
        return vectors[0]


class VectorStoreManager:
    """Manages ChromaDB vector collection persistence and dense search."""

    COLLECTION_NAME = "az_labour_code"

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
    ):
        """Initializes Chroma persistent client and collection.

        Args:
            persist_dir: Directory path for Chroma persistence.
            embedding_manager: Optional EmbeddingManager instance.
        """
        self.persist_dir = str(persist_dir or settings.CHROMA_PERSIST_DIR)
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = None

    @property
    def collection(self) -> chromadb.Collection:
        """Returns the Chroma collection with cosine distance configuration."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def reset_collection(self) -> None:
        """Deletes and recreates the collection to rebuild from scratch."""
        try:
            self.client.delete_collection(name=self.COLLECTION_NAME)
            logger.info(f"Deleted existing collection: {self.COLLECTION_NAME}")
        except Exception:
            pass
        self._collection = self.client.create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Created fresh collection: {self.COLLECTION_NAME}")

    def add_chunks(self, chunks: List[ArticleChunk]) -> None:
        """Embeds and inserts article chunks into the collection.

        Args:
            chunks: List of ArticleChunk objects to index.
        """
        if not chunks:
            logger.warning("No chunks provided to index.")
            return

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "article_number": c.article_number,
                "title": c.title,
                "section": c.section,
                "chapter": c.chapter,
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
                "page_start": c.page_start,
                "page_end": c.page_end,
            }
            for c in chunks
        ]

        logger.info(f"Computing embeddings for {len(chunks)} chunks...")
        embeddings = self.embedding_manager.embed_texts(texts)

        logger.info(f"Inserting {len(chunks)} vectors into ChromaDB...")
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Successfully populated vector store.")

    def query(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Queries the vector store using dense cosine similarity.

        Args:
            query_text: User search query.
            top_k: Maximum number of matches to retrieve.

        Returns:
            List of matching records with chunk data and similarity scores.
        """
        query_vector = self.embedding_manager.embed_query(query_text)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        matches: List[Dict[str, Any]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return matches

        ids = results["ids"][0]
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        for cid, doc, meta, dist in zip(ids, docs, metas, distances):
            # Chroma returns cosine distance in [0, 2]
            # similarity = 1 - distance
            similarity = max(0.0, 1.0 - dist)
            matches.append({
                "chunk_id": cid,
                "text": doc,
                "score": similarity,
                "article_number": meta.get("article_number", ""),
                "title": meta.get("title", ""),
                "section": meta.get("section", ""),
                "chapter": meta.get("chapter", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "total_chunks": meta.get("total_chunks", 1),
                "page_start": meta.get("page_start", 1),
                "page_end": meta.get("page_end", 1),
            })

        return matches
