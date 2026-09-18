"""Unit and integration tests for indexing components.

Tests chunking logic, BM25 lexical indexing & scoring, and ChromaDB vector
store operations.
"""

import sys
import tempfile
from pathlib import Path

from src.indexing.bm25_indexer import BM25Manager, tokenize_azerbaijani
from src.indexing.chunker import ArticleChunk, chunk_article
from src.indexing.vector_store import EmbeddingManager, VectorStoreManager
from src.ingestion.article_parser import ArticleRecord


def test_chunking_short_and_long():
    """Tests chunking logic on both short and multi-clause articles."""
    # Short article
    short_art = ArticleRecord(
        article_number="74",
        title="Tərəflərin iradəsindən asılı olmayan hallar",
        section="BÖLMƏ I",
        chapter="FƏSİL 10",
        content="İşçi həqiqi hərbi xidmətə çağırıldıqda xitam verilir.",
        full_text=(
            "Maddə 74. Tərəflərin iradəsindən asılı olmayan hallar\n\n"
            "İşçi həqiqi hərbi xidmətə çağırıldıqda xitam verilir."
        ),
        page_start=2,
        page_end=2,
        is_suspicious=False,
    )
    chunks = chunk_article(short_art, max_chunk_chars=500)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "art_74_chunk_0"
    assert "Maddə 74." in chunks[0].text

    # Long article with sub-clauses
    clause_1 = "1. Birinci bəndin mətni geniş hüquqi izah verir. " * 15
    clause_2 = "\n2. İkinci bənd işəgötürənin vəzifələrini təsvir edir. " * 15
    long_content = clause_1 + clause_2
    long_art = ArticleRecord(
        article_number="70",
        title="Əmək müqaviləsinin ləğv edilməsi",
        section="BÖLMƏ I",
        chapter="FƏSİL 10",
        content=long_content,
        full_text="Maddə 70...",
        page_start=1,
        page_end=1,
        is_suspicious=False,
    )
    long_chunks = chunk_article(long_art, max_chunk_chars=600)
    assert len(long_chunks) > 1
    for i, c in enumerate(long_chunks):
        assert c.article_number == "70"
        assert c.chunk_index == i
        assert c.total_chunks == len(long_chunks)
        assert f"Maddə 70." in c.text
    print(f"PASS: Chunking test ({len(long_chunks)} sub-chunks generated)")


def test_bm25_indexing_and_query():
    """Tests BM25 tokenization, index building, persistence, and querying."""
    chunks = [
        ArticleChunk(
            chunk_id="art_70_chunk_0",
            article_number="70",
            title="İşəgötürən tərəfindən ləğv",
            section="BÖLMƏ I",
            chapter="FƏSİL 10",
            text=(
                "Maddə 70. İşçilərin ştatları və ya sayı ixtisarı "
                "edildikdə ləğv edilir."
            ),
            chunk_index=0,
            total_chunks=1,
            page_start=1,
            page_end=1,
        ),
        ArticleChunk(
            chunk_id="art_74_chunk_0",
            article_number="74",
            title="Tərəflərin iradəsindən asılı olmayan",
            section="BÖLMƏ I",
            chapter="FƏSİL 10",
            text=(
                "Maddə 74. İşçi həqiqi hərbi xidmətə çağırıldıqda "
                "əmək müqaviləsi ləğv olunur."
            ),
            chunk_index=0,
            total_chunks=1,
            page_start=2,
            page_end=2,
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        index_file = Path(tmpdir) / "bm25.pkl"

        mgr = BM25Manager()
        mgr.build_index(chunks)
        mgr.save(index_file)

        mgr_loaded = BM25Manager()
        mgr_loaded.load(index_file)

        # Query for dismissal / staff reduction
        results = mgr_loaded.query("ştat ixtisarı", top_k=2)
        assert len(results) > 0
        assert results[0]["article_number"] == "70"

        # Query for military service
        results_mil = mgr_loaded.query("hərbi xidmət", top_k=2)
        assert len(results_mil) > 0
        assert results_mil[0]["article_number"] == "74"

    print("PASS: BM25 lexical indexing & query test")


def test_vector_store_with_chroma():
    """Tests ChromaDB vector persistence with EmbeddingManager."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        # Lightweight multilingual sentence transformer for fast unit testing
        test_model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        emb_mgr = EmbeddingManager(model_name=test_model_name)
        v_mgr = VectorStoreManager(
            persist_dir=tmpdir, embedding_manager=emb_mgr
        )
        v_mgr.reset_collection()

        chunk = ArticleChunk(
            chunk_id="art_70_chunk_0",
            article_number="70",
            title="Əmək müqaviləsinin ləğvi",
            section="BÖLMƏ I",
            chapter="FƏSİL 10",
            text="Maddə 70. İşçilərin sayı və ya ştatları ixtisarı edildikdə.",
            chunk_index=0,
            total_chunks=1,
            page_start=1,
            page_end=1,
        )
        v_mgr.add_chunks([chunk])

        results = v_mgr.query("işdən çıxarılma və ştat ixtisarı", top_k=1)
        assert len(results) == 1
        assert results[0]["article_number"] == "70"
        assert results[0]["score"] > 0.0

    print("PASS: ChromaDB vector store integration test")


if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    test_chunking_short_and_long()
    test_bm25_indexing_and_query()
    test_vector_store_with_chroma()
    print("ALL INDEXING TESTS PASSED!")
