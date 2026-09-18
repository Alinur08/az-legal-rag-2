# az-legal-rag: Phase Completion Reports & Architecture Walkthrough

This document records the design decisions, implemented files, testing outputs, and validation steps for each phase of the **az-legal-rag** project.

---

## Phase 0 — Conventions, Architecture, and Libraries

### Decisions & Approved Specifications
- **Python Version**: Python 3.12 (standardized on 3.12.10).
- **Core Dependencies**:
  - `pymupdf`: PDF text extraction with layout and Unicode font mapping.
  - `sentence-transformers` & `torch`: Dense vector representation using multilingual embeddings (`BAAI/bge-m3`).
  - `rank-bm25`: BM25Okapi lexical retrieval with positive IDF flooring.
  - `chromadb`: Local-first persistent vector store (`cosine` distance).
  - `openai`: LLM generation and structured slot extraction client (`gpt-4o-mini`).
  - `scikit-learn`: Multinomial Logistic Regression dispute-category classifier.
  - `pydantic` & `pydantic-settings`: Strongly-typed schema validation and environment management.
  - `streamlit`: Multi-turn conversational interface.
  - `pycodestyle`: Strict PEP 8 code quality enforcement.
- **Conventions**:
  - Pydantic models for external data/schemas; `@dataclass` for lightweight internal data transfer objects.
  - Google-style docstrings across all modules, classes, and functions.
  - Centralized settings in [src/config.py](file:///c:/Users/User/Desktop/legal-rag/az-legal-rag/src/config.py).
  - Standardized logging via [src/utils/logger.py](file:///c:/Users/User/Desktop/legal-rag/az-legal-rag/src/utils/logger.py).
- **Legal Safety Guardrails**:
  - Fixed Azerbaijani fallback: *"Bu mövzu haqqında Əmək Məcəlləsində dəqiq maddə tapılmadı. Zəhmət olmasa sualınızı və ya vəziyyətinizi daha ətraflı qeyd edin."*
  - Mandatory disclaimer on all outputs: *"XƏBƏRDARLIQ: Bu sistem yalnız məlumatlandırma xarakteri daşıyır və peşəkar hüquqi məsləhət hesab edilmir."*

---

## Phase 1 — PDF Ingestion & Article Parsing

### Implemented Modules
- **[src/config.py](file:///c:/Users/User/Desktop/legal-rag/az-legal-rag/src/config.py)**: Central configuration for directory paths, thresholds, and legal disclaimers.
- **[src/ingestion/pdf_extractor.py](file:///c:/Users/User/Desktop/legal-rag/az-legal-rag/src/ingestion/pdf_extractor.py)**: PyMuPDF-based text extractor preserving page numbers and text blocks.
- **[src/ingestion/text_cleaner.py](file:///c:/Users/User/Desktop/legal-rag/az-legal-rag/src/ingestion/text_cleaner.py)**: Normalizes Azerbaijani Latin diacritics (`ə, ğ, ı, ö, ş, ü, ç`), repairs Cyrillic lookalikes (`\u04D9` -> `ə`), rejoins hyphenated line breaks, and flags suspicious or corrupted extractions.
- **[src/ingestion/article_parser.py](file:///c:/Users/User/Desktop/legal-rag/az-legal-rag/src/ingestion/article_parser.py)**: State machine tracking Section (`BÖLMƏ`), Chapter (`FƏSİL`), and Article (`Maddə N`) boundaries. Serializes structured records to `data/processed/articles.json` and `.jsonl`.
- **[src/ingestion/validate_parser.py](file:///c:/Users/User/Desktop/legal-rag/az-legal-rag/src/ingestion/validate_parser.py)**: Spot-check validator verifying key legal articles (e.g., Maddə 70 and Maddə 74) for domain terms and completeness.
- **[tests/test_ingestion.py](file:///c:/Users/User/Desktop/legal-rag/az-legal-rag/tests/test_ingestion.py)**: End-to-end integration test creating Azerbaijani legal PDF pages and validating parsing.

### Verification Results
- `python -m tests.test_ingestion`:
  - `OK: No suspicious articles detected.`
  - `PASS: Maddə 70 - 'Əmək müqaviləsinin işəgötürən tərəfindən ləğv edilməsinin əsasları'`
  - `PASS: Maddə 74 - 'Tərəflərin iradəsindən asılı olmayan hallarda əmək müqaviləsinə xitam verilməsinin əsasları'`
  - `Validation Result: PASSED`
- `python -m pycodestyle src tests`: 0 warnings, 100% compliant.

---

## Phase 2 — Indexing (Chunking, Vector Store, & BM25)

### Implemented Modules
- **[src/indexing/chunker.py](file:///c:/Users/User/Desktop/legal-rag/az-legal-rag/src/indexing/chunker.py)**:
  - Generates `ArticleChunk` instances.
  - Preserves article number and title prefix in every chunk (`Maddə N. [Title] (Hissə X/Y)`) so that chunks retain full legal context and citation traceability.
- **[src/indexing/vector_store.py](file:///c:/Users/User/Desktop/legal-rag/az-legal-rag/src/indexing/vector_store.py)**:
  - `EmbeddingManager`: Lazy-loaded `SentenceTransformer` with batch vector normalization.
  - `VectorStoreManager`: ChromaDB persistent client managing collection `az_labour_code` with cosine distance.
- **[src/indexing/bm25_indexer.py](file:///c:/Users/User/Desktop/legal-rag/az-legal-rag/src/indexing/bm25_indexer.py)**:
  - Azerbaijani legal tokenizer (`tokenize_azerbaijani`).
  - `BM25Manager`: Fits BM25 with an IDF floor to handle small collections and edge cases; serializes to `data/processed/bm25_index.pkl`.
- **[src/indexing/build_index.py](file:///c:/Users/User/Desktop/legal-rag/az-legal-rag/src/indexing/build_index.py)**:
  - CLI entrypoint (`python -m src.indexing.build_index`) to rebuild both ChromaDB and BM25 indices from scratch.
- **[tests/test_indexing.py](file:///c:/Users/User/Desktop/legal-rag/az-legal-rag/tests/test_indexing.py)**:
  - Validates chunking, BM25 exact term matching, and ChromaDB vector persistence.

### Verification Results
- `python -m tests.test_indexing`:
  - `PASS: Chunking test (3 sub-chunks generated)`
  - `PASS: BM25 lexical indexing & query test`
  - `PASS: ChromaDB vector store integration test`
  - `ALL INDEXING TESTS PASSED!`
- `python -m src.indexing.build_index`:
  - ChromaDB persisted at `data/chroma_db` with `BAAI/bge-m3` (1024-dim)
  - BM25 persisted at `data/processed/bm25_index.pkl`
- `python -m pycodestyle src tests`: 0 warnings, 100% compliant.

---

## Phase 3 — Hybrid Retrieval (Dense + Lexical BM25 Fusion)

### Implemented Modules
- **[src/retrieval/retriever.py](file:///c:/Users/User/Desktop/legal-rag/az-legal-rag/src/retrieval/retriever.py)**:
  - `RetrievedChunk`: Dataclass containing `chunk_id`, `article_number`, `title`, `text`, `score` (combined hybrid confidence in $[0.0, 1.0]$), `dense_score`, `bm25_score`, and `rank`.
  - `LegalRetriever`: Combines ChromaDB vector search and BM25 lexical retrieval using Reciprocal Rank Fusion (RRF, $k=60$) alongside linear score confidence blending.
  - Supports an optional `case_state` parameter for query expansion during later multi-turn dialogue phases.
  - Standalone module-level convenience function `retrieve(query: str, top_k: int = 5)`.
- **[src/retrieval/__init__.py](file:///c:/Users/User/Desktop/legal-rag/az-legal-rag/src/retrieval/__init__.py)**:
  - Clean exports for `LegalRetriever`, `RetrievedChunk`, and `retrieve`.
- **[tests/test_retrieval.py](file:///c:/Users/User/Desktop/legal-rag/az-legal-rag/tests/test_retrieval.py)**:
  - Validates exact keyword retrieval (e.g. `işçilərin ştatları ixtisarı` $\rightarrow$ Maddə 70).
  - Validates Azerbaijani semantic paraphrase queries (e.g. `işçi orduya hərbi xidmətə çağırıldıqda əmək müqaviləsi` $\rightarrow$ Maddə 74).
  - Validates empty query handling and score boundedness ($[0.0, 1.0]$).

### Verification Results
- `python -m tests.test_retrieval`:
  - `PASS: Exact keyword retrieval for Maddə 70`
  - `PASS: Semantic retrieval for Maddə 74`
  - `PASS: Empty query edge case`
  - `PASS: Module-level retrieve() entrypoint`
  - `ALL RETRIEVAL TESTS PASSED SUCCESSFULLY!`
- `python -m pycodestyle src tests`: 0 warnings, 100% compliant.
