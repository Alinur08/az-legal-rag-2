"""Lightweight relevance tests with no models, databases, or network calls."""

import unittest
from unittest.mock import MagicMock, patch

from src.guardrails.relevance import is_relevant


class TestRelevance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the real dataclass without loading retrieval dependencies.
        with patch.dict("sys.modules", {
            "src.indexing.bm25_indexer": MagicMock(),
            "src.indexing.vector_store": MagicMock(),
        }):
            from src.retrieval.retriever import RetrievedChunk
            cls.chunk_type = RetrievedChunk

    def make_chunk(self, dense_score, score=1.0, rank=1):
        return self.chunk_type(
            chunk_id=f"chunk_{rank}",
            article_number="1",
            title="Example article",
            section="",
            chapter="",
            text="Example legal text.",
            score=score,
            dense_score=dense_score,
            bm25_score=1.0,
            rank=rank,
            page_start=1,
            page_end=1,
        )

    def test_empty_results(self):
        self.assertFalse(is_relevant("Maaşımı vermirlər", []))

    def test_moderate_score_without_keyword_despite_high_final_score(self):
        results = [self.make_chunk(0.52), self.make_chunk(0.36, rank=2)]
        self.assertFalse(is_relevant("Sabah hava necə olacaq?", results))

    def test_strong_score_without_keyword(self):
        for score in (0.55, 0.62):
            with self.subTest(score=score):
                self.assertTrue(is_relevant("Sual", [self.make_chunk(score)]))

    def test_low_score_even_with_keyword(self):
        self.assertFalse(is_relevant(
            "Maaşımı vermirlər", [self.make_chunk(0.4499)]
        ))

    def test_moderate_score_with_keyword(self):
        self.assertTrue(is_relevant(
            "İşəgötürən mənə məzuniyyət vermir", [self.make_chunk(0.50)]
        ))

    def test_dismissal_example(self):
        self.assertTrue(is_relevant(
            "Məni heç bir səbəb demədən işdən çıxardılar",
            [self.make_chunk(0.4836)],
        ))

    def test_weather_example(self):
        self.assertFalse(is_relevant(
            "Sabah hava necə olacaq?", [self.make_chunk(0.5218)]
        ))

    def test_cats_example(self):
        self.assertFalse(is_relevant(
            "Pişiklər niyə çox yatır?", [self.make_chunk(0.4047)]
        ))

    def test_moderate_boundary_is_inclusive(self):
        self.assertTrue(is_relevant(
            "Maaşımı vermirlər", [self.make_chunk(0.45)]
        ))

    def test_just_below_strong_threshold_without_keyword(self):
        self.assertFalse(is_relevant("Sual", [self.make_chunk(0.5499)]))

    def test_keywords_are_case_insensitive(self):
        for query in ("İŞÇİ", "İŞƏGÖTÜRƏN", "İŞDƏN ÇIXARDILAR", "ƏMƏK HAQQI"):
            with self.subTest(query=query):
                self.assertTrue(is_relevant(query, [self.make_chunk(0.48)]))

    def test_best_dense_score_is_not_first(self):
        results = [
            self.make_chunk(0.40, score=0.95),
            self.make_chunk(0.65, score=0.70, rank=2),
        ]
        self.assertTrue(is_relevant("Sual", results))

    def test_best_moderate_score_is_not_first(self):
        results = [self.make_chunk(0.40), self.make_chunk(0.48, rank=2)]
        self.assertTrue(is_relevant("Maaşımı vermirlər", results))

    def test_custom_thresholds(self):
        results = [self.make_chunk(0.62)]
        self.assertFalse(is_relevant("Sual", results, strong_threshold=0.70))
        self.assertTrue(is_relevant("Sual", results, strong_threshold=0.62))
        self.assertTrue(is_relevant(
            "Maaş", results, strong_threshold=0.70, moderate_threshold=0.62
        ))
        self.assertFalse(is_relevant(
            "Maaş", results, strong_threshold=0.70, moderate_threshold=0.63
        ))


if __name__ == "__main__":
    unittest.main()
