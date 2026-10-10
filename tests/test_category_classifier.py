"""Offline tests for the Phase 6.5 classifier.

A deterministic hashed character n-gram embedder stands in for bge-m3,
so these tests verify the code (fit, threshold, save/load, data
format) without downloading models. They say nothing about the real
accuracy; that is measured in the training notebook.
"""

import csv
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from sklearn.model_selection import train_test_split

from src.dialogue.category_classifier import (
    CategoryClassifier,
    CategoryPrediction,
)
from src.dialogue.slot_schema import DisputeCategory

DATA_PATH = Path("data/training_data/dispute_categories.csv")
DIM = 512


def fake_embed(texts):
    """Hashes character trigrams into L2-normalised vectors."""
    out = np.zeros((len(texts), DIM))
    for i, text in enumerate(texts):
        text = f"  {text.lower()}  "
        for j in range(len(text) - 2):
            out[i, zlib.crc32(text[j:j + 3].encode()) % DIM] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-9)


def load_rows():
    """Reads (message, category) rows from the training CSV."""
    with open(DATA_PATH, encoding="utf-8", newline="") as f:
        return [(r["message"], r["category"]) for r in csv.DictReader(f)]


class TestTrainingData(unittest.TestCase):
    """Validates the labelled dataset."""

    def test_format_and_balance(self):
        rows = load_rows()
        self.assertGreaterEqual(len(rows), 40)
        valid = {c.value for c in DisputeCategory}
        counts = {}
        for message, category in rows:
            self.assertTrue(message.strip())
            self.assertIn(category, valid)
            counts[category] = counts.get(category, 0) + 1
        self.assertEqual(set(counts), valid)
        self.assertGreaterEqual(min(counts.values()), 10)
        self.assertEqual(len({m for m, _ in rows}), len(rows))


class TestCategoryClassifier(unittest.TestCase):
    """Checks classifier behaviour with a stand-in embedder."""

    @classmethod
    def setUpClass(cls):
        rows = load_rows()
        messages = [m for m, _ in rows]
        labels = [c for _, c in rows]
        cls.X = fake_embed(messages)
        cls.messages, cls.labels = messages, labels

    def make(self, threshold=0.0):
        clf = CategoryClassifier(threshold=threshold, embed_fn=fake_embed)
        return clf.fit(self.X, self.labels)

    def test_pipeline_smoke_accuracy(self):
        idx = np.arange(len(self.labels))
        train, test = train_test_split(
            idx, test_size=0.2, stratify=self.labels, random_state=42)
        clf = CategoryClassifier(embed_fn=fake_embed)
        clf.fit(self.X[train], [self.labels[i] for i in train])
        probs = clf.predict_proba(self.X[test])
        pred = [clf.classes[k].value for k in probs.argmax(axis=1)]
        acc = np.mean([p == self.labels[i] for p, i in zip(pred, test)])
        self.assertGreater(acc, 0.5)

    def test_predict_returns_prediction(self):
        pred = self.make().predict(self.messages[0])
        self.assertIsInstance(pred, CategoryPrediction)
        self.assertIsInstance(pred.category, DisputeCategory)
        self.assertTrue(0.0 <= pred.confidence <= 1.0)

    def test_low_confidence_returns_none(self):
        clf = self.make(threshold=1.01)
        self.assertIsNone(clf.predict(self.messages[0]))

    def test_empty_message_returns_none(self):
        self.assertIsNone(self.make().predict("   "))

    def test_unfitted_predict_does_not_embed(self):
        embed_fn = Mock(side_effect=AssertionError("Embedding was called"))
        clf = CategoryClassifier(embed_fn=embed_fn)
        with self.assertRaisesRegex(RuntimeError, "is not fitted"):
            clf.predict("sual")
        embed_fn.assert_not_called()

    def test_unfitted_empty_message_returns_none(self):
        embed_fn = Mock(side_effect=AssertionError("Embedding was called"))
        clf = CategoryClassifier(embed_fn=embed_fn)
        for message in ["", "   ", "\t\n"]:
            with self.subTest(message=message):
                self.assertIsNone(clf.predict(message))
        embed_fn.assert_not_called()

    def test_save_and_load_roundtrip(self):
        clf = self.make(threshold=0.3)
        with tempfile.TemporaryDirectory() as tmp:
            path = clf.save(Path(tmp) / "nested" / "clf.pkl")
            loaded = CategoryClassifier.load(path, embed_fn=fake_embed)
        self.assertEqual(loaded.threshold, 0.3)
        np.testing.assert_allclose(
            clf.predict_proba(self.X[:5]),
            loaded.predict_proba(self.X[:5]),
        )

    def test_unfitted_and_missing_file_errors(self):
        clf = CategoryClassifier(embed_fn=fake_embed)
        with self.assertRaises(RuntimeError):
            clf.predict("sual")
        with self.assertRaises(RuntimeError):
            clf.save(Path("unused.pkl"))
        with self.assertRaises(FileNotFoundError):
            CategoryClassifier.load(Path("does_not_exist.pkl"))

    def test_invalid_label_rejected(self):
        clf = CategoryClassifier(embed_fn=fake_embed)
        with self.assertRaises(ValueError):
            clf.fit(self.X[:4], ["A", "B", "C", "D"])


if __name__ == "__main__":
    unittest.main()
