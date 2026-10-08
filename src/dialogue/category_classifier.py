"""Trained dispute-category classifier (Phase 6.5).

Logistic Regression on bge-m3 sentence embeddings. It predicts which of
the four dispute categories a user's first message belongs to, with a
confidence score. Below the confidence threshold it returns None so the
dialogue can ask the user to clarify instead of guessing.

The embedding layer is the same ``EmbeddingManager`` used by indexing
and retrieval, so the classifier sees text exactly as retrieval does.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.config import settings
from src.dialogue.slot_schema import DisputeCategory
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_PATH = (
    Path(__file__).parent / "models" / "category_classifier.pkl"
)

EmbedFn = Callable[[List[str]], Sequence[Sequence[float]]]
Label = Union[str, DisputeCategory]


@dataclass
class CategoryPrediction:
    """A predicted category with its probability.

    Attributes:
        category: Predicted dispute category.
        confidence: Predicted class probability in [0.0, 1.0].
    """

    category: DisputeCategory
    confidence: float


class CategoryClassifier:
    """Embedding-based dispute-category classifier.

    Attributes:
        threshold: Minimum confidence required to return a category.
        model: The underlying scikit-learn LogisticRegression.
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        embed_fn: Optional[EmbedFn] = None,
        C: float = 10.0,
    ):
        """Initializes an unfitted classifier.

        Args:
            threshold: Confidence threshold. Defaults to
                ``settings.CLASSIFIER_CONFIDENCE_THRESHOLD``.
            embed_fn: Callable mapping a list of texts to embeddings.
                Defaults to the project's bge-m3 EmbeddingManager,
                loaded lazily on first use.
            C: Inverse L2 regularisation strength for the model.
        """
        self.threshold = (
            settings.CLASSIFIER_CONFIDENCE_THRESHOLD
            if threshold is None else threshold
        )
        self._embed_fn = embed_fn
        self._embedding_model_name = settings.EMBEDDING_MODEL_NAME
        self.model = LogisticRegression(C=C, max_iter=1000)
        self.is_fitted = False

    @property
    def classes(self) -> List[DisputeCategory]:
        """Categories in the column order of ``predict_proba``."""
        return [DisputeCategory(c) for c in self.model.classes_]

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embeds texts with the shared embedding layer.

        Args:
            texts: Messages to embed.

        Returns:
            Array of shape (len(texts), embedding_dim).
        """
        if self._embed_fn is None:
            # Imported lazily: loads chromadb and sentence-transformers.
            from src.indexing.vector_store import EmbeddingManager
            manager = EmbeddingManager(self._embedding_model_name)
            self._embed_fn = manager.embed_texts
        return np.asarray(self._embed_fn(list(texts)), dtype=float)

    def fit(
        self, embeddings: np.ndarray, labels: Sequence[Label]
    ) -> "CategoryClassifier":
        """Trains the model on precomputed embeddings.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim).
            labels: Category per sample (enum or its string value).

        Returns:
            self, to allow chaining.

        Raises:
            ValueError: If a label is not a valid DisputeCategory.
        """
        values = [DisputeCategory(label).value for label in labels]
        self.model.fit(np.asarray(embeddings, dtype=float), values)
        self.is_fitted = True
        return self

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Returns class probabilities for precomputed embeddings.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim).

        Returns:
            Array of shape (n_samples, n_classes); columns follow
            :attr:`classes`.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("CategoryClassifier is not fitted.")
        return self.model.predict_proba(
            np.asarray(embeddings, dtype=float))

    def predict(self, message: str) -> Optional[CategoryPrediction]:
        """Predicts the dispute category of a user message.

        Args:
            message: The user's message.

        Returns:
            A CategoryPrediction, or None if the message is empty or
            the top probability is below the threshold.
        """
        if not message or not message.strip():
            return None
        probs = self.predict_proba(self.embed([message.strip()]))[0]
        best = int(np.argmax(probs))
        confidence = float(probs[best])
        category = self.classes[best]
        if confidence < self.threshold:
            logger.info(
                "Low confidence (%.2f < %.2f) for %s; returning None.",
                confidence, self.threshold, category.value,
            )
            return None
        return CategoryPrediction(category, confidence)

    def save(self, path: Optional[Path] = None) -> Path:
        """Saves the fitted model to a .pkl file.

        Only the scikit-learn model and metadata are stored, not the
        embedding model, which is reloaded from settings.

        Args:
            path: Destination file. Defaults to DEFAULT_MODEL_PATH.

        Returns:
            The path written.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted classifier.")
        path = Path(path or DEFAULT_MODEL_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "threshold": self.threshold,
                "embedding_model": self._embedding_model_name,
            },
            path,
        )
        logger.info("Saved classifier to %s", path)
        return path

    @classmethod
    def load(
        cls,
        path: Optional[Path] = None,
        embed_fn: Optional[EmbedFn] = None,
    ) -> "CategoryClassifier":
        """Loads a classifier saved with :meth:`save`.

        Args:
            path: Model file. Defaults to DEFAULT_MODEL_PATH.
            embed_fn: Optional embedding callable (e.g. for tests).

        Returns:
            A fitted CategoryClassifier.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        path = Path(path or DEFAULT_MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"No trained classifier at {path}. Run "
                "notebooks/train_category_classifier.ipynb first."
            )
        data = joblib.load(path)
        clf = cls(threshold=data["threshold"], embed_fn=embed_fn)
        clf.model = data["model"]
        clf._embedding_model_name = data["embedding_model"]
        clf.is_fitted = True
        return clf
