from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticMatcher:
    """Semantic similarity with Sentence Transformers and a TF-IDF fallback."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self.backend = "tfidf"
        self.error: str | None = None
        self._load_backend()

    def _load_backend(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.backend = "sentence-transformers"
        except Exception as exc:
            self._model = None
            self.backend = "tfidf"
            self.error = str(exc)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Return a score from 0 to 100."""
        if not text_a.strip() or not text_b.strip():
            return 0.0

        if self._model is not None:
            embeddings = self._model.encode(
                [text_a, text_b],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            score = float(np.dot(embeddings[0], embeddings[1]))
        else:
            vectorizer = TfidfVectorizer(stop_words="english")
            matrix = vectorizer.fit_transform([text_a, text_b])
            score = float(cosine_similarity(matrix[0], matrix[1])[0][0])

        # Cosine can technically be negative for some embeddings.
        return max(0.0, min(100.0, score * 100.0))

    def rank_chunks(
        self,
        query: str,
        chunks: list[str],
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        if not query.strip() or not chunks:
            return []

        top_k = min(top_k, len(chunks))

        if self._model is not None:
            vectors = self._model.encode(
                [query, *chunks],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            query_vec = vectors[0]
            chunk_vecs = vectors[1:]
            scores = chunk_vecs @ query_vec
        else:
            vectorizer = TfidfVectorizer(stop_words="english")
            matrix = vectorizer.fit_transform([query, *chunks])
            scores = cosine_similarity(matrix[0], matrix[1:]).flatten()

        order = np.argsort(scores)[::-1][:top_k]
        return [(chunks[i], float(scores[i])) for i in order]
