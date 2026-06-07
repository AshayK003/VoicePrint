"""Semantic similarity gate — Ensures meaning is preserved.

Uses sentence-transformers/all-MiniLM-L6-v2 for fast CPU inference.
Threshold: cosine similarity ≥ 0.78 (from paniccow/humanizer research).
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .config import Config, load_config

# Module-level cache (loaded once)
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the sentence transformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def compute_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts."""
    model = _get_model()
    embeddings = model.encode([text_a, text_b])
    return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])


def check_similarity(
    original: str,
    humanized: str,
    config: Config | None = None,
) -> float:
    """Check if humanized text is similar enough to original.

    Returns the similarity score. Use config.similarity_threshold to gate.
    """
    config = config or load_config()
    score = compute_similarity(original, humanized)

    if score < config.similarity_threshold:
        import logging
        logging.warning(
            f"Similarity {score:.3f} below threshold "
            f"{config.similarity_threshold}. Meaning may be degraded."
        )

    return score
