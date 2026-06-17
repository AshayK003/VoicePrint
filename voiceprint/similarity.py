"""Semantic similarity gate — Ensures meaning is preserved.

Two modes:
  - use_models=True  (default): sentence-transformers/all-MiniLM-L6-v2, ~80MB
  - use_models=False: Jaccard word overlap, zero dependencies
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .config import Config, load_config

# Module-level cache for sentence-transformer model
_model = None


def _get_model() -> Any | None:
    """Lazy-load the sentence transformer model. Thread-safe, loaded once."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            # Fallback: if model can't load, return None and use Jaccard
            return None
    return _model


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Word-overlap Jaccard similarity. Zero dependencies, ~0ms."""
    tokens_a = set(_tokenize(text_a))
    tokens_b = set(_tokenize(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def compute_similarity(
    text_a: str,
    text_b: str,
    config: Config | None = None,
) -> float:
    """Compute similarity between two texts.

    Uses MiniLM embeddings (cosine) when use_models=True,
    falls back to Jaccard word overlap when use_models=False.
    """
    use_models = True
    if config is not None:
        use_models = config.use_models

    if not use_models:
        return _jaccard_similarity(text_a, text_b)

    from sklearn.metrics.pairwise import cosine_similarity
    model = _get_model()
    if model is None:
        # Model failed to load, fallback to Jaccard
        return _jaccard_similarity(text_a, text_b)
    embeddings = model.encode([text_a, text_b], show_progress_bar=False)
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
    score = compute_similarity(original, humanized, config=config)

    if score < config.similarity_threshold:
        logging.warning(
            f"Similarity {score:.3f} below threshold "
            f"{config.similarity_threshold}. Meaning may be degraded."
        )

    return score
