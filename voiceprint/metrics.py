"""Text metrics — Burstiness, perplexity, and readability scoring."""

from __future__ import annotations

import re

import numpy as np

from ._text import sentences as _split_sentences
import textstat


def sentence_lengths(text: str) -> list[int]:
    """Split text into sentences and return word counts."""
    sents = _split_sentences(text.strip())
    return [len(s.split()) for s in sents if s.strip()]


def burstiness(text: str) -> float:
    """Calculate burstiness (sentence length variance).

    Human text: 0.4–0.7
    AI text: < 0.3 (uniform)
    """
    lengths = sentence_lengths(text)
    if len(lengths) < 2:
        return 0.0
    mean_len = np.mean(lengths)
    std_len = np.std(lengths)
    return float(std_len / mean_len) if mean_len > 0 else 0.0


def burstiness_report(text: str) -> dict:
    """Detailed burstiness analysis."""
    lengths = sentence_lengths(text)
    if not lengths:
        return {"burstiness": 0.0, "mean_length": 0, "std": 0, "min": 0, "max": 0}

    mean_len = float(np.mean(lengths))
    std_len = float(np.std(lengths))
    b = std_len / mean_len if mean_len > 0 else 0.0

    return {
        "burstiness": round(b, 3),
        "mean_length": round(mean_len, 1),
        "std": round(std_len, 1),
        "min": min(lengths),
        "max": max(lengths),
        "sentence_count": len(lengths),
    }


def readability_scores(text: str) -> dict:
    """Calculate readability metrics."""
    return {
        "flesch_reading_ease": round(textstat.flesch_reading_ease(text), 1),
        "flesch_kincaid_grade": round(textstat.flesch_kincaid_grade(text), 1),
        "gunning_fog": round(textstat.gunning_fog(text), 1),
        "text_standard": textstat.text_standard(text, float_output=False),
    }



