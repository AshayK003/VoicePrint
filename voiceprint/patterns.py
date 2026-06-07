"""AI-pattern fingerprint signals.

13 statistical signals that detect AI-generated text.
Used for pre-analysis and post-validation.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np


# AI-favorite words (from paniccow/humanizer research)
AI_FAVORITE_WORDS = frozenset({
    "delve", "tapestry", "landscape", "ecosystem", "synergy",
    "holistic", "robust", "comprehensive", "cutting-edge",
    "state-of-the-art", "innovative", "transformative", "paradigm",
    "unlock", "empower", "foster", "spearhead", "underpins",
    "underscores", "signifies", "exemplifies", "epitomizes",
    "furthermore", "moreover", "additionally", "consequently",
    "nevertheless", "nonetheless", "leverage", "utilize", "facilitate",
    "endeavor", "commence", "terminate", "myriad", "plethora",
    "aforementioned", "subsequent", "henceforth", "herein", "therein",
})

# Transition words detectors look for
TRANSITION_WORDS = frozenset({
    "furthermore", "moreover", "additionally", "consequently",
    "nevertheless", "nonetheless", "however", "therefore",
    "hence", "thus", "accordingly", "subsequently",
})


def signal_ai_vocabulary(text: str) -> float:
    """Fraction of words that are AI-favorite vocabulary."""
    words = text.lower().split()
    if not words:
        return 0.0
    ai_count = sum(1 for w in words if w.strip(".,;:!?") in AI_FAVORITE_WORDS)
    return ai_count / len(words)


def signal_transition_density(text: str) -> float:
    """Fraction of sentences starting with transition words."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return 0.0
    starts_with_transition = sum(
        1 for s in sentences
        if s.strip().split()[0].lower().strip(".,;:!?") in TRANSITION_WORDS
    )
    return starts_with_transition / len(sentences)


def signal_sentence_start_uniformity(text: str) -> float:
    """How uniform are sentence starters. Higher = more uniform = more AI."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) < 3:
        return 0.0
    starters = [s.strip().split()[0].lower()[:4] for s in sentences if s.strip()]
    counts = Counter(starters)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(starters)


def signal_tricolons(text: str) -> float:
    """Density of X, Y, and Z patterns."""
    pattern = r"\b\w+,\s+\w+,\s+and\s+\w+\b"
    matches = re.findall(pattern, text)
    words = text.split()
    if not words:
        return 0.0
    return len(matches) / (len(words) / 100)  # Per 100 words


def signal_em_dash_density(text: str) -> float:
    """Em-dash usage density."""
    em_dashes = text.count("—") + text.count("–")
    words = text.split()
    if not words:
        return 0.0
    return em_dashes / (len(words) / 100)


def signal_hedging(text: str) -> float:
    """Hedging language density."""
    hedges = [
        "it is important to note", "it should be noted",
        "it goes without saying", "in this day and age",
        "at the end of the day", "it is crucial to",
        "it is essential to", "plays a pivotal role",
    ]
    text_lower = text.lower()
    count = sum(1 for h in hedges if h in text_lower)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return 0.0
    return count / len(sentences)


def signal_contraction_deficit(text: str) -> float:
    """Score 0–1 where 1 means no contractions (AI-like)."""
    contractions = re.findall(r"\b\w+'t\b|\b\w+'re\b|\b\w+'ve\b|\b\w+'ll\b|\b\w+'s\b|\bI'm\b", text)
    words = text.split()
    if not words:
        return 1.0
    return 1.0 - min(1.0, len(contractions) / (len(words) / 20))


def signal_ngram_repetition(text: str) -> float:
    """Fraction of repeated bigrams."""
    words = text.lower().split()
    if len(words) < 4:
        return 0.0
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    counts = Counter(bigrams)
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / len(bigrams)


def signal_type_token_ratio(text: str) -> float:
    """Lexical diversity: unique words / total words."""
    words = [w.lower().strip(".,;:!?") for w in text.split()]
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def signal_passive_voice(text: str) -> float:
    """Approximate passive voice fraction."""
    passive_patterns = [
        r"\b(is|was|were|are|been|being)\s+\w+ed\b",
        r"\b(is|was|were|are|been|being)\s+\w+en\b",
    ]
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return 0.0
    passive_count = sum(
        1 for s in sentences
        if any(re.search(p, s, re.IGNORECASE) for p in passive_patterns)
    )
    return passive_count / len(sentences)


def signal_abstract_subjects(text: str) -> float:
    """Fraction of sentences starting with abstract subjects."""
    abstract_starts = [
        "it is", "there are", "there is", "this is", "that is",
        "the fact", "the concept", "the idea", "the notion",
    ]
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return 0.0
    count = sum(
        1 for s in sentences
        if any(s.lower().startswith(a) for a in abstract_starts)
    )
    return count / len(sentences)


def signal_modality_overload(text: str) -> float:
    """Excessive use of modal verbs (could, would, should, might)."""
    modals = ["could", "would", "should", "might", "may", "must", "shall"]
    words = text.lower().split()
    if not words:
        return 0.0
    modal_count = sum(1 for w in words if w in modals)
    return modal_count / len(words)


def compute_all_signals(text: str) -> dict[str, float]:
    """Compute all 13 AI-pattern signals."""
    return {
        "ai_vocabulary": round(signal_ai_vocabulary(text), 4),
        "transition_density": round(signal_transition_density(text), 4),
        "sentence_start_uniformity": round(signal_sentence_start_uniformity(text), 4),
        "tricolons": round(signal_tricolons(text), 4),
        "em_dash_density": round(signal_em_dash_density(text), 4),
        "hedging": round(signal_hedging(text), 4),
        "contraction_deficit": round(signal_contraction_deficit(text), 4),
        "ngram_repetition": round(signal_ngram_repetition(text), 4),
        "type_token_ratio": round(signal_type_token_ratio(text), 4),
        "passive_voice": round(signal_passive_voice(text), 4),
        "abstract_subjects": round(signal_abstract_subjects(text), 4),
        "modality_overload": round(signal_modality_overload(text), 4),
    }


def pattern_score(text: str) -> float:
    """Aggregate pattern score (0 = very human, 1 = very AI).

    Based on paniccow/humanizer's 13-signal aggregate.
    """
    signals = compute_all_signals(text)
    values = list(signals.values())
    return round(float(np.mean(values)), 4)
