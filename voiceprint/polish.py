"""Stage 4: Local Style Polish — Final text transformations.

Converts passive → active, injects rhetorical questions and sentence
fragments, normalizes burstiness. Pure Python, no model needed.
"""

import random
import re
from typing import Callable


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

_rules: list[Callable[[str], str]] = []


def rule(func: Callable[[str], str]) -> Callable[[str], str]:
    """Decorator to register a polish rule."""
    _rules.append(func)
    return func


# ---------------------------------------------------------------------------
# Passive → active voice (additional patterns beyond scrub.py)
# ---------------------------------------------------------------------------

ACTIVE_CONVERSIONS: list[tuple[str, str]] = [
    (r"The study was conducted by (.+)", r"\1 conducted the study"),
    (r"The experiment was performed by (.+)", r"\1 performed the experiment"),
    (r"The data was analyzed by (.+)", r"\1 analyzed the data"),
    (r"The results were obtained by (.+)", r"\1 obtained the results"),
    (r"The model was trained by (.+)", r"\1 trained the model"),
    (r"The paper was written by (.+)", r"\1 wrote the paper"),
    (r"The research was carried out by (.+)", r"\1 carried out the research"),
    (r"The survey was conducted among (.+)", r"We surveyed \1"),
    (r"The analysis was performed on (.+)", r"We analyzed \1"),
]


@rule
def convert_passive_to_active(text: str) -> str:
    """Convert common passive constructions to active voice."""
    for pattern, replacement in ACTIVE_CONVERSIONS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------------------------
# Rhetorical question injection
# ---------------------------------------------------------------------------
RHETORICAL_QUESTIONS: list[str] = [
    "But does it really work that way?",
    "So what does this mean in practice?",
    "Why does this matter?",
    "And here's the real question — is that enough?",
    "The thing is, who actually benefits?",
    "Right, but what's the catch?",
    "Sounds good on paper. But in reality?",
]


@rule
def inject_rhetorical_questions(text: str) -> str:
    """Insert a rhetorical question after every 4th sentence."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) < 6:
        return text

    result = []
    for i, sent in enumerate(sentences):
        result.append(sent)
        if (i + 1) % 4 == 0 and i < len(sentences) - 1:
            q = random.choice(RHETORICAL_QUESTIONS)
            result.append(q)

    return " ".join(result)


# ---------------------------------------------------------------------------
# Sentence fragment injection
# ---------------------------------------------------------------------------

FRAGMENTS: list[str] = [
    "Not bad.",
    "Here's why.",
    "That's the key.",
    "Makes sense, right?",
    "And that changes everything.",
    "Or does it?",
    "That's the real issue.",
    "Simple enough.",
    "But wait.",
    "Here's the thing.",
]


@rule
def inject_fragments(text: str) -> str:
    """Add short sentence fragments for natural rhythm."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) < 8:
        return text

    result = []
    for i, sent in enumerate(sentences):
        result.append(sent)
        # Inject fragment before a long sentence
        if i < len(sentences) - 1 and len(sentences[i + 1].split()) > 20:
            if random.random() < 0.4:  # 40% chance
                fragment = random.choice(FRAGMENTS)
                result.append(fragment)

    return " ".join(result)


# ---------------------------------------------------------------------------
# Duplicate sentence removal
# ---------------------------------------------------------------------------

@rule
def remove_duplicates(text: str) -> str:
    """Remove exact duplicate sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    seen = set()
    result = []
    for sent in sentences:
        normalized = sent.strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            result.append(sent)
    return " ".join(result)


# ---------------------------------------------------------------------------
# Final punctuation normalization
# ---------------------------------------------------------------------------

@rule
def normalize_punctuation(text: str) -> str:
    """Fix double spaces, trailing punctuation issues."""
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"!{2,}", "!", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def polish(text: str) -> str:
    """Apply all polish rules to text."""
    for rule_fn in _rules:
        text = rule_fn(text)
    return text
