"""Stage 1: Heuristic Scrub — Deterministic text transformations.

No model required. Pure Python regex + rule engine.
Replaces AI transition phrases, forces burstiness, injects contractions,
breaks tricolons, and reduces em-dash density.
"""

import re
from typing import Callable

# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

_rules: list[Callable[[str], str]] = []


def rule(func: Callable[[str], str]) -> Callable[[str], str]:
    """Decorator to register a scrub rule."""
    _rules.append(func)
    return func


# ---------------------------------------------------------------------------
# AI transition phrases → human equivalents
# ---------------------------------------------------------------------------

TRANSITION_MAP: dict[str, str] = {
    "Furthermore,": "Also,",
    "Moreover,": "Plus,",
    "In addition,": "On top of that,",
    "Additionally,": "And,",
    "Consequently,": "So,",
    "As a result,": "Because of that,",
    "However,": "But,",
    "Nevertheless,": "Still,",
    "Nonetheless,": "That said,",
    "On the other hand,": "Then again,",
    "In contrast,": "By comparison,",
    "Conversely,": "On the flip side,",
    "In conclusion,": "So to wrap up,",
    "To summarize,": "In short,",
    "In summary,": "Bottom line,",
    "It is important to note that": "Worth noting:",
    "It is worth noting that": "Worth noting:",
    "It should be noted that": "Keep in mind:",
    "It goes without saying that": "Obviously,",
    "Undoubtedly,": "No question,",
    "Indisputably,": "Without a doubt,",
    "Unquestionably,": "Clearly,",
    "It is essential to": "We need to",
    "It is crucial to": "We have to",
    "plays a pivotal role": "matters a lot",
    "plays a crucial role": "really matters",
    "plays a vital role": "is key to",
    "in this day and age": "nowadays",
    "at the end of the day": "ultimately",
    "leverage": "use",
    "utilize": "use",
    "facilitate": "help",
    "demonstrate": "show",
    "necessitate": "require",
    "aforementioned": "this",
    "subsequent": "next",
    "prior to": "before",
    "endeavor": "try",
    "commence": "start",
    "terminate": "end",
    "endeavors": "efforts",
    "myriad": "many",
    "plethora": "lots of",
    "tapestry": "mix",
    "delve": "dig in",
    "embark": "start",
    "landscape": "field",
    "ecosystem": "system",
    "synergy": "collaboration",
    "holistic": "complete",
    "robust": "strong",
    "comprehensive": "full",
    "cutting-edge": "newest",
    "state-of-the-art": "latest",
    "innovative": "new",
    "transformative": "big",
    "paradigm shift": "major change",
    "game-changer": "breakthrough",
    "unlock": "enable",
    "empower": "let",
    "foster": "encourage",
    "spearhead": "lead",
    "underpins": "supports",
    "underscores": "highlights",
    "signifies": "shows",
    "exemplifies": "shows",
    "epitomizes": "embodies",
    "furthermore": "also",
    "moreover": "plus",
    "additionally": "and",
    "consequently": "so",
    "nevertheless": "still",
    "nonetheless": "that said",
    "henceforth": "from now on",
    "herein": "here",
    "therein": "there",
    "whereby": "by which",
    "wherein": "in which",
}


@rule
def replace_transitions(text: str) -> str:
    """Replace AI transition phrases with human equivalents."""
    for ai, human in TRANSITION_MAP.items():
        # Case-insensitive replacement
        pattern = re.compile(re.escape(ai), re.IGNORECASE)
        text = pattern.sub(human, text)
    return text


# ---------------------------------------------------------------------------
# Tricolon breaking (X, Y, and Z → X and Y)
# ---------------------------------------------------------------------------

@rule
def break_tricolons(text: str) -> str:
    """Break the AI-favorite 'X, Y, and Z' pattern."""
    pattern = r"(\b\w+\b),\s+(\b\w+\b),\s+and\s+(\b\w+\b)"
    return re.sub(pattern, r"\1 and \2", text)


# ---------------------------------------------------------------------------
# Em-dash density reduction
# ---------------------------------------------------------------------------

@rule
def reduce_em_dashes(text: str) -> str:
    """Replace excessive em-dashes with commas or periods."""
    # Replace em-dash + surrounding spaces with comma
    text = re.sub(r"\s*—\s*", ", ", text)
    return text


# ---------------------------------------------------------------------------
# Contraction injection
# ---------------------------------------------------------------------------

CONTRACTIONS: dict[str, str] = {
    "do not": "don't",
    "does not": "doesn't",
    "did not": "didn't",
    "is not": "isn't",
    "are not": "aren't",
    "was not": "wasn't",
    "were not": "weren't",
    "has not": "hasn't",
    "have not": "haven't",
    "had not": "hadn't",
    "will not": "won't",
    "would not": "wouldn't",
    "could not": "couldn't",
    "should not": "shouldn't",
    "cannot": "can't",
    "can not": "can't",
    "it is": "it's",
    "that is": "that's",
    "there is": "there's",
    "here is": "here's",
    "what is": "what's",
    "let us": "let's",
    "I am": "I'm",
    "I have": "I've",
    "I will": "I'll",
    "I would": "I'd",
    "we are": "we're",
    "we have": "we've",
    "we will": "we'll",
    "we would": "we'd",
    "they are": "they're",
    "they have": "they've",
    "they will": "they'll",
    "they would": "they'd",
    "you are": "you're",
    "you have": "you've",
    "you will": "you'll",
    "you would": "you'd",
    "he is": "he's",
    "he has": "he's",
    "he will": "he'll",
    "he would": "he'd",
    "she is": "she's",
    "she has": "she's",
    "she will": "she'll",
    "she would": "she'd",
    "that will": "that'll",
    "that would": "that'd",
    "who is": "who's",
    "who has": "who's",
    "who will": "who'll",
    "who would": "who'd",
}


@rule
def inject_contractions(text: str) -> str:
    """Replace expanded forms with natural contractions."""
    for full, contracted in CONTRACTIONS.items():
        pattern = re.compile(re.escape(full), re.IGNORECASE)
        text = pattern.sub(contracted, text)
    return text


# ---------------------------------------------------------------------------
# Passive voice simplification (basic patterns)
# ---------------------------------------------------------------------------

PASSIVE_PATTERNS: list[tuple[str, str]] = [
    (r"It was decided that (.+)", r"We decided that \1"),
    (r"It was found that (.+)", r"We found that \1"),
    (r"It was observed that (.+)", r"We observed that \1"),
    (r"It was noted that (.+)", r"Note that \1"),
    (r"It has been shown that (.+)", r"Studies show that \1"),
    (r"It has been demonstrated that (.+)", r"Research shows that \1"),
]


@rule
def simplify_passive(text: str) -> str:
    """Convert common passive constructions to active voice."""
    for pattern, replacement in PASSIVE_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------------------------
# Redundant hedging removal
# ---------------------------------------------------------------------------

HEDGES: list[str] = [
    r"\bvery\b",
    r"\bquite\b",
    r"\brather\b",
    r"\bfairly\b",
    r"\bsomewhat\b",
    r"\bperhaps\b",
    r"\bit seems that\b",
    r"\bit appears that\b",
    r"\bin a sense\b",
    r"\bto some extent\b",
]


@rule
def remove_hedges(text: str) -> str:
    """Remove hedging language that weakens writing."""
    for hedge in HEDGES:
        text = re.sub(hedge, "", text, flags=re.IGNORECASE)
    # Clean up double spaces
    text = re.sub(r"\s{2,}", " ", text)
    return text


# ---------------------------------------------------------------------------
# Numbered list conversion (AI loves "1. 2. 3.")
# ---------------------------------------------------------------------------

@rule
def cleanup_numbered_lists(text: str) -> str:
    """Remove extra spaces after numbered list markers."""
    text = re.sub(r"(\d+)\.\s{2,}", r"\1. ", text)
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrub(text: str) -> str:
    """Apply all scrub rules to text. Returns cleaned text."""
    for rule_fn in _rules:
        text = rule_fn(text)
    return text


def get_rules() -> list[str]:
    """Return list of registered rule names (for debugging)."""
    return [fn.__name__ for fn in _rules]
