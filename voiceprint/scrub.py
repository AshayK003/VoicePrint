"""Stage 1: Heuristic Scrub — Deterministic text transformations.

No model required. Pure Python regex + rule engine.
Replaces AI transition phrases, forces burstiness, injects contractions,
breaks tricolons, and reduces em-dash density.
"""

import hashlib
import random
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
    # Expanded banned-word replacements (2025/2026 detector research)
    "multifaceted": "many-sided",
    "nuanced": "subtle",
    "nuance": "subtlety",
    "intricate": "complex",
    "intricacies": "details",
    "meticulous": "thorough",
    "bolster": "back",
    "paramount": "key",
    "groundbreaking": "big",
    "seamless": "smooth",
    "revolutionize": "change",
    "unprecedented": "unlike anything before",
    "remarkable": "amazing",
    "profound": "deep",
    "vibrant": "lively",
    "beacon": "model",
    "cornerstone": "foundation",
    "trajectory": "path",
    "spectrum": "range",
    "confluence": "mix",
    "pain points": "problems",
    "pain point": "problem",
    "pivotal": "critical",
    "bespoke": "custom",
    "hyper-personalized": "personalized",
    "omnichannel": "multi-channel",
    "actionable insights": "useful tips",
    "dot-com": "online",
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
    """Break the AI-favorite 'X, Y, and Z' pattern (single-word items)."""
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
    r"\bquite frankly\b",
    r"\bin a sense\b",
    r"\bto some extent\b",
    r"\bit seems that\b",
    r"\bit appears that\b",
    r"\bbroadly speaking\b",
    r"\bmore often than not\b",
    r"\bfor the most part\b",
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
# Filler phrase removal (AI-specific)
# ---------------------------------------------------------------------------

FILLER_PHRASES: list[tuple[str, str]] = [
    (r"\bIt is important to note that\b", ""),
    (r"\bIt is worth noting that\b", ""),
    (r"\bIt should be noted that\b", ""),
    (r"\bIt goes without saying that\b", ""),
    (r"\bIt is essential to\b", "We need to"),
    (r"\bIt is crucial to\b", "We have to"),
    (r"\bIn this day and age\b", "Nowadays"),
    (r"\bAt the end of the day\b", "Ultimately"),
    (r"\bWhen it comes to\b", "For"),
    (r"\bIn terms of\b", "For"),
    (r"\bAs a matter of fact\b", "Actually"),
    (r"\bFor all intents and purposes\b", "Basically"),
    (r"\bBy and large\b", "Mostly"),
    (r"\bOn a regular basis\b", "Regularly"),
    (r"\bDue to the fact that\b", "Because"),
    (r"\bIn light of the fact that\b", "Since"),
    (r"\bGiven the fact that\b", "Since"),
    (r"\bWith regard to\b", "About"),
    (r"\bWith respect to\b", "About"),
    (r"\bIn the event that\b", "If"),
    (r"\bFor the purpose of\b", "To"),
    (r"\bIn order to\b", "To"),
    (r"\bIs able to\b", "Can"),
    (r"\bHas the ability to\b", "Can"),
    (r"\bIs in a position to\b", "Can"),
    (r"\bIn the context of\b", "For"),
    (r"\bTaking into account\b", "Considering"),
    (r"\bOn a broader scale\b", "Broadly"),
    (r"\bFrom a holistic perspective\b", ""),
    (r"\bAt the core of this\b", "At the heart of this"),
    (r"\bIt is worth mentioning that\b", ""),
    (r"\bA wide range of\b", "Many"),
    (r"\bIn today. world\b", "Now"),
    (r"\bIt is evident that\b", "Clearly"),
    (r"\bThere is a growing need for\b", "We need"),
    (r"\bThe importance of\b", ""),
    (r"\bThe goal of this\b", "This"),
    (r"\bThe purpose of this\b", "This"),
    (r"\bAs previously mentioned\b", "As noted"),
    (r"\bAs discussed earlier\b", ""),
    (r"\bAs shown above\b", ""),
    (r"\bAs can be seen\b", ""),
]


@rule
def remove_filler_phrases(text: str) -> str:
    """Remove AI filler phrases and wordy constructions."""
    for pattern, replacement in FILLER_PHRASES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text)
    return text


# ---------------------------------------------------------------------------
# Modal verb reduction
# ---------------------------------------------------------------------------

MODALS: list[str] = [
    r"\bcould\b", r"\bwould\b", r"\bshould\b", r"\bmight\b",
    r"\bmay\b", r"\bmust\b", r"\bshall\b",
]


_reduce_modals_rng = random.Random()


@rule
def reduce_modals(text: str) -> str:
    """Reduce excessive modal verb usage (AI tends to over-hedge)."""
    _reduce_modals_rng.seed(hashlib.md5(text.encode()).hexdigest())

    for modal_pattern in MODALS:
        matches = list(re.finditer(modal_pattern, text, re.IGNORECASE))
        for m in reversed(matches):
            if _reduce_modals_rng.random() < 0.15:
                text = text[:m.start()] + text[m.end():]
    text = re.sub(r"\s{2,}", " ", text)
    return text


# ---------------------------------------------------------------------------
# Abstract subject conversion
# ---------------------------------------------------------------------------

ABSTRACT_SUBJECTS: list[tuple[str, str]] = [
    (r"\bIt is clear that\b", "Clearly"),
    (r"\bIt is evident that\b", "Clearly"),
    (r"\bIt is apparent that\b", "Clearly"),
    (r"\bIt is true that\b", "Surely"),
    (r"\bThere is no doubt that\b", "Undoubtedly"),
    (r"\bThere is a need to\b", "We need to"),
    (r"\bThere are many\b", "Many"),
    (r"\bThere are several\b", "Several"),
    (r"\bThis is a\b", "It's a"),
    (r"\bThat is a\b", "It's a"),
    (r"\bThe fact is that\b", "Actually"),
    (r"\bThe reality is that\b", "Actually"),
    (r"\bThe thing is that\b", "Actually"),
]


@rule
def convert_abstract_subjects(text: str) -> str:
    """Replace abstract/vague sentence openers with concrete ones."""
    for pattern, replacement in ABSTRACT_SUBJECTS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------------------------
# Parallel structure breaking
# ---------------------------------------------------------------------------

@rule
def break_parallelism(text: str) -> str:
    """Break AI's tendency toward symmetrical parallel structures."""
    # Break "not just X, not just Y, not just Z" → "not just X or Y or Z"
    text = re.sub(
        r"not just (\w+),\s*not just (\w+),\s*and not just (\w+)",
        r"not just \1, \2, or \3",
        text,
        flags=re.IGNORECASE,
    )

    # Break triple "it's X, it's Y, and it's Z" parallel structures
    text = re.sub(
        r"(it's \w+),\s+it's \w+,\s+and it's (\w+)",
        r"\1 and \2",
        text,
        flags=re.IGNORECASE,
    )

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
