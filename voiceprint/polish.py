"""Stage 4: Local Style Polish — Final text transformations.

Converts passive → active, injects rhetorical questions and sentence
fragments, normalizes burstiness. Pure Python, no model needed.
"""

import hashlib
import random
import re
from typing import Callable

from ._text import sentences as _split_sentences


# ---------------------------------------------------------------------------
# Dedicated per-function RNGs (deterministic per input text)
# ---------------------------------------------------------------------------

_rng_questions = random.Random()
_rng_fragments = random.Random()
_rng_dysfluency = random.Random()
_rng_narrative = random.Random()
_rng_vocab = random.Random()
_rng_selfdoubt = random.Random()
_rng_opinion = random.Random()
_rng_openers = random.Random()
_rng_topicshift = random.Random()


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
    (r"The study was conducted by ([^.]+)\.", r"\1 conducted the study."),
    (r"The experiment was performed by ([^.]+)\.", r"\1 performed the experiment."),
    (r"The data was analyzed by ([^.]+)\.", r"\1 analyzed the data."),
    (r"The results were obtained by ([^.]+)\.", r"\1 obtained the results."),
    (r"The model was trained by ([^.]+)\.", r"\1 trained the model."),
    (r"The paper was written by ([^.]+)\.", r"\1 wrote the paper."),
    (r"The research was carried out by ([^.]+)\.", r"\1 carried out the research."),
    (r"The survey was conducted among ([^.]+)\.", r"We surveyed \1."),
    (r"The analysis was performed on ([^.]+)\.", r"We analyzed \1."),
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
    """Occasionally insert a rhetorical question, with variable spacing."""
    _rng_questions.seed(hashlib.md5(text.encode()).hexdigest())
    sents = _split_sentences(text)
    if len(sents) < 8:
        return text

    result = []
    # Track how many sentences since last question — avoid any fixed pattern
    since_last = 99
    for i, sent in enumerate(sents):
        result.append(sent)
        since_last += 1
        if since_last > 3 and _rng_questions.random() < 0.08:
            if i < len(sents) - 2:
                q = _rng_questions.choice(RHETORICAL_QUESTIONS)
                result.append(q)
                since_last = 0

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
    _rng_fragments.seed(hashlib.md5(text.encode()).hexdigest())
    sents = _split_sentences(text)
    if len(sents) < 8:
        return text

    result = []
    for i, sent in enumerate(sents):
        result.append(sent)
        # Inject fragment before a long sentence
        if i < len(sents) - 1 and len(sents[i + 1].split()) > 20:
            if _rng_fragments.random() < 0.15:  # 15% chance
                fragment = _rng_fragments.choice(FRAGMENTS)
                result.append(fragment)

    return " ".join(result)


# ---------------------------------------------------------------------------
# Dysfluency injection — natural speech disfluencies (um, well, I mean)
# Research shows these significantly reduce detection scores
# ---------------------------------------------------------------------------

DYSFLUENCIES_MID_SENTENCE = [
    "well,",
    "I mean,",
    "you know,",
    "actually,",
    "honestly,",
    "basically,",
    "the thing is,",
    "like,",
]

DYSFLUENCIES_SELF_CORRECT = [
    " — actually, that's not quite right —",
    " — or rather,",
    " — well, maybe not —",
    " — let me rephrase that —",
    " — I should say —",
]


@rule
def inject_dysfluencies(text: str) -> str:
    """Insert natural disfluencies and self-corrections.

    Adds mid-sentence fillers (well, I mean, you know) at sentence-initial
    positions and occasional self-corrections in longer sentences.
    """
    _rng_dysfluency.seed(hashlib.md5(text.encode()).hexdigest())
    sents = _split_sentences(text)
    if len(sents) < 4:
        return text

    result = []
    for i, sent in enumerate(sents):
        words = sent.split()
        # Skip very short sentences
        if len(words) < 5:
            result.append(sent)
            continue

        # ~5% chance to add a mid-sentence dysfluency at sentence start
        if _rng_dysfluency.random() < 0.05 and i > 0:
            filler = _rng_dysfluency.choice(DYSFLUENCIES_MID_SENTENCE)
            first_word = sent.split()[0]
            if first_word in ("I", "I'm", "I'll", "I've", "I'd"):
                sent = filler[0].upper() + filler[1:] + " " + sent
            else:
                sent = filler[0].upper() + filler[1:] + " " + sent[0].lower() + sent[1:]

        # ~3% chance to insert a self-correction in longer sentences
        if len(words) > 15 and _rng_dysfluency.random() < 0.03:
            insert_pos = len(words) // 2
            correction = _rng_dysfluency.choice(DYSFLUENCIES_SELF_CORRECT)
            words.insert(insert_pos, correction)
            sent = " ".join(words)

        result.append(sent)

    return " ".join(result)


# ---------------------------------------------------------------------------
# Vocabulary variety — swap common "safe" words for rarer alternatives
# Creates perplexity "spikes" that detectors interpret as human-like.
# Humans occasionally use unusual words; AI avoids them.
# ---------------------------------------------------------------------------

VOCAB_SWAPS: list[tuple[str, str]] = [
    (r"\bbig\b", "massive"),
    (r"\bgood\b", "decent"),
    (r"\bimportant\b", "crucial"),
    (r"\bdifficult\b", "tricky"),
    (r"\bend\b", "wrap up"),
    (r"\bclear\b", "obvious"),
    (r"\bvery\b", "pretty"),
    (r"\bshow\b", "prove"),
]


@rule
def inject_vocabulary_variety(text: str) -> str:
    """Replace common words with rarer alternatives at low probability.

    Creates sporadic perplexity spikes — the kind of unexpected word
    choices that naturally occur in human writing but AI avoids.
    """
    _rng_vocab.seed(hashlib.md5(text.encode()).hexdigest())
    sents = _split_sentences(text)
    if len(sents) < 3:
        return text

    result = []
    for i, sent in enumerate(sents):
        words = sent.split()
        if len(words) < 6:
            result.append(sent)
            continue

        # ~4% chance per applicable sentence to swap one word
        if _rng_vocab.random() < 0.04:
            candidates = [(pattern, repl) for pattern, repl in VOCAB_SWAPS
                          if re.search(pattern, sent, re.IGNORECASE)]
            if candidates:
                pattern, repl = _rng_vocab.choice(candidates)
                if _rng_vocab.random() < 0.5:  # 50% swap rate when triggered
                    sent = re.sub(pattern, repl, sent, count=1, flags=re.IGNORECASE)

        result.append(sent)

    return " ".join(result)


# ---------------------------------------------------------------------------
# Formal-to-casual tone conversion — replaces stiff language with natural alternatives
# ---------------------------------------------------------------------------

FORMAL_TO_CASUAL: list[tuple[str, str]] = [
    (r"\bHowever,\b", "But"),
    (r"\bHowever\b", "though"),
    (r"\bTherefore,\b", "So"),
    (r"\bThus,\b", "So"),
    (r"\bHence,\b", "So"),
    (r"\bConsequently,\b", "Because of that,"),
    (r"\bIn order to\b", "To"),
    (r"\bIn the event that\b", "If"),
    (r"\bRegarding\b", "About"),
    (r"\bWith regard to\b", "About"),
    (r"\bIn regard to\b", "About"),
    (r"\bDue to the fact that\b", "Because"),
    (r"\bOn the other hand,\b", "But"),
    (r"\bIn contrast,\b", "Meanwhile,"),
    (r"\bNevertheless,\b", "Still,"),
    (r"\bNonetheless,\b", "Even so,"),
    (r"\bAs a result,\b", "So"),
    (r"\bIn addition,\b", "Plus,"),
    (r"\bAdditionally,\b", "And"),
    (r"\bMoreover,\b", "Plus,"),
    (r"\bFurthermore,\b", "Also,"),
    (r"\bIt should be noted that\b", ""),
    (r"\bIt is worth noting that\b", ""),
    (r"\bIt is important to note that\b", ""),
    (r"\bIn conclusion,\b", "So"),
    (r"\bTo summarize,\b", "Basically,"),
    (r"\bIn summary,\b", "Long story short,"),
    (r"\bsubsequently\b", "later"),
    (r"\bprior to\b", "before"),
    (r"\bapproximately\b", "about"),
    (r"\bsufficient\b", "enough"),
    (r"\bnecessitate\b", "need"),
    (r"\bendeavor\b", "try"),
    (r"\bcommence\b", "start"),
    (r"\bterminate\b", "end"),
    (r"\brequest\b", "ask"),
    (r"\bassist\b", "help"),
    (r"\brequire\b", "need"),
    (r"\bprovide\b", "give"),
    (r"\bobtain\b", "get"),
    (r"\bdetermine\b", "figure out"),
    (r"\bconstruct\b", "build"),
    (r"\bdevelop\b", "create"),
    (r"\bestablish\b", "set up"),
    (r"\bidentify\b", "find"),
    (r"\bnotify\b", "tell"),
    (r"\bpossess\b", "have"),
    (r"\bretain\b", "keep"),
    (r"\bcontrary to\b", "unlike"),
    (r"\bconstitute\b", "make up"),
    (r"\bdemonstrate\b", "show"),
    (r"\bencounter\b", "come across"),
    (r"\bgenerate\b", "create"),
    (r"\bimplement\b", "put in place"),
    (r"\binitiate\b", "start"),
    (r"\bmaintain\b", "keep"),
    (r"\bparticipate\b", "take part"),
    (r"\bperform\b", "do"),
    (r"\bproceed\b", "go ahead"),
    (r"\bproduce\b", "make"),
    (r"\bremain\b", "stay"),
    (r"\brender\b", "make"),
    (r"\brequire\b", "need"),
    (r"\btransmit\b", "send"),
    (r"\butilize\b", "use"),
]


@rule
def convert_formal_to_casual(text: str) -> str:
    """Replace formal/literary words with casual conversational alternatives."""
    for pattern, replacement in FORMAL_TO_CASUAL:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text)
    return text


# ---------------------------------------------------------------------------
# Personal narrative injection — adds first-person framing
# Research shows personal voice is a strong human signal
# ---------------------------------------------------------------------------

PERSONAL_FRAMINGS = [
    "In my experience, ",
    "I've found that ",
    "I think ",
    "What I've noticed is ",
    "From what I can tell, ",
    "I remember when ",
    "Personally, I ",
    "If you ask me, ",
]

PERSONAL_ASIDES = [
    " — at least that's how I see it.",
    " — or at least that's what I've noticed.",
    " — I could be wrong, but that's my take.",
    " — in my opinion, anyway.",
]


@rule
def inject_personal_narrative(text: str) -> str:
    """Add first-person framing and personal asides.

    Wraps opening with personal experience framing and adds
    opinion asides to create a more authentic human voice.
    """
    _rng_narrative.seed(hashlib.md5(text.encode()).hexdigest())
    sents = _split_sentences(text)
    if len(sents) < 3:
        return text

    result = []
    for i, sent in enumerate(sents):
        words = sent.split()
        if len(words) < 6:
            result.append(sent)
            continue

        # ~5% chance to frame a sentence with personal experience
        if _rng_narrative.random() < 0.05:
            framing = _rng_narrative.choice(PERSONAL_FRAMINGS)
            first_word = sent.split()[0]
            if first_word in ("I", "I'm", "I'll", "I've", "I'd"):
                sent = framing + sent
            else:
                sent = framing + sent[0].lower() + sent[1:]

        # ~4% chance to add a personal aside at end of longer sentences
        if len(words) > 12 and _rng_narrative.random() < 0.04:
            aside = _rng_narrative.choice(PERSONAL_ASIDES)
            sent = sent.rstrip(".!?") + aside

        result.append(sent)

    return " ".join(result)


# ---------------------------------------------------------------------------
# Duplicate sentence removal
# ---------------------------------------------------------------------------

@rule
def remove_duplicates(text: str) -> str:
    """Remove exact duplicate sentences."""
    sents = _split_sentences(text)
    seen = set()
    result = []
    for sent in sents:
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
# Smoothing — fixes common artifacts from previous rules
# ---------------------------------------------------------------------------

@rule
def smooth_text(text: str) -> str:
    """Fix common artifacts from transformation rules.

    - Standalone "i" becomes "I"
    - Consecutive duplicate leading conjunctions ("But but", "So so")
    - Redundant double transitions ("But however" → "But")
    - Punctuation greed (".?." → "?", "?!?..." → "?")
    """
    # Fix lowercase "i" standing alone
    text = re.sub(r"(?<!\w)i(?!\w)", "I", text)

    # Fix duplicate leading words ("But but" → "But", "So so" → "So")
    text = re.sub(r"\b(But|So|And|Well|However) \1\b", r"\1", text, flags=re.IGNORECASE)

    # Fix redundant "But however", "So therefore", "And also"
    text = re.sub(r"\bBut however\b", "But", text, flags=re.IGNORECASE)
    text = re.sub(r"\bSo therefore\b", "So", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAnd also\b", "And", text, flags=re.IGNORECASE)

    # Remove period after ? or !  ("Right?." → "Right?")
    text = re.sub(r"[?!]\.", lambda m: m.group()[0], text)

    return text


# ---------------------------------------------------------------------------
# Self-doubt injection — natural uncertainty signals
# AI avoids uncertainty; humans express doubt naturally
# ---------------------------------------------------------------------------

SELF_DOUBT_FRAMINGS = [
    "I could be wrong, but ",
    "I'm not entirely sure, but ",
    "If I remember right, ",
    "I think ",
    "Honestly, I'm not 100% sure on this, but ",
    "From what I recall, ",
    "I might be mixing this up, but ",
]

SELF_DOUBT_ASIDES = [
    " — at least that's how I understand it.",
    " — but don't quote me on that.",
    " — I think?",
    " — or something like that.",
    " — if I'm not mistaken.",
]


@rule
def inject_self_doubt(text: str) -> str:
    _rng_selfdoubt.seed(hashlib.md5(text.encode()).hexdigest())
    sents = _split_sentences(text)
    if len(sents) < 6:
        return text

    result = []
    for i, sent in enumerate(sents):
        words = sent.split()
        if len(words) < 5:
            result.append(sent)
            continue

        if _rng_selfdoubt.random() < 0.03:
            framing = _rng_selfdoubt.choice(SELF_DOUBT_FRAMINGS)
            first_word = sent.split()[0]
            if first_word in ("I", "I'm", "I'll", "I've", "I'd"):
                sent = framing + sent
            else:
                sent = framing + sent[0].lower() + sent[1:]

        if len(words) > 15 and _rng_selfdoubt.random() < 0.02:
            aside = _rng_selfdoubt.choice(SELF_DOUBT_ASIDES)
            sent = sent.rstrip(".!?") + aside

        result.append(sent)

    return " ".join(result)


# ---------------------------------------------------------------------------
# Opinion injection — take a stance where natural
# AI tends to stay neutral; humans express judgment
# ---------------------------------------------------------------------------

OPINION_MARKERS = [
    "The thing is, ",
    "Here's what I really think: ",
    "Honestly? ",
    "If you ask me, ",
    "What's interesting is ",
    "The key thing to understand is ",
    "What most people miss is ",
]

OPINION_FRAMINGS = [
    ", and that's a good thing.",
    ", which honestly makes all the difference.",
    " — and that's not a bad thing at all.",
    ", and frankly that matters a lot.",
    ", which is exactly what you'd hope for.",
]


@rule
def inject_opinion_framing(text: str) -> str:
    _rng_opinion.seed(hashlib.md5(text.encode()).hexdigest())
    sents = _split_sentences(text)
    if len(sents) < 5:
        return text

    result = []
    for i, sent in enumerate(sents):
        words = sent.split()
        if len(words) < 8:
            result.append(sent)
            continue

        if _rng_opinion.random() < 0.04:
            marker = _rng_opinion.choice(OPINION_MARKERS)
            first_word = sent.split()[0]
            if first_word in ("I", "I'm", "I'll", "I've", "I'd"):
                sent = marker + sent
            else:
                sent = marker + sent[0].lower() + sent[1:]

        result.append(sent)

    return " ".join(result)


# ---------------------------------------------------------------------------
# Varied sentence openers — break GPTZero's sentence-start pattern detection
# ---------------------------------------------------------------------------

ALTERNATIVE_OPENERS = [
    "Actually, ",
    "Honestly, ",
    "The truth is, ",
    "What's funny is, ",
    "Here's the deal: ",
    "Look, ",
    "Sure, ",
    "Of course, ",
    "At the end of the day, ",
    "When you get right down to it, ",
]

OPENERS_EXCLUDE = {"But", "So", "And", "Or", "Well", "Actually", "Honestly", "Look", "Sure"}


@rule
def vary_sentence_openers(text: str) -> str:
    _rng_openers.seed(hashlib.md5(text.encode()).hexdigest())
    sents = _split_sentences(text)
    if len(sents) < 5:
        return text

    result = []
    for i, sent in enumerate(sents):
        result.append(sent)
        if i >= len(sents) - 1:
            break
        next_sent = sents[i + 1]
        first_word = next_sent.split()[0] if next_sent.split() else ""
        stripped = first_word.strip("'\"")

        if stripped in OPENERS_EXCLUDE:
            continue

        if _rng_openers.random() < 0.03 and len(next_sent.split()) > 6:
            opener = _rng_openers.choice(ALTERNATIVE_OPENERS)
            sents[i + 1] = opener + next_sent[0].lower() + next_sent[1:]

    return " ".join(result)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def polish(text: str) -> str:
    """Apply all polish rules to text."""
    for rule_fn in _rules:
        text = rule_fn(text)
    return text
