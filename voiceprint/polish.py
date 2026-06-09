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
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) < 4:
        return text

    result = []
    for i, sent in enumerate(sentences):
        words = sent.split()
        # Skip very short sentences
        if len(words) < 5:
            result.append(sent)
            continue

        # ~15% chance to add a mid-sentence dysfluency at sentence start
        if random.random() < 0.15 and i > 0:
            filler = random.choice(DYSFLUENCIES_MID_SENTENCE)
            sent = filler[0].upper() + filler[1:] + " " + sent[0].lower() + sent[1:]

        # ~8% chance to insert a self-correction in longer sentences
        if len(words) > 15 and random.random() < 0.08:
            insert_pos = len(words) // 2
            correction = random.choice(DYSFLUENCIES_SELF_CORRECT)
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
    (r"\bbad\b", "lousy"),
    (r"\bimportant\b", "crucial"),
    (r"\bdifficult\b", "tricky"),
    (r"\beasy\b", "painless"),
    (r"\binteresting\b", "fascinating"),
    (r"\bchange\b", "shift"),
    (r"\bhelp\b", "lend a hand"),
    (r"\bstart\b", "kick off"),
    (r"\bend\b", "wrap up"),
    (r"\bwork\b", "pull off"),
    (r"\bproblem\b", "headache"),
    (r"\bsolution\b", "fix"),
    (r"\bclear\b", "obvious"),
    (r"\bquickly\b", "in a snap"),
    (r"\bvery\b", "pretty"),
    (r"\breally\b", "honestly"),
    (r"\bknow\b", "get"),
    (r"\bunderstand\b", "grasp"),
    (r"\bthink\b", "figure"),
    (r"\bshow\b", "prove"),
    (r"\btell\b", "lay out"),
]


@rule
def inject_vocabulary_variety(text: str) -> str:
    """Replace common words with rarer alternatives at low probability.

    Creates sporadic perplexity spikes — the kind of unexpected word
    choices that naturally occur in human writing but AI avoids.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) < 3:
        return text

    result = []
    for i, sent in enumerate(sentences):
        words = sent.split()
        if len(words) < 6:
            result.append(sent)
            continue

        # ~8% chance per applicable sentence to swap one word
        if random.random() < 0.08:
            candidates = [(pattern, repl) for pattern, repl in VOCAB_SWAPS
                          if re.search(pattern, sent, re.IGNORECASE)]
            if candidates:
                pattern, repl = random.choice(candidates)
                if random.random() < 0.5:  # 50% swap rate when triggered
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
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) < 3:
        return text

    result = []
    for i, sent in enumerate(sentences):
        words = sent.split()
        if len(words) < 6:
            result.append(sent)
            continue

        # ~12% chance to frame a sentence with personal experience
        if random.random() < 0.12:
            framing = random.choice(PERSONAL_FRAMINGS)
            sent = framing + sent[0].lower() + sent[1:]

        # ~10% chance to add a personal aside at end of longer sentences
        if len(words) > 12 and random.random() < 0.10:
            aside = random.choice(PERSONAL_ASIDES)
            sent = sent.rstrip(".!?") + aside

        result.append(sent)

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
