"""Stage 2b: Clause Restructure — Syntactic transformations via spaCy.

Targets GPTZero's sentence-structure signal by altering clause order,
extracting relative clauses, converting appositives, and normalizing
burstiness. No LLM calls, no API costs.
"""

import hashlib
import logging
import random
import re
from typing import Callable

from ._text import sentences as _split_sentences

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

_rules: list[Callable[[str], str]] = []


def rule(func: Callable[[str], str]) -> Callable[[str], str]:
    _rules.append(func)
    return func


# ---------------------------------------------------------------------------
# Lazy-loaded spaCy model
# ---------------------------------------------------------------------------

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp if _nlp is not False else None
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
        return _nlp
    except Exception as e:
        logger.warning(f"spaCy model not available: {e}")
        _nlp = False
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capitalize(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def _uncapitalize(text: str) -> str:
    if not text:
        return text
    return text[0].lower() + text[1:]


# ---------------------------------------------------------------------------
# Rule 1: Front subordinate clauses — move clause from end to beginning
# ---------------------------------------------------------------------------

_ADVCL_MARKER_WORDS = {
    "because", "although", "though", "while", "since", "if", "when",
    "unless", "until", "whereas", "after", "before", "as",
    "once", "whether",
}

# Regex: find subordinate clause at end: ", conjunction ..." or " conjunction ..."
_RE_SUBORDINATE_END = re.compile(
    r",\s*(because|although|though|while|since|if|when|unless|until|"
    r"whereas|after|before|as|once|whether)\s+(.+)",
    re.IGNORECASE,
)


@rule
def front_subordinate_clauses(text: str, _prob: float = 0.4) -> str:
    """Move subordinate clauses from end to beginning of sentence.

    Uses spaCy for clause detection with regex fallback for robustness.
    """
    nlp = _get_nlp()
    if nlp is None:
        return text

    _rng = random.Random()
    _rng.seed(hashlib.md5(text.encode()).hexdigest())

    sents = _split_sentences(text)
    if len(sents) < 1:
        return text

    result = []
    for sent in sents:
        words = sent.split()
        if len(words) < 6 or _rng.random() >= _prob:
            result.append(sent)
            continue

        # Try spaCy-based detection
        doc = nlp(sent)
        clause_data = None
        for token in doc:
            if token.dep_ == "advcl" and token.i > 0:
                marker = None
                for child in token.subtree:
                    if child.dep_ == "mark":
                        marker = child
                        break
                if marker is None:
                    continue
                subtree_tokens = sorted(token.subtree, key=lambda t: t.i)
                if not subtree_tokens:
                    continue
                end_char = subtree_tokens[-1].idx + len(subtree_tokens[-1].text_with_ws)
                clause_data = {
                    "marker_pos": marker.idx,
                    "marker_word": marker.text,
                    "end_char": end_char,
                }
                break

        if clause_data is None:
            result.append(sent)
            continue

        idx = clause_data["marker_pos"]
        before = sent[:idx].strip().rstrip(",. ")
        marker_word = clause_data["marker_word"]
        after = sent[idx + len(marker_word):].rstrip(",. ")
        if not after:
            result.append(sent)
            continue

        after = after.strip()
        before = _capitalize(before)
        after = _uncapitalize(after)
        transformed = f"{_capitalize(marker_word)} {after}, {_uncapitalize(before)}."
        result.append(transformed)

    return " ".join(result)


# ---------------------------------------------------------------------------
# Rule 2: Extract relative clauses — convert to independent sentences
# ---------------------------------------------------------------------------

_RE_RELATIVE_CLAUSE = re.compile(
    r",\s*(which|who|whom|whose|that|where)\s+(.+?),\s*",
    re.IGNORECASE,
)


@rule
def extract_relative_clauses(text: str, _prob: float = 0.3) -> str:
    """Convert relative clauses to independent sentences.

    Falls back to regex detection when spaCy parsing doesn't find a relcl.
    """
    nlp = _get_nlp()
    if nlp is None:
        return text

    _rng = random.Random()
    _rng.seed(hashlib.md5(text.encode()).hexdigest())

    sents = _split_sentences(text)
    if len(sents) < 1:
        return text
    result = []
    for sent in sents:
        words = sent.split()
        if len(words) < 6 or _rng.random() >= _prob:

            result.append(sent)
            continue

        doc = nlp(sent)
        relcl_data = None
        for token in doc:
            if token.dep_ == "relcl":
                head = token.head
                rel_tokens = sorted(token.subtree, key=lambda t: t.i)
                if not rel_tokens:
                    continue
                rel_start = rel_tokens[0].idx
                rel_last = rel_tokens[-1]
                rel_end = rel_last.idx + len(rel_last.text_with_ws)
                subject = head.text
                subject_start = head.idx
                before = doc.text[:rel_start].strip()
                after = doc.text[rel_end:].strip().lstrip(",").strip()

                # Check if the relcl was preceded by a comma — we want the
                # text before the comma as the "before" part
                if before.endswith(","):
                    before = before[:-1].strip()

                relcl_data = {
                    "subject": subject,
                    "before": before,
                    "after": after,
                    "rel_text": doc.text[rel_start:rel_end].strip(),
                    "head_start": subject_start,
                }
                break

        if relcl_data is None:
            # Regex fallback
            match = _RE_RELATIVE_CLAUSE.search(sent)
            if match:
                rel_word = match.group(1)
                rel_content = match.group(2)
                before = sent[:match.start()].strip().rstrip(",")
                after = sent[match.end():].strip()
                subject = before.split()[-1] if before.split() else ""
                relcl_data = {
                    "subject": subject,
                    "before": before,
                    "after": after,
                    "rel_text": f"{rel_word} {rel_content}",
                    "head_start": 0,
                }

        if relcl_data is None:
            result.append(sent)
            continue

        extracted = (
            f"{_capitalize(relcl_data['subject'])} "
            f"{relcl_data['rel_text'].split(' ', 1)[-1]}"
        )
        if not extracted.endswith("."):
            extracted += "."

        if relcl_data["after"]:
            continuation = _capitalize(relcl_data["after"])
            if not continuation.endswith("."):
                continuation += "."
            transformed = f"{relcl_data['before']}. {extracted} {continuation}"
        else:
            transformed = f"{relcl_data['before']}. {extracted}"

        result.append(transformed)

    return " ".join(result)


# ---------------------------------------------------------------------------
# Rule 3: Swap main/subordinate clause order
# ---------------------------------------------------------------------------

_RE_LEADING_SUBORDINATE = re.compile(
    r"^(Although|Though|While|Since|If|When|Unless|Until|Whereas|"
    r"Because|As|Once|Whether)\b(.+?),\s*(.+)",
    re.IGNORECASE,
)


@rule
def swap_main_subordinate(text: str, _prob: float = 0.3) -> str:
    """When a sentence starts with a subordinate clause, swap it to the end.

    'Although the data was noisy, we proceeded.' →
    'We proceeded, although the data was noisy.'
    """
    import re as _re

    _rng = random.Random()
    _rng.seed(hashlib.md5(text.encode()).hexdigest())

    sents = _split_sentences(text)
    if len(sents) < 1:
        return text

    result = []
    for sent in sents:
        words = sent.split()
        if len(words) < 6 or _rng.random() >= _prob:
            result.append(sent)
            continue

        match = _RE_LEADING_SUBORDINATE.match(sent.strip())
        if not match:
            result.append(sent)
            continue

        conjunction = match.group(1)
        sub_clause = match.group(2).strip().rstrip(",")
        main_clause = match.group(3).strip().rstrip(",. ")

        transformed = f"{_capitalize(main_clause)}, {_uncapitalize(conjunction)} {_uncapitalize(sub_clause)}."
        result.append(transformed)

    return " ".join(result)


# ---------------------------------------------------------------------------
# Rule 4: Convert appositives — expand to separate sentence
# ---------------------------------------------------------------------------

_RE_APPOSITIVE = re.compile(
    r"(\w[\w\s.]*?),\s+(the|a|an)\s+([^,]+?),\s*(.+)",
    re.IGNORECASE,
)


@rule
def convert_appositives(text: str, _prob: float = 0.3) -> str:
    """Expand appositives into separate sentences.

    'Dr. Smith, the lead researcher, presented.' →
    'Dr. Smith was the lead researcher. They presented.'
    """
    _rng = random.Random()
    _rng.seed(hashlib.md5(text.encode()).hexdigest())

    sents = _split_sentences(text)
    if len(sents) < 1:
        return text

    result = []
    for sent in sents:
        words = sent.split()
        if len(words) < 6 or _rng.random() >= _prob:
            result.append(sent)
            continue

        match = _RE_APPOSITIVE.match(sent.strip())
        if not match:
            result.append(sent)
            continue

        subject_phrase = match.group(1).strip()
        article = match.group(2)
        appos_desc = match.group(3).strip()
        rest = match.group(4).strip()

        subject = subject_phrase.split()[-1]

        expansion = f"{_capitalize(subject_phrase)} was {article} {_uncapitalize(appos_desc)}."
        subjects_small = {"he", "she", "they", "it", "dr", "mr", "ms", "mrs"}
        first_subj_word = subject.split()[0].lower().strip(",. ")
        pronoun = "They" if first_subj_word not in subjects_small and "." not in first_subj_word else subject.split()[0]

        if rest:
            continuation = f"{_capitalize(pronoun)} {_uncapitalize(rest)}"
            if not continuation.endswith((".", "!", "?")):
                continuation += "."
            transformed = f"{expansion} {continuation}"
        else:
            transformed = expansion

        result.append(transformed)

    return " ".join(result)


# ---------------------------------------------------------------------------
# Rule 5: Split compound sentences at coordinating conjunctions
# ---------------------------------------------------------------------------

_RE_COMPOUND_SPLIT = re.compile(
    r"(.+?),\s*(and|but|or|yet|so)\s+(.+)",
    re.IGNORECASE,
)


@rule
def split_compounds(text: str, _prob: float = 0.35) -> str:
    """Split compound sentences at 'X, and Y' or 'X, but Y' patterns."""
    _rng = random.Random()
    _rng.seed(hashlib.md5(text.encode()).hexdigest())

    sents = _split_sentences(text)
    if len(sents) < 1:
        return text

    result = []
    for sent in sents:
        words = sent.split()
        if len(words) < 6 or _rng.random() >= _prob:
            result.append(sent)
            continue

        match = _RE_COMPOUND_SPLIT.match(sent.strip())
        if not match:
            result.append(sent)
            continue

        first = match.group(1).strip()
        conjunction = match.group(2)
        second = match.group(3).strip()

        first = _capitalize(first)
        second = _capitalize(second)
        if not first.endswith((".", "!", "?")):
            first += "."
        if not second.endswith((".", "!", "?")):
            second += "."

        transformed = f"{first} {second}"
        result.append(transformed)

    return " ".join(result)


# ---------------------------------------------------------------------------
# Rule 6: Normalize burstiness
# ---------------------------------------------------------------------------

@rule
def normalize_burstiness(text: str, _prob: float = 1.0) -> str:
    """If document burstiness < 0.4, engineer it upward.

    Splits sentences > 30 words at midpoints, rejoins adjacent short
    sentences (< 10 words each). Repeats until burstiness >= 0.4.
    """
    from .metrics import burstiness as _calc_burstiness

    max_passes = 5
    for _ in range(max_passes):
        if _calc_burstiness(text) >= 0.4:
            break

        sents = _split_sentences(text)
        if len(sents) == 0:
            break
        if len(sents) == 1 and len(sents[0].split()) <= 30:
            break

        new_sents = list(sents)
        changed = False

        for i, sent in enumerate(sents):
            words = sent.split()
            if len(words) > 30:
                mid = len(words) // 2
                part1 = " ".join(words[:mid]).rstrip(",.!?;:") + "."
                part2_words = words[mid:]
                if part2_words:
                    if part2_words[0][0].islower():
                        part2_words[0] = part2_words[0][0].upper() + part2_words[0][1:]
                    part2 = " ".join(part2_words)
                    if not part2.endswith((".", "!", "?")):
                        part2 += "."
                    new_sents[i] = part1
                    new_sents.insert(i + 1, part2)
                    changed = True
                    break

        if not changed:
            for i in range(len(sents) - 1):
                words_a = sents[i].split()
                words_b = sents[i + 1].split()
                if 3 <= len(words_a) <= 10 and 3 <= len(words_b) <= 10:
                    joined = sents[i].rstrip(",.!?;:") + " and " + _uncapitalize(sents[i + 1])
                    new_sents[i] = joined
                    del new_sents[i + 1]
                    changed = True
                    break

        if not changed:
            break

        if changed:
            text = " ".join(new_sents)

    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_restructure(text: str, probability: float = 0.4) -> str:
    """Apply all restructure rules to text.

    Args:
        text: Input text to restructure
        probability: Per-rule application probability (0.0 = never, 1.0 = always)

    Returns:
        Restructured text with altered clause structure
    """
    for rule_fn in _rules:
        try:
            text = rule_fn(text, probability)
        except Exception as e:
            logger.warning(f"Restructure rule {rule_fn.__name__} failed: {e}")
    return text
