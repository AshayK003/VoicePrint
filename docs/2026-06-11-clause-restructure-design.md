# Clause Restructure Module — Design Spec

## Problem

GPTZero's 2026 Paraphraser Shield detects synonym-level paraphrasing with 93.5% recall. VoicePrint's current pipeline (scrub → paraphrase → polish → detect) changes words but preserves sentence skeleton, clause order, and syntactic structure — exactly what the Shield exploits.

## Solution

A new `voiceprint/restructure.py` module that uses spaCy dependency parsing to identify clause boundaries and apply syntactic transformations. No LLM calls, no API costs.

## Impact

Clause-level restructuring is the most effective single bypass technique per academic research (Cheng et al., NeurIPS 2025), dropping GPTZero scores ~35%. It directly counters the Paraphraser Shield by altering syntax, not vocabulary.

## Design

### 6 Transformation Rules (via `@rule` decorator)

| # | Rule | Mechanism | Before | After |
|---|------|-----------|--------|-------|
| 1 | `front_subordinate_clauses` | Move `advcl`/`mark` from sentence end → front | "Data was significant **because** samples were large." | "**Because** samples were large, data was significant." |
| 2 | `extract_relative_clauses` | Convert `relcl` to independent sentence | "The study, **which** Smith led, showed..." | "Smith led the study. It showed..." |
| 3 | `swap_main_subordinate` | Invert main/subordinate clause order | "**Although** data was noisy, we proceeded." | "Data was noisy. We proceeded anyway." |
| 4 | `convert_appositives` | Expand `appos` → separate sentence | "Dr. Smith, **lead researcher**, presented." | "Dr. Smith was the lead researcher. They presented." |
| 5 | `split_compounds` | Break `conj` sentences at coordinating conjunction | "Results were significant **and** team celebrated." | "Results were significant. The team celebrated." |
| 6 | `normalize_burstiness` | Measure document burstiness. If < 0.4: split sentences > 30 words into 2 at clause boundary; rejoin adjacent sentences < 10 words into one. Repeat until burstiness >= 0.4 or no more adjustments possible. | — | — |

### Module Structure (`voiceprint/restructure.py`)

```
_rules = []                          # Registered via @rule
_cached_nlp = None                   # Global spaCy model (lazy-loaded)

@rule
front_subordinate_clauses(text) → str

@rule
extract_relative_clauses(text) → str

@rule
swap_main_subordinate(text) → str

@rule
convert_appositives(text) → str

@rule
split_compounds(text) → str

@rule
normalize_burstiness(text) → str

restructure(text) → str              # Applies all rules in sequence
```

### spaCy Dependency Labels Used

| Label | Meaning | Used By |
|-------|---------|---------|
| `mark` | Subordinating conjunction (because, although, while) | Rules 1, 3 |
| `advcl` | Adverbial clause modifier | Rules 1, 3 |
| `relcl` | Relative clause modifier | Rule 2 |
| `appos` | Appositional modifier | Rule 4 |
| `conj` | Conjunct | Rule 5 |
| `cc` | Coordinating conjunction (and, but, or) | Rule 5 |
| `nsubj` | Nominal subject | Rules 2, 4 |
| `punct` | Punctuation | All (boundary detection) |

### Pipeline Integration

Insert restructure after paraphrase, before polish, inside the iterative loop:

```python
# In pipeline.py iteration loop:
current = scrub(current)                                    # Stage 1 (once)
for iteration in range(max_iterations):
    candidates = generate_candidates(current, ...)           # Stage 2
    current = select_best(original, candidates, ...)
    current = restructure(current)                           # NEW: Stage 2b
    current = polish(current)                                # Stage 3
    result = detect(current)                                 # Stage 4
    ...
```

This placement ensures:
- LLM paraphrasing runs first (changes vocabulary)
- Restructure then changes clause structure (counters Paraphraser Shield)
- Polish applies finishing touches (fragments, dysfluencies, etc.)
- Transformations compound across iterations

### Determinism & Randomness

Follow existing pattern: each probabilistic rule uses a dedicated `random.Random()` seeded from `hashlib.md5(text.encode())`. Same input always produces same output.

### Error Handling

- **spaCy model not found**: Log warning, return text unchanged (graceful degradation)
- **Parsing failure on a sentence**: Skip that sentence, continue with next
- **Sentence too short (< 10 tokens)**: Skip (no clauses to restructure)
- **No applicable clause structure**: Skip, return sentence unchanged

### Configuration

Add to `Config` dataclass in `config.py`:

```python
@dataclass
class Config:
    ...
    use_restructure: bool = True        # Enable/disable clause restructuring
    restructure_probability: float = 0.6  # Probability each rule applies per eligible sentence (seeded deterministically)
```

### Files Changed

| File | Change |
|------|--------|
| `voiceprint/restructure.py` | **New** — 150-200 lines |
| `voiceprint/pipeline.py` | Add restructure stage in iteration loop |
| `voiceprint/__init__.py` | Export `restructure` |
| `voiceprint/config.py` | Add `use_restructure`, `restructure_probability` |
| `requirements.txt` | Add `spacy>=3.8` (already installed transitively) |
| `tests/test_restructure.py` | **New** — unit tests for each rule + integration |

### Test Plan

| Test | Type |
|------|------|
| Each transformation rule with known input/output | Unit |
| Empty/edge inputs (empty string, single word, no clauses) | Edge |
| Long/compound/complex sentences | Stress |
| Determinism: same input → same output | Property |
| Pipeline integration: restructure runs within full pipeline | Integration |
| spaCy model missing → graceful fallback | Resilience |

### Out of Scope (v1)

- **Multi-parse iteration**: Not retrying spaCy analysis with different models
- **Tree-level optimization**: Not computing optimal clause order per document
- **Per-detector restructuring**: Not targeting GPTZero vs Turnitin differently
- **Style scorer integration**: Not using style scorer to guide transformations

## Verification

- `pytest tests/test_restructure.py -v` — all new tests pass
- `pytest tests/ -v` — all 345 existing tests still pass
