# VoicePrint — Full Project Reference

## Project Overview

VoicePrint is a multi-stage AI text humanization pipeline that transforms AI-generated text into natural, human-like writing capable of bypassing state-of-the-art AI detectors (GPTZero, Turnitin, Originality.ai, ZeroGPT).

**Core Principle:** A multi-stage pipeline is 2.3x more effective than any single technique (research-backed).

**Constraints:**
- No GPU required — cloud APIs only
- Streamlit web app frontend
- Balanced stealth vs readability (not maximum evasion at quality cost)
- Bypass ALL major detectors (universal approach)
- All dependencies must be MIT/Apache-2.0 licensed

---

## Architecture: 5-Stage Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT: AI-Generated Text              │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 1: Heuristic Scrub (No Model - Pure Python)     │
│  • Replace AI transition phrases + banned words        │
│  • Force burstiness via 3-1-5 pattern                   │
│  • Inject contractions                                  │
│  • Break tricolons, em-dash overuse                     │
│  • Remove hedging + filler phrases                      │
│  Module: voiceprint/scrub.py                            │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2: Adversarial Paraphrasing (Cloud LLM API)     │
│  • NINJA_PROMPTS: 3 persona levels (convos/blog/raw)   │
│  • Detection-guided feedback (prev_p_ai injected)       │
│  • Best-of-N sampling (default 8, configurable)         │
│  • Perplexity-aware candidate rejection (ppl < 30)     │
│  • Multi-provider: OpenCode Zen / Gemini / Groq / ...  │
│  Module: voiceprint/paraphrase.py                       │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2b: Clause Restructuring (No Model - spaCy)     │
│  • 6 syntactic transformation rules                     │
│  • spaCy dependency parsing for clause boundaries       │
│  • Front subordinate clauses, extract relative clauses  │
│  • Swap main/subordinate, convert appositives           │
│  • Split compounds, normalize burstiness                │
│  • Seeded deterministic randomness (per-rule @rule)     │
│  Module: voiceprint/restructure.py                      │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 3: Style Polish (No Model - Pure Python)         │
│  • Convert passive → active voice                       │
│  • Dysfluency injection (well, I mean, you know)        │
│  • Personal narrative injection (first-person framing)  │
│  • Formal-to-casual tone conversion (60+ replacements)  │
│  • Vocabulary variety (perplexity spikes via rare words)│
│  • Rhetorical questions + sentence fragments            │
│  • **Naturalize**: final pass reduces over-injected     │
│    artifacts (dysfluencies, narratives, fragments)      │
│    for natural readability                              │
│  Module: voiceprint/polish.py                           │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 4: AI-Pattern Detection & Scoring                │
│  • Statistical pre-filter (burstiness + 15+ signals)   │
│  • Perplexity scoring via GPT-2 (lazy-loaded)           │
│  • RoBERTa-large-openai detector (when loaded)          │
│  • chatgpt-detector-roberta (Hello-SimpleAI)            │
│  • Binoculars zero-shot metric                          │
│  • Select lowest p_ai candidate; tiebreak on perplexity │
│  Module: voiceprint/detect.py                           │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  OUTPUT: Humanized Text + Detection Scores              │
└─────────────────────────────────────────────────────────┘
```

---

## File Structure

```
VoicePrint/
├── AGENTS.md                 # This file — project context for agents
├── .gitignore
├── requirements.txt
├── README.md
├── app.py                    # Streamlit frontend
├── benchmark.py              # One-off benchmark script
├── voiceprint/
│   ├── __init__.py           # Exports: perplexity_score, raw_perplexity, PromptMemory
│   ├── config.py             # API keys, thresholds, model names, env/registry
│   ├── scrub.py              # Stage 1: Heuristic rule engine (50+ rules)
│   ├── paraphrase.py         # Stage 2: LLM API, NINJA_PROMPTS, candidate selection
│   ├── restructure.py        # Stage 2b: Clause restructuring (6 rules, spaCy)
│   ├── polish.py             # Stage 3: Style post-processing (8 rules)
│   ├── detect.py             # Stage 4: AI-text detection ensemble + perplexity
│   ├── style_scorer.py       # Style quality scoring
│   ├── metrics.py            # Burstiness, readability calculations
│   ├── similarity.py         # Semantic similarity gate
│   ├── patterns.py           # AI-pattern fingerprint signals (15+, optional pystylometry)
│   ├── memory.py             # PromptMemory - adaptive feedback loop
│   ├── perplexity.py         # GPT-2 based perplexity scoring
│   ├── _text.py              # Shared text utilities (sentences(), etc.)
│   ├── service.py            # Service layer (business logic between UI and pipeline)
│   └── pipeline.py           # Main orchestrator + retry loop + PromptMemory
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Stubs torch/transformers for fast tests
│   ├── test_scrub.py
│   ├── test_metrics.py
│   ├── test_pipeline.py
│   ├── test_memory.py
│   ├── test_perplexity.py
│   ├── test_paraphrase.py
│   ├── test_polish.py
│   ├── test_patterns.py
│   ├── test_service.py
│   ├── test_similarity.py
│   ├── test_detect.py
│   ├── test_config.py
│   ├── test_app.py
│   ├── test_restructure.py       # Clause restructuring unit tests
│   ├── test_split_sentences.py   # Sentence splitting tests
│   ├── test_perplexity_spikes.py # Perplexity spike injection tests
│   └── test_burstiness_gate.py   # Burstiness gate tests
├── tools/
│   └── analyze_banned_words.py  # Dataset-based banned word pruning (gsingh1-py/train)
├── .dependencygraph/         # Dependency graph artifacts
├── opencode.json             # OpenCode agent configuration
└── docs/
    ├── techniques.md         # Research notes
    └── 2026-06-11-clause-restructure-design.md  # Clause restructure spec
```

---

## Dependencies

| Package | Version | Purpose | License |
|---------|---------|---------|---------|
| `streamlit` | ≥1.35 | Web frontend | Apache-2.0 |
| `litellm` | ≥1.40 | Unified LLM API client | MIT |
| `transformers` | ≥4.40 | AI-text detector models | Apache-2.0 |
| `torch` | ≥2.0 | Model inference (CPU) | BSD-3 |
| `sentence-transformers` | ≥3.0 | Semantic similarity | Apache-2.0 |
| `textstat` | ≥0.7 | Readability metrics | MIT |
| `numpy` | ≥1.24 | Math operations | BSD-3 |
| `nltk` | ≥3.8 | Tokenization | Apache-2.0 |
| `pystylometry` | ≥1.4 | Optional lexical diversity metrics | MIT |

---

## Key Open-Source Components

### Detectors (Ensemble)
1. **`openai-community/roberta-large-openai-detector`** — MIT, ~1.4GB, primary detector
2. **`Hello-SimpleAI/chatgpt-detector-roberta`** — MIT, ~500MB, secondary detector
3. **Binoculars zero-shot** — Custom implementation (perplexity ratio between two models)

### Similarity Gate
- **`sentence-transformers/all-MiniLM-L6-v2`** — Apache-2.0, ~80MB
- Threshold: cosine similarity ≥ 0.68 between original and humanized text

### Perplexity Scoring
- **`gpt2`** — ~525MB cached locally, evaluates how predictable/surprising text is
- Human-like text: ~26 perplexity (more predictable)
- AI-generated text: ~138 perplexity (more surprising, buzzword-heavy)
- Normalized score: `min(max((80 - ppl) / 60, 0), 1)` — higher = more human
- Lazy loaded, globally cached, graceful fallback if model unavailable
- Mock guard rejects MagicMock stubs from conftest.py

### LLM Providers (Free Tiers)
- **OpenCode Zen** (default) — `openai/nemotron-3-ultra-free` (best free model), `openai/mimo-v2.5-free` (fallback)
- Google Gemini Flash (15 RPM, free)
- Groq Llama 3.3 70B (30 RPM, free)
- Mistral Large (1 req/sec, free)

---

## Implementation Phases

### Phase 1: Heuristic Engine (scrub.py) — COMPLETED
- 50+ rules for AI phrase replacement, including data-pruned banned words (47 AI tells with ≥2x ratio)
- 3-1-5 burstiness pattern, contraction injection, tricolon breaking, em-dash density reduction
- Filler phrase removal, hedging removal, discourse marker cleanup

### Phase 2: Detection Ensemble (detect.py) — COMPLETED
- Statistical pre-filter with 15+ pattern signals (burstiness, vocabulary, modality, etc.)
- Perplexity scoring integration (tiebreaker, candidate rejection)
- Binoculars zero-shot metric
- Ensemble summary with perplexity line
- Model loading deferred (statistical-only when models unavailable)

### Phase 3: LLM Paraphrasing (paraphrase.py) — COMPLETED
- 3 persona-based NINJA_PROMPT levels: conversational, blog-style, casual/raw
- Detection-guided refinement: prev_p_ai feedback escalates retry tone
- Best-of-N sampling (default 1, configurable via sidebar)
- Perplexity-aware candidate rejection (ppl < 30 filtered)
- Multi-provider fallback via litellm

### Phase 4: Quality Gates — COMPLETED
- Semantic similarity checking (MiniLM / Jaccard fallback)
- Burstiness calculation + readability scoring (textstat)
- Perplexity scoring via GPT-2 (lazy-loaded, 0-1 normalized)
- PromptMemory adaptive feedback loop (records per-level p_ai, recommends best prompt)

### Phase 5: Style Polish (polish.py) — COMPLETED
- Passive → active voice conversion
- Dysfluency injection (well, I mean, you know — ~15% per sentence)
- Personal narrative injection (first-person framing — ~12% per sentence)
- Formal-to-casual tone conversion (60+ replacement patterns)
- Vocabulary variety (common→rare word swaps at ~8%)
- Rhetorical question injection + sentence fragment insertion
- Duplicate removal + punctuation normalization

### Phase 6: Clause Restructuring (restructure.py) — COMPLETED
- 6 syntactic transformation rules via spaCy dependency parsing
- Front subordinate clauses, extract relative clauses, swap main/subordinate
- Convert appositives, split compounds, normalize burstiness
- Seeded deterministic randomness per rule (follows polish.py @rule pattern)
- Integrated as Stage 2b (after paraphrase, before polish in iteration loop)

### Phase 7: Pipeline Orchestrator (pipeline.py) — COMPLETED
- All 5 stages wired with configurable toggles
- Detection-guided retry loop (up to N iterations, tracks best by p_ai + perplexity tiebreak)
- PromptMemory integration (adaptive prompt selection, outcome recording)
- Parallel metrics computation (ThreadPoolExecutor)
- Progress reporting callback for UI

### Phase 8: Naturalize & Readability Polish — COMPLETED
- Added `naturalize()` as the final polish rule: reduces over-applied dysfluency, personal narrative, and self-doubt markers
- Merges short fragments back into adjacent sentences for flow
- Bumped `max_iterations` to 5 for more aggressive evasion cycles

### Phase 9: Streamlit UI (app.py) — COMPLETED
- Text input area with clear button
- Pipeline controls (aggressiveness slider, stage toggles)
- Side-by-side comparison view (diff highlighting)
- Detection score dashboard
- Paraphrase: Applied / Skipped badge
- Styled dark-panel output
- API key registry detection for sidebar badge

---

## Design Decisions

1. **Why cloud APIs instead of local models?** — No GPU available. Cloud APIs are cheaper for infrequent use and avoid 2GB+ model downloads for the paraphraser.

2. **Why 3-1-5 pattern?** — Research shows human writing follows a rhythm: 3-4 regular sentences, 1 short punchy sentence (3-8 words), 1 long complex sentence (35+ words). This breaks the uniform structural signatures detectors look for.

3. **Why best-of-N instead of single-pass?** — Adversarial Paraphrasing paper (NeurIPS 2025) shows best-of-N adds 5-15% attack success rate for free. Default N=1 for speed (~24s), users can increase.

4. **Why similarity gate at 0.65?** — Empirical tuning on this pipeline. Lowered from 0.68 to 0.65 to allow more aggressive restructure/paraphrase transformations while still preserving meaning. The original 0.78 from paniccow/humanizer was too restrictive.

5. **Why deterministic scrub BEFORE LLM?** — Removes the easy-to-fix AI tells (transitions, vocabulary) so the LLM can focus on structural transformation instead of surface-level cleanup.

6. **Why GPT-2 for perplexity (not GPTZero's approach)?** — GPT-2 is 525MB, loads on CPU, and finds human-like text more predictable than AI buzzword-heavy text. The inversion (human ppl~26 < AI ppl~138) is what makes it a useful detection signal.

7. **Why normalize perplexity (80→0, 20→1)?** — Empirically, ppl≥80 indicates AI-generated text (surprising to GPT-2), while ppl≤20 indicates predictable human-like text. The clamped linear transform creates a clean 0-1 score where higher = more human.

8. **Why PromptMemory?** — Different prompt styles work better for different texts. PromptMemory tracks which prompt level (0=conversational, 1=blog, 2=raw) produces the lowest p_ai per session, and biases future selections toward that level.

9. **Why mock guard in perplexity.py?** — conftest.py stubs transformers for fast test collection. perplexity detects MagicMock via `isinstance(model, Mock)` and returns None instead of producing meaningless output.

10. **Why default n_candidates=8?** — Best-of-8 balances speed and quality. Testing showed 8 candidates achieves ~20% lower p_ai than 1 candidate with reasonable runtime. Users can adjust via sidebar.

11. **Why persona-based prompts instead of detection-avoidance?** — Models are good at "sound like a person explaining this" (training data) and bad at "avoid pattern X" (abstract goal). Switched all 3 NINJA_PROMPT levels to persona framing.

12. **Why perplexity-aware candidate rejection?** — Candidates with raw_perplexity < 30 are too predictable/AI-like. Filtering them before detection scoring improves quality.

13. **Why registry fallback for API keys?** — Windows registry (`HKCU\Environment\OPENCODE_API_KEY`) persists across sessions without a `.env` file. Resolution: explicit > env var > registry.

14. **Why post-paraphrase scrub?** — LLM tends to re-introduce AI patterns (formal transitions, hedging, banned words). Scrub runs again after paraphrasing to catch these.

15. **Why dedicated per-function RNGs?** — Each polish rule uses its own `random.Random()` instance seeded from `hashlib.md5(text.encode())`. This guarantees determinism (same input → same output), prevents global seed contamination between rules, and makes tests reproducible without `random.seed()`.

16. **Why inline `sentences()` utility?** — 12+ copies of `re.split(r"(?<=[.!?])\s+", ...)` were scattered across 5 modules. Extracted to `voiceprint/_text.py` for a single source of truth. Import as `from ._text import sentences as _split_sentences` to avoid local variable shadowing.

17. **Why lazy imports in detect.py?** — `import torch` and `from transformers import ...` at module level cause `OSError: [WinError 1114]` on Windows when torch DLL is unloadable. Moving them inside methods defers the crash to the actual model load attempt, allowing statistical-only fallback.

18. **Why `load_config` doesn't mutate `os.environ`?** — Previous implementation wrote `os.environ["OPENCODE_API_KEY"] = reg_key` as a side effect, leaking registry state into the process environment. Setting `config.api_key` directly avoids invisible coupling for other callers of `os.getenv`.

19. **Why clause restructuring as Stage 2b (after paraphrase, before polish)?** — LLM paraphrasing changes vocabulary first, restructuring then alters sentence skeleton (counters GPTZero's Paraphraser Shield which exploits preserved syntax), and polish applies finishing touches (dysfluencies, fragments). Transformations compound across iterations. Research shows clause restructuring drops GPTZero scores ~35% (Cheng et al., NeurIPS 2025).

20. **Why default restructure_probability 0.6?** — Tuned empirically: 0.4 was too conservative and left too many sentences with unmodified clause structure, reducing bypass effectiveness. 0.6 applies rules to ~60% of eligible sentences, balancing structural variety with content preservation.

21. **Why a final `naturalize()` pass?** — Polish rules compound and can make text feel "forced" (too many dysfluencies, personal framings, fragments). The final pass detects over-application and prunes excess markers, so the output reads naturally while still bypassing detectors. It also merges very short fragments (<4 words) into adjacent sentences to avoid choppy rhythm.

---

## Testing Strategy

- **345 tests total**, ~19s suite duration
- Unit tests for each module (scrub, detect, metrics, similarity, patterns, perplexity, memory, polish, paraphrase, restructure, config, service, app)
- Integration test for full pipeline
- Test against known AI-generated texts
- Verify detection scores improve after humanization
- Check semantic similarity stays above threshold
- conftest.py stubs torch/transformers so pure-Python tests run fast without GPU libs
- Perplexity tests skip model-dependent assertions when GPT-2 is unavailable
- Mock guard rejects stubs via `type(_MODEL).__module__.startswith("unittest.mock")`

---

## Reference Repositories

| Repo | Stars | What to Learn |
|------|-------|---------------|
| `paniccow/humanizer` | 1 | Full pipeline architecture, scrub rules, detection ensemble |
| `chengez/Adversarial-Paraphrasing` | 44 | Core adversarial algorithm (NeurIPS 2025) |
| `lynote-ai/humanize-text` | 1.1k | Production pipeline patterns, methodology reference |
| `rudra496/StealthHumanizer` | 41 | UI/UX architecture, style engine design |
| `stef41/lmscan` | 17 | Zero-dependency statistical detection |

---

## Agent Notes

- When editing `scrub.py`, follow the existing rule pattern: each rule is a function that takes `text: str` and returns `str`
- Detection models are loaded once via `@st.cache_resource` in Streamlit — never reload per request
- LLM calls go through `litellm` for provider-agnostic code
- API keys: resolved via `build_config()` in service.py with priority: explicit > env var > Windows registry
- Registry fallback (`_read_registry_env`) reads `HKCU\Environment\OPENCODE_API_KEY`
- Never commit `.env` or API keys. The `.env` file is not used — registry or sidebar input only.
- Use type hints on all functions
- Keep modules under 300 lines — split if growing larger
- New modules should be exported from `__init__.py`
- Perplexity module has a mock guard — `type(_MODEL).__module__.startswith("unittest.mock")` — returns None under conftest stubs. Does NOT use `isinstance(model, Mock)` or import from `unittest.mock`.
- Paraphrase prompts are in `paraphrase.py` as `NINJA_PROMPTS` list (3 levels). Add/modify levels there.
- Polish rules follow the `@rule` decorator pattern with `text: str → str` signature — restructure.py follows the same pattern
- Polish randomness uses dedicated per-function `random.Random()` instances seeded from `hashlib.md5(text.encode())` — deterministic per input text, no global seed contamination — restructure.py follows the same pattern
- Restructure uses spaCy dependency parsing (lazy-loaded, globally cached `_cached_nlp`). Graceful degradation if model not found.
- Use `from ._text import sentences as _split_sentences` for sentence splitting — don't write `re.split(r"(?<=[.!?])\s+", ...)` inline
- `load_config()` does NOT mutate `os.environ` — registry/API key resolution sets `config.api_key` directly
- `detect.py` lazily imports `torch` and `transformers` inside methods, never at module level
- `pystylometry` is optional (MIT license). Detect availability via `importlib.util.find_spec("pystylometry")` — see `patterns.py`.
- Default provider: OpenCode Zen (`openai/nemotron-3-ultra-free` via OpenCode Zen API at `https://opencode.ai/zen/v1`)
- API key auto-detection: `sk-` keys longer than 60 chars → OpenCode Zen; under 60 → OpenAI
- `Config.__post_init__` is intentionally a no-op. env/registry resolution is in `build_config()` and `load_config()`.
- `load_config()` does NOT mutate `os.environ` — registry/API key resolution sets `config.api_key` directly.
