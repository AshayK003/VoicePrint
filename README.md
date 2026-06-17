# VoicePrint — AI Text Humanizer

[![CI](https://github.com/AshayK003/VoicePrint/actions/workflows/ci.yml/badge.svg)](https://github.com/AshayK003/VoicePrint/actions/workflows/ci.yml) ![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue) ![tests 345/345](https://img.shields.io/badge/tests-345%2F345-green) ![license MIT](https://img.shields.io/badge/license-MIT-green)

Multi-stage pipeline that rewrites AI-generated text to bypass GPTZero, Turnitin, Originality.ai, and ZeroGPT. Combines heuristic rules, LLM paraphrasing, detection-feedback selection, and style polish in a single pass.

A multi-stage approach is **2.3x more effective** than any single humanization technique. Stage 2 (paraphrasing) is the only API-dependent step — stages 1, 3, and 4 run locally with zero network calls.

---

## Architecture

```
Input (AI text) → Scrub → Paraphrase (LLM) → Restructure → Polish → Detect → Output + metrics
                   │           │                │            │        │
                 stage 1     stage 2          stage 2b     stage 4  stage 3
```

| Stage | File | What it does | Cost |
|-------|------|-------------|------|
| 1. Scrub | `scrub.py` | Replaces 130+ AI-favorite phrases, forces burstiness, injects contractions, breaks tricolons, removes hedging | Pure Python, free |
| 2. Paraphrase | `paraphrase.py` | LLM generates N candidates using escalation prompts. Detection-guided feedback adapts tone per retry. Selects lowest-AI candidate via detection ensemble | API call |
| 2b. Restructure | `restructure.py` | Syntactic clause transformations via spaCy — front/back subclause swaps, appositive extraction, compound splits | spaCy (local) |
| 3. Detect | `detect.py` | 3-tier ensemble: statistical pre-filter (instant) → perplexity (GPT-2) → RoBERTa classifiers. Skips model loading for clear human/AI text | Local ML (~4GB) |
| 4. Polish | `polish.py` | Dysfluency injection, personal narrative, formal→casual conversion, vocabulary variety, passive→active | Pure Python, free |

### Detection decision tree

```
Statistical pre-filter (burstiness + pattern signals)
  ├─ p_ai < 0.20 → clear human → return stat result (no models loaded)
  ├─ p_ai > 0.80 → clear AI    → return stat result (no models loaded)
  └─ 0.20–0.80  → ambiguous    → run GPT-2 perplexity + RoBERTa ensemble
                                   ├─ Binoculars zero-shot (weight 0.30)
                                   └─ 2× RoBERTa classifiers (weight 0.70)
```

### Key design decisions

- **Scrub runs twice** — once before the LLM (removes easy surface-level tells), once after (LLM re-introduces formal transitions). The LLM focuses on structural transformation, not vocabulary swaps.
- **Persona-based prompts, not detection-avoidance** — models are good at "sound like a person explaining this" (in their training data) and bad at "avoid pattern X" (abstract goal). 8 escalation levels, each targeting specific GPTZero signals.
- **Best-of-N with perplexity gate** — generate N candidates, reject perplexity < 30 (too predictable = AI-like), detect the top 3, pick lowest p_ai. Adds 5–15% evasion rate over single-pass.
- **Detection-guided refinement** — previous p_ai is injected into the next paraphrase prompt as feedback. Score < 0.3 → minor polish, 0.3–0.6 → push harder, ≥ 0.6 → rewrite from scratch.
- **PromptMemory** — tracks which prompt levels produce the lowest p_ai per session. Biases future selections toward what worked. No persistence across sessions.
- **Similarity gate at 0.68** — prevents the LLM from drifting too far. Below 0.68, text starts losing critical facts. Fallback: Jaccard word overlap when MiniLM is unavailable.
- **Per-function deterministic RNGs** — each polish/scrub rule seeds a private `random.Random()` from `hashlib.md5(text.encode())`. Same input always produces the same output. No global seed contamination.
- **Skip-on-fail detection** — when a detector raises (OOM, model error), the failed candidate is skipped, not assigned a neutral score. Fallback to similarity ranking when all detections fail.
- **Lazy model loading** — all heavy ML deps (torch, transformers, sentence-transformers) import inside functions, not at module level. `requirements.txt` keeps only core deps for fast cold starts.

---

## Setup

### Prerequisites

- Python 3.12+
- An API key from one supported provider (free tiers work for all)

### Install

```bash
git clone https://github.com/AshayK003/VoicePrint.git
cd VoicePrint
pip install -r requirements.txt
```

Optional — enables model-based detection (RoBERTa, GPT-2 perplexity, MiniLM similarity):

```bash
# Install all at once:
pip install torch transformers sentence-transformers scikit-learn spacy
python -m spacy download en_core_web_sm

# Or let them auto-install on first use — the app prompts with the exact command.
```

### Configure

```bash
cp .env.example .env
```

Set your API key in `.env`. Only one key is needed. The UI auto-detects provider from the key prefix.

### Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Paste AI text, optionally toggle pipeline stages, click **Humanize**. No model downloads on first run — those trigger on first detection use.

### Use as a library

```python
from voiceprint.service import humanize, detect

# Full pipeline
result = humanize("Your AI-generated text here")
print(result.text)
print(f"AI probability: {result.ai_probability:.1%}")
print(f"Similarity: {result.similarity:.1%}")

# Detection-only pre-check
pre = detect("Some text")
print(pre["summary"])
```

---

## Environment variables

### API keys (set at least one)

| Variable | Provider | Rate limit | Get key |
|----------|----------|-----------|---------|
| `OPENCODE_API_KEY` | OpenCode Zen (default) | No observed limits | https://opencode.ai |
| `GEMINI_API_KEY` | Google Gemini | 15 RPM free | https://aistudio.google.com/app/apikey |
| `GROQ_API_KEY` | Groq Cloud | 30 RPM free | https://console.groq.com/keys |
| `MISTRAL_API_KEY` | Mistral AI | 1 req/s free | https://console.mistral.ai/api-keys |
| `OPENAI_API_KEY` | OpenAI | Pay-as-you-go | https://platform.openai.com/api-keys |
| `ANTHROPIC_API_KEY` | Anthropic | Pay-as-you-go | https://console.anthropic.com/ |

Resolution priority: **explicit arg > env var > Windows registry** (`HKCU\Environment\*KEY_NAME*`).

### Pipeline settings

| Variable | Default | Purpose |
|----------|---------|---------|
| `VOICEPRINT_LLM_MODEL` | Provider default | Override the LLM model name |
| `VOICEPRINT_SIMILARITY_THRESHOLD` | `0.68` | Minimum cosine similarity to original |
| `VOICEPRINT_HUMANIZER_MODEL` | `models/humanizer/mistral-7b-humanizer.gguf` | Path to optional GGUF model |

---

## Project structure

```
VoicePrint/
├── app.py                        # Streamlit frontend
├── voiceprint/
│   ├── __init__.py               # Package entry, exports public API
│   ├── service.py                # Entry point: humanize(), detect()
│   ├── pipeline.py               # Orchestrator: wires stages + retry loop
│   ├── scrub.py                  # Stage 1: 50+ heuristic rules
│   ├── paraphrase.py             # Stage 2: LLM client + prompt library
│   ├── restructure.py            # Stage 2b: spaCy clause transformations
│   ├── detect.py                 # Stage 3: 3-tier detection ensemble
│   ├── polish.py                 # Stage 4: 10 style-postprocessing rules
│   ├── config.py                 # Dataclass, provider presets, validation
│   ├── humanizer_model.py        # Optional GGUF model inference
│   ├── metrics.py                # Burstiness, readability scoring
│   ├── patterns.py               # 12 AI-pattern fingerprint signals
│   ├── perplexity.py             # GPT-2 perplexity (lazy-loaded)
│   ├── memory.py                 # Adaptive prompt-level feedback loop
│   ├── similarity.py             # MiniLM / Jaccard similarity gate
│   ├── _text.py                  # Shared sentence splitter (pysbd)
│   └── static/
│       └── style.css
├── tests/                        # 345 tests, all mocked, no network
├── tools/
│   └── analyze_banned_words.py
├── .github/workflows/ci.yml      # Pytest on push/PR (3.11 + 3.12)
├── .streamlit/config.toml
├── pyproject.toml                # Ruff, pytest config
├── .env.example
└── requirements.txt
```

Layering is strict: `app.py → service.py → pipeline.py → modules`. Each layer only imports from the layer below. No circular dependencies.

---

## Local development flow

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Run tests (this is fast — all mocked, no network)
pytest tests/ -v

# 3. Start the app
streamlit run app.py

# 4. Make changes
#    - New scrub/polish rule? Add a function with @rule decorator.
#    - New signal? Add it in patterns.py, update _WEIGHTS and compute_all_signals().
#    - New endpoint? Add it in service.py, wire into pipeline.py.

# 5. Lint before committing
ruff check voiceprint/ tests/
```

### Adding a new scrub rule

```python
# voiceprint/scrub.py
@rule
def lowercase_specific_terms(text: str) -> str:
    """Convert specific AI-favorite capitalized terms to lowercase."""
    return re.sub(r"\bInternet\b", "internet", text)
```

The `@rule` decorator auto-registers it. No other wiring needed. Same pattern for polish rules.

### Adding a new detection signal

1. Add the signal function in `patterns.py` (signature: `text: str → float`, 0–1 range)
2. Add it to `compute_all_signals()`
3. Add its weight to `_WEIGHTS`
4. `pattern_score()` handles the rest

---

## Testing

```bash
# Full suite (345 pass, 2 skipped, ~16s)
pytest tests/ -v

# Single module
pytest tests/test_scrub.py -v

# With coverage
pytest tests/ --cov=voiceprint

# With coverage report
pytest tests/ --cov=voiceprint --cov-report=html
```

### Test strategy

- **No API calls** — all LLM and detection calls mocked via `unittest.mock`
- **No model downloads** — torch, transformers, spaCy, sklearn stubbed in `conftest.py`
- **Perplexity**: mock guard rejects stubs (`type(_MODEL).__module__.startswith("unittest.mock")` returns `None`), so tests never load GPT-2
- **Coverage**: each module has a corresponding `tests/test_*.py`. Tests cover edge cases, error paths, and boundary conditions, not just happy paths.

### Mocking external calls

```python
@patch("voiceprint.paraphrase.litellm.completion")
def test_my_feature(self, mock_completion):
    mock_completion.return_value.choices[0].message.content = "Humanized text"
    result = humanize("Some AI text")
    assert result.success
```

---

## Deployment

### Streamlit Community Cloud (easiest)

1. Push to GitHub
2. Go to https://share.streamlit.io, connect repo, set `app.py` as entry point
3. Add API key in **Settings → Secrets**:
   ```toml
   OPENCODE_API_KEY = "your-key"
   ```
4. Done. Cold start is ~10s — only lightweight core deps are in `requirements.txt`

### Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

### Production notes

- Detection models (RoBERTa + Binoculars) are ~4GB total, cached in `~/.cache/huggingface/`
- GPT-2 for perplexity is ~525MB, same cache
- Rate limiter: 10 `humanize()` calls per 60 seconds per process
- On Streamlit Cloud, set `fastReruns = true` in `.streamlit/config.toml` (done)
- ML models are lazy-loaded: first detection call after deploy triggers download

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| `[Errno 22]` on startup | litellm file cache bug on Windows | Already handled (cache disabled in code) |
| Model takes forever to load | First download of ~4GB HuggingFace models | Wait once. Subsequent loads are cached. |
| "Base URL validation failed" | Custom endpoint without HTTPS | Only HTTP allowed for localhost (ollama, LM Studio) |
| "No API key set" | Missing key in both env and sidebar | Set `OPENCODE_API_KEY` in `.env` or paste key in sidebar |
| API call fails silently | Free tier rate limit hit | Wait 60s or switch providers |
| Similarity = 0.0 | Model download failed, fell back to Jaccard on non-overlapping text | Run `pip install sentence-transformers` or set `use_models=False` |
| Perplexity shows as N/A | Missing `import os` in `perplexity.py` | Update to latest commit (fix in `7cc04d2`) |
| spaCy errors | Model not downloaded | `python -m spacy download en_core_web_sm` |
| Streamlit keeps rerunning | Widget state change triggers full-script rerun | Expected. Results survive in `st.session_state`. |

---

## Contributing

### Principles

- **No paid dependencies.** All libraries must be MIT/Apache-2.0 licensed.
- **No GPU required.** Cloud APIs for LLM, local CPU for detection. Always.
- **Business logic lives in `voiceprint/`**, not in `app.py`. The Streamlit frontend is a thin shell.
- **Test in isolation.** Mock external calls. Never require API keys for tests.
- **Type hints** on all function signatures. Keep modules under 300 lines.

### Workflow

1. Open an issue first — describe what you're changing and why
2. Follow the module structure (one stage = one file)
3. Scrub/polish rules use the `@rule` decorator with `text: str → str` signature
4. Run the full suite before submitting: `pytest tests/ -v` (must pass, 345 tests)
5. Add tests for new functionality. Remove tests only when removing dead code.
6. Run `ruff check voiceprint/ tests/` for lint

### Code of conduct

Be constructive. This is a research-oriented project; questions and experiments are welcome.

---

## License

MIT
