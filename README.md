# VoicePrint — AI Text Humanizer

[![CI](https://github.com/AshayK003/VoicePrint/actions/workflows/ci.yml/badge.svg)](https://github.com/AshayK003/VoicePrint/actions/workflows/ci.yml)

Multi-stage pipeline that transforms AI-generated text into human-like writing that bypasses GPTZero, Turnitin, Originality.ai, and ZeroGPT.

Research shows a multi-stage pipeline is 2.3x more effective than any single technique. VoicePrint combines heuristic rules, LLM paraphrasing, detection-feedback selection, and style polish in a single pass.

## How it works

```
Input (AI text) → Scrub → Paraphrase (LLM) → Polish → Detect → Output + scores
                    |          |               |         |
                    stage 1    stage 2         stage 4   stage 3
```

| Stage | What | Cost |
|-------|------|------|
| 1. Heuristic Scrub | Replaces AI phrases, forces burstiness, injects contractions, breaks tricolons, removes hedging | Free (pure Python) |
| 2. Adversarial Paraphrasing | LLM generates N candidates with persona-based prompts (explain-to-friend, blog, Reddit). Detection-guided feedback escalates tone per retry | API call |
| 3. Detection Scoring | Statistical pre-filter + perplexity scoring + optional RoBERTa ensemble. Selects lowest-AI candidate | Free (local models) |
| 4. Style Polish | Dysfluency injection, personal narrative, formal→casual conversion, vocabulary variety, passive→active | Free (pure Python) |

Stage 2 is the only API-dependent step. Stages 1, 3, and 4 run locally with no network calls.

### Detection tiers

- **Statistical pre-filter** (instant) — burstiness + pattern signals. Skips model loading for clear human/AI text.
- **Perplexity scoring** (fast, ~500MB GPT-2 cached) — measures how predictable text is. Human-like text: ~26 perplexity, AI-generated: ~138 perplexity. Normalized to 0-1 score.
- **RoBERTa ensemble** (heavy, ~1.9GB) — two fine-tuned classifiers in parallel. Only loads when statistical pre-filter returns ambiguous.

### Design decisions

- **Deterministic scrub before LLM paraphrase** — removes easy-to-fix surface-level AI tells so the LLM focuses on structural transformation instead.
- **Persona-based prompts** (not detection-avoidance) — models are good at "sound like a person explaining this" (training data) and bad at "avoid pattern X" (abstract goal). Three escalation levels: conversational, blog-style, casual/raw.
- **Best-of-N sampling** — generate N candidates, reject those with perplexity < 30 (too predictable = AI-like), pick the lowest detection score. Adds 5–15% evasion rate.
- **Detection-guided iterative refinement** — prev_p_ai feedback injected into next paraphrase prompt. Score < 0.3 → minor polish, 0.3-0.6 → push harder, ≥ 0.6 → rewrite from scratch.
- **PromptMemory adaptive feedback loop** — tracks which prompt levels produce lowest p_ai per session. Biases future selections toward what worked.
- **Post-paraphrase scrub** — LLM tends to re-introduce AI patterns (formal transitions, hedging). Scrub runs again after paraphrasing.
- **Iterative retry loop** — if detection doesn't pass (p_ai < 0.5), pipeline retries up to N times, tracking the best result (tiebreak on higher perplexity).
- **Similarity gate at 0.68** — prevents the LLM from drifting too far from original meaning. Below 0.68, text starts losing critical facts.
- **API key resolution** — checks env var, then config, then sidebar input. No side effects on `os.environ`.
- **Dedicated per-function RNGs (polish.py)** — each rule seeds a private `random.Random()` from `hashlib.md5(text.encode())`. Same text always produces same output. No global seed contamination across tests.
- **Shared `sentences()` utility** — 12 regex copies unified into `voiceprint/_text.py`. Single source of truth for sentence boundary splitting.
- **Lazy torch/transformers imports in detect.py** — no import-time crash on Windows when torch DLL is unloadable.

## Setup

### Prerequisites

- Python 3.12+
- Windows, macOS, or Linux
- An API key from one supported provider (free tiers work)

### Install

```bash
git clone <repo-url> && cd VoicePrint
pip install -r requirements.txt
python -m nltk.downloader punkt tabulate
```

### Configure

```bash
cp .env.example .env
```

Edit `.env` and set the API key for your chosen provider. Only one key is needed.

### Run

```bash
streamlit run app.py
```

Opens at http://localhost:8501. Paste AI text, optionally toggle pipeline stages, click **Humanize**.

### CLI usage (via service layer)

```python
from voiceprint.service import humanize, detect

result = humanize("Your AI-generated text here")
print(result.text)
print(f"AI probability: {result.ai_probability:.1%}")
print(f"Similarity: {result.similarity:.1%}")
print(f"Perplexity: {result.perplexity:.1f}")

# Detection-only pre-check
pre = detect("Some text")
print(pre["summary"])
```

## Environment variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `OPENCODE_API_KEY` | For OpenCode Zen | — | OpenCode AI key (default provider) |
| `GEMINI_API_KEY` | For Gemini provider | — | Google AI Studio key (free) |
| `GROQ_API_KEY` | For Groq provider | — | Groq Cloud key (free) |
| `MISTRAL_API_KEY` | For Mistral provider | — | Mistral AI key (free) |
| `OPENAI_API_KEY` | For OpenAI provider | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | For Anthropic provider | — | Anthropic API key |
| `VOICEPRINT_LLM_MODEL` | No | Provider default | Override the LLM model |
| `VOICEPRINT_SIMILARITY_THRESHOLD` | No | `0.68` | Min cosine similarity to original |
| `VOICEPRINT_HUMANIZER_MODEL` | No | `models/humanizer/mistral-7b-humanizer.gguf` | Path to GGUF model file |

At least one API key must be set (either via env var, sidebar, or Windows registry). The UI auto-detects provider from the key prefix. OpenCode Zen keys (79 chars, `sk-`) are auto-detected by length > 60.

API key resolution priority: **explicit arg > env var > Windows registry** (`HKCU\Environment\OPENCODE_API_KEY`).

### Free provider key links

- **OpenCode Zen** — https://opencode.ai (default, no rate limits observed on free tier)
- **Gemini** — https://aistudio.google.com/app/apikey (15 RPM free)
- **Groq** — https://console.groq.com/keys (30 RPM free)
- **Mistral** — https://console.mistral.ai/api-keys (1 req/s free)

## Project structure

```
VoicePrint/
├── app.py                     # Streamlit frontend
├── voiceprint/
│   ├── __init__.py            # Package entry, exports public API
│   ├── service.py             # Entry point: humanize(), detect()
│   ├── pipeline.py            # Orchestrator: wires all stages + retry loop + PromptMemory
│   ├── scrub.py               # Stage 1: heuristic rule engine (50+ AI-pattern rules)
│   ├── paraphrase.py          # Stage 2: LLM API client + NINJA_PROMPTS + candidate selection
│   ├── detect.py              # Stage 3: detection ensemble (stat + Binoculars + RoBERTa + perplexity)
│   ├── polish.py              # Stage 4: style post-processing (dysfluency, narrative, etc.)
│   ├── restructure.py         # Stage 2b: syntactic clause restructuring via spaCy
│   ├── config.py              # Config dataclass, provider presets, env/registry, validation
│   ├── humanizer_model.py     # Phase 2: fine-tuned GGUF model inference (optional)
│   ├── metrics.py             # Burstiness, readability scoring
│   ├── patterns.py            # AI-pattern fingerprint signals (15+ signals, optional pystylometry)
│   ├── perplexity.py          # GPT-2 based perplexity scoring (lazy-loaded, 0-1 normalized)
│   ├── memory.py              # PromptMemory — adaptive prompt_level feedback loop
│   ├── similarity.py          # Semantic similarity (MiniLM / Jaccard fallback)
│   ├── _text.py               # Shared text utilities (sentences() splitter)
│   └── static/
│       └── style.css          # App stylesheet
├── .github/
│   └── workflows/
│       └── ci.yml                # CI: pytest on push/PR (3.11 + 3.12), Ruff lint
├── pyproject.toml                 # Project config (Ruff, pytest)
├── tests/                     # 345 tests, all mocked (no API calls, no model downloads)
├── tools/
│   └── analyze_banned_words.py  # Dataset-based banned word analysis (gsingh1-py/train)
├── .env.example               # Environment variable template
└── requirements.txt
```

Layering is strict: `app.py` → `service.py` → `pipeline.py` → individual modules. Each layer only imports from the layer below. No circular dependencies.

## Testing

```bash
# Run all tests (345 total, ~20s)
pytest tests/ -v

# Run a specific module
pytest tests/test_scrub.py -v

# Run with coverage
pytest tests/ --cov=voiceprint
```

### Test strategy

- No API calls — all LLM and detection calls are mocked via `unittest.mock`
- No model downloads — `torch`, `transformers`, `sentence_transformers`, `spaCy`, and `sklearn` are stubbed in `tests/conftest.py`
- Tests cover: each scrub rule, pattern signal, paraphrase candidate flow, perplexity scoring, PromptMemory, pipeline retry logic, restructure rules, service validation, rate limiting, URL validation, XSS safety, and config edge cases
- Perplexity tests skip model-dependent assertions when GPT-2 is unavailable; mock guard rejects stubs via `type(_MODEL).__module__`

### Adding tests

Each test class mirrors a production module. Mock at the boundary where your module calls an external dependency:

```python
@patch("voiceprint.paraphrase.litellm.completion")
def test_my_feature(self, mock_completion):
    mock_completion.return_value = fake_response()
```

## Deployment

### Streamlit Community Cloud

1. Push to GitHub
2. Go to https://share.streamlit.io
3. Connect the repo, set `app.py` as entry point
4. Add API keys as **Secrets** (Streamlit Cloud → Settings → Secrets)

```toml
# .streamlit/secrets.toml
OPENCODE_API_KEY = "your-key"
```

### Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt && python -m nltk.downloader punkt tabulate
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

### Notes

- Detection models (RoBERTa, Binoculars) are ~4GB total and download on first use. They are cached in HuggingFace's default cache dir (`~/.cache/huggingface/`).
- GPT-2 for perplexity scoring (~525MB) is cached the same way, lazy-loaded on first call.
- Streamlit's `@st.cache_resource` prevents reloading models on every rerun.
- The rate limiter allows 10 `humanize()` calls per 60 seconds per process.

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| `[Errno 22]` on startup | litellm file operation bug on Windows | Already handled in code (litellm cache disabled) |
| Model takes forever to load | First download of ~4GB of model weights | Wait once. Subsequent loads are cached. |
| "Base URL validation failed" | Custom endpoint without HTTPS | Only HTTP allowed for localhost (ollama, LM Studio) |
| "No API key set" | Neither env var nor sidebar input | Set `OPENCODE_API_KEY` in `.env` or paste key in sidebar |
| API call fails silently | Free tier rate limit hit | Wait 60s or switch providers |
| Similarity score is 0.0 | Model failed to load, fell back to Jaccard on non-overlapping text | Check model download or set `use_models=False` for Jaccard-only mode |

## Contributing

### Principles

- **No paid dependencies** — all libraries must be MIT/Apache-2.0 licensed
- **No GPU required** — cloud APIs only for LLM, local CPU for detection
- **Business logic lives in `voiceprint/`**, not in `app.py`
- **Test in isolation** — mock external calls, never require API keys for tests
- **Type hints** on all function signatures

### Guidelines

1. Open an issue before starting work
2. Follow the existing module structure — each stage is one file
3. Keep modules focused; split only when a clear maintenance boundary emerges
4. Scrub rules follow the decorator pattern: `@rule` with `text: str → str` signature
5. Run `pytest tests/ -v` before submitting — all 345 tests must pass
6. Add tests for new functionality; remove tests only when removing dead code

### Code of conduct

Be constructive. This is a research-oriented project; questions and experiments are welcome.

## Support

<a href="https://chai4.me/darkcharon3301" target="_blank" title="Support darkcharon3301 on Chai4Me" style="display:inline-flex;flex-direction:column;align-items:center;justify-content:center;background:#ffffff;padding:8px 32px;border-radius:16px;text-decoration:none;border:1px solid #e5e7eb;box-shadow:0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.05);"><img src="https://chai4.me/icons/wordmark.png" alt="Chai4Me" style="height:32px;object-fit:contain;"/></a>

## License

MIT
