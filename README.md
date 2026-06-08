# VoicePrint — AI Text Humanizer

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
| 1. Heuristic Scrub | Replaces AI phrases, forces burstiness, injects contractions | Free (pure Python) |
| 2. Adversarial Paraphrasing | LLM generates N candidates with detection-avoidance prompt | API call |
| 3. Detection Scoring | Ensembles RoBERTa + Binoculars detectors; picks lowest-AI candidate | Free (local models) |
| 4. Style Polish | Passive→active voice, rhetorical questions, sentence fragments | Free (pure Python) |

Stage 2 is the only API-dependent step. Stages 1, 3, and 4 run locally with no network calls.

### Detection tiers

- **Statistical pre-filter** (instant) — burstiness + pattern signals. Skips model loading for clear human/AI text.
- **Binoculars zero-shot** (fast, ~2GB) — perplexity ratio between GPT-2 and GPT-2-medium.
- **RoBERTa ensemble** (heavy, ~1.9GB) — two fine-tuned classifiers in parallel.

Models only load when the statistical pre-filter returns ambiguous (20–80%). Most AI text scores >70% on statistics alone, so models are often skipped entirely.

### Design decisions

- **Deterministic scrub before LLM paraphrase** — removes easy-to-fix surface-level AI tells so the LLM focuses on structural transformation instead.
- **Best-of-N sampling** — generate 8 candidates, pick the one with lowest detection score. Adds 5–15% evasion rate for free.
- **Post-paraphrase scrub** — LLM tends to re-introduce AI patterns (formal transitions, hedging). Scrub runs again after paraphrasing.
- **Iterative retry loop** — if detection doesn't pass (p_ai < 0.5), pipeline retries up to N times, tracking the best result.
- **Similarity gate at 0.68** — prevents the LLM from drifting too far from original meaning. Below 0.68, text starts losing critical facts.

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
from voiceprint.config import Config, load_config

config = load_config()
config.api_key = "your-key"  # or set env var
config.provider = "Google Gemini (Free)"

result = humanize("Your AI-generated text here")
print(result.text)
print(f"AI probability: {result.ai_probability:.1%}")
print(f"Similarity: {result.similarity:.1%}")

# Detection-only pre-check
pre = detect("Some text")
print(pre["summary"])
```

## Environment variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `GEMINI_API_KEY` | For Gemini provider | — | Google AI Studio key (free) |
| `GROQ_API_KEY` | For Groq provider | — | Groq Cloud key (free) |
| `MISTRAL_API_KEY` | For Mistral provider | — | Mistral AI key (free) |
| `OPENAI_API_KEY` | For OpenAI provider | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | For Anthropic provider | — | Anthropic API key |
| `OPENCODE_API_KEY` | For OpenCode Zen | — | OpenCode AI key |
| `VOICEPRINT_LLM_MODEL` | No | Provider default | Override the LLM model |
| `VOICEPRINT_SIMILARITY_THRESHOLD` | No | `0.68` | Min cosine similarity to original |

At least one API key must be set (either via env var or the sidebar). The UI auto-detects provider from the key prefix.

### Free provider key links

- **Gemini** — https://aistudio.google.com/app/apikey (15 RPM free)
- **Groq** — https://console.groq.com/keys (30 RPM free)
- **Mistral** — https://console.mistral.ai/api-keys (1 req/s free)

## Project structure

```
VoicePrint/
├── app.py                  # Streamlit frontend
├── voiceprint/
│   ├── service.py          # Entry point: humanize(), detect()
│   ├── pipeline.py         # Orchestrator: wires all 4 stages
│   ├── scrub.py            # Stage 1: heuristic rule engine
│   ├── paraphrase.py       # Stage 2: LLM API client + candidate selection
│   ├── detect.py           # Stage 3: detection ensemble (stat + Binoculars + RoBERTa)
│   ├── polish.py           # Stage 4: style post-processing
│   ├── config.py           # Config dataclass, provider presets, validation
│   ├── metrics.py          # Burstiness, readability scoring
│   ├── patterns.py         # AI-pattern fingerprint signals
│   └── similarity.py       # Semantic similarity (MiniLM / Jaccard fallback)
├── tests/                  # 255+ tests, all mocked (no API calls, no model downloads)
├── .env.example            # Environment variable template
└── requirements.txt
```

Layering is strict: `app.py` → `service.py` → `pipeline.py` → individual modules. Each layer only imports from the layer below. No circular dependencies.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run a specific module
pytest tests/test_scrub.py -v

# Run with coverage
pytest tests/ --cov=voiceprint
```

### Test strategy

- No API calls — all LLM and detection calls are mocked via `unittest.mock`
- No model downloads — `torch`, `transformers`, `sentence_transformers`, and `sklearn` are stubbed in `tests/conftest.py`
- Tests cover: each scrub rule, pattern signal, paraphrase candidate flow, pipeline retry logic, service validation, rate limiting, URL validation, XSS safety, and config edge cases
- Dead code branches (`use_models=False`, `_run_roberta`) are tested through their callers, not directly

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
GEMINI_API_KEY = "your-key"
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
- Streamlit's `@st.cache_resource` prevents reloading models on every rerun.
- The rate limiter allows 10 `humanize()` calls per 60 seconds per process.

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| `[Errno 22]` on startup | litellm file operation bug on Windows | Already handled in code (litellm cache disabled) |
| Model takes forever to load | First download of ~4GB of model weights | Wait once. Subsequent loads are cached. |
| "Base URL validation failed" | Custom endpoint without HTTPS | Only HTTP allowed for localhost (ollama, LM Studio) |
| "No API key set" | Neither env var nor sidebar input | Set `GEMINI_API_KEY` in `.env` or paste key in sidebar |
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
3. Keep modules under ~350 lines; split if growing larger
4. Scrub rules follow the decorator pattern: `@rule` with `text: str → str` signature
5. Run `pytest tests/ -v` before submitting — all 255+ tests must pass
6. Add tests for new functionality; remove tests only when removing dead code

### Code of conduct

Be constructive. This is a research-oriented project; questions and experiments are welcome.

## License

MIT
