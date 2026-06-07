# VoicePrint — AI Text Humanizer

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

## Architecture: 4-Stage Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT: AI-Generated Text              │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 1: Heuristic Scrub (No Model - Pure Python)     │
│  • Replace AI transition phrases                        │
│  • Force burstiness via 3-1-5 pattern                   │
│  • Inject contractions                                  │
│  • Break tricolons, em-dash overuse                     │
│  Module: voiceprint/scrub.py                            │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2: Adversarial Paraphrasing (Cloud LLM API)     │
│  • Use GPT-4o-mini / Gemini Flash / Groq as paraphraser │
│  • Custom prompt targeting perplexity + burstiness      │
│  • Best-of-N sampling (8 candidates)                    │
│  • Similarity filter (≥ 0.78 cosine)                    │
│  Module: voiceprint/paraphrase.py                       │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 3: AI-Pattern Detection & Scoring                │
│  • RoBERTa-large-openai detector                        │
│  • chatgpt-detector-roberta (Hello-SimpleAI)            │
│  • Binoculars zero-shot metric                          │
│  • Select lowest p_ai candidate                         │
│  Module: voiceprint/detect.py                           │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 4: Local Style Polish (No Model - Pure Python)   │
│  • Convert passive → active voice                       │
│  • Inject rhetorical questions                          │
│  • Add sentence fragments                               │
│  • Final burstiness normalization                       │
│  Module: voiceprint/polish.py                           │
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
├── voiceprint/
│   ├── __init__.py
│   ├── config.py             # API keys, thresholds, model names
│   ├── scrub.py              # Stage 1: Heuristic rule engine
│   ├── paraphrase.py         # Stage 2: LLM API paraphrasing
│   ├── detect.py             # Stage 3: AI-text detection ensemble
│   ├── polish.py             # Stage 4: Style post-processing
│   ├── metrics.py            # Burstiness, perplexity calculations
│   ├── similarity.py         # Semantic similarity gate
│   ├── patterns.py           # AI-pattern fingerprint signals
│   └── pipeline.py           # Main orchestrator (ties all stages)
├── tests/
│   ├── __init__.py
│   ├── test_scrub.py
│   ├── test_metrics.py
│   └── test_pipeline.py
├── app.py                    # Streamlit frontend
└── docs/
    └── techniques.md         # Research notes
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

---

## Key Open-Source Components

### Detectors (Ensemble)
1. **`openai-community/roberta-large-openai-detector`** — MIT, ~1.4GB, primary detector
2. **`Hello-SimpleAI/chatgpt-detector-roberta`** — MIT, ~500MB, secondary detector
3. **Binoculars zero-shot** — Custom implementation (perplexity ratio between two models)

### Similarity Gate
- **`sentence-transformers/all-MiniLM-L6-v2`** — Apache-2.0, ~80MB
- Threshold: cosine similarity ≥ 0.78 between original and humanized text

### LLM Providers (Free Tiers)
- Google Gemini Flash (15 RPM, free)
- Groq Llama 3.3 70B (30 RPM, free)
- Mistral Large (1 req/sec, free)

---

## Implementation Phases

### Phase 1: Heuristic Engine (scrub.py) — NO API COSTS
- Build 50+ rules for AI phrase replacement
- Implement 3-1-5 burstiness pattern
- Contraction injection
- Tricolon breaking
- Em-dash density reduction

### Phase 2: Detection Ensemble (detect.py)
- Load RoBERTa detectors via HuggingFace
- Build ensemble scoring (weighted average)
- Implement Binoculars zero-shot metric
- Create detection report format

### Phase 3: LLM Paraphrasing (paraphrase.py)
- Design adversarial paraphrasing prompt
- Implement best-of-N sampling (N=8)
- Add similarity filtering
- Multi-provider fallback (Gemini → Groq → Mistral)

### Phase 4: Quality Gates (similarity.py, metrics.py)
- Semantic similarity checking (MiniLM)
- Burstiness calculation
- Readability scoring (textstat)
- Reject/retry if quality too low

### Phase 5: Style Polish (polish.py)
- Passive → active voice conversion
- Rhetorical question injection
- Sentence fragment insertion
- Final burstiness normalization

### Phase 6: Pipeline Orchestrator (pipeline.py)
- Wire all stages together
- Configurable pipeline (toggle stages)
- Progress reporting
- Result caching

### Phase 7: Streamlit UI (app.py)
- Text input area
- Pipeline controls (aggressiveness slider)
- Side-by-side comparison view
- Detection score dashboard
- Export functionality

---

## Design Decisions

1. **Why cloud APIs instead of local models?** — No GPU available. Cloud APIs are cheaper for infrequent use and avoid 2GB+ model downloads for the paraphraser.

2. **Why 3-1-5 pattern?** — Research shows human writing follows a rhythm: 3-4 regular sentences, 1 short punchy sentence (3-8 words), 1 long complex sentence (35+ words). This breaks the uniform structural signatures detectors look for.

3. **Why best-of-N instead of single-pass?** — Adversarial Paraphrasing paper (NeurIPS 2025) shows best-of-N adds 5-15% attack success rate for free. Generate 8 candidates, pick the one with lowest detection score.

4. **Why similarity gate at 0.78?** — paniccow/humanizer found this threshold balances evasion vs meaning preservation. Below 0.78, text starts losing critical facts.

5. **Why deterministic scrub BEFORE LLM?** — Removes the easy-to-fix AI tells (transitions, vocabulary) so the LLM can focus on structural transformation instead of surface-level cleanup.

---

## Testing Strategy

- Unit tests for each module (scrub, detect, metrics, similarity)
- Integration test for full pipeline
- Test against known AI-generated texts
- Verify detection scores improve after humanization
- Check semantic similarity stays above threshold

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

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt
python -m nltk.downloader punkt tabulate

# Run Streamlit app
streamlit run app.py

# Run tests
pytest tests/ -v
```

---

## Agent Notes

- When editing `scrub.py`, follow the existing rule pattern: each rule is a function that takes `text: str` and returns `str`
- Detection models are loaded once via `@st.cache_resource` in Streamlit — never reload per request
- LLM calls go through `litellm` for provider-agnostic code
- All API keys go in `.env` file, loaded via `config.py`
- Never commit `.env` or API keys
- Use type hints on all functions
- Keep modules under 300 lines — split if growing larger
