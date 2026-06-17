"""Stage 2: Adversarial Paraphrasing — LLM-based text rewriting.

Uses cloud LLM APIs (via litellm) to generate N candidate paraphrases,
then selects the best one based on detection scores and similarity.
"""

from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

# litellm is imported lazily inside functions that use it (~1.5s import)
# to avoid slowing down startup. Configure once via _ensure_litellm().

_LITELLM_READY = False


def _ensure_litellm():
    global _LITELLM_READY
    if _LITELLM_READY:
        return
    import litellm
    litellm.cache = None
    litellm.success_callback = []
    litellm.failure_callback = []
    litellm.set_verbose = False
    os.environ.setdefault("LITELLM_LOG", "WARNING")
    os.environ.setdefault("OPENAI_LOG", "WARN")
    _LITELLM_READY = True


from .config import Config, load_config
from .similarity import check_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trained model integration
# ---------------------------------------------------------------------------

_HUMANIZER_CACHE: dict[str, object] = {}


def generate_trained_candidate(
    text: str,
    model_path: str | None = None,
    temperature: float = 1.0,
) -> str | None:
    """Generate a candidate using the trained HumanizerModel.

    Falls back gracefully if the model isn't available or fails to load.
    Returns None on failure so callers can fall back to LLM API.

    The model is cached globally after first load.
    """
    global _HUMANIZER_CACHE
    cache_key = model_path or "default"

    if cache_key not in _HUMANIZER_CACHE:
        try:
            from .humanizer_model import HumanizerModel
            _HUMANIZER_CACHE[cache_key] = HumanizerModel(model_path)
        except (FileNotFoundError, ImportError, OSError) as e:
            logger.debug(f"Trained model not available: {e}")
            _HUMANIZER_CACHE[cache_key] = None

    model = _HUMANIZER_CACHE.get(cache_key)
    if model is None:
        return None

    try:
        return model.humanize(text, temperature=temperature)
    except Exception as e:
        logger.warning(f"Trained model candidate failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Error sanitization — redact API keys from log messages
# ---------------------------------------------------------------------------

_API_KEY_PATTERNS = [
    r"(?i)(api[_-]?key|apikey)\s*['\"]?[=:]\s*['\"]?(sk-[A-Za-z0-9]{10,})",
    r"(?i)(AIza[0-9A-Za-z_-]{35})",       # Google Gemini
    r"(?i)(gsk_[A-Za-z0-9]{30,})",         # Groq
    r"(?i)(ak-[A-Za-z0-9]{20,})",          # Mistral
    r"(?i)(sk-ant-[A-Za-z0-9]{20,})",      # Anthropic
    r"(?i)(sk-[A-Za-z0-9]{20,})",          # OpenAI
    r"(?i)(Authorization:\s*Bearer\s+\S+)",
]



# Wide temperature range for maximum candidate diversity
_CANDIDATE_TEMPERATURES = [0.5, 0.8, 1.0, 1.2, 1.4, 1.1, 0.9, 1.3]


def _sanitize_error(msg: str) -> str:
    """Redact API keys and bearer tokens from error messages before logging."""
    result = msg
    for pattern in _API_KEY_PATTERNS:
        result = re.sub(pattern, "[API KEY REDACTED]", result)
    return result


# ---------------------------------------------------------------------------
# Paraphrasing prompt
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Ninja-mode progressive paraphrase prompts
# Eight escalating levels: 0-7 targeting increasing detection evasion
# The pipeline cycles through levels across retry iterations.
# Each level targets specific GPTZero detection signals.
# ---------------------------------------------------------------------------

NINJA_PROMPTS = [
    # Level 0 — conversational rewrite: sound like a real person
    """\
Rewrite the following text in a natural, conversational voice. Make it sound like someone explaining this to a friend — not writing an essay.

- Use natural contractions and casual phrasing throughout
- Vary sentence lengths: mix short punchy sentences with longer flowing ones
- Keep the same facts and meaning — don't add or remove information
- Don't sound formal, academic, or like a marketing brochure

Text:
{text}""",

    # Level 1 — direct and personal: confident but natural
    """\
Rewrite this text to sound like someone with real experience on the topic. Warm, direct, personal — like a knowledgeable colleague explaining something clearly.

- Write as if you're talking to one person, not an audience
- Use "you" naturally where it fits
- Mix up sentence openings — don't start every sentence the same way
- Keep every fact accurate. Don't invent anything.
- Drop the formal tone entirely. This should read like someone's honest take.

Here's the text:
{text}""",

    # Level 2 — raw and unfiltered: like a smart person's casual take
    """\
Rewrite this to sound completely natural — like a thoughtful person sharing their perspective in a conversation, not writing for publication.

- Write how people actually talk: not perfectly polished, but clear
- Use one or two natural asides or tangents if they fit
- Vary rhythm: short sentences land harder when surrounded by longer ones
- Avoid corporate or academic language completely
- Keep all facts and meaning exactly as they are
- This should read like someone's honest, unfiltered take

Text:
{text}""",

    # Level 3 — anti-mechanical precision: break GPTZero's "Mechanical Precision" signal
    # GPTZero penalizes uniform sentence structure and perfect grammar.
    # This level deliberately varies writing quality across sentences.
    """\
Rewrite this text so it reads like someone writing quickly — not perfectly polished, but clear and natural. The key is to make each sentence feel independently written, not like they all came from the same template.

Vary the writing quality across sentences. Some sentences should feel carefully thought out. Others should feel more like quick thoughts. Mix short fragments with longer, more complex sentences. Let the rhythm feel organic, not mechanical.

Keep every fact accurate. Don't add or remove information. The goal is to make this read like someone's authentic writing, not like a polished article.

Text:
{text}""",

    # Level 4 — anti-robotic formality: opinionated and conversational
    # GPTZero flags writing without personality or stance as "Robotic Formality"
    # Inject opinion, take a side, use natural conviction
    """\
Rewrite this so it sounds like someone who genuinely knows this stuff sharing their honest perspective. Take a stance where appropriate — AI-generated text tends to stay neutral on everything, which is actually a dead giveaway.

Use natural conviction. If something is clearly good or bad, say so. If there's debate around the topic, acknowledge it with a real human perspective. But don't force opinions where they don't belong — keep it authentic.

Use natural conversational rhythms: rhetorical questions, light asides, the occasional parenthetical thought. Write like a smart person talking, not writing for publication. Keep all facts accurate.

Text:
{text}""",

    # Level 5 — anti-sophisticated clarity: break GPTZero's "Sophisticated Clarity"
    # GPTZero flags overly clean, perfectly clear reasoning as AI.
    # This level deliberately roughens the presentation — not incorrect, just not pristine.
    """\
Rewrite this text to read like someone with real, hands-on experience — the kind of person who knows their stuff but isn't trying to impress anyone with how well they write.

The writing should feel direct and a little rough around the edges. Use punchy sentences. Lead with the most interesting point, not a careful transition. Drop any unnecessary setup or context. If something is obvious, say it plainly instead of building up to it.

The tone should suggest someone who's been around the block on this topic — confident enough to be casual, direct enough to skip the formalities. Keep every fact accurate, nothing invented.

Text:
{text}""",

    # Level 6 — anti-pattern: break every structural pattern GPTZero looks for
    # GPTZero detects consistent patterns in sentence structure, opening variety, and paragraph flow
    # This level deliberately varies the writing style sentence by sentence
    """\
Rewrite this text so that no two consecutive sentences follow the same structural pattern. The goal is to make the writing feel genuinely organic — like you're reading something a person actually wrote, not text that follows consistent formatting rules.

Each sentence should feel like it was written independently. Vary how sentences start from one to the next. Some should begin with subjects, others with verbs, others with prepositional phrases, others with conjunctions. Vary the length dramatically — some sentences under 5 words, some over 30.

The overall effect should feel natural, not forced. Don't draw attention to the variation — it should just read like normal human writing where no two sentences are mechanically identical.

Preserve all facts. Don't add or remove information.

Text:
{text}""",

    # Level 7 — maximum evasion: aggressive humanization for stubbornly AI-detected text
    # Used when all other levels fail. Simulates a dictation-style, highly personal voice.
    """\
Rewrite this text completely from scratch. Imagine you're explaining this to a friend over coffee — you're knowledgeable about it, but you're not delivering a prepared speech. You're thinking as you speak.

Use a dictation-like quality: sentences run into each other naturally, some trail off, some restart. Use contractions heavily. Throw in an occasional self-correction ("actually that's not quite right — "). Use one or two natural digressions. 

The most important thing: this should sound unmistakably like a human being wrote it. Not a writer being casual for effect — just a normal person sharing what they know. Keep it genuine. Don't overdo the casual affectations. Just write like a real person talks.

Keep every fact intact. Don't invent anything.

Text:
{text}""",
]

# Backward-compatible alias
PARAPHRASE_PROMPT = NINJA_PROMPTS[0]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _validate_base_url(url: str) -> str:
    """Validate and normalize base URL. Rejects non-HTTPS URLs unless localhost.
    Raises ValueError on invalid URLs.
    """
    url = url.strip()
    if not url:
        return url
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(
            f"Invalid base URL '{url}'. Must include scheme and host, "
            f"e.g. https://api.openai.com/v1"
        )
    if parsed.scheme == "http":
        host = parsed.hostname or ""
        # Allow HTTP only for localhost/loopback (e.g. local LM proxy)
        if host not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError(
                f"HTTP base URL '{url}' is not allowed. "
                f"Use HTTPS, or for local proxies use localhost."
            )
    elif parsed.scheme != "https":
        raise ValueError(f"Unsupported scheme '{parsed.scheme}' in base URL. Use https://.")
    return url


def test_llm_connection(config: Config, timeout: int = 15) -> dict:
    """Test LLM connectivity. Returns dict with 'connected' bool and optional 'error'.

    Makes a minimal 1-token completion call. No streaming, no side effects.
    """
    _ensure_litellm()

    try:
        kwargs = _litellm_kwargs(config, temperature=0.5)
        kwargs["messages"][0]["content"] = "Say hi"
        kwargs["max_tokens"] = 2
        kwargs["timeout"] = timeout

        import litellm
        resp = litellm.completion(**kwargs)
        if resp and resp.choices and len(resp.choices) > 0:
            return {"connected": True, "error": None}
        return {"connected": False, "error": "No response from API"}
    except Exception as e:
        return {"connected": False, "error": _sanitize_error(str(e))}


def _litellm_kwargs(config: Config, temperature: float) -> dict:
    """Build kwargs dict for litellm.completion, passing provider config."""
    model = config.llm_model.strip() if config.llm_model else ""
    if not model:
        raise ValueError("Model name is empty. Select a model in the sidebar.")

    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": ""}],
        "temperature": temperature,
        "max_tokens": config.llm_max_tokens,
    }

    # API key — strip whitespace; litellm expects it via api_key param or env var
    if config.api_key:
        kwargs["api_key"] = config.api_key.strip()

    # Base URL — validate before passing to litellm (SSRF prevention)
    if config.base_url:
        kwargs["base_url"] = _validate_base_url(config.base_url)

    return kwargs


def generate_candidate(
    text: str,
    config: Config | None = None,
    temperature: float | None = None,
    prompt_level: int = 0,
    prev_p_ai: float | None = None,
) -> str:
    """Generate a single paraphrase candidate via LLM API.

    Args:
        prompt_level: Index into NINJA_PROMPTS (0=standard, 1=aggressive, 2=extreme).
            Clamped to valid range.
        prev_p_ai: Previous detection score (0-1). When provided, a feedback
            line is injected into the prompt for detection-guided refinement.
    """
    config = config or load_config()
    temp = temperature if temperature is not None else config.llm_temperature

    prompt_idx = max(0, min(prompt_level, len(NINJA_PROMPTS) - 1))
    prompt_template = NINJA_PROMPTS[prompt_idx]

    # Detection-guided refinement: inject previous score into prompt
    # The thresholds target GPTZero's stricter detection (p_ai > 0.2 is flagged)
    feedback = ""
    if prev_p_ai is not None:
        if prev_p_ai < 0.3:
            feedback = f"\nNOTE: Previous attempt scored {prev_p_ai:.2f} — almost there but still shows subtle AI patterns. Vary sentence openings and add more natural rhythm.\n"
        elif prev_p_ai < 0.5:
            feedback = f"\nNOTE: Previous attempt scored {prev_p_ai:.2f} — still too clean. Make sentences less uniform. Add some personality. Vary the writing quality between sentences.\n"
        else:
            feedback = f"\nNOTE: Previous attempt scored {prev_p_ai:.2f} — clearly AI-detected. Rewrite from scratch. Break every obvious pattern. Each sentence should feel independently written, with different lengths and structures. Use contractions. Drop transitions. Write like someone thinking out loud, not composing text.\n"

    content = prompt_template.format(text=text)
    if feedback:
        content = content.rstrip() + feedback

    _ensure_litellm()
    kwargs = _litellm_kwargs(config, temp)
    kwargs["messages"] = [
        {"role": "user", "content": content}
    ]

    import litellm
    response = litellm.completion(**kwargs)
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("LLM returned empty response")
    return content.strip()


def generate_candidates(
    text: str,
    n: int | None = None,
    config: Config | None = None,
    prompt_level: int = 0,
    prev_p_ai: float | None = None,
) -> list[str]:
    """Generate N paraphrase candidates in parallel with varied temperatures.

    Args:
        prompt_level: Index into NINJA_PROMPTS passed to each candidate.
        prev_p_ai: Previous detection score for detection-guided refinement.
    """
    config = config or load_config()
    n = n or config.n_candidates

    temperatures = [_CANDIDATE_TEMPERATURES[i % len(_CANDIDATE_TEMPERATURES)] for i in range(n)]

    candidates: list[str] = []

    def _generate_one(idx: int, temp: float) -> str | None:
        try:
            return generate_candidate(
                text, config, temperature=temp, prompt_level=prompt_level,
                prev_p_ai=prev_p_ai,
            )
        except Exception as e:
            logger.warning(f"Candidate {idx+1} failed: {_sanitize_error(str(e))}")
            return None

    # Run all candidates in parallel
    with ThreadPoolExecutor(max_workers=min(n, 4)) as executor:
        futures = {
            executor.submit(_generate_one, i, t): i
            for i, t in enumerate(temperatures)
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                candidates.append(result)

    return candidates


def select_best(
    original: str,
    candidates: list[str],
    config: Config | None = None,
    detector=None,
    min_sim: float | None = None,
    use_trained_model: bool = False,
    trained_model_path: str | None = None,
) -> tuple[str, float]:
    """Select the best candidate by detection score, not just similarity.

    Optimization: pre-filter by similarity (cheap), then detect only top candidates (expensive).
    This reduces detection calls from N to min(N, 3).

    When use_trained_model=True, also generates a candidate from the trained
    HumanizerModel and includes it in the pool. Falls back gracefully if the
    model isn't available.

    Args:
        detector: Optional DetectorEnsemble instance. If None, creates one.
        min_sim: Minimum similarity gate for candidate pool. Derived from
            config.similarity_threshold if None (typically threshold - 0.13).
        use_trained_model: Whether to include a trained model candidate.
        trained_model_path: Path to trained model. None = use default.

    Returns (best_candidate, similarity_score).
    """
    config = config or load_config()

    if not candidates:
        return original, 1.0

    # Include trained model candidate if available
    if use_trained_model:
        trained_candidate = generate_trained_candidate(
            original, model_path=trained_model_path
        )
        if trained_candidate and trained_candidate.strip():
            # De-duplicate against existing candidates
            is_duplicate = any(
                c.strip().lower() == trained_candidate.strip().lower()
                for c in candidates
            )
            if not is_duplicate:
                candidates = candidates + [trained_candidate]
                logger.debug("Added trained model candidate to pool")

    if min_sim is None:
        min_sim = max(0.5, config.similarity_threshold - 0.13)

    # Step 1: Compute all similarity scores (cheap — no models)
    sim_map = {}
    for c in candidates:
        sim_map[c] = check_similarity(original, c, config=config)

    # Step 2: Filter by minimum similarity, sort by sim descending
    scored = [(c, sim_map[c]) for c in candidates if sim_map[c] >= min_sim]
    if not scored:
        scored = [(c, sim_map[c]) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Step 2b: Reject candidates with perplexity below threshold (too predictable = AI-like)
    try:
        from .perplexity import raw_perplexity as _raw_ppl
        filtered = []
        for c, s in scored:
            try:
                ppl = _raw_ppl(c)
                if ppl is None or ppl >= 30.0:
                    filtered.append((c, s))
            except Exception:
                filtered.append((c, s))
        scored = filtered
    except Exception:
        pass
    if not scored:
        scored = [(c, sim_map[c]) for c in candidates] if candidates else []

    if not scored:
        return original, 1.0

    # Step 3: Run detection ONLY on top candidates (expensive — model inference)
    top_n = min(len(scored), 3)
    top_candidates = [c for c, _ in scored[:top_n]]

    if detector is None:
        from .detect import DetectorEnsemble
        detector = DetectorEnsemble(config)

    best = None
    best_pai = 1.0
    best_sim = 0.0

    for candidate in top_candidates:
        sim = sim_map[candidate]
        try:
            detection = detector.detect(candidate)
            p_ai = detection.p_ai
        except Exception:
            p_ai = 0.5

        effective_pai = p_ai

        if effective_pai < best_pai:
            best = candidate
            best_pai = p_ai  # Store real p_ai (not effective) for display
            best_sim = sim

    # Fallback: if detection failed on all, pick highest similarity
    if best is None:
        if scored:
            best = scored[0][0]
            best_sim = scored[0][1]
        else:
            best, best_sim = candidates[0], sim_map.get(candidates[0], 1.0)

    return best, best_sim



