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

import litellm

# Disable litellm file operations (avoids [Errno 22] on Windows)
litellm.cache = None
litellm.success_callback = []
litellm.failure_callback = []
litellm.set_verbose = False
# Suppress litellm's own HTTP request logging (avoids spilling API keys to stdout)
os.environ.setdefault("LITELLM_LOG", "WARNING")
os.environ.setdefault("OPENAI_LOG", "WARN")

from .config import Config, load_config
from .similarity import check_similarity

logger = logging.getLogger(__name__)


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

# Deliberately lower than config.similarity_threshold (0.68). This gates
# candidate pool size, not final quality — higher values reject too many
# valid candidates before detection can evaluate them.
_MIN_CANDIDATE_SIM = 0.55

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

PARAPHRASE_PROMPT = """\
You are a human editor rewriting AI-generated text to pass as human-written.

DETECTION AVOIDANCE — these patterns get flagged instantly:
- Uniform sentence length (all 15-25 words) → break this
- Starting multiple sentences with "The", "This", "It", "In" → vary openers
- Perfect parallel structure → break it unevenly
- Modal verbs stacked (could, would, should) → remove most
- No contractions → use them naturally
- Transition words at sentence starts (Furthermore, Moreover, However) → cut or rephrase
- Abstract subjects (It is important, There are) → use concrete subjects
- Clean, symmetrical paragraph structure → make it messy

REWRITE RULES:
1. Sentence rhythm: mix 4-word punches with 40-word long sentences. Never uniform.
2. Openers: start sentences with pronouns, verbs, prepositions, fragments, questions. Never two in a row with the same.
3. Voice: write like you're texting a friend who asked you to explain this. First/second person OK.
4. Imperfection: add one short sentence fragment. Use a dash. Start a sentence with "And" or "But".
5. Vocabulary: swap AI words (leverage→use, facilitate→help, comprehensive→full, robust→strong, utilize→use, furthermore→also, moreover→plus, nevertheless→still, consequently→so)
6. Structure: merge short sentences. Split long ones. Rearrange order where meaning allows.
7. Keep ALL facts, data, numbers, and core meaning identical. Do NOT add new information.

Original text:
{text}

Rewrite it now:"""


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
) -> str:
    """Generate a single paraphrase candidate via LLM API."""
    config = config or load_config()
    temp = temperature if temperature is not None else config.llm_temperature

    kwargs = _litellm_kwargs(config, temp)
    kwargs["messages"] = [
        {"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}
    ]

    response = litellm.completion(**kwargs)

    return response.choices[0].message.content.strip()


def generate_candidates(
    text: str,
    n: int | None = None,
    config: Config | None = None,
) -> list[str]:
    """Generate N paraphrase candidates in parallel with varied temperatures."""
    config = config or load_config()
    n = n or config.n_candidates

    temperatures = [_CANDIDATE_TEMPERATURES[i % len(_CANDIDATE_TEMPERATURES)] for i in range(n)]

    candidates: list[str] = []

    def _generate_one(idx: int, temp: float) -> str | None:
        try:
            return generate_candidate(text, config, temperature=temp)
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
) -> tuple[str, float]:
    """Select the best candidate by detection score, not just similarity.

    Optimization: pre-filter by similarity (cheap), then detect only top candidates (expensive).
    This reduces detection calls from N to min(N, 3).

    Args:
        detector: Optional DetectorEnsemble instance. If None, creates one.

    Returns (best_candidate, similarity_score).
    """
    config = config or load_config()

    if not candidates:
        return original, 1.0

    # Step 1: Compute all similarity scores (cheap — no models)
    sim_map = {}
    for c in candidates:
        sim_map[c] = check_similarity(original, c, config=config)

    # Step 2: Filter by minimum similarity, sort by sim descending
    scored = [(c, sim_map[c]) for c in candidates if sim_map[c] >= _MIN_CANDIDATE_SIM]
    if not scored:
        scored = [(c, sim_map[c]) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)

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
        if p_ai < best_pai:
            best = candidate
            best_pai = p_ai
            best_sim = sim

    # Fallback: if detection failed on all, pick highest similarity
    if best is None:
        best = scored[0][0]
        best_sim = scored[0][1]

    return best, best_sim



