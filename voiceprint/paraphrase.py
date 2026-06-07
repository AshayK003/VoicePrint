"""Stage 2: Adversarial Paraphrasing — LLM-based text rewriting.

Uses cloud LLM APIs (via litellm) to generate N candidate paraphrases,
then selects the best one based on detection scores and similarity.
"""

from __future__ import annotations

import litellm

from .config import Config, load_config
from .similarity import check_similarity


# ---------------------------------------------------------------------------
# Paraphrasing prompt
# ---------------------------------------------------------------------------

PARAPHRASE_PROMPT = """\
Rewrite the following AI-generated text to sound like a real human wrote it.

Rules:
- Vary sentence length dramatically (some short 5-10 words, some long 30+ words)
- Use casual, conversational phrasing where appropriate
- Add natural imperfections — not everything needs to be perfectly structured
- Replace formal AI vocabulary with everyday words
- Keep ALL facts, data, and meaning exactly the same
- Do NOT add new information that wasn't in the original
- Write as if explaining to a smart friend, not submitting to a professor

Original text:
{text}

Humanized version:"""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _litellm_kwargs(config: Config, temperature: float) -> dict:
    """Build kwargs dict for litellm.completion, passing provider config."""
    kwargs: dict = {
        "model": config.llm_model,
        "messages": [{"role": "user", "content": ""}],
        "temperature": temperature,
        "max_tokens": config.llm_max_tokens,
    }

    # API key — litellm expects it via api_key param or env var
    if config.api_key:
        kwargs["api_key"] = config.api_key

    # Base URL — for custom OpenAI-compatible endpoints
    if config.base_url:
        kwargs["api_base"] = config.base_url

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
    """Generate N paraphrase candidates with varied temperatures."""
    config = config or load_config()
    n = n or config.n_candidates

    candidates = []
    # Vary temperature across candidates for diversity
    temps = [0.7, 0.9, 1.0, 1.1, 1.2, 1.0, 0.8, 1.1]

    for i in range(n):
        temp = temps[i % len(temps)]
        try:
            candidate = generate_candidate(text, config, temperature=temp)
            candidates.append(candidate)
        except Exception as e:
            print(f"Warning: Candidate {i+1} failed: {e}")
            continue

    return candidates


def select_best(
    original: str,
    candidates: list[str],
    config: Config | None = None,
) -> tuple[str, float]:
    """Select the best candidate based on similarity to original.

    Returns (best_candidate, similarity_score).
    """
    config = config or load_config()

    best_text = ""
    best_score = 0.0

    for candidate in candidates:
        sim = check_similarity(original, candidate)
        if sim >= config.similarity_threshold and sim > best_score:
            best_text = candidate
            best_score = sim

    # Fallback: if no candidate passes threshold, pick highest similarity
    if not best_text and candidates:
        scores = [(c, check_similarity(original, c)) for c in candidates]
        best_text, best_score = max(scores, key=lambda x: x[1])

    return best_text, best_score


def paraphrase(
    text: str,
    config: Config | None = None,
) -> tuple[str, float]:
    """Full paraphrasing pipeline: generate N candidates, select best.

    Returns (humanized_text, similarity_score).
    """
    config = config or load_config()

    candidates = generate_candidates(text, config=config)
    if not candidates:
        return text, 1.0

    return select_best(text, candidates, config=config)
