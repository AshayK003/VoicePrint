"""Service layer — Pure business logic between UI and pipeline.

No Streamlit, no framework deps. Every function is testable in isolation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from .config import Config, validate_config, load_config
from .pipeline import HumanizePipeline, PipelineResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate limiter — in-process sliding window to prevent runaway API costs
# ---------------------------------------------------------------------------

import time

_LIMIT_WINDOW = 60  # seconds
_LIMIT_MAX_CALLS = 10  # max humanize calls per window
_last_calls: list[float] = []


def _check_rate_limit() -> None:
    """Raise InputError if rate limit exceeded. Per-process sliding window."""
    global _last_calls
    now = time.time()
    cutoff = now - _LIMIT_WINDOW
    _last_calls = [t for t in _last_calls if t > cutoff]
    if len(_last_calls) >= _LIMIT_MAX_CALLS:
        raise InputError(
            f"Rate limit exceeded: max {_LIMIT_MAX_CALLS} humanizations "
            f"per {_LIMIT_WINDOW} seconds. Please wait."
        )
    _last_calls.append(now)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

_MIN_TEXT_LENGTH = 10
_MAX_TEXT_LENGTH = 100_000


class InputError(Exception):
    """Raised when user input is invalid."""


def validate_input(text: str) -> str:
    """Validate and sanitize input text. Returns stripped text.

    Raises InputError on problems.
    """
    if not text:
        raise InputError("Text cannot be empty.")
    text = text.strip()
    if len(text) < _MIN_TEXT_LENGTH:
        raise InputError(
            f"Text too short ({len(text)} chars). "
            f"Minimum is {_MIN_TEXT_LENGTH} characters."
        )
    if len(text) > _MAX_TEXT_LENGTH:
        raise InputError(
            f"Text too long ({len(text):,} chars). "
            f"Maximum is {_MAX_TEXT_LENGTH:,} characters."
        )
    return text


# ---------------------------------------------------------------------------
# Config building (pure — no Streamlit deps)
# ---------------------------------------------------------------------------

def build_config(
    provider: str = "",
    api_key: str = "",
    base_url: str = "",
    model: str = "",
) -> Config:
    """Build a Config from explicit parameters. No hidden deps.

    Falls back to env vars when api_key is empty.
    """
    config = Config()

    if provider:
        config.provider = provider
    if model:
        config.llm_model = model
    if base_url:
        config.base_url = base_url

    # Resolve API key: explicit > env var
    config.api_key = api_key
    if not config.api_key:
        preset = None
        if provider:
            from .config import PROVIDER_PRESETS
            preset = PROVIDER_PRESETS.get(provider)
        if preset and preset.get("env_key"):
            config.api_key = os.getenv(preset["env_key"], "")

    return config


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

@dataclass
class HumanizeResult:
    """Structured result from humanize(). Wraps PipelineResult for clarity."""
    success: bool
    text: str
    original: str
    ai_probability: float
    similarity: float
    burstiness: float
    pattern_score: float
    detection_summary: str
    readability: dict
    burstiness_detail: dict
    signals: dict
    stages: list[str]
    error: str | None = None


def humanize(
    text: str,
    config: Config | None = None,
    *,
    use_scrub: bool = True,
    use_paraphrase: bool = True,
    use_polish: bool = True,
    n_candidates: int | None = None,
    progress_callback: callable | None = None,
) -> HumanizeResult:
    """Run the full humanization pipeline. Returns structured result.

    This is the main entry point for any caller (UI, CLI, API).
    """
    text = validate_input(text)
    config = config or load_config()
    validate_config(config)
    _check_rate_limit()

    if n_candidates is not None:
        config.n_candidates = n_candidates

    pipe = HumanizePipeline(config=config)

    try:
        result: PipelineResult = pipe.run(
            text,
            use_scrub=use_scrub,
            use_paraphrase=use_paraphrase,
            use_polish=use_polish,
            n_candidates=config.n_candidates,
            progress_callback=progress_callback,
        )
    except Exception as e:
        from .paraphrase import _sanitize_error
        logger.error(f"Pipeline failed: {_sanitize_error(str(e))}", exc_info=True)
        return HumanizeResult(
            success=False,
            text="",
            original=text,
            ai_probability=0.0,
            similarity=0.0,
            burstiness=0.0,
            pattern_score=0.0,
            detection_summary="",
            readability={},
            burstiness_detail={},
            signals={},
            stages=[],
            error=str(e),
        )

    return HumanizeResult(
        success=True,
        text=result.humanized,
        original=result.original,
        ai_probability=result.detection.p_ai,
        similarity=result.similarity,
        burstiness=result.burstiness,
        pattern_score=result.pattern_score,
        detection_summary=result.detection.summary(),
        readability=result.readability,
        burstiness_detail=result.burstiness_detail,
        signals=result.signals,
        stages=result.stages_applied,
    )


def detect(text: str, config: Config | None = None) -> dict:
    """Run detection only (no humanization). Returns dict for JSON serialization."""
    text = validate_input(text)
    config = config or load_config()

    pipe = HumanizePipeline(config=config)
    result = pipe.detect_only(text)

    return {
        "p_ai": result.p_ai,
        "passed": result.passed,
        "method": result.method,
        "summary": result.summary(),
        "detectors": [
            {"name": d.name, "p_ai": d.p_ai, "label": d.label}
            for d in result.detectors
        ],
    }
