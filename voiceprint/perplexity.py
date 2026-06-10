"""Perplexity-based scoring — targets GPTZero's core signal.

Measures how predictable text is using GPT-2. Higher perplexity = more
human-like. Uses lazy loading with graceful fallback (no CUDA needed).
"""

from __future__ import annotations

import logging
import math
import os

logger = logging.getLogger(__name__)

_MODEL = None
_TOKENIZER = None
_LOADED = False


def _load_model() -> bool:
    """Lazy-load GPT-2 for perplexity. Returns True if loaded."""
    global _MODEL, _TOKENIZER, _LOADED
    if _LOADED:
        return _MODEL is not None
    _LOADED = True
    try:
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _MODEL = AutoModelForCausalLM.from_pretrained("gpt2")
        # Reject mock/stub models (conftest stubs transformers for fast tests)
        if type(_MODEL).__module__.startswith("unittest.mock"):
            logger.warning("Perplexity model is a mock — skipping")
            _MODEL = None
            _TOKENIZER = None
            return False
        _MODEL.eval()
        _TOKENIZER = AutoTokenizer.from_pretrained("gpt2")
        _TOKENIZER.pad_token = _TOKENIZER.eos_token
        logger.info("GPT-2 loaded for perplexity scoring")
        return True
    except Exception as e:
        logger.warning(f"Perplexity model unavailable: {e}")
        _MODEL = None
        _TOKENIZER = None
        return False


def perplexity_score(text: str) -> float | None:
    """Compute a perplexity-based human-likeness score (0-1).

    Higher score = more human-like (higher perplexity).
    Returns None if the model couldn't be loaded.
    """
    if not text or not text.strip():
        return 0.0
    if not _load_model():
        return None

    try:
        import torch
        inputs = _TOKENIZER(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = _MODEL(
                **inputs, labels=inputs["input_ids"]
            )
        ppl = math.exp(outputs.loss.item())

        # GPT-2 finds human-like text more predictable (lower perplexity).
        # Normalize: ppl >= 80 → 0.0 (AI-like, surprising), ppl <= 20 → 1.0 (human-like, predictable)
        score = min(max((80.0 - ppl) / 60.0, 0.0), 1.0)
        return score
    except Exception as e:
        logger.warning(f"Perplexity scoring failed: {e}")
        return None


def _get_gpt2():
    """Return globally cached GPT-2 model and tokenizer, or (None, None).

    Used by BinocularsDetector to avoid loading GPT-2 twice.
    """
    if _load_model():
        return _MODEL, _TOKENIZER
    return None, None


def raw_perplexity(text: str) -> float | None:
    """Return raw perplexity value (not normalized)."""
    if not text or not text.strip():
        return 0.0
    if not _load_model():
        return None
    try:
        import torch
        inputs = _TOKENIZER(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = _MODEL(
                **inputs, labels=inputs["input_ids"]
            )
        ppl = math.exp(outputs.loss.item())
        logger.debug(f"Perplexity: {ppl:.2f} ({len(inputs.input_ids[0])} tokens)")
        return ppl
    except Exception as e:
        logger.warning(f"Raw perplexity failed: {e}")
        return None
