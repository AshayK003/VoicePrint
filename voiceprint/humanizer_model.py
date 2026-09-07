"""
VoicePrint — HumanizerModel
============================
Loads the fine-tuned Mistral/GGUF model and provides inference for
AI-to-human text rewriting, with graceful CPU fallback.

The model is loaded lazily and cached globally. If the model file
isn't found, all methods return None (callers fall back to LLM API).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths — user can override with env var or constructor arg
# ---------------------------------------------------------------------------
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "humanizer"
DEFAULT_MODEL_PATH = os.getenv("VOICEPRINT_HUMANIZER_MODEL", str(DEFAULT_MODEL_DIR / "mistral-7b-humanizer.gguf"))


class HumanizerModel:
    """Thin wrapper around a GGUF model fine-tuned for AI→human rewriting.

    Uses llama-cpp-python for CPU inference. The model file should be a
    merged GGUF (base weights + LoRA adapter baked in).

    Usage:
        model = _HUMANIZER_CACHE.get(...) or HumanizerModel(...)
        result = model.humanize("AI-generated text here...")

    Falls back to None everywhere if the model file cannot be loaded.
    Instantiate via the module-level humanize() function to get global caching.
    """

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self._model = None
        self._llm = None
        self._load()

    # ------------------------------------------------------------------
    # Prompt template — matches the style targets from Phase 1
    # ------------------------------------------------------------------
    SYSTEM_PROMPT = (
        "You rewrite AI-generated text to sound like it was written by a human. "
        "Your output is natural, varied in sentence structure, uses contractions "
        "appropriately, and avoids the polished uniformity that marks AI text. "
        "You preserve all facts and meaning exactly."
    )

    USER_TEMPLATE = "Rewrite this to sound human:\n{text}"

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load(self) -> bool:
        """Load the GGUF model via llama-cpp-python. Returns True on success."""
        model_file = Path(self.model_path)
        if not model_file.exists():
            logger.info(f"Humanizer model not found at {self.model_path} — will use LLM API fallback")
            return False

        try:
            from llama_cpp import Llama

            logger.info(f"Loading Humanizer model from {model_file} ({model_file.stat().st_size / 1024**3:.1f} GB)...")
            self._llm = Llama(
                model_path=str(model_file),
                n_ctx=2048,
                n_threads=os.cpu_count() or 4,
                n_gpu_layers=0,  # CPU-only
                verbose=False,
            )
            logger.info("Humanizer model loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to load Humanizer model: {e}")
            self._llm = None
            return False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def humanize(self, text: str, temperature: float = 0.8) -> str | None:
        """Rewrite AI text to sound human.

        Args:
            text: Input text (typically AI-generated).
            temperature: Creativity (0.0-1.5). Higher = more varied output.

        Returns:
            Humanized text, or None if the model isn't loaded.
        """
        if self._llm is None:
            return None

        if not text or not text.strip():
            return text

        try:
            response = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": self.USER_TEMPLATE.format(text=text)},
                ],
                temperature=temperature,
                max_tokens=len(text.split()) * 3,  # generous headroom
                stop=None,
            )
            content = response["choices"][0]["message"]["content"]
            return content.strip() if content else None
        except Exception as e:
            logger.warning(f"Humanizer inference failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    def __bool__(self) -> bool:
        return self.is_loaded


# ---------------------------------------------------------------------------
# Module-level convenience (used by voiceprint.__init__)
# ---------------------------------------------------------------------------

_HUMANIZER_CACHE: dict[str, HumanizerModel] = {}

def humanize(text: str, model_path: str | None = None, temperature: float = 0.8) -> str | None:
    """Module-level convenience. Lazy-caches the HumanizerModel."""
    global _HUMANIZER_CACHE
    cache_key = model_path or DEFAULT_MODEL_PATH
    if cache_key not in _HUMANIZER_CACHE:
        _HUMANIZER_CACHE[cache_key] = HumanizerModel(cache_key)
    model = _HUMANIZER_CACHE.get(cache_key)
    if model is None:
        return None
    return model.humanize(text, temperature=temperature)
