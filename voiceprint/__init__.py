"""VoicePrint — AI Text Humanizer"""

from .perplexity import perplexity_score, raw_perplexity, get_gpt2
from .memory import PromptMemory
from .restructure import apply_restructure

# Phase 2: Trained humanizer model (lazy — only imported when actually used)
try:
    from .humanizer_model import HumanizerModel, humanize
except ImportError:
    # Optional dep — not installed on Streamlit Cloud
    import warnings as _w
    _w.warn("humanizer_model not available (install with: pip install llama-cpp-python)", stacklevel=2)

    class HumanizerModel:  # type: ignore
        """Dummy stub — real class available after `pip install llama-cpp-python`."""
        def __init__(self, *a, **kw): raise ImportError("humanizer_model not installed")

    def humanize(text: str, **kw) -> dict:  # type: ignore
        raise ImportError("humanizer_model not installed")

__version__ = "0.1.0"
