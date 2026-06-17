"""VoicePrint — AI Text Humanizer"""

from .perplexity import perplexity_score, raw_perplexity, get_gpt2
from .memory import PromptMemory
from .restructure import apply_restructure

# Phase 2: Trained humanizer model (lazy — only fails when actually used)
from .humanizer_model import HumanizerModel, humanize

__version__ = "0.1.0"
