"""Style Scorer — evaluates human-likeness of text style.

Stub: gracefully degrades when model is unavailable.
"""


class StyleScorer:
    """Stub style scorer. Always degrades to unloaded state."""

    def __init__(self):
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def human_score(self, text: str) -> float | None:
        return None
