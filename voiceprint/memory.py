"""PromptMemory — adaptive prompt_level selection feedback loop.

Tracks which prompt levels produce the lowest p_ai scores and biases
future selections toward what worked.
"""

from __future__ import annotations

import threading
from collections import defaultdict


class PromptMemory:
    """Thread-safe memory of per-level prompt success rates.

    Records (prompt_level, p_ai) outcomes and recommends the best level.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._history: dict[int, list[float]] = defaultdict(list)

    def record(self, prompt_level: int, p_ai: float) -> None:
        """Record the p_ai result for a given prompt level."""
        with self._lock:
            self._history[prompt_level].append(p_ai)

    def success_rate(self, prompt_level: int) -> float:
        """Fraction of attempts at this level that passed (p_ai < 0.5)."""
        with self._lock:
            scores = self._history.get(prompt_level, [])
            if not scores:
                return 0.0
            return sum(1 for s in scores if s < 0.5) / len(scores)

    def avg_p_ai(self, prompt_level: int) -> float | None:
        """Average p_ai for this level. None if no data."""
        with self._lock:
            scores = self._history.get(prompt_level, [])
            return sum(scores) / len(scores) if scores else None

    def best_level(self, default: int = 0) -> int:
        """Prompt level with the lowest average p_ai. Falls back to default."""
        with self._lock:
            best = default
            best_avg = float("inf")
            for level, scores in self._history.items():
                avg = sum(scores) / len(scores)
                if avg < best_avg:
                    best_avg = avg
                    best = level
            return best

    def total_runs(self) -> int:
        """Total number of recorded outcomes."""
        with self._lock:
            return sum(len(v) for v in self._history.values())

    def counts(self) -> dict[int, int]:
        """Number of attempts per prompt level."""
        with self._lock:
            return {k: len(v) for k, v in self._history.items()}

    def reset(self) -> None:
        """Clear all history."""
        with self._lock:
            self._history.clear()
