"""Pipeline orchestrator — Ties all 4 stages together.

Usage:
    from voiceprint.pipeline import HumanizePipeline
    pipe = HumanizePipeline()
    result = pipe.run("Your AI-generated text here")
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config, load_config
from .scrub import scrub
from .paraphrase import paraphrase, generate_candidates, select_best
from .detect import DetectorEnsemble, EnsembleResult
from .polish import polish
from .similarity import check_similarity
from .metrics import burstiness, burstiness_report, readability_scores
from .patterns import pattern_score, compute_all_signals


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Full pipeline output."""
    original: str
    humanized: str
    detection: EnsembleResult
    similarity: float
    burstiness: float
    pattern_score: float
    readability: dict
    burstiness_detail: dict
    signals: dict
    stages_applied: list[str]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class HumanizePipeline:
    """4-stage humanization pipeline."""

    def __init__(self, config: Config | None = None):
        self.config = config or load_config()
        self.ensemble = DetectorEnsemble(self.config)

    def run(
        self,
        text: str,
        use_scrub: bool = True,
        use_paraphrase: bool = True,
        use_polish: bool = True,
        n_candidates: int | None = None,
    ) -> PipelineResult:
        """Run the full pipeline.

        Args:
            text: Input AI-generated text
            use_scrub: Enable Stage 1 (heuristic scrub)
            use_paraphrase: Enable Stage 2 (LLM paraphrasing)
            use_polish: Enable Stage 4 (style polish)
            n_candidates: Override number of LLM candidates

        Returns:
            PipelineResult with humanized text and all metrics
        """
        stages = []
        current = text

        # Stage 1: Heuristic scrub
        if use_scrub:
            current = scrub(current)
            stages.append("scrub")

        # Stage 2: Adversarial paraphrasing
        if use_paraphrase:
            candidates = generate_candidates(
                current, n=n_candidates, config=self.config
            )
            if candidates:
                current, sim = select_best(current, candidates, self.config)
                stages.append("paraphrase")

        # Stage 3: Detection (always runs for scoring)
        detection = self.ensemble.detect(current)
        stages.append("detect")

        # Stage 4: Style polish
        if use_polish:
            current = polish(current)
            stages.append("polish")

        # Final metrics
        final_detection = self.ensemble.detect(current)
        final_sim = check_similarity(text, current, self.config)
        final_burstiness = burstiness(current)
        final_pattern = pattern_score(current)

        return PipelineResult(
            original=text,
            humanized=current,
            detection=final_detection,
            similarity=final_sim,
            burstiness=final_burstiness,
            pattern_score=final_pattern,
            readability=readability_scores(current),
            burstiness_detail=burstiness_report(current),
            signals=compute_all_signals(current),
            stages_applied=stages,
        )

    def detect_only(self, text: str) -> EnsembleResult:
        """Run detection without humanization (for pre-check)."""
        return self.ensemble.detect(text)

    def scrub_only(self, text: str) -> str:
        """Run only the heuristic scrub (no API calls)."""
        return scrub(text)
