"""Pipeline orchestrator — Ties all 4 stages together.

Usage:
    from voiceprint.pipeline import HumanizePipeline
    pipe = HumanizePipeline()
    result = pipe.run("Your AI-generated text here")
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .config import Config, load_config
from .scrub import scrub
from .paraphrase import generate_candidates, select_best
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
        progress_callback: callable | None = None,
    ) -> PipelineResult:
        """Run the full pipeline.

        Args:
            text: Input AI-generated text
            use_scrub: Enable Stage 1 (heuristic scrub)
            use_paraphrase: Enable Stage 2 (LLM paraphrasing)
            use_polish: Enable Stage 4 (style polish)
            n_candidates: Override number of LLM candidates
            progress_callback: fn(progress_float, message_str) for UI updates

        Returns:
            PipelineResult with humanized text and all metrics
        """
        if not text or not text.strip():
            return PipelineResult(
                original=text,
                humanized=text,
                detection=EnsembleResult(p_ai=0.0, detectors=[], passed=True),
                similarity=1.0,
                burstiness=0.0,
                pattern_score=0.0,
                readability=readability_scores(text),
                burstiness_detail=burstiness_report(text),
                signals=compute_all_signals(text),
                stages_applied=[],
            )
        stages: list[str] = []
        max_iter = self.config.max_iterations

        def _report(pct: float, msg: str):
            if progress_callback:
                progress_callback(pct, msg)

        # Stage 1: Heuristic scrub (runs once, before iteration loop)
        scrubbed = text
        if use_scrub:
            _report(0.05, "Stage 1: Heuristic scrub...")
            scrubbed = scrub(text)
            stages.append("scrub")

        # Iterative loop: paraphrase → polish → detect
        best_text = scrubbed
        best_pai = 1.0
        best_detection: EnsembleResult | None = None

        for iteration in range(max_iter):
            current = best_text if iteration > 0 and best_text != scrubbed else scrubbed
            iter_label = f" (attempt {iteration + 1}/{max_iter})" if max_iter > 1 else ""

            # Stage 2: Adversarial paraphrasing
            if use_paraphrase:
                _report(0.15, f"Stage 2: Generating candidates{iter_label}...")
                try:
                    candidates = generate_candidates(
                        current, n=n_candidates, config=self.config
                    )
                    if candidates:
                        _report(0.65, f"Stage 2: Selecting best candidate{iter_label}...")
                        current, _sim = select_best(current, candidates, self.config)
                        if "paraphrase" not in stages:
                            stages.append("paraphrase")
                        # Post-paraphrase scrub: LLM re-introduces AI patterns
                        current = scrub(current)
                except Exception as e:
                    _report(0.65, f"Stage 2 skipped: {e}")

            # Stage 3: Style polish
            if use_polish:
                _report(0.70, f"Stage 3: Style polish{iter_label}...")
                current = polish(current)
                if "polish" not in stages:
                    stages.append("polish")

            # Stage 4: Detection
            _report(0.80, f"Stage 4: Running detection{iter_label}...")
            try:
                detection = self.ensemble.detect(current)
                if "detect" not in stages:
                    stages.append("detect")
            except Exception as e:
                logging.warning(f"Detection failed: {e}")
                detection = EnsembleResult(p_ai=0.5, detectors=[], passed=False)

            # Track best result across iterations
            if detection.p_ai < best_pai:
                best_pai = detection.p_ai
                best_text = current
                best_detection = detection

            # Early exit if detection passes
            if detection.passed:
                _report(0.90, f"Detection passed on attempt {iteration + 1}!")
                break

        # Final metrics on best result (compute in parallel)
        _report(0.95, "Computing final metrics...")

        def _compute_similarity():
            try:
                return check_similarity(text, best_text, self.config)
            except Exception as e:
                logging.warning(f"Similarity check failed: {e}")
                return 0.0

        def _compute_burstiness():
            return burstiness(best_text)

        def _compute_pattern_score():
            return pattern_score(best_text)

        def _compute_readability():
            return readability_scores(best_text)

        def _compute_burstiness_detail():
            return burstiness_report(best_text)

        def _compute_signals():
            return compute_all_signals(best_text)

        with ThreadPoolExecutor(max_workers=3) as executor:
            fut_sim = executor.submit(_compute_similarity)
            fut_bur = executor.submit(_compute_burstiness)
            fut_ps = executor.submit(_compute_pattern_score)
            fut_read = executor.submit(_compute_readability)
            fut_bur_det = executor.submit(_compute_burstiness_detail)
            fut_sig = executor.submit(_compute_signals)

            final_sim = fut_sim.result()
            final_burstiness = fut_bur.result()
            final_ps = fut_ps.result()
            final_readability = fut_read.result()
            final_bur_det = fut_bur_det.result()
            final_signals = fut_sig.result()

        _report(1.0, "Done!")
        return PipelineResult(
            original=text,
            humanized=best_text,
            detection=best_detection or EnsembleResult(p_ai=0.5, detectors=[], passed=False),
            similarity=final_sim,
            burstiness=final_burstiness,
            pattern_score=final_ps,
            readability=final_readability,
            burstiness_detail=final_bur_det,
            signals=final_signals,
            stages_applied=stages,
        )

    def detect_only(self, text: str) -> EnsembleResult:
        """Run detection without humanization (for pre-check)."""
        return self.ensemble.detect(text)
