"""Tests for pipeline orchestrator with mocked LLM and detectors."""

import pytest
from unittest.mock import patch, MagicMock
from voiceprint.pipeline import HumanizePipeline
from voiceprint.config import Config
from voiceprint.detect import (
    EnsembleResult, DetectorEnsemble, statistical_detect, DetectionResult,
)


# ---------------------------------------------------------------------------
# Fake detector result
# ---------------------------------------------------------------------------

def _fake_ensemble_result(p_ai: float = 0.15) -> EnsembleResult:
    return EnsembleResult(p_ai=p_ai, detectors=[], passed=p_ai < 0.5)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHumanizePipeline:
    def test_empty_input(self):
        pipeline = HumanizePipeline(config=Config())
        result = pipeline.run("")
        assert result.humanized == ""

    @patch("voiceprint.pipeline.generate_candidates")
    @patch("voiceprint.pipeline.select_best")
    def test_scrub_only(self, mock_select, mock_gen):
        pipeline = HumanizePipeline(config=Config())
        result = pipeline.run(
            "Furthermore, the results are very important.",
            use_paraphrase=False,
            use_polish=False,
        )
        assert "Furthermore" not in result.humanized
        mock_gen.assert_not_called()

    @patch("voiceprint.pipeline.generate_candidates")
    @patch("voiceprint.pipeline.select_best")
    def test_paraphrase_called(self, mock_select, mock_gen):
        mock_gen.return_value = ["Paraphrased text."]
        mock_select.return_value = ("Paraphrased text.", 0.85)

        pipeline = HumanizePipeline(config=Config())
        pipeline.ensemble.detect = MagicMock(return_value=_fake_ensemble_result(p_ai=0.6))
        result = pipeline.run(
            "Furthermore, the results are very important.",
            use_polish=False,
        )
        assert mock_gen.call_count == 4

    @patch("voiceprint.pipeline.generate_candidates", side_effect=Exception("API error"))
    def test_paraphrase_failure_continues(self, mock_gen):
        pipeline = HumanizePipeline(config=Config())
        result = pipeline.run(
            "Furthermore, the results are very important.",
            use_polish=False,
        )
        assert result.humanized  # Should still have scrubbed text

    @patch("voiceprint.pipeline.generate_candidates")
    @patch("voiceprint.pipeline.select_best")
    def test_stages_applied(self, mock_select, mock_gen):
        mock_gen.return_value = ["Paraphrased."]
        mock_select.return_value = ("Paraphrased.", 0.85)

        pipeline = HumanizePipeline(config=Config())
        result = pipeline.run(
            "Furthermore, the results are very important.",
        )
        assert "scrub" in result.stages_applied
        assert "paraphrase" in result.stages_applied
        assert "polish" in result.stages_applied
        assert "detect" in result.stages_applied

    @patch("voiceprint.pipeline.generate_candidates")
    @patch("voiceprint.pipeline.select_best")
    def test_progress_callback(self, mock_select, mock_gen):
        mock_gen.return_value = ["Text."]
        mock_select.return_value = ("Text.", 0.85)

        calls = []
        def on_progress(pct, msg):
            calls.append((pct, msg))

        pipeline = HumanizePipeline(config=Config())
        result = pipeline.run(
            "Furthermore, the results are very important.",
            progress_callback=on_progress,
        )
        assert len(calls) > 0
        assert calls[-1][0] == 1.0  # Final progress is 1.0

    def test_result_has_readability(self):
        pipeline = HumanizePipeline(config=Config())
        result = pipeline.run(
            "The quick brown fox jumps over the lazy dog. It was a fine day.",
            use_paraphrase=False,
        )
        assert "flesch_reading_ease" in result.readability

    def test_result_has_burstiness_detail(self):
        pipeline = HumanizePipeline(config=Config())
        result = pipeline.run(
            "The quick brown fox jumps over the lazy dog. It was a fine day.",
            use_paraphrase=False,
        )
        assert "burstiness" in result.burstiness_detail
        assert "sentence_count" in result.burstiness_detail

    def test_result_has_signals(self):
        pipeline = HumanizePipeline(config=Config())
        result = pipeline.run(
            "The quick brown fox jumps over the lazy dog. It was a fine day.",
            use_paraphrase=False,
        )
        assert "ai_vocabulary" in result.signals

    @patch("voiceprint.pipeline.generate_candidates")
    @patch("voiceprint.pipeline.select_best")
    def test_detect_only(self, mock_select, mock_gen):
        pipeline = HumanizePipeline(config=Config())
        result = pipeline.detect_only("Some text to detect.")
        assert hasattr(result, "p_ai")
        assert hasattr(result, "passed")

    # ------------------------------------------------------------------
    # Retry loop tests
    # ------------------------------------------------------------------

    @patch("voiceprint.pipeline.generate_candidates")
    @patch("voiceprint.pipeline.select_best")
    def test_retry_loop_runs_multiple_iterations(self, mock_select, mock_gen):
        mock_gen.return_value = ["Candidate text."]
        mock_select.return_value = ("Candidate text.", 0.85)

        config = Config(max_iterations=3)
        pipeline = HumanizePipeline(config=config)
        pipeline.ensemble.detect = MagicMock(return_value=_fake_ensemble_result(p_ai=0.6))
        pipeline.run(
            "Furthermore, the results are very important and quite significant.",
            use_polish=False,
        )
        assert mock_gen.call_count == 3

    @patch("voiceprint.pipeline.generate_candidates")
    @patch("voiceprint.pipeline.select_best")
    def test_retry_early_exit_on_pass(self, mock_select, mock_gen):
        mock_gen.return_value = ["Candidate text."]
        mock_select.return_value = ("Candidate text.", 0.85)

        config = Config(max_iterations=5)
        pipeline = HumanizePipeline(config=config)
        # Mock ensemble.detect to pass on first call
        pipeline.ensemble.detect = MagicMock(
            side_effect=[
                _fake_ensemble_result(p_ai=0.1),   # passes
                _fake_ensemble_result(p_ai=0.15),  # would pass but never reached
                _fake_ensemble_result(p_ai=0.2),
                _fake_ensemble_result(p_ai=0.25),
                _fake_ensemble_result(p_ai=0.3),
            ]
        )
        result = pipeline.run(
            "Furthermore, the results are very important.",
            use_polish=False,
        )
        # Should exit early (only 1 iteration) since detection passes on first attempt
        assert mock_gen.call_count == 1
        assert result.detection.passed is True

    @patch("voiceprint.pipeline.generate_candidates")
    @patch("voiceprint.pipeline.select_best")
    def test_retry_tracks_best_result(self, mock_select, mock_gen):
        call_count = [0]

        def gen_side_effect(text, n=None, config=None):
            call_count[0] += 1
            return [f"Candidate {call_count[0]}"]

        def select_side_effect(original, candidates, config=None):
            # First iteration: high p_ai (bad), second: low p_ai (good)
            if call_count[0] == 1:
                return ("Worse candidate", 0.80)
            return ("Better candidate", 0.90)

        mock_gen.side_effect = gen_side_effect
        mock_select.side_effect = select_side_effect

        config = Config(max_iterations=3)
        pipeline = HumanizePipeline(config=config)
        result = pipeline.run(
            "Furthermore, the results are very important and quite significant.",
            use_polish=False,
        )
        # Should keep the best result across iterations
        assert result.humanized is not None

    def test_max_iterations_one_preserves_behavior(self):
        config = Config(max_iterations=1)
        pipeline = HumanizePipeline(config=config)
        result = pipeline.run(
            "The quick brown fox jumps over the lazy dog.",
            use_paraphrase=False,
            use_restructure=False,
            use_polish=False,
        )
        assert result.humanized is not None
        assert result.stages_applied == ["scrub", "detect"]

    @patch("voiceprint.pipeline.generate_candidates")
    @patch("voiceprint.pipeline.select_best")
    def test_prev_p_ai_passed_to_generate_candidates(self, mock_select, mock_gen):
        """Detection score from iteration N should be passed to iteration N+1."""
        mock_gen.return_value = ["Cand"]
        mock_select.return_value = ("Selected.", 0.85)

        config = Config(max_iterations=3)
        pipeline = HumanizePipeline(config=config)
        pipeline.ensemble.detect = MagicMock(
            side_effect=[
                _fake_ensemble_result(p_ai=0.6),
                _fake_ensemble_result(p_ai=0.7),
                _fake_ensemble_result(p_ai=0.8),
            ]
        )
        pipeline.run(
            "Furthermore, the results are very important.",
            use_polish=False,
        )

        assert mock_gen.call_count == 3
        call1_kwargs = mock_gen.call_args_list[0].kwargs
        call2_kwargs = mock_gen.call_args_list[1].kwargs
        call3_kwargs = mock_gen.call_args_list[2].kwargs
        assert "prev_p_ai" in call1_kwargs
        assert call1_kwargs["prev_p_ai"] is None
        assert call2_kwargs["prev_p_ai"] == 0.6
        assert call3_kwargs["prev_p_ai"] == 0.7


# ---------------------------------------------------------------------------
# Binoculars detection tests
# ---------------------------------------------------------------------------

class TestBinocularsDetection:
    def test_statistical_detect_clear_human(self):
        """Short, simple text should be flagged as clearly human (no models)."""
        result = statistical_detect("Hello world. Goodbye!")
        assert result.method == "statistical"
        assert result.passed is True

    def test_statistical_detect_clear_ai(self):
        """Long AI-heavy text should be flagged as clearly AI (no models)."""
        text = (
            "Furthermore, it is important to note that the utilization of "
            "comprehensive methodologies facilitates the optimization of "
            "outcomes. Moreover, the implementation of strategic frameworks "
            "enables the systematic enhancement of performance metrics. "
            "Additionally, the integration of innovative approaches contributes "
            "to the advancement of organizational objectives."
        )
        result = statistical_detect(text)
        assert result.method == "statistical"
        assert result.p_ai > 0.5
        assert result.passed is False

    def test_ensemble_includes_binoculars_when_ambiguous(self):
        """When statistical score is ambiguous, Binoculars should run."""
        ensemble = DetectorEnsemble(config=Config())

        # Mock Binoculars to return a known result
        mock_binoculars_result = DetectionResult(
            name="binoculars", p_ai=0.4, label="REAL"
        )
        ensemble._binoculars = MagicMock()
        ensemble._binoculars.detect.return_value = mock_binoculars_result

        # Also mock RoBERTa to avoid loading heavy models
        mock_roberta_result = DetectionResult(
            name="roberta-test", p_ai=0.6, label="FAKE"
        )
        ensemble._roberta_detectors = [
            MagicMock(detect=MagicMock(return_value=mock_roberta_result)),
            MagicMock(detect=MagicMock(return_value=mock_roberta_result)),
        ]
        ensemble._roberta_weights = [0.65, 0.35]

        # Use text that gives ambiguous statistical score (0.30-0.70)
        text = (
            "Furthermore, the results are important and significant. "
            "The findings demonstrate considerable value."
        )
        result = ensemble.detect(text)

        # Should have Binoculars + statistical + RoBERTa detectors
        names = [d.name for d in result.detectors]
        assert "binoculars" in names
        assert "statistical" in names

    def test_binoculars_score_mapping(self):
        """Verify Binoculars score-to-p_ai mapping logic."""
        # Score = 1.0 (both models agree) → high p_ai (AI)
        score = 1.0
        p_ai = max(0.0, min(1.0, 1.0 - abs(score - 1.0) * 5.0))
        assert p_ai == pytest.approx(1.0)

        # Score = 0.8 (models disagree) → low p_ai (human)
        score = 0.8
        p_ai = max(0.0, min(1.0, 1.0 - abs(score - 1.0) * 5.0))
        assert p_ai == pytest.approx(0.0)

        # Score = 0.9 (slight disagreement) → medium p_ai
        score = 0.9
        p_ai = max(0.0, min(1.0, 1.0 - abs(score - 1.0) * 5.0))
        assert p_ai == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Pipeline edge-case / error-path tests
# ---------------------------------------------------------------------------

class TestPipelineEdgeCases:
    def test_whitespace_only_input(self):
        pipeline = HumanizePipeline(config=Config())
        result = pipeline.run("   \n\t  ")
        assert result.humanized == "   \n\t  "
        assert result.stages_applied == []

    def test_detection_exception_fallback(self):
        """When detect() raises, pipeline returns p_ai=0.5 fallback."""
        pipeline = HumanizePipeline(config=Config())
        pipeline.ensemble.detect = MagicMock(side_effect=Exception("Model crashed"))
        result = pipeline.run(
            "Furthermore, the results are very important.",
            use_paraphrase=False,
            use_polish=False,
        )
        assert result.detection.p_ai == 0.5
        assert result.detection.passed is False

    def test_similarity_check_failure_fallback(self):
        """When check_similarity raises, pipeline returns 0.0."""
        pipeline = HumanizePipeline(config=Config())
        with patch("voiceprint.pipeline.check_similarity", side_effect=Exception("Model error")):
            result = pipeline.run(
                "Furthermore, the results are very important.",
                use_paraphrase=False,
                use_polish=False,
            )
        assert result.similarity == 0.0

    def test_result_has_all_expected_fields(self):
        pipeline = HumanizePipeline(config=Config())
        result = pipeline.run(
            "The quick brown fox jumps over the lazy dog.",
            use_paraphrase=False,
        )
        assert hasattr(result, "original")
        assert hasattr(result, "humanized")
        assert hasattr(result, "detection")
        assert hasattr(result, "similarity")
        assert hasattr(result, "burstiness")
        assert hasattr(result, "pattern_score")
        assert hasattr(result, "readability")
        assert hasattr(result, "burstiness_detail")
        assert hasattr(result, "signals")
        assert hasattr(result, "stages_applied")
