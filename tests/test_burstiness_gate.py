"""Tests for burstiness validation gate in pipeline.py."""
import pytest
from unittest.mock import patch, MagicMock
from voiceprint.pipeline import HumanizePipeline, PipelineResult
from voiceprint.detect import EnsembleResult


class TestBurstinessGate:
    def test_low_burstiness_triggers_retry(self):
        """When burstiness < 0.3 AND detection fails, pipeline should retry."""
        config = MagicMock()
        config.max_iterations = 3
        config.similarity_min = 0.55
        config.ensemble = MagicMock()
        config.style_scorer_path = None
        config.n_candidates = 3
        config.provider = "test"
        config.model = "test"
        config.api_key = "test"
        config.api_base = ""

        pipe = HumanizePipeline(config=config)

        # Detection FAILS (passed=False) — this is when burstiness gate should kick in
        call_count = 0

        def mock_detect(text):
            nonlocal call_count
            call_count += 1
            return EnsembleResult(p_ai=0.6, detectors=[], passed=False)

        pipe.ensemble.detect = mock_detect

        # Mock generate_candidates to return a candidate
        with patch("voiceprint.pipeline.generate_candidates") as mock_gen, \
             patch("voiceprint.pipeline.select_best") as mock_sel, \
             patch("voiceprint.pipeline.scrub", side_effect=lambda x: x), \
             patch("voiceprint.pipeline.polish", side_effect=lambda x: x), \
             patch("voiceprint.pipeline.burstiness", return_value=0.2), \
             patch("voiceprint.pipeline.check_similarity", return_value=0.8):

            mock_gen.return_value = ["candidate text here"]
            mock_sel.return_value = ("candidate text here", 0.8)

            result = pipe.run("Some AI text here.", use_scrub=True,
                            use_paraphrase=True, use_polish=True)

            # With low burstiness, detection failing, and max_iter=3,
            # should run multiple iterations (burstiness gate forces retry)
            assert call_count >= 2, (
                f"Expected >=2 detect calls (retry due to low burstiness), got {call_count}"
            )

    def test_high_burstiness_no_extra_retry(self):
        """When burstiness >= 0.3, no extra retry needed."""
        config = MagicMock()
        config.max_iterations = 2
        config.similarity_min = 0.55
        config.ensemble = MagicMock()
        config.style_scorer_path = None
        config.n_candidates = 3
        config.provider = "test"
        config.model = "test"
        config.api_key = "test"
        config.api_base = ""

        pipe = HumanizePipeline(config=config)

        call_count = 0

        def mock_detect(text):
            nonlocal call_count
            call_count += 1
            return EnsembleResult(p_ai=0.1, detectors=[], passed=True)

        pipe.ensemble.detect = mock_detect

        with patch("voiceprint.pipeline.generate_candidates") as mock_gen, \
             patch("voiceprint.pipeline.select_best") as mock_sel, \
             patch("voiceprint.pipeline.scrub", side_effect=lambda x: x), \
             patch("voiceprint.pipeline.polish", side_effect=lambda x: x), \
             patch("voiceprint.pipeline.burstiness", return_value=0.6), \
             patch("voiceprint.pipeline.check_similarity", return_value=0.8):

            mock_gen.return_value = ["candidate text here"]
            mock_sel.return_value = ("candidate text here", 0.8)

            result = pipe.run("Some AI text here.", use_scrub=True,
                            use_paraphrase=True, use_polish=True)

            # Detection passes on first attempt → early exit, only 1 detect call
            assert call_count == 1, (
                f"Expected 1 detect call (no retry needed), got {call_count}"
            )

    def test_burstiness_gate_respects_max_iterations(self):
        """Burstiness gate should not exceed max_iterations."""
        config = MagicMock()
        config.max_iterations = 2
        config.similarity_min = 0.55
        config.ensemble = MagicMock()
        config.style_scorer_path = None
        config.n_candidates = 3
        config.provider = "test"
        config.model = "test"
        config.api_key = "test"
        config.base_url = ""

        pipe = HumanizePipeline(config=config)

        call_count = 0

        def mock_detect(text):
            nonlocal call_count
            call_count += 1
            return EnsembleResult(p_ai=0.5, detectors=[], passed=False)

        pipe.ensemble.detect = mock_detect

        with patch("voiceprint.pipeline.generate_candidates") as mock_gen, \
             patch("voiceprint.pipeline.select_best") as mock_sel, \
             patch("voiceprint.pipeline.scrub", side_effect=lambda x: x), \
             patch("voiceprint.pipeline.polish", side_effect=lambda x: x), \
             patch("voiceprint.pipeline.burstiness", return_value=0.15), \
             patch("voiceprint.pipeline.check_similarity", return_value=0.8):

            mock_gen.return_value = ["candidate text"]
            mock_sel.return_value = ("candidate text", 0.8)

            result = pipe.run("AI text.", use_scrub=True,
                            use_paraphrase=True, use_polish=True)

            # max_iter=2 → at most 2 detect calls
            assert call_count <= 2, (
                f"Expected <=2 detect calls (max_iter=2), got {call_count}"
            )
