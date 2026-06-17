"""Tests for paraphrase module — LLM integration layer (all mocked)."""

import pytest
from unittest.mock import patch, MagicMock
from voiceprint.config import Config
from voiceprint.paraphrase import (
    _litellm_kwargs,
    generate_candidate,
    generate_candidates,
    select_best,
    PARAPHRASE_PROMPT,
)


# ---------------------------------------------------------------------------
# _litellm_kwargs
# ---------------------------------------------------------------------------

class TestLitellmKwargs:
    def test_basic_kwargs(self):
        config = Config(llm_model="gpt-4o-mini", api_key="sk-test", llm_max_tokens=1024)
        kwargs = _litellm_kwargs(config, temperature=0.9)
        assert kwargs["model"] == "gpt-4o-mini"
        assert kwargs["temperature"] == 0.9
        assert kwargs["max_tokens"] == 1024
        assert kwargs["api_key"] == "sk-test"

    def test_empty_model_raises(self):
        config = Config(llm_model="")
        with pytest.raises(ValueError, match="Model name is empty"):
            _litellm_kwargs(config, temperature=1.0)

    def test_whitespace_model_raises(self):
        config = Config(llm_model="   ")
        with pytest.raises(ValueError, match="Model name is empty"):
            _litellm_kwargs(config, temperature=1.0)

    def test_base_url_included(self):
        config = Config(llm_model="custom/model", base_url="https://api.example.com/v1")
        kwargs = _litellm_kwargs(config, temperature=1.0)
        assert kwargs["base_url"] == "https://api.example.com/v1"

    def test_no_base_url_omitted(self):
        config = Config(llm_model="gpt-4o-mini", base_url="")
        kwargs = _litellm_kwargs(config, temperature=1.0)
        assert "base_url" not in kwargs

    def test_api_key_stripped(self):
        config = Config(llm_model="gpt-4o-mini", api_key="  sk-test  ")
        kwargs = _litellm_kwargs(config, temperature=1.0)
        assert kwargs["api_key"] == "sk-test"

    def test_no_api_key_omitted(self):
        config = Config(llm_model="gpt-4o-mini", api_key="")
        kwargs = _litellm_kwargs(config, temperature=1.0)
        # api_key omitted when empty (env/registry fallback handled by build_config)
        assert "api_key" not in kwargs


# ---------------------------------------------------------------------------
# generate_candidate
# ---------------------------------------------------------------------------

class TestGenerateCandidate:
    @patch("litellm.completion")
    def test_returns_stripped_response(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="  Humanized text.  "))]
        mock_completion.return_value = mock_response

        config = Config(llm_model="gpt-4o-mini", api_key="sk-test")
        result = generate_candidate("Original text.", config=config)
        assert result == "Humanized text."
        mock_completion.assert_called_once()

    @patch("litellm.completion")
    def test_prompt_contains_original_text(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Paraphrased."))]
        mock_completion.return_value = mock_response

        config = Config(llm_model="gpt-4o-mini", api_key="sk-test")
        generate_candidate("Test input text", config=config)

        call_kwargs = mock_completion.call_args
        prompt_content = call_kwargs.kwargs["messages"][0]["content"]
        assert "Test input text" in prompt_content

    @patch("litellm.completion")
    def test_custom_temperature_passed(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Done."))]
        mock_completion.return_value = mock_response

        config = Config(llm_model="gpt-4o-mini", api_key="sk-test")
        generate_candidate("Text", config=config, temperature=0.7)

        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["temperature"] == 0.7

    @patch("litellm.completion")
    def test_prev_p_ai_includes_feedback(self, mock_completion):
        """Detection-guided refinement: prev_p_ai should add feedback to prompt."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Rewritten."))]
        mock_completion.return_value = mock_response

        config = Config(llm_model="gpt-4o-mini", api_key="sk-test")
        generate_candidate("Original text.", config=config, prev_p_ai=0.35)

        call_kwargs = mock_completion.call_args
        prompt_content = call_kwargs.kwargs["messages"][0]["content"]
        assert "Previous attempt scored" in prompt_content
        assert "0.35" in prompt_content

    @patch("litellm.completion")
    def test_prev_p_ai_high_escalates_tone(self, mock_completion):
        """prev_p_ai ≥ 0.6 should add 'clearly AI' feedback."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Rewritten."))]
        mock_completion.return_value = mock_response

        config = Config(llm_model="gpt-4o-mini", api_key="sk-test")
        generate_candidate("Text.", config=config, prev_p_ai=0.75)

        call_kwargs = mock_completion.call_args
        prompt_content = call_kwargs.kwargs["messages"][0]["content"]
        assert "clearly AI-detected" in prompt_content
        assert "Rewrite from scratch" in prompt_content

    @patch("litellm.completion")
    def test_prev_p_ai_none_no_feedback(self, mock_completion):
        """When prev_p_ai is None, no feedback line should appear."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Rewritten."))]
        mock_completion.return_value = mock_response

        config = Config(llm_model="gpt-4o-mini", api_key="sk-test")
        generate_candidate("Text.", config=config, prev_p_ai=None)

        call_kwargs = mock_completion.call_args
        prompt_content = call_kwargs.kwargs["messages"][0]["content"]
        assert "Previous attempt scored" not in prompt_content


# ---------------------------------------------------------------------------
# generate_candidates
# ---------------------------------------------------------------------------

class TestGenerateCandidates:
    @patch("voiceprint.paraphrase.generate_candidate")
    def test_returns_n_candidates(self, mock_gen):
        mock_gen.side_effect = lambda text, config, temperature, **kw: f"Candidate {temperature}"
        config = Config(llm_model="gpt-4o-mini", n_candidates=3)
        candidates = generate_candidates("Text", n=3, config=config)
        assert len(candidates) == 3
        assert mock_gen.call_count == 3

    @patch("voiceprint.paraphrase.generate_candidate")
    def test_filters_none_results(self, mock_gen):
        mock_gen.side_effect = ["OK", None, "OK"]
        config = Config(llm_model="gpt-4o-mini", n_candidates=3)
        candidates = generate_candidates("Text", n=3, config=config)
        assert len(candidates) == 2

    @patch("voiceprint.paraphrase.generate_candidate")
    def test_all_failures_returns_empty(self, mock_gen):
        mock_gen.side_effect = Exception("API error")
        config = Config(llm_model="gpt-4o-mini", n_candidates=2)
        candidates = generate_candidates("Text", n=2, config=config)
        assert candidates == []

    @patch("voiceprint.paraphrase.generate_candidate")
    def test_default_n_from_config(self, mock_gen):
        mock_gen.return_value = "Candidate"
        config = Config(llm_model="gpt-4o-mini", n_candidates=4)
        candidates = generate_candidates("Text", config=config)
        assert len(candidates) == 4


# ---------------------------------------------------------------------------
# select_best
# ---------------------------------------------------------------------------

class TestSelectBest:
    def _mock_detector(self, p_ai_scores):
        """Create a mock detector that returns p_ai values in order."""
        from voiceprint.detect import EnsembleResult, DetectionResult
        mock = MagicMock()
        results = []
        for p in p_ai_scores:
            results.append(EnsembleResult(
                p_ai=p,
                detectors=[DetectionResult(name="test", p_ai=p, label="test")],
                passed=p < 0.5,
            ))
        mock.detect.side_effect = results
        return mock

    @patch("voiceprint.paraphrase.check_similarity")
    def test_picks_lowest_pai_above_min_similarity(self, mock_sim):
        sim_map = {"Candidate A": 0.85, "Candidate B": 0.92, "Candidate C": 0.80}
        mock_sim.side_effect = lambda orig, cand, config=None: sim_map.get(cand, 0.5)
        # Detector returns p_ai in call order: B=0.7, A=0.3, C=0.5
        detector = self._mock_detector([0.7, 0.3, 0.5])
        best, score = select_best(
            "Original",
            ["Candidate A", "Candidate B", "Candidate C"],
            config=Config(),
            detector=detector,
        )
        # Sorted by sim desc: B(0.92), A(0.85), C(0.80). Detection order: B, A, C.
        # B gets p_ai=0.7, A gets p_ai=0.3 (lowest), C gets p_ai=0.5
        assert best == "Candidate A"  # Lowest p_ai (0.3) among top 3

    @patch("voiceprint.paraphrase.check_similarity")
    def test_fallback_picks_lowest_pai_when_all_below_min_sim(self, mock_sim):
        sim_map = {"Candidate A": 0.35, "Candidate B": 0.40, "Candidate C": 0.45}
        mock_sim.side_effect = lambda orig, cand, config=None: sim_map.get(cand, 0.5)
        detector = self._mock_detector([0.7, 0.3, 0.5])
        best, score = select_best(
            "Original",
            ["Candidate A", "Candidate B", "Candidate C"],
            config=Config(),
            detector=detector,
        )
        assert best == "Candidate B"  # Lowest p_ai (0.3)

    def test_empty_candidates_returns_original(self):
        config = Config()
        best, score = select_best("Original", [], config=config)
        assert best == "Original"
        assert score == 1.0

    @patch("voiceprint.paraphrase.check_similarity")
    def test_single_candidate(self, mock_sim):
        mock_sim.return_value = 0.88
        detector = self._mock_detector([0.4])
        best, score = select_best("Original", ["Only one"], config=Config(), detector=detector)
        assert best == "Only one"
        assert score == 0.88

    @patch("voiceprint.paraphrase.check_similarity")
    @patch("voiceprint.perplexity.raw_perplexity")
    def test_rejects_low_perplexity_candidates(self, mock_ppl, mock_sim):
        """Candidates with perplexity < 30 should be filtered out."""
        mock_sim.return_value = 0.85
        mock_ppl.side_effect = [25.0, 45.0, 20.0]
        detector = self._mock_detector([0.3])
        best, score = select_best(
            "Original",
            ["Low ppl A", "Mid ppl B", "Low ppl C"],
            config=Config(),
            detector=detector,
        )
        assert best == "Mid ppl B"

    @patch("voiceprint.paraphrase.check_similarity")
    @patch("voiceprint.perplexity.raw_perplexity")
    def test_perplexity_all_low_falls_back_to_all(self, mock_ppl, mock_sim):
        """If all candidates fail perplexity check, fall back to full list."""
        mock_sim.return_value = 0.85
        mock_ppl.side_effect = [20.0, 15.0, 10.0]
        detector = self._mock_detector([0.3, 0.5, 0.2])
        best, score = select_best(
            "Original",
            ["Low A", "Low B", "Low C"],
            config=Config(),
            detector=detector,
        )
        assert best is not None

    @patch("voiceprint.paraphrase.check_similarity")
    @patch("voiceprint.perplexity.raw_perplexity")
    def test_perplexity_none_does_not_filter(self, mock_ppl, mock_sim):
        """When perplexity returns None (model unavailable), no filtering."""
        mock_sim.return_value = 0.85
        mock_ppl.return_value = None
        detector = self._mock_detector([0.3])
        best, score = select_best(
            "Original",
            ["Candidate A"],
            config=Config(),
            detector=detector,
        )
        assert best == "Candidate A"

    @patch("voiceprint.paraphrase.check_similarity")
    def test_detector_exception_skips_failed_candidate(self, mock_sim):
        mock_sim.side_effect = [0.85, 0.80]
        detector = MagicMock()
        from voiceprint.detect import EnsembleResult
        detector.detect.side_effect = [Exception("crash"), EnsembleResult(p_ai=0.3, detectors=[], passed=True)]
        best, score = select_best(
            "Original",
            ["Candidate A", "Candidate B"],
            config=Config(),
            detector=detector,
        )
        assert best == "Candidate B"


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

class TestPrompt:
    def test_prompt_contains_placeholder(self):
        assert "{text}" in PARAPHRASE_PROMPT

    def test_prompt_format_works(self):
        formatted = PARAPHRASE_PROMPT.format(text="Test input")
        assert "Test input" in formatted
        assert "{text}" not in formatted
