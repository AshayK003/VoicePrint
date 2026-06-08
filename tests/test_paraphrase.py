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
        assert "api_key" not in kwargs


# ---------------------------------------------------------------------------
# generate_candidate
# ---------------------------------------------------------------------------

class TestGenerateCandidate:
    @patch("voiceprint.paraphrase.litellm.completion")
    def test_returns_stripped_response(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="  Humanized text.  "))]
        mock_completion.return_value = mock_response

        config = Config(llm_model="gpt-4o-mini", api_key="sk-test")
        result = generate_candidate("Original text.", config=config)
        assert result == "Humanized text."
        mock_completion.assert_called_once()

    @patch("voiceprint.paraphrase.litellm.completion")
    def test_prompt_contains_original_text(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Paraphrased."))]
        mock_completion.return_value = mock_response

        config = Config(llm_model="gpt-4o-mini", api_key="sk-test")
        generate_candidate("Test input text", config=config)

        call_kwargs = mock_completion.call_args
        prompt_content = call_kwargs.kwargs["messages"][0]["content"]
        assert "Test input text" in prompt_content

    @patch("voiceprint.paraphrase.litellm.completion")
    def test_custom_temperature_passed(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Done."))]
        mock_completion.return_value = mock_response

        config = Config(llm_model="gpt-4o-mini", api_key="sk-test")
        generate_candidate("Text", config=config, temperature=0.7)

        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["temperature"] == 0.7


# ---------------------------------------------------------------------------
# generate_candidates
# ---------------------------------------------------------------------------

class TestGenerateCandidates:
    @patch("voiceprint.paraphrase.generate_candidate")
    def test_returns_n_candidates(self, mock_gen):
        mock_gen.side_effect = lambda text, config, temperature: f"Candidate {temperature}"
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
    def test_detector_exception_uses_neutral_pai(self, mock_sim):
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
