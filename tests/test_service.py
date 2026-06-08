"""Tests for service layer — pure business logic, no Streamlit deps."""

import pytest
from unittest.mock import patch, MagicMock
from voiceprint.config import Config
from voiceprint.service import (
    validate_input,
    build_config,
    humanize,
    detect,
    InputError,
)


# ---------------------------------------------------------------------------
# validate_input
# ---------------------------------------------------------------------------

class TestValidateInput:
    def test_valid_text_returns_stripped(self):
        assert validate_input("  Hello world, this is long enough.  ") == "Hello world, this is long enough."

    def test_empty_text_raises(self):
        with pytest.raises(InputError, match="cannot be empty"):
            validate_input("")

    def test_none_text_raises(self):
        with pytest.raises((InputError, TypeError)):
            validate_input(None)

    def test_too_short_raises(self):
        with pytest.raises(InputError, match="too short"):
            validate_input("short")

    def test_exactly_min_length(self):
        text = "a" * 10
        assert validate_input(text) == text

    def test_too_long_raises(self):
        with pytest.raises(InputError, match="too long"):
            validate_input("a" * 100_001)


# ---------------------------------------------------------------------------
# build_config
# ---------------------------------------------------------------------------

class TestBuildConfig:
    def test_explicit_params(self):
        config = build_config(
            provider="OpenAI",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
        )
        assert config.provider == "OpenAI"
        assert config.api_key == "sk-test"
        assert config.base_url == "https://api.openai.com/v1"
        assert config.llm_model == "gpt-4o"

    def test_empty_params_uses_defaults(self):
        config = build_config()
        assert config.provider == "Google Gemini (Free)"
        assert config.llm_model == "gemini/gemini-2.0-flash"

    @patch.dict("os.environ", {"GEMINI_API_KEY": "env-key-123"})
    def test_env_var_fallback(self):
        config = build_config(provider="Google Gemini (Free)")
        assert config.api_key == "env-key-123"

    def test_explicit_key_takes_precedence(self):
        config = build_config(api_key="explicit-key", provider="Google Gemini (Free)")
        assert config.api_key == "explicit-key"

    def test_custom_provider_no_env_key(self):
        config = build_config(
            provider="Custom (OpenAI-compatible)",
            model="my-model",
            api_key="my-key",
        )
        assert config.llm_model == "my-model"
        assert config.api_key == "my-key"


# ---------------------------------------------------------------------------
# humanize (mocked pipeline)
# ---------------------------------------------------------------------------

class TestHumanize:
    def test_empty_input_raises(self):
        with pytest.raises(InputError):
            humanize("")

    def test_short_input_raises(self):
        with pytest.raises(InputError):
            humanize("short")

    @patch("voiceprint.service.HumanizePipeline")
    def test_successful_humanize(self, mock_pipe_cls):
        mock_pipe = MagicMock()
        mock_pipe_cls.return_value = mock_pipe
        mock_result = MagicMock()
        mock_result.humanized = "Humanized text here for testing."
        mock_result.original = "Original text here for testing."
        mock_result.detection.p_ai = 0.2
        mock_result.detection.summary.return_value = "0.200 HUMAN"
        mock_result.similarity = 0.85
        mock_result.burstiness = 0.55
        mock_result.pattern_score = 0.08
        mock_result.readability = {"flesch_reading_ease": 70.0}
        mock_result.burstiness_detail = {"burstiness": 0.55}
        mock_result.signals = {"ai_vocabulary": 0.02}
        mock_result.stages_applied = ["scrub", "paraphrase"]
        mock_pipe.run.return_value = mock_result

        result = humanize(
            "Original text here for testing.",
            config=Config(llm_model="gpt-4o-mini"),
        )
        assert result.success is True
        assert result.text == "Humanized text here for testing."
        assert result.ai_probability == 0.2
        assert result.similarity == 0.85

    @patch("voiceprint.service.HumanizePipeline")
    def test_pipeline_exception_returns_error(self, mock_pipe_cls):
        mock_pipe = MagicMock()
        mock_pipe_cls.return_value = mock_pipe
        mock_pipe.run.side_effect = RuntimeError("LLM API down")

        result = humanize(
            "Original text here for testing.",
            config=Config(llm_model="gpt-4o-mini"),
        )
        assert result.success is False
        assert "LLM API down" in result.error
        assert result.text == ""


# ---------------------------------------------------------------------------
# detect (mocked pipeline)
# ---------------------------------------------------------------------------

class TestDetect:
    def test_empty_input_raises(self):
        with pytest.raises(InputError):
            detect("")

    @patch("voiceprint.service.HumanizePipeline")
    def test_successful_detect(self, mock_pipe_cls):
        mock_pipe = MagicMock()
        mock_pipe_cls.return_value = mock_pipe
        mock_ensemble = MagicMock()
        mock_pipe.detect_only.return_value = mock_ensemble
        mock_ensemble.p_ai = 0.75
        mock_ensemble.passed = False
        mock_ensemble.method = "model"
        mock_ensemble.summary.return_value = "0.750 AI"
        mock_ensemble.detectors = []

        result = detect(
            "Original text here for testing.",
            config=Config(use_models=False),
        )
        assert result["p_ai"] == 0.75
        assert result["passed"] is False
        assert result["method"] == "model"
        assert "detectors" in result
