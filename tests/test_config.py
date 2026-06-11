"""Tests for config module — provider presets, Config dataclass."""

import pytest
from voiceprint.config import (
    Config,
    PROVIDER_PRESETS,
    PROVIDER_MODELS,
    PROVIDER_BASE_URLS,
    load_config,
    detect_provider_from_key,
    validate_config,
    ConfigError,
)


# ---------------------------------------------------------------------------
# PROVIDER_PRESETS
# ---------------------------------------------------------------------------

class TestProviderPresets:
    def test_has_known_providers(self):
        assert "Google Gemini (Free)" in PROVIDER_PRESETS
        assert "Groq (Free)" in PROVIDER_PRESETS
        assert "OpenAI" in PROVIDER_PRESETS
        assert "Anthropic" in PROVIDER_PRESETS
        assert "Mistral (Free)" in PROVIDER_PRESETS

    def test_each_preset_has_required_keys(self):
        for name, preset in PROVIDER_PRESETS.items():
            assert "model" in preset, f"{name} missing 'model'"
            assert "base_url" in preset, f"{name} missing 'base_url'"
            assert "env_key" in preset, f"{name} missing 'env_key'"


# ---------------------------------------------------------------------------
# PROVIDER_MODELS
# ---------------------------------------------------------------------------

class TestProviderModels:
    def test_keys_match_presets(self):
        for key in PROVIDER_MODELS:
            assert key in PROVIDER_PRESETS, f"Model key '{key}' not in presets"

    def test_all_lists_nonempty(self):
        for name, models in PROVIDER_MODELS.items():
            assert len(models) > 0, f"{name} has empty model list"

    def test_gemini_has_flash(self):
        models = PROVIDER_MODELS["Google Gemini (Free)"]
        assert any("gemini-2.0-flash" in m for m in models)


# ---------------------------------------------------------------------------
# PROVIDER_BASE_URLS
# ---------------------------------------------------------------------------

class TestProviderBaseUrls:
    def test_keys_match_presets(self):
        for key in PROVIDER_BASE_URLS:
            assert key in PROVIDER_PRESETS, f"Base URL key '{key}' not in presets"

    def test_all_lists_nonempty(self):
        for name, urls in PROVIDER_BASE_URLS.items():
            assert len(urls) > 0, f"{name} has empty base URL list"


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

class TestConfig:
    def test_defaults(self):
        config = Config()
        assert config.provider == "OpenCode Zen"
        assert config.llm_model == "openai/nemotron-3-ultra-free"
        assert config.base_url == "https://opencode.ai/zen/v1"
        assert config.n_candidates == 2
        assert config.similarity_threshold == 0.65
        assert config.detection_threshold == 0.5

    def test_custom_values(self):
        config = Config(
            provider="Groq (Free)",
            llm_model="groq/llama-3.3-70b-versatile",
            n_candidates=5,
        )
        assert config.provider == "Groq (Free)"
        assert config.n_candidates == 5


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_returns_config(self):
        config = load_config()
        assert isinstance(config, Config)

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("VOICEPRINT_LLM_MODEL", "custom/model")
        config = load_config()
        assert config.llm_model == "custom/model"

    def test_no_env_uses_default(self, monkeypatch):
        monkeypatch.delenv("VOICEPRINT_LLM_MODEL", raising=False)
        config = load_config()
        assert config.llm_model == "openai/nemotron-3-ultra-free"

    def test_similarity_threshold_env_override(self, monkeypatch):
        monkeypatch.setenv("VOICEPRINT_SIMILARITY_THRESHOLD", "0.85")
        config = load_config()
        assert config.similarity_threshold == 0.85

    def test_invalid_similarity_threshold_env(self, monkeypatch):
        monkeypatch.setenv("VOICEPRINT_SIMILARITY_THRESHOLD", "not-a-number")
        with pytest.raises(ValueError):
            load_config()


# ---------------------------------------------------------------------------
# detect_provider_from_key
# ---------------------------------------------------------------------------

class TestDetectProviderFromKey:
    def test_gemini_key(self):
        result = detect_provider_from_key("AIzaSyD...")
        assert result is not None
        assert result["provider"] == "Google Gemini (Free)"
        assert result["model"] == "gemini/gemini-2.0-flash"

    def test_openai_key(self):
        result = detect_provider_from_key("sk-abc123...")
        assert result is not None
        assert result["provider"] == "OpenAI"
        assert result["model"] == "gpt-4o-mini"

    def test_anthropic_key(self):
        result = detect_provider_from_key("sk-ant-api03-...")
        assert result is not None
        assert result["provider"] == "Anthropic"
        assert result["model"] == "claude-3-5-haiku-20241022"

    def test_openai_not_mistaken_for_anthropic(self):
        result = detect_provider_from_key("sk-proj-abc123")
        assert result is not None
        assert result["provider"] == "OpenAI"

    def test_groq_key(self):
        result = detect_provider_from_key("gsk_abc123...")
        assert result is not None
        assert result["provider"] == "Groq (Free)"
        assert "llama" in result["model"]

    def test_mistral_key(self):
        result = detect_provider_from_key("ak-abc123...")
        assert result is not None
        assert result["provider"] == "Mistral (Free)"
        assert "mistral" in result["model"]

    def test_unknown_key_returns_none(self):
        assert detect_provider_from_key("random-key-12345") is None

    def test_empty_key_returns_none(self):
        assert detect_provider_from_key("") is None

    def test_whitespace_key_returns_none(self):
        assert detect_provider_from_key("   ") is None

    def test_leading_whitespace(self):
        result = detect_provider_from_key("  AIzaSyD...")
        assert result is not None
        assert result["provider"] == "Google Gemini (Free)"


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------

class TestValidateConfig:
    def test_valid_config_passes(self):
        validate_config(Config())  # Should not raise

    def test_empty_model_raises(self):
        with pytest.raises(ConfigError, match="Model name is required"):
            validate_config(Config(llm_model=""))

    def test_zero_candidates_raises(self):
        with pytest.raises(ConfigError, match="n_candidates"):
            validate_config(Config(n_candidates=0))

    def test_zero_iterations_raises(self):
        with pytest.raises(ConfigError, match="max_iterations"):
            validate_config(Config(max_iterations=0))
