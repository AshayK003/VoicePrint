"""Configuration — API keys, thresholds, model names."""

import os
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Provider presets — common free/cheap providers
# ---------------------------------------------------------------------------

PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "Google Gemini (Free)": {
        "model": "gemini/gemini-2.0-flash",
        "base_url": "",
        "env_key": "GEMINI_API_KEY",
    },
    "Groq (Free)": {
        "model": "groq/llama-3.3-70b-versatile",
        "base_url": "",
        "env_key": "GROQ_API_KEY",
    },
    "Mistral (Free)": {
        "model": "mistral/mistral-large-latest",
        "base_url": "",
        "env_key": "MISTRAL_API_KEY",
    },
    "OpenAI": {
        "model": "gpt-4o-mini",
        "base_url": "",
        "env_key": "OPENAI_API_KEY",
    },
    "Anthropic": {
        "model": "claude-3-5-haiku-20241022",
        "base_url": "",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "OpenCode Zen": {
        "model": "opencode/mimo-v2.5-free",
        "base_url": "",
        "env_key": "OPENCODE_API_KEY",
    },
    "Custom (OpenAI-compatible)": {
        "model": "",
        "base_url": "",
        "env_key": "",
    },
}


# ---------------------------------------------------------------------------
# Common models per provider (for dropdown suggestions)
# ---------------------------------------------------------------------------

PROVIDER_MODELS: dict[str, list[str]] = {
    "Google Gemini (Free)": [
        "gemini/gemini-2.0-flash",
        "gemini/gemini-2.0-flash-lite",
        "gemini/gemini-1.5-flash",
        "gemini/gemini-1.5-pro",
    ],
    "OpenAI": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-5-mini",
        "gpt-5",
    ],
    "Anthropic": [
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20241022",
        "claude-3-haiku-20240307",
        "claude-3-opus-20240229",
    ],
    "Groq (Free)": [
        "groq/llama-3.3-70b-versatile",
        "groq/llama-4-scout-17b-16e-instruct",
        "groq/qwen-3-32b",
        "groq/llama-3.1-8b-instant",
        "groq/deepseek-r1-distill-llama-70b",
    ],
    "Mistral (Free)": [
        "mistral/mistral-large-latest",
        "mistral/mistral-medium-latest",
        "mistral/mistral-small-latest",
        "mistral/open-mistral-nemo",
        "mistral/codestral-latest",
    ],
    "OpenCode Zen": [
        "opencode/mimo-v2.5-free",
        "opencode/mimo-v2.5-pro",
    ],
    "Custom (OpenAI-compatible)": [
        "",
    ],
}


# ---------------------------------------------------------------------------
# Common base URLs per provider (dropdown suggestions)
# ---------------------------------------------------------------------------

PROVIDER_BASE_URLS: dict[str, list[str]] = {
    "Google Gemini (Free)": [
        "(default)",
    ],
    "OpenAI": [
        "(default)",
        "https://api.openai.com/v1",
    ],
    "Anthropic": [
        "(default)",
    ],
    "Groq (Free)": [
        "(default)",
    ],
    "Mistral (Free)": [
        "(default)",
    ],
    "OpenCode Zen": [
        "(default)",
    ],
    "Custom (OpenAI-compatible)": [
        "(default)",
        "https://api.openai.com/v1",
    ],
}


@dataclass
class Config:
    # LLM Provider
    provider: str = "Google Gemini (Free)"
    api_key: str = ""
    base_url: str = ""
    llm_model: str = "gemini/gemini-2.0-flash"
    llm_temperature: float = 1.0
    llm_max_tokens: int = 2048

    # Paraphrasing
    n_candidates: int = 8  # Best-of-N (8 for better evasion diversity)
    similarity_threshold: float = 0.68  # Min cosine similarity to original (lower = more transformation allowed)
    max_iterations: int = 2  # Max paraphrase→polish→detect retry cycles

    # Detection
    primary_detector: str = "openai-community/roberta-large-openai-detector"
    secondary_detector: str = "Hello-SimpleAI/chatgpt-detector-roberta"
    detection_threshold: float = 0.5  # Below this = "human"
    use_models: bool = True  # False = skip all model loading, use heuristics only


def load_config() -> Config:
    """Load configuration with environment variable overrides."""
    config = Config()

    if model := os.getenv("VOICEPRINT_LLM_MODEL"):
        config.llm_model = model
    if key := os.getenv("VOICEPRINT_SIMILARITY_THRESHOLD"):
        config.similarity_threshold = float(key)

    return config


# ---------------------------------------------------------------------------
# Provider auto-detection from API key prefix
# ---------------------------------------------------------------------------

def detect_provider_from_key(api_key: str) -> dict[str, str] | None:
    """Detect provider, model, and base URL from API key prefix.

    Returns dict with provider, model, base_url keys — or None if unknown.
    Pure function, no side effects.
    """
    key = api_key.strip()

    if key.startswith("AIza"):
        return {
            "provider": "Google Gemini (Free)",
            "model": "gemini/gemini-2.0-flash",
            "base_url": "",
        }
    if key.startswith("sk-") and not key.startswith("sk-ant-"):
        return {
            "provider": "OpenAI",
            "model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1",
        }
    if key.startswith("sk-ant-"):
        return {
            "provider": "Anthropic",
            "model": "claude-3-5-haiku-20241022",
            "base_url": "",
        }
    if key.startswith("gsk_"):
        return {
            "provider": "Groq (Free)",
            "model": "groq/llama-3.3-70b-versatile",
            "base_url": "",
        }
    if key.startswith("ak-"):
        return {
            "provider": "Mistral (Free)",
            "model": "mistral/mistral-large-latest",
            "base_url": "",
        }

    return None


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    """Raised when configuration is invalid."""


def validate_config(config: Config) -> None:
    """Validate a Config instance. Raises ConfigError on problems."""
    if not config.llm_model or not config.llm_model.strip():
        raise ConfigError("Model name is required.")
    if config.n_candidates < 1:
        raise ConfigError("n_candidates must be at least 1.")
    if config.max_iterations < 1:
        raise ConfigError("max_iterations must be at least 1.")
    if not 0.0 <= config.similarity_threshold <= 1.0:
        raise ConfigError("similarity_threshold must be between 0.0 and 1.0.")
    if not 0.0 <= config.detection_threshold <= 1.0:
        raise ConfigError("detection_threshold must be between 0.0 and 1.0.")
