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
        "model": "openai/nemotron-3-ultra-free",
        "base_url": "https://opencode.ai/zen/v1",
        "env_key": "OPENCODE_API_KEY",
    },
    "Lightning AI": {
        "model": "openai/gpt-4o-mini",
        "base_url": "https://lightning.ai/api/v1",
        "env_key": "LIGHTNING_API_KEY",
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
        "gemini/gemini-1.5-flash",
        "gemini/gemini-1.5-pro",
    ],
    "OpenAI": [
        "gpt-4o-mini",
        "gpt-4.1-mini",
        "o4-mini",
        "gpt-4o",
        "gpt-5-mini",
        "gpt-5",
    ],
    "Anthropic": [
        "claude-3-haiku-20240307",
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20241022",
        "claude-4-sonnet",
    ],
    "Groq (Free)": [
        "groq/llama-3.3-70b-versatile",
        "groq/llama-4-scout-17b-16e-instruct",
        "groq/llama-4-maverick-17b-128e-instruct",
        "groq/qwen-2.5-32b",
        "groq/deepseek-r1-distill-llama-70b",
        "groq/mixtral-8x7b-32768",
    ],
    "Mistral (Free)": [
        "mistral/mistral-tiny",
        "mistral/open-mistral-nemo",
        "mistral/mistral-small-latest",
        "mistral/mistral-medium-latest",
        "mistral/mistral-large-latest",
        "mistral/codestral-latest",
    ],
    "OpenCode Zen": [
        "openai/nemotron-3-ultra-free",
        "openai/mimo-v2.5-free",
        "openai/llama-3.3-70b-arcee-free",
        "openai/deepseek-v3-0615-free",
        "openai/qwen-3-235b-a22b-free",
    ],
    "Lightning AI": [
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "openai/gpt-5",
        "anthropic/claude-3-5-sonnet-20241022",
        "google/gemini-2.0-flash",
        "google/gemini-2.5-flash",
        "mistral/mixtral-8x22b",
        "meta-llama/llama-3.1-70b",
        "deepseek/deepseek-chat",
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
    "Lightning AI": [
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
    provider: str = "OpenCode Zen"
    api_key: str = ""
    base_url: str = "https://opencode.ai/zen/v1"
    llm_model: str = "openai/nemotron-3-ultra-free"
    llm_temperature: float = 1.0
    llm_max_tokens: int = 2048

    # Paraphrasing
    n_candidates: int = 2  # Best-of-N (increase for better evasion at cost of speed)
    similarity_threshold: float = 0.65  # Min cosine similarity to original (lower = more transformation allowed)
    max_iterations: int = 5  # Max paraphrase→polish→detect retry cycles

    # Detection
    # Clause restructuring
    use_restructure: bool = True  # Enable/disable syntactic clause restructuring
    restructure_probability: float = 0.6  # Per-rule application probability per eligible sentence

    # Detection
    primary_detector: str = "openai-community/roberta-large-openai-detector"
    secondary_detector: str = "Hello-SimpleAI/chatgpt-detector-roberta"
    detection_threshold: float = 0.5  # Below this = "human"
    use_models: bool = True  # False = skip all model loading, use heuristics only

    # Trained model (Phase 2 — fine-tuned on AI→Human style transfer)
    trained_model_path: str = ""  # Path to GGUF file; "" = use default

    def __post_init__(self) -> None:
        """No automatic env/registry fallback here.
        That is handled by build_config() and load_config() explicitly.
        """


def _read_registry_env(name: str) -> str:
    """Fallback: read User-level env var from Windows registry."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            return winreg.QueryValueEx(key, name)[0]
    except (ImportError, FileNotFoundError, OSError):
        return ""


def load_config() -> Config:
    """Load configuration with environment variable overrides."""
    config = Config()

    if model := os.getenv("VOICEPRINT_LLM_MODEL"):
        config.llm_model = model
    if key := os.getenv("VOICEPRINT_SIMILARITY_THRESHOLD"):
        config.similarity_threshold = float(key)

    # API key resolution: env var first, then Windows registry fallback.
    # Does NOT mutate os.environ — avoids side effects for other callers.
    if not config.api_key:
        config.api_key = os.getenv("OPENCODE_API_KEY", "")
    if not config.api_key:
        config.api_key = _read_registry_env("OPENCODE_API_KEY")

    # Trained model path from env var
    if model_path := os.getenv("VOICEPRINT_HUMANIZER_MODEL"):
        config.trained_model_path = model_path

    return config


# ---------------------------------------------------------------------------
# Provider auto-detection from API key prefix
# ---------------------------------------------------------------------------

def detect_provider_from_key(api_key: str) -> dict[str, str] | None:
    """Detect provider, model, and base URL from API key prefix.

    Uses PROVIDER_PRESETS for model/base URL — single source of truth.
    Returns dict with provider, model, base_url keys — or None if unknown.
    Pure function, no side effects.
    """
    key = api_key.strip()

    if key.startswith("AIza"):
        provider_name = "Google Gemini (Free)"
    elif key.startswith("sk-ant-"):
        provider_name = "Anthropic"
    elif key.startswith("gsk_"):
        provider_name = "Groq (Free)"
    elif key.startswith("ak-"):
        provider_name = "Mistral (Free)"
    elif key.startswith("opc-") or key.startswith("oc-"):
        provider_name = "OpenCode Zen"
    elif key.startswith("sk-lit-"):
        provider_name = "Lightning AI"
    elif key.startswith("sk-"):
        # OpenAI keys are typically sk-... with ~51 chars
        # OpenCode Zen uses longer keys when behind opencodedotai prefix
        provider_name = "OpenCode Zen" if len(key) > 60 else "OpenAI"
    else:
        return None

    preset = PROVIDER_PRESETS.get(provider_name)
    if not preset:
        return None
    return {
        "provider": provider_name,
        "model": preset["model"],
        "base_url": preset["base_url"],
    }


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
