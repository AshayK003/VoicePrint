"""Configuration — API keys, thresholds, model names."""

import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Config:
    # LLM Provider (set via environment variable)
    llm_model: str = "gemini/gemini-2.0-flash"
    llm_temperature: float = 1.0
    llm_max_tokens: int = 2048

    # Paraphrasing
    n_candidates: int = 8  # Best-of-N
    similarity_threshold: float = 0.78  # Min cosine similarity to original
    max_retries: int = 3

    # Detection
    primary_detector: str = "openai-community/roberta-large-openai-detector"
    secondary_detector: str = "Hello-SimpleAI/chatgpt-detector-roberta"
    detection_threshold: float = 0.5  # Below this = "human"

    # Burstiness
    burstiness_target: float = 0.55  # Human text range: 0.4-0.7
    min_sentence_words: int = 3
    max_sentence_words: int = 45

    # Paths
    cache_dir: Path = field(default_factory=lambda: Path(".cache"))


def load_config() -> Config:
    """Load configuration with environment variable overrides."""
    config = Config()

    if model := os.getenv("VOICEPRINT_LLM_MODEL"):
        config.llm_model = model
    if key := os.getenv("VOICEPRINT_SIMILARITY_THRESHOLD"):
        config.similarity_threshold = float(key)

    return config
