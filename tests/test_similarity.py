"""Tests for semantic similarity gate (MiniLM-L6-v2 + Jaccard fallback)."""

from unittest.mock import patch

import pytest

from voiceprint.config import Config
from voiceprint.similarity import _jaccard_similarity, check_similarity, compute_similarity

# ---------------------------------------------------------------------------
# Mock model to avoid downloading during tests
# ---------------------------------------------------------------------------

class FakeSentenceTransformer:
    def encode(self, texts, **kwargs):
        if isinstance(texts, list):
            return [[0.1] * 384 for _ in texts]
        return [0.1] * 384


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeSimilarity:
    @patch("voiceprint.similarity._get_model", return_value=FakeSentenceTransformer())
    def test_identical_text_high_score(self, mock_model):
        score = compute_similarity("Hello world", "Hello world")
        assert score > 0.8

    @patch("voiceprint.similarity._get_model", return_value=FakeSentenceTransformer())
    def test_score_in_range(self, mock_model):
        score = compute_similarity("Some text", "Other text")
        assert -1.0 <= score <= 1.0

    def test_use_models_false_uses_jaccard(self):
        config = Config(use_models=False)
        score = compute_similarity("the cat sat", "the dog sat", config=config)
        expected = _jaccard_similarity("the cat sat", "the dog sat")
        assert score == pytest.approx(expected)


class TestCheckSimilarity:
    @patch("voiceprint.similarity._get_model", return_value=FakeSentenceTransformer())
    def test_returns_float(self, mock_model):
        score = check_similarity("Hello world", "Hello world")
        assert isinstance(score, float)

    @patch("voiceprint.similarity._get_model", return_value=FakeSentenceTransformer())
    def test_above_threshold_no_warning(self, mock_model):
        score = check_similarity("Hello world", "Hello world")
        assert score > 0.78

    def test_below_threshold_triggers_warning(self, caplog):
        """When similarity is below threshold, a warning should be logged."""
        config = Config(similarity_threshold=0.99)
        with caplog.at_level("WARNING"):
            check_similarity("completely different text A", "totally unrelated B", config=config)
        assert "below threshold" in caplog.text

    def test_config_none_uses_defaults(self):
        """When config=None, check_similarity should use load_config defaults."""
        with patch("voiceprint.config.load_config") as mock_load:
            mock_config = Config(similarity_threshold=0.78)
            mock_load.return_value = mock_config
            score = check_similarity("Hello", "Hello", config=None)
            assert isinstance(score, float)


class TestModelPathFailureFallback:
    def test_sklearn_value_error_falls_back_to_jaccard(self):
        """numpy ABI drift raises ValueError (not ImportError) — must still fall back."""
        import builtins

        real_import = builtins.__import__

        def bad_import(name, *args, **kwargs):
            if name == "sklearn.metrics.pairwise":
                raise ValueError("numpy.dtype size changed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=bad_import):
            score = compute_similarity("the cat sat", "the cat sat", config=Config())
        assert score == 1.0

    def test_broken_encode_falls_back_to_jaccard(self):
        """Encoder crash mid-path must return Jaccard, not raise."""

        class BrokenEncoder:
            def encode(self, texts, **kwargs):
                raise RuntimeError("CUDA OOM")

        with patch("voiceprint.similarity._get_model", return_value=BrokenEncoder()):
            score = compute_similarity("the cat sat", "the cat sat", config=Config())
        assert score == 1.0
