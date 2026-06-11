"""Tests for the split_long_sentences scrub rule (burstiness engineering)."""
import pytest
from voiceprint.metrics import burstiness, sentence_lengths
from voiceprint.scrub import split_long_sentences


# ---------------------------------------------------------------------------
# split_long_sentences
# ---------------------------------------------------------------------------

class TestSplitLongSentences:
    def test_short_sentence_unchanged(self):
        """Sentences under 25 words pass through untouched."""
        text = "The quick brown fox jumps over the lazy dog."
        assert split_long_sentences(text) == text

    def test_long_sentence_split(self):
        """A 35+ word sentence gets split into two shorter ones."""
        text = (
            "The research team conducted an extensive series of experiments "
            "over the course of several months and found that the new algorithm "
            "performed significantly better than the previous version in terms "
            "of both accuracy and computational efficiency across all benchmarks."
        )
        result = split_long_sentences(text)
        # Should be split — result should contain at least 2 sentences
        sentences = [s.strip() for s in result.split(".") if s.strip()]
        assert len(sentences) >= 2, f"Expected split, got: {result}"

    def test_split_improves_burstiness(self):
        """Splitting should increase burstiness of uniform-length text."""
        # Generate 10 uniform long sentences (classic AI pattern)
        long_sent = (
            "This is a moderately long sentence that contains enough words "
            "to demonstrate how AI-generated text tends to produce uniform "
            "sentence lengths that detectors can easily identify as non-human."
        )
        text = ". ".join([long_sent] * 10) + "."
        original_burstiness = burstiness(text)
        result = split_long_sentences(text)
        new_burstiness = burstiness(result)
        assert new_burstiness > original_burstiness, (
            f"Burstiness should increase: {original_burstiness:.3f} → {new_burstiness:.3f}"
        )

    def test_preserves_content(self):
        """Split sentences should preserve all original words."""
        text = (
            "The comprehensive analysis of the data revealed several important "
            "findings that were consistent with the initial hypotheses and "
            "provided strong evidence for the proposed theoretical framework."
        )
        result = split_long_sentences(text)
        # All key content words should survive
        for word in ["comprehensive", "analysis", "findings", "hypotheses", "framework"]:
            assert word in result.lower(), f"Word '{word}' missing from result"

    def test_empty_string(self):
        assert split_long_sentences("") == ""

    def test_single_word(self):
        assert split_long_sentences("Hello") == "Hello"

    def test_mixed_lengths(self):
        """Short sentences stay short, long ones get split."""
        text = "Short. " + (
            "This is a very long sentence that goes on and on with many words "
            "and should definitely be split into two parts for better rhythm."
        ) + " Also short."
        result = split_long_sentences(text)
        assert "Short." in result
        assert "Also short." in result

    def test_deterministic(self):
        """Same input always produces same output."""
        text = (
            "The implementation leverages a comprehensive set of algorithms "
            "that work together to produce seamless results across multiple "
            "platforms and use cases in the modern technological landscape."
        )
        assert split_long_sentences(text) == split_long_sentences(text)

    def test_fragment_injection_creates_short_sentences(self):
        """Long sentence gets split into varied lengths."""
        text = (
            "The development team spent months refining the algorithm to ensure "
            "it could handle edge cases gracefully while maintaining performance "
            "at scale across distributed systems with varying network conditions "
            "and hardware configurations that are common in production environments."
        )
        result = split_long_sentences(text)
        lengths = sentence_lengths(result)
        # Split should produce at least 2 sentences with varied lengths
        assert len(lengths) >= 2, f"Expected split into 2+ sentences, got lengths: {lengths}"
        # The lengths should vary (not identical)
        assert len(set(lengths)) >= 2, f"Expected varied lengths, got: {lengths}"
