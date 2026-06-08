"""Tests for metrics module — burstiness, readability, word/char counts."""

import pytest
from voiceprint.metrics import (
    sentence_lengths,
    burstiness,
    burstiness_report,
    readability_scores,
)


# ---------------------------------------------------------------------------
# sentence_lengths
# ---------------------------------------------------------------------------

class TestSentenceLengths:
    def test_basic(self):
        result = sentence_lengths("Hello world. This is a test. Goodbye.")
        assert result == [2, 4, 1]

    def test_single_sentence(self):
        result = sentence_lengths("Hello world.")
        assert result == [2]

    def test_empty_string(self):
        assert sentence_lengths("") == []

    def test_whitespace_only(self):
        assert sentence_lengths("   ") == []

    def test_exclamation_and_question(self):
        result = sentence_lengths("Wow! Really? Yes.")
        assert result == [1, 1, 1]

    def test_long_sentence(self):
        text = "The quick brown fox jumps over the lazy dog."
        result = sentence_lengths(text)
        assert result == [9]


# ---------------------------------------------------------------------------
# burstiness
# ---------------------------------------------------------------------------

class TestBurstiness:
    def test_uniform_sentences_low_burstiness(self):
        text = "This is a test. This is a test. This is a test."
        b = burstiness(text)
        assert b < 0.1

    def test_varied_sentences_high_burstiness(self):
        text = "Short. This is a much longer sentence with many words in it."
        b = burstiness(text)
        assert b > 0.3

    def test_single_sentence_zero(self):
        assert burstiness("Hello.") == 0.0

    def test_empty_string_zero(self):
        assert burstiness("") == 0.0

    def test_human_range(self):
        """Human text should have burstiness 0.4-0.7."""
        text = (
            "The cat sat on the mat. "
            "A very long sentence with many important words follows here for testing. "
            "Short. "
            "Another moderately long sentence that has a reasonable number of words."
        )
        b = burstiness(text)
        assert 0.2 < b < 1.0  # Should be in a reasonable range


# ---------------------------------------------------------------------------
# burstiness_report
# ---------------------------------------------------------------------------

class TestBurstinessReport:
    def test_basic_report(self):
        report = burstiness_report("Hello world. This is a test.")
        assert "burstiness" in report
        assert "mean_length" in report
        assert "std" in report
        assert "min" in report
        assert "max" in report
        assert "sentence_count" in report
        assert report["sentence_count"] == 2
        assert report["min"] == 2
        assert report["max"] == 4

    def test_empty_text(self):
        report = burstiness_report("")
        assert report["burstiness"] == 0.0
        assert report["mean_length"] == 0

    def test_single_sentence(self):
        report = burstiness_report("Hello.")
        assert report["sentence_count"] == 1
        assert report["min"] == 1
        assert report["max"] == 1


# ---------------------------------------------------------------------------
# readability_scores
# ---------------------------------------------------------------------------

class TestReadabilityScores:
    def test_basic(self):
        scores = readability_scores("The cat sat on the mat. It was happy.")
        assert "flesch_reading_ease" in scores
        assert "flesch_kincaid_grade" in scores
        assert "gunning_fog" in scores
        assert "text_standard" in scores

    def test_simple_text_high_ease(self):
        scores = readability_scores("The cat sat. The dog ran.")
        assert scores["flesch_reading_ease"] > 50

    def test_complex_text_low_ease(self):
        scores = readability_scores(
            "The implementation of comprehensive algorithmic frameworks "
            "necessitates a thorough understanding of fundamental principles."
        )
        assert scores["flesch_reading_ease"] < 50



