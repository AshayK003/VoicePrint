"""Tests for Stage 4: Style Polish rules."""

import random
import pytest
from voiceprint.polish import (
    polish,
    convert_passive_to_active,
    inject_rhetorical_questions,
    inject_fragments,
    remove_duplicates,
    normalize_punctuation,
)


# ---------------------------------------------------------------------------
# convert_passive_to_active
# ---------------------------------------------------------------------------

class TestConvertPassiveToActive:
    def test_basic_conversion(self):
        result = convert_passive_to_active("The study was conducted by the team.")
        assert result == "the team conducted the study."

    def test_case_insensitive(self):
        result = convert_passive_to_active("the data was analyzed by the lab.")
        assert result == "the lab analyzed the data."

    def test_no_passive(self):
        text = "We conducted the study."
        assert convert_passive_to_active(text) == text

    def test_multiple_passives(self):
        text = "The study was conducted by A. The experiment was performed by B."
        result = convert_passive_to_active(text)
        assert "A conducted the study" in result
        assert "B performed the experiment" in result

    def test_empty_string(self):
        assert convert_passive_to_active("") == ""


# ---------------------------------------------------------------------------
# inject_rhetorical_questions (seeded)
# ---------------------------------------------------------------------------

class TestInjectRhetoricalQuestions:
    def test_short_text_no_injection(self):
        """Fewer than 6 sentences: no injection."""
        text = "One. Two. Three. Four. Five."
        result = inject_rhetorical_questions(text)
        assert result == text

    def test_injection_at_4th_sentence(self):
        """6+ sentences: injection after 4th."""
        import re
        random.seed(42)
        text = "S1. S2. S3. S4. S5. S6."
        result = inject_rhetorical_questions(text)
        assert len(result) > len(text)
        # Should have 7 sentences now (original 6 + 1 question)
        parts = re.split(r"(?<=[.!?])\s+", result)
        assert len(parts) >= 7

    def test_deterministic_with_seed(self):
        random.seed(123)
        text = "A. B. C. D. E. F. G. H."
        result1 = inject_rhetorical_questions(text)
        random.seed(123)
        result2 = inject_rhetorical_questions(text)
        assert result1 == result2

    def test_empty_string(self):
        assert inject_rhetorical_questions("") == ""


# ---------------------------------------------------------------------------
# inject_fragments (seeded)
# ---------------------------------------------------------------------------

class TestInjectFragments:
    def test_short_text_no_injection(self):
        """Fewer than 8 sentences: no injection."""
        text = "One. Two. Three. Four. Five."
        result = inject_fragments(text)
        assert result == text

    def test_empty_string(self):
        assert inject_fragments("") == ""


# ---------------------------------------------------------------------------
# remove_duplicates
# ---------------------------------------------------------------------------

class TestRemoveDuplicates:
    def test_exact_duplicates_removed(self):
        text = "Hello world. Hello world. Different sentence."
        result = remove_duplicates(text)
        assert result.count("Hello world") == 1

    def test_case_insensitive_dedup(self):
        text = "Hello World. hello world. Different."
        result = remove_duplicates(text)
        assert result.count("Hello World") == 1
        assert result.count("hello world") == 0

    def test_no_duplicates(self):
        text = "First. Second. Third."
        result = remove_duplicates(text)
        assert result == text

    def test_empty_string(self):
        assert remove_duplicates("") == ""

    def test_single_sentence(self):
        text = "Hello."
        assert remove_duplicates(text) == text


# ---------------------------------------------------------------------------
# normalize_punctuation
# ---------------------------------------------------------------------------

class TestNormalizePunctuation:
    def test_double_spaces(self):
        result = normalize_punctuation("Hello  world.  How  are  you?")
        assert "  " not in result

    def test_double_dots(self):
        result = normalize_punctuation("Hello..")
        assert result == "Hello."

    def test_double_question_marks(self):
        result = normalize_punctuation("Really??")
        assert result == "Really?"

    def test_double_exclamation(self):
        result = normalize_punctuation("Wow!!")
        assert result == "Wow!"

    def test_trailing_whitespace(self):
        result = normalize_punctuation("  Hello.  ")
        assert result == "Hello."

    def test_empty_string(self):
        assert normalize_punctuation("") == ""


# ---------------------------------------------------------------------------
# polish (full pipeline)
# ---------------------------------------------------------------------------

class TestPolishFull:
    def test_polish_applies_all_rules(self):
        text = "The study was conducted by the team. It was found that the results are significant."
        result = polish(text)
        assert "team conducted the study" in result

    def test_polish_preserves_meaning(self):
        text = "We found that the data is valid. The results are clear."
        result = polish(text)
        assert "found" in result
        assert "data" in result
        assert "results" in result

    def test_polish_empty_string(self):
        assert polish("") == ""

    def test_polish_single_sentence(self):
        text = "Hello world."
        result = polish(text)
        assert result == "Hello world."
