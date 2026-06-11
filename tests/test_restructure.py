"""Tests for Stage 2b: Clause Restructure rules."""

from voiceprint.restructure import (
    apply_restructure,
    front_subordinate_clauses,
    extract_relative_clauses,
    swap_main_subordinate,
    convert_appositives,
    split_compounds,
    normalize_burstiness,
)


# ---------------------------------------------------------------------------
# front_subordinate_clauses
# ---------------------------------------------------------------------------

class TestFrontSubordinateClauses:
    def test_basic_fronting(self):
        text = "The results were significant because the sample size was large."
        result = front_subordinate_clauses(text, _prob=1.0)
        assert result != text
        assert "Because" in result
        assert "the sample size was large" in result

    def test_no_subordinate_clause(self):
        text = "The results were significant."
        result = front_subordinate_clauses(text, _prob=1.0)
        assert result == text

    def test_short_sentence_skipped(self):
        text = "It was good."
        result = front_subordinate_clauses(text, _prob=1.0)
        assert result == text

    def test_empty_string(self):
        assert front_subordinate_clauses("", _prob=1.0) == ""

    def test_deterministic(self):
        text = "The results were significant because the sample size was large."
        r1 = front_subordinate_clauses(text, _prob=1.0)
        r2 = front_subordinate_clauses(text, _prob=1.0)
        assert r1 == r2


# ---------------------------------------------------------------------------
# extract_relative_clauses
# ---------------------------------------------------------------------------

class TestExtractRelativeClauses:
    def test_basic_extraction(self):
        text = "The study, which was conducted by Smith, showed promising results."
        result = extract_relative_clauses(text, _prob=1.0)
        assert result != text

    def test_no_relative_clause(self):
        text = "The study showed promising results."
        result = extract_relative_clauses(text, _prob=1.0)
        assert result == text

    def test_short_sentence_skipped(self):
        text = "It was good."
        result = extract_relative_clauses(text, _prob=1.0)
        assert result == text

    def test_empty_string(self):
        assert extract_relative_clauses("", _prob=1.0) == ""

    def test_deterministic(self):
        text = "The study, which was conducted by Smith, showed promising results."
        r1 = extract_relative_clauses(text, _prob=1.0)
        r2 = extract_relative_clauses(text, _prob=1.0)
        assert r1 == r2


# ---------------------------------------------------------------------------
# swap_main_subordinate
# ---------------------------------------------------------------------------

class TestSwapMainSubordinate:
    def test_basic_swap(self):
        text = "Although the data was noisy, the team proceeded with the analysis."
        result = swap_main_subordinate(text, _prob=1.0)
        assert result != text

    def test_no_subordinate_first(self):
        text = "The team proceeded with the analysis."
        result = swap_main_subordinate(text, _prob=1.0)
        assert result == text

    def test_short_sentence_skipped(self):
        text = "It was good."
        result = swap_main_subordinate(text, _prob=1.0)
        assert result == text

    def test_empty_string(self):
        assert swap_main_subordinate("", _prob=1.0) == ""

    def test_deterministic(self):
        text = "Although the data was noisy, the team proceeded with the analysis."
        r1 = swap_main_subordinate(text, _prob=1.0)
        r2 = swap_main_subordinate(text, _prob=1.0)
        assert r1 == r2


# ---------------------------------------------------------------------------
# convert_appositives
# ---------------------------------------------------------------------------

class TestConvertAppositives:
    def test_basic_conversion(self):
        text = "Dr. Smith, the lead researcher, presented the findings."
        result = convert_appositives(text, _prob=1.0)
        assert result != text

    def test_no_appositive(self):
        text = "Dr. Smith presented the findings."
        result = convert_appositives(text, _prob=1.0)
        assert result == text

    def test_short_sentence_skipped(self):
        text = "It was good."
        result = convert_appositives(text, _prob=1.0)
        assert result == text

    def test_empty_string(self):
        assert convert_appositives("", _prob=1.0) == ""

    def test_deterministic(self):
        text = "Dr. Smith, the lead researcher, presented the findings."
        r1 = convert_appositives(text, _prob=1.0)
        r2 = convert_appositives(text, _prob=1.0)
        assert r1 == r2


# ---------------------------------------------------------------------------
# split_compounds
# ---------------------------------------------------------------------------

class TestSplitCompounds:
    def test_basic_split(self):
        text = "The results were significant, and the team celebrated their success."
        result = split_compounds(text, _prob=1.0)
        assert result != text

    def test_no_compound(self):
        text = "The results were significant."
        result = split_compounds(text, _prob=1.0)
        assert result == text

    def test_short_sentence_skipped(self):
        text = "It was good."
        result = split_compounds(text, _prob=1.0)
        assert result == text

    def test_empty_string(self):
        assert split_compounds("", _prob=1.0) == ""

    def test_deterministic(self):
        text = "The results were significant, and the team celebrated their success."
        r1 = split_compounds(text, _prob=1.0)
        r2 = split_compounds(text, _prob=1.0)
        assert r1 == r2


# ---------------------------------------------------------------------------
# normalize_burstiness
# ---------------------------------------------------------------------------

class TestNormalizeBurstiness:
    def test_long_sentence_split(self):
        text = ("This is an extremely long sentence that just keeps going on "
                "and on without any sign of stopping or even slowing down at "
                "all because it needs to exceed the thirty word threshold for "
                "splitting into multiple shorter sentences for burstiness." * 2)
        result = normalize_burstiness(text, _prob=1.0)
        assert len(result.split(".")) > len(text.split("."))

    def test_short_sentences_rejoined(self):
        text = "One. Two. Three. Four. Five. Six. Seven."
        result = normalize_burstiness(text, _prob=1.0)
        assert len(result.split(".")) <= len(text.split("."))

    def test_empty_string(self):
        assert normalize_burstiness("", _prob=1.0) == ""


# ---------------------------------------------------------------------------
# apply_restructure (full pipeline)
# ---------------------------------------------------------------------------

class TestApplyRestructure:
    def test_basic_restructure(self):
        text = ("The experiment yielded significant results because the methodology "
                "was sound. Although the sample size was limited, the conclusions "
                "were validated by subsequent studies.")
        result = apply_restructure(text, probability=1.0)
        assert result != text

    def test_probability_zero(self):
        text = ("The experiment yielded significant results because the methodology "
                "was sound. Although the sample size was limited, the conclusions "
                "were validated by subsequent studies.")
        result = apply_restructure(text, probability=0.0)
        assert result == text

    def test_empty_string(self):
        assert apply_restructure("", probability=1.0) == ""

    def test_determinism(self):
        text = ("The experiment yielded significant results because the methodology "
                "was sound. Although the sample size was limited, the conclusions "
                "were validated by subsequent studies.")
        r1 = apply_restructure(text, probability=1.0)
        r2 = apply_restructure(text, probability=1.0)
        assert r1 == r2

    def test_short_text_unchanged(self):
        text = "Hello world. This is fine."
        result = apply_restructure(text, probability=1.0)
        assert result == text

    def test_no_spacy_fallback(self):
        """Simulate spaCy unavailable by monkeypatching."""
        import voiceprint.restructure as rmod
        old_nlp = rmod._nlp
        rmod._nlp = False
        try:
            text = ("The experiment yielded significant results because the methodology "
                    "was sound.")
            result = apply_restructure(text, probability=1.0)
            assert result == text
        finally:
            rmod._nlp = old_nlp
