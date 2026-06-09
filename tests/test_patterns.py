"""Tests for AI-pattern fingerprint signals."""

import pytest
from voiceprint.patterns import (
    PYSTYLOMETRY_AVAILABLE,
    signal_ai_vocabulary,
    signal_transition_density,
    signal_sentence_start_uniformity,
    signal_tricolons,
    signal_em_dash_density,
    signal_hedging,
    signal_contraction_deficit,
    signal_ngram_repetition,
    signal_type_token_ratio,
    signal_passive_voice,
    signal_abstract_subjects,
    signal_modality_overload,
    compute_all_signals,
    pattern_score,
)


# ---------------------------------------------------------------------------
# signal_ai_vocabulary
# ---------------------------------------------------------------------------

class TestSignalAIVocabulary:
    def test_clean_text_zero(self):
        assert signal_ai_vocabulary("The cat sat on the mat.") == 0.0

    def test_ai_text_high(self):
        text = "We need to leverage our robust holistic ecosystem."
        score = signal_ai_vocabulary(text)
        assert score > 0.2

    def test_empty_text(self):
        assert signal_ai_vocabulary("") == 0.0

    def test_punctuation_stripped(self):
        text = "delve, leverage, utilize."
        score = signal_ai_vocabulary(text)
        assert score > 0.5


# ---------------------------------------------------------------------------
# signal_transition_density
# ---------------------------------------------------------------------------

class TestSignalTransitionDensity:
    def test_all_transitions(self):
        text = "Furthermore, we move. Moreover, we act. Consequently, we win."
        score = signal_transition_density(text)
        assert score > 0.8

    def test_no_transitions(self):
        text = "We move. We act. We win."
        score = signal_transition_density(text)
        assert score == 0.0

    def test_empty_text(self):
        assert signal_transition_density("") == 0.0


# ---------------------------------------------------------------------------
# signal_sentence_start_uniformity
# ---------------------------------------------------------------------------

class TestSignalSentenceStartUniformity:
    def test_uniform_starters(self):
        text = "The cat sat. The dog ran. The bird flew."
        score = signal_sentence_start_uniformity(text)
        assert score > 0.5

    def test_varied_starters(self):
        text = "Cats are fun. Dogs are great. Birds fly high."
        score = signal_sentence_start_uniformity(text)
        assert score < 0.5

    def test_too_few_sentences(self):
        text = "Hello. World."
        assert signal_sentence_start_uniformity(text) == 0.0


# ---------------------------------------------------------------------------
# signal_tricolons
# ---------------------------------------------------------------------------

class TestSignalTricolons:
    def test_tricolon_present(self):
        text = "Cats, dogs, and birds are pets. " * 3
        score = signal_tricolons(text)
        assert score > 0

    def test_no_tricolon(self):
        text = "Cats and dogs are pets. Birds fly. Fish swim."
        score = signal_tricolons(text)
        assert score == 0.0

    def test_empty_text(self):
        assert signal_tricolons("") == 0.0


# ---------------------------------------------------------------------------
# signal_em_dash_density
# ---------------------------------------------------------------------------

class TestSignalEmDashDensity:
    def test_em_dashes_present(self):
        text = "The result — which was surprising — was clear. " * 3
        score = signal_em_dash_density(text)
        assert score > 0

    def test_no_em_dashes(self):
        text = "The result was clear."
        score = signal_em_dash_density(text)
        assert score == 0.0

    def test_en_dash(self):
        text = "A – B – C – D – E " * 3
        score = signal_em_dash_density(text)
        assert score > 0

    def test_empty_text(self):
        assert signal_em_dash_density("") == 0.0


# ---------------------------------------------------------------------------
# signal_hedging
# ---------------------------------------------------------------------------

class TestSignalHedging:
    def test_hedging_present(self):
        text = "It is important to note that we should proceed. It goes without saying that we must act."
        score = signal_hedging(text)
        assert score > 0

    def test_no_hedging(self):
        text = "We should proceed. We must act."
        score = signal_hedging(text)
        assert score == 0.0

    def test_empty_text(self):
        assert signal_hedging("") == 0.0


# ---------------------------------------------------------------------------
# signal_contraction_deficit
# ---------------------------------------------------------------------------

class TestSignalContractionDeficit:
    def test_no_contractions_high_deficit(self):
        text = "I am happy. We will go. They are here."
        score = signal_contraction_deficit(text)
        assert score > 0.8

    def test_with_contractions_low_deficit(self):
        text = "I'm happy. We'll go. They're here."
        score = signal_contraction_deficit(text)
        assert score < 0.5

    def test_empty_text(self):
        assert signal_contraction_deficit("") == 1.0


# ---------------------------------------------------------------------------
# signal_ngram_repetition
# ---------------------------------------------------------------------------

class TestSignalNgramRepetition:
    def test_repetitive_text(self):
        text = "the cat the cat the cat the cat the cat"
        score = signal_ngram_repetition(text)
        assert score > 0.3

    def test_diverse_text(self):
        text = "The quick brown fox jumps over the lazy dog near the river bank"
        score = signal_ngram_repetition(text)
        assert score < 0.2

    def test_short_text(self):
        assert signal_ngram_repetition("Hi there") == 0.0


# ---------------------------------------------------------------------------
# signal_type_token_ratio
# ---------------------------------------------------------------------------

class TestTypeTokenRatio:
    def test_low_diversity(self):
        text = "the the the the the the the the"
        score = signal_type_token_ratio(text)
        assert score < 0.2

    def test_high_diversity(self):
        text = "one two three four five six seven eight"
        score = signal_type_token_ratio(text)
        assert score > 0.8

    def test_empty_text(self):
        assert signal_type_token_ratio("") == 0.0


# ---------------------------------------------------------------------------
# signal_passive_voice
# ---------------------------------------------------------------------------

class TestSignalPassiveVoice:
    def test_passive_present(self):
        text = "The cake was eaten. The door was opened. The song was sung."
        score = signal_passive_voice(text)
        assert score > 0.5

    def test_active_voice(self):
        text = "We ate the cake. They opened the door. She sang the song."
        score = signal_passive_voice(text)
        assert score == 0.0

    def test_empty_text(self):
        assert signal_passive_voice("") == 0.0


# ---------------------------------------------------------------------------
# signal_abstract_subjects
# ---------------------------------------------------------------------------

class TestSignalAbstractSubjects:
    def test_abstract_start(self):
        text = "It is important. There are many. This is true."
        score = signal_abstract_subjects(text)
        assert score > 0.5

    def test_concrete_start(self):
        text = "Cats are fun. Dogs are loyal. Birds fly."
        score = signal_abstract_subjects(text)
        assert score == 0.0

    def test_empty_text(self):
        assert signal_abstract_subjects("") == 0.0


# ---------------------------------------------------------------------------
# signal_modality_overload
# ---------------------------------------------------------------------------

class TestSignalModalityOverload:
    def test_many_modals(self):
        text = "We could go. We should stay. We might try. We would win."
        score = signal_modality_overload(text)
        assert score > 0.1

    def test_no_modals(self):
        text = "We went. We stayed. We tried. We won."
        score = signal_modality_overload(text)
        assert score == 0.0

    def test_empty_text(self):
        assert signal_modality_overload("") == 0.0


# ---------------------------------------------------------------------------
# compute_all_signals / pattern_score
# ---------------------------------------------------------------------------

class TestComputeAllSignals:
    BASE_KEYS = {
        "ai_vocabulary", "transition_density", "sentence_start_uniformity",
        "tricolons", "em_dash_density", "hedging", "contraction_deficit",
        "ngram_repetition", "type_token_ratio", "passive_voice",
        "abstract_subjects", "modality_overload",
    }
    PYSTYLOMETRY_KEYS = {"pystylometry_mtld", "pystylometry_yule_k", "pystylometry_hapax"}

    def test_returns_all_base_signals(self):
        signals = compute_all_signals("Hello world.")
        assert len(signals) >= 12
        assert self.BASE_KEYS.issubset(set(signals.keys()))

    def test_all_values_float(self):
        signals = compute_all_signals("Hello world.")
        for v in signals.values():
            assert isinstance(v, float)

    def test_all_values_in_range(self):
        signals = compute_all_signals("Some normal text with enough words to test all signals properly throughout the analysis pipeline.")
        for v in signals.values():
            assert 0.0 <= v <= 1.0

    @pytest.mark.skipif(not PYSTYLOMETRY_AVAILABLE, reason="pystylometry not installed")
    def test_pystylometry_signals_present_for_long_text(self):
        signals = compute_all_signals("This is a longer piece of text with enough words to trigger the pystylometry lexical analysis and verify the new signals are returned correctly for AI detection purposes.")
        for key in self.PYSTYLOMETRY_KEYS:
            assert key in signals, f"Missing pystylometry signal: {key}"
        assert len(signals) >= 15

    @pytest.mark.skipif(not PYSTYLOMETRY_AVAILABLE, reason="pystylometry not installed")
    def test_pystylometry_short_text_no_signals(self):
        signals = compute_all_signals("Too short.")
        for key in self.PYSTYLOMETRY_KEYS:
            assert key not in signals


class TestPatternScore:
    def test_returns_float(self):
        score = pattern_score("Hello world.")
        assert isinstance(score, float)

    def test_in_range(self):
        score = pattern_score("Some normal text.")
        assert 0.0 <= score <= 1.0

    def test_ai_text_higher_score(self):
        ai_text = "Furthermore, we need to leverage our robust holistic ecosystem. Moreover, this is innovative and transformative."
        normal_text = "The cat sat on the mat. It was a nice day."
        ai_score = pattern_score(ai_text)
        normal_score = pattern_score(normal_text)
        assert ai_score > normal_score

    def test_type_token_ratio_not_inverted(self):
        text = "one two three four five six seven eight nine ten eleven twelve"
        score = pattern_score(text)
        assert score < 0.5
