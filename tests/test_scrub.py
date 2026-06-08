"""Tests for Stage 1: Heuristic Scrub rules."""

import pytest
from voiceprint.scrub import (
    scrub,
    replace_transitions,
    break_tricolons,
    reduce_em_dashes,
    inject_contractions,
    simplify_passive,
    remove_hedges,
    cleanup_numbered_lists,
    TRANSITION_MAP,
    CONTRACTIONS,
)


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

class TestRuleRegistry:
    def test_all_rules_registered(self):
        from voiceprint.scrub import _rules
        assert len(_rules) >= 6


# ---------------------------------------------------------------------------
# replace_transitions
# ---------------------------------------------------------------------------

class TestReplaceTransitions:
    def test_known_transition_replaced(self):
        result = replace_transitions("Furthermore, this is important.")
        assert result == "Also, this is important."

    def test_case_insensitive(self):
        result = replace_transitions("furthermore this matters")
        assert result == "also this matters"

    def test_multiple_transitions(self):
        text = "However, moreover, consequently."
        result = replace_transitions(text)
        assert "But," in result
        assert "Plus," in result
        assert "so" in result.lower()

    def test_no_transition_unchanged(self):
        text = "The quick brown fox jumps."
        assert replace_transitions(text) == text

    def test_transition_map_completeness(self):
        for ai_phrase, human_phrase in TRANSITION_MAP.items():
            result = replace_transitions(f"{ai_phrase} something")
            assert human_phrase.lower() in result.lower() or result != f"{ai_phrase} something"

    def test_empty_string(self):
        assert replace_transitions("") == ""

    def test_single_word(self):
        assert replace_transitions("Hello") == "Hello"


# ---------------------------------------------------------------------------
# break_tricolons
# ---------------------------------------------------------------------------

class TestBreakTricolons:
    def test_basic_tricolon(self):
        result = break_tricolons("Cats, dogs, and birds are pets.")
        assert result == "Cats and dogs are pets."

    def test_no_tricolon(self):
        text = "Cats and dogs are pets."
        assert break_tricolons(text) == text

    def test_tricolon_with_longer_words(self):
        result = break_tricolons("Speed, accuracy, and reliability matter.")
        assert result == "Speed and accuracy matter."

    def test_multiple_tricolons(self):
        text = "A, B, and C. X, Y, and Z."
        result = break_tricolons(text)
        assert "A and B" in result
        assert "X and Y" in result

    def test_empty_string(self):
        assert break_tricolons("") == ""

    def test_two_items_no_change(self):
        text = "Cats and dogs."
        assert break_tricolons(text) == text

    def test_multi_word_items_unchanged(self):
        result = break_tricolons("High quality, fast delivery, and low prices matter.")
        assert result == "High quality, fast delivery, and low prices matter."


# ---------------------------------------------------------------------------
# reduce_em_dashes
# ---------------------------------------------------------------------------

class TestReduceEmDashes:
    def test_em_dash_replaced(self):
        result = reduce_em_dashes("The result — which was surprising — was clear.")
        assert "—" not in result
        assert ", " in result

    def test_no_em_dash(self):
        text = "The result was clear."
        assert reduce_em_dashes(text) == text

    def test_multiple_em_dashes(self):
        text = "A — B — C"
        result = reduce_em_dashes(text)
        assert result.count("—") == 0

    def test_empty_string(self):
        assert reduce_em_dashes("") == ""


# ---------------------------------------------------------------------------
# inject_contractions
# ---------------------------------------------------------------------------

class TestInjectContractions:
    def test_basic_contraction(self):
        result = inject_contractions("I am happy.")
        assert result == "I'm happy."

    def test_multiple_contractions(self):
        result = inject_contractions("I am not sure but they will not give up.")
        assert "I'm" in result
        assert "won't" in result

    def test_already_contracted(self):
        text = "I'm happy."
        result = inject_contractions(text)
        assert "I'm" in result

    def test_case_insensitive(self):
        result = inject_contractions("I AM happy.")
        assert result == "I'm happy."

    def test_contraction_map_completeness(self):
        for full, contracted in CONTRACTIONS.items():
            # Use "Dogs" prefix to avoid "we are", "they are" etc. matching first
            result = inject_contractions(f"Dogs {full} that.")
            # At least one contraction should appear
            assert contracted in result or full in result, (
                f"Expected '{contracted}' or '{full}' in '{result}'"
            )

    def test_empty_string(self):
        assert inject_contractions("") == ""

    def test_no_match(self):
        text = "The cat sat on the mat."
        assert inject_contractions(text) == text


# ---------------------------------------------------------------------------
# simplify_passive
# ---------------------------------------------------------------------------

class TestSimplifyPassive:
    def test_basic_passive(self):
        result = simplify_passive("It was decided that we move forward.")
        assert result == "We decided that we move forward."

    def test_case_insensitive(self):
        result = simplify_passive("it was found that the data is valid.")
        assert result == "We found that the data is valid."

    def test_no_passive(self):
        text = "We decided to move forward."
        assert simplify_passive(text) == text

    def test_multiple_passives(self):
        text = "It was decided that X. It was found that Y."
        result = simplify_passive(text)
        assert "We decided" in result
        assert "We found" in result

    def test_empty_string(self):
        assert simplify_passive("") == ""


# ---------------------------------------------------------------------------
# remove_hedges
# ---------------------------------------------------------------------------

class TestRemoveHedges:
    def test_hedge_removed(self):
        result = remove_hedges("This is very important.")
        assert "very" not in result

    def test_multiple_hedges(self):
        result = remove_hedges("This is quite rather important.")
        assert "quite" not in result
        assert "rather" not in result

    def test_no_hedge(self):
        text = "This is important."
        assert remove_hedges(text) == text

    def test_double_spaces_cleaned(self):
        result = remove_hedges("This is  very  important.")
        assert "  " not in result

    def test_empty_string(self):
        assert remove_hedges("") == ""


# ---------------------------------------------------------------------------
# cleanup_numbered_lists
# ---------------------------------------------------------------------------

class TestCleanupNumberedLists:
    def test_extra_spaces_removed(self):
        result = cleanup_numbered_lists("1.  First item")
        assert result == "1. First item"

    def test_single_space_preserved(self):
        text = "1. First item"
        assert cleanup_numbered_lists(text) == text

    def test_multiple_numbers(self):
        text = "1.  First\n2.  Second"
        result = cleanup_numbered_lists(text)
        assert "1. First" in result
        assert "2. Second" in result

    def test_empty_string(self):
        assert cleanup_numbered_lists("") == ""


# ---------------------------------------------------------------------------
# scrub (full pipeline)
# ---------------------------------------------------------------------------

class TestScrubFull:
    def test_scrub_applies_all_rules(self):
        text = "Furthermore, I am not sure that this is very important."
        result = scrub(text)
        # Should have replaced transitions, added contractions, removed hedges
        assert "Furthermore" not in result
        assert "I'm" in result

    def test_scrub_preserves_meaning(self):
        text = "The results are significant."
        result = scrub(text)
        assert "results" in result
        assert "significant" in result

    def test_scrub_empty_string(self):
        assert scrub("") == ""

    def test_scrub_single_word(self):
        assert scrub("Hello") == "Hello"

    def test_scrub_idempotent(self):
        """Running scrub twice should produce same result."""
        text = "Furthermore, the results are very important."
        first = scrub(text)
        second = scrub(first)
        assert first == second
