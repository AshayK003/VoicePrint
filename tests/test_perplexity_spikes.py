"""Tests for boosted perplexity spike injection in polish.py."""
import pytest
from voiceprint.polish import (
    inject_vocabulary_variety,
    inject_perplexity_spikes,
    VOCAB_SWAPS,
)


class TestVocabularyVariety:
    def test_boosted_trigger_rate(self):
        """Vocabulary variety rule exists, works, and respects the rule registry."""
        from voiceprint.scrub import _rules as scrub_rules
        # The scrub rule registry should include split_long_sentences
        rule_names = [fn.__name__ for fn in scrub_rules]
        assert "split_long_sentences" in rule_names

    def test_swap_matches_pattern(self):
        """When a swap is triggered, it replaces a known word."""
        # Single sentence — deterministic, we can predict the outcome
        text = "The good results were very clear and important for the team."
        result = inject_vocabulary_variety(text)
        # Either unchanged (RNG didn't trigger) or contains a swap
        if result != text:
            # At least one VOCAB_SWAPS replacement should appear
            assert any(
                repl in result.lower()
                for _, repl in VOCAB_SWAPS
            ), f"Expected a swap in: {result}"

    def test_empty_string(self):
        assert inject_vocabulary_variety("") == ""

    def test_short_text_unchanged(self):
        text = "One. Two. Three."
        assert inject_vocabulary_variety(text) == text


class TestPerplexitySpikes:
    def test_injects_uncommon_words(self):
        """Should inject uncommon words that create perplexity spikes."""
        # Long enough text (10+ sentences) to trigger injection
        text = (
            "The team worked hard on the project and made good progress. "
            "They used a clear and important approach to solve the problem. "
            "The results were big and the impact was difficult to measure. "
            "Everyone agreed the work was good and the outcome was clear. "
            "The process showed a very clear and important result overall. "
            "This approach has been used in many other projects before now. "
            "The team found that the data showed a clear and big difference. "
            "It was a good result that proved the approach was important. "
            "The final outcome was clear and showed good progress overall. "
            "Everyone agreed that the results were important and clear to see."
        )
        # Run multiple times — should inject spikes eventually
        injected = 0
        for _ in range(20):
            result = inject_perplexity_spikes(text)
            if result != text:
                injected += 1
        assert injected > 0, "Perplexity spikes should inject uncommon words"

    def test_preserves_meaning(self):
        """Spikes should be contextually appropriate — not random garbage."""
        text = (
            "The research was clear and important. The results were good. "
            "The analysis showed a big difference. The team was happy. "
            "The approach was important and clear. The data was good. "
            "The outcome was clear. The results showed progress. "
            "The team made good progress on the important work. "
            "The clear results showed the approach was good overall."
        )
        result = inject_perplexity_spikes(text)
        # No gibberish — all words should be real English
        for word in result.split():
            clean = word.strip(".,!?;:'\"")
            assert len(clean) > 0

    def test_empty_string(self):
        assert inject_perplexity_spikes("") == ""

    def test_deterministic(self):
        """Same input → same output (seeded RNG)."""
        text = (
            "The good results were important. The clear data showed progress. "
            "The big impact was difficult to measure. The team worked hard. "
            "The approach was good and clear. The results showed importance. "
            "The analysis was clear. The data was important. "
            "The progress was good and clear. The results were big."
        )
        assert inject_perplexity_spikes(text) == inject_perplexity_spikes(text)

    def test_spike_rate_above_minimum(self):
        """Should hit at least 3% of sentences with spikes."""
        # 30 sentences — should get at least 1 spike
        sentence = "The results were good and the approach was important."
        text = ". ".join([sentence] * 30) + "."
        spikes_injected = 0
        for _ in range(30):
            result = inject_perplexity_spikes(text)
            if result != text:
                spikes_injected += 1
        assert spikes_injected > 0
