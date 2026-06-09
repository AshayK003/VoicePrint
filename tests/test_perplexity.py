"""Tests for GPT-2 based perplexity scoring."""


def _can_score():
    """Check if GPT-2 is available (not stubbed by conftest)."""
    from voiceprint.perplexity import raw_perplexity
    return raw_perplexity("test") is not None


class TestPerplexity:
    def test_raw_perplexity_returns_number(self):
        from voiceprint.perplexity import raw_perplexity
        ppl = raw_perplexity("The quick brown fox jumps over the lazy dog.")
        assert ppl is None or isinstance(ppl, float)
        if ppl is not None:
            assert ppl > 0

    def test_perplexity_score_between_0_and_1(self):
        from voiceprint.perplexity import perplexity_score
        score = perplexity_score("The quick brown fox jumps over the lazy dog.")
        assert score is None or (0.0 <= score <= 1.0)

    def test_ai_text_higher_perplexity(self):
        if not _can_score():
            return
        from voiceprint.perplexity import raw_perplexity
        ai_text = (
            "In todays rapidly evolving digital landscape it is essential to leverage "
            "cutting edge solutions that fundamentally transform the way we approach "
            "innovative paradigms Our synergistic framework optimizes core competencies "
            "across verticals while driving best of breed methodologies"
        )
        human_text = (
            "So I went to the store yesterday and guess what They were out Completely out "
            "The guy just shrugged at me I stood there like an idiot for five minutes before "
            "he even looked up from his phone Then he said check back Tuesday Tuesday I drove "
            "all the way across town for nothing"
        )
        ai_ppl = raw_perplexity(ai_text)
        human_ppl = raw_perplexity(human_text)
        assert ai_ppl is not None and human_ppl is not None
        # GPT-2 finds AI-generated buzzword text more surprising than conversational text
        assert human_ppl < ai_ppl, (
            f"Expected human text perplexity ({human_ppl:.1f}) < AI text ({ai_ppl:.1f})"
        )

    def test_short_text_does_not_crash(self):
        from voiceprint.perplexity import raw_perplexity
        ppl = raw_perplexity("Hello world.")
        assert ppl is None or ppl > 0

    def test_empty_text_returns_0(self):
        from voiceprint.perplexity import raw_perplexity
        assert raw_perplexity("") == 0.0

    def test_lazy_load_no_error(self):
        from voiceprint.perplexity import raw_perplexity
        p1 = raw_perplexity("first call")
        p2 = raw_perplexity("second call")
        if p1 is not None and p2 is not None:
            assert p1 > 0
            assert p2 > 0
