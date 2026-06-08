"""Tests for app.py pure functions (no Streamlit runtime needed).

Note: detect_provider_from_key has been moved to voiceprint.config.
Tests for it are in test_config.py. This file tests app.py helpers only.
"""

import pytest
from app import _word_diff_html, _copy_button_html, _status_badge, _metric_help


# ---------------------------------------------------------------------------
# _word_diff_html
# ---------------------------------------------------------------------------

class TestWordDiffHtml:
    def test_identical_text_no_highlighting(self):
        html = _word_diff_html("hello world", "hello world")
        assert "<span" not in html
        assert "hello" in html
        assert "world" in html

    def test_replaced_words_highlighted(self):
        html = _word_diff_html("hello world", "hello earth")
        assert "vp-diff-del" in html
        assert "vp-diff-ins" in html
        assert "world" in html
        assert "earth" in html

    def test_inserted_words(self):
        html = _word_diff_html("hello", "hello beautiful world")
        assert "vp-diff-ins" in html
        assert "beautiful" in html

    def test_deleted_words(self):
        html = _word_diff_html("hello beautiful world", "hello")
        assert "vp-diff-del" in html
        assert "beautiful" in html

    def test_html_escapes_special_chars(self):
        html = _word_diff_html("<script>alert('xss')</script>", "safe text")
        assert "<script>" not in html  # Should be escaped
        assert "&lt;script&gt;" in html


# ---------------------------------------------------------------------------
# _copy_button_html
# ---------------------------------------------------------------------------

class TestCopyButtonHtml:
    def test_returns_button_html(self):
        js = _copy_button_html("Hello world")
        assert "<button" in js
        assert "this.dataset.text" in js
        assert "Copy" in js

    def test_escapes_double_quotes(self):
        js = _copy_button_html('Quote "test"')
        assert "data-text=\"Quote &quot;test&quot;\"" in js

    def test_escapes_html_chars(self):
        js = _copy_button_html("<script>alert('xss')</script>")
        assert "&lt;script&gt;" in js
        assert "<script>" not in js.rsplit("data-text=", 1)[1].split(">", 1)[0]

    def test_preserves_newlines(self):
        js = _copy_button_html("Line1\nLine2")
        assert "data-text=" in js
        assert "navigator.clipboard" in js

    def test_empty_text(self):
        js = _copy_button_html("")
        assert "<button" in js
        assert "data-text=\"\"" in js

    def test_has_aria_label(self):
        js = _copy_button_html("test")
        assert 'aria-label="Copy text to clipboard"' in js

    def test_no_inner_html(self):
        js = _copy_button_html("test")
        assert "innerHTML" not in js


# ---------------------------------------------------------------------------
# _status_badge
# ---------------------------------------------------------------------------

class TestStatusBadge:
    def test_success_badge(self):
        html = _status_badge("OK", "success")
        assert "vp-badge-success" in html
        assert "OK" in html

    def test_warning_badge(self):
        html = _status_badge("Low", "warning")
        assert "vp-badge-warning" in html

    def test_error_badge(self):
        html = _status_badge("Failed", "error")
        assert "vp-badge-error" in html


# ---------------------------------------------------------------------------
# _metric_help
# ---------------------------------------------------------------------------

class TestMetricHelp:
    def test_known_metric(self):
        html = _metric_help("AI Probability")
        assert "Below 50%" in html

    def test_unknown_metric(self):
        html = _metric_help("Unknown")
        assert '<div class="vp-metric-context">' in html
        assert "</div>" in html
