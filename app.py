"""VoicePrint — Streamlit Frontend.

Run: streamlit run app.py
"""

import json

import streamlit as st

from voiceprint.config import (
    PROVIDER_PRESETS,
    PROVIDER_MODELS,
    PROVIDER_BASE_URLS,
    detect_provider_from_key,
)
from voiceprint.service import build_config, humanize, detect, InputError
from voiceprint.paraphrase import test_llm_connection


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VoicePrint — AI Text Humanizer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — cached to avoid re-injection on every rerun
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_css() -> str:
    return """
<style>
:root {
    --vp-spacing-xs: 4px; --vp-spacing-sm: 8px; --vp-spacing-md: 16px;
    --vp-spacing-lg: 24px; --vp-spacing-xl: 32px;
    --vp-radius: 8px; --vp-radius-sm: 4px;
    --vp-color-text: #1a1a2e; --vp-color-text-muted: #6b7280;
    --vp-color-surface: #f8f9fa; --vp-color-border: #e5e7eb;
    --vp-color-success: #059669; --vp-color-warning: #d97706;
    --vp-color-error: #dc2626; --vp-color-info: #2563eb;
}
:focus-visible { outline: 2px solid var(--vp-color-info); outline-offset: 2px; }
.vp-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 500; line-height: 1; }
.vp-badge-success { background: #d1fae5; color: #065f46; }
.vp-badge-warning { background: #fef3c7; color: #92400e; }
.vp-badge-error { background: #fee2e2; color: #991b1b; }
.vp-empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 48px 32px; text-align: center; border: 2px dashed var(--vp-color-border); border-radius: var(--vp-radius); color: var(--vp-color-text-muted); min-height: 300px; }
.vp-empty-state h3 { margin: 0 0 8px 0; color: var(--vp-color-text); font-size: 18px; }
.vp-empty-state p { margin: 0 0 4px 0; font-size: 14px; max-width: 360px; }
.vp-stage-list { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.vp-stage { display: inline-flex; align-items: center; gap: 4px; padding: 4px 12px; border-radius: 16px; font-size: 12px; font-weight: 500; background: var(--vp-color-surface); border: 1px solid var(--vp-color-border); color: var(--vp-color-text); }
.vp-metric-context { font-size: 11px; color: var(--vp-color-text-muted); margin-top: 2px; }
.vp-copy-btn { display: inline-flex; align-items: center; gap: 6px; padding: 6px 14px; border: 1px solid var(--vp-color-border); border-radius: var(--vp-radius-sm); background: white; color: var(--vp-color-text); font-size: 13px; font-weight: 500; cursor: pointer; transition: background 150ms, border-color 150ms; float: right; }
.vp-copy-btn:hover { background: var(--vp-color-surface); border-color: #d1d5db; }
.vp-copy-btn:focus-visible { outline: 2px solid var(--vp-color-info); outline-offset: 2px; }
.vp-copy-btn:active { background: #e5e7eb; }
.vp-diff-container { display: flex; gap: 16px; max-height: 400px; }
.vp-diff-pane { flex: 1; overflow: auto; padding: 12px; border: 1px solid var(--vp-color-border); border-radius: var(--vp-radius); font-size: 14px; line-height: 1.7; }
.vp-diff-pane::-webkit-scrollbar { width: 6px; }
.vp-diff-pane::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 3px; }
.vp-diff-label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: var(--vp-color-text-muted); margin-bottom: 8px; }
.vp-diff-del { background: #fecaca; text-decoration: line-through; border-radius: 2px; padding: 0 1px; }
.vp-diff-ins { background: #bbf7d0; border-radius: 2px; padding: 0 1px; }
.vp-kbd { display: inline-block; padding: 2px 6px; border: 1px solid #d1d5db; border-radius: 4px; background: #f9fafb; font-family: monospace; font-size: 11px; color: var(--vp-color-text-muted); line-height: 1.4; }
@media (max-width: 768px) {
    .stColumns > div { min-width: 100% !important; flex-basis: 100% !important; }
    .vp-diff-container { flex-direction: column; }
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 12px; }
</style>
"""

st.markdown(_load_css(), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — Provider & API Configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### VoicePrint")
    st.caption("AI Text Humanizer v0.1.0")

    st.divider()

    # --- Provider section ---
    st.markdown("**LLM Provider**")

    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="Paste your API key",
        help="Provider, model, and base URL are auto-detected from the key prefix.",
        label_visibility="visible",
    )

    auto = detect_provider_from_key(api_key) if api_key else None

    if "provider" not in st.session_state:
        st.session_state.provider = "OpenCode Zen"
    if "base_url" not in st.session_state:
        st.session_state.base_url = ""
    if "model" not in st.session_state:
        st.session_state.model = PROVIDER_PRESETS[st.session_state.provider]["model"]

    if auto:
        if st.session_state.provider != auto["provider"]:
            st.toast(f"Auto-detected: {auto['provider']}", icon="🔍")
        st.session_state.provider = auto["provider"]
        st.session_state.model = auto["model"]
        st.session_state.base_url = auto["base_url"]

    provider = st.selectbox(
        "Provider",
        options=list(PROVIDER_PRESETS.keys()),
        index=list(PROVIDER_PRESETS.keys()).index(st.session_state.provider),
        key="provider_select",
        label_visibility="visible",
    )
    st.session_state.provider = provider

    preset = PROVIDER_PRESETS[provider]

    model_options = PROVIDER_MODELS.get(provider, [])
    model_options_with_custom = list(model_options) + ["Custom..."]
    current_model = st.session_state.model

    try:
        model_idx = model_options.index(current_model)
    except ValueError:
        model_idx = len(model_options_with_custom) - 1

    chosen_model = st.selectbox(
        "Model",
        options=model_options_with_custom,
        index=model_idx,
        key="model_select",
        label_visibility="visible",
    )

    if chosen_model == "Custom...":
        model = st.text_input(
            "Custom model name",
            value="" if current_model in model_options else current_model,
            placeholder="e.g. gpt-4-turbo",
            key="model_custom",
            label_visibility="visible",
        )
    else:
        model = chosen_model
    st.session_state.model = model

    base_url_options = PROVIDER_BASE_URLS.get(provider, [])
    base_url_options_with_custom = base_url_options + ["Custom..."]
    current_base_url = st.session_state.base_url

    try:
        bu_idx = base_url_options.index(current_base_url)
    except ValueError:
        bu_idx = len(base_url_options_with_custom) - 1

    chosen_base_url = st.selectbox(
        "Base URL",
        options=base_url_options_with_custom,
        index=bu_idx if current_base_url else 0,
        key="base_url_select",
        label_visibility="visible",
    )

    if chosen_base_url == "Custom...":
        base_url = st.text_input(
            "Custom base URL",
            value="" if current_base_url in base_url_options else current_base_url,
            placeholder="https://your-api.com/v1",
            key="base_url_custom",
            label_visibility="visible",
        )
    elif chosen_base_url == "(default)":
        base_url = ""
    else:
        base_url = chosen_base_url
    st.session_state.base_url = base_url

    # Provider status — cached env key check
    @st.cache_data
    def _check_env_key(env_key_name: str) -> bool:
        import os
        if os.getenv(env_key_name):
            return True
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                return bool(winreg.QueryValueEx(key, env_key_name)[0])
        except Exception:
            return False

    env_key_name = preset.get("env_key", "")
    has_env_key = _check_env_key(env_key_name) if env_key_name else False
    has_manual_key = bool(api_key)

    if has_manual_key:
        if auto:
            st.markdown(
                f'<div class="vp-badge vp-badge-success">'
                f'🔍 Detected: {auto["provider"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="vp-badge vp-badge-warning">'
                f'Using: {provider}</div>',
                unsafe_allow_html=True,
            )
    elif has_env_key:
        st.markdown(
            f'<div class="vp-badge vp-badge-success">'
            f'Using env: {env_key_name}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="vp-badge vp-badge-error">'
            'No API key set</div>',
            unsafe_allow_html=True,
        )

    # --- Test Connection button ---
    has_any_key = has_manual_key or has_env_key or bool(api_key)
    if has_any_key:
        test_col1, test_col2 = st.columns([2, 1])
        with test_col1:
            test_clicked = st.button("🔌 Test Connection", use_container_width=True, key="test_btn")
        with test_col2:
            if "conn_status" in st.session_state:
                cs = st.session_state.conn_status
                if cs["connected"]:
                    st.markdown(
                        '<div class="vp-badge vp-badge-success">✓</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="vp-badge vp-badge-error">✗</div>',
                        unsafe_allow_html=True,
                    )

        if test_clicked:
            with st.spinner("Connecting..."):
                cfg = build_config(
                    provider=st.session_state.provider,
                    api_key=api_key or "",
                    base_url=st.session_state.base_url,
                    model=st.session_state.model,
                )
                cs = test_llm_connection(cfg)
                st.session_state.conn_status = cs
                if cs["connected"]:
                    st.toast("Connected successfully!", icon="✅")
                else:
                    st.toast(f"Connection failed", icon="❌")
                    if cs.get("error"):
                        st.caption(f"Error: {cs['error']}")

    # --- Best free model indicator ---
    _best_free_models = {
        "Google Gemini (Free)": "gemini-2.0-flash (stable)\ngemini-1.5-flash (fallback)",
        "OpenCode Zen": "deepseek-v3-0615-free (best overall)\nnemotron-3-ultra-free (fast)",
        "Groq (Free)": "llama-3.3-70b (fastest)\nllama-4-scout (latest)",
        "Mistral (Free)": "mistral-tiny (fast)\nmistral-small-latest (quality)",
        "OpenAI": "gpt-4o-mini (cheapest)",
        "Anthropic": "claude-3-haiku (cheapest)",
        "Custom (OpenAI-compatible)": "",
    }
    tip = _best_free_models.get(provider, "")
    if tip:
        st.markdown(
            f'<div style="font-size:11px;color:#6b7280;margin-top:4px;'
            f'padding:6px 10px;background:#f3f4f6;border-radius:6px;line-height:1.5;">'
            f'⭐ Best free models:<br>{tip}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # --- Pipeline settings section ---
    st.markdown("**Pipeline Stages**")

    use_scrub = st.checkbox("Heuristic scrub", value=True, key="use_scrub")
    use_paraphrase = st.checkbox("LLM paraphrasing", value=True, key="use_paraphrase")
    use_polish = st.checkbox("Style polish", value=True, key="use_polish")
    n_candidates = st.slider(
        "Candidates (N)",
        min_value=1,
        max_value=16,
        value=8,
        help="More candidates = better output but slower. 8 is a good default.",
    )

    st.divider()

    # --- About section ---
    with st.expander("About VoicePrint", expanded=False):
        st.markdown("""
Multi-stage pipeline that transforms AI-generated text
into natural, human-like writing.

**Stages:**
1. Heuristic scrub (no model)
2. Adversarial paraphrasing (LLM API)
3. Detection scoring (ensemble)
4. Style polish (no model)
        """)

    st.divider()
    st.markdown(
        '<div style="text-align:center;padding:4px 0">'
        '<a href="https://chai4.me/darkcharon3301" target="_blank" '
        'title="Support darkcharon3301 on Chai4Me" '
        'style="display:inline-flex;flex-direction:column;align-items:center;justify-content:center;'
        'background:#ffffff;padding:8px 32px;border-radius:16px;text-decoration:none;'
        'border:1px solid #e5e7eb;box-shadow:0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.05);">'
        '<img src="https://chai4.me/icons/wordmark.png" alt="Chai4Me" style="height:32px;object-fit:contain;"/>'
        '</a></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Helpers: diff viewer, copy button
# ---------------------------------------------------------------------------

def _word_diff_html(original: str, humanized: str) -> str:
    """Return HTML with word-level diff highlighting."""
    import difflib
    from html import escape

    orig_words = original.split()
    human_words = humanized.split()
    matcher = difflib.SequenceMatcher(None, orig_words, human_words)
    out = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            out.extend(escape(w) + " " for w in orig_words[i1:i2])
        elif tag == "replace":
            out.append('<span class="vp-diff-del">')
            out.extend(escape(w) + " " for w in orig_words[i1:i2])
            out.append('</span><span class="vp-diff-ins">')
            out.extend(escape(w) + " " for w in human_words[j1:j2])
            out.append("</span>")
        elif tag == "delete":
            out.append('<span class="vp-diff-del">')
            out.extend(escape(w) + " " for w in orig_words[i1:i2])
            out.append("</span>")
        elif tag == "insert":
            out.append('<span class="vp-diff-ins">')
            out.extend(escape(w) + " " for w in human_words[j1:j2])
            out.append("</span>")
    return "".join(out)


def _copy_button_html(text: str, button_id: str = "copy-btn") -> str:
    """Safe copy button — text in data-* attribute via HTML character references.
    No inline JSON, no innerHTML, no onclick injection surface.
    The browser decodes character references when reading dataset.text.
    """
    # HTML-attribute-safe encoding: character references for all dangerous chars
    safe = text.replace("&", "&amp;")
    safe = safe.replace('"', "&quot;")
    safe = safe.replace("<", "&lt;")
    safe = safe.replace(">", "&gt;")
    return (
        f'<button id="{button_id}" class="vp-copy-btn" '
        f'data-text="{safe}" '
        f'onclick="(async()=>{{'
        f'await navigator.clipboard.writeText(this.dataset.text);'
        f'this.textContent=\'Copied!\';'
        f'setTimeout(()=>{{this.textContent=\'📋 Copy\'}},1500)'
        f'}})()" '
        f'aria-label="Copy text to clipboard">'
        f'📋 Copy</button>'
    )


def _status_badge(text: str, variant: str = "success") -> str:
    """HTML status badge."""
    return (
        f'<span class="vp-badge vp-badge-{variant}">{text}</span>'
    )


def _metric_help(label: str) -> str:
    """Context label for metrics."""
    helps = {
        "AI Probability": "Below 50% = looks human-written",
        "Similarity": "Above 68% = meaning preserved",
        "Burstiness": "0.4–0.7 = natural rhythm",
        "Pattern Score": "Below 0.1 = few AI patterns",
    }
    return f'<div class="vp-metric-context">{helps.get(label, "")}</div>'


# ---------------------------------------------------------------------------
# Stage indicator
# ---------------------------------------------------------------------------

_STAGE_NAMES = ["Scrub", "Paraphrase", "Polish", "Detect", "Metrics"]


def _parse_stage(msg: str, pct: float) -> str | None:
    """Determine active pipeline stage from progress message + percentage."""
    if "Stage 1" in msg or "scrub" in msg.lower():
        return "Scrub"
    if "Stage 2" in msg or "skipped" in msg:
        return "Paraphrase"
    if "Stage 3" in msg:
        return "Polish"
    if "Stage 4" in msg:
        return "Detect"
    if "Stage 5" in msg:
        return "Metrics"
    if pct < 0.15:
        return None
    if pct < 0.70:
        return "Paraphrase"
    if pct < 0.80:
        return "Polish"
    if pct < 0.95:
        return "Detect"
    return "Metrics"


def _stages_html(completed: list[str], active: str | None, error: str | None = None) -> str:
    """Render pipeline stage progression as inline HTML steps."""
    parts = []
    for i, stage in enumerate(_STAGE_NAMES):
        if i > 0:
            parts.append(
                '<span style="color:#d1d5db;margin:0 4px;font-size:12px">\u2192</span>'
            )

        if stage == error:
            bg, fg = "#fee2e2", "#991b1b"
            icon = "\u2717"
        elif stage in completed:
            bg, fg = "#d1fae5", "#065f46"
            icon = "\u2713"
        elif stage == active:
            bg, fg = "#dbeafe", "#1e40af"
            icon = "\u25cf"
        else:
            bg, fg = "#f3f4f6", "#9ca3af"
            icon = "\u25cb"

        border = ';border:1px solid #93c5fd' if stage == active else ''
        parts.append(
            f'<span style="display:inline-flex;align-items:center;gap:3px;'
            f'padding:4px 10px;border-radius:12px;font-size:12px;font-weight:500;'
            f'background:{bg};color:{fg}{border}">'
            f'{icon} {stage}</span>'
        )

    return '<div style="display:flex;align-items:center;flex-wrap:wrap">' + "".join(parts) + "</div>"


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("VoicePrint")
st.caption("Transform AI-generated text into natural, human-like writing.")

col_input, col_output = st.columns(2)

with col_input:
    st.markdown("**Input**")
    input_text = st.text_area(
        "Text to humanize",
        height=300,
        placeholder="Paste your AI-generated text here...",
        label_visibility="visible",
        key="input_text",
    )
    char_count = len(input_text.strip()) if input_text else 0
    if input_text:
        if char_count < 10:
            st.caption(f":red[{char_count} chars — minimum 10 required]")
        else:
            st.caption(f"{char_count:,} characters")


# ---------------------------------------------------------------------------
# Action row — Humanize button + keyboard shortcut
# ---------------------------------------------------------------------------

col_btn, col_shortcut = st.columns([3, 1])
with col_btn:
    humanize_clicked = st.button(
        "Humanize",
        type="primary",
        use_container_width=True,
        key="humanize_btn",
    )
with col_shortcut:
    st.markdown(
        '<div style="padding-top:10px;text-align:center;color:#6b7280;font-size:13px">'
        '<span class="vp-kbd">Ctrl</span> + <span class="vp-kbd">Enter</span>'
        '</div>',
        unsafe_allow_html=True,
    )

# Inject JS for Ctrl+Enter shortcut (guarded: only one listener per session)
st.markdown("""
<script>
if (!window._vpKeyHandler) {
    window._vpKeyHandler = function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            var btn = document.querySelector('[data-testid="stButton"][kind="primary"] button');
            if (btn) btn.click();
        }
    };
    document.addEventListener('keydown', window._vpKeyHandler);
}
</script>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

if humanize_clicked:
    if not input_text or not input_text.strip():
        st.warning("Please enter some text to humanize.")
    elif not api_key and not preset.get("env_key"):
        st.error("Add an API key in the sidebar, or set an environment variable.")
    else:
        config = build_config(
            provider=st.session_state.provider,
            api_key=api_key,
            base_url=st.session_state.base_url,
            model=st.session_state.model,
        )

        with st.status("Initializing pipeline...", expanded=True) as status:
            progress = st.progress(0.0)
            stage_display = st.empty()
            stage_display.markdown(_stages_html([], None), unsafe_allow_html=True)

            _stage_track = {"completed": [], "active": None}  # mutable container for closure

            def _on_progress(pct: float, msg: str):
                progress.progress(pct)

                stage = _parse_stage(msg, pct)
                if stage and stage != _stage_track["active"]:
                    if _stage_track["active"]:
                        _stage_track["completed"].append(_stage_track["active"])
                    _stage_track["active"] = stage

                is_err = "skipped" in msg
                state = "error" if is_err else ("complete" if pct >= 1.0 else "running")
                status.update(label=msg, state=state)

                error_stage = "Paraphrase" if is_err else None
                stage_display.markdown(
                    _stages_html(_stage_track["completed"], _stage_track["active"], error_stage),
                    unsafe_allow_html=True,
                )

            try:
                result = humanize(
                    input_text,
                    config=config,
                    use_scrub=use_scrub,
                    use_paraphrase=use_paraphrase,
                    use_polish=use_polish,
                    n_candidates=n_candidates,
                    progress_callback=_on_progress,
                )
            except InputError as e:
                status.update(label="Invalid input", state="error")
                st.warning(str(e))
                st.stop()
            except Exception as e:
                status.update(label="Pipeline failed", state="error")
                st.error(f"Pipeline failed: {e}")
                st.stop()

            if not result.success:
                status.update(label="Pipeline failed", state="error")
                st.error(f"Pipeline failed: {result.error}")
                st.stop()

            # Cache result in session_state to avoid recomputation on UI interactions
            st.session_state["last_result"] = result
            st.session_state["last_input"] = input_text

# Use cached result if available (avoids recomputation on tab/expander clicks)
result = st.session_state.get("last_result")
cached_input = st.session_state.get("last_input", "")

# ---------------------------------------------------------------------------
# Render results (from fresh run or cache)
# ---------------------------------------------------------------------------

if result:
    with col_output:
        st.markdown("**Output**")

        if result.text:
            st.markdown(
                _copy_button_html(result.text),
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div style="background:#1a1a2e;border:1px solid #333;border-radius:8px;'
            f'padding:16px;min-height:270px;max-height:400px;overflow-y:auto;'
            f'font-size:14px;line-height:1.7;color:#e0e0e0;white-space:pre-wrap;'
            f'font-family:-apple-system,BlinkMacSystemFont,sans-serif;">'
            f'{result.text}</div>',
            unsafe_allow_html=True,
        )

        if result.stages:
            stage_html = "".join(
                f'<span class="vp-stage">{s}</span>' for s in result.stages
            )
            st.markdown(
                f'<div class="vp-stage-list">{stage_html}</div>',
                unsafe_allow_html=True,
            )

    # Paraphrase status badge
    has_llm = "paraphrase" in (result.stages or [])
    if has_llm:
        st.markdown(
            '<div class="vp-badge vp-badge-success">Paraphrase: Applied</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="vp-badge vp-badge-warning">Paraphrase: Skipped (no LLM)</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    method_tag = (
        "Heuristic" if "statistical" in result.detection_summary
        else "Model"
    )
    st.markdown(f"**Detection Scores** {_status_badge(method_tag)}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ai_pct = result.ai_probability
        st.metric(
            "AI Probability",
            f"{ai_pct:.1%}",
            delta="Human" if ai_pct < 0.5 else "AI",
            delta_color="normal" if ai_pct < 0.5 else "inverse",
        )
        st.markdown(_metric_help("AI Probability"), unsafe_allow_html=True)
    with c2:
        sim = result.similarity
        st.metric(
            "Similarity",
            f"{sim:.1%}",
            delta="OK" if sim >= 0.68 else "Low",
            delta_color="normal" if sim >= 0.68 else "inverse",
        )
        st.markdown(_metric_help("Similarity"), unsafe_allow_html=True)
    with c3:
        bur = result.burstiness
        st.metric(
            "Burstiness",
            f"{bur:.2f}",
            delta="Human" if 0.4 <= bur <= 0.7 else "AI-like",
            delta_color="normal" if 0.4 <= bur <= 0.7 else "inverse",
        )
        st.markdown(_metric_help("Burstiness"), unsafe_allow_html=True)
    with c4:
        ps = result.pattern_score
        st.metric(
            "Pattern Score",
            f"{ps:.3f}",
            delta="Low" if ps < 0.1 else "High",
            delta_color="normal" if ps < 0.1 else "inverse",
        )
        st.markdown(_metric_help("Pattern Score"), unsafe_allow_html=True)

    # --- Diff viewer (lazy: only compute HTML when expander is opened) ---
    with st.expander("Side-by-Side Diff", expanded=False):
        diff_html = _word_diff_html(result.original, result.text)
        orig_paragraphs = "".join(
            f"<p>{line}</p>" for line in result.original.strip().split("\n")
        )
        st.markdown(
            '<div class="vp-diff-container">'
            '<div class="vp-diff-pane">'
            '<div class="vp-diff-label">Original</div>'
            + orig_paragraphs
            + "</div>"
            '<div class="vp-diff-pane">'
            '<div class="vp-diff-label">Humanized</div>'
            + diff_html
            + "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    # --- Detail tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Detection", "Readability", "Signals", "Stages"]
    )
    with tab1:
        st.code(result.detection_summary, language=None)
    with tab2:
        cols = st.columns(2)
        with cols[0]:
            st.json(result.readability)
        with cols[1]:
            st.json(result.burstiness_detail)
    with tab3:
        st.json(result.signals)
    with tab4:
        for s in result.stages:
            st.markdown(f"- {s}")

    # --- Download row ---
    st.divider()
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download humanized text",
            data=result.text,
            file_name="humanized_text.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with dl2:
        report = {
            "provider": st.session_state.provider,
            "model": st.session_state.model,
            "ai_probability": result.ai_probability,
            "similarity": result.similarity,
            "burstiness": result.burstiness,
            "pattern_score": result.pattern_score,
            "readability": result.readability,
            "signals": result.signals,
        }
        st.download_button(
            "Download report (JSON)",
            data=json.dumps(report, indent=2),
            file_name="detection_report.json",
            mime="application/json",
            use_container_width=True,
        )

else:
    # --- Empty state ---
    with col_output:
        if input_text and input_text.strip():
            st.info("Click **Humanize** to start the pipeline.")
        else:
            st.markdown("""
<div class="vp-empty-state">
    <h3>Paste AI text, get human writing</h3>
    <p>Enter AI-generated text in the input panel, then click Humanize.</p>
    <p>The pipeline scrubs patterns, paraphrases with an LLM, polishes style, and checks detection scores.</p>
    <div class="vp-stage-list" style="margin-top:16px">
        <span class="vp-stage">1. Scrub</span>
        <span class="vp-stage">2. Paraphrase</span>
        <span class="vp-stage">3. Polish</span>
        <span class="vp-stage">4. Detect</span>
    </div>
</div>
            """, unsafe_allow_html=True)

        if input_text and input_text.strip():
            if st.button("Quick Detection Check", use_container_width=False):
                with st.spinner("Analyzing..."):
                    try:
                        pre_result = detect(input_text)
                    except InputError as e:
                        st.warning(str(e))
                        st.stop()

                st.metric(
                    "Pre-check AI Probability",
                    f"{pre_result['p_ai']:.1%}",
                )
                st.code(pre_result["summary"], language=None)
