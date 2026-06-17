"""VoicePrint — Streamlit Frontend.

Run: streamlit run app.py
"""

import json
from pathlib import Path

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
# Custom CSS — loaded from external file, cached to avoid re-read on every rerun
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_css() -> str:
    css_path = Path(__file__).resolve().parent / "voiceprint" / "static" / "style.css"
    if css_path.exists():
        return f"<style>\n{css_path.read_text()}\n</style>"
    return ""

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

    # Provider status — inline badge
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
        label = f'Detected: {auto["provider"]}' if auto else f'Using: {provider}'
        variant = "success" if auto else "warning"
        st.markdown(
            f'<div class="vp-badge vp-badge-{variant}">{label}</div>',
            unsafe_allow_html=True,
        )
    elif has_env_key:
        st.markdown(
            f'<div class="vp-badge vp-badge-success">Using env: {env_key_name}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="vp-badge vp-badge-error">No API key set</div>',
            unsafe_allow_html=True,
        )

    # --- Test Connection button ---
    has_any_key = has_manual_key or has_env_key or bool(api_key)
    if has_any_key:
        test_col1, test_col2 = st.columns([2, 1])
        with test_col1:
            test_clicked = st.button("Test Connection", type="secondary", use_container_width=True, key="test_btn")
        with test_col2:
            if "conn_status" in st.session_state:
                cs = st.session_state.conn_status
                variant = "success" if cs["connected"] else "error"
                st.markdown(
                    f'<div class="vp-badge vp-badge-{variant}">'
                    f'{"Connected" if cs["connected"] else "Failed"}</div>',
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
                cs["provider"] = st.session_state.provider
                cs["model_tested"] = st.session_state.model
                st.session_state.conn_status = cs
                if cs["connected"]:
                    st.toast("Connected successfully!", icon="✅")
                else:
                    st.toast("Connection failed", icon="❌")

        # Show persistent error from last connection test
        if "conn_status" in st.session_state and not st.session_state.conn_status["connected"]:
            cs = st.session_state.conn_status
            err = cs.get("error", "")
            prov = cs.get("provider", "?")
            mdl = cs.get("model_tested", "?")
            if err:
                st.markdown(
                    f'<div class="vp-conn-error">'
                    f'<strong>{prov}</strong> · {mdl}<br>{err}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

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
        lines = tip.replace("\n", "<br>")
        st.markdown(
            f'<div style="font-size:11px;margin-top:4px;'
            f'padding:6px 10px;background:var(--vp-bg-surface);border-radius:6px;line-height:1.5;'
            f'color:var(--vp-text-muted);font-family:var(--vp-font-sans);">'
            f'Recommended models:<br>{lines}</div>',
            unsafe_allow_html=True,
        )

    # --- Pipeline settings (collapsible) ---
    with st.expander("Pipeline Settings", expanded=False):
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
        'border:2px solid #ffffff;box-shadow:0 2px 8px rgba(0,0,0,0.15);">'
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
    """Copy button with clipboard API + textarea fallback."""
    safe = text.replace("&", "&amp;")
    safe = safe.replace('"', "&quot;")
    safe = safe.replace("<", "&lt;")
    safe = safe.replace(">", "&gt;")
    safe_nl = safe.replace("\n", "\\n").replace("\r", "\\r")
    return (
        f'<button id="{button_id}" class="vp-copy-btn" '
        f'data-text="{safe}" '
        f'onclick="(function(){{'
        f"var t=this.getAttribute('data-text');"
        f'var cp=function(){{'
        f"var ta=document.createElement('textarea');"
        f"ta.value=t;ta.style.position='fixed';ta.style.left='-9999px';"
        f"document.body.appendChild(ta);ta.select();"
        f"document.execCommand('copy');document.body.removeChild(ta);"
        f"}};try{{navigator.clipboard.writeText(t).then(cp).catch(cp);}}"
        f"catch(e){{cp();}}"
        f"this.textContent='Copied';var btn=this;"
        f"setTimeout(function(){{btn.textContent='Copy';}},1500);"
        f'}})()" '
        f'aria-label="Copy text to clipboard">'
        f'Copy</button>'
    )


def _status_badge(text: str, variant: str = "success") -> str:
    """HTML status badge."""
    return (
        f'<span class="vp-badge vp-badge-{variant}">{text}</span>'
    )



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
                '<span style="color:var(--vp-text-dim);margin:0 4px;font-size:11px;font-weight:300;">→</span>'
            )

        if stage == error:
            bg, fg = "var(--vp-badge-error-bg)", "var(--vp-badge-error-fg)"
            icon = "✗"
        elif stage in completed:
            bg, fg = "var(--vp-badge-success-bg)", "var(--vp-badge-success-fg)"
            icon = "✓"
        elif stage == active:
            bg, fg = "#1e3a5f", "var(--vp-primary)"
            icon = "●"
        else:
            bg, fg = "var(--vp-bg-surface)", "var(--vp-text-dim)"
            icon = "○"

        border = 'border:1px solid var(--vp-primary);' if stage == active else ''
        parts.append(
            f'<span style="display:inline-flex;align-items:center;gap:3px;'
            f'padding:3px 10px;border-radius:14px;font-size:11px;font-weight:600;'
            f'background:{bg};color:{fg};{border}line-height:1.3;'
            f'font-family:-apple-system,BlinkMacSystemFont,sans-serif;">'
            f'{icon} {stage}</span>'
        )

    return '<div style="display:flex;align-items:center;flex-wrap:wrap;gap:2px;">' + "".join(parts) + "</div>"


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("VoicePrint")
st.caption("Transform AI-generated text into natural, human-like writing.")

col_input, col_output = st.columns(2)

with col_input:
    st.markdown("**Input**")
    input_text = st.text_area(
        "Input text",
        height=300,
        placeholder="Paste your AI-generated text here...",
        label_visibility="collapsed",
        key="input_text",
    )
    char_count = len(input_text.strip()) if input_text else 0
    if input_text:
        if char_count < 10:
            st.caption(f":red[{char_count} chars — minimum 10 required]")
        else:
            pct = min(char_count / 500, 1.0)
            bar_color = "var(--vp-success)" if char_count >= 10 else "var(--vp-error)"
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-top:2px;">'
                f'<div style="flex:1;height:4px;background:var(--vp-border);border-radius:2px;overflow:hidden;">'
                f'<div style="width:{pct*100:.0f}%;height:100%;background:{bar_color};'
                f'border-radius:2px;transition:width 200ms;"></div></div>'
                f'<span style="font-size:12px;color:var(--vp-text-muted);white-space:nowrap;font-variant-numeric:tabular-nums;">'
                f'{char_count:,} chars</span></div>',
                unsafe_allow_html=True,
            )


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
        '<div style="padding-top:10px;text-align:center;color:var(--vp-text-muted);font-size:12px">'
        '<kbd class="vp-kbd">Ctrl</kbd> + <kbd class="vp-kbd">Enter</kbd>'
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
        st.warning("Please enter some text to humanize.", icon="⚠️")
    elif not api_key and not preset.get("env_key"):
        st.error(
            "No API key configured. Add one in the sidebar, or set the "
            f"`{preset.get('env_key', 'OPENCODE_API_KEY')}` environment variable.",
            icon="❌",
        )
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
            f'<div class="vp-card" style="min-height:270px;max-height:400px;overflow-y:auto;'
            f'font-size:14px;line-height:1.7;white-space:pre-wrap;">'
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

    st.markdown("<div style='margin-top:16px;margin-bottom:16px;'></div>", unsafe_allow_html=True)

    method_tag = (
        "Statistical" if "statistical" in result.detection_summary
        else "Model"
    )
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
        f'<span style="font-size:14px;font-weight:600;">Detection Scores</span>'
        f'<span class="vp-badge vp-badge-success">{method_tag}</span></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ai_pct = result.ai_probability
        good = ai_pct < 0.5
        st.metric(
            "AI Probability",
            f"{ai_pct:.1%}",
            delta="Human" if good else "AI",
            delta_color="normal" if good else "inverse",
        )
        st.caption("Below 50% = human-like")
    with c2:
        sim = result.similarity
        good = sim >= 0.68
        st.metric(
            "Similarity",
            f"{sim:.1%}",
            delta="Preserved" if good else "Low",
            delta_color="normal" if good else "inverse",
        )
        st.caption("Above 68% = meaning kept")
    with c3:
        bur = result.burstiness
        good = 0.4 <= bur <= 0.7
        st.metric(
            "Burstiness",
            f"{bur:.2f}",
            delta="Natural" if good else "Extreme",
            delta_color="normal" if good else "inverse",
        )
        st.caption("0.4–0.7 = natural rhythm")
    with c4:
        ps = result.pattern_score
        good = ps < 0.1
        st.metric(
            "Pattern Score",
            f"{ps:.3f}",
            delta="Clean" if good else "AI patterns",
            delta_color="normal" if good else "inverse",
        )
        st.caption("Below 0.1 = few AI signals")

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
        st.text(result.detection_summary)
        if hasattr(result, "perplexity") and result.perplexity is not None:
            st.caption(f"Perplexity score: {result.perplexity:.1f} (higher = more human-like)")
    with tab2:
        r = result.readability
        bd = result.burstiness_detail
        cols = st.columns(2)
        if r:
            with cols[0]:
                st.markdown("**Readability**")
                for k, v in r.items():
                    label = k.replace("_", " ").title()
                    if isinstance(v, float):
                        st.markdown(f"- **{label}:** {v:.2f}")
                    else:
                        st.markdown(f"- **{label}:** {v}")
        if bd:
            with cols[1]:
                st.markdown("**Burstiness Detail**")
                for k, v in bd.items():
                    label = k.replace("_", " ").title()
                    if isinstance(v, float):
                        st.markdown(f"- **{label}:** {v:.4f}")
                    else:
                        st.markdown(f"- **{label}:** {v}")
    with tab3:
        sig = result.signals
        if sig:
            for k, v in sig.items():
                label = k.replace("_", " ").title()
                if isinstance(v, float):
                    st.markdown(f"- **{label}:** {v:.4f}")
                else:
                    st.markdown(f"- **{label}:** {v}")
    with tab4:
        for s in result.stages:
            st.markdown(f"- {s}")

    # --- Download row ---
    st.divider()
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download text",
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
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--vp-text-dim)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4L16.5 3.5z"/>
    </svg>
    <h3>Ready to humanize</h3>
    <p>Enter AI-generated text in the input panel, then click Humanize (Ctrl+Enter).</p>
    <p style="font-size:12px;color:var(--vp-text-dim);margin-top:8px;">4-stage pipeline &mdash; scrub, paraphrase, polish, detect</p>
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
