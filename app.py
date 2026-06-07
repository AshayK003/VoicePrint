"""VoicePrint — Streamlit Frontend.

Run: streamlit run app.py
"""

import json

import streamlit as st

from voiceprint.config import (
    Config,
    PROVIDER_PRESETS,
    PROVIDER_MODELS,
    PROVIDER_BASE_URLS,
)
from voiceprint.pipeline import HumanizePipeline


# ---------------------------------------------------------------------------
# API Key auto-detection
# ---------------------------------------------------------------------------

def detect_provider_from_key(api_key: str) -> dict | None:
    """Detect provider, model, and base URL from API key prefix.

    Returns dict with provider, model, base_url or None if unknown.
    """
    key = api_key.strip()

    # Google Gemini — starts with "AIza"
    if key.startswith("AIza"):
        return {
            "provider": "Google Gemini (Free)",
            "model": "gemini/gemini-2.0-flash",
            "base_url": "",
        }

    # OpenAI — starts with "sk-" (but not "sk-ant-")
    if key.startswith("sk-") and not key.startswith("sk-ant-"):
        return {
            "provider": "OpenAI",
            "model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1",
        }

    # Anthropic — starts with "sk-ant-"
    if key.startswith("sk-ant-"):
        return {
            "provider": "Anthropic",
            "model": "claude-3-5-haiku-20241022",
            "base_url": "",
        }

    # Groq — starts with "gsk_"
    if key.startswith("gsk_"):
        return {
            "provider": "Groq (Free)",
            "model": "groq/llama-3.3-70b-versatile",
            "base_url": "",
        }

    # Mistral — starts with "ak-"
    if key.startswith("ak-"):
        return {
            "provider": "Mistral (Free)",
            "model": "mistral/mistral-large-latest",
            "base_url": "",
        }

    return None

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VoicePrint — AI Text Humanizer",
    page_icon="🎭",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — Provider & API Configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🎭 VoicePrint")
    st.caption("AI Text Humanizer v0.1.0")

    st.divider()

    # --- Provider selection ---
    st.subheader("🔑 LLM Provider")

    # --- API Key (input first so auto-detect can update provider) ---
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="Paste your API key — auto-detects provider",
        help="Paste any API key. Provider, model, and base URL are auto-detected.",
    )

    # Auto-detect provider from key prefix
    auto = detect_provider_from_key(api_key) if api_key else None

    # Session state for provider/model/base_url
    if "provider" not in st.session_state:
        st.session_state.provider = list(PROVIDER_PRESETS.keys())[0]
    if "base_url" not in st.session_state:
        st.session_state.base_url = ""
    if "model" not in st.session_state:
        st.session_state.model = PROVIDER_PRESETS[st.session_state.provider]["model"]

    # Update session state when auto-detect fires
    if auto:
        if st.session_state.provider != auto["provider"]:
            st.toast(f"Auto-detected: {auto['provider']}", icon="🔍")
        st.session_state.provider = auto["provider"]
        st.session_state.model = auto["model"]
        st.session_state.base_url = auto["base_url"]

    # Provider dropdown (reflects auto-detection)
    provider = st.selectbox(
        "Provider",
        options=list(PROVIDER_PRESETS.keys()),
        index=list(PROVIDER_PRESETS.keys()).index(st.session_state.provider),
        key="provider_select",
    )
    st.session_state.provider = provider

    preset = PROVIDER_PRESETS[provider]

    # --- Model dropdown ---
    model_options = PROVIDER_MODELS.get(provider, [])
    model_options_with_custom = list(model_options) + ["Custom..."]
    current_model = st.session_state.model

    # Find closest match in model options
    try:
        model_idx = model_options.index(current_model)
    except ValueError:
        model_idx = len(model_options_with_custom) - 1  # "Custom..."

    chosen_model = st.selectbox(
        "Model",
        options=model_options_with_custom,
        index=model_idx,
        key="model_select",
    )

    if chosen_model == "Custom...":
        model = st.text_input(
            "Custom model",
            value="" if current_model in model_options else current_model,
            placeholder="e.g. gpt-4-turbo",
            key="model_custom",
        )
    else:
        model = chosen_model
    st.session_state.model = model

    # --- Base URL dropdown ---
    base_url_options = PROVIDER_BASE_URLS.get(provider, [])
    base_url_options_with_custom = base_url_options + ["Custom..."]
    current_base_url = st.session_state.base_url

    try:
        bu_idx = base_url_options.index(current_base_url)
    except ValueError:
        bu_idx = len(base_url_options_with_custom) - 1  # "Custom..."

    chosen_base_url = st.selectbox(
        "Base URL",
        options=base_url_options_with_custom,
        index=bu_idx if current_base_url else 0,
        key="base_url_select",
    )

    if chosen_base_url == "Custom...":
        base_url = st.text_input(
            "Custom base URL",
            value="" if current_base_url in base_url_options else current_base_url,
            placeholder="https://your-api.com/v1",
            key="base_url_custom",
        )
    elif chosen_base_url == "(default)":
        base_url = ""
    else:
        base_url = chosen_base_url
    st.session_state.base_url = base_url

    # --- Status indicator ---
    if api_key:
        if auto:
            st.success(f"✅ {auto['provider']} — auto-detected")
        else:
            st.info(f"ℹ️ Key entered — using {provider}")
    else:
        st.warning(f"⚠️ No API key")

    st.divider()

    # --- Pipeline settings ---
    st.subheader("⚙️ Pipeline Settings")
    use_scrub = st.checkbox("Stage 1: Heuristic Scrub", value=True)
    use_paraphrase = st.checkbox("Stage 2: LLM Paraphrasing", value=True)
    use_polish = st.checkbox("Stage 4: Style Polish", value=True)
    n_candidates = st.slider("Candidates (N)", 1, 16, 8)

    st.divider()

    # --- About ---
    st.subheader("About")
    st.markdown("""
    Multi-stage pipeline that transforms AI-generated text
    into natural, human-like writing.

    **Stages:**
    1. Heuristic scrub (no model)
    2. Adversarial paraphrasing (LLM API)
    3. Detection scoring (ensemble)
    4. Style polish (no model)
    """)

# ---------------------------------------------------------------------------
# Build Config from sidebar inputs
# ---------------------------------------------------------------------------

def build_config() -> Config:
    """Construct a Config from sidebar inputs."""
    config = Config()
    config.provider = st.session_state.provider
    config.api_key = api_key
    config.base_url = st.session_state.base_url
    config.llm_model = st.session_state.model

    # Resolve env var if no manual key provided
    if not config.api_key and preset["env_key"]:
        import os
        config.api_key = os.getenv(preset["env_key"], "")

    return config

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("🎭 VoicePrint")
st.markdown("Transform AI-generated text into natural, human-like writing.")

# Text input
col_input, col_output = st.columns(2)

with col_input:
    st.subheader("Input")
    input_text = st.text_area(
        "Paste your AI-generated text here:",
        height=300,
        placeholder="Enter text to humanize...",
        label_visibility="collapsed",
    )

# Humanize button
if st.button("🎨 Humanize", type="primary", use_container_width=True):
    if not input_text.strip():
        st.warning("Please enter some text to humanize.")
    elif not api_key and not preset["env_key"]:
        st.error("Please enter an API key in the sidebar.")
    else:
        with st.spinner("Running pipeline..."):
            config = build_config()
            pipe = HumanizePipeline(config=config)
            result = pipe.run(
                input_text,
                use_scrub=use_scrub,
                use_paraphrase=use_paraphrase,
                use_polish=use_polish,
                n_candidates=n_candidates,
            )

        # Display output
        with col_output:
            st.subheader("Output")
            st.text_area(
                "Humanized text:",
                value=result.humanized,
                height=300,
                disabled=True,
                label_visibility="collapsed",
            )

        # Metrics dashboard
        st.divider()
        st.subheader("📊 Detection Scores")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "AI Probability",
                f"{result.detection.p_ai:.1%}",
                delta="Human" if result.detection.passed else "AI",
                delta_color="normal" if result.detection.passed else "inverse",
            )
        with col2:
            st.metric(
                "Similarity",
                f"{result.similarity:.1%}",
                delta=f"≥ 78%" if result.similarity >= 0.78 else "Below threshold",
            )
        with col3:
            st.metric(
                "Burstiness",
                f"{result.burstiness:.2f}",
                delta="Human range" if 0.4 <= result.burstiness <= 0.7 else "Outside range",
            )
        with col4:
            st.metric(
                "Pattern Score",
                f"{result.pattern_score:.3f}",
                delta="Low = human" if result.pattern_score < 0.1 else "High = AI",
            )

        # Detailed breakdown
        with st.expander("🔍 Detailed Detection Results"):
            st.code(result.detection.summary(), language=None)

        with st.expander("📈 Readability Scores"):
            cols = st.columns(2)
            with cols[0]:
                st.json(result.readability)
            with cols[1]:
                st.json(result.burstiness_detail)

        with st.expander("🧬 AI Pattern Signals"):
            st.json(result.signals)

        with st.expander("⚙️ Stages Applied"):
            st.write(f"Pipeline stages: {', '.join(result.stages_applied)}")

        # Export
        st.divider()
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "📥 Download Result",
                data=result.humanized,
                file_name="humanized_text.txt",
                mime="text/plain",
            )
        with col_dl2:
            report = {
                "provider": st.session_state.provider,
                "model": st.session_state.model,
                "ai_probability": result.detection.p_ai,
                "similarity": result.similarity,
                "burstiness": result.burstiness,
                "pattern_score": result.pattern_score,
                "readability": result.readability,
                "signals": result.signals,
            }
            st.download_button(
                "📥 Download Report (JSON)",
                data=json.dumps(report, indent=2),
                file_name="detection_report.json",
                mime="application/json",
            )
else:
    # Pre-check mode
    if input_text.strip():
        with col_output:
            st.info("Click **Humanize** to start the pipeline.")

        # Quick pre-check (no API key needed — uses local detectors)
        if st.button("🔍 Quick Detection Check", use_container_width=False):
            with st.spinner("Checking..."):
                pipe = HumanizePipeline()
                pre_result = pipe.detect_only(input_text)

            st.metric(
                "Pre-check AI Probability",
                f"{pre_result.p_ai:.1%}",
            )
            st.code(pre_result.summary(), language=None)
