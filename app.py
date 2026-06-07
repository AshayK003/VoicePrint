"""VoicePrint — Streamlit Frontend.

Run: streamlit run app.py
"""

import json

import streamlit as st

from voiceprint.config import Config, PROVIDER_PRESETS
from voiceprint.pipeline import HumanizePipeline

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

    provider = st.selectbox(
        "Provider",
        options=list(PROVIDER_PRESETS.keys()),
        index=0,
    )

    preset = PROVIDER_PRESETS[provider]

    # --- API Key ---
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="sk-... or AIza...",
        help="Your API key. Stored in session only — never saved to disk.",
    )

    # --- Base URL ---
    base_url = st.text_input(
        "Base URL (optional)",
        placeholder="https://your-api.com/v1",
        help="Custom OpenAI-compatible endpoint. Leave empty for default.",
    )

    # --- Model ---
    model = st.text_input(
        "Model",
        value=preset["model"],
        help="Model identifier for the selected provider.",
    )

    # --- Status indicator ---
    if api_key:
        st.success(f"✅ {provider} configured")
    else:
        st.warning(f"⚠️ No API key for {provider}")

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
    config.provider = provider
    config.api_key = api_key
    config.base_url = base_url
    config.llm_model = model

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
                "provider": provider,
                "model": model,
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
