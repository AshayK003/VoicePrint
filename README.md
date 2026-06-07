# VoicePrint — AI Text Humanizer

Transform AI-generated text into natural, human-like writing that bypasses state-of-the-art detectors.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
python -m nltk.downloader punkt tabulate

# Set up API key (get free Gemini key at https://aistudio.google.com/apikey)
cp .env.example .env
# Edit .env and add your key

# Run the app
streamlit run app.py
```

## Architecture

4-stage pipeline:

1. **Heuristic Scrub** — Replace AI phrases, force burstiness, inject contractions
2. **Adversarial Paraphrasing** — LLM generates N candidates, select best
3. **Detection Scoring** — Ensemble of RoBERTa + Binoculars detectors
4. **Style Polish** — Passive→active, rhetorical questions, fragments

## Cost

~$0.01–0.03 per humanization using free-tier LLM APIs.

## License

MIT
