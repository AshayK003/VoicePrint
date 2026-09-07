# Changelog

All notable changes to VoicePrint are documented here. Format follows Keep a Changelog.

## [0.2.0] — 2026-09-07

### Security
- Reject prompt-injection patterns (`ignore previous instructions`, `<<SYS>>`, `<|system|>`, `<|user|>`) in input validation before text reaches LLM prompts.
- Escape humanized output and original text before rendering into `unsafe_allow_html` blocks (stored XSS via LLM or user input).

### Fixed
- Restored CI (`lint-and-test`: ruff + pytest on push/PR).
- Full suite collects without optional ML deps: lazy `textstat` import, real `Tensor` on the torch test stub, spaCy-gated restructure tests skip cleanly when `en_core_web_sm` is absent.
- `ThreadPoolExecutor` in the metrics stage capped at 4 workers.
- Removed dead code flagged by ruff (unused re-exports, dead locals, unused test imports); combined nested ifs; 64 ruff findings → 0.
- Tests that never asserted now assert on pipeline output; 3 new tests pin the injection guard.

### Changed
- `select_best` perplexity floor is now a named constant (`PERPLEXITY_FLOOR = 30.0`) with debug logging on skipped candidates.

## [0.1.0] — 2026-07-26
- Initial 4-stage pipeline (scrub → paraphrase → restructure → detect → polish) with detection-guided refinement, 345-test suite, Streamlit UI.
