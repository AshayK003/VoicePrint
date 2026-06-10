"""Stage 3: AI-Pattern Detection & Scoring.

Three-tier detection:
  1. Statistical pre-filter (instant, ~0MB) — burstiness + pattern signals
  2. Binoculars zero-shot (fast, ~2GB) — perplexity ratio between two LMs
  3. RoBERTa ensemble (heavy, ~1.9GB) — fine-tuned classifier

Most AI text scores >0.7 on statistics alone, so models are skipped entirely.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from .config import Config, load_config
from .metrics import burstiness
from .patterns import pattern_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level model caches (survive across Streamlit reruns)
# ---------------------------------------------------------------------------

_tokenizer_cache: dict[str, Any] = {}
_classifier_cache: dict[str, Any] = {}
_binoculars_cache: dict[str, Any] = {}  # {model_name: (tokenizer, model)}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Result from a single detector."""
    name: str
    p_ai: float  # Probability of being AI-generated (0 = human, 1 = AI)
    label: str   # "REAL" or "FAKE"


@dataclass
class EnsembleResult:
    """Combined result from the detection ensemble."""
    p_ai: float          # Weighted average probability
    detectors: list[DetectionResult]
    passed: bool         # True if below detection threshold
    method: str = "model"  # "statistical" or "model"
    perplexity: float | None = None  # Raw perplexity (higher = more human)
    perplexity_score: float | None = None  # Normalized 0-1 (1 = most human-like)

    def summary(self) -> str:
        tag = "[STATISTICAL]" if self.method == "statistical" else "[MODEL]"
        lines = [f"{tag} Ensemble p_ai: {self.p_ai:.3f} ({'HUMAN' if self.passed else 'AI'})"]
        for d in self.detectors:
            lines.append(f"  {d.name}: {d.p_ai:.3f} ({d.label})")
        if self.perplexity is not None:
            lines.append(f"  perplexity: {self.perplexity:.1f} (score: {self.perplexity_score:.2f})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Statistical pre-filter (instant, no models loaded)
# ---------------------------------------------------------------------------

# Thresholds: if statistical score is outside this range, skip models
_STAT_LOW = 0.20   # Below this → almost certainly human
_STAT_HIGH = 0.80  # Above this → almost certainly AI


def _statistical_score(text: str) -> float:
    """Compute a fast AI-likelihood score from statistics alone.

    Combines burstiness (low = AI) and pattern_score (high = AI)
    into a single 0-1 score where 1 = definitely AI.
    """
    b = burstiness(text)
    p = pattern_score(text)

    # Invert burstiness: low burstiness → high AI score
    # Human burstiness is 0.4-0.7, AI is <0.3
    b_score = max(0.0, min(1.0, 1.0 - (b - 0.1) * 2.0))

    # Weighted blend: pattern_score is more reliable
    return 0.4 * b_score + 0.6 * p


def statistical_detect(text: str, threshold: float = 0.5) -> EnsembleResult:
    """Instant detection using statistics only. No models loaded."""
    score = _statistical_score(text)
    passed = score < threshold
    label = "FAKE" if score > 0.5 else "REAL"

    return EnsembleResult(
        p_ai=score,
        detectors=[DetectionResult(name="statistical", p_ai=score, label=label)],
        passed=passed,
        method="statistical",
    )


# ---------------------------------------------------------------------------
# RoBERTa detector
# ---------------------------------------------------------------------------

class RoBERTaDetector:
    """RoBERTa-based AI text detector."""

    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        if device:
            self.device = device
        else:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        if self.model_name in _tokenizer_cache:
            self.tokenizer = _tokenizer_cache[self.model_name]
            self.model = _classifier_cache[self.model_name]
        else:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch  # noqa: F401 — ensures torch is importable before model loads
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            ).to(self.device)
            _tokenizer_cache[self.model_name] = self.tokenizer
            _classifier_cache[self.model_name] = self.model
        self.model.eval()
        self._loaded = True

    def detect(self, text: str) -> DetectionResult:
        """Run detection on a single text."""
        self._load()
        import torch
        import torch.nn.functional as F

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)

        p_ai = probs[0][1].item()
        label = "FAKE" if p_ai > 0.5 else "REAL"

        short_name = self.model_name.split("/")[-1]
        return DetectionResult(name=short_name, p_ai=p_ai, label=label)


# ---------------------------------------------------------------------------
# Binoculars zero-shot detector (perplexity ratio)
# ---------------------------------------------------------------------------

_BINOCULARS_MODEL_A = "gpt2"
_BINOCULARS_MODEL_B = "gpt2-medium"


class BinocularsDetector:
    """Zero-shot AI detection via perplexity ratio between two LMs.

    Uses GPT-2 and GPT-2-medium. AI text has ratio ~1.0 (both models
    find it equally natural). Human text has more variance.
    """

    def __init__(self, device: str | None = None):
        if device:
            self.device = device
        else:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        import torch  # noqa: F401
        for name in (_BINOCULARS_MODEL_A, _BINOCULARS_MODEL_B):
            if name not in _binoculars_cache:
                if name == _BINOCULARS_MODEL_A:
                    from .perplexity import _get_gpt2
                    model, tok = _get_gpt2()
                    if model is not None:
                        _binoculars_cache[name] = (tok, model)
                        continue
                from transformers import AutoTokenizer, AutoModelForCausalLM
                tok = AutoTokenizer.from_pretrained(name)
                mod = AutoModelForCausalLM.from_pretrained(name).to(self.device)
                mod.eval()
                _binoculars_cache[name] = (tok, mod)
        self._loaded = True

    def _perplexity(self, text: str, model_name: str) -> float:
        import torch
        tok, mod = _binoculars_cache[model_name]
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = mod(**inputs, labels=inputs["input_ids"])
        return torch.exp(outputs.loss).item()

    def detect(self, text: str) -> DetectionResult:
        """Compute Binoculars score and classify."""
        self._load()
        ppl_a = self._perplexity(text, _BINOCULARS_MODEL_A)
        ppl_b = self._perplexity(text, _BINOCULARS_MODEL_B)
        score = ppl_a / ppl_b if ppl_b > 0 else 1.0

        # Thresholds from the Binoculars paper:
        # AI text: ratio close to 1.0 (both models equally "natural")
        # Human text: ratio deviates from 1.0
        # Map ratio to 0-1 p_ai: 1.0 → high p_ai, deviation → low p_ai
        p_ai = max(0.0, min(1.0, 1.0 - abs(score - 1.0) * 5.0))
        label = "FAKE" if p_ai > 0.5 else "REAL"

        return DetectionResult(
            name="binoculars",
            p_ai=p_ai,
            label=label,
        )


# ---------------------------------------------------------------------------
# Ensemble (statistical pre-filter → model fallback)
# ---------------------------------------------------------------------------

class DetectorEnsemble:
    """Three-tier detection ensemble.

    Tier 1: Statistical pre-filter (instant) — burstiness + pattern signals.
    Tier 2: Binoculars zero-shot + RoBERTa models — only if ambiguous.
    """

    def __init__(self, config: Config | None = None):
        config = config or load_config()
        self.config = config

        # Lazy-loaded detectors (only created if needed)
        self._roberta_detectors = None
        self._binoculars: BinocularsDetector | None = None

    def _ensure_roberta(self):
        """Lazily initialize RoBERTa detectors on first model-based detection."""
        if self._roberta_detectors is not None:
            return
        self._roberta_detectors = [
            RoBERTaDetector(self.config.primary_detector),   # weight 0.65
            RoBERTaDetector(self.config.secondary_detector),  # weight 0.35
        ]
        self._roberta_weights = [0.65, 0.35]

    def _detect_one(self, detector: RoBERTaDetector, text: str) -> DetectionResult | None:
        """Run a single detector, returning None on failure."""
        try:
            return detector.detect(text)
        except Exception as e:
            logger.warning(f"{detector.__class__.__name__} failed: {e}")
            return None

    def _run_binoculars(self, text: str) -> DetectionResult | None:
        """Run Binoculars zero-shot detection, returning None on failure."""
        if self._binoculars is None:
            self._binoculars = BinocularsDetector()
        try:
            return self._binoculars.detect(text)
        except Exception as e:
            logger.warning(f"Binoculars failed: {e}")
            return None

    def detect(self, text: str) -> EnsembleResult:
        """Detection: statistics first, models only if ambiguous and enabled."""
        # Tier 1: Instant statistical check
        stat = statistical_detect(text, self.config.detection_threshold)

        # If models disabled or score is clear, return statistical result
        if not self.config.use_models or stat.p_ai < _STAT_LOW or stat.p_ai > _STAT_HIGH:
            if not self.config.use_models:
                reason = "models disabled"
            elif stat.p_ai < _STAT_LOW:
                reason = "clear human"
            else:
                reason = "clear AI"
            logger.info(
                f"Statistical pre-filter: {stat.p_ai:.3f} → {reason} (models skipped)"
            )
            return stat

        # Tier 2: Ambiguous — run Binoculars + RoBERTa in parallel
        logger.info(f"Statistical pre-filter: {stat.p_ai:.3f} → ambiguous, running models")

        all_detectors: list[DetectionResult] = [stat.detectors[0]]
        weighted_sum = 0.0
        total_weight = 0.0

        with ThreadPoolExecutor(max_workers=3) as executor:
            # Binoculars: weight 0.30
            binoculars_future = executor.submit(self._run_binoculars, text)
            # RoBERTa: weight 0.70 total (0.455 primary, 0.245 secondary)
            self._ensure_roberta()
            roberta_futures = {
                executor.submit(self._detect_one, det, text): (det, w * 0.70)
                for det, w in zip(self._roberta_detectors, self._roberta_weights)
            }

            # Collect Binoculars result
            binoculars_result = binoculars_future.result()
            if binoculars_result is not None:
                all_detectors.append(binoculars_result)
                weighted_sum += binoculars_result.p_ai * 0.30
                total_weight += 0.30

            # Collect RoBERTa results
            for future in as_completed(roberta_futures):
                det, weight = roberta_futures[future]
                result = future.result()
                if result is not None:
                    all_detectors.append(result)
                    weighted_sum += result.p_ai * weight
                    total_weight += weight

        p_ai = weighted_sum / total_weight if total_weight > 0 else stat.p_ai
        blended_passed = p_ai < self.config.detection_threshold

        return EnsembleResult(
            p_ai=p_ai,
            detectors=all_detectors,
            passed=blended_passed,
            method="model",
        )
