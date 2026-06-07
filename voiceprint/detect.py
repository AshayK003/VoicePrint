"""Stage 3: AI-Pattern Detection & Scoring.

Ensemble of open-source detectors to measure how "AI-like" text is.
Primary: RoBERTa-large-openai-detector
Secondary: chatgpt-detector-roberta
Zero-shot: Binoculars-style perplexity ratio
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .config import Config, load_config


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

    def summary(self) -> str:
        lines = [f"Ensemble p_ai: {self.p_ai:.3f} ({'HUMAN' if self.passed else 'AI'})"]
        for d in self.detectors:
            lines.append(f"  {d.name}: {d.p_ai:.3f} ({d.label})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Detector classes
# ---------------------------------------------------------------------------

class RoBERTaDetector:
    """RoBERTa-based AI text detector."""

    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self.device)
        self.model.eval()
        self._loaded = True

    def detect(self, text: str) -> DetectionResult:
        """Run detection on a single text."""
        self._load()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Model outputs [REAL, FAKE] — index 1 is AI probability
        p_ai = probs[0][1].item()
        label = "FAKE" if p_ai > 0.5 else "REAL"

        short_name = self.model_name.split("/")[-1]
        return DetectionResult(name=short_name, p_ai=p_ai, label=label)


class BinocularsDetector:
    """Zero-shot detection via perplexity ratio (simplified).

    Uses two models to compute a Binoculars-style score.
    Higher ratio → more likely AI-generated.
    """

    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        # Use two models of different sizes for perplexity comparison
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model_a = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device)
        self.model_b = AutoModelForCausalLM.from_pretrained("gpt2-medium").to(self.device)
        self.model_a.eval()
        self.model_b.eval()
        self._loaded = True

    def _perplexity(self, model, tokenizer, text: str) -> float:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        return torch.exp(outputs.loss).item()

    def detect(self, text: str) -> DetectionResult:
        """Run Binoculars-style detection."""
        self._load()
        ppl_a = self._perplexity(self.model_a, self.tokenizer, text)
        ppl_b = self._perplexity(self.model_b, self.tokenizer, text)

        # Binoculars score: ratio of perplexities
        # AI text tends to have ratio closer to 1.0 (both models find it predictable)
        # Human text has more variance between models
        ratio = ppl_a / ppl_b if ppl_b > 0 else 1.0

        # Map ratio to probability (calibrated empirically)
        # Lower ratio → more likely human
        p_ai = max(0.0, min(1.0, 1.0 - (ratio - 0.8) * 2.5))
        label = "FAKE" if p_ai > 0.5 else "REAL"

        return DetectionResult(name="binoculars", p_ai=p_ai, label=label)


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class DetectorEnsemble:
    """Ensemble of AI text detectors with weighted scoring."""

    def __init__(self, config: Config | None = None):
        config = config or load_config()
        self.config = config
        self.detectors = []
        self.weights = []

        # Primary: RoBERTa-large
        self.detectors.append(RoBERTaDetector(config.primary_detector))
        self.weights.append(0.5)

        # Secondary: chatgpt-detector-roberta
        self.detectors.append(RoBERTaDetector(config.secondary_detector))
        self.weights.append(0.3)

        # Zero-shot: Binoculars
        self.detectors.append(BinocularsDetector())
        self.weights.append(0.2)

    def detect(self, text: str) -> EnsembleResult:
        """Run all detectors and compute weighted average."""
        results = []
        weighted_sum = 0.0
        total_weight = 0.0

        for detector, weight in zip(self.detectors, self.weights):
            try:
                result = detector.detect(text)
                results.append(result)
                weighted_sum += result.p_ai * weight
                total_weight += weight
            except Exception as e:
                print(f"Warning: {detector.__class__.__name__} failed: {e}")
                continue

        p_ai = weighted_sum / total_weight if total_weight > 0 else 0.5
        passed = p_ai < self.config.detection_threshold

        return EnsembleResult(p_ai=p_ai, detectors=results, passed=passed)
