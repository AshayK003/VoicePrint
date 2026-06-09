"""Benchmark VoicePrint on lynote-ai showcase examples.

Measures statistical detection scores, timing per stage,
and pystylometry signal contribution.
"""

import time
import statistics

from voiceprint.patterns import pattern_score, compute_all_signals, PYSTYLOMETRY_AVAILABLE
from voiceprint.scrub import scrub
from voiceprint.polish import polish
from voiceprint.metrics import burstiness, readability_scores, burstiness_report

SAMPLES = [
    {
        "name": "01_Quantum_Computing",
        "text": "Quantum computing has been proposed as a paradigm shift for solving certain optimization and simulation tasks. Unlike classical bits, qubits can represent superpositions, enabling parallel computation in specific problem structures. Although real-world advantage remains limited by hardware instability and error rates, research in quantum algorithms continues to expand. For industries such as logistics, chemistry, and cryptography, quantum readiness is becoming a strategic conversation. Companies can start by training technical teams and identifying quantum-friendly problem classes. Even before full-scale deployment, a quantum roadmap can act as a signal of innovation leadership and future resilience in computational competitiveness.",
    },
    {
        "name": "03_Sustainable_Supply_Chains",
        "text": "Sustainable supply chains require coordination across sourcing, production, transportation, and disposal. Academic research highlights that transparency reduces environmental and social risks, but implementation is challenging due to fragmented suppliers and inconsistent reporting standards. Technologies like blockchain and digital product passports are sometimes proposed, though their real impact depends on data accuracy and governance. For brands, sustainability claims must be evidence-based to avoid greenwashing concerns. Companies that invest in traceability can strengthen compliance readiness and customer trust. In a marketing context, transparency stories when supported by audit data create stronger differentiation and can justify premium pricing ethically.",
    },
    {
        "name": "04_Financial_Literacy",
        "text": "Financial literacy is correlated with improved household resilience and long-term wealth outcomes. Studies show that individuals who understand compound interest, budgeting, and risk diversification are less likely to accumulate high-cost debt. Yet financial education initiatives often fail if they remain abstract and not connected to real behavior. Practical tools such as automated savings, spending categorization, and goal-based planning support behavioral adoption. Fintech platforms increasingly package education with product features, making learning more actionable. From a societal perspective, scaling financial capability reduces economic vulnerability. From a business angle, helping users succeed financially can increase retention and trust in financial products.",
    },
    {
        "name": "05_Peer_Review_in_Science",
        "text": "Peer review remains a foundational mechanism for quality control in scientific publishing, yet it is not without limitations. Critics note issues such as reviewer bias, slow turnaround times, and inconsistent standards. Nevertheless, peer review provides a filter that often improves clarity, methodology, and replicability. Emerging models include open peer review, preprint feedback systems, and post-publication evaluation. For research institutions, improving review practices can enhance credibility and accelerate innovation cycles. In an increasingly competitive academic environment, transparent and efficient peer review not only strengthens scientific integrity but also supports broader knowledge dissemination and collaboration opportunities.",
    },
]


def _format_timing(timings: list[float]) -> str:
    avg = statistics.mean(timings) * 1000
    low = min(timings) * 1000
    high = max(timings) * 1000
    return f"{avg:.1f}ms (min={low:.1f}, max={high:.1f})"


def run_benchmark(runs: int = 5):
    print(f"=== VoicePrint Benchmark ===")
    print(f"Pystylometry available: {PYSTYLOMETRY_AVAILABLE}")
    print(f"Runs per sample: {runs}")
    print(f"Samples: {len(SAMPLES)}")
    print()

    for sample in SAMPLES:
        text = sample["text"]
        name = sample["name"]

        signal_times: list[float] = []
        scrub_times: list[float] = []
        polish_times: list[float] = []

        for _ in range(runs):
            # signals on original
            t0 = time.perf_counter()
            orig_signals = compute_all_signals(text)
            orig_score = pattern_score(text)
            t1 = time.perf_counter()
            signal_times.append(t1 - t0)

            # scrub
            t2 = time.perf_counter()
            scrubbed = scrub(text)
            t3 = time.perf_counter()
            scrub_times.append(t3 - t2)

            # polish
            polished = polish(scrubbed)
            t4 = time.perf_counter()
            polish_times.append(t4 - t3)

        # Final results (from last run)
        final_signals = compute_all_signals(polished)
        final_score = pattern_score(polished)
        scrub_score = pattern_score(scrubbed)
        improv = (orig_score - final_score) / orig_score * 100

        # Burstiness & readability on final output
        bur = burstiness(polished)
        read = readability_scores(polished)

        print(f"--- {name} {'-' * (40 - len(name))}")
        print(f"  Length:          {len(text.split())} words, {len(text)} chars")
        print(f"  Pattern scores:")
        print(f"    Original:      {orig_score:.4f}")
        print(f"    After scrub:   {scrub_score:.4f}")
        print(f"    Final:         {final_score:.4f}")
        print(f"    Improvement:   {improv:.1f}%")
        print(f"  Burstiness:      {bur:.4f}")
        print(f"  Readability:     Flesch {read.get('flesch_reading_ease', 'N/A')}, "
              f"GF {read.get('gunning_fog', 'N/A')}")
        print(f"  Signal count:    {len(orig_signals)} -> {len(final_signals)}")

        # Pystylometry signals
        pyst_keys = [k for k in final_signals if k.startswith("pystylometry_")]
        if pyst_keys:
            print(f"  Pystylometry:")
            for k in pyst_keys:
                print(f"    {k:25s}  {final_signals[k]:.4f}")

        # Timing
        print(f"  Timing:")
        print(f"    compute_signals:  {_format_timing(signal_times)}")
        print(f"    scrub:            {_format_timing(scrub_times)}")
        print(f"    polish:           {_format_timing(polish_times)}")

        # Signal breakdown (non-pystylometry)
        print(f"  Top AI signals (original):")
        sorted_sigs = sorted(orig_signals.items(), key=lambda x: x[1], reverse=True)
        for k, v in sorted_sigs[:5]:
            print(f"    {k:30s} {v:.4f}")
        print()


if __name__ == "__main__":
    run_benchmark()
