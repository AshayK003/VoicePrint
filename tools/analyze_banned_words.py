"""Analyze AI_FAVORITE_WORDS frequency in human vs AI text.

Loads gsingh1-py/train (Comprehensive 2025 dataset) — 7.3k rows of NYT
articles rewritten by 6 LLMs. Computes occurrences per 1000 words for
each banned word and reports the AI/human ratio.

Usage:
    python tools/analyze_banned_words.py [--max-rows 2000]
"""

import re
import sys
from collections import Counter

from datasets import load_dataset

# The banned words from patterns.py
BANNED_WORDS = {
    "delve", "tapestry", "multifaceted", "nuanced", "nuance",
    "intricate", "intricacies", "meticulous", "bolster", "garner",
    "paramount", "groundbreaking", "seamless", "revolutionize",
    "unprecedented", "remarkable", "profound", "vibrant",
    "beacon", "cornerstone", "trajectory", "spectrum", "confluence",
    "landscape", "ecosystem", "synergy",
    "holistic", "robust", "comprehensive", "cutting-edge",
    "state-of-the-art", "innovative", "transformative", "paradigm",
    "unlock", "empower", "foster", "spearhead", "underpins",
    "underscores", "signifies", "exemplifies", "epitomizes",
    "furthermore", "moreover", "additionally", "consequently",
    "nevertheless", "nonetheless", "leverage", "utilize", "facilitate",
    "endeavor", "commence", "terminate", "myriad", "plethora",
    "aforementioned", "subsequent", "henceforth", "herein", "therein",
}

AI_COLUMNS = [
    "gemma-2-9b", "mistral-7B", "qwen-2-72B",
    "llama-8B", "accounts/yi-01-ai/models/yi-large", "GPT_4-o",
]
HUMAN_COL = "Human_story"


def word_count(text: str) -> int:
    return len(text.split())


def find_words(text: str) -> Counter:
    found: Counter = Counter()
    for w in re.findall(r"[a-zA-Z]+(?:-[a-zA-Z]+)*", text.lower()):
        if w in BANNED_WORDS:
            found[w] += 1
    return found


def main():
    max_rows = 2000
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.startswith("--max-rows="):
            max_rows = int(arg.split("=", 1)[1])

    ds = load_dataset("gsingh1-py/train", split="train", streaming=True)

    human_counts: Counter = Counter()
    ai_counts: Counter = Counter()
    human_total_words = 0
    ai_total_words = 0
    rows_processed = 0

    for i, row in enumerate(ds):
        if i >= max_rows:
            break

        human = row.get(HUMAN_COL, "")
        human_total_words += word_count(human)
        human_counts += find_words(human)

        for col in AI_COLUMNS:
            ai_text = row.get(col, "")
            ai_total_words += word_count(ai_text)
            ai_counts += find_words(ai_text)

        rows_processed += 1
        if rows_processed % 200 == 0:
            print(f"  Processed {rows_processed} rows...", file=sys.stderr)

    print(f"\nProcessed {rows_processed} rows ({human_total_words:,} human words, {ai_total_words:,} AI words)\n")
    print(f"{'Word':<25} {'Human/1k':>10} {'AI/1k':>10} {'Ratio':>8}  {'Keep?':>6}")
    print("-" * 65)

    words_to_remove = []
    words_to_keep = []

    for word in sorted(BANNED_WORDS, key=str.lower):
        h_freq = human_counts.get(word, 0) / max(human_total_words, 1) * 1000
        a_freq = ai_counts.get(word, 0) / max(ai_total_words, 1) * 1000
        ratio = a_freq / max(h_freq, 0.0001)

        keep = "KEEP" if ratio >= 2.0 else "REMOVE"
        if ratio >= 2.0:
            words_to_keep.append(word)
        else:
            words_to_remove.append(word)

        print(f"{word:<25} {h_freq:>10.4f} {a_freq:>10.4f} {ratio:>7.2f}x  {keep:>6}")

    print(f"\n--- Summary ---")
    print(f"  Keep ({len(words_to_keep)}): {', '.join(sorted(words_to_keep, key=str.lower))}")
    print(f"  Remove ({len(words_to_remove)}): {', '.join(sorted(words_to_remove, key=str.lower))}")


if __name__ == "__main__":
    main()
