"""
VoicePrint — Training Data Preparation
=======================================
Processes downloaded datasets into unified training-ready format.

Steps:
  1. HC3           → {question → (human_answers, chatgpt_answers)} pairs
  2. CMU Human-AI  → {human_chunk → model_continuation} pairs (doc_id aligned)
  3. ai2human      → Human reference passages (source for style analysis)

Output: data/processed/ with train/val/test splits in JSONL format.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
TEST_RATIO = 0.10
VAL_RATIO = 0.10

# Path for intermediate + output files
HC3_IN = DATA_DIR / "training" / "hc3"
CMU_IN = DATA_DIR / "training" / "human-ai-parallel"
AI2HUMAN_IN = DATA_DIR / "training" / "ai2human"
OUT_DIR = DATA_DIR / "processed"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Process HC3
# ---------------------------------------------------------------------------

HC3_DOMAINS = {
    "finance.jsonl": "finance",
    "medicine.jsonl": "medicine",
    "open_qa.jsonl": "open_qa",
    "reddit_eli5.jsonl": "reddit_eli5",
    "wiki_csai.jsonl": "wiki_csai",
}


def process_hc3() -> list[dict]:
    """Convert HC3 parallel data into (ai_text, human_text) pairs.

    HC3 stores: question, human_answers (list), chatgpt_answers (list).
    We pair each ChatGPT answer with the corresponding human answer.
    When multiple human answers exist, we pair each one.
    """
    records = []
    skipped_domains = 0

    for fname, domain in HC3_DOMAINS.items():
        path = HC3_IN / fname
        if not path.exists():
            logger.warning(f"  HC3 file missing: {fname} — skipping")
            skipped_domains += 1
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue

                question = d.get("question", "").strip()
                human_answers = d.get("human_answers", [])
                chatgpt_answers = d.get("chatgpt_answers", [])

                if not question or not human_answers or not chatgpt_answers:
                    continue

                # Pair each ChatGPT answer with a human answer
                for i, ai_text in enumerate(chatgpt_answers):
                    if not ai_text or not ai_text.strip():
                        continue
                    # Match with human answer (cycle if more AI answers)
                    human_text = human_answers[i % len(human_answers)]
                    if not human_text or not human_text.strip():
                        continue

                    records.append({
                        "id": f"hc3-{domain}-{len(records)}",
                        "source_text": ai_text.strip(),
                        "target_text": human_text.strip(),
                        "dataset": "hc3",
                        "domain": domain,
                        "question": question,
                    })

        logger.info(f"  {fname}: {sum(1 for r in records if r['domain'] == domain)} HC3 pairs")

    if skipped_domains:
        logger.warning(f"  Skipped {skipped_domains} missing HC3 domain files")

    logger.info(f"  TOTAL HC3 pairs: {len(records)}")
    return records


# ---------------------------------------------------------------------------
# 2. Process CMU Human-AI Parallel Corpus
# ---------------------------------------------------------------------------

def process_cmu() -> list[dict]:
    """Convert CMU parquet corpus into (ai_text, human_text) pairs.

    The corpus has doc_id-aligned parquet files:
      - hape-text_human-chunk-1.parquet — human-written text (chunk 1)
      - hape-text_human-chunk-2.parquet — human-written text (chunk 2)
      - hape-text_gpt-4o-*.parquet       — GPT-4o continuations
      - hape-text_gpt-4o-mini-*.parquet  — GPT-4o-mini continuations
      - hape-text_llama-3-*.parquet      — Llama 3 continuations

    Strategy: pair human chunks with model continuations via doc_id.
    For VoicePrint, each human chunk is paired with each AI model's
    continuation of the same doc_id.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        logger.error("pyarrow not installed. Run: pip install pyarrow")
        return []

    records = []

    # Load human chunks (doc_id → text)
    human_chunk_files = sorted(CMU_IN.glob("hape-text_human-chunk-*.parquet"))
    if not human_chunk_files:
        logger.warning("  No human chunk parquet files found — skipping CMU")
        return []

    human_by_doc = defaultdict(list)
    for hf in human_chunk_files:
        pf = pq.ParquetFile(str(hf))
        tbl = pf.read()
        for doc_id, text in zip(
            tbl.column("doc_id").to_pylist(),
            tbl.column("text").to_pylist()
        ):
            if text and text.strip():
                human_by_doc[doc_id].append(text.strip())

    logger.info(f"  Loaded {len(human_by_doc)} unique doc_ids from human chunks")

    # Load model-generated parquet files
    model_files = sorted(CMU_IN.glob("hape-text_*.parquet"))
    model_files = [f for f in model_files if "human-chunk" not in f.name]

    if not model_files:
        logger.warning("  No AI model parquet files found — skipping CMU pairs")
        return records

    for mf in model_files:
        # Extract model name from filename
        model_name = mf.stem.replace("hape-text_", "")
        pf = pq.ParquetFile(str(mf))
        tbl = pf.read()

        pair_count = 0
        for doc_id, text in zip(
            tbl.column("doc_id").to_pylist(),
            tbl.column("text").to_pylist()
        ):
            if not text or not text.strip():
                continue
            ai_text = text.strip()

            # Find matching human text
            # doc_id format: acad_0001@model-name
            base_doc = doc_id.split("@")[0]
            human_key = f"{base_doc}"
            # Try matching with chunk suffix
            human_texts = []
            for hkey, htexts in human_by_doc.items():
                if hkey.startswith(base_doc):
                    human_texts.extend(htexts)

            if not human_texts:
                continue

            # Use the first human chunk as the "target" for rewriting
            # (The human text is the original, the AI text is the continuation)
            human_text = human_texts[0]

            records.append({
                "id": f"cmu-{base_doc}-{model_name}-{len(records)}",
                "source_text": ai_text,
                "target_text": human_text,
                "dataset": "cmu",
                "domain": "academic",
                "doc_id": doc_id,
            })
            pair_count += 1

        if pair_count:
            logger.info(f"  {model_name}: {pair_count} pairs")

    logger.info(f"  TOTAL CMU pairs: {len(records)}")
    return records


# ---------------------------------------------------------------------------
# 3. Process ai2human data
# ---------------------------------------------------------------------------

def process_ai2human() -> list[dict]:
    """Extract human-written passages from ai2human data.

    The ai2human repo provides 10K human-written student essays.
    We use these as pure human-reference text for style analysis.
    They can also supplement the target_text pool.

    Format: text separated by blank lines (paragraphs or passages).
    """
    records = []

    human_path = AI2HUMAN_IN / "human_sentences_10k.txt"
    if not human_path.exists():
        # Try FINAL_10k.txt as fallback (same content)
        human_path = AI2HUMAN_IN / "FINAL_10k.txt"
    if not human_path.exists():
        logger.warning("  ai2human human passages not found — skipping")
        return records

    with open(human_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by double-newlines (passages)
    passages = [p.strip() for p in content.split("\n\n") if p.strip()]

    for i, passage in enumerate(passages):
        if len(passage) < 50:  # Skip too-short passages
            continue
        records.append({
            "id": f"ai2human-{i}",
            "source_text": "",  # We don't have the AI counterpart
            "target_text": passage,
            "dataset": "ai2human",
            "domain": "student_essay",
        })

    logger.info(f"  TOTAL ai2human human passages: {len(records)}")
    return records


# ---------------------------------------------------------------------------
# 4. Combine, deduplicate, split
# ---------------------------------------------------------------------------

def split_data(records: list[dict], val_ratio: float, test_ratio: float):
    """Split into train/val/test while preserving dataset proportions."""
    # Group by dataset
    by_dataset = defaultdict(list)
    for r in records:
        by_dataset[r["dataset"]].append(r)

    train, val, test = [], [], []
    for ds_name, ds_records in by_dataset.items():
        random.shuffle(ds_records)
        n = len(ds_records)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_test - n_val

        if n_train <= 0:
            # Very small dataset: put most in train, 1 each in val/test
            n_test = min(1, n // 5)
            n_val = min(1, n // 5)
            n_train = n - n_test - n_val

        ds_test = ds_records[:n_test]
        ds_val = ds_records[n_test:n_test + n_val]
        ds_train = ds_records[n_test + n_val:]

        test.extend(ds_test)
        val.extend(ds_val)
        train.extend(ds_train)

        logger.info(f"  {ds_name}: {len(ds_train)} train, {len(ds_val)} val, {len(ds_test)} test")

    # Shuffle within each split
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def compute_stats(records: list[dict]) -> dict:
    """Compute basic dataset statistics."""
    if not records:
        return {}
    total_chars = sum(len(r.get("source_text", "")) + len(r.get("target_text", "")) for r in records)
    source_avg = sum(len(r.get("source_text", "")) for r in records) / len(records)
    target_avg = sum(len(r.get("target_text", "")) for r in records) / len(records)

    return {
        "total_records": len(records),
        "total_chars": total_chars,
        "avg_source_len_chars": round(source_avg, 1),
        "avg_target_len_chars": round(target_avg, 1),
        "datasets": list(set(r["dataset"] for r in records)),
        "domains": sorted(set(r.get("domain", "unknown") for r in records)),
    }


def write_jsonl(path: Path, records: list[dict]):
    """Write records to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"  Written: {path} ({len(records)} records, {path.stat().st_size / 1024 / 1024:.1f} MB)")


def write_stats(stats: dict, path: Path):
    """Write dataset statistics as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"  Written: {path}")


# ---------------------------------------------------------------------------
# 5. Simple inspection utility
# ---------------------------------------------------------------------------

def inspect_processed(records: list[dict] = None):
    """Print current state of processed data."""
    train_path = OUT_DIR / "train.jsonl"
    val_path = OUT_DIR / "val.jsonl"
    test_path = OUT_DIR / "test.jsonl"
    stats_path = OUT_DIR / "stats.json"

    if records is None:
        # Read from disk
        records = []
        for p in [train_path, val_path, test_path]:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    records.extend([json.loads(line) for line in f])

    if not records:
        print("\n❌ No processed data found. Run: python scripts/prepare_training_data.py")
        return

    print("=" * 60)
    print("📊 VoicePrint — Training Data Summary")
    print("=" * 60)

    ds_counts = defaultdict(int)
    for r in records:
        ds_counts[r.get("dataset", "unknown")] += 1

    print(f"\n📦 Total records: {len(records):,}")
    print(f"\n📁 By dataset:")
    for ds, cnt in sorted(ds_counts.items()):
        print(f"   {ds}: {cnt:,}")

    domains = set(r.get("domain", "unknown") for r in records if r.get("domain"))
    print(f"\n🏷️  Domains: {', '.join(sorted(domains))}")
    print(f"\n📐 Splits:")
    if train_path.exists():
        print(f"   Train: {len([r for r in records if r.get('_split', '') == 'train'])}")
    if val_path.exists():
        print(f"   Val:   {len([r for r in records if r.get('_split', '') == 'val'])}")
    if test_path.exists():
        print(f"   Test:  {len([r for r in records if r.get('_split', '') == 'test'])}")

    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        print(f"\n📈 Stats:")
        print(f"   Avg source length: {stats.get('avg_source_len_chars', '?')} chars")
        print(f"   Avg target length: {stats.get('avg_target_len_chars', '?')} chars")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare VoicePrint training data")
    parser.add_argument("--skip-hc3", action="store_true", help="Skip HC3 processing")
    parser.add_argument("--skip-cmu", action="store_true", help="Skip CMU corpus processing")
    parser.add_argument("--skip-ai2human", action="store_true", help="Skip ai2human processing")
    parser.add_argument("--inspect", action="store_true", help="Show current data state")
    parser.add_argument("--force", action="store_true", help="Re-process even if output exists")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO, help="Validation ratio")
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO, help="Test ratio")
    args = parser.parse_args()

    if args.inspect:
        inspect_processed()
        return

    random.seed(SEED)
    t_start = time.time()

    # -----------------------------------------------------------------------
    # Process each dataset
    # -----------------------------------------------------------------------
    all_records = []

    if not args.skip_hc3:
        print("\n" + "=" * 60)
        print("📦 Processing HC3")
        print("=" * 60)
        hc3_records = process_hc3()
        all_records.extend(hc3_records)

    if not args.skip_cmu:
        print("\n" + "=" * 60)
        print("📦 Processing CMU Human-AI Parallel Corpus")
        print("=" * 60)
        cmu_records = process_cmu()
        all_records.extend(cmu_records)

    if not args.skip_ai2human:
        print("\n" + "=" * 60)
        print("📦 Processing ai2human (human references)")
        print("=" * 60)
        ai2h_records = process_ai2human()
        all_records.extend(ai2h_records)

    if not all_records:
        logger.error("No records produced. Check that datasets are downloaded.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Split
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📐 Splitting into train/val/test")
    print("=" * 60)
    train, val, test = split_data(all_records, args.val_ratio, args.test_ratio)

    # Mark splits
    for r in train:
        r["_split"] = "train"
    for r in val:
        r["_split"] = "val"
    for r in test:
        r["_split"] = "test"

    # -----------------------------------------------------------------------
    # Write
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("💾 Writing output files")
    print("=" * 60)
    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "val.jsonl", val)
    write_jsonl(OUT_DIR / "test.jsonl", test)

    # Write stats
    stats = compute_stats(train + val + test)
    stats["n_train"] = len(train)
    stats["n_val"] = len(val)
    stats["n_test"] = len(test)
    stats["splits"] = {
        "train": {"size": len(train), "datasets": list(set(r["dataset"] for r in train))},
        "val": {"size": len(val), "datasets": list(set(r["dataset"] for r in val))},
        "test": {"size": len(test), "datasets": list(set(r["dataset"] for r in test))},
    }
    write_stats(stats, OUT_DIR / "stats.json")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("✅ Data Preparation Complete")
    print("=" * 60)
    print(f"  Total pairs: {len(all_records):,}")
    print(f"  Train: {len(train):,}  |  Val: {len(val):,}  |  Test: {len(test):,}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"  Output: {OUT_DIR.resolve()}")
    print()
    print("  🔗 To load in training:")
    print(f'     from datasets import load_dataset')
    print(f'     ds = load_dataset("json", data_files={{"train": "data/processed/train.jsonl",')
    print(f'                                            "val": "data/processed/val.jsonl",')
    print(f'                                            "test": "data/processed/test.jsonl"}})')
    print()


if __name__ == "__main__":
    main()
