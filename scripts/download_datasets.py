"""
VoicePrint — Dataset Downloader
================================
Downloads and prepares datasets for training the AI-to-Human style transfer model.

Datasets:
  1. HC3 (Human ChatGPT Comparison Corpus) — Hello-SimpleAI/HC3
     Question → {human_answers, chatgpt_answers} pairs
  2. CMU Human-AI Parallel Corpus — browndw/human-ai-parallel-corpus
     Human-written text vs GPT-4o/Llama 3 continuations (parquet)
  3. ai2human-style-transfer — Bansnetsajak007/ai2human-style-transfer (GitHub)
     Paired AI→Human passages (10K pairs)

Usage:
    python scripts/download_datasets.py            # Download all
    python scripts/download_datasets.py --hc3      # Only HC3
    python scripts/download_datasets.py --cmu      # Only CMU corpus
    python scripts/download_datasets.py --list     # List what's already downloaded

Output layout:
    data/
    ├── raw/                  # Existing human-written articles
    ├── processed/            # Processed/tokenized outputs (populated by prepare script)
    └── training/
        ├── hc3/              # HC3 JSONL files
        ├── human-ai-parallel/ # CMU corpus parquet files
        └── ai2human/         # ai2human-style-transfer data
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Project root (two levels up from scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# 1. HC3 — Hello-SimpleAI/HC3
# ---------------------------------------------------------------------------

HC3_REPO = "Hello-SimpleAI/HC3"
HC3_FILES = [
    "all.jsonl",
    "finance.jsonl",
    "medicine.jsonl",
    "open_qa.jsonl",
    "reddit_eli5.jsonl",
    "wiki_csai.jsonl",
]
HC3_OUT = DATA_DIR / "training" / "hc3"

def download_hc3(force: bool = False) -> bool:
    """Download HC3 JSONL files from HuggingFace Hub using hf_hub_download."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return False

    HC3_OUT.mkdir(parents=True, exist_ok=True)
    success = True

    for fname in HC3_FILES:
        dest = HC3_OUT / fname
        if dest.exists() and not force:
            logger.info(f"  SKIP {fname} — already exists ({dest.stat().st_size / 1024:.0f} KB)")
            continue

        logger.info(f"  Downloading {fname} ...")
        try:
            cached = hf_hub_download(HC3_REPO, fname, repo_type="dataset")
            shutil.copy2(cached, dest)
            logger.info(f"    → {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        except Exception as e:
            logger.error(f"    FAILED: {e}")
            success = False

    return success


def inspect_hc3() -> dict:
    """Return metadata about downloaded HC3 files."""
    result = {}
    for fname in HC3_FILES:
        path = HC3_OUT / fname
        if not path.exists():
            result[fname] = None
            continue
        lines = 0
        human_count = 0
        chatgpt_count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                lines += 1
                if lines <= 1000:
                    d = json.loads(line)
                    human_count += len(d.get("human_answers", []))
                    chatgpt_count += len(d.get("chatgpt_answers", []))
        # Estimate totals from sample
        if lines > 0:
            ratio = max(1, lines / 1000) if lines > 1000 else 1
            est_human = int(human_count * ratio)
            est_chatgpt = int(chatgpt_count * ratio)
        else:
            est_human = est_chatgpt = 0
        result[fname] = {
            "records": lines,
            "est_human_answers": est_human,
            "est_chatgpt_answers": est_chatgpt,
            "size_kb": path.stat().st_size / 1024,
        }
    return result


# ---------------------------------------------------------------------------
# 2. CMU Human-AI Parallel Corpus — browndw/human-ai-parallel-corpus
# ---------------------------------------------------------------------------

CMU_REPO = "browndw/human-ai-parallel-corpus"
CMU_FILES = [
    "text_data/hape-text_human-chunk-1.parquet",
    "text_data/hape-text_human-chunk-2.parquet",
    "text_data/hape-text_gpt-4o-2024-08-06.parquet",
    "text_data/hape-text_gpt-4o-mini-2024-07-18.parquet",
    "text_data/hape-text_llama-3-8B.parquet",
    "text_data/hape-text_llama-3-8B-Instruct.parquet",
    "text_data/hape-text_llama-3-70B.parquet",
    "text_data/hape-text_llama-3-70B-Instruct.parquet",
]
CMU_OUT = DATA_DIR / "training" / "human-ai-parallel"

def download_cmu(force: bool = False) -> bool:
    """Download CMU Human-AI Parallel parquet files."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return False

    CMU_OUT.mkdir(parents=True, exist_ok=True)
    success = True

    for fname in CMU_FILES:
        dest = CMU_OUT / Path(fname).name
        if dest.exists() and not force:
            logger.info(f"  SKIP {Path(fname).name} — already exists ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
            continue

        logger.info(f"  Downloading {fname} ...")
        try:
            cached = hf_hub_download(CMU_REPO, fname, repo_type="dataset")
            shutil.copy2(cached, dest)
            logger.info(f"    → {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        except Exception as e:
            logger.error(f"    FAILED: {e}")
            success = False

    return success


def inspect_cmu() -> dict:
    """Return metadata about downloaded CMU parquet files."""
    result = {}
    for fname in CMU_FILES:
        basename = Path(fname).name
        path = CMU_OUT / basename
        if not path.exists():
            result[basename] = None
            continue
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            schema_names = pf.schema.names
            ncols = len(schema_names)
            result[basename] = {
                "rows": pf.metadata.num_rows,
                "columns": ncols,
                "size_mb": path.stat().st_size / 1024 / 1024,
                "schema": schema_names,
            }
        except ImportError:
            result[basename] = {
                "size_mb": path.stat().st_size / 1024 / 1024,
            }
    return result


# ---------------------------------------------------------------------------
# 3. ai2human-style-transfer — from GitHub
# ---------------------------------------------------------------------------

AI2HUMAN_REPO = "https://github.com/Bansnetsajak007/ai2human-style-transfer.git"
AI2HUMAN_OUT = DATA_DIR / "training" / "ai2human"

def download_ai2human(force: bool = False) -> bool:
    """Clone (shallow) the ai2human-style-transfer repo to get the data/ folder."""
    if AI2HUMAN_OUT.exists() and not force:
        logger.info(f"  ai2human directory already exists at {AI2HUMAN_OUT}")
        return True

    AI2HUMAN_OUT.mkdir(parents=True, exist_ok=True)

    # Use sparse checkout to only get the data/ directory
    tmp_dir = PROJECT_ROOT / "tmp_ai2human_clone"
    try:
        logger.info("  Cloning ai2human-style-transfer repo (shallow, data only) ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             AI2HUMAN_REPO, str(tmp_dir)],
            check=True, capture_output=True, text=True, timeout=120,
        )
        subprocess.run(
            ["git", "-C", str(tmp_dir), "sparse-checkout", "set", "data"],
            check=True, capture_output=True, text=True, timeout=30,
        )
        subprocess.run(
            ["git", "-C", str(tmp_dir), "checkout"],
            check=True, capture_output=True, text=True, timeout=30,
        )

        # Copy data/ contents
        src_data = tmp_dir / "data"
        if src_data.exists():
            for item in src_data.iterdir():
                dest = AI2HUMAN_OUT / item.name
                if item.is_file():
                    shutil.copy2(item, dest)
                elif item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)

        # Also copy docs/ for the paper
        src_docs = tmp_dir / "docs"
        if src_docs.exists():
            docs_out = AI2HUMAN_OUT / "docs"
            docs_out.mkdir(exist_ok=True)
            for item in src_docs.iterdir():
                dest = docs_out / item.name
                if item.is_file():
                    shutil.copy2(item, dest)

        logger.info(f"    → Copied to {AI2HUMAN_OUT}")
        return True

    except subprocess.TimeoutExpired:
        logger.error("    TIMEOUT: Git clone took too long (network may be slow)")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"    Git error: {e.stderr[:200]}")
        return False
    except Exception as e:
        logger.error(f"    FAILED: {e}")
        return False
    finally:
        # Clean up temp directory
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def inspect_ai2human() -> dict:
    """Return metadata about downloaded ai2human files."""
    result = {"files": [], "total_size_kb": 0}
    if not AI2HUMAN_OUT.exists():
        return None
    for item in sorted(AI2HUMAN_OUT.rglob("*")):
        if item.is_file() and item.name != ".gitkeep":
            result["files"].append({
                "name": str(item.relative_to(AI2HUMAN_OUT)),
                "size_kb": item.stat().st_size / 1024,
            })
            result["total_size_kb"] += item.stat().st_size / 1024
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def list_datasets():
    """Print what's currently downloaded."""
    print("=" * 60)
    print("VoicePrint — Dataset Inventory")
    print("=" * 60)

    print("\n📦 HC3 (Hello-SimpleAI/HC3):")
    hc3_stats = inspect_hc3()
    for fname, info in hc3_stats.items():
        if info is None:
            print(f"  ❌ {fname} — not downloaded")
        else:
            print(f"  ✅ {fname}: {info['records']:,} records, "
                  f"~{info['est_human_answers']:,} human answers, "
                  f"~{info['est_chatgpt_answers']:,} ChatGPT answers, "
                  f"{info['size_kb']/1024:.1f} MB")

    print("\n📦 CMU Human-AI Parallel Corpus:")
    cmu_stats = inspect_cmu()
    for fname, info in cmu_stats.items():
        if info is None:
            print(f"  ❌ {fname} — not downloaded")
        else:
            rows = info.get("rows", "?")
            cols = info.get("columns", "?")
            schema = info.get("schema", [])
            print(f"  ✅ {fname}: {rows:,} rows × {cols} cols, "
                  f"{info['size_mb']:.1f} MB"
                  f"{' — schema: ' + str(schema) if schema else ''}")

    print("\n📦 ai2human-style-transfer:")
    ai2h = inspect_ai2human()
    if ai2h is None:
        print("  ❌ Not downloaded")
    else:
        print(f"  ✅ {ai2h['total_size_kb']/1024:.1f} MB — {len(ai2h['files'])} files")
        for f in ai2h["files"][:10]:
            print(f"     📄 {f['name']} ({f['size_kb']:.1f} KB)")
        if len(ai2h["files"]) > 10:
            print(f"     ... and {len(ai2h['files'])-10} more files")


def main():
    parser = argparse.ArgumentParser(description="VoicePrint Dataset Downloader")
    parser.add_argument("--hc3", action="store_true", help="Download only HC3")
    parser.add_argument("--cmu", action="store_true", help="Download only CMU corpus")
    parser.add_argument("--ai2human", action="store_true", help="Download only ai2human data")
    parser.add_argument("--list", action="store_true", help="Show downloaded datasets")
    parser.add_argument("--force", action="store_true", help="Re-download even if exists")
    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    # Determine which datasets to download
    all_datasets = not (args.hc3 or args.cmu or args.ai2human)

    if all_datasets or args.hc3:
        print("\n" + "=" * 60)
        print("📦 Downloading HC3 (Human ChatGPT Comparison Corpus)")
        print("=" * 60)
        t0 = time.time()
        ok = download_hc3(force=args.force)
        dt = time.time() - t0
        logger.info(f"HC3 download {'OK' if ok else 'PARTIAL'} in {dt:.0f}s")
        hc3_stats = inspect_hc3()
        total_records = sum(v["records"] for v in hc3_stats.values() if v)
        total_human = sum(v["est_human_answers"] for v in hc3_stats.values() if v)
        total_chatgpt = sum(v["est_chatgpt_answers"] for v in hc3_stats.values() if v)
        logger.info(f"  Total: {total_records:,} questions, "
                    f"~{total_human:,} human answers, ~{total_chatgpt:,} ChatGPT answers")

    if all_datasets or args.cmu:
        print("\n" + "=" * 60)
        print("📦 Downloading CMU Human-AI Parallel Corpus")
        print("=" * 60)
        t0 = time.time()
        ok = download_cmu(force=args.force)
        dt = time.time() - t0
        logger.info(f"CMU download {'OK' if ok else 'PARTIAL'} in {dt:.0f}s")

    if all_datasets or args.ai2human:
        print("\n" + "=" * 60)
        print("📦 Downloading ai2human-style-transfer data")
        print("=" * 60)
        t0 = time.time()
        ok = download_ai2human(force=args.force)
        dt = time.time() - t0
        logger.info(f"ai2human download {'OK' if ok else 'PARTIAL'} in {dt:.0f}s")

    print("\n" + "=" * 60)
    print("📋 Final Dataset Inventory")
    print("=" * 60)
    list_datasets()


if __name__ == "__main__":
    main()
