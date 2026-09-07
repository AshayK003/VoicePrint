"""
Generate synthetic AI→Human training pairs from the raw Medium articles.

Strategy: for each human-written Medium article, split into paragraphs,
then use any available LLM API to generate an "AI version" that sounds
like GPT wrote it. Pair these (AI → human) as additional training data.

This gives us style-specific pairs matching the user's actual writing
voice, complementing the generic HC3/CMU corpora.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
SYNTH_OUT = OUT_DIR / "synthetic_pairs.jsonl"


def load_articles(raw_dir: Path = RAW_DIR) -> list[dict]:
    """Load all medium articles. Returns [{title, body_paragraphs, path}, ...]"""
    articles = []
    for fpath in sorted(raw_dir.glob("medium*.txt")):
        text = fpath.read_text(encoding="utf-8").strip()
        if not text:
            continue

        lines = text.split("\n")
        title = lines[0].strip() if lines else "Untitled"
        # Combine remaining lines into paragraphs (split on double newline)
        body = "\n".join(lines[1:]).strip()
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]

        articles.append({
            "title": title,
            "source": fpath.name,
            "paragraphs": paragraphs,
            "body": body,
        })
        logger.info(f"  {fpath.name}: {len(paragraphs)} paragraphs, {len(body.split())} words")
    return articles


def chunk_paragraphs(
    articles: list[dict],
    min_chars: int = 200,
    max_chars: int = 2000,
) -> tuple[list[str], list[str]]:
    """Split articles into training-length chunks.

    Returns (chunks, metadata) where metadata has title and source for each chunk.
    """
    chunks = []
    meta = []
    for art in articles:
        # Build chunks from paragraphs (combine small adjacent paragraphs)
        current = ""
        for para in art["paragraphs"]:
            # Skip very short fragments
            if len(para) < min_chars * 0.3:
                if current:
                    current += "\n\n" + para
                continue

            if len(current) + len(para) > max_chars and current:
                chunks.append(current.strip())
                meta.append({"title": art["title"], "source": art["source"]})
                current = para
            else:
                current = (current + "\n\n" + para) if current else para

        if current:
            chunks.append(current.strip())
            meta.append({"title": art["title"], "source": art["source"]})

    return chunks, meta


def main():
    logger.info("Loading Medium articles...")
    articles = load_articles()
    logger.info(f"Loaded {len(articles)} articles")

    chunks, meta = chunk_paragraphs(articles)
    logger.info(f"Created {len(chunks)} chunks from articles")

    # Write as synthetic pairs with placeholder source_text
    # The Colab notebook will fill source_text (AI version) by running
    # the untrained base model to generate "AI-sounding" versions.
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for i, (chunk, m) in enumerate(zip(chunks, meta)):
        records.append({
            "id": f"synth-{i:04d}",
            "source_text": "",  # filled by Colab notebook
            "target_text": chunk,
            "dataset": "synthetic-medium",
            "domain": "blog_article",
            "metadata": {
                "title": m["title"],
                "source": m["source"],
                "target_is_human": True,
            },
        })

    # Write
    with open(SYNTH_OUT, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(records)} synthetic pair templates to {SYNTH_OUT}")
    logger.info("NOTE: source_text is empty — Colab notebook fills it via base model inference")
    logger.info("")
    logger.info("Article stats:")
    for art in articles:
        char_len = len(art["body"])
        word_len = len(art["body"].split())
        logger.info(f"  {art['source']:30s} {char_len:6d} chars  {word_len:4d} words  {len(art['paragraphs']):2d} paragraphs")


if __name__ == "__main__":
    main()
