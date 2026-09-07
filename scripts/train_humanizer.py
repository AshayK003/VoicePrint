"""
VoicePrint — Phase 2 Training: Fine-tune a seq2seq model for AI→Human style transfer.

Trains FLAN-T5 (or any HuggingFace seq2seq model) on parallel AI→Human text pairs.

Usage:
    # Recommended: train on Colab (GPU required for practical speed)
    # See scripts/VoicePrint_Qwen_3.5_4B_Training.ipynb

    # Local CPU training (very slow — expect 24+ hours for FLAN-T5-small)
    python scripts/train_humanizer.py --model google/flan-t5-small --epochs 3 --batch-size 4

    # Full quality on GPU
    python scripts/train_humanizer.py --model google/flan-t5-base --epochs 5 --batch-size 8

    # Resume from checkpoint
    python scripts/train_humanizer.py --resume models/humanizer/checkpoint-1000

    # Evaluate only
    python scripts/train_humanizer.py --evaluate-only models/humanizer/final
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "humanizer"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VoicePrint humanizer model")

    # Data
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR),
                        help="Path to processed data dir with train.jsonl / val.jsonl / test.jsonl")
    parser.add_argument("--max-source-len", type=int, default=512,
                        help="Max source sequence length (tokens)")
    parser.add_argument("--max-target-len", type=int, default=512,
                        help="Max target sequence length (tokens)")

    # Model
    parser.add_argument("--model", type=str, default="google/flan-t5-small",
                        help="HuggingFace model ID or path")
    parser.add_argument("--output-dir", type=str, default=str(MODELS_DIR),
                        help="Where to save trained model and checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")

    # Training
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=500,
                        help="Warmup steps for scheduler")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing factor")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay for optimizer")
    parser.add_argument("--save-steps", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval-steps", type=int, default=500,
                        help="Evaluate every N steps")
    parser.add_argument("--logging-steps", type=int, default=100,
                        help="Log every N steps")

    # Evaluation only
    parser.add_argument("--evaluate-only", type=str, default=None,
                        help="Path to model checkpoint for evaluation only")

    # Misc
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Use mixed precision (fp16) — needs GPU")
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="Use bfloat16 — needs Ampere+ GPU")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0 = main process)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file, return list of records."""
    if not path.exists():
        logger.error(f"File not found: {path}")
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    logger.info(f"  Loaded {len(records):,} records from {path.name}")
    return records


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_rouge(references: list[str], predictions: list[str]) -> dict:
    """Compute ROUGE-1/2/L scores. Falls back to rough character-level if rouge_score not installed."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for ref, pred in zip(references, predictions):
            if not ref or not pred:
                continue
            result = scorer.score(ref, pred)
            for key in scores:
                scores[key].append(result[key].fmeasure)
        return {
            k: (sum(v) / len(v) if v else 0.0) * 100
            for k, v in scores.items()
        }
    except ImportError:
        logger.warning("  rouge_score not installed — using character-level fallback")
        # Simple character n-gram overlap as rough proxy
        total_f1 = []
        for ref, pred in zip(references, predictions):
            if not ref or not pred:
                continue
            # character 3-gram overlap
            ref_ngrams = set(ref[i:i+3] for i in range(len(ref)-2))
            pred_ngrams = set(pred[i:i+3] for i in range(len(pred)-2))
            if not ref_ngrams or not pred_ngrams:
                continue
            overlap = ref_ngrams & pred_ngrams
            precision = len(overlap) / len(pred_ngrams) if pred_ngrams else 0
            recall = len(overlap) / len(ref_ngrams) if ref_ngrams else 0
            if precision + recall > 0:
                total_f1.append(2 * precision * recall / (precision + recall))
        return {"char_ngram_f1": (sum(total_f1) / len(total_f1) * 100) if total_f1 else 0.0}


def compute_readability(text: str) -> dict:
    """Compute readability metrics using textstat. Gracefully handles missing package."""
    try:
        import textstat
        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            "avg_sentence_length": textstat.avg_sentence_length(text),
            "syllable_count": textstat.syllable_count(text),
        }
    except ImportError:
        return {}


def compute_style_metrics(text: str) -> dict:
    """Compute linguistic markers that separate AI from human text."""
    import re
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    contractions = len(re.findall(r"\b\w+'\w+\b", text))  # don't, it's, etc.
    avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
    lexical_diversity = len(set(w.lower() for w in words)) / len(words) if words else 0
    question_marks = text.count("?")
    exclamations = text.count("!")
    commas = text.count(",")
    sentence_len_var = 0
    if len(sentences) > 1:
        sent_lens = [len(s.split()) for s in sentences]
        avg_sl = sum(sent_lens) / len(sent_lens)
        sentence_len_var = sum((sl - avg_sl) ** 2 for sl in sent_lens) / len(sent_lens)

    return {
        "contractions": contractions,
        "avg_word_length": round(avg_word_len, 2),
        "lexical_diversity": round(lexical_diversity, 3),
        "question_marks": question_marks,
        "exclamations": exclamations,
        "commas": commas,
        "sentence_len_variance": round(sentence_len_var, 2),
        "n_sentences": len(sentences),
        "n_words": len(words),
    }


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Suppress TF warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # -------------------------------------------------------------------
    # Load datasets
    # -------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("📦 Loading datasets")
    logger.info("=" * 60)

    train_records = load_jsonl(data_dir / "train.jsonl")
    val_records = load_jsonl(data_dir / "val.jsonl")
    test_records = load_jsonl(data_dir / "test.jsonl")

    if not train_records and not args.evaluate_only:
        logger.error("No training data found. Run prepare_training_data.py first.")
        sys.exit(1)

    # Only evaluation mode
    if args.evaluate_only:
        logger.info(f"\n🔍 Evaluation-only mode: loading model from {args.evaluate_only}")
        _run_evaluation(args.evaluate_only, test_records or val_records, args)
        return

    # -------------------------------------------------------------------
    # Load model and tokenizer
    # -------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"🧠 Loading model: {args.model}")
    logger.info("=" * 60)

    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
        get_scheduler,
    )
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"  Device: {device.upper()}")

    # Load tokenizer & model
    model_path = args.resume or args.model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if args.bf16 and device == "cuda"
                else torch.float16 if args.fp16 and device == "cuda"
                else torch.float32,
    )

    # Log model size
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Parameters: {n_params:,} total, {n_trainable:,} trainable")

    # -------------------------------------------------------------------
    # Tokenize datasets
    # -------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("✂️  Tokenizing datasets")
    logger.info("=" * 60)

    from datasets import Dataset

    def tokenize_fn(batch):
        sources = batch["source_text"]
        targets = batch["target_text"]

        model_inputs = tokenizer(
            sources,
            max_length=args.max_source_len,
            truncation=True,
            padding=False,
        )

        labels = tokenizer(
            targets,
            max_length=args.max_target_len,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def make_dataset(records: list[dict]) -> Dataset:
        ds = Dataset.from_list(records)
        ds = ds.map(
            tokenize_fn,
            batched=True,
            remove_columns=ds.column_names,
            desc="Tokenizing",
        )
        return ds

    if train_records:
        train_dataset = make_dataset(train_records)
        logger.info(f"  Train: {len(train_dataset):,} examples")
    else:
        train_dataset = None

    if val_records:
        val_dataset = make_dataset(val_records)
        logger.info(f"  Val:   {len(val_dataset):,} examples")
    else:
        val_dataset = None

    # -------------------------------------------------------------------
    # Data collator
    # -------------------------------------------------------------------
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=tokenizer.pad_token_id,
    )

    # -------------------------------------------------------------------
    # Training arguments
    # -------------------------------------------------------------------
    from transformers import EarlyStoppingCallback

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        generation_max_length=args.max_target_len,
        generation_num_beams=4,
        fp16=args.fp16 and device == "cuda",
        bf16=args.bf16 and device == "cuda",
        dataloader_num_workers=args.num_workers,
        report_to=["tensorboard"] if device == "cuda" else [],
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        gradient_checkpointing=True if device == "cuda" else False,
        ddp_find_unused_parameters=False,
    )

    # -------------------------------------------------------------------
    # Compute metrics
    # -------------------------------------------------------------------
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds

        # Decode predictions and labels
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_preds = tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        # Replace -100 in labels
        labels = [[l for l in lab if l != -100] for lab in labels]
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )

        # Remove empty labels/predictions
        pairs = [(p.strip(), l.strip()) for p, l in zip(decoded_preds, decoded_labels)
                 if p.strip() and l.strip()]
        if not pairs:
            return {"loss": 0.0}
        preds_clean, labels_clean = zip(*pairs)

        # ROUGE
        rouge = compute_rouge(labels_clean, preds_clean)

        # Style metrics on predictions vs references
        avg_metrics = {}
        style_preds = [compute_style_metrics(p) for p in preds_clean]
        style_labels = [compute_style_metrics(l) for l in labels_clean]

        for key in ["contractions", "lexical_diversity", "question_marks",
                     "exclamations", "avg_word_length", "sentence_len_variance"]:
            pred_mean = sum(m[key] for m in style_preds) / len(style_preds)
            label_mean = sum(m[key] for m in style_labels) / len(style_labels)
            avg_metrics[f"pred_{key}"] = round(pred_mean, 3)
            avg_metrics[f"ref_{key}"] = round(label_mean, 3)
            # How close prediction is to human reference
            avg_metrics[f"gap_{key}"] = round(abs(pred_mean - label_mean), 3)

        return {**rouge, **avg_metrics}

    # -------------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------------
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if val_dataset else None,
    )

    # -------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("🚀 Training")
    logger.info("=" * 60)

    t0 = time.time()

    if args.resume:
        logger.info(f"  Resuming from checkpoint: {args.resume}")
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    train_time = time.time() - t0
    logger.info(f"  Training completed in {train_time:.0f}s ({train_time/60:.1f} min)")

    # -------------------------------------------------------------------
    # Save final
    # -------------------------------------------------------------------
    final_dir = output_dir / "final"
    logger.info(f"  Saving final model to {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save training args
    with open(final_dir / "training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    logger.info(f"  ✅ Model saved to {final_dir}")

    # -------------------------------------------------------------------
    # Evaluate on test set
    # -------------------------------------------------------------------
    if test_records:
        _run_evaluation(str(final_dir), test_records, args)

    logger.info("\n" + "=" * 60)
    logger.info("✅ Training complete!")
    logger.info("=" * 60)
    logger.info(f"  Model: {final_dir}")
    logger.info(f"  To use in paraphrase pipeline:")
    logger.info(f'    from voiceprint.humanizer_model import HumanizerModel')
    logger.info(f'    h = HumanizerModel("{final_dir}")')
    logger.info(f'    result = h.humanize("Your AI-generated text here")')


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _run_evaluation(model_path: str, test_records: list[dict], args: argparse.Namespace):
    """Run evaluation on test set."""
    logger.info("\n" + "=" * 60)
    logger.info("📊 Evaluation on test set")
    logger.info("=" * 60)

    if not test_records:
        logger.warning("  No test records to evaluate on.")
        return

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"  Device: {device.upper()}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    # Sample a subset for evaluation (to keep it fast)
    eval_sample = test_records[:min(len(test_records), 200)]
    logger.info(f"  Evaluating on {len(eval_sample)} examples (sampled from {len(test_records):,})")

    predictions = []
    references = []
    timer = 0.0

    for i, rec in enumerate(eval_sample):
        source = rec["source_text"]
        target = rec["target_text"]

        if not source.strip() or not target.strip():
            continue

        t0 = time.time()
        inputs = tokenizer(
            source,
            max_length=args.max_source_len,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=args.max_target_len,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        timer += time.time() - t0

        predictions.append(pred)
        references.append(target)

        if (i + 1) % 50 == 0:
            logger.info(f"    [{i+1}/{len(eval_sample)}] — avg {timer/(i+1):.2f}s per example")

    # Metrics
    logger.info("\n  --- Metrics ---")
    rouge = compute_rouge(references, predictions)
    for k, v in rouge.items():
        logger.info(f"  {k}: {v:.2f}")

    # Style shift analysis
    logger.info("\n  --- Style Shift (avg per passage) ---")
    ai_style = compute_style_metrics(" ".join(predictions) if predictions else "")
    human_style = compute_style_metrics(" ".join(references) if references else "")
    for key in ["contractions", "avg_word_length", "lexical_diversity",
                "question_marks", "exclamations", "sentence_len_variance",
                "n_sentences", "n_words"]:
        ai_val = ai_style.get(key, 0)
        human_val = human_style.get(key, 0)
        gap = abs(ai_val - human_val)
        logger.info(f"  {key:25s}  pred={ai_val:<10}  ref={human_val:<10}  gap={gap:.2f}")

    # Readability
    logger.info("\n  --- Readability ---")
    pred_read = compute_readability(" ".join(predictions)) if predictions else {}
    ref_read = compute_readability(" ".join(references)) if references else {}
    for key in pred_read:
        pv = pred_read.get(key, 0)
        rv = ref_read.get(key, 0)
        logger.info(f"  {key:25s}  pred={pv:.2f}  ref={rv:.2f}")

    avg_time = timer / len(eval_sample) if eval_sample else 0
    logger.info(f"\n  Avg inference time: {avg_time:.2f}s per example")

    # Save evaluation results
    out_dir = Path(model_path)
    eval_path = out_dir / "eval_results.json"
    with open(eval_path, "w") as f:
        json.dump({
            "rouge": rouge,
            "style_shift": {
                "predictions": ai_style,
                "references": human_style,
            },
            "readability": {"predictions": pred_read, "references": ref_read},
            "avg_inference_time_s": avg_time,
            "n_evaluated": len(eval_sample),
        }, f, indent=2)
    logger.info(f"\n  Results saved to {eval_path}")

    # Show 3 examples
    logger.info("\n  --- Sample Outputs ---")
    for i in range(min(3, len(predictions))):
        logger.info(f"\n  Example {i+1}:")
        logger.info(f"    AI Input:    {eval_sample[i]['source_text'][:120]}...")
        logger.info(f"    Human Ref:   {references[i][:120]}...")
        logger.info(f"    Prediction:  {predictions[i][:120]}...")


if __name__ == "__main__":
    main()
