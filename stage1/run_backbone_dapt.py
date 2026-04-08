"""
run_backbone_dapt.py
--------------------
OPTIONAL. Run this LAST, only after baseline pipeline is stable.

Domain-Adaptive Pretraining (DAPT) via Masked Language Modeling (MLM).
Produces a domain-adapted backbone that can be used as a drop-in replacement
in any of the training scripts.

Priority note: If time is short, use ProsusAI/finbert directly as the
"financial pretrained backbone" reference. This script is for producing
a custom DAPT checkpoint using YOUR corpus.

Usage:
    python run_backbone_dapt.py \
        --backbone bert-base-uncased \
        --corpus_dir data/financial_corpus/ \
        --output_dir checkpoints/dapt_bert_base/ \
        --max_steps 10000

Financial corpus sources (collect at least one):
    - FPB / SEntFiN training text (already available)
    - Public financial news (Reuters, SEC filings)
    - Target: 50M-500M tokens for meaningful DAPT
    - Minimum viable: 10M tokens (will show modest improvement)
"""

import argparse
import json
import os
from pathlib import Path
from typing import List

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset as HFDataset


# ------------------------------------------------------------------
# Corpus loading
# ------------------------------------------------------------------

def load_corpus_texts(corpus_dir: str, max_files: int = None) -> List[str]:
    """
    Load .txt files from corpus_dir. Each file = one document.
    Returns list of document strings.
    """
    corpus_path = Path(corpus_dir)
    files = sorted(corpus_path.glob("*.txt"))
    if max_files:
        files = files[:max_files]

    texts = []
    for fp in files:
        with open(fp, encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
            if content:
                texts.append(content)

    print(f"[OK] Loaded {len(texts)} documents from {corpus_dir}")
    return texts


def chunk_texts(texts: List[str], tokenizer, chunk_size: int = 512) -> List[str]:
    """
    Chunk documents into fixed-size token windows.
    Simple approach: split by whitespace into words, then regroup.
    """
    all_tokens = []
    for doc in texts:
        tokens = tokenizer.tokenize(doc)
        all_tokens.extend(tokens)

    chunks = []
    for i in range(0, len(all_tokens), chunk_size):
        chunk_tokens = all_tokens[i:i + chunk_size]
        if len(chunk_tokens) < chunk_size // 2:
            continue  # Skip very short trailing chunks
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)

    print(f"[OK] Created {len(chunks)} text chunks (chunk_size={chunk_size})")
    return chunks


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

def build_mlm_dataset(chunks: List[str], tokenizer, max_length: int) -> HFDataset:
    hf_dataset = HFDataset.from_dict({"text": chunks})

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

    tokenized = hf_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    return tokenized


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load tokenizer and model
    print(f"[INFO] Loading backbone: {args.backbone}")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    model     = AutoModelForMaskedLM.from_pretrained(args.backbone)

    # 2. Load and chunk corpus
    texts  = load_corpus_texts(args.corpus_dir, max_files=args.max_files)
    chunks = chunk_texts(texts, tokenizer, chunk_size=args.chunk_size)

    if len(chunks) < 1000:
        print(
            f"[WARN] Only {len(chunks)} chunks available. "
            f"DAPT requires substantial text to be meaningful. "
            f"Consider using ProsusAI/finbert directly instead."
        )

    # 3. Split 95/5 for train/eval
    n_eval  = max(100, len(chunks) // 20)
    train_chunks = chunks[:-n_eval]
    eval_chunks  = chunks[-n_eval:]

    train_dataset = build_mlm_dataset(train_chunks, tokenizer, args.max_length)
    eval_dataset  = build_mlm_dataset(eval_chunks,  tokenizer, args.max_length)

    # 4. Data collator for MLM (15% masking by default)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    # 5. Training
    training_args = TrainingArguments(
        output_dir              = str(output_dir / "trainer_output"),
        max_steps               = args.max_steps,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        learning_rate           = args.learning_rate,
        warmup_ratio            = 0.05,
        weight_decay            = 0.01,
        evaluation_strategy     = "steps",
        eval_steps              = args.eval_steps,
        save_steps              = args.eval_steps,
        logging_steps           = 100,
        save_total_limit        = 2,
        fp16                    = args.fp16,
        dataloader_num_workers  = 2,
        seed                    = args.seed,
        report_to               = "none",
    )

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_dataset,
        eval_dataset  = eval_dataset,
        data_collator = data_collator,
    )

    print(f"\n[INFO] Starting DAPT for {args.max_steps} steps...")
    trainer.train()

    # 6. Save final backbone (encoder only, no MLM head)
    final_model_dir = output_dir / "backbone"
    model.save_pretrained(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    print(f"[OK] DAPT backbone saved to {final_model_dir}")

    # 7. Save version metadata
    meta = {
        "base_backbone":    args.backbone,
        "dapt_output":      str(final_model_dir),
        "corpus_dir":       args.corpus_dir,
        "n_chunks":         len(chunks),
        "max_steps":        args.max_steps,
        "chunk_size":       args.chunk_size,
        "mlm_probability":  0.15,
        "seed":             args.seed,
        "usage": (
            "Use this path as --backbone in train_baseline_*.py "
            "to compare DAPT vs non-DAPT performance."
        ),
    }
    with open(output_dir / "dapt_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Metadata written to {output_dir / 'dapt_metadata.json'}")
    print(
        f"\n[NEXT] To use this backbone:\n"
        f"  python train_baseline_target.py \\\n"
        f"      --backbone {final_model_dir} \\\n"
        f"      --output_dir checkpoints/target_dapt_marker/"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",      default="bert-base-uncased",
                        help="Base backbone to continue pretraining")
    parser.add_argument("--corpus_dir",    required=True,
                        help="Directory of .txt files for financial domain corpus")
    parser.add_argument("--output_dir",    default="checkpoints/dapt/")
    parser.add_argument("--max_files",     type=int,   default=None,
                        help="Max corpus files to load (None = all)")
    parser.add_argument("--max_length",    type=int,   default=512)
    parser.add_argument("--chunk_size",    type=int,   default=512)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_steps",     type=int,   default=10000)
    parser.add_argument("--eval_steps",    type=int,   default=500)
    parser.add_argument("--fp16",          action="store_true")
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()
    main(args)
