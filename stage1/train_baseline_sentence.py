"""
train_baseline_sentence.py
--------------------------
Trains a sentence-level financial sentiment classifier on FPB.
This is the floor baseline: no target awareness, no retrieval, no routing.

Usage:
    python train_baseline_sentence.py \
        --backbone ProsusAI/finbert \
        --data_dir data/processed/ \
        --output_dir checkpoints/sentence_finbert/ \
        --seed 42

To compare multiple backbones, run once per backbone with different
--backbone and --output_dir arguments.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from schema import (
    LABEL2ID,
    ID2LABEL,
    FinSentSample,
    PredictionRecord,
    TrainConfig,
    save_predictions,
)
from utils import (
    FinSentDataset,
    build_model_and_tokenizer,
    compute_metrics,
    compute_subset_metrics,
    load_split,
    save_checkpoint,
    set_seed,
)


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(loader)


# ------------------------------------------------------------------
# Evaluation loop
# ------------------------------------------------------------------

def evaluate(
    model,
    loader,
    samples: List[FinSentSample],
    device,
    model_alias: str,
    backbone: str,
    input_mode: str,
) -> List[PredictionRecord]:
    model.eval()
    all_records = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            indices        = batch["sample_idx"].tolist()

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

            probs      = torch.softmax(logits, dim=-1).cpu().tolist()
            pred_ids   = torch.argmax(logits, dim=-1).cpu().tolist()

            for i, idx in enumerate(indices):
                s = samples[idx]
                rec = PredictionRecord(
                    sample_id   = s.sample_id,
                    source      = s.source,
                    split       = s.split,
                    text        = s.text,
                    target      = s.target,
                    true_label  = s.label,
                    pred_label  = ID2LABEL[pred_ids[i]],
                    pred_probs  = [round(p, 6) for p in probs[i]],
                    subset_tags = s.subset_tags,
                    model_alias = model_alias,
                    backbone    = backbone,
                    input_mode  = input_mode,
                )
                all_records.append(rec)

    return all_records


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OK] Device: {device}")

    # --- Backbone alias for naming ---
    backbone_alias = args.backbone.replace("/", "_")
    model_alias = f"sentence_{backbone_alias}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data (FPB only for sentence baseline) ---
    print("[INFO] Loading FPB data...")
    train_samples = load_split(
        os.path.join(args.data_dir, "train.jsonl"),
        sources=["fpb"],
    )
    dev_samples = load_split(
        os.path.join(args.data_dir, "val.jsonl"),
        sources=["fpb"],
    )
    test_samples = load_split(
        os.path.join(args.data_dir, "test.jsonl"),
        sources=["fpb"],
    )
    print(f"  Train: {len(train_samples)}  Dev: {len(dev_samples)}  Test: {len(test_samples)}")

    # --- Build model ---
    # Sentence mode does not need special tokens
    model, tokenizer = build_model_and_tokenizer(
        args.backbone,
        add_special_tokens=False,
    )
    model.to(device)

    # --- Datasets and loaders ---
    train_dataset = FinSentDataset(
        train_samples, tokenizer, input_mode="sentence", max_length=args.max_length
    )
    dev_dataset = FinSentDataset(
        dev_samples, tokenizer, input_mode="sentence", max_length=args.max_length
    )
    test_dataset = FinSentDataset(
        test_samples, tokenizer, input_mode="sentence", max_length=args.max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_dataset,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    # --- Optimizer and scheduler ---
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # --- Save config for reproducibility ---
    config = TrainConfig(
        model_alias=model_alias,
        backbone=args.backbone,
        input_mode="sentence",
        task_type="sentence",
        datasets_used=["fpb"],
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
    )
    config.save(output_dir / "train_config.json")

    # --- Training loop ---
    best_dev_f1 = 0.0
    best_epoch = -1
    train_log = []

    print(f"\n[INFO] Starting training: {model_alias}")
    for epoch in range(1, args.num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)

        dev_records = evaluate(
            model, dev_loader, dev_samples, device,
            model_alias, args.backbone, "sentence"
        )
        dev_metrics = compute_metrics(
            [r.true_label for r in dev_records],
            [r.pred_label for r in dev_records],
        )
        dev_f1 = dev_metrics["macro_f1"]

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "dev_macro_f1": dev_f1,
        }
        train_log.append(log_entry)
        print(f"  Epoch {epoch:02d} | loss={train_loss:.4f} | dev_macro_f1={dev_f1:.4f}")

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_epoch = epoch
            save_checkpoint(model, tokenizer, str(output_dir / "best_checkpoint"))
            save_predictions(dev_records, str(output_dir / "dev_predictions_best.jsonl"))

    print(f"\n[OK] Best dev macro-F1: {best_dev_f1:.4f} at epoch {best_epoch}")

    # --- Final test evaluation using best checkpoint ---
    from utils import load_checkpoint
    best_model, _ = load_checkpoint(str(output_dir / "best_checkpoint"))
    best_model.to(device)

    test_records = evaluate(
        best_model, test_loader, test_samples, device,
        model_alias, args.backbone, "sentence"
    )
    test_metrics = compute_metrics(
        [r.true_label for r in test_records],
        [r.pred_label for r in test_records],
    )
    test_subset_metrics = compute_subset_metrics(test_records)

    # --- Save everything ---
    save_predictions(test_records, str(output_dir / "test_predictions.jsonl"))

    results = {
        "model_alias":        model_alias,
        "backbone":           args.backbone,
        "input_mode":         "sentence",
        "task_type":          "sentence",
        "best_dev_macro_f1":  best_dev_f1,
        "best_epoch":         best_epoch,
        "test_macro_f1":      test_metrics["macro_f1"],
        "test_per_class":     test_metrics["per_class"],
        "test_subset":        test_subset_metrics,
        "seed":               args.seed,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "train_log.json", "w") as f:
        json.dump(train_log, f, indent=2)

    print(f"\n[RESULTS] Test macro-F1: {test_metrics['macro_f1']:.4f}")
    for tag, m in test_subset_metrics.items():
        f1_str = f"{m['macro_f1']:.4f}" if m["macro_f1"] is not None else "N/A"
        print(f"  subset/{tag}: {f1_str} (n={m['n_samples']})")

    print(f"\n[OK] All outputs written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone",      default="ProsusAI/finbert")
    parser.add_argument("--data_dir",      default="data/processed/")
    parser.add_argument("--output_dir",    default="checkpoints/sentence_finbert/")
    parser.add_argument("--max_length",    type=int,   default=128)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs",    type=int,   default=5)
    parser.add_argument("--warmup_ratio",  type=float, default=0.1)
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()
    main(args)
