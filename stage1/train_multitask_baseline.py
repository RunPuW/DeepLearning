"""
train_multitask_baseline.py
---------------------------
Multi-task baseline: shared backbone, sentence head (FPB) + target head (SEntFiN/FinEntity).

Training strategy: alternating batches (one sentence batch, one target batch per step).
This is simpler than a mixed dataloader and much easier to debug when something goes wrong.

Loss: L_total = alpha * L_sentence + beta * L_target
      Default: alpha=1.0, beta=1.0

Loss weight sensitivity: run with alpha_beta_pairs to check stability.
Supported pairs: "1:1" (default), "1:2" (upweight target), "2:1" (upweight sentence)

Usage:
    python train_multitask_baseline.py \
        --backbone ProsusAI/finbert \
        --input_mode marker \
        --alpha_beta 1:1 \
        --output_dir checkpoints/multitask_finbert_1_1/ \
        --seed 42
"""

import argparse
import json
import os
from itertools import cycle
from pathlib import Path
from typing import Iterator, List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from schema import (
    ID2LABEL,
    LABEL2ID,
    NUM_LABELS,
    FinSentSample,
    PredictionRecord,
    TrainConfig,
    save_predictions,
)
from utils import (
    FinSentDataset,
    compute_metrics,
    compute_subset_metrics,
    load_split,
    save_checkpoint,
    set_seed,
)


# ------------------------------------------------------------------
# Multi-task model
# ------------------------------------------------------------------

class MultiTaskSentimentModel(nn.Module):
    """
    Shared encoder backbone with two independent classification heads:
      - sentence_head: for FPB sentence-level task
      - target_head:   for SEntFiN / FinEntity target-level task
    """

    def __init__(self, backbone_name: str, num_labels: int = NUM_LABELS):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_name)
        hidden_size  = self.encoder.config.hidden_size

        self.sentence_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels),
        )
        self.target_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def _encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        return outputs.last_hidden_state[:, 0, :]

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        task: str = "target",   # "sentence" | "target"
    ):
        cls_repr = self._encode(input_ids, attention_mask)

        if task == "sentence":
            logits = self.sentence_head(cls_repr)
        else:
            logits = self.target_head(cls_repr)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return loss, logits

    def resize_token_embeddings(self, new_size: int):
        self.encoder.resize_token_embeddings(new_size)


# ------------------------------------------------------------------
# Alternating training step
# ------------------------------------------------------------------

def train_epoch(
    model,
    sentence_loader: DataLoader,
    target_loader: DataLoader,
    optimizer,
    scheduler,
    device,
    alpha: float,
    beta: float,
) -> Tuple[float, float]:
    """
    One epoch of alternating batch training.
    Returns (sentence_loss_avg, target_loss_avg).
    """
    model.train()
    sent_loss_total, tgt_loss_total = 0.0, 0.0
    n_sent, n_tgt = 0, 0

    # Cycle the shorter loader so both run for len(target_loader) steps
    sent_iter: Iterator = cycle(sentence_loader)
    tgt_iter: Iterator  = iter(target_loader)

    for tgt_batch in tgt_iter:
        sent_batch = next(sent_iter)

        # --- Sentence step ---
        optimizer.zero_grad()
        sent_loss, _ = model(
            input_ids      = sent_batch["input_ids"].to(device),
            attention_mask = sent_batch["attention_mask"].to(device),
            labels         = sent_batch["labels"].to(device),
            task           = "sentence",
        )
        (alpha * sent_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # --- Target step ---
        optimizer.zero_grad()
        tgt_loss, _ = model(
            input_ids      = tgt_batch["input_ids"].to(device),
            attention_mask = tgt_batch["attention_mask"].to(device),
            labels         = tgt_batch["labels"].to(device),
            task           = "target",
        )
        (beta * tgt_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        sent_loss_total += sent_loss.item()
        tgt_loss_total  += tgt_loss.item()
        n_sent += 1
        n_tgt  += 1

    return sent_loss_total / max(n_sent, 1), tgt_loss_total / max(n_tgt, 1)


# ------------------------------------------------------------------
# Evaluation (target head only -- this is what matters for comparison)
# ------------------------------------------------------------------

def evaluate_target(
    model,
    loader: DataLoader,
    samples: List[FinSentSample],
    device,
    model_alias: str,
    backbone: str,
    input_mode: str,
) -> List[PredictionRecord]:
    model.eval()
    records = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            indices        = batch["sample_idx"].tolist()

            _, logits  = model(input_ids=input_ids, attention_mask=attention_mask, task="target")
            probs      = torch.softmax(logits, dim=-1).cpu().tolist()
            pred_ids   = torch.argmax(logits, dim=-1).cpu().tolist()

            for i, idx in enumerate(indices):
                s = samples[idx]
                records.append(PredictionRecord(
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
                ))

    return records


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse alpha:beta
    alpha_str, beta_str = args.alpha_beta.split(":")
    alpha = float(alpha_str)
    beta  = float(beta_str)
    print(f"[OK] Loss weights: alpha (sentence)={alpha}  beta (target)={beta}")

    backbone_alias = args.backbone.replace("/", "_")
    ab_tag         = args.alpha_beta.replace(":", "_")
    model_alias    = f"multitask_{backbone_alias}_{args.input_mode}_{ab_tag}"
    output_dir     = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    SENT_SOURCES   = ["fpb"]
    TARGET_SOURCES = ["sentfin", "finentity"]

    train_sent  = load_split(os.path.join(args.data_dir, "train.jsonl"), SENT_SOURCES)
    train_tgt   = load_split(os.path.join(args.data_dir, "train.jsonl"), TARGET_SOURCES)
    train_tgt   = [s for s in train_tgt if s.target]

    dev_tgt     = load_split(os.path.join(args.data_dir, "val.jsonl"),   TARGET_SOURCES)
    dev_tgt     = [s for s in dev_tgt   if s.target]
    test_tgt    = load_split(os.path.join(args.data_dir, "test.jsonl"),  TARGET_SOURCES)
    test_tgt    = [s for s in test_tgt  if s.target]

    print(f"  Train sentence: {len(train_sent)}  |  Train target: {len(train_tgt)}")
    print(f"  Dev target: {len(dev_tgt)}  |  Test target: {len(test_tgt)}")

    # --- Build model ---
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    if args.input_mode == "marker":
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["[TGT]", "[/TGT]"]}
        )

    model = MultiTaskSentimentModel(args.backbone)
    if args.input_mode == "marker":
        model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # --- Datasets ---
    # Sentence loader uses "sentence" mode; target loader uses chosen input_mode
    sent_dataset  = FinSentDataset(train_sent, tokenizer, "sentence",       args.max_length)
    tgt_dataset   = FinSentDataset(train_tgt,  tokenizer, args.input_mode,  args.max_length)
    dev_dataset   = FinSentDataset(dev_tgt,    tokenizer, args.input_mode,  args.max_length)
    test_dataset  = FinSentDataset(test_tgt,   tokenizer, args.input_mode,  args.max_length)

    sent_loader   = DataLoader(sent_dataset,  batch_size=args.batch_size, shuffle=True)
    tgt_loader    = DataLoader(tgt_dataset,   batch_size=args.batch_size, shuffle=True)
    dev_loader    = DataLoader(dev_dataset,   batch_size=args.batch_size, shuffle=False)
    test_loader   = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    # --- Optimizer ---
    # Steps based on target loader length (the primary task)
    optimizer    = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps  = len(tgt_loader) * args.num_epochs * 2  # x2 for two steps per batch
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    # --- Save config ---
    config = TrainConfig(
        model_alias   = model_alias,
        backbone      = args.backbone,
        input_mode    = args.input_mode,
        task_type     = "multitask",
        datasets_used = SENT_SOURCES + TARGET_SOURCES,
        max_length    = args.max_length,
        learning_rate = args.learning_rate,
        batch_size    = args.batch_size,
        num_epochs    = args.num_epochs,
        warmup_ratio  = args.warmup_ratio,
        seed          = args.seed,
        notes         = f"alpha={alpha} beta={beta}",
    )
    config.save(output_dir / "train_config.json")

    # --- Training ---
    best_dev_f1 = 0.0
    best_epoch  = -1
    train_log   = []

    print(f"\n[INFO] Training: {model_alias}")
    for epoch in range(1, args.num_epochs + 1):
        sent_loss, tgt_loss = train_epoch(
            model, sent_loader, tgt_loader, optimizer, scheduler, device, alpha, beta
        )

        dev_records = evaluate_target(
            model, dev_loader, dev_tgt, device,
            model_alias, args.backbone, args.input_mode
        )
        dev_f1 = compute_metrics(
            [r.true_label for r in dev_records],
            [r.pred_label for r in dev_records],
        )["macro_f1"]

        log_entry = {
            "epoch":          epoch,
            "sent_loss":      round(sent_loss, 4),
            "target_loss":    round(tgt_loss, 4),
            "dev_macro_f1":   dev_f1,
        }
        train_log.append(log_entry)
        print(
            f"  Epoch {epoch:02d} | "
            f"sent_loss={sent_loss:.4f} | tgt_loss={tgt_loss:.4f} | "
            f"dev_f1={dev_f1:.4f}"
        )

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_epoch  = epoch
            # Save full model state (not HF standard, so use torch.save)
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            tokenizer.save_pretrained(str(output_dir / "tokenizer"))
            save_predictions(dev_records, str(output_dir / "dev_predictions_best.jsonl"))

    print(f"\n[OK] Best dev macro-F1: {best_dev_f1:.4f} at epoch {best_epoch}")

    # --- Final test eval ---
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    model.to(device)

    test_records        = evaluate_target(
        model, test_loader, test_tgt, device,
        model_alias, args.backbone, args.input_mode
    )
    test_metrics        = compute_metrics(
        [r.true_label for r in test_records],
        [r.pred_label for r in test_records],
    )
    test_subset_metrics = compute_subset_metrics(test_records)

    save_predictions(test_records, str(output_dir / "test_predictions.jsonl"))

    results = {
        "model_alias":       model_alias,
        "backbone":          args.backbone,
        "input_mode":        args.input_mode,
        "task_type":         "multitask",
        "alpha":             alpha,
        "beta":              beta,
        "best_dev_macro_f1": best_dev_f1,
        "best_epoch":        best_epoch,
        "test_macro_f1":     test_metrics["macro_f1"],
        "test_per_class":    test_metrics["per_class"],
        "test_subset":       test_subset_metrics,
        "seed":              args.seed,
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
    parser.add_argument("--input_mode",    choices=["marker", "concat"], default="marker")
    parser.add_argument(
        "--alpha_beta",
        default="1:1",
        choices=["1:1", "1:2", "2:1"],
        help="Loss weight ratio alpha:beta (sentence:target). "
             "Run all 3 for sensitivity check.",
    )
    parser.add_argument("--data_dir",      default="data/processed/")
    parser.add_argument("--output_dir",    default="checkpoints/multitask_finbert_marker_1_1/")
    parser.add_argument("--max_length",    type=int,   default=128)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs",    type=int,   default=5)
    parser.add_argument("--warmup_ratio",  type=float, default=0.1)
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()
    main(args)
