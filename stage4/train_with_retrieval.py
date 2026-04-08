"""
train_with_retrieval.py  [v2b: path-B bypass]
---------------------------------------------
Same path-B bypass changes as eval_four_policies.py [v2b].
3-policy version (none / always_on / conditional), no router.

Path-B bypass logic:
  bypass=True  when: no context AND NOT conflict/multi_entity
  bypass=False when: has context OR is conflict/multi_entity
  -> conflict/multi_entity always go through fusion (preserved)
  -> none/ambiguous with no context skip fusion (zero-vector noise removed)

Usage:
  # Original (no bypass):
  python train_with_retrieval.py --stage train

  # Path-B bypass:
  python train_with_retrieval.py --stage train --use_bypass ^
      --output_dir F:\\stage4\\checkpoints\\retrieval_model_bypass
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from schema import (
    LABEL2ID,
    ID2LABEL,
    NUM_LABELS,
    FinSentSample,
    PredictionRecord,
    save_predictions,
)
from utils import (
    compute_metrics,
    compute_subset_metrics,
    load_split,
    set_seed,
)
from alignment_head import (
    MarketAlignmentHead,
    AlignmentLoss,
    encode_market_labels,
)
from trigger_policy import RetrievalTrigger, batch_retrieve


# ------------------------------------------------------------------
# Local checkpoint loader
# ------------------------------------------------------------------

def load_local_checkpoint(checkpoint_path: Path):
    config_file = checkpoint_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError("No config.json in {}".format(checkpoint_path))
    with open(config_file, encoding="utf-8") as f:
        config_dict = json.load(f)
    model_type = config_dict.get("model_type", "bert")
    config_dict.pop("model_type", None)
    config = AutoConfig.for_model(model_type, **config_dict)
    config.num_labels = config_dict.get("num_labels", NUM_LABELS)
    model = AutoModelForSequenceClassification.from_config(config)
    sf_file  = checkpoint_path / "model.safetensors"
    bin_file = checkpoint_path / "pytorch_model.bin"
    if sf_file.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(sf_file))
    elif bin_file.exists():
        state_dict = torch.load(str(bin_file), map_location="cpu")
    else:
        raise FileNotFoundError(
            "No model.safetensors or pytorch_model.bin in {}".format(checkpoint_path))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("[WARN] Missing keys: {}".format(len(missing)))
    if unexpected:
        print("[WARN] Unexpected keys: {}".format(len(unexpected)))
    return model


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

class RetrievalAugmentedModel(nn.Module):
    """
    FinBERT + conditional fusion head.
    Path-B bypass: bypass_mask=True only for no-context non-hard samples.
    """

    def __init__(self, backbone: str, alpha: float = 0.15):
        super().__init__()
        self.encoder     = AutoModel.from_pretrained(backbone)
        hidden_size      = self.encoder.config.hidden_size
        self.hidden_size = hidden_size
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, NUM_LABELS),
        )
        self.align_head = MarketAlignmentHead(hidden_size=hidden_size)
        self.align_loss = AlignmentLoss(alpha=alpha)
        self.main_loss  = nn.CrossEntropyLoss()

    def _cls(self, input_ids, attention_mask) -> torch.Tensor:
        return self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]

    def forward(
        self,
        input_ids:       torch.Tensor,
        attention_mask:  torch.Tensor,
        context_vectors: Optional[torch.Tensor] = None,
        labels:          Optional[torch.Tensor] = None,
        market_labels:   Optional[torch.Tensor] = None,
        bypass_mask:     Optional[torch.Tensor] = None,  # (B,) bool
    ) -> Dict:
        cls           = self._cls(input_ids, attention_mask)
        batch_size    = cls.size(0)
        market_logits = self.align_head(cls)

        if bypass_mask is not None and bypass_mask.any():
            sent_logits = torch.empty(batch_size, NUM_LABELS, device=cls.device)
            bypass_idx  = bypass_mask.nonzero(as_tuple=True)[0]
            fusion_idx  = (~bypass_mask).nonzero(as_tuple=True)[0]

            if len(bypass_idx) > 0:
                sent_logits[bypass_idx] = self.classifier(cls[bypass_idx])

            if len(fusion_idx) > 0:
                if context_vectors is not None:
                    ctx_f = context_vectors[fusion_idx]
                else:
                    ctx_f = torch.zeros(
                        len(fusion_idx), self.hidden_size, device=cls.device)
                fused = self.fusion_layer(
                    torch.cat([cls[fusion_idx], ctx_f], dim=1))
                sent_logits[fusion_idx] = self.classifier(fused)
        else:
            if context_vectors is None:
                context_vectors = torch.zeros(
                    batch_size, self.hidden_size, device=cls.device)
            fused       = self.fusion_layer(torch.cat([cls, context_vectors], dim=1))
            sent_logits = self.classifier(fused)

        total_loss = None
        if labels is not None:
            main_loss  = self.main_loss(sent_logits, labels)
            align_loss = torch.tensor(0.0, device=cls.device)
            if market_labels is not None:
                align_loss = self.align_loss(market_logits, market_labels)
            total_loss = main_loss + align_loss

        return {
            "loss":             total_loss,
            "sentiment_logits": sent_logits,
            "market_logits":    market_logits,
        }

    def resize_token_embeddings(self, size: int):
        self.encoder.resize_token_embeddings(size)


# ------------------------------------------------------------------
# Subset metrics helper  [v2b: includes none subset]
# ------------------------------------------------------------------

def compute_all_subset_metrics(records: List[PredictionRecord]) -> dict:
    subset_metrics = compute_subset_metrics(records)
    none_records   = [r for r in records if not r.subset_tags]
    if none_records:
        true_labels = [r.true_label for r in none_records]
        pred_labels = [r.pred_label for r in none_records]
        none_m = compute_metrics(true_labels, pred_labels)
        subset_metrics["none"] = {
            "macro_f1":  none_m["macro_f1"],
            "n_samples": len(none_records),
        }
    return subset_metrics


# ------------------------------------------------------------------
# Dataset  [v2b: path-B bypass logic]
# ------------------------------------------------------------------

_HARD_SUBSETS = frozenset(["conflict", "multi_entity"])


class RetrievalDataset(Dataset):
    def __init__(
        self,
        samples:                List[FinSentSample],
        tokenizer,
        retrieval_cache:        Dict[str, dict],
        context_encoder,
        device:                 torch.device,
        policy:                 str = "always_on",
        threshold:              float = 0.5,
        max_length:             int = 128,
        is_train:               bool = False,
        retrieval_dropout_rate: float = 0.5,
    ):
        self.samples                = samples
        self.tokenizer              = tokenizer
        self.retrieval_cache        = retrieval_cache
        self.context_encoder        = context_encoder
        self.device                 = device
        self.policy                 = policy
        self.threshold              = threshold
        self.max_length             = max_length
        self.is_train               = is_train
        self.retrieval_dropout_rate = retrieval_dropout_rate

    def __len__(self):
        return len(self.samples)

    def _get_context_vec(self, contexts: List[str]) -> Optional[torch.Tensor]:
        if not contexts:
            return None
        ctx_enc = self.tokenizer(
            contexts, max_length=64, padding=True,
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            out = self.context_encoder(
                input_ids=ctx_enc["input_ids"].to(self.device),
                attention_mask=ctx_enc["attention_mask"].to(self.device),
            )
        return out.last_hidden_state[:, 0, :].mean(dim=0).cpu()

    def __getitem__(self, idx):
        s    = self.samples[idx]
        text = "{} [SEP] {}".format(s.text, s.target) if s.target else s.text
        enc  = self.tokenizer(
            text, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        entry       = self.retrieval_cache.get(s.sample_id, {})
        uncertainty = entry.get("uncertainty", 0.0)
        contexts    = entry.get("contexts", [])

        if self.is_train:
            use_context = (
                len(contexts) > 0
                and random.random() > self.retrieval_dropout_rate
            )
        else:
            if self.policy == "always_on":
                use_context = len(contexts) > 0
            elif self.policy == "conditional":
                use_context = uncertainty > self.threshold and len(contexts) > 0
            else:  # none
                use_context = False

        # [v2b] Path-B bypass: True only when no context AND not hard subset
        is_hard = any(t in _HARD_SUBSETS for t in (s.subset_tags or []))
        bypass  = (not use_context) and (not is_hard)

        context_vec = self._get_context_vec(contexts) if use_context else None
        raw_market  = getattr(s, "market_label", None)

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(LABEL2ID[s.label], dtype=torch.long),
            "market_label":   raw_market,
            "context_vec":    context_vec,
            "sample_idx":     idx,
            "bypass":         bypass,
        }


def collate_fn(batch):
    input_ids      = torch.stack([b["input_ids"]      for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels         = torch.stack([b["label"]          for b in batch])
    sample_indices = [b["sample_idx"] for b in batch]
    market_labels  = encode_market_labels([b["market_label"] for b in batch])
    bypass_mask    = torch.tensor([b["bypass"] for b in batch], dtype=torch.bool)

    ctx_list = [b["context_vec"] for b in batch]
    if all(c is None for c in ctx_list):
        context_vectors = None
    else:
        h       = next(c for c in ctx_list if c is not None).shape[0]
        stacked = [c if c is not None else torch.zeros(h) for c in ctx_list]
        context_vectors = torch.stack(stacked)

    return {
        "input_ids":       input_ids,
        "attention_mask":  attention_mask,
        "labels":          labels,
        "market_labels":   market_labels,
        "context_vectors": context_vectors,
        "sample_indices":  sample_indices,
        "bypass_mask":     bypass_mask,
    }


# ------------------------------------------------------------------
# Cache helpers
# ------------------------------------------------------------------

def build_retrieval_cache(samples, trigger, baseline, tokenizer,
                          device, batch_size=32, max_length=128):
    print("[INFO] Building retrieval cache...")
    cache = {}
    baseline.eval()
    for start in range(0, len(samples), batch_size):
        batch_s = samples[start:start + batch_size]
        texts   = [
            "{} [SEP] {}".format(s.text, s.target) if s.target else s.text
            for s in batch_s
        ]
        enc = tokenizer(texts, max_length=max_length, padding=True,
                        truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = baseline(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits
        probs  = torch.softmax(logits, dim=-1).cpu()
        uncert = (1.0 - probs.max(dim=-1).values).tolist()
        retrieved = batch_retrieve(
            trigger, texts=[s.text for s in batch_s],
            entities=[s.target or "" for s in batch_s], policy="always_on",
        )
        for i, s in enumerate(batch_s):
            cache[s.sample_id] = {
                "uncertainty": uncert[i],
                "contexts":    [r["context"] for r in retrieved[i]],
            }
        print("  cached {}/{}".format(
            min(start + batch_size, len(samples)), len(samples)), end="\r")
    print("\n[OK] Cache: {} entries".format(len(cache)))
    return cache


def save_cache(cache, path):
    with open(path, "w", encoding="utf-8") as f:
        for sid, entry in cache.items():
            f.write(json.dumps({"sample_id": sid, **entry}) + "\n")
    print("[OK] Cache saved: {}".format(path))


def load_cache(path):
    cache = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d   = json.loads(line.strip())
            sid = d.pop("sample_id")
            cache[sid] = d
    print("[OK] Cache loaded: {} entries".format(len(cache)))
    return cache


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, device,
                use_bypass=False) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        ctx = batch["context_vectors"]
        if ctx is not None:
            ctx = ctx.to(device)
        bypass_mask = batch["bypass_mask"].to(device) if use_bypass else None
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            context_vectors=ctx,
            labels=batch["labels"].to(device),
            market_labels=batch["market_labels"].to(device),
            bypass_mask=bypass_mask,
        )
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += out["loss"].item()
    return total_loss / len(loader)


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate(model, samples, tokenizer, cache, context_encoder, device,
             policy, model_alias, backbone, batch_size=32, max_length=128,
             threshold=0.5, use_bypass=False) -> List[PredictionRecord]:
    dataset = RetrievalDataset(
        samples, tokenizer, cache, context_encoder, device,
        policy=policy, threshold=threshold, max_length=max_length,
        is_train=False,
    )
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=False, collate_fn=collate_fn)
    model.eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            ctx = batch["context_vectors"]
            if ctx is not None:
                ctx = ctx.to(device)
            bypass_mask = batch["bypass_mask"].to(device) if use_bypass else None
            out      = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                context_vectors=ctx,
                bypass_mask=bypass_mask,
            )
            logits   = out["sentiment_logits"]
            probs    = torch.softmax(logits, dim=-1).cpu().tolist()
            pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()
            for i, idx in enumerate(batch["sample_indices"]):
                s   = samples[idx]
                tag = "bypass" if use_bypass else "nbypass"
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
                    model_alias = "retrieval_{}_{}".format(policy, tag),
                    backbone    = backbone,
                    input_mode  = "concat",
                ))
    return records


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[OK] Device: {}  bypass={} (path-B)".format(device, args.use_bypass))

    TARGET_SOURCES = ["sentfin", "finentity"]
    MARKET_SOURCES = ["finmarba"]

    train_samples = [s for s in load_split(
        os.path.join(args.data_dir, "train.jsonl"), TARGET_SOURCES) if s.target]
    val_samples   = [s for s in load_split(
        os.path.join(args.data_dir, "val.jsonl"),   TARGET_SOURCES) if s.target]
    test_main     = [s for s in load_split(
        os.path.join(args.data_dir, "test.jsonl"),  TARGET_SOURCES) if s.target]
    test_market   = list(load_split(
        os.path.join(args.data_dir, "test.jsonl"),  MARKET_SOURCES))
    test_all      = test_main + test_market

    print("  Train: {}  Val: {}  Test: {}".format(
        len(train_samples), len(val_samples), len(test_main)))

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    if args.stage == "cache":
        baseline = load_local_checkpoint(Path(args.baseline_checkpoint).resolve())
        baseline.to(device)
        trigger = RetrievalTrigger(
            host=args.host, port=args.port,
            db_name=args.db_name, collection=args.collection,
            backbone=args.backbone, device=device,
        )
        cache = build_retrieval_cache(
            train_samples + val_samples + test_all,
            trigger, baseline, tokenizer, device, args.batch_size,
        )
        save_cache(cache, args.cache)
        return

    if not Path(args.cache).exists():
        print("[FAIL] Cache not found. Run --stage cache first.")
        return

    cache = load_cache(args.cache)
    model = RetrievalAugmentedModel(backbone=args.backbone, alpha=args.alpha)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    context_encoder = model.encoder
    output_dir      = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Train ----
    train_dataset = RetrievalDataset(
        train_samples, tokenizer, cache, context_encoder, device,
        policy="always_on", max_length=args.max_length,
        is_train=True, retrieval_dropout_rate=args.retrieval_dropout_rate,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn,
    )
    optimizer    = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps  = len(train_loader) * args.num_epochs
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps)

    best_val_f1, best_epoch, train_log = 0.0, -1, []
    print("\n[INFO] Training (bypass={} path-B) dropout={}".format(
        args.use_bypass, args.retrieval_dropout_rate))

    for epoch in range(1, args.num_epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            use_bypass=args.use_bypass)
        val_records = evaluate(
            model, val_samples, tokenizer, cache, context_encoder, device,
            policy="always_on", model_alias="retrieval", backbone=args.backbone,
            batch_size=args.batch_size, max_length=args.max_length,
            use_bypass=args.use_bypass)
        val_f1 = compute_metrics(
            [r.true_label for r in val_records],
            [r.pred_label for r in val_records])["macro_f1"]
        train_log.append({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "val_macro_f1": val_f1, "use_bypass": args.use_bypass})
        print("  Epoch {:02d} | loss={:.4f} | val_f1={:.4f}".format(
            epoch, train_loss, val_f1))
        if val_f1 > best_val_f1:
            best_val_f1, best_epoch = val_f1, epoch
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            tokenizer.save_pretrained(str(output_dir / "tokenizer"))

    print("\n[OK] Best val F1: {:.4f} at epoch {}".format(best_val_f1, best_epoch))
    with open(output_dir / "train_log.json", "w") as f:
        json.dump(train_log, f, indent=2)

    # ---- Eval: three policies ----
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    model.to(device)

    tag = "bypass" if args.use_bypass else "nbypass"
    results_summary = {}

    for policy in ("none", "always_on", "conditional"):
        records_main = evaluate(
            model, test_main, tokenizer, cache, context_encoder, device,
            policy=policy, model_alias="retrieval", backbone=args.backbone,
            batch_size=args.batch_size, max_length=args.max_length,
            threshold=args.uncertainty_threshold, use_bypass=args.use_bypass)
        records_all = evaluate(
            model, test_all, tokenizer, cache, context_encoder, device,
            policy=policy, model_alias="retrieval", backbone=args.backbone,
            batch_size=args.batch_size, max_length=args.max_length,
            threshold=args.uncertainty_threshold, use_bypass=args.use_bypass)

        metrics        = compute_metrics(
            [r.true_label for r in records_main],
            [r.pred_label for r in records_main])
        subset_metrics = compute_all_subset_metrics(records_main)  # [v2b]

        save_predictions(records_main,
            str(output_dir / "test_predictions_{}_{}.jsonl".format(policy, tag)))
        save_predictions(records_all,
            str(output_dir / "test_predictions_{}_{}_with_market.jsonl".format(policy, tag)))

        results_summary[policy] = {
            "overall_macro_f1": metrics["macro_f1"],
            "subset":           subset_metrics,
            "use_bypass":       args.use_bypass,
        }

    result_file = output_dir / "retrieval_policy_comparison_{}.json".format(tag)
    with open(result_file, "w") as f:
        json.dump(results_summary, f, indent=2)

    # [v2b] Fixed policy comparison with correct None column
    print("\n[POLICY COMPARISON]  bypass={} (path-B)".format(args.use_bypass))
    print("  {:<14} {:>8} {:>12} {:>10} {:>10} {:>8}".format(
        "Policy", "Overall", "MultiEnt", "Conflict", "Ambiguous", "None"))
    print("  " + "-" * 66)
    for policy, r in results_summary.items():
        overall   = r["overall_macro_f1"]
        conflict  = r["subset"].get("conflict",     {}).get("macro_f1") or 0.0
        ambiguous = r["subset"].get("ambiguous",    {}).get("macro_f1") or 0.0
        multi     = r["subset"].get("multi_entity", {}).get("macro_f1") or 0.0
        none_f1   = r["subset"].get("none",         {}).get("macro_f1") or 0.0
        print("  {:<14} {:>8.4f} {:>12.4f} {:>10.4f} {:>10.4f} {:>8.4f}".format(
            policy, overall, multi, conflict, ambiguous, none_f1))

    print("\n[OK] Results -> {}".format(result_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",    choices=["cache", "train"], default="train")
    parser.add_argument("--backbone", default="ProsusAI/finbert")
    parser.add_argument("--data_dir", default="F:\\stage4\\data")
    parser.add_argument("--host",     default="180.184.67.7")
    parser.add_argument("--port",     default="19530")
    parser.add_argument("--db_name",  default="DL_project")
    parser.add_argument("--collection", default="financial_corpus")
    parser.add_argument("--cache",      default="F:\\stage4\\retrieval_cache.jsonl")
    parser.add_argument("--output_dir", default="F:\\stage4\\checkpoints\\retrieval_model")
    parser.add_argument("--baseline_checkpoint",
                        default="F:\\stage1\\checkpoints\\target_finbert_concat\\best_checkpoint")
    parser.add_argument("--use_bypass",   action="store_true", default=False,
                        help="Path-B bypass: skip fusion for no-context non-hard samples.")
    parser.add_argument("--alpha",                  type=float, default=0.15)
    parser.add_argument("--retrieval_dropout_rate", type=float, default=0.5)
    parser.add_argument("--max_length",             type=int,   default=128)
    parser.add_argument("--batch_size",             type=int,   default=16)
    parser.add_argument("--learning_rate",          type=float, default=2e-5)
    parser.add_argument("--num_epochs",             type=int,   default=5)
    parser.add_argument("--warmup_ratio",           type=float, default=0.1)
    parser.add_argument("--uncertainty_threshold",  type=float, default=0.5)
    parser.add_argument("--seed",                   type=int,   default=42)
    args = parser.parse_args()
    main(args)
