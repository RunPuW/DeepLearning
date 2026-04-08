"""
utils.py
--------
Shared utilities: dataset loading, input formatting, model building,
metric computation. All training scripts import from here.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from schema import (
    LABEL2ID,
    ID2LABEL,
    NUM_LABELS,
    FinSentSample,
    PredictionRecord,
)


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def _adapt_record(raw: dict) -> dict:
    """
    Adapts the actual data schema to FinSentSample fields.

    Actual fields:  uid, dataset, task_type, text, label, target,
                    market_label, split, hard_subset
    Expected fields: sample_id, source, text, label, target,
                     split, subset_tags, source_text_id
    """
    if "subset_tags" in raw and isinstance(raw["subset_tags"], list):
        subset_tags = raw["subset_tags"]
    else:
        hard_subset = raw.get("hard_subset")
        if isinstance(hard_subset, str) and hard_subset:
            subset_tags = [hard_subset]
        elif isinstance(hard_subset, list):
            subset_tags = [t for t in hard_subset if t]
        else:
            subset_tags = []

    # Infer source_text_id from uid prefix
    # e.g. "finent_000690" -> "finent_000"  (groups same-headline samples)
    uid = raw.get("uid", "")
    parts = uid.split("_")
    source_text_id = "_".join(parts[:-1]) if len(parts) > 1 else uid

    return {
        "sample_id":      uid,
        "text":           raw["text"],
        "target":         raw.get("target"),
        "label":          raw["label"],
        "source":         raw.get("dataset", "unknown"),
        "split":          raw.get("split", "train"),
        "subset_tags":    subset_tags,
        "source_text_id": source_text_id,
    }


def load_split(path: str, sources: Optional[List[str]] = None) -> List[FinSentSample]:
    """
    Load a split JSONL. Handles both raw data schema (uid/dataset fields)
    and canonical FinSentSample schema automatically.
    Optionally filter to specific sources.
    """
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            # Auto-detect raw schema by presence of 'uid' field
            if "uid" in raw:
                raw = _adapt_record(raw)
            s = FinSentSample.from_dict(raw)
            if sources is None or s.source in sources:
                samples.append(s)
    return samples


# ------------------------------------------------------------------
# Input formatting
# ------------------------------------------------------------------

def format_input_sentence(sample: FinSentSample) -> str:
    """
    Mode: sentence
    Input is just the text. Used for FPB sentence-level baseline.
    """
    return sample.text


def format_input_marker(sample: FinSentSample) -> str:
    """
    Mode: marker
    Wraps the target with special tokens inside the text.
    If target not found as substring, prepends [TGT] ... [/TGT].
    """
    if sample.target is None:
        return sample.text

    text   = sample.text
    target = sample.target

    pattern_idx = text.lower().find(target.lower())
    if pattern_idx != -1:
        end_idx = pattern_idx + len(target)
        marked = (
            text[:pattern_idx]
            + "[TGT] "
            + text[pattern_idx:end_idx]
            + " [/TGT]"
            + text[end_idx:]
        )
    else:
        marked = f"[TGT] {target} [/TGT] {text}"

    return marked


def format_input_concat(sample: FinSentSample) -> str:
    """
    Mode: concat
    Simple text [SEP] target concatenation.
    """
    if sample.target is None:
        return sample.text
    return f"{sample.text} [SEP] {sample.target}"


INPUT_FORMATTERS = {
    "sentence": format_input_sentence,
    "marker":   format_input_marker,
    "concat":   format_input_concat,
}


# ------------------------------------------------------------------
# PyTorch Dataset
# ------------------------------------------------------------------

class FinSentDataset(Dataset):
    def __init__(
        self,
        samples: List[FinSentSample],
        tokenizer,
        input_mode: str,
        max_length: int = 128,
    ):
        self.samples   = samples
        self.tokenizer = tokenizer
        self.input_mode = input_mode
        self.max_length = max_length
        self.formatter  = INPUT_FORMATTERS[input_mode]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s    = self.samples[idx]
        text = self.formatter(s)
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(LABEL2ID[s.label], dtype=torch.long),
            "sample_idx":     idx,
        }


# ------------------------------------------------------------------
# Model building
# ------------------------------------------------------------------

def build_model_and_tokenizer(
    backbone: str,
    add_special_tokens: bool = False,
) -> Tuple:
    """
    Returns (model, tokenizer).
    add_special_tokens: add [TGT] / [/TGT] tokens for marker mode.
    """
    tokenizer = AutoTokenizer.from_pretrained(backbone)

    if add_special_tokens:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["[TGT]", "[/TGT]"]}
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        backbone,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    if add_special_tokens:
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def compute_metrics(
    true_labels: List[str],
    pred_labels: List[str],
    label_names: Optional[List[str]] = None,
) -> Dict:
    if label_names is None:
        label_names = list(LABEL2ID.keys())

    macro_f1 = f1_score(
        true_labels,
        pred_labels,
        labels=label_names,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        true_labels,
        pred_labels,
        labels=label_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "macro_f1": round(macro_f1, 4),
        "per_class": {
            lbl: {
                "precision": round(report[lbl]["precision"], 4),
                "recall":    round(report[lbl]["recall"], 4),
                "f1":        round(report[lbl]["f1-score"], 4),
                "support":   report[lbl]["support"],
            }
            for lbl in label_names
        },
    }


def compute_subset_metrics(
    records: List[PredictionRecord],
    tags: List[str] = ("multi_entity", "conflict", "ambiguous"),
) -> Dict:
    results = {}
    for tag in tags:
        subset = [r for r in records if tag in r.subset_tags]
        if not subset:
            results[tag] = {"macro_f1": None, "n_samples": 0}
            continue
        true_labels = [r.true_label for r in subset]
        pred_labels = [r.pred_label for r in subset]
        macro_f1 = f1_score(
            true_labels,
            pred_labels,
            average="macro",
            zero_division=0,
        )
        results[tag] = {
            "macro_f1":  round(macro_f1, 4),
            "n_samples": len(subset),
        }
    return results


# ------------------------------------------------------------------
# Checkpoint I/O
# ------------------------------------------------------------------

def save_checkpoint(model, tokenizer, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[OK] Checkpoint saved to {output_dir}")


def load_checkpoint(checkpoint_dir: str) -> Tuple:
    model     = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    return model, tokenizer
