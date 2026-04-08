"""
make_splits.py
--------------
Loads train/val/test.jsonl from data dir, re-derives subset_tags
using gold-label-based rules (not the original hard_subset field),
and writes canonical splits with corrected tags.

Run this ONCE before re-training. Output overwrites existing splits.

Rules (frozen before any model training):
  multi_entity : same text has >= 2 distinct targets
  conflict     : same text has >= 2 targets with DIFFERENT gold labels
  ambiguous    : text satisfies ANY of:
                 (a) token count < 12
                 (b) target not found as substring in text
                 (c) hedge word present AND no sentiment anchor present

Usage:
    python make_splits.py --data_dir F:\\stage1\\data\\
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


# ------------------------------------------------------------------
# Lexicons (frozen, do not change after protocol is set)
# ------------------------------------------------------------------

HEDGE_LEXICON = {
    "could", "may", "might", "possibly", "potentially", "reportedly",
    "seems", "appears", "likely", "unlikely", "expected", "projected",
    "anticipated", "estimated", "suggests", "indicates",
}

SENTIMENT_LEXICON = {
    "surge", "soar", "jump", "rise", "gain", "profit", "beat",
    "exceed", "strong", "record", "growth", "rally", "upgrade",
    "fall", "drop", "decline", "loss", "miss", "weak", "cut",
    "downgrade", "warn", "risk", "collapse", "plunge", "slump",
}


# ------------------------------------------------------------------
# Ambiguity rules (text-based only, no model signal)
# ------------------------------------------------------------------

def is_ambiguous(text: str, target: str) -> bool:
    tokens = text.lower().split()

    # Rule (a): very short text
    if len(tokens) < 12:
        return True

    # Rule (b): target not found as substring (implied reference)
    if target and target.lower() not in text.lower():
        return True

    # Rule (c): hedged AND no sentiment anchor
    token_set = set(tokens)
    has_hedge = bool(token_set & HEDGE_LEXICON)
    has_anchor = bool(token_set & SENTIMENT_LEXICON)
    if has_hedge and not has_anchor:
        return True

    return False


# ------------------------------------------------------------------
# Core tagging logic
# ------------------------------------------------------------------

def derive_subset_tags(records: List[dict]) -> List[dict]:
    """
    Groups records by text, derives multi_entity and conflict from
    gold labels, derives ambiguous from text rules.
    Returns records with corrected subset_tags field.
    """
    # Group by full text (most reliable key for same-headline grouping)
    text_groups: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(records):
        text_groups[r["text"]].append(i)

    # Derive tags per group
    for text, indices in text_groups.items():
        group = [records[i] for i in indices]
        targets = [r.get("target") for r in group if r.get("target")]
        labels  = [r["label"] for r in group]

        unique_targets = set(t for t in targets if t)
        unique_labels  = set(labels)

        is_multi    = len(unique_targets) >= 2
        is_conflict = is_multi and len(unique_labels) > 1

        for i in indices:
            r = records[i]
            tags = []
            if is_multi:
                tags.append("multi_entity")
            if is_conflict:
                tags.append("conflict")
            target = r.get("target") or ""
            if is_ambiguous(r["text"], target):
                tags.append("ambiguous")
            r["subset_tags"] = tags

    return records


# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------

def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def print_stats(records: List[dict], split_name: str) -> None:
    from collections import Counter
    tag_counter: Counter = Counter()
    for r in records:
        tags = r.get("subset_tags", [])
        if not tags:
            tag_counter["none"] += 1
        for t in tags:
            tag_counter[t] += 1

    print(f"\n  [{split_name.upper()}] {len(records)} samples")
    for tag in ["none", "multi_entity", "conflict", "ambiguous"]:
        count = tag_counter.get(tag, 0)
        pct = 100 * count / len(records) if records else 0
        print(f"    {tag:<14}: {count:>5}  ({pct:.1f}%)")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(args):
    data_dir = Path(args.data_dir)

    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            print(f"[SKIP] {path} not found")
            continue

        records = load_jsonl(str(path))
        print(f"[OK] Loaded {len(records)} records from {path}")

        records = derive_subset_tags(records)
        save_jsonl(records, str(path))
        print(f"[OK] Written back to {path}")
        print_stats(records, split)

    print("\n[OK] Done. Re-run training and eval to get updated subset metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="F:\\stage1\\data\\")
    args = parser.parse_args()
    main(args)
