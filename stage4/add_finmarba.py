"""
add_finmarba.py
---------------
Standalone script: reads the downloaded FinMarBa arrow file,
converts to the project 8-field schema, runs subset tagging,
and appends records to test.jsonl for alignment analysis.

Does NOT touch train.jsonl or val.jsonl.
Does NOT require rerunning preprocess.py.

Usage:
    python add_finmarba.py \
        --arrow  F:\\stage4\\FinMarBa\\train\\data-00000-of-00001.arrow \
        --data_dir F:\\stage4\\data
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from datasets import Dataset as HFDataset


# ------------------------------------------------------------------
# Label normalization (same mapping as preprocess.py)
# ------------------------------------------------------------------

MARKET_LABEL_MAP = {
    "positive": "positive", "pos": "positive",  "1":  "positive",
    "neutral":  "neutral",  "neu": "neutral",   "0":  "neutral",
    "negative": "negative", "neg": "negative",  "-1": "negative",
    "bull":     "positive",
    "bear":     "negative",
}


def normalize_market_label(raw):
    if raw is None:
        return None
    try:
        if pd.isna(raw):
            return None
    except Exception:
        pass
    s = str(raw).strip().lower()
    return MARKET_LABEL_MAP.get(s)


# ------------------------------------------------------------------
# Subset tagging (same rules as make_splits.py)
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


def is_ambiguous(text: str) -> bool:
    tokens = text.lower().split()
    if len(tokens) < 12:
        return True
    ts = set(tokens)
    if ts & HEDGE_LEXICON and not ts & SENTIMENT_LEXICON:
        return True
    return False


def tag_subsets(records):
    """
    FinMarBa records have no target, so multi_entity and conflict
    do not apply. Only ambiguous is tagged.
    """
    for r in records:
        tags = []
        if is_ambiguous(r["text"]):
            tags.append("ambiguous")
        r["subset_tags"] = tags
    return records


# ------------------------------------------------------------------
# Load FinMarBa arrow file
# ------------------------------------------------------------------

def load_finmarba(arrow_path: str):
    print(f"[INFO] Loading FinMarBa from {arrow_path}")
    ds = HFDataset.from_file(arrow_path)
    df = ds.to_pandas()
    print(f"[OK] Raw rows: {len(df)}")
    print(f"[OK] Columns: {df.columns.tolist()}")

    # Detect text column
    text_col = None
    for candidate in ("Title", "headline", "text", "Headline", "sentence"):
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        raise ValueError(f"No text column found. Columns: {df.columns.tolist()}")
    print(f"[OK] Text column: '{text_col}'")

    # Detect market label column
    # Priority: "Global Sentiment" is the aggregate signal (integer: 1/0/-1).
    # "Sentiment" is a per-ticker dict and must be skipped.
    label_col = None
    candidates = [
        "Global Sentiment", "Aggregate Sentiment", "aggregate_sentiment",
        "market_label", "Label", "label",
    ]
    for candidate in candidates:
        if candidate in df.columns:
            sample = df[candidate].dropna().head(10).tolist()
            # Skip columns that look like dicts (per-entity breakdowns)
            if any(isinstance(v, dict) or (isinstance(v, str) and v.startswith("{"))
                   for v in sample):
                continue
            label_col = candidate
            break
    if label_col is None:
        print(f"[WARN] No market label column found. market_label will be None.")
    else:
        print(f"[OK] Market label column: '{label_col}'")

    records = []
    for idx, row in df.iterrows():
        text = str(row[text_col]).strip() if row[text_col] is not None else None
        try:
            if pd.isna(row[text_col]):
                text = None
        except Exception:
            pass
        if not text or len(text) < 10:
            continue

        market_label = None
        if label_col is not None:
            market_label = normalize_market_label(row[label_col])

        records.append({
            "uid":          f"finmarba_{idx:05d}",
            "dataset":      "finmarba",
            "task_type":    "market_aux",
            "text":         text,
            "label":        "neutral",   # placeholder so schema is valid
            "target":       None,
            "market_label": market_label,
            "split":        "test",
            "hard_subset":  None,
            "subset_tags":  [],
        })

    valid_market = sum(1 for r in records if r["market_label"] is not None)
    print(f"[OK] Parsed: {len(records)} records "
          f"({valid_market} with valid market_label)")
    return records


# ------------------------------------------------------------------
# Append to test.jsonl
# ------------------------------------------------------------------

def append_to_test(records, test_path: str):
    # Check for duplicates: don't append if finmarba already exists
    existing_uids = set()
    finmarba_count = 0
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            existing_uids.add(d.get("uid") or d.get("sample_id"))
            if d.get("dataset") == "finmarba":
                finmarba_count += 1

    if finmarba_count > 0:
        print(f"[WARN] test.jsonl already contains {finmarba_count} finmarba records.")
        print(f"       Skipping to avoid duplicates. Delete existing finmarba rows first.")
        return 0

    new_records = [r for r in records if r["uid"] not in existing_uids]
    with open(test_path, "a", encoding="utf-8") as f:
        for r in new_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] Appended {len(new_records)} FinMarBa records to {test_path}")
    return len(new_records)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(args):
    arrow_path = args.arrow
    test_path  = os.path.join(args.data_dir, "test.jsonl")

    if not Path(arrow_path).exists():
        print(f"[FAIL] Arrow file not found: {arrow_path}")
        print(f"       Expected at: FinMarBa/train/data-00000-of-00001.arrow")
        return

    if not Path(test_path).exists():
        print(f"[FAIL] test.jsonl not found: {test_path}")
        return

    # Load and tag
    records = load_finmarba(arrow_path)
    if not records:
        print("[FAIL] No records loaded from FinMarBa.")
        return

    records = tag_subsets(records)

    # Show market label distribution
    c = Counter(r["market_label"] for r in records)
    print(f"\n[MARKET LABEL DISTRIBUTION]")
    for label, count in c.most_common():
        print(f"  {label}: {count}")

    # Append to test.jsonl
    n_added = append_to_test(records, test_path)
    if n_added == 0:
        return

    # Verify
    print(f"\n[VERIFY] test.jsonl after append:")
    total = 0
    by_dataset: Counter = Counter()
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            total += 1
            by_dataset[d.get("dataset", "unknown")] += 1
    print(f"  Total records: {total}")
    for ds, count in by_dataset.most_common():
        print(f"  {ds}: {count}")

    print(f"\n[OK] Done. Now re-run train_with_retrieval.py --stage train")
    print(f"     and analyze_alignment.py to get alignment results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arrow",
        default="F:\\stage4\\FinMarBa\\train\\data-00000-of-00001.arrow",
    )
    parser.add_argument("--data_dir", default="F:\\stage4\\data")
    args = parser.parse_args()
    main(args)
