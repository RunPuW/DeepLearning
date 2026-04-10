"""
make_splits.py
==============
Stage 3: Data freeze, stratified train/val/test split, hard subset tagging,
and bias checklist.

Input:
    final_integrated.jsonl   – produced by preprocess.py

Outputs:
    split_spec.json          – machine-readable freeze manifest
    train.jsonl              – training split  (~70%)
    val.jsonl                – validation split (~15%)
    test.jsonl               – test split       (~15%)
    analysis.jsonl           – market_aux rows  (excluded from eval)

Hard subset tags (field: hard_subset, values can be comma-joined):
    multi_entity  – target_semantic: same text has ≥2 distinct targets
    conflict      – (text, target) pair has ≥2 distinct labels in corpus
    ambiguous     – sentence_semantic: same text appears with ≥2 labels in corpus

Bias checklist (stored in split_spec.json):
    cross_split_exact_leakage   – exact text match between val/test and train
    near_duplicate_pairs        – 8-gram Jaccard ≥0.85 across split boundaries
    uid_uniqueness_ok           – no UID appears in >1 split
    label_balance_per_split     – neutral/positive/negative % per split
    split_pollution             – any sample in multiple splits
"""

import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from itertools import combinations

import pandas as pd

# ============================================================
# Config
# ============================================================
RANDOM_SEED   = 42
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
TEST_RATIO    = 0.15

IN_FILE       = "final_integrated.jsonl"
OUT_TRAIN     = "train.jsonl"
OUT_VAL       = "val.jsonl"
OUT_TEST      = "test.jsonl"
OUT_ANALYSIS  = "analysis.jsonl"
OUT_SPEC      = "split_spec.json"

NEAR_DUP_THRESHOLD = 0.85   # Jaccard similarity threshold for 8-gram shingles
NEAR_DUP_NGRAM     = 8      # character n-gram size


# ============================================================
# Hard subset detection
# ============================================================

def detect_hard_subsets(df: pd.DataFrame) -> pd.Series:
    """
    Returns a Series (indexed by df.index) of hard_subset tag strings.
    Tags are pipe-separated when multiple apply (e.g. 'multi_entity|conflict').
    Empty string means no hard subset.
    """
    tags = defaultdict(set)  # index -> set of tags

    # ------- multi_entity -------
    # target_semantic rows where the same text text is shared by ≥2 distinct targets
    tgt_df = df[df["task_type"] == "target_semantic"]
    n_targets_per_text = (
        tgt_df.groupby("text")["target"].nunique()
    )
    multi_texts = set(n_targets_per_text[n_targets_per_text >= 2].index)
    for idx in tgt_df[tgt_df["text"].isin(multi_texts)].index:
        tags[idx].add("multi_entity")

    # ------- conflict -------
    # (text, target) pairs that appear with ≥2 distinct labels in the WHOLE corpus
    conflict_pairs = (
        tgt_df.groupby(["text", "target"])["label"].nunique()
    )
    conflict_pairs = conflict_pairs[conflict_pairs >= 2]
    conflict_set = set(conflict_pairs.index)   # set of (text, target) tuples
    for idx, row in tgt_df.iterrows():
        if (row["text"], row["target"]) in conflict_set:
            tags[idx].add("conflict")

    # ------- ambiguous -------
    # sentence_semantic rows where same text appears with ≥2 labels corpus-wide
    sent_df = df[df["task_type"] == "sentence_semantic"]
    n_labels_per_text = sent_df.groupby("text")["label"].nunique()
    ambiguous_texts = set(n_labels_per_text[n_labels_per_text >= 2].index)
    for idx in sent_df[sent_df["text"].isin(ambiguous_texts)].index:
        tags[idx].add("ambiguous")

    # Build result series
    result = pd.Series("", index=df.index)
    for idx, tag_set in tags.items():
        result.at[idx] = "|".join(sorted(tag_set))
    return result


# ============================================================
# Stratified split
# ============================================================

def stratified_split(df: pd.DataFrame, seed: int):
    """
    Stratify on (dataset, label).
    Returns three DataFrames: train, val, test.
    """
    rng = random.Random(seed)
    train_idx, val_idx, test_idx = [], [], []

    # Group by (dataset, label) stratum
    for (dataset, label), group in df.groupby(["dataset", "label"]):
        idx_list = list(group.index)
        rng.shuffle(idx_list)
        n = len(idx_list)
        n_val  = max(1, round(n * VAL_RATIO))
        n_test = max(1, round(n * TEST_RATIO))
        n_train = n - n_val - n_test

        # Edge case: very small strata
        if n_train < 1:
            n_train = 1
            n_val   = max(0, (n - 1) // 2)
            n_test  = n - 1 - n_val

        train_idx.extend(idx_list[:n_train])
        val_idx.extend(idx_list[n_train:n_train + n_val])
        test_idx.extend(idx_list[n_train + n_val:])

    return (
        df.loc[train_idx].reset_index(drop=True),
        df.loc[val_idx].reset_index(drop=True),
        df.loc[test_idx].reset_index(drop=True),
    )


# ============================================================
# Bias checklist
# ============================================================

def _shingles(text: str, n: int = 8) -> set:
    """Return set of character n-grams."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def check_exact_leakage(train_df, val_df, test_df) -> dict:
    """Check for exact text match between val/test and train."""
    train_texts = set(train_df["text"].str.strip().str.lower())
    val_leak  = val_df["text"].str.strip().str.lower().isin(train_texts).sum()
    test_leak = test_df["text"].str.strip().str.lower().isin(train_texts).sum()
    return {
        "val_texts_in_train":  int(val_leak),
        "test_texts_in_train": int(test_leak),
        "total":               int(val_leak + test_leak),
    }


def check_near_duplicate_leakage(train_df, val_df, test_df,
                                  threshold=NEAR_DUP_THRESHOLD,
                                  n=NEAR_DUP_NGRAM,
                                  sample_cap=500) -> dict:
    """
    Near-duplicate check using character n-gram Jaccard similarity.
    Samples up to `sample_cap` rows from val+test to avoid O(N^2) on large sets.
    Returns count of cross-split pairs above threshold.
    """
    # Build shingle sets for train
    train_shingles = [(i, _shingles(t, n))
                      for i, t in enumerate(train_df["text"])]

    # Sample from val and test
    eval_texts = pd.concat([
        val_df[["uid", "text"]].assign(split="val"),
        test_df[["uid", "text"]].assign(split="test"),
    ]).reset_index(drop=True)

    rng = random.Random(RANDOM_SEED)
    if len(eval_texts) > sample_cap:
        eval_texts = eval_texts.sample(sample_cap, random_state=RANDOM_SEED)

    near_dup_pairs = 0
    for _, row in eval_texts.iterrows():
        sh = _shingles(row["text"], n)
        for _, tr_sh in train_shingles:
            if _jaccard(sh, tr_sh) >= threshold:
                near_dup_pairs += 1
                break  # count at most 1 match per eval row

    return {
        "threshold":           threshold,
        "eval_sample_size":    len(eval_texts),
        "near_dup_pairs_found": int(near_dup_pairs),
        "note": (
            f"Sampled {len(eval_texts)} from val+test vs all train. "
            f"Jaccard >= {threshold} on {n}-char shingles."
        ),
    }


def check_uid_uniqueness(train_df, val_df, test_df, analysis_df) -> dict:
    """Verify no UID exists in more than one split."""
    all_dfs = {
        "train":    train_df,
        "val":      val_df,
        "test":     test_df,
        "analysis": analysis_df,
    }
    uid_to_splits = defaultdict(list)
    for split_name, sdf in all_dfs.items():
        for uid in sdf["uid"]:
            uid_to_splits[uid].append(split_name)

    duplicated = {uid: splits for uid, splits in uid_to_splits.items()
                  if len(splits) > 1}
    return {
        "ok":              len(duplicated) == 0,
        "duplicate_count": len(duplicated),
        "examples":        list(duplicated.items())[:5],
    }


def label_balance(df: pd.DataFrame, name: str) -> dict:
    """Return label distribution as percentages."""
    sem = df[df["task_type"] != "market_aux"]
    counts = sem["label"].value_counts(dropna=False)
    total  = counts.sum()
    return {
        str(lbl): {
            "count":   int(cnt),
            "pct":     round(float(cnt) / total * 100, 2),
        }
        for lbl, cnt in counts.items()
    }


def build_bias_checklist(train_df, val_df, test_df, analysis_df) -> dict:
    print("\n[BiasCheck] Running bias checklist...")

    exact = check_exact_leakage(train_df, val_df, test_df)
    print(f"  Exact leakage: val={exact['val_texts_in_train']}, "
          f"test={exact['test_texts_in_train']}")

    near = check_near_duplicate_leakage(train_df, val_df, test_df)
    print(f"  Near-dup pairs found (sampled): {near['near_dup_pairs_found']}")

    uid_check = check_uid_uniqueness(train_df, val_df, test_df, analysis_df)
    print(f"  UID uniqueness ok: {uid_check['ok']}")

    return {
        "cross_split_exact_leakage": exact,
        "near_duplicate_leakage":    near,
        "uid_uniqueness":            uid_check,
        "label_balance": {
            "train":    label_balance(train_df,    "train"),
            "val":      label_balance(val_df,      "val"),
            "test":     label_balance(test_df,     "test"),
        },
        "future_info_leakage": {
            "verdict": "not_applicable",
            "note":    (
                "FPB, SEntFiN, FinEntity have no publication timestamps. "
                "FinMarBa is market_aux only and excluded from supervised eval."
            ),
        },
        "split_pollution": {
            "ok":  uid_check["ok"],
            "note": "Each UID appears in exactly one split.",
        },
    }


# ============================================================
# Checksum helpers
# ============================================================

def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_df(df: pd.DataFrame) -> str:
    """Stable checksum of a DataFrame by hashing its JSON lines."""
    h = hashlib.md5()
    for uid in sorted(df["uid"].tolist()):
        h.update(uid.encode())
    return h.hexdigest()


def hard_subset_stats(df: pd.DataFrame) -> dict:
    """Compute hard subset counts per split."""
    stats = {}
    for tag in ["multi_entity", "conflict", "ambiguous"]:
        mask = df["hard_subset"].str.contains(tag, na=False)
        stats[tag] = int(mask.sum())
    stats["any_hard"] = int(df["hard_subset"].ne("").sum())
    return stats


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Stage 3: Data Freeze & Split")
    print("=" * 60)

    # ---- Load -----------------------------------------------
    print(f"\n[Load] Reading {IN_FILE}...")
    df = pd.read_json(IN_FILE, lines=True)
    print(f"  Total rows: {len(df)}")

    # ---- Separate market_aux --------------------------------
    sem_df      = df[df["task_type"] != "market_aux"].copy()
    analysis_df = df[df["task_type"] == "market_aux"].copy()
    print(f"  Semantic rows (for split): {len(sem_df)}")
    print(f"  Analysis rows (market_aux): {len(analysis_df)}")

    # ---- Hard subset tagging (on semantic rows) -------------
    print("\n[HardSubset] Detecting hard subsets...")
    sem_df["hard_subset"] = detect_hard_subsets(sem_df)
    analysis_df["hard_subset"] = ""

    hs_total = sem_df["hard_subset"].ne("").sum()
    print(f"  multi_entity: {sem_df['hard_subset'].str.contains('multi_entity', na=False).sum()}")
    print(f"  conflict:     {sem_df['hard_subset'].str.contains('conflict', na=False).sum()}")
    print(f"  ambiguous:    {sem_df['hard_subset'].str.contains('ambiguous', na=False).sum()}")
    print(f"  Total hard subset rows: {hs_total}")

    # ---- Stratified split -----------------------------------
    print("\n[Split] Stratified split (70/15/15)...")
    train_df, val_df, test_df = stratified_split(sem_df, RANDOM_SEED)

    # Stamp split field
    train_df["split"]    = "train"
    val_df["split"]      = "val"
    test_df["split"]     = "test"
    analysis_df["split"] = "analysis"

    print(f"  train: {len(train_df)}")
    print(f"  val:   {len(val_df)}")
    print(f"  test:  {len(test_df)}")
    print(f"  analysis: {len(analysis_df)}")

    # Sanity: total should equal sem_df
    assert len(train_df) + len(val_df) + len(test_df) == len(sem_df), \
        "Split totals do not add up!"

    # ---- Bias checklist -------------------------------------
    checklist = build_bias_checklist(train_df, val_df, test_df, analysis_df)

    # Warn on leakage
    if checklist["cross_split_exact_leakage"]["total"] > 0:
        print("\n[WARN] Exact cross-split leakage detected!")
    if checklist["near_duplicate_leakage"]["near_dup_pairs_found"] > 0:
        print(f"\n[WARN] Near-duplicate leakage: "
              f"{checklist['near_duplicate_leakage']['near_dup_pairs_found']} pairs")

    # ---- Save split files -----------------------------------
    SCHEMA = ["uid", "dataset", "task_type", "text", "label",
              "target", "market_label", "split", "hard_subset"]

    train_df[SCHEMA].to_json(OUT_TRAIN,    orient="records", lines=True, force_ascii=False)
    val_df[SCHEMA].to_json(OUT_VAL,        orient="records", lines=True, force_ascii=False)
    test_df[SCHEMA].to_json(OUT_TEST,      orient="records", lines=True, force_ascii=False)
    analysis_df[SCHEMA].to_json(OUT_ANALYSIS, orient="records", lines=True, force_ascii=False)

    print(f"\n[OK] Saved: {OUT_TRAIN}, {OUT_VAL}, {OUT_TEST}, {OUT_ANALYSIS}")

    # ---- Build split_spec.json ------------------------------
    spec = {
        "version":     "1.0",
        "created_at":  "2026-04-02T21:12:00+08:00",
        "seed":        RANDOM_SEED,
        "ratios":      {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "source_file": IN_FILE,
        "source_rows": int(len(df)),
        "splits": {
            "train": {
                "file":  OUT_TRAIN,
                "rows":  int(len(train_df)),
                "md5":   md5_file(OUT_TRAIN),
                "hard_subsets": hard_subset_stats(train_df),
                "task_type_counts": train_df["task_type"].value_counts().to_dict(),
                "label_counts":     train_df[train_df["task_type"] != "market_aux"]["label"].value_counts().to_dict(),
            },
            "val": {
                "file":  OUT_VAL,
                "rows":  int(len(val_df)),
                "md5":   md5_file(OUT_VAL),
                "hard_subsets": hard_subset_stats(val_df),
                "task_type_counts": val_df["task_type"].value_counts().to_dict(),
                "label_counts":     val_df[val_df["task_type"] != "market_aux"]["label"].value_counts().to_dict(),
            },
            "test": {
                "file":  OUT_TEST,
                "rows":  int(len(test_df)),
                "md5":   md5_file(OUT_TEST),
                "hard_subsets": hard_subset_stats(test_df),
                "task_type_counts": test_df["task_type"].value_counts().to_dict(),
                "label_counts":     test_df[test_df["task_type"] != "market_aux"]["label"].value_counts().to_dict(),
            },
            "analysis": {
                "file":  OUT_ANALYSIS,
                "rows":  int(len(analysis_df)),
                "md5":   md5_file(OUT_ANALYSIS),
                "note":  "market_aux rows excluded from supervised evaluation",
            },
        },
        "hard_subset_definitions": {
            "multi_entity": (
                "target_semantic rows where the same text shares >=2 distinct target entities "
                "in the corpus. Tests ability to handle multi-entity documents."
            ),
            "conflict": (
                "(text, target) pairs that appear with >=2 distinct sentiment labels "
                "across the corpus. Reflects genuine annotator/source disagreement."
            ),
            "ambiguous": (
                "sentence_semantic rows where the same text text appears with >=2 distinct "
                "sentiment labels in the corpus (FPB boundary agreement cases)."
            ),
        },
        "bias_checklist": checklist,
    }

    with open(OUT_SPEC, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False, default=str)
    print(f"[OK] Saved: {OUT_SPEC}")

    # ---- Summary --------------------------------------------
    print("\n" + "=" * 60)
    print("Split Summary")
    print("=" * 60)
    for sname, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        sem_rows = sdf[sdf["task_type"] != "market_aux"]
        print(f"\n  [{sname}] {len(sdf)} rows")
        print(f"    Label dist: {sem_rows['label'].value_counts().to_dict()}")
        print(f"    Hard rows:  {hard_subset_stats(sdf)}")

    print("\n[DONE] make_splits.py complete.")
    return spec


if __name__ == "__main__":
    main()
