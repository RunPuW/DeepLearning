"""
run_eval.py
===========
Unified evaluation script for FinSentMulti-v1.

Usage
-----
  # Standard evaluation
  python run_eval.py \\
      --predictions predictions.jsonl \\
      --data_file   test.jsonl \\
      --split_spec  split_spec.json \\
      --output_dir  eval_results/

  # Self-test mode (generate dummy predictions internally)
  python run_eval.py --self_test perfect --data_file test.jsonl --split_spec split_spec.json
  python run_eval.py --self_test random  --data_file test.jsonl --split_spec split_spec.json

Prediction format (JSONL, one per line)
---------------------------------------
  {"uid": "fpb_00001", "pred": "positive"}

Outputs
-------
  eval_results/metrics.json           – all scalar metrics
  eval_results/metrics_by_dataset.csv – per-dataset breakdown
  eval_results/confusion_matrix.csv   – 3x3 confusion matrix
  eval_results/hard_subset_metrics.json – hard subset breakdown
  eval_results/eval_report.md         – human-readable summary
"""

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict

import pandas as pd

# ============================================================
# Metric helpers
# ============================================================

LABELS = ["positive", "neutral", "negative"]


def _confusion_matrix(y_true, y_pred, labels=LABELS):
    """Return dict[actual][predicted] = count."""
    cm = {l: {p: 0 for p in labels} for l in labels}
    for yt, yp in zip(y_true, y_pred):
        if yt in cm and yp in cm:
            cm[yt][yp] += 1
    return cm


def _f1_from_cm(cm, label):
    tp = cm[label][label]
    fp = sum(cm[other][label] for other in LABELS if other != label)
    fn = sum(cm[label][other] for other in LABELS if other != label)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}


def _accuracy(y_true, y_pred):
    if not y_true:
        return 0.0
    return sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)


def _mcc(cm, labels=LABELS):
    """Matthews Correlation Coefficient for multiclass via macro approach."""
    # Use the standard formula from the confusion matrix
    n = sum(cm[a][p] for a in labels for p in labels)
    if n == 0:
        return 0.0
    # Compute per-class MCC-like components, then average
    mccs = []
    for k in labels:
        tp = cm[k][k]
        fp = sum(cm[other][k] for other in labels if other != k)
        fn = sum(cm[k][other] for other in labels if other != k)
        tn = n - tp - fp - fn
        denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mccs.append((tp * tn - fp * fn) / denom if denom > 0 else 0.0)
    return sum(mccs) / len(mccs)


def compute_metrics(y_true, y_pred, labels=LABELS):
    """
    Returns a dict of metrics:
        accuracy, mcc, macro_f1,
        per_class: {label: {precision, recall, f1, support}}
    """
    cm = _confusion_matrix(y_true, y_pred, labels)
    per_class = {lbl: _f1_from_cm(cm, lbl) for lbl in labels}
    macro_f1  = sum(per_class[l]["f1"] for l in labels) / len(labels)
    return {
        "n":          len(y_true),
        "accuracy":   round(_accuracy(y_true, y_pred), 4),
        "macro_f1":   round(macro_f1, 4),
        "mcc":        round(_mcc(cm, labels), 4),
        "per_class":  {l: {k: round(v, 4) for k, v in d.items()}
                       for l, d in per_class.items()},
        "confusion_matrix": cm,
    }


# ============================================================
# Data loading
# ============================================================

def load_data(data_file: str, split_spec_file: str) -> pd.DataFrame:
    """Load test data and optionally verify MD5."""
    print(f"[Load] Reading {data_file}...")
    df = pd.read_json(data_file, lines=True)

    # Verify MD5 if split_spec is available
    if split_spec_file and os.path.exists(split_spec_file):
        with open(split_spec_file, encoding="utf-8") as f:
            spec = json.load(f)

        # Determine which split this file corresponds to by name
        fname = os.path.basename(data_file)
        split_name = fname.replace(".jsonl", "")
        if split_name in spec.get("splits", {}):
            expected_md5 = spec["splits"][split_name]["md5"]
            actual_md5   = _md5_file(data_file)
            if actual_md5 != expected_md5:
                print(f"[WARN] MD5 mismatch for {data_file}!")
                print(f"       Expected: {expected_md5}")
                print(f"       Actual:   {actual_md5}")
                print("       Proceeding, but data integrity is not guaranteed.")
            else:
                print(f"[OK] MD5 verified: {actual_md5}")
        else:
            print(f"[INFO] Split '{split_name}' not found in split_spec.json. MD5 not verified.")

    print(f"  Rows: {len(df)}")
    return df


def _md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_predictions(pred_file: str) -> dict:
    """Load predictions JSONL -> dict[uid -> pred]."""
    print(f"[Load] Reading predictions from {pred_file}...")
    preds = {}
    with open(pred_file, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {line_no}: JSON parse error: {e}")
                continue
            uid  = obj.get("uid")
            pred = obj.get("pred")
            if not uid:
                print(f"[WARN] Line {line_no}: missing 'uid' field")
                continue
            if pred not in LABELS:
                print(f"[WARN] Line {line_no}: invalid pred '{pred}' for uid '{uid}'. Skipping.")
                continue
            preds[uid] = pred
    print(f"  Predictions loaded: {len(preds)}")
    return preds


# ============================================================
# Self-test prediction generators
# ============================================================

def generate_perfect_predictions(df: pd.DataFrame) -> dict:
    """Return predictions that exactly match ground truth labels."""
    return {row["uid"]: row["label"] for _, row in df.iterrows()
            if row["label"] in LABELS}


def generate_random_predictions(df: pd.DataFrame, seed: int = 42) -> dict:
    """Return uniformly random predictions."""
    rng = random.Random(seed)
    return {row["uid"]: rng.choice(LABELS)
            for _, row in df.iterrows()
            if row["label"] in LABELS}


def generate_majority_predictions(df: pd.DataFrame) -> dict:
    """Return predictions that always predict the majority class."""
    sem = df[df["label"].isin(LABELS)]
    majority = sem["label"].value_counts().index[0]
    return {row["uid"]: majority for _, row in df.iterrows()
            if row["label"] in LABELS}


# ============================================================
# Evaluation core
# ============================================================

def evaluate(df: pd.DataFrame, preds: dict) -> dict:
    """
    Run full evaluation suite on df (test split rows).
    df must have columns: uid, label, dataset, task_type, hard_subset.
    preds is dict[uid -> pred_label].

    Returns nested dict of all results.
    """
    # Filter to semantic rows with valid labels
    sem = df[df["label"].isin(LABELS)].copy()

    # Coverage check
    missing_uids = [uid for uid in sem["uid"] if uid not in preds]
    extra_uids   = [uid for uid in preds if uid not in set(sem["uid"])]
    if missing_uids:
        print(f"[ERROR] {len(missing_uids)} test UIDs have no prediction!")
        print(f"  Sample: {missing_uids[:5]}")
        sys.exit(1)
    if extra_uids:
        print(f"[WARN] {len(extra_uids)} prediction UIDs not in test set (ignored)")

    sem = sem.copy()
    sem["pred"] = sem["uid"].map(preds)

    y_true_all = sem["label"].tolist()
    y_pred_all = sem["pred"].tolist()

    # --- Overall metrics ---
    overall = compute_metrics(y_true_all, y_pred_all)
    print(f"\n[Overall] n={overall['n']}, "
          f"macro_F1={overall['macro_f1']:.4f}, "
          f"accuracy={overall['accuracy']:.4f}, "
          f"MCC={overall['mcc']:.4f}")

    # --- Per-task metrics ---
    per_task = {}
    for task in sem["task_type"].unique():
        sub = sem[sem["task_type"] == task]
        per_task[task] = compute_metrics(sub["label"].tolist(), sub["pred"].tolist())
        print(f"  [{task}] n={per_task[task]['n']}, "
              f"macro_F1={per_task[task]['macro_f1']:.4f}")

    # --- Per-dataset metrics ---
    per_dataset = {}
    for ds in sem["dataset"].unique():
        sub = sem[sem["dataset"] == ds]
        per_dataset[ds] = compute_metrics(sub["label"].tolist(), sub["pred"].tolist())
        print(f"  [{ds}] n={per_dataset[ds]['n']}, "
              f"macro_F1={per_dataset[ds]['macro_f1']:.4f}")

    # --- Hard subset metrics ---
    hard_subsets = {}
    for tag in ["multi_entity", "conflict", "ambiguous"]:
        if "hard_subset" not in sem.columns:
            break
        sub = sem[sem["hard_subset"].str.contains(tag, na=False)]
        if len(sub) == 0:
            hard_subsets[tag] = {"n": 0, "note": "no rows with this tag in split"}
        else:
            hard_subsets[tag] = compute_metrics(
                sub["label"].tolist(), sub["pred"].tolist()
            )
            print(f"  [hard:{tag}] n={hard_subsets[tag]['n']}, "
                  f"macro_F1={hard_subsets[tag].get('macro_f1', 'N/A')}")

    return {
        "overall":      overall,
        "per_task":     per_task,
        "per_dataset":  per_dataset,
        "hard_subsets": hard_subsets,
        "coverage": {
            "test_rows":     int(len(sem)),
            "missing_preds": int(len(missing_uids)),
            "extra_preds":   int(len(extra_uids)),
        },
    }


# ============================================================
# Report writers
# ============================================================

def write_metrics_json(results: dict, out_dir: str):
    path = os.path.join(out_dir, "metrics.json")

    def _clean(obj):
        """Recursively convert sets/non-serializable to serializable."""
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, float):
            return round(obj, 6)
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(results), f, indent=2, ensure_ascii=False)
    print(f"[OK] {path}")


def write_metrics_by_dataset_csv(results: dict, out_dir: str):
    rows = []
    for ds, m in results["per_dataset"].items():
        row = {
            "dataset":    ds,
            "n":          m["n"],
            "macro_f1":   m["macro_f1"],
            "accuracy":   m["accuracy"],
            "mcc":        m["mcc"],
        }
        for lbl in LABELS:
            row[f"f1_{lbl}"]        = m["per_class"][lbl]["f1"]
            row[f"precision_{lbl}"] = m["per_class"][lbl]["precision"]
            row[f"recall_{lbl}"]    = m["per_class"][lbl]["recall"]
            row[f"support_{lbl}"]   = m["per_class"][lbl]["support"]
        rows.append(row)

    # Also add per-task rows
    for task, m in results["per_task"].items():
        row = {
            "dataset":    f"[task] {task}",
            "n":          m["n"],
            "macro_f1":   m["macro_f1"],
            "accuracy":   m["accuracy"],
            "mcc":        m["mcc"],
        }
        for lbl in LABELS:
            row[f"f1_{lbl}"]        = m["per_class"][lbl]["f1"]
            row[f"precision_{lbl}"] = m["per_class"][lbl]["precision"]
            row[f"recall_{lbl}"]    = m["per_class"][lbl]["recall"]
            row[f"support_{lbl}"]   = m["per_class"][lbl]["support"]
        rows.append(row)

    path = os.path.join(out_dir, "metrics_by_dataset.csv")
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.4f")
    print(f"[OK] {path}")


def write_confusion_matrix_csv(results: dict, out_dir: str):
    cm = results["overall"]["confusion_matrix"]
    rows = []
    for actual in LABELS:
        row = {"actual \\ predicted": actual}
        for pred in LABELS:
            row[pred] = cm[actual][pred]
        rows.append(row)
    path = os.path.join(out_dir, "confusion_matrix.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[OK] {path}")


def write_hard_subset_metrics_json(results: dict, out_dir: str):
    path = os.path.join(out_dir, "hard_subset_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results["hard_subsets"], f, indent=2, ensure_ascii=False)
    print(f"[OK] {path}")


def write_eval_report_md(results: dict, out_dir: str,
                          pred_file: str, data_file: str):
    ov = results["overall"]
    pc = ov["per_class"]
    cm = ov["confusion_matrix"]

    lines = [
        "# Evaluation Report — FinSentMulti-v1",
        "",
        f"- **Data file:** `{os.path.basename(data_file)}`",
        f"- **Predictions:** `{os.path.basename(pred_file)}`",
        "",
        "---",
        "",
        "## Overall Metrics",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| **Macro-F1** | **{ov['macro_f1']:.4f}** |",
        f"| Accuracy     | {ov['accuracy']:.4f} |",
        f"| MCC          | {ov['mcc']:.4f} |",
        f"| N (test rows)| {ov['n']} |",
        "",
        "## Per-Class F1",
        "",
        "| Class    | Precision | Recall | F1     | Support |",
        "|---|---|---|---|---|",
    ]
    for lbl in LABELS:
        d = pc[lbl]
        lines.append(
            f"| {lbl:8s} | {d['precision']:.4f}    | {d['recall']:.4f} "
            f"| {d['f1']:.4f} | {d['support']:7d} |"
        )

    lines += [
        "",
        "## Confusion Matrix",
        "",
        "| actual \\ pred | positive | neutral | negative |",
        "|---|---|---|---|",
    ]
    for actual in LABELS:
        lines.append(
            f"| {actual:8s}      | "
            + " | ".join(str(cm[actual][p]) for p in LABELS)
            + " |"
        )

    lines += ["", "## Per-Dataset Metrics", ""]
    lines.append("| Dataset | N | Macro-F1 | F1-pos | F1-neu | F1-neg |")
    lines.append("|---|---|---|---|---|---|")
    for ds, m in sorted(results["per_dataset"].items()):
        lines.append(
            f"| {ds} | {m['n']} | {m['macro_f1']:.4f} | "
            f"{m['per_class']['positive']['f1']:.4f} | "
            f"{m['per_class']['neutral']['f1']:.4f} | "
            f"{m['per_class']['negative']['f1']:.4f} |"
        )

    lines += ["", "## Hard Subset Metrics", ""]
    lines.append("| Subset | N | Macro-F1 | F1-pos | F1-neu | F1-neg |")
    lines.append("|---|---|---|---|---|---|")
    for tag, m in results["hard_subsets"].items():
        if m.get("n", 0) == 0:
            lines.append(f"| {tag} | 0 | — | — | — | — |")
        else:
            lines.append(
                f"| {tag} | {m['n']} | {m.get('macro_f1', 0):.4f} | "
                f"{m['per_class']['positive']['f1']:.4f} | "
                f"{m['per_class']['neutral']['f1']:.4f} | "
                f"{m['per_class']['negative']['f1']:.4f} |"
            )

    path = os.path.join(out_dir, "eval_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] {path}")


# ============================================================
# Main / CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="FinSentMulti-v1 unified evaluation script"
    )
    p.add_argument("--predictions", default=None,
                   help="Path to predictions JSONL")
    p.add_argument("--data_file", required=True,
                   help="Path to test JSONL (e.g. test.jsonl)")
    p.add_argument("--split_spec", default="split_spec.json",
                   help="Path to split_spec.json for MD5 verification")
    p.add_argument("--output_dir", default="eval_results",
                   help="Directory to write all output files")
    p.add_argument("--self_test", choices=["perfect", "random", "majority"],
                   default=None,
                   help="Generate internal dummy predictions for self-testing")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("FinSentMulti-v1 Evaluation")
    print("=" * 60)

    # Load test data
    df = load_data(args.data_file, args.split_spec)

    # Load or generate predictions
    if args.self_test:
        print(f"\n[SelfTest] Generating '{args.self_test}' predictions...")
        if args.self_test == "perfect":
            preds = generate_perfect_predictions(df)
        elif args.self_test == "random":
            preds = generate_random_predictions(df)
        else:
            preds = generate_majority_predictions(df)

        # Save generated predictions for reproducibility
        pred_file = os.path.join(args.output_dir,
                                  f"self_test_{args.self_test}_predictions.jsonl")
        with open(pred_file, "w", encoding="utf-8") as f:
            for uid, pred in preds.items():
                f.write(json.dumps({"uid": uid, "pred": pred}) + "\n")
        print(f"[OK] Self-test predictions saved: {pred_file}")
        display_pred_file = pred_file
    else:
        if not args.predictions:
            print("[ERROR] --predictions is required unless --self_test is specified")
            sys.exit(1)
        preds         = load_predictions(args.predictions)
        display_pred_file = args.predictions

    # Run evaluation
    print("\n[Eval] Computing metrics...")
    results = evaluate(df, preds)

    # Write all outputs
    print(f"\n[Write] Saving outputs to {args.output_dir}/...")
    write_metrics_json(results, args.output_dir)
    write_metrics_by_dataset_csv(results, args.output_dir)
    write_confusion_matrix_csv(results, args.output_dir)
    write_hard_subset_metrics_json(results, args.output_dir)
    write_eval_report_md(results, args.output_dir,
                          display_pred_file, args.data_file)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    ov = results["overall"]
    print(f"  N (test semantic rows): {ov['n']}")
    print(f"  Macro-F1:  {ov['macro_f1']:.4f}")
    print(f"  Accuracy:  {ov['accuracy']:.4f}")
    print(f"  MCC:       {ov['mcc']:.4f}")
    print(f"  F1(pos):   {ov['per_class']['positive']['f1']:.4f}")
    print(f"  F1(neu):   {ov['per_class']['neutral']['f1']:.4f}")
    print(f"  F1(neg):   {ov['per_class']['negative']['f1']:.4f}")
    print(f"\n  Output dir: {os.path.abspath(args.output_dir)}/")
    print("[DONE]")


if __name__ == "__main__":
    main()
