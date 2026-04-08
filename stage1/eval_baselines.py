"""
eval_baselines.py
-----------------
Unified evaluation and leaderboard generator.
Reads ALL results.json files from a checkpoints directory tree,
computes consistent metrics, and outputs the final baseline leaderboard.

This is the single source of truth for all comparisons.
Run this AFTER all training scripts complete.

Usage:
    python eval_baselines.py \
        --results_dir checkpoints/ \
        --output_dir reports/ \
        --test_split data/processed/test.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from schema import (
    PredictionRecord,
    load_predictions,
)
from utils import compute_metrics, compute_subset_metrics


# ------------------------------------------------------------------
# Leaderboard row schema
# ------------------------------------------------------------------

LEADERBOARD_COLUMNS = [
    "model_alias",
    "backbone",
    "input_mode",
    "task_type",
    "test_macro_f1",
    "subset/multi_entity",
    "subset/conflict",
    "subset/ambiguous",
    "n_multi_entity",
    "n_conflict",
    "n_ambiguous",
    "seed",
    "checkpoint_dir",
]


# ------------------------------------------------------------------
# Discovery: find all results.json files
# ------------------------------------------------------------------

def discover_result_dirs(results_dir: str) -> List[Path]:
    """
    Walk the results_dir tree and return all directories containing results.json.
    """
    root = Path(results_dir)
    dirs = sorted([p.parent for p in root.rglob("results.json")])
    print(f"[OK] Found {len(dirs)} result directories")
    for d in dirs:
        print(f"  {d}")
    return dirs


# ------------------------------------------------------------------
# Load results from a single checkpoint directory
# ------------------------------------------------------------------

def load_result(checkpoint_dir: Path) -> Optional[Dict]:
    results_file = checkpoint_dir / "results.json"
    pred_file    = checkpoint_dir / "test_predictions.jsonl"

    if not results_file.exists():
        print(f"[WARN] Missing results.json in {checkpoint_dir}")
        return None
    if not pred_file.exists():
        print(f"[WARN] Missing test_predictions.jsonl in {checkpoint_dir}")
        return None

    with open(results_file) as f:
        results = json.load(f)

    # Recompute subset metrics from predictions for consistency
    # (guards against results.json being stale after a re-run)
    records = load_predictions(str(pred_file))
    subset_metrics = compute_subset_metrics(records)

    # Also recompute overall from predictions (single source of truth)
    true_labels = [r.true_label for r in records]
    pred_labels = [r.pred_label for r in records]
    overall = compute_metrics(true_labels, pred_labels)

    row = {
        "model_alias":          results.get("model_alias", checkpoint_dir.name),
        "backbone":             results.get("backbone", "?"),
        "input_mode":           results.get("input_mode", "?"),
        "task_type":            results.get("task_type", "?"),
        "test_macro_f1":        overall["macro_f1"],
        "per_class":            overall["per_class"],
        "subset/multi_entity":  subset_metrics.get("multi_entity", {}).get("macro_f1"),
        "subset/conflict":      subset_metrics.get("conflict",      {}).get("macro_f1"),
        "subset/ambiguous":     subset_metrics.get("ambiguous",     {}).get("macro_f1"),
        "n_multi_entity":       subset_metrics.get("multi_entity", {}).get("n_samples", 0),
        "n_conflict":           subset_metrics.get("conflict",      {}).get("n_samples", 0),
        "n_ambiguous":          subset_metrics.get("ambiguous",     {}).get("n_samples", 0),
        "seed":                 results.get("seed", -1),
        "checkpoint_dir":       str(checkpoint_dir),
        # Extras for the detailed report
        "best_dev_macro_f1":    results.get("best_dev_macro_f1"),
        "best_epoch":           results.get("best_epoch"),
        "alpha":                results.get("alpha"),
        "beta":                 results.get("beta"),
    }
    return row


# ------------------------------------------------------------------
# Formatting
# ------------------------------------------------------------------

def format_f1(v) -> str:
    if v is None:
        return "  N/A  "
    return f"{v:.4f}"


def print_leaderboard(rows: List[Dict]) -> None:
    sorted_rows = sorted(rows, key=lambda r: r["test_macro_f1"], reverse=True)

    header = (
        f"{'Model Alias':<45} "
        f"{'Backbone':<22} "
        f"{'Mode':<8} "
        f"{'Task':<10} "
        f"{'Overall':>8} "
        f"{'MultiEnt':>9} "
        f"{'Conflict':>9} "
        f"{'Ambig':>7}"
    )
    sep = "-" * len(header)

    print("\n" + sep)
    print("BASELINE LEADERBOARD (sorted by test macro-F1)")
    print(sep)
    print(header)
    print(sep)

    for r in sorted_rows:
        print(
            f"{r['model_alias']:<45} "
            f"{r['backbone']:<22} "
            f"{r['input_mode']:<8} "
            f"{r['task_type']:<10} "
            f"{format_f1(r['test_macro_f1']):>8} "
            f"{format_f1(r['subset/multi_entity']):>9} "
            f"{format_f1(r['subset/conflict']):>9} "
            f"{format_f1(r['subset/ambiguous']):>7}"
        )

    print(sep)
    best = sorted_rows[0]
    print(f"[BEST] {best['model_alias']}  |  overall={best['test_macro_f1']:.4f}")
    print(sep + "\n")


# ------------------------------------------------------------------
# Backbone comparison table
# ------------------------------------------------------------------

def print_backbone_comparison(rows: List[Dict]) -> None:
    backbones = sorted(set(r["backbone"] for r in rows))
    print("\n[BACKBONE COMPARISON]")
    print(f"{'Backbone':<30} {'Best Overall F1':>16} {'Best Model':>45}")
    print("-" * 95)

    for bb in backbones:
        bb_rows = [r for r in rows if r["backbone"] == bb]
        best    = max(bb_rows, key=lambda r: r["test_macro_f1"])
        print(
            f"{bb:<30} "
            f"{best['test_macro_f1']:>16.4f} "
            f"{best['model_alias']:>45}"
        )


# ------------------------------------------------------------------
# Input mode comparison table
# ------------------------------------------------------------------

def print_input_mode_comparison(rows: List[Dict]) -> None:
    print("\n[INPUT MODE COMPARISON: marker vs concat]")
    print(f"{'Backbone':<25} {'Mode':<8} {'Overall':>8} {'MultiEnt':>9} {'Conflict':>9} {'Ambig':>7}")
    print("-" * 70)

    target_rows = [r for r in rows if r["task_type"] == "target"]
    for r in sorted(target_rows, key=lambda r: (r["backbone"], r["input_mode"])):
        print(
            f"{r['backbone']:<25} "
            f"{r['input_mode']:<8} "
            f"{format_f1(r['test_macro_f1']):>8} "
            f"{format_f1(r['subset/multi_entity']):>9} "
            f"{format_f1(r['subset/conflict']):>9} "
            f"{format_f1(r['subset/ambiguous']):>7}"
        )


# ------------------------------------------------------------------
# Multi-task sensitivity table
# ------------------------------------------------------------------

def print_multitask_sensitivity(rows: List[Dict]) -> None:
    mt_rows = [r for r in rows if r["task_type"] == "multitask"]
    if not mt_rows:
        return

    print("\n[MULTI-TASK LOSS WEIGHT SENSITIVITY]")
    print(f"{'Model Alias':<50} {'Alpha':>6} {'Beta':>6} {'Overall':>8} {'Conflict':>9}")
    print("-" * 85)

    for r in sorted(mt_rows, key=lambda r: r["test_macro_f1"], reverse=True):
        alpha = r.get("alpha", "?")
        beta  = r.get("beta", "?")
        print(
            f"{r['model_alias']:<50} "
            f"{str(alpha):>6} "
            f"{str(beta):>6} "
            f"{format_f1(r['test_macro_f1']):>8} "
            f"{format_f1(r['subset/conflict']):>9}"
        )


# ------------------------------------------------------------------
# Generate backbone selection recommendation
# ------------------------------------------------------------------

def generate_backbone_report(rows: List[Dict], output_path: Path) -> None:
    backbones = sorted(set(r["backbone"] for r in rows))
    report = {"backbone_analysis": []}

    for bb in backbones:
        bb_rows = [r for r in rows if r["backbone"] == bb]
        best    = max(bb_rows, key=lambda r: r["test_macro_f1"])
        report["backbone_analysis"].append({
            "backbone":              bb,
            "best_overall_f1":       best["test_macro_f1"],
            "best_model_alias":      best["model_alias"],
            "best_conflict_f1":      best["subset/conflict"],
            "best_multi_entity_f1":  best["subset/multi_entity"],
            "n_models_tested":       len(bb_rows),
        })

    # Sort by overall F1
    report["backbone_analysis"].sort(
        key=lambda x: x["best_overall_f1"], reverse=True
    )
    report["recommendation"] = report["backbone_analysis"][0]["backbone"]

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[OK] Backbone selection report written to {output_path}")
    print(f"[RECOMMEND] Use backbone: {report['recommendation']}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover all checkpoint directories
    checkpoint_dirs = discover_result_dirs(args.results_dir)
    if not checkpoint_dirs:
        print("[FAIL] No result directories found. Run training scripts first.")
        return

    # 2. Load all results
    rows = []
    for d in checkpoint_dirs:
        row = load_result(d)
        if row is not None:
            rows.append(row)

    if not rows:
        print("[FAIL] No valid results loaded.")
        return

    print(f"\n[OK] Loaded {len(rows)} model results")

    # 3. Print comparison tables
    print_leaderboard(rows)
    print_backbone_comparison(rows)
    print_input_mode_comparison(rows)
    print_multitask_sensitivity(rows)

    # 4. Save leaderboard JSON and CSV
    leaderboard_path = output_dir / "baseline_leaderboard.json"
    with open(leaderboard_path, "w") as f:
        json.dump(sorted(rows, key=lambda r: r["test_macro_f1"], reverse=True), f, indent=2)
    print(f"\n[OK] Leaderboard written to {leaderboard_path}")

    # CSV
    import csv
    csv_path = output_dir / "baseline_leaderboard.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEADERBOARD_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["test_macro_f1"], reverse=True):
            writer.writerow(row)
    print(f"[OK] CSV leaderboard written to {csv_path}")

    # 5. Backbone selection report
    generate_backbone_report(rows, output_dir / "backbone_selection_report.json")

    # 6. Hard subset report (separate file for 2/3/4 -- reference)
    hard_subset_rows = [
        {
            "model_alias":         r["model_alias"],
            "subset/multi_entity": r["subset/multi_entity"],
            "subset/conflict":     r["subset/conflict"],
            "subset/ambiguous":    r["subset/ambiguous"],
            "n_multi_entity":      r["n_multi_entity"],
            "n_conflict":          r["n_conflict"],
            "n_ambiguous":         r["n_ambiguous"],
        }
        for r in sorted(rows, key=lambda r: r["test_macro_f1"], reverse=True)
    ]
    with open(output_dir / "hard_subset_report.json", "w") as f:
        json.dump(hard_subset_rows, f, indent=2)
    print(f"[OK] Hard subset report written to {output_dir / 'hard_subset_report.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir",  default="checkpoints/",
                        help="Root directory to scan for results.json files")
    parser.add_argument("--output_dir",   default="reports/")
    parser.add_argument("--test_split",   default="data/processed/test.jsonl",
                        help="Test split path (for any ad-hoc re-evaluation)")
    args = parser.parse_args()
    main(args)
