"""
analyze_alignment.py
--------------------
Reads model prediction files (the *_with_market.jsonl variants which
include FinMarBa samples) and analyzes disagreement between semantic
sentiment predictions and market labels.

Produces:
  - disagreement_report.jsonl : all disagreement cases
  - alignment_summary.json    : aggregated stats by subset and policy

Usage:
    python analyze_alignment.py \
        --predictions_dir F:\\stage4\\checkpoints\\retrieval_model \
        --data_dir        F:\\stage4\\data \
        --output_dir      F:\\stage4\\reports \
        --policy          always_on
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional


# ------------------------------------------------------------------
# Load market labels from raw data
# ------------------------------------------------------------------

def load_market_labels(data_dir: str) -> Dict[str, Optional[str]]:
    market_map = {}
    for split in ("train", "val", "test"):
        path = Path(data_dir) / f"{split}.jsonl"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d   = json.loads(line)
                uid = d.get("uid") or d.get("sample_id")
                if uid:
                    market_map[uid] = d.get("market_label")
    print(f"[OK] Loaded market labels for {len(market_map)} samples")
    return market_map


# ------------------------------------------------------------------
# Load predictions
# ------------------------------------------------------------------

def load_predictions(path: str) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ------------------------------------------------------------------
# Disagreement analysis
# ------------------------------------------------------------------

def analyze(
    records:    List[dict],
    market_map: Dict[str, Optional[str]],
) -> tuple:
    disagreements  = []
    total          = 0
    has_market     = 0
    disagree_count = 0
    subset_stats: Dict[str, Counter] = defaultdict(Counter)

    for r in records:
        total  += 1
        sid     = r["sample_id"]
        market  = market_map.get(sid)

        if market is None:
            continue
        has_market += 1

        semantic    = r["pred_label"]
        is_disagree = (semantic != market)
        tags        = r.get("subset_tags", []) or []

        for tag in (tags if tags else ["none"]):
            subset_stats[tag]["total"] += 1
            if is_disagree:
                subset_stats[tag]["disagree"] += 1

        if is_disagree:
            disagree_count += 1
            disagreements.append({
                "sample_id":     sid,
                "text":          r["text"],
                "target":        r.get("target"),
                "true_label":    r["true_label"],
                "semantic_pred": semantic,
                "market_label":  market,
                "subset_tags":   tags,
                "pred_probs":    r.get("pred_probs"),
            })

    stats = {
        "total_samples":       total,
        "samples_with_market": has_market,
        "disagreement_count":  disagree_count,
        "disagreement_rate":   round(disagree_count / has_market, 4)
                               if has_market > 0 else 0.0,
        "by_subset": {
            tag: {
                "total":         counts["total"],
                "disagree":      counts["disagree"],
                "disagree_rate": round(counts["disagree"] / counts["total"], 4)
                                 if counts["total"] > 0 else 0.0,
            }
            for tag, counts in subset_stats.items()
        },
    }
    return disagreements, stats


def analyze_directions(disagreements: List[dict]) -> Dict:
    c: Counter = Counter()
    for d in disagreements:
        key = f"semantic={d['semantic_pred']} | market={d['market_label']}"
        c[key] += 1
    return dict(c.most_common())


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prefer the *_with_market.jsonl file which includes FinMarBa samples
    pred_dir      = Path(args.predictions_dir)
    market_file   = pred_dir / f"test_predictions_{args.policy}_with_market.jsonl"
    fallback_file = pred_dir / f"test_predictions_{args.policy}.jsonl"

    if market_file.exists():
        pred_file = market_file
        print(f"[INFO] Using market-enriched predictions: {pred_file.name}")
    elif fallback_file.exists():
        pred_file = fallback_file
        print(f"[WARN] _with_market file not found, using: {pred_file.name}")
        print(f"       Re-run train_with_retrieval.py --stage train to generate market file.")
    else:
        print(f"[FAIL] No prediction file found for policy '{args.policy}' in {pred_dir}")
        return

    records    = load_predictions(str(pred_file))
    market_map = load_market_labels(args.data_dir)

    print(f"[OK] Loaded {len(records)} prediction records")

    disagreements, stats = analyze(records, market_map)
    directions           = analyze_directions(disagreements)

    # Print summary
    print(f"\n[ALIGNMENT ANALYSIS SUMMARY]")
    print(f"  Total samples         : {stats['total_samples']}")
    print(f"  Samples with market   : {stats['samples_with_market']}")
    print(f"  Disagreement count    : {stats['disagreement_count']}")
    if stats['samples_with_market'] == 0:
        print(f"  [WARN] No samples with market labels found.")
        print(f"         Ensure FinMarBa samples are in your data splits")
        print(f"         and that train_with_retrieval.py generated *_with_market.jsonl files.")
    else:
        print(f"  Disagreement rate     : {stats['disagreement_rate']:.2%}")

    print(f"\n  [By subset]")
    for tag, s in stats["by_subset"].items():
        print(f"    {tag:<16}: {s['disagree']}/{s['total']}  rate={s['disagree_rate']:.2%}")

    print(f"\n  [Disagreement directions]")
    for direction, count in directions.items():
        print(f"    {direction}: {count}")

    # Save outputs
    report_path = output_dir / "disagreement_report.jsonl"
    with open(report_path, "w", encoding="utf-8") as f:
        for d in disagreements:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"\n[OK] Disagreement report -> {report_path}")

    summary = {
        "policy":                    args.policy,
        "predictions_file":          str(pred_file),
        "stats":                     stats,
        "disagreement_directions":   directions,
    }
    summary_path = output_dir / "alignment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Alignment summary  -> {summary_path}")

    print(f"\n[TOP 5 DISAGREEMENT CASES]")
    for i, d in enumerate(disagreements[:5]):
        print(f"\n  Case {i+1}:")
        print(f"    Target  : {d['target']}")
        print(f"    Text    : {d['text'][:100]}...")
        print(f"    Semantic: {d['semantic_pred']}  |  Market: {d['market_label']}")
        print(f"    Subsets : {d['subset_tags']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir",
                        default="F:\\stage4\\checkpoints\\retrieval_model")
    parser.add_argument("--data_dir",    default="F:\\stage4\\data")
    parser.add_argument("--output_dir",  default="F:\\stage4\\reports")
    parser.add_argument("--policy",
                        choices=["none", "always_on", "conditional"],
                        default="always_on")
    args = parser.parse_args()
    main(args)
