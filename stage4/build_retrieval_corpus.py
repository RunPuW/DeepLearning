"""
build_retrieval_corpus.py
-------------------------
Reads train/val/test.jsonl to collect unique target entities,
then scans fnspid_sample_50k.jsonl to extract matching financial
news articles. Outputs retrieval_corpus.jsonl for indexing.

doc_type heuristic:
  headline : text token count <= 25 (short, title-style)
  article  : text token count >  25 (full news body)

Usage:
    python build_retrieval_corpus.py \
        --data_dir   F:\\stage1\\data \
        --fnspid     F:\\stage1\\data\\fnspid_sample_50k.jsonl \
        --output     F:\\stage1\\retrieval_corpus.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Set


# ------------------------------------------------------------------
# Step 1: collect unique targets from all splits
# ------------------------------------------------------------------

def collect_targets(data_dir: str) -> Dict[str, str]:
    """
    Returns {target_lower: target_original} from train/val/test splits.
    """
    targets: Dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = Path(data_dir) / f"{split}.jsonl"
        if not path.exists():
            print(f"[WARN] {path} not found, skipping")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                raw = d.get("target")
                if raw and isinstance(raw, str):
                    t = raw.strip()
                    if t:
                        targets[t.lower()] = t
        print(f"[OK] Loaded targets from {split}.jsonl")

    print(f"[OK] {len(targets)} unique target entities collected")
    return targets


# ------------------------------------------------------------------
# Step 2: classify doc_type by text length
# ------------------------------------------------------------------

def classify_doc_type(text: str) -> str:
    token_count = len(text.split())
    if token_count <= 25:
        return "headline"
    return "article"


# ------------------------------------------------------------------
# Step 3: scan FNSPID and build corpus
# ------------------------------------------------------------------

def build_corpus(fnspid_path: str, targets: Dict[str, str], output_path: str) -> int:
    """
    Scans fnspid_sample_50k.jsonl line by line.
    For each article, checks if any target entity appears in text+ticker.
    Writes matching articles to output_path.
    Returns count of matched documents.
    """
    if not Path(fnspid_path).exists():
        raise FileNotFoundError(f"[FAIL] FNSPID file not found: {fnspid_path}")

    match_count = 0
    scan_count = 0

    with open(fnspid_path, encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue
            scan_count += 1

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            text   = item.get("text", "")
            ticker = str(item.get("ticker", ""))
            if not text:
                continue

            search_scope = (text + " " + ticker).lower()

            matched_target = None
            for target_lower, target_original in targets.items():
                if target_lower in search_scope:
                    matched_target = target_original
                    break

            if matched_target is None:
                continue

            doc = {
                "doc_id":       f"fnspid_{match_count}",
                "entity":       matched_target,
                "context":      text.strip()[:2000],
                "doc_type":     classify_doc_type(text),
                "ticker":       ticker,
                "published_at": item.get("published_at", ""),
                "source":       item.get("source", ""),
                "url":          item.get("url", ""),
            }
            f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            match_count += 1

            if scan_count % 10000 == 0:
                print(f"  scanned {scan_count} | matched {match_count}")

    return match_count


# ------------------------------------------------------------------
# Step 4: print corpus stats
# ------------------------------------------------------------------

def print_corpus_stats(output_path: str) -> None:
    from collections import Counter
    doc_types: Counter = Counter()
    entities: Counter  = Counter()

    with open(output_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            doc_types[d["doc_type"]] += 1
            entities[d["entity"]]    += 1

    print(f"\n[CORPUS STATS]")
    print(f"  Total documents : {sum(doc_types.values())}")
    for dt, count in doc_types.most_common():
        print(f"  doc_type/{dt:<12}: {count}")
    print(f"  Unique entities : {len(entities)}")
    print(f"  Top 10 entities :")
    for entity, count in entities.most_common(10):
        print(f"    {entity:<30} {count}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(args):
    print("[INFO] Building retrieval corpus")

    targets = collect_targets(args.data_dir)
    if not targets:
        print("[FAIL] No targets found. Check data_dir.")
        return

    print(f"\n[INFO] Scanning {args.fnspid}")
    match_count = build_corpus(args.fnspid, targets, args.output)

    print(f"\n[OK] Matched {match_count} documents -> {args.output}")
    if match_count > 0:
        print_corpus_stats(args.output)
    else:
        print("[WARN] No documents matched. Check entity names vs FNSPID content.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="F:\\stage1\\data")
    parser.add_argument("--fnspid",   default="F:\\stage1\\data\\fnspid_sample_50k.jsonl")
    parser.add_argument("--output",   default="F:\\stage1\\retrieval_corpus.jsonl")
    args = parser.parse_args()
    main(args)
