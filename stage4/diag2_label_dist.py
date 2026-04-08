"""
diag2_label_dist.py
-------------------
诊断实验二：ambiguous vs none 子集的 gold label 分布分析。

目的：验证 ambiguous 的规则代理定义是否具有统计区分度。
如果两个子集的 label 分布高度相似，说明 ambiguous 标签
本身没有捕捉到真正的"语义歧义"，router 无法学到有效的
路由信号是合理的——信号本来就不存在。

用法：
  cd F:\\stage4
  python diag2_label_dist.py
"""

import json
from collections import Counter
import math

PRED_FILE = r"F:\stage4\checkpoints\retrieval_model\test_predictions_none.jsonl"


def entropy(dist, total):
    """Shannon entropy of a label distribution."""
    h = 0.0
    for v in dist.values():
        p = v / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def js_divergence(dist1, total1, dist2, total2):
    """Jensen-Shannon divergence between two label distributions."""
    all_labels = set(dist1) | set(dist2)
    p = {l: dist1.get(l, 0) / total1 for l in all_labels}
    q = {l: dist2.get(l, 0) / total2 for l in all_labels}
    m = {l: 0.5 * (p[l] + q[l]) for l in all_labels}

    def kl(a, b):
        return sum(a[l] * math.log2(a[l] / b[l]) for l in all_labels if a[l] > 0)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    samples = [json.loads(l) for l in open(PRED_FILE, encoding="utf-8")]
    print("[OK] Loaded {} samples from {}".format(len(samples), PRED_FILE))
    print()

    # ── 子集划分 ──────────────────────────────────────────
    subsets = {
        "ambiguous": [],
        "conflict":  [],
        "multi_entity": [],
        "none":      [],
    }
    for s in samples:
        tags = s.get("subset_tags") or []
        placed = False
        for tag in ["conflict", "multi_entity", "ambiguous"]:
            if tag in tags:
                subsets[tag].append(s)
                placed = True
        if not placed:
            subsets["none"].append(s)

    # ── 逐子集统计 ────────────────────────────────────────
    print("[LABEL DISTRIBUTION BY SUBSET]")
    print("{:<15} {:>8} {:>12} {:>12} {:>12} {:>10} {:>8}".format(
        "Subset", "N", "positive", "neutral", "negative", "neutral%", "entropy"))
    print("-" * 80)

    dist_map = {}
    for tag in ["ambiguous", "none", "conflict", "multi_entity"]:
        rows  = subsets[tag]
        total = len(rows)
        if total == 0:
            continue
        labels = Counter(s["true_label"] for s in rows)
        pos = labels.get("positive", 0)
        neu = labels.get("neutral",  0)
        neg = labels.get("negative", 0)
        neu_ratio = neu / total
        h = entropy(labels, total)
        dist_map[tag] = (labels, total)
        print("{:<15} {:>8} {:>12.1f} {:>12.1f} {:>12.1f} {:>9.3f} {:>8.3f}".format(
            tag, total,
            100 * pos / total,
            100 * neu / total,
            100 * neg / total,
            neu_ratio, h))

    # ── ambiguous vs none 的 JS 散度 ─────────────────────
    print()
    if "ambiguous" in dist_map and "none" in dist_map:
        jsd = js_divergence(*dist_map["ambiguous"], *dist_map["none"])
        print("[JS DIVERGENCE] ambiguous vs none: {:.4f}".format(jsd))
        print()
        if jsd < 0.01:
            print("[CONCLUSION] JS divergence < 0.01: label distributions are")
            print("  virtually identical. The rule-based ambiguous proxy")
            print("  (short text / target absent / hedge words) does NOT")
            print("  capture a statistically distinct subset. This explains")
            print("  why the router cannot learn to distinguish ambiguous")
            print("  from none samples -- the signal does not exist in the")
            print("  label space.")
        elif jsd < 0.05:
            print("[CONCLUSION] JS divergence in [0.01, 0.05]: weak separation.")
            print("  The ambiguous proxy has limited discriminative power.")
            print("  Router training signal is insufficient for convergence.")
        else:
            print("[CONCLUSION] JS divergence >= 0.05: meaningful separation.")
            print("  The ambiguous proxy does capture a distinct distribution.")
            print("  Router failure is not caused by label indistinguishability.")

    # ── ambiguous 三子类细分 ──────────────────────────────
    print()
    print("[AMBIGUOUS SUBTYPE BREAKDOWN]")
    print("(Decomposing by which proxy rule triggered the tag)")
    print()

    ambig_samples = subsets["ambiguous"]

    short_text  = [s for s in ambig_samples
                   if len(s.get("text", "").split()) < 12]
    target_abs  = [s for s in ambig_samples
                   if s.get("target") and
                   s["target"].lower() not in s.get("text", "").lower()]
    # remaining are hedge / no anchor word (approximation)
    short_ids   = {id(s) for s in short_text}
    abs_ids     = {id(s) for s in target_abs}
    hedge_only  = [s for s in ambig_samples
                   if id(s) not in short_ids and id(s) not in abs_ids]

    for name, rows in [("short_text (<12 tok)", short_text),
                        ("target_absent",        target_abs),
                        ("hedge/no_anchor",       hedge_only)]:
        if not rows:
            continue
        total  = len(rows)
        labels = Counter(s["true_label"] for s in rows)
        acc    = sum(1 for s in rows if s["true_label"] == s["pred_label"])
        print("  {:<22} n={:>5}  acc={:.4f}  labels={}".format(
            name, total, acc / total, dict(labels)))


if __name__ == "__main__":
    main()
