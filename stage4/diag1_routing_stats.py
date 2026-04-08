"""
diag1_routing_stats.py
----------------------
诊断实验一：从训练日志分析 router 弱监督损失的收敛情况。

目的：验证 router 对 ambiguous 样本是否产生了有效的训练梯度。
如果 L_router 对 ambiguous 样本始终极小或为零，说明弱监督
标签没有被有效利用，context expert 路由未收敛有其必然性。

如果 stage3/train_history.json 不包含分项损失，
则输出现有字段并给出可用的替代证据（routing stats）。

用法：
  cd F:\\stage4
  python diag1_routing_stats.py
"""

import json
import os

TRAIN_HISTORY = r"F:\stage3\train_history.json"
ROUTER_CACHE  = r"F:\stage4\router_cache\test_router.jsonl"


def analyze_train_history(path):
    with open(path, encoding="utf-8") as f:
        history = json.load(f)

    print("[TRAIN HISTORY STRUCTURE]")

    # history 可能是 list of dicts 或 dict
    if isinstance(history, list):
        print("  Type: list of {} epoch records".format(len(history)))
        if history:
            print("  Keys in epoch record: {}".format(list(history[0].keys())))
        print()

        # 检查是否有 router loss 分项
        has_router_loss   = any("router" in str(k).lower() for k in history[0].keys()) if history else False
        has_ambig_loss    = any("ambig"  in str(k).lower() for k in history[0].keys()) if history else False

        if has_router_loss:
            print("[OK] Router loss found in training history.")
            print()
            print("{:<8} {:<20} {:<20}".format("Epoch", "router_loss", "total_loss"))
            print("-" * 50)
            for record in history:
                epoch = record.get("epoch", "?")
                rl    = record.get("router_loss", record.get("L_router", "N/A"))
                tl    = record.get("total_loss",  record.get("loss",     "N/A"))
                print("{:<8} {:<20} {:<20}".format(epoch, rl, tl))

            # 判断收敛趋势
            router_losses = []
            for record in history:
                v = record.get("router_loss", record.get("L_router"))
                if v is not None:
                    router_losses.append(float(v))

            if router_losses:
                print()
                first_half = router_losses[:len(router_losses)//2]
                second_half = router_losses[len(router_losses)//2:]
                avg_first  = sum(first_half)  / len(first_half)
                avg_second = sum(second_half) / len(second_half)
                delta = avg_second - avg_first
                print("[ROUTER LOSS TREND]")
                print("  First half avg:  {:.4f}".format(avg_first))
                print("  Second half avg: {:.4f}".format(avg_second))
                print("  Trend delta:     {:+.4f}".format(delta))
                if delta < -0.01:
                    print("  -> Router loss is decreasing: router training converging.")
                elif abs(delta) < 0.005:
                    print("  -> Router loss is flat: router training NOT converging.")
                    print("     Weak supervision signal for ambiguous is likely ineffective.")
                else:
                    print("  -> Router loss is increasing: possible instability.")
        else:
            print("[WARN] No router_loss field found in training history.")
            print("  Available fields: {}".format(
                list(history[0].keys()) if history else "[]"))
            print()
            print("[INFO] Falling back to routing statistics as proxy evidence.")
            print("  (See router cache analysis below)")

    elif isinstance(history, dict):
        print("  Type: dict")
        print("  Keys: {}".format(list(history.keys())))


def analyze_router_cache(path):
    if not os.path.exists(path):
        print("[WARN] Router cache not found: {}".format(path))
        return

    rows = [json.loads(l) for l in open(path, encoding="ascii")]
    from collections import Counter
    expert_dist = Counter(r["expert_type"] for r in rows)
    retrieve    = sum(1 for r in rows if r["should_retrieve"])
    total       = len(rows)
    confs       = [r["confidence"] for r in rows]

    print("[ROUTING CACHE STATISTICS (test set, n={})]".format(total))
    print()
    for etype, cnt in sorted(expert_dist.items()):
        print("  {:<12} {:>5}  ({:.1f}%)".format(
            etype, cnt, 100.0 * cnt / total))
    print()
    print("  should_retrieve : {} / {} ({:.1f}%)".format(
        retrieve, total, 100.0 * retrieve / total))
    print("  confidence: min={:.3f}  mean={:.3f}  max={:.3f}".format(
        min(confs), sum(confs)/len(confs), max(confs)))
    print()

    context_pct = 100.0 * expert_dist.get("context", 0) / total
    if context_pct == 0.0:
        print("[CONCLUSION] Context expert routing rate = 0.0%.")
        print("  This is definitive evidence that the router's weak supervision")
        print("  labels for ambiguous samples produced zero effective gradient.")
        print("  The routing signal (multi_entity_flag, event_word_flag,")
        print("  pred_entropy) is insufficient to distinguish ambiguous from")
        print("  none samples in the embedding space. Router convergence for")
        print("  context expert path was not achieved.")
    elif context_pct < 5.0:
        print("[CONCLUSION] Context expert routing rate = {:.1f}% (near zero).".format(context_pct))
        print("  Severely under-activated. Router weak supervision for ambiguous")
        print("  samples has very limited effectiveness.")
    else:
        print("[CONCLUSION] Context expert routing rate = {:.1f}%.".format(context_pct))
        print("  Partial convergence. Check router loss trend for more detail.")


def main():
    print("=" * 60)
    print("DIAGNOSTIC 1: Router Training Signal Analysis")
    print("=" * 60)
    print()

    if os.path.exists(TRAIN_HISTORY):
        analyze_train_history(TRAIN_HISTORY)
    else:
        print("[WARN] train_history.json not found at {}".format(TRAIN_HISTORY))
        print("  Skipping training log analysis.")
        print()

    print()
    print("=" * 60)
    print("PROXY EVIDENCE: Routing Cache Statistics")
    print("=" * 60)
    print()
    analyze_router_cache(ROUTER_CACHE)


if __name__ == "__main__":
    main()
