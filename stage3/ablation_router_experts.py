"""
ablation_router_experts.py
消融实验脚本（分工2 验收核心）

验证 Router + Experts 各模块的必要性，回答四个关键问题：
  1. Router 是否真的在工作？
  2. Expert 是否真的有分工？
  3. 模型提升是不是主要来自 conflict / ambiguous 样本？
  4. 去掉某个 expert，性能会不会掉？

消融配置（共 5 种）：
  full        : 完整模型（Router + 3 Experts + L_div）
  no_router   : 均匀路由（α = [1/3, 1/3, 1/3]）
  no_conflict : 关闭 Conflict Expert（权重归零 + 重归一化）
  no_context  : 关闭 Context Expert（权重归零 + 重归一化）
  no_div      : 不使用 L_div（从 checkpoint 加载时不影响推理，
                记录 router 熵和 expert 多样性指标以间接验证）

用法（需要先训练好模型）：
  python ablation_router_experts.py \
      --checkpoint  checkpoints/router_experts/best_model.pt \
      --test_file   test.jsonl \
      --output_dir  ablation_results/ \
      --backbone    ProsusAI/finbert

输出：
  ablation_results/
    ablation_summary.json   : 所有配置的 Macro-F1 对比表
    ablation_report.md      : 可读版分析报告
    routing_analysis.json   : 路由分布和 expert 多样性诊断
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.experts import FinSentModel, LABEL2ID, ID2LABEL
from losses.diversity_loss import analyze_expert_diversity, build_subset_masks
from train_conflict_router import FinSentDataset, collate_fn, compute_macro_f1


# ─────────────────────────────────────────────────────────
# 消融配置定义
# ─────────────────────────────────────────────────────────
ABLATION_CONFIGS = {
    "full": {
        "desc":             "完整模型（Router + Base/Conflict/Context Expert + L_div）",
        "no_router":        False,
        "no_conflict":      False,
        "no_context":       False,
    },
    "no_router": {
        "desc":             "均匀路由消融（α = [1/3, 1/3, 1/3]，验证 Router 的必要性）",
        "no_router":        True,
        "no_conflict":      False,
        "no_context":       False,
    },
    "no_conflict_expert": {
        "desc":             "关闭 Conflict Expert（验证冲突专家对 multi_entity/conflict 样本的价值）",
        "no_router":        False,
        "no_conflict":      True,
        "no_context":       False,
    },
    "no_context_expert": {
        "desc":             "关闭 Context Expert（验证上下文专家对 ambiguous 样本的价值）",
        "no_router":        False,
        "no_conflict":      False,
        "no_context":       True,
    },
    "base_expert_only": {
        "desc":             "只用 Base Expert（关闭 Conflict + Context，等价于单专家 baseline）",
        "no_router":        False,
        "no_conflict":      True,
        "no_context":       True,
    },
}


# ─────────────────────────────────────────────────────────
# 单次评估
# ─────────────────────────────────────────────────────────
@torch.no_grad()
def run_ablation(
    model: FinSentModel,
    loader: DataLoader,
    device: torch.device,
    config_name: str,
    config: dict,
) -> dict:
    """
    运行单个消融配置的评估。

    Returns:
        完整评估结果字典，包含：
          - 整体 Macro-F1 / Accuracy
          - 每类 F1
          - Hard subset F1（multi_entity / conflict / ambiguous）
          - Task type F1（sentence_semantic / target_semantic）
          - 路由权重统计
          - Expert 多样性分析
    """
    model.eval()

    all_preds, all_labels = [], []
    all_task_types, all_hard_subsets = [], []
    routing_weights_all = []
    routing_entropy_all = []
    expert_reprs_all    = [[], [], []]     # base / conflict / context

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_expert_repr=True,
            ablation_no_router=config["no_router"],
            ablation_no_conflict=config["no_conflict"],
            ablation_no_context=config["no_context"],
        )

        preds = out["logits"].argmax(dim=-1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(batch["label"].tolist())
        all_task_types.extend(batch["task_type"])
        all_hard_subsets.extend(batch["hard_subset"])

        rw = out["routing_weights"].cpu()
        routing_weights_all.append(rw)

        entropy = -(rw * (rw + 1e-8).log()).sum(dim=-1)
        routing_entropy_all.append(entropy)

        if "expert_reprs" in out:
            for i, r in enumerate(out["expert_reprs"]):
                expert_reprs_all[i].append(r.cpu())

    # ── 整体指标 ──
    macro_f1 = compute_macro_f1(all_preds, all_labels)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    report = classification_report(
        all_labels, all_preds,
        target_names=["positive", "neutral", "negative"],
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2]).tolist()

    result = {
        "config_name": config_name,
        "desc":        config["desc"],
        "macro_f1":    macro_f1,
        "accuracy":    accuracy,
        "per_class_f1": {
            "positive": report["positive"]["f1-score"],
            "neutral":  report["neutral"]["f1-score"],
            "negative": report["negative"]["f1-score"],
        },
        "confusion_matrix": cm,
    }

    # ── Task type 分别统计 ──
    for tt in ("sentence_semantic", "target_semantic"):
        idx = [i for i, t in enumerate(all_task_types) if t == tt]
        if idx:
            pp = [all_preds[i]  for i in idx]
            ll = [all_labels[i] for i in idx]
            result[f"macro_f1_{tt}"] = compute_macro_f1(pp, ll)
            result[f"n_{tt}"]         = len(idx)

    # ── Hard subset 分别统计 ──
    for tag in ("multi_entity", "conflict", "ambiguous", "any_hard"):
        idx = [
            i for i, hs in enumerate(all_hard_subsets)
            if (tag == "any_hard" and hs.strip()) or (tag != "any_hard" and tag in hs)
        ]
        if idx:
            pp = [all_preds[i]  for i in idx]
            ll = [all_labels[i] for i in idx]
            result[f"macro_f1_hard_{tag}"] = compute_macro_f1(pp, ll)
            result[f"n_hard_{tag}"]         = len(idx)

    # ── 路由权重统计 ──
    rw_cat = torch.cat(routing_weights_all, dim=0)          # (N, 3)
    re_cat = torch.cat(routing_entropy_all, dim=0)          # (N,)

    result["routing"] = {
        "mean_weights":    rw_cat.mean(0).tolist(),
        "std_weights":     rw_cat.std(0).tolist(),
        "mean_entropy":    re_cat.mean().item(),
        "max_entropy":     float(torch.log(torch.tensor(3.0))),  # 均匀分布的最大熵
        "entropy_ratio":   (re_cat.mean() / torch.log(torch.tensor(3.0))).item(),
    }

    # ── Expert 多样性分析 ──
    if all(expert_reprs_all[i] for i in range(3)):
        reprs = [torch.cat(expert_reprs_all[i], dim=0) for i in range(3)]
        diversity_info = analyze_expert_diversity(reprs, all_hard_subsets)
        result["expert_diversity"] = diversity_info

    return result


# ─────────────────────────────────────────────────────────
# 报告生成
# ─────────────────────────────────────────────────────────
def generate_report(all_results: dict, output_dir: str) -> str:
    """生成可读版 Markdown 分析报告。"""
    lines = [
        "# 消融实验报告：Router + Experts",
        "",
        f"> 生成时间：{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 1. 整体 Macro-F1 对比",
        "",
        "| 配置 | Macro-F1 | Accuracy | 描述 |",
        "|------|----------|----------|------|",
    ]

    # 按 macro_f1 降序排列
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]["macro_f1"], reverse=True)

    full_f1 = all_results.get("full", {}).get("macro_f1", 0.0)

    for name, res in sorted_results:
        f1   = res["macro_f1"]
        acc  = res["accuracy"]
        desc = res["desc"]
        delta = f" (+{f1-full_f1:+.4f})" if name != "full" else ""
        lines.append(f"| **{name}** | {f1:.4f}{delta} | {acc:.4f} | {desc} |")

    lines += [
        "",
        "## 2. Hard Subset 分析",
        "",
        "（只展示 full 与关键消融配置的对比）",
        "",
        "| 配置 | multi_entity F1 | conflict F1 | ambiguous F1 | any_hard F1 |",
        "|------|-----------------|-------------|--------------|-------------|",
    ]

    for name in ["full", "no_router", "no_conflict_expert", "no_context_expert", "base_expert_only"]:
        if name not in all_results:
            continue
        res = all_results[name]
        me  = res.get("macro_f1_hard_multi_entity", "N/A")
        cf  = res.get("macro_f1_hard_conflict",     "N/A")
        amb = res.get("macro_f1_hard_ambiguous",    "N/A")
        ah  = res.get("macro_f1_hard_any_hard",     "N/A")
        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
        lines.append(f"| {name} | {fmt(me)} | {fmt(cf)} | {fmt(amb)} | {fmt(ah)} |")

    lines += [
        "",
        "## 3. 路由分析",
        "",
        "| 配置 | Base权重 | Conflict权重 | Context权重 | 平均熵 | 熵比率 |",
        "|------|----------|--------------|-------------|--------|--------|",
    ]

    for name, res in sorted_results:
        rw = res.get("routing", {})
        mw = rw.get("mean_weights", [0, 0, 0])
        ent = rw.get("mean_entropy", 0)
        er  = rw.get("entropy_ratio", 0)
        lines.append(
            f"| {name} | {mw[0]:.3f} | {mw[1]:.3f} | {mw[2]:.3f} | {ent:.4f} | {er:.4f} |"
        )

    lines += [
        "",
        "## 4. Expert 多样性诊断",
        "",
        "（余弦相似度越低 = experts 越分化 = L_div 效果越好）",
        "",
        "| 配置 | 整体平均余弦相似度 | Hard样本余弦相似度 | Base-Conflict | Base-Context | Conflict-Context |",
        "|------|-------------------|-------------------|---------------|--------------|-----------------|",
    ]

    for name, res in sorted_results:
        ed = res.get("expert_diversity", {})
        overall = ed.get("mean_cosine_sim_all",  "N/A")
        hard    = ed.get("mean_cosine_sim_hard",  "N/A")
        pw      = ed.get("pairwise_sims",        {})
        bc  = pw.get("Base-Conflict",    "N/A")
        bct = pw.get("Base-Context",     "N/A")
        cct = pw.get("Conflict-Context", "N/A")
        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
        lines.append(f"| {name} | {fmt(overall)} | {fmt(hard)} | {fmt(bc)} | {fmt(bct)} | {fmt(cct)} |")

    lines += [
        "",
        "## 5. 验收标准回答",
        "",
        "### Q1: Router 是否真的在工作？",
    ]
    if "full" in all_results and "no_router" in all_results:
        delta = all_results["full"]["macro_f1"] - all_results["no_router"]["macro_f1"]
        if delta > 0:
            lines.append(f"✅ **是**。完整 Router 比均匀路由 Macro-F1 高出 **{delta:+.4f}**，"
                         f"说明路由器学到了有意义的分发策略。")
        elif delta > -0.005:
            lines.append(f"⚠️ **差距较小**（{delta:+.4f}）。可能需要更多训练轮次或调整 λ_router。")
        else:
            lines.append(f"❌ **Router 当前无效**（{delta:+.4f}）。建议检查路由损失权重和弱监督标签质量。")

    lines += [
        "",
        "### Q2: Expert 是否真的有分工？",
    ]
    if "full" in all_results:
        ed = all_results["full"].get("expert_diversity", {})
        sim = ed.get("mean_cosine_sim_hard", ed.get("mean_cosine_sim_all", None))
        if sim is not None:
            if sim < 0.7:
                lines.append(f"✅ **是**。Hard 样本上 Expert 表征平均余弦相似度 = {sim:.4f}（< 0.7），"
                             f"说明 Expert 已分化，L_div 有效。")
            else:
                lines.append(f"⚠️ Expert 表征相似度 = {sim:.4f}（> 0.7），可能需要增大 λ_div 或调整 margin。")

    lines += [
        "",
        "### Q3: 提升主要来自 conflict / ambiguous 样本？",
    ]
    if "full" in all_results and "base_expert_only" in all_results:
        full_me  = all_results["full"].get("macro_f1_hard_multi_entity", 0)
        base_me  = all_results["base_expert_only"].get("macro_f1_hard_multi_entity", 0)
        full_amb = all_results["full"].get("macro_f1_hard_ambiguous", 0)
        base_amb = all_results["base_expert_only"].get("macro_f1_hard_ambiguous", 0)
        if full_me > base_me or full_amb > base_amb:
            lines.append(
                f"✅ **是**。完整模型在 multi_entity 上提升 {full_me-base_me:+.4f}，"
                f"在 ambiguous 上提升 {full_amb-base_amb:+.4f}。"
            )
        else:
            lines.append("❌ 当前数据未显示明确的 hard subset 提升，需进一步调优。")

    lines += [
        "",
        "### Q4: 去掉某个 Expert，性能会不会掉？",
    ]
    if "full" in all_results:
        for name, tag in [("no_conflict_expert", "Conflict Expert"), ("no_context_expert", "Context Expert")]:
            if name in all_results:
                delta = all_results["full"]["macro_f1"] - all_results[name]["macro_f1"]
                direction = "✅ 掉" if delta > 0 else "❌ 未掉"
                lines.append(f"  - 去掉 {tag}：{direction}（delta = {delta:+.4f}）")

    lines.append("")
    lines.append("---")
    lines.append("*本报告由 `ablation_router_experts.py` 自动生成*")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Router + Experts 消融实验")

    p.add_argument("--checkpoint",  required=True,
                   help="训练好的模型 checkpoint 路径（best_model.pt）")
    p.add_argument("--test_file",   default="test.jsonl",
                   help="评估数据集（默认使用 test split）")
    p.add_argument("--output_dir",  default="ablation_results")
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device",      default="auto")
    p.add_argument("--configs",     nargs="+",
                   default=list(ABLATION_CONFIGS.keys()),
                   help="要运行的消融配置名，默认全部")

    return p.parse_args()


def main():
    args = parse_args()

    # ── 设备 ──
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 加载 checkpoint ──
    print(f"[Ablation] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    saved_args = ckpt.get("args", {})

    backbone_name  = saved_args.get("backbone",        "ProsusAI/finbert")
    hidden_size    = saved_args.get("hidden_size",     768)
    bottleneck     = saved_args.get("bottleneck_size", 256)
    dropout        = saved_args.get("dropout",         0.1)

    print(f"[Ablation] Backbone: {backbone_name} | Hidden: {hidden_size}")

    tokenizer = AutoTokenizer.from_pretrained(backbone_name)

    # ── 数据集 ──
    test_dataset = FinSentDataset(
        args.test_file, tokenizer,
        max_length=saved_args.get("max_length", 256),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    print(f"[Ablation] Test samples: {len(test_dataset)}")

    # ── 模型 ──
    model = FinSentModel(
        backbone_name=backbone_name,
        hidden_size=hidden_size,
        bottleneck_size=bottleneck,
        dropout=0.0,     # 推理时关闭 dropout
        use_alignment_head=saved_args.get("use_alignment_head", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    val_f1 = ckpt.get("val_metrics", {}).get("macro_f1", "N/A")
    print(f"[Ablation] Model loaded (val Macro-F1 at save: {val_f1})")

    # ── 运行消融实验 ──
    all_results = {}

    for config_name in args.configs:
        if config_name not in ABLATION_CONFIGS:
            print(f"[Warning] Unknown config: {config_name}, skipping.")
            continue

        config = ABLATION_CONFIGS[config_name]
        print(f"\n{'─'*50}")
        print(f"[Ablation] Running: {config_name}")
        print(f"  {config['desc']}")

        result = run_ablation(model, test_loader, device, config_name, config)
        all_results[config_name] = result

        print(f"  Macro-F1: {result['macro_f1']:.4f} | Acc: {result['accuracy']:.4f}")
        if "macro_f1_hard_multi_entity" in result:
            print(f"  Hard F1: multi_entity={result.get('macro_f1_hard_multi_entity', 0):.4f}, "
                  f"conflict={result.get('macro_f1_hard_conflict', 0):.4f}, "
                  f"ambiguous={result.get('macro_f1_hard_ambiguous', 0):.4f}")
        rw = result.get("routing", {})
        if rw:
            print(f"  Router weights: {[f'{w:.3f}' for w in rw.get('mean_weights', [])]} | "
                  f"entropy ratio: {rw.get('entropy_ratio', 0):.4f}")

    # ── 保存结果 ──
    summary_path = Path(args.output_dir) / "ablation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[Ablation] Results saved to {summary_path}")

    # ── 生成报告 ──
    report_md = generate_report(all_results, args.output_dir)
    report_path = Path(args.output_dir) / "ablation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[Ablation] Report saved to {report_path}")

    # ── 打印摘要表格 ──
    print(f"\n{'='*60}")
    print("消融实验结果摘要")
    print(f"{'='*60}")
    print(f"{'配置':<25} {'Macro-F1':>10} {'vs full':>8}")
    print(f"{'─'*45}")

    full_f1 = all_results.get("full", {}).get("macro_f1", None)
    for name, res in sorted(all_results.items(), key=lambda x: x[1]["macro_f1"], reverse=True):
        f1 = res["macro_f1"]
        delta = ""
        if full_f1 is not None and name != "full":
            delta = f"{f1 - full_f1:+.4f}"
        print(f"{name:<25} {f1:>10.4f} {delta:>8}")


if __name__ == "__main__":
    main()
