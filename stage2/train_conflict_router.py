"""
train_conflict_router.py
训练脚本：Router + Experts 主模型训练

用法：
  python train_conflict_router.py \
      --train_file  train.jsonl \
      --val_file    val.jsonl \
      --output_dir  checkpoints/ \
      --backbone    ProsusAI/finbert \
      --epochs      5 \
      --batch_size  32 \
      --lr          2e-5 \
      --lambda_router 0.3 \
      --lambda_div    0.1

总损失：
  L_total = L_sent + λ_target * L_target + λ_router * L_router + L_div

  L_sent   : sentence_semantic 样本的主任务交叉熵
  L_target : target_semantic   样本的主任务交叉熵（权重可调）
  L_router : 路由器弱监督软交叉熵
  L_div    : Expert 表征多样性约束（只在 hard 样本上）

评估指标：Macro-F1（与 eval_protocol.md 一致）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report

# 确保项目根目录在 sys.path 中
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.experts import FinSentModel, LABEL2ID, ID2LABEL
from models.router import compute_router_loss, build_aux_signals
from losses.diversity_loss import DiversityLoss


# ─────────────────────────────────────────────────────────
# 数据集
# ─────────────────────────────────────────────────────────
class FinSentDataset(Dataset):
    """
    金融情感分析数据集。
    加载 JSONL 格式，过滤 market_aux（不参与主任务训练）。
    支持 sentence_semantic 和 target_semantic 两种输入模式。
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        include_market_aux: bool = False,
    ):
        self.samples     = []
        self.tokenizer   = tokenizer
        self.max_length  = max_length

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if not include_market_aux and d["task_type"] == "market_aux":
                    continue
                if d.get("label") is None:
                    continue
                self.samples.append(d)

        print(f"[Dataset] Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        d = self.samples[idx]
        text        = d["text"]
        target      = d.get("target") or ""
        label_str   = d["label"]
        task_type   = d["task_type"]
        hard_subset = d.get("hard_subset", "") or ""

        label_id = LABEL2ID[label_str]

        # Target-aware 输入：text + target 作为 pair
        if task_type == "target_semantic" and target:
            enc = self.tokenizer(
                text,
                target,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
        else:
            enc = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids", torch.zeros(1, dtype=torch.long)).squeeze(0),
            "label":          torch.tensor(label_id, dtype=torch.long),
            "task_type":      task_type,
            "hard_subset":    hard_subset,
            "text":           text,
            "uid":            d.get("uid", ""),
        }


def collate_fn(batch: list[dict]) -> dict:
    """批次整理函数（处理不等长字段）。"""
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "token_type_ids": torch.stack([b["token_type_ids"] for b in batch]),
        "label":          torch.stack([b["label"]          for b in batch]),
        "task_type":      [b["task_type"]   for b in batch],
        "hard_subset":    [b["hard_subset"] for b in batch],
        "text":           [b["text"]        for b in batch],
        "uid":            [b["uid"]         for b in batch],
    }


# ─────────────────────────────────────────────────────────
# 训练工具函数
# ─────────────────────────────────────────────────────────
def compute_macro_f1(preds: list[int], labels: list[int]) -> float:
    """计算三类 Macro-F1（与 eval_protocol 一致）。"""
    return f1_score(labels, preds, average="macro", labels=[0, 1, 2], zero_division=0)


def compute_losses(
    model_out: dict,
    labels: torch.Tensor,
    task_types: list[str],
    hard_subset_list: list[str],
    div_loss_fn: DiversityLoss,
    lambda_target: float = 1.0,
    lambda_router: float = 0.3,
    return_expert_repr: bool = True,
) -> dict:
    """
    计算所有损失项。

    Returns:
        dict:
          'total' : 总损失
          'sent'  : L_sent
          'target': L_target
          'router': L_router
          'div'   : L_div
    """
    # 样本掩码
    sent_mask   = torch.tensor(
        [t == "sentence_semantic" for t in task_types], dtype=torch.bool
    ).to(labels.device)
    target_mask = ~sent_mask

    logits = model_out["logits"]

    # ── L_sent：sentence_semantic 样本 ──
    if sent_mask.any():
        l_sent = F.cross_entropy(logits[sent_mask], labels[sent_mask])
    else:
        l_sent = torch.tensor(0.0, device=labels.device)

    # ── L_target：target_semantic 样本 ──
    if target_mask.any():
        l_target = F.cross_entropy(logits[target_mask], labels[target_mask])
    else:
        l_target = torch.tensor(0.0, device=labels.device)

    # ── L_router：弱监督路由损失 ──
    if "weak_labels" in model_out:
        l_router = compute_router_loss(
            model_out["routing_logits"],
            model_out["weak_labels"],
        )
    else:
        l_router = torch.tensor(0.0, device=labels.device)

    # ── L_div：Expert 表征多样性约束 ──
    if return_expert_repr and "expert_reprs" in model_out:
        l_div = div_loss_fn(model_out["expert_reprs"], hard_subset_list)
    else:
        l_div = torch.tensor(0.0, device=labels.device)

    # ── 总损失 ──
    l_total = (
        l_sent
        + lambda_target * l_target
        + lambda_router * l_router
        + l_div
    )

    return {
        "total":  l_total,
        "sent":   l_sent.detach(),
        "target": l_target.detach(),
        "router": l_router.detach(),
        "div":    l_div.detach() if isinstance(l_div, torch.Tensor) else l_div,
    }


# ─────────────────────────────────────────────────────────
# 训练与评估
# ─────────────────────────────────────────────────────────
def train_epoch(
    model: FinSentModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    div_loss_fn: DiversityLoss,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
) -> dict:
    model.train()
    total_losses = defaultdict(float)
    all_preds, all_labels = [], []

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["label"].to(device)
        task_types     = batch["task_type"]
        hard_subset    = batch["hard_subset"]
        texts          = batch["text"]

        # 构建辅助路由信号（第一轮 pred_entropy 为 None）
        aux_signals = build_aux_signals(texts, hard_subset).to(device)

        # 前向（需要 expert_repr 用于 diversity loss）
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            aux_signals=aux_signals,
            hard_subset=hard_subset,
            return_expert_repr=True,
        )

        losses = compute_losses(
            model_out=out,
            labels=labels,
            task_types=task_types,
            hard_subset_list=hard_subset,
            div_loss_fn=div_loss_fn,
            lambda_target=args.lambda_target,
            lambda_router=args.lambda_router,
            return_expert_repr=True,
        )

        loss = losses["total"]
        loss.backward()

        # 梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # 统计
        for k, v in losses.items():
            total_losses[k] += v.item() if isinstance(v, torch.Tensor) else v

        preds = out["logits"].argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

        if (step + 1) % 50 == 0:
            lr = scheduler.get_last_lr()[0]
            macro_f1 = compute_macro_f1(all_preds[-100:], all_labels[-100:])
            print(
                f"  Epoch {epoch} | Step {step+1}/{len(loader)} | "
                f"Loss {losses['total'].item():.4f} "
                f"(sent={losses['sent'].item():.3f}, "
                f"tgt={losses['target'].item():.3f}, "
                f"rtr={losses['router'].item():.3f}, "
                f"div={losses['div'].item() if isinstance(losses['div'], torch.Tensor) else losses['div']:.3f}) | "
                f"LR {lr:.2e} | recent macro-F1 {macro_f1:.4f}"
            )

    n = len(loader)
    avg = {k: v / n for k, v in total_losses.items()}
    avg["macro_f1"] = compute_macro_f1(all_preds, all_labels)
    return avg


@torch.no_grad()
def evaluate(
    model: FinSentModel,
    loader: DataLoader,
    device: torch.device,
    split_name: str = "val",
) -> dict:
    """
    在验证/测试集上评估。
    返回：Macro-F1、每类 F1、per-dataset F1、hard subset F1。
    """
    model.eval()
    all_preds, all_labels = [], []
    all_task_types, all_hard_subsets, all_datasets = [], [], []
    routing_weights_list = []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)

        out = model(input_ids, attention_mask, token_type_ids)
        preds = out["logits"].argmax(dim=-1).cpu().tolist()

        all_preds.extend(preds)
        all_labels.extend(batch["label"].tolist())
        all_task_types.extend(batch["task_type"])
        all_hard_subsets.extend(batch["hard_subset"])
        routing_weights_list.append(out["routing_weights"].cpu())

    # ── 整体指标 ──
    macro_f1 = compute_macro_f1(all_preds, all_labels)
    report = classification_report(
        all_labels, all_preds,
        target_names=["positive", "neutral", "negative"],
        output_dict=True, zero_division=0,
    )

    results = {
        "macro_f1":   macro_f1,
        "accuracy":   sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels),
        "per_class":  {
            "positive": report["positive"]["f1-score"],
            "neutral":  report["neutral"]["f1-score"],
            "negative": report["negative"]["f1-score"],
        },
    }

    # ── Task type 分别统计 ──
    for tt in ("sentence_semantic", "target_semantic"):
        mask_idx = [i for i, t in enumerate(all_task_types) if t == tt]
        if mask_idx:
            pp = [all_preds[i]  for i in mask_idx]
            ll = [all_labels[i] for i in mask_idx]
            results[f"macro_f1_{tt}"] = compute_macro_f1(pp, ll)

    # ── Hard subset 分别统计 ──
    for tag in ("multi_entity", "conflict", "ambiguous"):
        mask_idx = [i for i, hs in enumerate(all_hard_subsets) if tag in hs]
        if mask_idx:
            pp = [all_preds[i]  for i in mask_idx]
            ll = [all_labels[i] for i in mask_idx]
            results[f"macro_f1_hard_{tag}"] = compute_macro_f1(pp, ll)
            results[f"n_hard_{tag}"]         = len(mask_idx)

    # ── 路由权重统计 ──
    rw = torch.cat(routing_weights_list, dim=0)  # (N, 3)
    results["mean_routing_weights"] = rw.mean(0).tolist()

    # 打印摘要
    print(f"\n[{split_name}] Macro-F1: {macro_f1:.4f} | "
          f"Acc: {results['accuracy']:.4f}")
    print(f"  Per-class F1: pos={results['per_class']['positive']:.4f}, "
          f"neu={results['per_class']['neutral']:.4f}, "
          f"neg={results['per_class']['negative']:.4f}")
    if "macro_f1_sentence_semantic" in results:
        print(f"  Task F1: sent={results.get('macro_f1_sentence_semantic', 0):.4f}, "
              f"target={results.get('macro_f1_target_semantic', 0):.4f}")
    if "macro_f1_hard_multi_entity" in results:
        print(f"  Hard subset F1: multi_entity={results.get('macro_f1_hard_multi_entity', 0):.4f}, "
              f"conflict={results.get('macro_f1_hard_conflict', 0):.4f}, "
              f"ambiguous={results.get('macro_f1_hard_ambiguous', 0):.4f}")
    print(f"  Mean routing weights (base/conflict/context): "
          f"{[f'{w:.3f}' for w in results['mean_routing_weights']]}")

    return results


# ─────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FinSentModel (Router + Experts)")

    # 数据
    p.add_argument("--train_file",  default="train.jsonl")
    p.add_argument("--val_file",    default="val.jsonl")
    p.add_argument("--output_dir",  default="checkpoints/router_experts")
    p.add_argument("--max_length",  type=int, default=256,
                   help="最大 token 长度（含 target 时建议 256）")

    # 模型
    p.add_argument("--backbone",          default="ProsusAI/finbert",
                   help="HuggingFace 模型标识符，如 ProsusAI/finbert 或 bert-base-uncased")
    p.add_argument("--hidden_size",       type=int, default=768)
    p.add_argument("--bottleneck_size",   type=int, default=256,
                   help="Expert Adapter 瓶颈维度")
    p.add_argument("--dropout",           type=float, default=0.1)
    p.add_argument("--use_alignment_head", action="store_true",
                   help="启用语义-市场对齐辅助头（需要 analysis.jsonl 中的 market_aux 样本）")

    # 训练超参
    p.add_argument("--epochs",          type=int,   default=5)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=2e-5)
    p.add_argument("--warmup_ratio",    type=float, default=0.1,
                   help="线性 warmup 步数占总步数的比例")
    p.add_argument("--weight_decay",    type=float, default=0.01)

    # 损失权重
    p.add_argument("--lambda_target",   type=float, default=1.0,
                   help="L_target（target_semantic 任务）权重")
    p.add_argument("--lambda_router",   type=float, default=0.3,
                   help="L_router（弱监督路由损失）权重")
    p.add_argument("--lambda_div",      type=float, default=0.1,
                   help="L_div（Expert 多样性约束）权重")
    p.add_argument("--div_margin",      type=float, default=0.0,
                   help="Diversity loss 余弦相似度松弛量")

    # 其他
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--device",          default="auto",
                   help="'auto' / 'cuda' / 'cpu' / 'mps'")
    p.add_argument("--num_workers",     type=int,   default=2)
    p.add_argument("--save_every",      type=int,   default=1,
                   help="每 N 个 epoch 保存一次 checkpoint")

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
    print(f"[Config] Device: {device}")

    # ── 随机种子 ──
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── 输出目录 ──
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Tokenizer ──
    print(f"[Config] Loading tokenizer: {args.backbone}")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    # ── 数据集 ──
    train_dataset = FinSentDataset(args.train_file, tokenizer, args.max_length)
    val_dataset   = FinSentDataset(args.val_file,   tokenizer, args.max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True,  collate_fn=collate_fn,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2,
        shuffle=False, collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    # ── 模型 ──
    print(f"[Config] Building model (backbone={args.backbone})")
    model = FinSentModel(
        backbone_name=args.backbone,
        hidden_size=args.hidden_size,
        bottleneck_size=args.bottleneck_size,
        dropout=args.dropout,
        use_alignment_head=args.use_alignment_head,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Config] Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # ── 损失函数 ──
    div_loss_fn = DiversityLoss(weight=args.lambda_div, margin=args.div_margin)

    # ── 优化器（backbone 和非 backbone 参数分开学习率）──
    backbone_params = list(model.backbone.parameters())
    backbone_ids    = set(id(p) for p in backbone_params)
    other_params    = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr},
            {"params": other_params,    "lr": args.lr * 5},   # router/experts/heads 用更大 lr
        ],
        weight_decay=args.weight_decay,
    )

    total_steps   = len(train_loader) * args.epochs
    warmup_steps  = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f"[Config] Epochs={args.epochs}, Steps/epoch={len(train_loader)}, "
          f"Total={total_steps}, Warmup={warmup_steps}")
    print(f"[Config] λ_target={args.lambda_target}, λ_router={args.lambda_router}, "
          f"λ_div={args.lambda_div}")

    # ── 训练循环 ──
    best_macro_f1 = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            div_loss_fn, args, device, epoch,
        )
        print(
            f"\n[Train] Macro-F1: {train_metrics['macro_f1']:.4f} | "
            f"Loss: {train_metrics['total']:.4f} "
            f"(sent={train_metrics['sent']:.3f}, "
            f"tgt={train_metrics['target']:.3f}, "
            f"rtr={train_metrics['router']:.3f}, "
            f"div={train_metrics['div']:.3f})"
        )

        val_metrics = evaluate(model, val_loader, device, "val")

        elapsed = time.time() - t0
        print(f"[Epoch {epoch}] Time: {elapsed:.1f}s")

        history.append({
            "epoch":        epoch,
            "train":        train_metrics,
            "val":          val_metrics,
        })

        # ── 保存最优模型 ──
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_path = Path(args.output_dir) / "best_model.pt"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_metrics": val_metrics,
                "args":        vars(args),
            }, best_path)
            print(f"  ✓ Best model saved (Macro-F1={best_macro_f1:.4f}) → {best_path}")

        # ── 定期 checkpoint ──
        if epoch % args.save_every == 0:
            ckpt_path = Path(args.output_dir) / f"epoch_{epoch}.pt"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "scheduler":   scheduler.state_dict(),
                "val_metrics": val_metrics,
                "args":        vars(args),
            }, ckpt_path)

    # ── 保存训练历史 ──
    history_path = Path(args.output_dir) / "train_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*60}")
    print(f"Training complete. Best val Macro-F1: {best_macro_f1:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
