"""
diag3_oracle_routing.py
-----------------------
诊断实验三：Oracle 路由实验。

目的：区分 context expert 路由失败的根本原因：
  - 假说 A：Router 是瓶颈。Context expert 本身有能力，
            只是 router 从未把 ambiguous 样本送过去。
  - 假说 B：Context expert 是瓶颈。即使强制路由，
            检索库质量不足导致 context expert 无法改善预测。

方法：
  对 ambiguous 样本强制设 routing_weights = [0, 0, 1]
  （100% context expert），其余样本正常路由。
  对比强制路由前后的 ambiguous 子集 macro-F1。

结论：
  F1 提升 -> 假说 A 成立，修复方向是改进路由信号
  F1 不变或下降 -> 假说 B 成立，修复方向是改进检索库

用法：
  cd F:\\stage4
  python diag3_oracle_routing.py
"""

import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# ── 路径 ───────────────────────────────────────────────────
STAGE3_DIR      = r"F:\stage3"
STAGE1_DIR      = r"F:\stage1"
MODEL_PATH      = r"F:\stage3\best_model.pt"
BACKBONE        = "ProsusAI/finbert"
DATA_PATH       = r"F:\stage4\data\test.jsonl"
RETRIEVAL_CACHE = r"F:\stage4\retrieval_cache.jsonl"
OUTPUT_FILE     = r"F:\stage4\reports\diag3_oracle_routing.json"

TARGET_SOURCES  = ["sentfin", "finentity"]
BATCH_SIZE      = 32
MAX_LENGTH      = 128
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, STAGE3_DIR)
sys.path.insert(0, STAGE1_DIR)

from models.experts import FinSentModel, ID2LABEL, LABEL2ID  # noqa
from models.router  import build_aux_signals                 # noqa
from utils          import load_split                        # noqa


# ── 工具 ───────────────────────────────────────────────────

def load_retrieval_cache(path):
    cache = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d   = json.loads(line.strip())
            sid = d.pop("sample_id")
            cache[sid] = d
    print("[OK] Retrieval cache: {} entries".format(len(cache)))
    return cache


def macro_f1(true_labels, pred_labels):
    classes = list(set(true_labels))
    f1s = []
    for c in classes:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == c and p == c)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != c and p == c)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


# ── Dataset ────────────────────────────────────────────────

class OracleDataset(Dataset):
    def __init__(self, samples, tokenizer, retrieval_cache, max_length):
        self.samples         = samples
        self.tokenizer       = tokenizer
        self.retrieval_cache = retrieval_cache
        self.max_length      = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s    = self.samples[idx]
        text = "{} [SEP] {}".format(s.text, s.target) if s.target else s.text
        enc  = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        uncertainty = self.retrieval_cache.get(
            s.sample_id, {}
        ).get("uncertainty", 0.0)

        is_ambiguous = "ambiguous" in (s.subset_tags or [])

        return {
            "input_ids":    enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "sample_idx":   idx,
            "uncertainty":  float(uncertainty),
            "subset_tags":  " ".join(s.subset_tags) if s.subset_tags else "",
            "raw_text":     s.text,
            "true_label":   s.label,
            "is_ambiguous": is_ambiguous,
        }


def collate(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "sample_indices": [b["sample_idx"]   for b in batch],
        "uncertainties":  [b["uncertainty"]  for b in batch],
        "subset_tags":    [b["subset_tags"]  for b in batch],
        "raw_texts":      [b["raw_text"]     for b in batch],
        "true_labels":    [b["true_label"]   for b in batch],
        "is_ambiguous":   [b["is_ambiguous"] for b in batch],
    }


# ── 推理核心 ───────────────────────────────────────────────

def run_inference(model, loader, samples, oracle_mode, device):
    """
    oracle_mode=False : 正常路由（router 自由决策）
    oracle_mode=True  : ambiguous 样本强制 routing_weights=[0,0,1]
    """
    model.eval()
    results = []

    with torch.no_grad():
        for batch in loader:
            pred_entropy = torch.tensor(
                batch["uncertainties"], dtype=torch.float32
            )
            aux_signals = build_aux_signals(
                texts            = batch["raw_texts"],
                hard_subset_list = batch["subset_tags"],
                pred_entropy     = pred_entropy,
                retrieval_utility = None,
            ).to(device)

            out = model.forward(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                aux_signals    = aux_signals,
            )

            rw     = out["routing_weights"]   # (B, 3)
            logits = out["logits"]            # (B, 3)

            if oracle_mode:
                # 对 ambiguous 样本强制路由到 context expert
                oracle_rw = rw.clone()
                for i, is_amb in enumerate(batch["is_ambiguous"]):
                    if is_amb:
                        oracle_rw[i] = torch.tensor(
                            [0.0, 0.0, 1.0], device=device
                        )
                # 用强制路由权重重新过 expert adapters
                # 直接调用 expert 前向（复用已有 CLS 表征）
                cls_hidden = model.encode(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
                base_h     = model.base_expert(cls_hidden)["hidden"]
                conflict_h = model.conflict_expert(cls_hidden)["hidden"]
                context_h  = model.context_expert(cls_hidden)["hidden"]

                fused = (
                    oracle_rw[:, 0:1] * base_h +
                    oracle_rw[:, 1:2] * conflict_h +
                    oracle_rw[:, 2:3] * context_h
                )
                logits = model.classifier(fused)
                rw     = oracle_rw

            probs    = F.softmax(logits, dim=-1).cpu()
            pred_ids = logits.argmax(dim=-1).cpu().tolist()

            for i, sidx in enumerate(batch["sample_indices"]):
                s = samples[sidx]
                results.append({
                    "sample_id":    s.sample_id,
                    "true_label":   s.label,
                    "pred_label":   ID2LABEL[pred_ids[i]],
                    "subset_tags":  s.subset_tags or [],
                    "routing_weights": rw[i].cpu().tolist(),
                    "expert_chosen": ["base","conflict","context"][rw[i].argmax().item()],
                })

    return results


# ── 评估 ───────────────────────────────────────────────────

def evaluate_results(results, subset_filter=None):
    """
    subset_filter: None (all) | 'ambiguous' | 'none' | 'conflict'
    Returns: macro_f1, n_samples, accuracy
    """
    if subset_filter is None:
        rows = results
    elif subset_filter == "none":
        rows = [r for r in results if not r["subset_tags"]]
    else:
        rows = [r for r in results if subset_filter in r["subset_tags"]]

    if not rows:
        return None, 0, None

    true_labels = [r["true_label"] for r in rows]
    pred_labels = [r["pred_label"] for r in rows]
    acc = sum(1 for t, p in zip(true_labels, pred_labels) if t == p) / len(rows)
    mf1 = macro_f1(true_labels, pred_labels)
    return mf1, len(rows), acc


# ── 主流程 ──────────────────────────────────────────────────

def main():
    os.makedirs(r"F:\stage4\reports", exist_ok=True)

    # 1. 加载数据
    samples = [
        s for s in load_split(DATA_PATH, TARGET_SOURCES)
        if s.target
    ]
    print("[OK] Loaded {} test samples".format(len(samples)))

    retrieval_cache = load_retrieval_cache(RETRIEVAL_CACHE)
    tokenizer       = AutoTokenizer.from_pretrained(BACKBONE)

    # 2. 加载模型
    model = FinSentModel(backbone_name=BACKBONE, use_alignment_head=False)
    ckpt  = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(DEVICE)
    print("[OK] FinSentModel loaded (strict=True)")

    dataset = OracleDataset(samples, tokenizer, retrieval_cache, MAX_LENGTH)
    loader  = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
    )

    # 3. 正常路由推理
    print("\n[INFO] Running normal routing...")
    normal_results = run_inference(model, loader, samples, oracle_mode=False, device=DEVICE)

    # 4. Oracle 路由推理
    print("[INFO] Running oracle routing (ambiguous -> context expert forced)...")
    oracle_results = run_inference(model, loader, samples, oracle_mode=True, device=DEVICE)

    # 5. 对比评估
    subsets = ["ambiguous", "conflict", "multi_entity", "none", None]
    subset_labels = {
        "ambiguous":   "ambiguous",
        "conflict":    "conflict",
        "multi_entity":"multi_entity",
        "none":        "none",
        None:          "overall",
    }

    comparison = {}
    print("\n[ORACLE ROUTING COMPARISON]")
    print("{:<15} {:>8} {:>10} {:>10} {:>10}".format(
        "Subset", "N", "Normal F1", "Oracle F1", "Delta"))
    print("-" * 56)

    for sf in subsets:
        n_f1, n_count, n_acc = evaluate_results(normal_results, sf)
        o_f1, o_count, o_acc = evaluate_results(oracle_results, sf)
        if n_f1 is None:
            continue
        delta = o_f1 - n_f1
        label = subset_labels[sf]
        print("{:<15} {:>8} {:>10.4f} {:>10.4f} {:>+10.4f}".format(
            label, n_count, n_f1, o_f1, delta))
        comparison[label] = {
            "n":         n_count,
            "normal_f1": round(n_f1, 4),
            "oracle_f1": round(o_f1, 4),
            "delta":     round(delta, 4),
        }

    # 6. 路由分布对比（正常 vs oracle）
    print("\n[ROUTING DISTRIBUTION - ambiguous samples only]")
    ambig_normal = [r for r in normal_results if "ambiguous" in r["subset_tags"]]
    ambig_oracle = [r for r in oracle_results if "ambiguous" in r["subset_tags"]]

    from collections import Counter
    normal_dist = Counter(r["expert_chosen"] for r in ambig_normal)
    oracle_dist = Counter(r["expert_chosen"] for r in ambig_oracle)

    print("  Normal routing : {}".format(dict(normal_dist)))
    print("  Oracle routing : {}".format(dict(oracle_dist)))

    # 7. 结论判断
    print()
    ambig_delta = comparison.get("ambiguous", {}).get("delta", 0.0)
    if ambig_delta > 0.005:
        conclusion = "HYPOTHESIS_A"
        conclusion_text = (
            "Oracle routing improves ambiguous F1 by {:.4f}. "
            "HYPOTHESIS A supported: the router is the bottleneck. "
            "Context expert has latent capability; fix routing signal.".format(ambig_delta)
        )
    elif ambig_delta < -0.005:
        conclusion = "HYPOTHESIS_B"
        conclusion_text = (
            "Oracle routing degrades ambiguous F1 by {:.4f}. "
            "HYPOTHESIS B supported: context expert is the bottleneck. "
            "Even perfect routing cannot help; fix retrieval corpus.".format(abs(ambig_delta))
        )
    else:
        conclusion = "INCONCLUSIVE"
        conclusion_text = (
            "Oracle routing changes ambiguous F1 by only {:.4f} (|delta| <= 0.005). "
            "Neither router nor context expert provides signal. "
            "Root cause is likely the label distribution indistinguishability "
            "(see diag2_label_dist.py).".format(ambig_delta)
        )

    print("[CONCLUSION] {}".format(conclusion))
    print("  {}".format(conclusion_text))

    # 8. 保存结果
    output = {
        "comparison":    comparison,
        "conclusion":    conclusion,
        "conclusion_text": conclusion_text,
        "routing_dist":  {
            "ambiguous_normal": dict(normal_dist),
            "ambiguous_oracle": dict(oracle_dist),
        },
    }
    with open(OUTPUT_FILE, "w", encoding="ascii") as f:
        json.dump(output, f, indent=2, ensure_ascii=True)
    print("\n[OK] Results saved to {}".format(OUTPUT_FILE))


if __name__ == "__main__":
    main()
