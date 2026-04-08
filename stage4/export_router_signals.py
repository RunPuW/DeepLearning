"""
export_router_signals.py
------------------------
F:\\stage4\\ 目录下运行。

从 stage3 的 best_model.pt 对 test split 跑推理，
提取每条样本的路由信号，写入 router_cache/test_router.jsonl。

输出格式（每行一条）：
  {"id": "...", "should_retrieve": true/false,
   "expert_type": "base"|"conflict"|"context", "confidence": 0.xxxx}

should_retrieve 规则（与 trigger_policy.router_triggered_policy 一致）：
  - expert_type == "context"  -> 始终检索
  - 其他 expert, confidence < 0.7 -> 检索
  - 否则不检索

用法：
  cd F:\\stage4
  python export_router_signals.py
"""

import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# ── stage3 模块路径 ────────────────────────────────────────
STAGE3_DIR = r"F:\stage3"
sys.path.insert(0, STAGE3_DIR)
from models.experts import FinSentModel          # noqa: E402
from models.router  import build_aux_signals     # noqa: E402

# ── stage1 schema（load_split） ───────────────────────────
STAGE1_DIR = r"F:\stage1"
sys.path.insert(0, STAGE1_DIR)
from utils import load_split                     # noqa: E402

# ── 路径配置 ───────────────────────────────────────────────
MODEL_PATH      = r"F:\stage3\best_model.pt"
BACKBONE        = "ProsusAI/finbert"
DATA_PATH       = r"F:\stage4\data\test.jsonl"
RETRIEVAL_CACHE = r"F:\stage4\retrieval_cache.jsonl"   # 含 uncertainty
OUTPUT_DIR      = r"F:\stage4\router_cache"
OUTPUT_FILE     = os.path.join(OUTPUT_DIR, "test_router.jsonl")

TARGET_SOURCES       = ["sentfin", "finentity"]
BATCH_SIZE           = 32
MAX_LENGTH           = 128
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPERT_NAMES         = {0: "base", 1: "conflict", 2: "context"}
CONFIDENCE_THRESHOLD = 0.7


# ── 工具函数 ───────────────────────────────────────────────

def load_retrieval_cache(path: str) -> dict:
    cache = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d   = json.loads(line.strip())
            sid = d.pop("sample_id")
            cache[sid] = d
    print("[OK] Retrieval cache loaded: {} entries".format(len(cache)))
    return cache


def derive_should_retrieve(expert_type: str, confidence: float) -> bool:
    if expert_type == "context":
        return True
    return confidence < CONFIDENCE_THRESHOLD


# ── Dataset ────────────────────────────────────────────────

class SimpleDataset(Dataset):
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
        # pred_entropy: 用 retrieval cache 里 baseline 的 uncertainty 作为代理
        uncertainty = self.retrieval_cache.get(
            s.sample_id, {}
        ).get("uncertainty", 0.0)

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "sample_idx":     idx,
            "uncertainty":    float(uncertainty),
            # subset_tags 传成空格拼接字符串，与 build_aux_signals 期望格式一致
            "subset_tags":    " ".join(s.subset_tags) if s.subset_tags else "",
            "raw_text":       s.text,
        }


def collate(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "sample_indices": [b["sample_idx"]  for b in batch],
        "uncertainties":  [b["uncertainty"] for b in batch],
        "subset_tags":    [b["subset_tags"] for b in batch],
        "raw_texts":      [b["raw_text"]    for b in batch],
    }


# ── 主流程 ──────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载数据
    samples = [
        s for s in load_split(DATA_PATH, TARGET_SOURCES)
        if s.target
    ]
    print("[OK] Loaded {} test samples".format(len(samples)))

    # 2. 加载 retrieval cache（获取 uncertainty 作为 pred_entropy 代理）
    retrieval_cache = load_retrieval_cache(RETRIEVAL_CACHE)

    # 3. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)

    # 4. 加载 FinSentModel
    model = FinSentModel(
        backbone_name      = BACKBONE,
        use_alignment_head = False,
    )
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(DEVICE)
    model.eval()
    print("[OK] FinSentModel loaded (strict=True, 0 missing keys)")

    # 5. DataLoader
    dataset = SimpleDataset(samples, tokenizer, retrieval_cache, MAX_LENGTH)
    loader  = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate
    )

    # 6. 推理 + 写出
    written = 0
    with open(OUTPUT_FILE, "w", encoding="ascii") as fout:
        for batch in loader:
            # 构造 aux_signals (B, 4): [multi_entity, pred_entropy, event_word, retrieval_utility]
            pred_entropy = torch.tensor(
                batch["uncertainties"], dtype=torch.float32
            )                                                   # (B,)
            aux_signals = build_aux_signals(
                texts             = batch["raw_texts"],
                hard_subset_list  = batch["subset_tags"],
                pred_entropy      = pred_entropy,
                retrieval_utility = None,                       # 当前占位 0
            ).to(DEVICE)                                        # (B, 4)

            with torch.no_grad():
                out = model.forward(
                    input_ids      = batch["input_ids"].to(DEVICE),
                    attention_mask = batch["attention_mask"].to(DEVICE),
                    aux_signals    = aux_signals,
                )

            rw              = out["routing_weights"].cpu()      # (B, 3)
            conf, expert_id = rw.max(dim=-1)                   # (B,), (B,)

            for i, sidx in enumerate(batch["sample_indices"]):
                s          = samples[sidx]
                etype      = EXPERT_NAMES[expert_id[i].item()]
                confidence = round(conf[i].item(), 4)
                row = {
                    "id":              s.sample_id,
                    "should_retrieve": derive_should_retrieve(etype, confidence),
                    "expert_type":     etype,
                    "confidence":      confidence,
                }
                fout.write(json.dumps(row, ensure_ascii=True) + "\n")
                written += 1

            print("  processed {}/{}".format(written, len(samples)), end="\r")

    print("\n[OK] Router cache written: {} entries -> {}".format(written, OUTPUT_FILE))

    # 7. 分布统计
    expert_counts  = {"base": 0, "conflict": 0, "context": 0}
    retrieve_count = 0
    with open(OUTPUT_FILE, encoding="ascii") as f:
        for line in f:
            row = json.loads(line)
            expert_counts[row["expert_type"]] += 1
            if row["should_retrieve"]:
                retrieve_count += 1

    total = written
    print("\n[ROUTING STATS]")
    for ename, cnt in expert_counts.items():
        print("  {:<10} {:>5}  ({:.1f}%)".format(
            ename, cnt, 100.0 * cnt / total))
    print("  should_retrieve: {} / {} ({:.1f}%)".format(
        retrieve_count, total, 100.0 * retrieve_count / total))


if __name__ == "__main__":
    main()
