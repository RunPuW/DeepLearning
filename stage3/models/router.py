"""
models/router.py
弱监督路由器 (Weakly Supervised Router)

根据多种信号决定样本最适合走哪条 Expert 路径：
  - CLS 表征（来自 backbone）
  - 多实体指示标志（multi_entity_flag）
  - 预测熵，即模型不确定性（pred_entropy）
  - 事件词存在性（event_word_flag）
  - 检索收益估计（retrieval_utility，由分工3/4填入，当前占位为0）

路由输出：三路 soft routing weights α ∈ R^3
  α[0] → Base Expert    (普通样本)
  α[1] → Conflict Expert (多实体/冲突样本)
  α[2] → Context Expert  (歧义/上下文不足样本)

弱监督标签来源：
  hard_subset 字段中含 'multi_entity'/'conflict' → 软标签偏向 Conflict Expert
  hard_subset 字段中含 'ambiguous'               → 软标签偏向 Context Expert
  其余                                             → 软标签偏向 Base Expert
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────
# 金融事件词词典（用于 event_word_flag 辅助信号）
# ─────────────────────────────────────────────────────────
FINANCIAL_EVENT_KEYWORDS = [
    "acqui", "merger", "merger", "bankrupt", "layoff", "restructur",
    "ipo ", " ipo", "delist", "dividend", "earning", "revenue",
    "forecast", "guidance", "downgrad", "upgrad", "recall",
    "investigat", "lawsuit", "settlement", "fine", "default",
    "insider", "fraud", "restat", "sanction", "writedown", "impairment",
]


def compute_event_word_flag(texts: list[str]) -> torch.Tensor:
    """
    检测文本中是否含有金融事件词。
    返回 float tensor (batch,)，含事件词为 1.0，否则 0.0。
    """
    flags = []
    for text in texts:
        tl = text.lower()
        found = any(kw in tl for kw in FINANCIAL_EVENT_KEYWORDS)
        flags.append(1.0 if found else 0.0)
    return torch.tensor(flags, dtype=torch.float32)


def build_aux_signals(
    texts: list[str],
    hard_subset_list: list[str],
    pred_entropy: torch.Tensor | None = None,
    retrieval_utility: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    构建路由器辅助信号向量（维度为 4）。

    维度定义：
      [0] multi_entity_flag   : hard_subset 含 'multi_entity' 时为 1.0
      [1] pred_entropy        : 模型预测熵（首轮为 0，后续更新）
      [2] event_word_flag     : 文本含金融事件词时为 1.0
      [3] retrieval_utility   : 检索收益估计（由检索模块提供，当前占位 0.0）

    Args:
        texts:              list of str, batch 文本
        hard_subset_list:   list of str, hard_subset 字段值
        pred_entropy:       (batch,) float tensor，上一轮预测熵；None 则用 0
        retrieval_utility:  (batch,) float tensor，检索收益；None 则用 0

    Returns:
        (batch, 4) float tensor
    """
    bsz = len(texts)

    # [0] multi_entity_flag
    me_flags = torch.tensor(
        [1.0 if "multi_entity" in hs else 0.0 for hs in hard_subset_list],
        dtype=torch.float32,
    )

    # [1] pred_entropy（归一化到 [0,1]，最大熵 = log(3) ≈ 1.099）
    if pred_entropy is None:
        pe = torch.zeros(bsz, dtype=torch.float32)
    else:
        pe = (pred_entropy / (torch.log(torch.tensor(3.0)) + 1e-8)).clamp(0.0, 1.0)

    # [2] event_word_flag
    ew_flags = compute_event_word_flag(texts)

    # [3] retrieval_utility
    if retrieval_utility is None:
        ru = torch.zeros(bsz, dtype=torch.float32)
    else:
        ru = retrieval_utility.float()

    aux = torch.stack([me_flags, pe, ew_flags, ru], dim=1)  # (batch, 4)
    return aux


# ─────────────────────────────────────────────────────────
# 路由器主模块
# ─────────────────────────────────────────────────────────
class WeaklySupervisedRouter(nn.Module):
    """
    三路专家弱监督路由器。

    结构：
      CLS 表征 → cls_proj (MLP) → cls_feat  (hidden/4)
      aux 信号  → aux_proj (MLP) → aux_feat  (hidden/4)
      concat    → router_head   → logits     (num_experts)
      softmax(logits / temperature) → routing_weights

    temperature 为可学习参数，防止路由过早坍缩为 one-hot。
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_experts: int = 3,
        aux_signal_dim: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_size:    backbone 输出维度（如 768 for BERT-base）
            num_experts:    专家数量，默认 3
            aux_signal_dim: 辅助信号维度，默认 4
            dropout:        dropout 概率
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts

        proj_dim = hidden_size // 4  # 192 for 768-dim backbone

        # 主特征投影
        self.cls_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # 辅助信号投影
        self.aux_proj = nn.Sequential(
            nn.Linear(aux_signal_dim, proj_dim),
            nn.GELU(),
        )

        # 路由输出层
        self.router_head = nn.Linear(proj_dim * 2, num_experts)

        # 可学习温度参数（初始化为 1）
        self.temperature = nn.Parameter(torch.ones(1))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.router_head.weight)
        nn.init.zeros_(self.router_head.bias)

    def forward(
        self,
        cls_hidden: torch.Tensor,
        aux_signals: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            cls_hidden:  (batch, hidden_size) backbone CLS 表征
            aux_signals: (batch, 4) 辅助信号；None 则用全零

        Returns:
            dict:
              'routing_weights': (batch, num_experts)  softmax 路由权重
              'routing_logits':  (batch, num_experts)  原始 logits（用于损失计算）
        """
        bsz = cls_hidden.size(0)
        device = cls_hidden.device

        if aux_signals is None:
            aux_signals = torch.zeros(bsz, 4, device=device)

        cls_feat = self.cls_proj(cls_hidden)          # (batch, proj_dim)
        aux_feat = self.aux_proj(aux_signals)          # (batch, proj_dim)

        combined = torch.cat([cls_feat, aux_feat], dim=-1)  # (batch, proj_dim*2)
        logits = self.router_head(combined)            # (batch, num_experts)

        temp = self.temperature.clamp(min=0.1)
        weights = F.softmax(logits / temp, dim=-1)    # (batch, num_experts)

        return {
            "routing_weights": weights,
            "routing_logits":  logits,
        }

    # ── 弱监督标签构建 ──────────────────────────────────────
    @staticmethod
    def build_weak_labels(
        hard_subset_list: list[str],
        device: torch.device,
    ) -> torch.Tensor:
        """
        从 hard_subset 字段派生弱监督路由软标签。

        规则（可根据实验调整权重）：
          含 'multi_entity' 或 'conflict' → [0.10, 0.80, 0.10]
          含 'ambiguous'                  → [0.10, 0.10, 0.80]
          其余（base samples）            → [0.80, 0.10, 0.10]
          多标签时：各规则权重平均

        Args:
            hard_subset_list: list of str
            device:           target device

        Returns:
            (batch, 3) float tensor，每行和为 1
        """
        BASE_LABEL     = torch.tensor([0.80, 0.10, 0.10])
        CONFLICT_LABEL = torch.tensor([0.10, 0.80, 0.10])
        CONTEXT_LABEL  = torch.tensor([0.10, 0.10, 0.80])

        labels = []
        for hs in hard_subset_list:
            tags = set(hs.split("|")) if hs.strip() else set()
            components = []
            if "multi_entity" in tags or "conflict" in tags:
                components.append(CONFLICT_LABEL)
            if "ambiguous" in tags:
                components.append(CONTEXT_LABEL)
            if not components:
                components.append(BASE_LABEL)
            # 多标签时取平均，再归一化
            lbl = torch.stack(components).mean(0)
            lbl = lbl / lbl.sum()
            labels.append(lbl)

        return torch.stack(labels).to(device)


# ─────────────────────────────────────────────────────────
# 路由损失函数
# ─────────────────────────────────────────────────────────
def compute_router_loss(
    routing_logits: torch.Tensor,
    weak_labels: torch.Tensor,
) -> torch.Tensor:
    """
    路由器弱监督损失：软交叉熵（等价于 KL 散度 + 常数项）。

    L_router = - Σ_i  weak_label_i * log(softmax(logit_i))

    Args:
        routing_logits: (batch, num_experts)
        weak_labels:    (batch, num_experts)，每行和为 1

    Returns:
        scalar loss
    """
    log_probs = F.log_softmax(routing_logits, dim=-1)   # (batch, num_experts)
    loss = -(weak_labels * log_probs).sum(dim=-1).mean()
    return loss
