"""
losses/diversity_loss.py
Anti-collapse 表征分化约束（Diversity Loss / L_div）

目标：防止三个 Expert 的特化表征坍缩为相同向量（expert collapse）。
只在 hard/conflict/ambiguous 样本上启用，避免对普通样本施加不必要约束。

损失定义（基于余弦相似度）：
  L_div = ReLU( mean_{i≠j}[ cosine_sim(E_i, E_j) ] - margin )

  其中 E_i 是第 i 个 Expert 的特化表征（expert_repr）。
  最小化 L_div ≡ 最大化 Expert 间的余弦距离。
  margin 为松弛量（默认 0），防止过度惩罚。

使用方式：
  from losses.diversity_loss import DiversityLoss

  div_loss_fn = DiversityLoss(weight=0.1, margin=0.0)
  loss = div_loss_fn(expert_reprs, hard_subset_list)  # scalar

  # 或使用底层函数
  from losses.diversity_loss import diversity_loss, build_hard_mask
  mask = build_hard_mask(hard_subset_list, device)
  loss = diversity_loss(expert_reprs, mask)
"""

from __future__ import annotations
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────
# Hard 样本掩码构建
# ─────────────────────────────────────────────────────────
def build_hard_mask(
    hard_subset_list: list[str],
    device: torch.device,
) -> torch.Tensor:
    """
    从 hard_subset 字段构建 hard 样本掩码。
    hard_subset 不为空（即有任意标签）的样本视为 hard。

    Args:
        hard_subset_list: list of str，每个元素为 hard_subset 字段值
        device:           target device

    Returns:
        (batch,) bool tensor
    """
    mask = [bool(hs.strip()) for hs in hard_subset_list]
    return torch.tensor(mask, dtype=torch.bool, device=device)


def build_subset_masks(
    hard_subset_list: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    构建各细粒度子集掩码（用于分析和调试）。

    Returns:
        dict:
          'multi_entity' : (batch,) bool
          'conflict'     : (batch,) bool
          'ambiguous'    : (batch,) bool
          'any_hard'     : (batch,) bool
    """
    me  = [("multi_entity" in hs) for hs in hard_subset_list]
    cf  = [("conflict"     in hs) for hs in hard_subset_list]
    amb = [("ambiguous"    in hs) for hs in hard_subset_list]
    any_hard = [bool(hs.strip())  for hs in hard_subset_list]

    return {
        "multi_entity": torch.tensor(me,       dtype=torch.bool, device=device),
        "conflict":     torch.tensor(cf,       dtype=torch.bool, device=device),
        "ambiguous":    torch.tensor(amb,      dtype=torch.bool, device=device),
        "any_hard":     torch.tensor(any_hard, dtype=torch.bool, device=device),
    }


# ─────────────────────────────────────────────────────────
# 底层 Diversity Loss 函数
# ─────────────────────────────────────────────────────────
def diversity_loss(
    expert_reprs: list[torch.Tensor],
    hard_mask: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """
    计算 Expert 表征分化损失。

    算法：
      1. 用 hard_mask 筛选 hard 样本
      2. 对每个 expert 表征做 L2 归一化
      3. 计算所有 expert pair 的余弦相似度
      4. 取平均后施加 ReLU(· - margin)

    Args:
        expert_reprs: list of (batch, hidden_size) tensor，
                      顺序为 [base_repr, conflict_repr, context_repr]
        hard_mask:    (batch,) bool tensor，True 为 hard 样本
        margin:       余弦相似度松弛量（默认 0.0）

    Returns:
        scalar loss（若无 hard 样本则返回 0.0）
    """
    device = expert_reprs[0].device

    # 若当前 batch 无 hard 样本，跳过（避免无效梯度）
    if not hard_mask.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 筛选 hard 样本
    hard_reprs = [r[hard_mask] for r in expert_reprs]          # 每个 (n_hard, d)

    # L2 归一化（cosine similarity = 归一化后的点积）
    norm_reprs = [F.normalize(r, p=2, dim=-1) for r in hard_reprs]

    # 计算所有 expert pair 的余弦相似度
    pair_sims = []
    for i, j in combinations(range(len(norm_reprs)), 2):
        # (n_hard,) → 每个样本上第 i、j expert 表征的余弦相似度
        sim = (norm_reprs[i] * norm_reprs[j]).sum(dim=-1)
        pair_sims.append(sim)

    if not pair_sims:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 均值余弦相似度（越低 = experts 越分化 = 越好）
    mean_sim = torch.stack(pair_sims, dim=0).mean()

    # ReLU(mean_sim - margin)：只在相似度超过 margin 时产生梯度
    loss = F.relu(mean_sim - margin)
    return loss


# ─────────────────────────────────────────────────────────
# 封装模块
# ─────────────────────────────────────────────────────────
class DiversityLoss(nn.Module):
    """
    Anti-collapse 表征分化约束，封装为 nn.Module。

    使用示例：
        div_loss_fn = DiversityLoss(weight=0.1, margin=0.0)

        # 在 forward 中：
        loss = div_loss_fn(
            expert_reprs=model_out['expert_reprs'],   # list of (batch, d)
            hard_subset_list=batch['hard_subset'],    # list of str
        )

    Args:
        weight: 总损失中 L_div 的权重系数 λ_div（建议 0.05~0.2）
        margin: 余弦相似度松弛量（建议 0.0~0.1）
    """

    def __init__(self, weight: float = 0.1, margin: float = 0.0):
        super().__init__()
        self.weight = weight
        self.margin = margin

    def forward(
        self,
        expert_reprs: list[torch.Tensor],
        hard_subset_list: list[str],
    ) -> torch.Tensor:
        """
        Args:
            expert_reprs:     list of (batch, hidden_size)
            hard_subset_list: list of str

        Returns:
            weighted scalar loss: weight * L_div
        """
        device = expert_reprs[0].device
        hard_mask = build_hard_mask(hard_subset_list, device)
        raw_loss  = diversity_loss(expert_reprs, hard_mask, self.margin)
        return self.weight * raw_loss

    def extra_repr(self) -> str:
        return f"weight={self.weight}, margin={self.margin}"


# ─────────────────────────────────────────────────────────
# 诊断工具：Expert 表征分析
# ─────────────────────────────────────────────────────────
@torch.no_grad()
def analyze_expert_diversity(
    expert_reprs: list[torch.Tensor],
    hard_subset_list: list[str] | None = None,
) -> dict:
    """
    计算 Expert 表征多样性诊断指标（推理/可视化用，不计算梯度）。

    Args:
        expert_reprs:     list of (batch, hidden_size)
        hard_subset_list: 可选，若提供则分别统计 hard/non-hard 样本

    Returns:
        dict:
          'mean_cosine_sim_all'   : 所有样本上的均值余弦相似度
          'mean_cosine_sim_hard'  : hard 样本上的均值余弦相似度（若提供）
          'mean_cosine_sim_easy'  : non-hard 样本上的均值（若提供）
          'pairwise_sims'         : dict of 'Base-Conflict'/'Base-Context'/'Conflict-Context' → mean sim
    """
    norm_reprs = [F.normalize(r, p=2, dim=-1) for r in expert_reprs]

    pair_names = ["Base-Conflict", "Base-Context", "Conflict-Context"]
    pairwise = {}
    all_pair_sims = []
    for (i, j), name in zip(combinations(range(len(norm_reprs)), 2), pair_names):
        sim = (norm_reprs[i] * norm_reprs[j]).sum(dim=-1)   # (batch,)
        pairwise[name] = sim.mean().item()
        all_pair_sims.append(sim)

    result = {
        "mean_cosine_sim_all": torch.stack(all_pair_sims).mean().item(),
        "pairwise_sims":       pairwise,
    }

    if hard_subset_list is not None:
        device = expert_reprs[0].device
        masks = build_subset_masks(hard_subset_list, device)
        hard_mask = masks["any_hard"]
        easy_mask = ~hard_mask

        if hard_mask.any():
            hard_sims = torch.stack([
                (norm_reprs[i][hard_mask] * norm_reprs[j][hard_mask]).sum(dim=-1)
                for i, j in combinations(range(len(norm_reprs)), 2)
            ])
            result["mean_cosine_sim_hard"] = hard_sims.mean().item()

        if easy_mask.any():
            easy_sims = torch.stack([
                (norm_reprs[i][easy_mask] * norm_reprs[j][easy_mask]).sum(dim=-1)
                for i, j in combinations(range(len(norm_reprs)), 2)
            ])
            result["mean_cosine_sim_easy"] = easy_sims.mean().item()

    return result
