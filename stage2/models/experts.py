"""
models/experts.py
三类专家模块 + 完整金融情感分析主模型 (FinSentModel)

Expert 分类：
  Base Expert    : 普通样本（单实体、直接语义），backbone 直接可处理
  Conflict Expert: 多实体/冲突样本，需分离不同实体的情绪极性
  Context Expert : 上下文不足/歧义强样本，配合检索结果做辅助判断

FinSentModel 总体结构：
  ┌─ Backbone (FinBERT/BERT encoder) ────────────────┐
  │  输入: [CLS] text [SEP] (target) [SEP]           │
  │  输出: CLS 表征 h ∈ R^d                           │
  └───────────────────────────────────────────────────┘
                │
  ┌─ Router ───────────────────────────────────────────┐
  │  输入: h + aux_signals                             │
  │  输出: 路由权重 α = [α_base, α_conf, α_ctx]        │
  └───────────────────────────────────────────────────┘
        α[0]      α[1]       α[2]
         │         │          │
    Base Expert  Conflict  Context
      Adapter    Expert    Expert
         │         │          │
         └────┬────┘          │
              │  加权融合      │
              └───────────────┘
                     │
              Classifier Head → logits (3 类)
                     │
           (可选) Alignment Head → logits (市场对齐)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from models.router import (
    WeaklySupervisedRouter,
    compute_router_loss,
    build_aux_signals,
)


# ─────────────────────────────────────────────────────────
# 标签映射
# ─────────────────────────────────────────────────────────
LABEL2ID = {"positive": 0, "neutral": 1, "negative": 2}
ID2LABEL = {0: "positive", 1: "neutral", 2: "negative"}

MARKET_LABEL2ID = {"positive": 0, "neutral": 1, "negative": 2}
MARKET_ID2LABEL = {0: "positive", 1: "neutral", 2: "negative"}


# ─────────────────────────────────────────────────────────
# Expert Adapter（轻量级瓶颈适配器）
# ─────────────────────────────────────────────────────────
class ExpertAdapter(nn.Module):
    """
    轻量级 Expert Adapter，作用于 backbone CLS 表征之上。

    结构（带残差）：
      x → Linear(d, bottleneck) → GELU → Dropout
        → Linear(bottleneck, d) → LayerNorm(x + output)
        → expert_repr_proj(h)   （用于 diversity loss 和可视化）

    参数初始化：up_proj 权重和偏置归零，确保训练开始时
    每个 expert 行为与 backbone 一致（稳定初始化）。
    """

    def __init__(
        self,
        hidden_size: int = 768,
        bottleneck_size: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.act       = nn.GELU()
        self.dropout   = nn.Dropout(dropout)
        self.up_proj   = nn.Linear(bottleneck_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

        # 专家特化投影，用于 diversity loss（线性，独立于主路径）
        self.expert_repr_proj = nn.Linear(hidden_size, hidden_size)

        # 稳定初始化：up_proj 设为零，residual 路径保持 identity
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (batch, hidden_size) CLS 表征

        Returns:
            dict:
              'hidden':       (batch, hidden_size) adapted 表征（含残差）
              'expert_repr':  (batch, hidden_size) 专家特化表征（用于 diversity loss）
        """
        # Adapter 前向（带残差）
        h = self.down_proj(x)
        h = self.act(h)
        h = self.dropout(h)
        h = self.up_proj(h)
        hidden = self.layer_norm(x + h)

        # 专家特化表征（仅用于 diversity loss 和可视化，不进入主预测路径）
        expert_repr = self.expert_repr_proj(hidden)

        return {"hidden": hidden, "expert_repr": expert_repr}


# ─────────────────────────────────────────────────────────
# 完整金融情感分析主模型
# ─────────────────────────────────────────────────────────
class FinSentModel(nn.Module):
    """
    结构化金融情感分析系统主模型（分工2：Router + Experts）。

    支持两种输入模式：
      sentence_semantic : tokenizer(text)             → [CLS] text [SEP]
      target_semantic   : tokenizer(text, target)     → [CLS] text [SEP] target [SEP]

    损失项（由 FinSentTrainer 调用，此处仅输出原始 logits）：
      L_sent   : sentence_semantic 样本的主任务交叉熵
      L_target : target_semantic  样本的主任务交叉熵
      L_router : 路由器弱监督软交叉熵
      L_div    : Expert 表征多样性约束（只在 hard 样本上）
      L_align  : 对齐辅助头损失（market_aux 样本，可选）

    Args:
        backbone_name:       HuggingFace 模型标识符，默认 'ProsusAI/finbert'
        num_labels:          情感类别数，默认 3
        hidden_size:         backbone 输出维度，默认 768
        bottleneck_size:     Expert Adapter 瓶颈维度，默认 256
        dropout:             dropout 概率
        use_alignment_head:  是否启用语义-市场对齐辅助头
        num_experts:         专家数量，默认 3（不建议修改）
    """

    def __init__(
        self,
        backbone_name: str = "ProsusAI/finbert",
        num_labels: int = 3,
        hidden_size: int = 768,
        bottleneck_size: int = 256,
        dropout: float = 0.1,
        use_alignment_head: bool = False,
        num_experts: int = 3,
    ):
        super().__init__()
        self.backbone_name       = backbone_name
        self.num_labels          = num_labels
        self.hidden_size         = hidden_size
        self.use_alignment_head  = use_alignment_head

        # ── Backbone ──────────────────────────────────────
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.input_dropout = nn.Dropout(dropout)

        # ── Router ────────────────────────────────────────
        self.router = WeaklySupervisedRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            dropout=dropout,
        )

        # ── Expert Adapters（三路并行）────────────────────
        self.base_expert     = ExpertAdapter(hidden_size, bottleneck_size, dropout)
        self.conflict_expert = ExpertAdapter(hidden_size, bottleneck_size, dropout)
        self.context_expert  = ExpertAdapter(hidden_size, bottleneck_size, dropout)

        # ── 主任务分类头（sentence/target 共享）──────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

        # ── 语义-市场对齐辅助头（可选）───────────────────
        if use_alignment_head:
            self.alignment_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 3),   # positive / neutral / negative (市场方向)
            )

    # ── 编码 ──────────────────────────────────────────────
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        用 backbone 编码输入，提取 CLS 表征。

        Returns:
            (batch, hidden_size)
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_hidden = outputs.last_hidden_state[:, 0, :]  # CLS token
        return self.input_dropout(cls_hidden)

    # ── 主前向 ────────────────────────────────────────────
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        aux_signals: torch.Tensor | None = None,
        hard_subset: list[str] | None = None,
        return_expert_repr: bool = False,
        # 消融选项（由 ablation 脚本使用）
        ablation_no_router: bool = False,
        ablation_no_conflict: bool = False,
        ablation_no_context: bool = False,
    ) -> dict:
        """
        Args:
            input_ids:          (batch, seq_len)
            attention_mask:     (batch, seq_len)
            token_type_ids:     (batch, seq_len) 可选
            aux_signals:        (batch, 4) 辅助路由信号；None 则自动用零填充
            hard_subset:        list of str，用于构建弱监督路由标签（训练时提供）
            return_expert_repr: True 时返回三路 expert 特化表征（用于 diversity loss）
            ablation_no_router:   消融：均匀路由（不使用 router）
            ablation_no_conflict: 消融：关闭 Conflict Expert（权重归零）
            ablation_no_context:  消融：关闭 Context Expert（权重归零）

        Returns:
            dict（根据参数不同，字段有所增减）：
              'logits'           : (batch, num_labels) 主任务 logits
              'routing_weights'  : (batch, num_experts) 路由权重
              'routing_logits'   : (batch, num_experts) 路由原始 logits
              'fused_hidden'     : (batch, hidden_size) 融合后表征
              'weak_labels'      : (batch, num_experts) 弱监督标签（提供 hard_subset 时）
              'alignment_logits' : (batch, 3) 对齐头 logits（启用 alignment_head 时）
              'expert_reprs'     : list of (batch, hidden_size)（return_expert_repr=True 时）
        """
        # 1. Backbone 编码
        cls_hidden = self.encode(input_ids, attention_mask, token_type_ids)

        # 2. Router（或均匀路由消融）
        if ablation_no_router:
            # 消融：uniform routing
            bsz = cls_hidden.size(0)
            routing_weights = torch.full(
                (bsz, 3), 1.0 / 3.0, device=cls_hidden.device
            )
            routing_logits = routing_weights.log()  # 占位
        else:
            router_out = self.router(cls_hidden, aux_signals)
            routing_weights = router_out["routing_weights"]   # (batch, 3)
            routing_logits  = router_out["routing_logits"]

        # 3. 消融：关闭特定 expert 并重归一化权重
        if ablation_no_conflict or ablation_no_context:
            rw = routing_weights.clone()
            if ablation_no_conflict:
                rw[:, 1] = 0.0
            if ablation_no_context:
                rw[:, 2] = 0.0
            rw_sum = rw.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            routing_weights = rw / rw_sum

        # 4. 三路 Expert 前向
        base_out     = self.base_expert(cls_hidden)
        conflict_out = self.conflict_expert(cls_hidden)
        context_out  = self.context_expert(cls_hidden)

        # 5. 加权融合
        fused = (
            routing_weights[:, 0:1] * base_out["hidden"]     +
            routing_weights[:, 1:2] * conflict_out["hidden"] +
            routing_weights[:, 2:3] * context_out["hidden"]
        )                                                       # (batch, hidden_size)

        # 6. 主任务分类头
        logits = self.classifier(fused)                        # (batch, num_labels)

        result = {
            "logits":          logits,
            "routing_weights": routing_weights,
            "routing_logits":  routing_logits,
            "fused_hidden":    fused,
        }

        # 7. 弱监督路由标签（训练时提供 hard_subset）
        if hard_subset is not None:
            weak_labels = WeaklySupervisedRouter.build_weak_labels(
                hard_subset, cls_hidden.device
            )
            result["weak_labels"] = weak_labels

        # 8. 语义-市场对齐辅助头
        if self.use_alignment_head and hasattr(self, "alignment_head"):
            result["alignment_logits"] = self.alignment_head(fused)

        # 9. Expert 特化表征（用于 diversity loss 和可视化）
        if return_expert_repr:
            result["expert_reprs"] = [
                base_out["expert_repr"],
                conflict_out["expert_repr"],
                context_out["expert_repr"],
            ]

        return result

    # ── 工具方法 ──────────────────────────────────────────
    def get_routing_entropy(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """
        计算路由分布熵 H = -Σ α_i log(α_i)。
        熵高 → 路由不确定 → 倾向 Context Expert。

        Returns:
            (batch,) float tensor
        """
        eps = 1e-8
        return -(routing_weights * (routing_weights + eps).log()).sum(dim=-1)

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> dict:
        """
        推理接口，返回预测标签、置信度、路由权重。

        Returns:
            dict:
              'pred_labels':     list of str
              'probs':           (batch, num_labels) 概率
              'routing_weights': (batch, 3) 路由权重
              'routing_entropy': (batch,) 路由熵
        """
        was_training = self.training
        self.eval()

        out = self.forward(input_ids, attention_mask, token_type_ids)
        probs    = F.softmax(out["logits"], dim=-1)
        pred_ids = probs.argmax(dim=-1)

        result = {
            "pred_labels":     [ID2LABEL[i.item()] for i in pred_ids],
            "probs":           probs,
            "routing_weights": out["routing_weights"],
            "routing_entropy": self.get_routing_entropy(out["routing_weights"]),
        }

        if was_training:
            self.train()
        return result
