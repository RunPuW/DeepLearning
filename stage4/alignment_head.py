"""
alignment_head.py
-----------------
Market alignment auxiliary head and loss.

The alignment head predicts whether the market reaction (price movement)
is consistent with the semantic sentiment. It is an AUXILIARY signal only.
The alpha weight keeps it from overriding the main sentiment task.

Market label mapping:
    negative -> 0
    neutral  -> 1
    positive -> 2
    missing  -> -1 (ignored in loss via ignore_index)

Usage:
    head    = MarketAlignmentHead(hidden_size=768)
    loss_fn = AlignmentLoss(alpha=0.15)
    logits  = head(cls_features)
    loss    = loss_fn(logits, market_labels)
"""

import torch
import torch.nn as nn
from typing import List, Optional


# ------------------------------------------------------------------
# Label mapping
# ------------------------------------------------------------------

MARKET_LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
MARKET_ID2LABEL = {v: k for k, v in MARKET_LABEL2ID.items()}
MARKET_IGNORE   = -1   # assigned when market_label is null


def encode_market_labels(raw_labels: List[Optional[str]]) -> torch.Tensor:
    """
    Converts a list of raw market label strings (or None) to a
    LongTensor. None maps to MARKET_IGNORE (-1) and is excluded
    from the loss calculation.
    """
    ids = []
    for label in raw_labels:
        if label is None or label not in MARKET_LABEL2ID:
            ids.append(MARKET_IGNORE)
        else:
            ids.append(MARKET_LABEL2ID[label])
    return torch.tensor(ids, dtype=torch.long)


# ------------------------------------------------------------------
# Alignment head
# ------------------------------------------------------------------

class MarketAlignmentHead(nn.Module):
    """
    Small MLP that predicts market reaction from the backbone CLS token.
    Operates as a side branch: it does not affect the main sentiment head.
    """

    def __init__(self, hidden_size: int = 768, num_classes: int = 3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, cls_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_features: (batch_size, hidden_size) CLS token from backbone
        Returns:
            logits: (batch_size, num_classes) market prediction logits
        """
        return self.network(cls_features)


# ------------------------------------------------------------------
# Alignment loss
# ------------------------------------------------------------------

class AlignmentLoss(nn.Module):
    """
    Auxiliary cross-entropy loss scaled by alpha.
    alpha must be small (default 0.15) to prevent the model from
    abandoning semantic sentiment in favor of market prediction.

    Samples with market_label = None are automatically excluded via
    ignore_index=-1.
    """

    def __init__(self, alpha: float = 0.15):
        super().__init__()
        if alpha >= 1.0:
            raise ValueError(
                f"alpha={alpha} is too large. Keep alpha < 0.3 to protect the main task."
            )
        self.alpha     = alpha
        self.criterion = nn.CrossEntropyLoss(ignore_index=MARKET_IGNORE)

    def forward(
        self,
        market_logits: torch.Tensor,
        market_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns alpha * cross_entropy(logits, labels).
        If all labels in the batch are MARKET_IGNORE, returns 0.
        """
        valid = (market_labels != MARKET_IGNORE).sum().item()
        if valid == 0:
            return torch.tensor(0.0, device=market_logits.device, requires_grad=True)
        base_loss = self.criterion(market_logits, market_labels)
        return self.alpha * base_loss


# ------------------------------------------------------------------
# Sanity check (run directly: python alignment_head.py)
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("[INFO] Running alignment head sanity check")

    dummy_features = torch.randn(4, 768)
    dummy_labels   = encode_market_labels(["positive", "negative", None, "neutral"])
    print(f"  market labels tensor: {dummy_labels}")

    head    = MarketAlignmentHead(hidden_size=768, num_classes=3)
    loss_fn = AlignmentLoss(alpha=0.15)

    logits = head(dummy_features)
    loss   = loss_fn(logits, dummy_labels)

    assert logits.shape == (4, 3), f"[FAIL] logits shape {logits.shape}"
    print(f"  logits shape : {logits.shape}  [PASS]")
    print(f"  alignment loss (scaled): {loss.item():.4f}  [PASS]")
    print("[OK] Alignment head is ready for integration")
