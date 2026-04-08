"""
schema.py
---------
Defines the canonical data schema and prediction output schema for the
entire project. ALL training and evaluation scripts must conform to these
structures. Define once, use everywhere.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


# ------------------------------------------------------------------
# Label space (fixed, do not change after first run)
# ------------------------------------------------------------------

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = 3

# ------------------------------------------------------------------
# Backbone candidate registry (fixed, do not add mid-project)
# Entry format: (hf_hub_name, short_alias, justification)
# ------------------------------------------------------------------

BACKBONE_CANDIDATES = [
    (
        "bert-base-uncased",
        "bert-base",
        "Minimal reference baseline. No domain adaptation. Sets the floor.",
    ),
    (
        "ProsusAI/finbert",
        "finbert",
        "Financial-domain continued pretraining on 4.9B tokens. Primary financial backbone candidate.",
    ),
    (
        "microsoft/deberta-v3-base",
        "deberta-v3-base",
        "Stronger general encoder with disentangled attention. Tests whether domain < architecture.",
    ),
    # Optional fourth candidate - only run if time permits
    # ("yiyanghkust/finbert-tone", "finbert-tone", "Tone-specific finbert variant. Optional."),
]

# ------------------------------------------------------------------
# Subset tag definitions (operational, frozen before any training)
# ------------------------------------------------------------------

SUBSET_DEFINITIONS = """
Subset tag definitions (must be applied at preprocessing time, not eval time):

  multi_entity : The same source text contains >= 2 distinct target entities.
                 Determined by grouping on (source_id, text) and counting unique targets.

  conflict     : The same source text contains >= 2 targets with DIFFERENT gold labels.
                 Requires multi_entity to also be true.
                 Determined by checking label set cardinality > 1 within same text group.

  ambiguous    : Text satisfies ANY of the following proxy rules (applied in order):
                 (a) text token count < 12 (very short, low context)
                 (b) text contains hedge words from HEDGE_LEXICON and has no clear
                     directional keyword from SENTIMENT_LEXICON
                 (c) target does not appear as a substring in the text
                     (implied reference, harder to resolve)

Note: ambiguous is defined by text properties, NOT by model prediction entropy.
      Using entropy would create a circular dependency with the model being evaluated.
"""

HEDGE_LEXICON = {
    "could", "may", "might", "possibly", "potentially", "reportedly",
    "seems", "appears", "likely", "unlikely", "expected", "projected",
    "anticipated", "estimated", "suggests", "indicates",
}

SENTIMENT_LEXICON = {
    "positive": {"surge", "soar", "jump", "rise", "gain", "profit", "beat",
                 "exceed", "strong", "record", "growth", "rally", "upgrade"},
    "negative": {"fall", "drop", "decline", "loss", "miss", "weak", "cut",
                 "downgrade", "warn", "risk", "collapse", "plunge", "slump"},
}


# ------------------------------------------------------------------
# Canonical sample schema
# ------------------------------------------------------------------

@dataclass
class FinSentSample:
    """
    Canonical schema for a single sample across all datasets.
    This is the intermediate format between raw data and model input.
    """
    sample_id: str                    # Globally unique: {source}_{split}_{idx}
    text: str                         # Raw financial text
    target: Optional[str]             # Entity/aspect; None for sentence-level samples
    label: str                        # "positive" | "neutral" | "negative"
    source: str                       # "fpb" | "sentfin" | "finentity" | "finmarba"
    split: str                        # "train" | "dev" | "test"
    subset_tags: List[str] = field(default_factory=list)  # ["multi_entity", "conflict", "ambiguous"]
    source_text_id: Optional[str] = None  # Groups samples from the same headline

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)


# ------------------------------------------------------------------
# Canonical prediction output schema
# ------------------------------------------------------------------

@dataclass
class PredictionRecord:
    """
    Canonical schema for model predictions.
    ALL training scripts must write predictions in this format.
    eval_baselines.py reads ONLY this format.
    """
    sample_id: str
    source: str
    split: str
    text: str
    target: Optional[str]
    true_label: str
    pred_label: str
    pred_probs: List[float]           # [P(neg), P(neu), P(pos)]
    subset_tags: List[str]
    model_alias: str                  # e.g. "finbert_target_marker"
    backbone: str                     # e.g. "ProsusAI/finbert"
    input_mode: str                   # "sentence" | "marker" | "concat"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)


# ------------------------------------------------------------------
# Prediction file I/O
# ------------------------------------------------------------------

def save_predictions(records: List[PredictionRecord], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")
    print(f"[OK] Saved {len(records)} predictions to {path}")


def load_predictions(path: str) -> List[PredictionRecord]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(PredictionRecord.from_dict(json.loads(line.strip())))
    return records


# ------------------------------------------------------------------
# Config schema for reproducibility
# ------------------------------------------------------------------

@dataclass
class TrainConfig:
    """
    Canonical training configuration. Saved alongside every checkpoint.
    """
    model_alias: str
    backbone: str
    input_mode: str
    task_type: str                    # "sentence" | "target" | "multitask"
    datasets_used: List[str]
    max_length: int
    learning_rate: float
    batch_size: int
    num_epochs: int
    warmup_ratio: float
    seed: int
    label2id: dict = field(default_factory=lambda: LABEL2ID)
    notes: str = ""

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            return cls(**json.load(f))
