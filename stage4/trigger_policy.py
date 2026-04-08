"""
trigger_policy.py
-----------------
Retrieval trigger policies using a remote Milvus server.

Three production policies:
  1. always_on_policy          : retrieves for every sample (baseline)
  2. conditional_policy        : retrieves only when uncertainty > threshold
  3. router_triggered_policy   : stub interface for 3-hao router integration

Budget control is enforced per batch via RetrievalBudget.

Usage:
    trigger = RetrievalTrigger(
        host="180.184.67.7", port="19530",
        db_name="DL_project", collection="financial_corpus",
        backbone="ProsusAI/finbert",
    )
    results = trigger.conditional_policy(text, entity, uncertainty=0.6)
"""

from typing import Dict, List, Optional

import torch
from transformers import AutoModel, AutoTokenizer
from pymilvus import MilvusClient


# ------------------------------------------------------------------
# Encoder
# ------------------------------------------------------------------

class FinbertEncoder:
    """
    Encodes query strings using mean pooling over finbert last hidden state.
    Must match the backbone used in index_builder.py.
    """

    def __init__(self, backbone: str, device: torch.device):
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.model     = AutoModel.from_pretrained(backbone)
        self.model.eval()
        self.model.to(device)
        self.device = device

    def encode(self, text: str, max_length: int = 128) -> List[float]:
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        mask      = attention_mask.unsqueeze(-1).float()
        summed    = (output.last_hidden_state * mask).sum(dim=1)
        counts    = mask.sum(dim=1).clamp(min=1e-9)
        embedding = (summed / counts).squeeze(0).cpu().numpy()
        return embedding.tolist()


# ------------------------------------------------------------------
# Budget controller
# ------------------------------------------------------------------

class RetrievalBudget:
    """
    Limits retrieval calls per batch to prevent latency spikes
    when many samples exceed the uncertainty threshold at once.
    """

    def __init__(self, max_per_batch: int = 16):
        self.max_per_batch = max_per_batch
        self.used          = 0

    def reset(self):
        self.used = 0

    def can_retrieve(self) -> bool:
        return self.used < self.max_per_batch

    def consume(self):
        self.used += 1


# ------------------------------------------------------------------
# Main trigger class
# ------------------------------------------------------------------

class RetrievalTrigger:
    """
    Wraps Milvus search with three retrieval policies.
    Connects to a remote Milvus server (no local file mode).

    All policies return List[Dict] with fields:
        doc_id, entity, context, doc_type, ticker, score
    Empty list means retrieval was skipped or returned no results.
    """

    def __init__(
        self,
        host:            str = "180.184.67.7",
        port:            str = "19530",
        db_name:         str = "DL_project",
        collection:      str = "financial_corpus",
        backbone:        str = "ProsusAI/finbert",
        device:          Optional[torch.device] = None,
        top_k:           int = 3,
        budget_per_batch: int = 16,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        uri           = f"http://{host}:{port}"
        self.client   = MilvusClient(uri=uri, db_name=db_name)
        self.collection = collection
        self.encoder  = FinbertEncoder(backbone, device)
        self.top_k    = top_k
        self.budget   = RetrievalBudget(max_per_batch=budget_per_batch)

        print(f"[OK] RetrievalTrigger connected to {uri}, db={db_name}, collection={collection}")

    # ------------------------------------------------------------------
    # Internal search
    # ------------------------------------------------------------------

    def _search(self, query: str) -> List[Dict]:
        query_vector = self.encoder.encode(query)
        raw = self.client.search(
            collection_name=self.collection,
            data=[query_vector],
            limit=self.top_k,
            output_fields=["doc_id", "entity", "context", "doc_type", "ticker"],
        )
        results = []
        for hits in raw:
            for hit in hits:
                results.append({
                    "doc_id":   hit["entity"]["doc_id"],
                    "entity":   hit["entity"]["entity"],
                    "context":  hit["entity"]["context"],
                    "doc_type": hit["entity"].get("doc_type", "article"),
                    "ticker":   hit["entity"]["ticker"],
                    "score":    float(hit["distance"]),
                })
        return results

    # ------------------------------------------------------------------
    # Policy 1: always-on
    # ------------------------------------------------------------------

    def always_on_policy(self, text: str, entity: str) -> List[Dict]:
        """
        Retrieves for every sample. Used as the always-on baseline.
        """
        query = f"Entity: {entity}. Context: {text}"
        return self._search(query)

    # ------------------------------------------------------------------
    # Policy 2: conditional
    # ------------------------------------------------------------------

    def conditional_policy(
        self,
        text:        str,
        entity:      str,
        uncertainty: float,
        threshold:   float = 0.5,
    ) -> List[Dict]:
        """
        Retrieves only when model uncertainty exceeds threshold.
        uncertainty = 1 - max(softmax(logits)), range [0, 1].
        Returns empty list when retrieval is skipped.
        """
        if uncertainty <= threshold:
            return []
        if not self.budget.can_retrieve():
            return []
        query   = f"Entity: {entity}. Context: {text}"
        results = self._search(query)
        if results:
            self.budget.consume()
        return results

    # ------------------------------------------------------------------
    # Policy 3: router-triggered (interface for 3-hao router)
    # ------------------------------------------------------------------

    def router_triggered_policy(
        self,
        text:          str,
        entity:        str,
        router_signal: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Interface for 3-hao router integration.

        router_signal expected fields:
            should_retrieve : bool   -- router decision to retrieve
            expert_type     : str    -- "base" | "conflict" | "context"
            confidence      : float  -- router confidence score

        Context expert always retrieves when should_retrieve=True.
        Other experts only retrieve when confidence < 0.7.
        If router_signal is None, returns [].

        3-hao: pass your router output dict directly as router_signal.
        """
        if router_signal is None:
            return []

        if not router_signal.get("should_retrieve", False):
            return []

        expert_type = router_signal.get("expert_type", "base")
        if expert_type != "context":
            if router_signal.get("confidence", 1.0) >= 0.7:
                return []

        if not self.budget.can_retrieve():
            return []

        query   = f"Entity: {entity}. Context: {text}"
        results = self._search(query)
        if results:
            self.budget.consume()
        return results

    def reset_budget(self):
        self.budget.reset()


# ------------------------------------------------------------------
# Batch retrieval helper
# ------------------------------------------------------------------

def batch_retrieve(
    trigger:         RetrievalTrigger,
    texts:           List[str],
    entities:        List[str],
    policy:          str = "always_on",
    uncertainties:   Optional[List[float]] = None,
    threshold:       float = 0.5,
    router_signals:  Optional[List[Optional[Dict]]] = None,
) -> List[List[Dict]]:
    """
    Run retrieval for a list of samples.
    policy: "always_on" | "conditional" | "router"
    Returns List[List[Dict]], one result list per sample.
    """
    trigger.reset_budget()
    results = []

    for i, (text, entity) in enumerate(zip(texts, entities)):
        if policy == "always_on":
            r = trigger.always_on_policy(text, entity)
        elif policy == "conditional":
            unc = uncertainties[i] if uncertainties else 0.0
            r   = trigger.conditional_policy(text, entity, unc, threshold)
        elif policy == "router":
            sig = router_signals[i] if router_signals else None
            r   = trigger.router_triggered_policy(text, entity, sig)
        else:
            raise ValueError(f"Unknown policy: {policy}")
        results.append(r)

    return results
