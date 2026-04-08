"""
index_builder.py
----------------
Encodes retrieval_corpus.jsonl using ProsusAI/finbert (mean pooling)
and inserts all vectors into a remote Milvus server.

Connection: host:port + db_name (no local file mode).

Usage:
    python index_builder.py \
        --corpus     F:\\stage4\\retrieval_corpus.jsonl \
        --backbone   ProsusAI/finbert \
        --host       180.184.67.7 \
        --port       19530 \
        --db_name    DL_project \
        --collection financial_corpus \
        --batch_size 64
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from pymilvus import MilvusClient


# ------------------------------------------------------------------
# Encoding
# ------------------------------------------------------------------

def load_encoder(backbone: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model     = AutoModel.from_pretrained(backbone)
    model.eval()
    model.to(device)
    return tokenizer, model


def encode_batch(
    texts:      List[str],
    tokenizer,
    model,
    device:     torch.device,
    max_length: int = 128,
) -> np.ndarray:
    encoded = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)

    mask       = attention_mask.unsqueeze(-1).float()
    summed     = (output.last_hidden_state * mask).sum(dim=1)
    counts     = mask.sum(dim=1).clamp(min=1e-9)
    embeddings = (summed / counts).cpu().numpy()
    return embeddings


def encode_corpus(
    documents:  List[str],
    tokenizer,
    model,
    device:     torch.device,
    batch_size: int,
) -> np.ndarray:
    all_embeddings = []
    total = len(documents)
    for start in range(0, total, batch_size):
        batch = documents[start:start + batch_size]
        emb   = encode_batch(batch, tokenizer, model, device)
        all_embeddings.append(emb)
        done = min(start + batch_size, total)
        print(f"  encoded {done}/{total}", end="\r")
    print()
    return np.vstack(all_embeddings)


# ------------------------------------------------------------------
# Corpus loading
# ------------------------------------------------------------------

def load_corpus(corpus_path: str):
    query_texts = []
    metadata    = []
    with open(corpus_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = f"Entity: {item['entity']}. Context: {item['context']}"
            query_texts.append(text)
            metadata.append({
                "id":       i,
                "doc_id":   item["doc_id"],
                "entity":   item["entity"],
                "context":  item["context"][:2000],
                "doc_type": item.get("doc_type", "article"),
                "ticker":   item.get("ticker", ""),
            })
    print(f"[OK] Loaded {len(query_texts)} documents from corpus")
    return query_texts, metadata


# ------------------------------------------------------------------
# Milvus server index
# ------------------------------------------------------------------

def build_index(
    embeddings:      np.ndarray,
    metadata:        list,
    host:            str,
    port:            str,
    db_name:         str,
    collection_name: str,
    insert_batch:    int = 2000,
) -> None:
    dim    = embeddings.shape[1]
    uri    = f"http://{host}:{port}"
    client = MilvusClient(uri=uri, db_name=db_name)
    print(f"[OK] Connected to Milvus at {uri}, db={db_name}")

    if client.has_collection(collection_name=collection_name):
        print(f"[INFO] Dropping existing collection '{collection_name}' to rebuild")
        client.drop_collection(collection_name=collection_name)

    client.create_collection(
        collection_name=collection_name,
        dimension=dim,
    )
    print(f"[OK] Collection '{collection_name}' created (dim={dim})")

    total_inserted = 0
    for start in range(0, len(metadata), insert_batch):
        batch_meta = metadata[start:start + insert_batch]
        batch_emb  = embeddings[start:start + insert_batch]

        insert_data = []
        for i, meta in enumerate(batch_meta):
            row = meta.copy()
            row["vector"] = batch_emb[i].tolist()
            insert_data.append(row)

        res = client.insert(collection_name=collection_name, data=insert_data)
        total_inserted += res["insert_count"]
        print(f"  inserted {total_inserted}/{len(metadata)}", end="\r")

    print(f"\n[OK] Inserted {total_inserted} vectors")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OK] Device: {device}")

    if not Path(args.corpus).exists():
        print(f"[FAIL] Corpus not found: {args.corpus}")
        print("       Run build_retrieval_corpus.py first.")
        return

    query_texts, metadata = load_corpus(args.corpus)
    if not query_texts:
        print("[FAIL] Empty corpus.")
        return

    print(f"[INFO] Loading encoder: {args.backbone}")
    tokenizer, model = load_encoder(args.backbone, device)

    print("[INFO] Encoding corpus...")
    embeddings = encode_corpus(query_texts, tokenizer, model, device, args.batch_size)
    print(f"[OK] Embeddings shape: {embeddings.shape}")

    print(f"[INFO] Building Milvus index on {args.host}:{args.port}")
    build_index(
        embeddings, metadata,
        host=args.host,
        port=args.port,
        db_name=args.db_name,
        collection_name=args.collection,
        insert_batch=2000,
    )

    print(f"[OK] Index complete.")
    print(f"     Server    : {args.host}:{args.port}")
    print(f"     DB        : {args.db_name}")
    print(f"     Collection: {args.collection}")
    print(f"     Documents : {len(metadata)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus",     default="F:\\stage4\\retrieval_corpus.jsonl")
    parser.add_argument("--backbone",   default="ProsusAI/finbert")
    parser.add_argument("--host",       default="180.184.67.7")
    parser.add_argument("--port",       default="19530")
    parser.add_argument("--db_name",    default="DL_project")
    parser.add_argument("--collection", default="financial_corpus")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(args)
