"""
preprocess.py
=============
Stage 1 data integration and preprocessing pipeline.

Inputs  (place in the same directory or adjust paths below):
    FinancialPhraseBank/Sentences_50Agree.txt  - Financial PhraseBank
    SEntFiN/SEntFiN.csv                        - SEntFiN entity-level sentiment
    FinEntity/                                 - FinEntity (HuggingFace Arrow dataset)
    FinMarBa/                                  - FinMarBa (HuggingFace Arrow dataset)

Outputs:
    integrated_raw.jsonl        - merged, legality-filtered, NOT yet deduped
    integrated_dedup_v1.jsonl   - after per-task deduplication
    final_integrated.jsonl      - clean 8-field schema, all task types
    audit_parse_errors.csv      - rows that failed to parse (sentfin)
    audit_drop_reasons.csv      - rows dropped during legality filtering
"""

import ast
import json
import os
import sys

import pandas as pd
from datasets import Dataset as HFDataset

# ============================================================
# Config: file paths
# ============================================================
FPB_PATH       = os.path.join("FinancialPhraseBank", "Sentences_50Agree.txt")
SENTFIN_PATH   = os.path.join("SEntFiN", "SEntFiN.csv")
FINENTITY_PATH = os.path.join("FinEntity", "train", "data-00000-of-00001.arrow")
FINMARBA_PATH  = os.path.join("FinMarBa", "train", "data-00000-of-00001.arrow")

OUT_RAW        = "integrated_raw.jsonl"
OUT_DEDUP      = "integrated_dedup_v1.jsonl"
OUT_TRAIN      = "final_integrated.jsonl"
OUT_PARSE_ERR  = "audit_parse_errors.csv"
OUT_DROP       = "audit_drop_reasons.csv"

# Valid label values for semantic tasks
VALID_LABELS = {"positive", "neutral", "negative"}

# Valid values for market_label
VALID_MARKET_LABELS = {"positive", "neutral", "negative"}

# Minimum text length (characters) after normalization
MIN_TEXT_LEN = 10

# ============================================================
# Schema and helpers
# ============================================================

SCHEMA_FIELDS = ["uid", "dataset", "task_type", "text",
                 "label", "target", "market_label", "split"]


def create_row():
    """Return an empty dict conforming to the 8-field schema."""
    return {f: None for f in SCHEMA_FIELDS}


def clean_text(text):
    """
    Normalize whitespace. Returns None for missing or empty input.
    Does NOT coerce NaN to empty string -- empty stays None.
    """
    if pd.isna(text):
        return None
    text = " ".join(str(text).split())
    return text if text else None


def normalize_label(raw):
    """
    Map raw label string to one of {positive, neutral, negative} or None.
    Handles common variant spellings seen in the four datasets.
    """
    if pd.isna(raw) or raw is None:
        return None
    s = str(raw).strip().lower()
    mapping = {
        "positive": "positive", "pos": "positive",
        "neutral":  "neutral",  "neu": "neutral",
        "negative": "negative", "neg": "negative",
    }
    return mapping.get(s, None)


def normalize_market_label(raw):
    """Map FinMarBa aggregate sentiment to {positive, neutral, negative} or None."""
    if pd.isna(raw) or raw is None:
        return None
    s = str(raw).strip().lower()
    mapping = {
        "positive": "positive", "pos": "positive",  "1": "positive",
        "neutral":  "neutral",  "neu": "neutral",   "0": "neutral",
        "negative": "negative", "neg": "negative", "-1": "negative",
        "bull":     "positive",
        "bear":     "negative",
    }
    return mapping.get(s, None)


# ============================================================
# Audit log helpers
# ============================================================

_parse_errors = []   # list[dict]  -- sentfin parse failures
_drop_log     = []   # list[dict]  -- legality-filter drops


def _log_parse_error(dataset, raw_id, error_type, error_msg):
    _parse_errors.append({
        "dataset":    dataset,
        "raw_id":     raw_id,
        "error_type": error_type,
        "error_msg":  str(error_msg)[:300],
    })


def _log_drop(dataset, uid, reason):
    _drop_log.append({
        "dataset": dataset,
        "uid":     uid,
        "reason":  reason,
    })


# ============================================================
# Loader: Financial PhraseBank (FPB)
# ============================================================

def load_fpb(path):
    """
    FPB .txt file: each line is  text@label\n
    Labels are already in {positive, neutral, negative}.
    FPB has no official split; we mark split='train' as the
    dataset is used entirely for training supervision.
    """
    print(f"[FPB] Loading from {path}")
    with open(path, "r", encoding="latin-1") as f:
        lines = f.readlines()
    print(f"[FPB] Raw lines: {len(lines)}")

    rows = []
    for idx, line in enumerate(lines):
        line = line.rstrip("\n").rstrip("\r")
        if not line:
            continue

        # Split on the LAST '@' to separate text from label
        at_pos = line.rfind("@")
        if at_pos == -1:
            _log_drop("fpb", f"fpb_{idx:05d}", "no_@_separator")
            continue

        text_orig = line[:at_pos]
        label_orig = line[at_pos + 1:]

        text  = clean_text(text_orig)
        label = normalize_label(label_orig)

        # Legality checks at load time
        if text is None:
            _log_drop("fpb", f"fpb_{idx:05d}", "empty_text")
            continue
        if label is None:
            _log_drop("fpb", f"fpb_{idx:05d}",
                      f"invalid_label:{label_orig}")
            continue

        row = create_row()
        row.update({
            "uid":       f"fpb_{idx:05d}",
            "dataset":   "fpb",
            "task_type": "sentence_semantic",
            "text":      text,
            "label":     label,
            "split":     "train",
        })
        rows.append(row)

    print(f"[FPB] Loaded: {len(rows)} rows (dropped {len(lines) - len(rows)})")
    return rows


# ============================================================
# Loader: SEntFiN
# ============================================================

def load_sentfin(path):
    """
    SEntFiN CSV columns: Title, Decisions (dict string: entity -> sentiment).
    Each headline expands into N (text, target, label) rows.
    """
    print(f"[SEntFiN] Loading from {path}")
    df = pd.read_csv(path)

    # Column existence guard
    for col in ("Title", "Decisions"):
        if col not in df.columns:
            raise ValueError(
                f"[SEntFiN] Required column '{col}' not found. "
                f"Available: {df.columns.tolist()}"
            )

    print(f"[SEntFiN] Raw rows: {len(df)}")

    rows = []
    for idx, r in df.iterrows():
        # -- Parse Decisions dict --
        try:
            decision_dict = ast.literal_eval(r["Decisions"])
        except Exception as e:
            _log_parse_error("sentfin", idx, "decision_parse_error", e)
            continue

        if not isinstance(decision_dict, dict) or not decision_dict:
            _log_parse_error("sentfin", idx, "empty_or_invalid_dict",
                             f"got: {type(decision_dict)}")
            continue

        text = clean_text(r["Title"])
        if text is None:
            _log_parse_error("sentfin", idx, "empty_text", "Title is null/empty")
            continue

        # -- Expand one headline into (text, target, label) rows --
        for ent, lab in decision_dict.items():
            target = str(ent).strip() if ent else None
            label  = normalize_label(lab)

            if target is None:
                _log_drop("sentfin",
                          f"sentfin_{idx:05d}_?",
                          "empty_target")
                continue
            if label is None:
                _log_drop("sentfin",
                          f"sentfin_{idx:05d}_{ent}",
                          f"invalid_label:{lab}")
                continue

            row = create_row()
            row.update({
                "uid":       f"sentfin_{idx:05d}_{len(rows):04d}",
                "dataset":   "sentfin",
                "task_type": "target_semantic",
                "text":      text,
                "target":    target,
                "label":     label,
                "split":     "train",
            })
            rows.append(row)

    if _parse_errors:
        recent = [e for e in _parse_errors if e["dataset"] == "sentfin"]
        print(f"[WARN] SEntFiN: {len(recent)} rows failed to parse Decisions")
    print(f"[SEntFiN] Loaded: {len(rows)} expanded rows")
    return rows


# ============================================================
# Loader: FinEntity
# ============================================================

def load_finentity(path):
    """
    FinEntity HuggingFace Arrow dataset: {content, annotations: [{value, label, ...}]}.
    Each annotation expands into one (text, target, label) row.
    """
    print(f"[FinEntity] Loading from {path}")
    ds = HFDataset.from_file(path)
    data = ds.to_list()
    print(f"[FinEntity] Raw entries: {len(data)}")

    rows = []
    for entry_idx, entry in enumerate(data):
        text = clean_text(entry.get("content"))
        if text is None:
            _log_drop("finentity",
                      f"finent_entry{entry_idx:06d}",
                      "empty_content")
            continue

        annotations = entry.get("annotations", [])
        if not annotations:
            # Entry has no annotations -- skip silently (no training signal)
            continue

        for ann in annotations:
            target = ann.get("value")
            if not target or not str(target).strip():
                _log_drop("finentity",
                          f"finent_entry{entry_idx:06d}",
                          "empty_target_value")
                continue

            label = normalize_label(ann.get("label"))
            if label is None:
                _log_drop("finentity",
                          f"finent_entry{entry_idx:06d}_{target}",
                          f"invalid_label:{ann.get('label')}")
                continue

            row = create_row()
            row.update({
                "uid":       f"finent_{len(rows):06d}",
                "dataset":   "finentity",
                "task_type": "target_semantic",
                "text":      text,
                "target":    str(target).strip(),
                "label":     label,
                "split":     "train",
            })
            rows.append(row)

    print(f"[FinEntity] Loaded: {len(rows)} expanded rows")
    return rows


# ============================================================
# Loader: FinMarBa
# ============================================================

def _detect_finmarba_label_col(df):
    """
    FinMarBa's aggregate sentiment column may have different names
    depending on the dataset version. This function inspects the
    actual columns and returns the best candidate column name.

    Priority order:
        Global Sentiment -> aggregate_label -> market_sentiment -> overall_label -> Sentiment

    If the chosen column contains dict-like strings (e.g. "{'spy': 1}"),
    it is NOT a valid single-value label column; we skip it and try the next.
    """
    candidates = ["Global Sentiment", "aggregate_label", "market_sentiment",
                  "overall_label", "Sentiment"]

    def _looks_like_dict_col(series):
        """Return True if >10% of non-null values look like dicts."""
        non_null = series.dropna().astype(str)
        if len(non_null) == 0:
            return False
        dict_like = non_null.str.startswith("{").sum()
        return (dict_like / len(non_null)) > 0.10

    for col in candidates:
        if col not in df.columns:
            continue
        if _looks_like_dict_col(df[col]):
            print(f"[WARN] FinMarBa: column '{col}' contains dict-like strings "
                  f"-- not a single-value label, skipping.")
            continue
        print(f"[FinMarBa] Using '{col}' as market_label source.")
        return col

    # Nothing usable found -- report and return None
    print(f"[ERROR] FinMarBa: could not find a valid aggregate label column.")
    print(f"        Available columns: {df.columns.tolist()}")
    print(f"        Sample of first row: {df.iloc[0].to_dict()}")
    return None


def load_finmarba(path):
    """
    FinMarBa HuggingFace Arrow dataset: headline + market sentiment label.
    target = None  (Tickers must NOT be placed in target field).
    split  = 'analysis'  (not used for supervised training).
    """
    print(f"[FinMarBa] Loading from {path}")
    ds = HFDataset.from_file(path)
    df = ds.to_pandas()
    print(f"[FinMarBa] Raw rows: {len(df)}")
    print(f"[FinMarBa] Columns: {df.columns.tolist()}")

    # -- Detect text column --
    text_col = None
    for candidate in ("Title", "headline", "text", "Headline"):
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        raise ValueError(
            f"[FinMarBa] No text column found. "
            f"Available: {df.columns.tolist()}"
        )
    print(f"[FinMarBa] Using '{text_col}' as text source.")

    # -- Detect label column --
    label_col = _detect_finmarba_label_col(df)

    rows = []
    for idx, r in df.iterrows():
        text = clean_text(r[text_col])
        if text is None:
            _log_drop("finmarba", f"finmarba_{idx:05d}", "empty_text")
            continue

        market_label = None
        if label_col is not None:
            market_label = normalize_market_label(r[label_col])
            if market_label is None:
                # Non-null but unmappable value -- log and keep row with None
                raw_val = r[label_col]
                if pd.notna(raw_val):
                    _log_drop("finmarba", f"finmarba_{idx:05d}",
                              f"invalid_market_label:{raw_val}")

        row = create_row()
        row.update({
            "uid":          f"finmarba_{idx:05d}",
            "dataset":      "finmarba",
            "task_type":    "market_aux",
            "text":         text,
            "label":        None,
            "target":       None,   # Tickers are NOT placed in target
            "market_label": market_label,
            "split":        "analysis",
        })
        rows.append(row)

    valid = sum(1 for r in rows if r["market_label"] is not None)
    print(f"[FinMarBa] Loaded: {len(rows)} rows "
          f"({valid} with valid market_label, "
          f"{len(rows) - valid} with market_label=None)")
    return rows


# ============================================================
# Dedup key per task type
# ============================================================

def _dedup_subset(task_type):
    if task_type == "sentence_semantic":
        return ["text", "label"]
    elif task_type == "target_semantic":
        return ["text", "target", "label"]
    elif task_type == "market_aux":
        # Current key: text + market_label only.
        # Scope: correct for simplified auxiliary usage where FinMarBa
        # provides aggregate market signal, not per-ticker events.
        # Limitation: if ticker-level or time-sensitive market modeling
        # is needed later, extend key to ["text", "market_label", "ticker"].
        return ["text", "market_label"]
    return ["text"]


# ============================================================
# Main integration pipeline
# ============================================================

def main_integration():
    print("=" * 60)
    print("Stage 1: Data Integration and Preprocessing")
    print("=" * 60)

    # ---- Load -----------------------------------------------
    all_rows = []
    all_rows.extend(load_fpb(FPB_PATH))
    all_rows.extend(load_sentfin(SENTFIN_PATH))
    all_rows.extend(load_finentity(FINENTITY_PATH))
    all_rows.extend(load_finmarba(FINMARBA_PATH))

    if not all_rows:
        print("[ERROR] No rows loaded. Check file paths.")
        sys.exit(1)

    df = pd.DataFrame(all_rows, columns=SCHEMA_FIELDS)
    print(f"\n[OK] Total rows after loading: {len(df)}")
    print(f"     Schema columns: {df.columns.tolist()}")

    # ---- Legality filtering ---------------------------------
    # Applied AFTER loaders (which already filter at load time)
    # to catch any edge cases missed per-loader.
    n_before = len(df)

    # 1. Empty text (should be caught already, but double-check)
    mask_text = df["text"].isna() | (df["text"].str.len() < MIN_TEXT_LEN)
    for uid in df.loc[mask_text, "uid"]:
        _log_drop("any", uid, "empty_or_short_text_post_load")
    df = df[~mask_text]

    # 2. sentence_semantic / target_semantic must have valid label
    mask_label = (
        df["task_type"].isin(["sentence_semantic", "target_semantic"])
        & (~df["label"].isin(VALID_LABELS))
    )
    for uid in df.loc[mask_label, "uid"]:
        _log_drop("any", uid, f"invalid_label_post_load")
    df = df[~mask_label]

    # 3. target_semantic must have non-null target
    mask_target = (
        (df["task_type"] == "target_semantic")
        & df["target"].isna()
    )
    for uid in df.loc[mask_target, "uid"]:
        _log_drop("any", uid, "null_target_post_load")
    df = df[~mask_target]

    n_dropped = n_before - len(df)
    print(f"\n[OK] Legality filter: {n_before} -> {len(df)} "
          f"(dropped {n_dropped})")

    # ---- Save raw (pre-dedup) --------------------------------
    df.to_json(OUT_RAW, orient="records", lines=True, force_ascii=False)
    print(f"[OK] {OUT_RAW} saved: {len(df)} rows")

    # ---- Deduplication per task type ------------------------
    deduped_parts = []
    for task in df["task_type"].unique():
        subset_df = df[df["task_type"] == task]
        key = _dedup_subset(task)
        deduped = subset_df.drop_duplicates(subset=key)
        n_removed = len(subset_df) - len(deduped)
        print(f"[Dedup] {task}: {len(subset_df)} -> {len(deduped)} "
              f"(removed {n_removed})")
        deduped_parts.append(deduped)

    df_dedup = pd.concat(deduped_parts, ignore_index=True)
    df_dedup.to_json(OUT_DEDUP, orient="records", lines=True,
                     force_ascii=False)
    print(f"[OK] {OUT_DEDUP} saved: {len(df_dedup)} rows")

    # ---- Final integrated file: clean 8-field schema only -----
    # Contains ALL task types (sentence_semantic, target_semantic, market_aux).
    # When training a semantic model, filter by task_type explicitly:
    #   df = pd.read_json("final_integrated.jsonl", lines=True)
    #   df_sem = df[df["task_type"] != "market_aux"]
    # Do NOT use the full file directly as a supervised training set.
    df_train = df_dedup[SCHEMA_FIELDS].copy()
    df_train.to_json(OUT_TRAIN, orient="records", lines=True,
                     force_ascii=False)
    print(f"[OK] {OUT_TRAIN} saved: {len(df_train)} rows (all task types)")
    sem_count = (df_train["task_type"] != "market_aux").sum()
    aux_count = (df_train["task_type"] == "market_aux").sum()
    print(f"     semantic supervision rows : {sem_count}")
    print(f"     market_aux (auxiliary) rows: {aux_count}")

    # ---- Save audit logs ------------------------------------
    if _parse_errors:
        pd.DataFrame(_parse_errors).to_csv(OUT_PARSE_ERR, index=False)
        print(f"[AUDIT] {OUT_PARSE_ERR}: {len(_parse_errors)} parse errors")
    else:
        print("[AUDIT] No parse errors.")

    if _drop_log:
        pd.DataFrame(_drop_log).to_csv(OUT_DROP, index=False)
        print(f"[AUDIT] {OUT_DROP}: {len(_drop_log)} dropped rows")
    else:
        print("[AUDIT] No rows dropped (all loaders clean).")

    # ---- Summary --------------------------------------------
    print("\n" + "=" * 60)
    print("Final dataset summary")
    print("=" * 60)
    print(df_train.groupby(["dataset", "task_type"]).size()
          .rename("count").to_string())

    print("\n--- label distribution (semantic tasks) ---")
    sem = df_train[df_train["task_type"] != "market_aux"]
    print(sem["label"].value_counts(dropna=False).to_string())

    print("\n--- market_label distribution (market_aux) ---")
    mkt = df_train[df_train["task_type"] == "market_aux"]
    print(mkt["market_label"].value_counts(dropna=False).to_string())

    print("\n--- null counts per field ---")
    print(df_train.isnull().sum().to_string())

    print("\n[DONE] Preprocessing complete.")
    return df_train


if __name__ == "__main__":
    main_integration()
