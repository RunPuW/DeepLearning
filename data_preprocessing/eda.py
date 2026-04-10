"""
eda.py
======
Stage 2 exploratory data analysis for the integrated financial dataset.

Input:
    final_train.jsonl           - produced by preprocess.py
    integrated_raw.jsonl        - produced by preprocess.py (for dedup comparison)
    integrated_dedup_v1.jsonl   - produced by preprocess.py (for dedup comparison)

Outputs (saved to ./eda_outputs/):
    eda_outputs/01_dataset_task_dist.png
    eda_outputs/02_label_dist.png
    eda_outputs/03_market_label_dist.png
    eda_outputs/04_text_length_dist.png
    eda_outputs/05_text_length_by_dataset.png
    eda_outputs/06_target_top20.png
    eda_outputs/07_label_by_dataset.png
    eda_outputs/eda_summary.json

Run:
    python eda.py
"""

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ============================================================
# Config
# ============================================================
TRAIN_PATH  = "final_integrated.jsonl"
RAW_PATH    = "integrated_raw.jsonl"
DEDUP_PATH  = "integrated_dedup_v1.jsonl"
OUT_DIR     = "eda_outputs"

os.makedirs(OUT_DIR, exist_ok=True)

# Plot style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi":      120,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


# ============================================================
# Load data
# ============================================================

def load_data():
    print("[EDA] Loading data files...")

    df_train = pd.read_json(TRAIN_PATH, lines=True)
    print(f"[OK] final_train.jsonl: {len(df_train)} rows")

    df_raw, df_dedup = None, None
    if os.path.exists(RAW_PATH):
        df_raw = pd.read_json(RAW_PATH, lines=True)
        print(f"[OK] integrated_raw.jsonl: {len(df_raw)} rows")
    if os.path.exists(DEDUP_PATH):
        df_dedup = pd.read_json(DEDUP_PATH, lines=True)
        print(f"[OK] integrated_dedup_v1.jsonl: {len(df_dedup)} rows")

    return df_train, df_raw, df_dedup


# ============================================================
# Section 1: Dedup comparison
# ============================================================

def report_dedup(df_train, df_raw, df_dedup):
    print("\n" + "=" * 60)
    print("Section 1: Deduplication Comparison")
    print("=" * 60)

    if df_raw is None or df_dedup is None:
        print("[SKIP] raw / dedup files not found.")
        return

    n_raw   = len(df_raw)
    n_dedup = len(df_dedup)
    n_train = len(df_train)

    print(f"  Raw (post legality filter):   {n_raw}")
    print(f"  After dedup:                  {n_dedup}  (removed {n_raw - n_dedup})")
    print(f"  Final train (>=10 char text): {n_train} (removed {n_dedup - n_train})")
    print(f"  Total removed from raw:       {n_raw - n_train} "
          f"({(n_raw - n_train) / n_raw * 100:.1f}%)")

    print("\n  Dedup by task type:")
    for task in df_raw["task_type"].unique():
        n_r = (df_raw["task_type"] == task).sum()
        n_d = (df_dedup["task_type"] == task).sum()
        print(f"    {task:<22} {n_r} -> {n_d}  (removed {n_r - n_d})")


# ============================================================
# Section 2: Dataset and task type distribution
# ============================================================

def plot_dataset_task_dist(df):
    print("\n" + "=" * 60)
    print("Section 2: Dataset and Task Type Distribution")
    print("=" * 60)

    print(df.groupby(["dataset", "task_type"]).size()
          .rename("count").to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: count by dataset
    dataset_counts = df["dataset"].value_counts()
    sns.barplot(x=dataset_counts.values, y=dataset_counts.index,
                ax=axes[0], orient="h")
    axes[0].set_title("Sample Count by Dataset")
    axes[0].set_xlabel("Count")
    for bar, val in zip(axes[0].patches, dataset_counts.values):
        axes[0].text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                     f"{val:,}", va="center", fontsize=8)

    # Right: stacked by task type within dataset
    cross = (df.groupby(["dataset", "task_type"])
             .size().unstack(fill_value=0))
    cross.plot(kind="bar", ax=axes[1], rot=0, colormap="Set2")
    axes[1].set_title("Task Type Distribution across Datasets")
    axes[1].set_xlabel("Dataset")
    axes[1].set_ylabel("Count")
    axes[1].legend(title="task_type", fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "01_dataset_task_dist.png")
    plt.savefig(out)
    plt.close()
    print(f"[OK] Saved {out}")


# ============================================================
# Section 3: Label distributions (semantic tasks)
# ============================================================

def plot_label_dist(df):
    print("\n" + "=" * 60)
    print("Section 3: Label Distribution (Semantic Tasks)")
    print("=" * 60)

    sem = df[df["task_type"].isin(["sentence_semantic", "target_semantic"])]
    label_counts = sem["label"].value_counts(dropna=False)
    print(label_counts.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall label distribution
    sns.barplot(x=label_counts.index.astype(str),
                y=label_counts.values, ax=axes[0],
                palette={"positive": "#4CAF50",
                         "neutral":  "#2196F3",
                         "negative": "#F44336"})
    axes[0].set_title("Overall Label Distribution (Semantic Tasks)")
    axes[0].set_xlabel("Label")
    axes[0].set_ylabel("Count")
    for bar in axes[0].patches:
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 30,
                     f"{int(bar.get_height()):,}",
                     ha="center", fontsize=9)

    # Label by dataset (semantic only)
    cross = (sem.groupby(["dataset", "label"])
             .size().unstack(fill_value=0))
    cross.plot(kind="bar", ax=axes[1], rot=0,
               color=["#F44336", "#2196F3", "#4CAF50"])
    axes[1].set_title("Label Distribution by Dataset (Semantic Tasks)")
    axes[1].set_xlabel("Dataset")
    axes[1].set_ylabel("Count")
    axes[1].legend(title="label", fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "02_label_dist.png")
    plt.savefig(out)
    plt.close()
    print(f"[OK] Saved {out}")


# ============================================================
# Section 4: Market label distribution (market_aux)
# ============================================================

def plot_market_label_dist(df):
    print("\n" + "=" * 60)
    print("Section 4: Market Label Distribution (market_aux)")
    print("=" * 60)

    mkt = df[df["task_type"] == "market_aux"]
    if mkt.empty:
        print("[SKIP] No market_aux rows found.")
        return

    ml_counts = mkt["market_label"].value_counts(dropna=False)
    print(ml_counts.to_string())

    # Check for problematic dict-like values
    dict_like = mkt["market_label"].dropna().astype(str).str.startswith("{").sum()
    if dict_like > 0:
        print(f"\n[WARN] {dict_like} market_label values look like dict strings.")
        print("       This indicates load_finmarba() read the wrong column.")
        print("       Sample values:")
        print(mkt["market_label"].dropna().astype(str).head(10).to_string())

    # Check for null market_label fraction
    null_frac = mkt["market_label"].isna().mean()
    print(f"\n  market_label null fraction: {null_frac:.1%}")
    if null_frac > 0.5:
        print("  [WARN] Over 50% of market_aux rows have no market_label. "
              "Check FinMarBa column mapping.")

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {
        "positive": "#4CAF50",
        "neutral":  "#2196F3",
        "negative": "#F44336",
    }
    bar_colors = [colors.get(str(k), "#9E9E9E")
                  for k in ml_counts.index]
    sns.barplot(x=ml_counts.index.astype(str),
                y=ml_counts.values, palette=bar_colors, ax=ax)
    ax.set_title("Market Label Distribution (FinMarBa)")
    ax.set_xlabel("market_label")
    ax.set_ylabel("Count")
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 10,
                f"{int(bar.get_height()):,}",
                ha="center", fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "03_market_label_dist.png")
    plt.savefig(out)
    plt.close()
    print(f"[OK] Saved {out}")


# ============================================================
# Section 5: Text length analysis
# ============================================================

def plot_text_length(df):
    print("\n" + "=" * 60)
    print("Section 5: Text Length Analysis")
    print("=" * 60)

    df = df.copy()
    df["text_len"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()

    print("\n--- Text length (chars) by task type ---")
    print(df.groupby("task_type")["text_len"]
          .describe().round(1).to_string())

    print("\n--- Word count by task type ---")
    print(df.groupby("task_type")["word_count"]
          .describe().round(1).to_string())

    # Very short texts
    short = df[df["text_len"] < 20]
    if len(short) > 0:
        print(f"\n[WARN] {len(short)} texts shorter than 20 chars:")
        print(short[["uid", "dataset", "text_len", "text"]].to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall text length histogram
    sns.histplot(data=df, x="text_len", bins=60, kde=True, ax=axes[0])
    axes[0].set_title("Text Length Distribution (chars)")
    axes[0].set_xlabel("Character count")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(df["text_len"].median(), color="red",
                    linestyle="--", label=f"Median={df['text_len'].median():.0f}")
    axes[0].legend(fontsize=8)

    # Box plot by task type
    sns.boxplot(data=df, x="task_type", y="text_len",
                order=["sentence_semantic", "target_semantic", "market_aux"],
                ax=axes[1])
    axes[1].set_title("Text Length by Task Type")
    axes[1].set_xlabel("Task Type")
    axes[1].set_ylabel("Character count")
    axes[1].tick_params(axis="x", rotation=10)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "04_text_length_dist.png")
    plt.savefig(out)
    plt.close()
    print(f"[OK] Saved {out}")

    # Per-dataset length distribution
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x="dataset", y="text_len", ax=ax)
    ax.set_title("Text Length by Dataset")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Character count")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "05_text_length_by_dataset.png")
    plt.savefig(out)
    plt.close()
    print(f"[OK] Saved {out}")


# ============================================================
# Section 6: Target / entity analysis
# ============================================================

def plot_target_analysis(df):
    print("\n" + "=" * 60)
    print("Section 6: Target / Entity Analysis")
    print("=" * 60)

    target_df = df[df["task_type"] == "target_semantic"]
    if target_df.empty:
        print("[SKIP] No target_semantic rows.")
        return

    print(f"  Total target_semantic rows: {len(target_df)}")
    print(f"  Unique targets: {target_df['target'].nunique()}")

    top20 = target_df["target"].value_counts().head(20)
    print("\n--- Top 20 most frequent targets ---")
    print(top20.to_string())

    # Multi-entity headlines: same text appears with >1 target
    multi = (target_df.groupby("text")["target"].nunique()
             .rename("n_targets"))
    multi_headline_count = (multi > 1).sum()
    print(f"\n  Headlines with multiple targets: {multi_headline_count} "
          f"({multi_headline_count / len(multi) * 100:.1f}%)")

    # Label conflict: same (text, target) with different labels
    conflicts = (target_df.groupby(["text", "target"])["label"]
                 .nunique().rename("n_labels"))
    conflict_count = (conflicts > 1).sum()
    print(f"  (text, target) pairs with conflicting labels: {conflict_count}")

    # Per-dataset target stats
    print("\n--- Unique targets per dataset ---")
    print(target_df.groupby("dataset")["target"]
          .nunique().rename("unique_targets").to_string())

    # Plot top 20 targets
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=top20.values, y=top20.index, ax=ax, orient="h",
                palette="Blues_r")
    ax.set_title("Top 20 Most Frequent Targets")
    ax.set_xlabel("Count")
    ax.set_ylabel("Target Entity")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "06_target_top20.png")
    plt.savefig(out)
    plt.close()
    print(f"[OK] Saved {out}")


# ============================================================
# Section 7: Label distribution per dataset (heatmap)
# ============================================================

def plot_label_by_dataset(df):
    print("\n" + "=" * 60)
    print("Section 7: Label Distribution per Dataset (Heatmap)")
    print("=" * 60)

    sem = df[df["task_type"].isin(["sentence_semantic", "target_semantic"])]
    pivot = (sem.groupby(["dataset", "label"])
             .size().unstack(fill_value=0))
    # Normalize to percentage
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    print("\n--- Label distribution (%) per dataset ---")
    print(pivot_pct.round(1).to_string())

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot_pct, annot=True, fmt=".1f", cmap="RdYlGn",
                linewidths=0.5, ax=ax, cbar_kws={"label": "%"})
    ax.set_title("Label Distribution (%) per Dataset")
    ax.set_xlabel("Label")
    ax.set_ylabel("Dataset")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "07_label_by_dataset.png")
    plt.savefig(out)
    plt.close()
    print(f"[OK] Saved {out}")


# ============================================================
# Section 8: Null / field coverage audit
# ============================================================

def report_null_audit(df):
    print("\n" + "=" * 60)
    print("Section 8: Null / Field Coverage Audit")
    print("=" * 60)

    print("\n--- Null counts per field ---")
    nulls = df.isnull().sum()
    print(nulls.to_string())

    print("\n--- Expected nulls (by task type) ---")
    print("  sentence_semantic: target=null, market_label=null  [OK]")
    print("  target_semantic:   market_label=null               [OK]")
    print("  market_aux:        label=null, target=null         [OK]")

    # Check unexpected nulls
    sent = df[df["task_type"] == "sentence_semantic"]
    if sent["label"].isna().any():
        print(f"[WARN] sentence_semantic: {sent['label'].isna().sum()} rows missing label")

    tgt = df[df["task_type"] == "target_semantic"]
    if tgt["label"].isna().any():
        print(f"[WARN] target_semantic: {tgt['label'].isna().sum()} rows missing label")
    if tgt["target"].isna().any():
        print(f"[WARN] target_semantic: {tgt['target'].isna().sum()} rows missing target")

    mkt = df[df["task_type"] == "market_aux"]
    if mkt["market_label"].isna().any():
        n = mkt["market_label"].isna().sum()
        frac = n / len(mkt) if len(mkt) > 0 else 0
        print(f"[INFO] market_aux: {n} rows ({frac:.1%}) have market_label=None")


# ============================================================
# Section 9: Save EDA summary JSON
# ============================================================

def save_summary(df, df_raw, df_dedup):
    summary = {
        "total_rows": int(len(df)),
        "raw_rows": int(len(df_raw)) if df_raw is not None else None,
        "dedup_rows": int(len(df_dedup)) if df_dedup is not None else None,
        "rows_removed_by_dedup": (
            int(len(df_raw) - len(df_dedup))
            if df_raw is not None and df_dedup is not None else None
        ),
        "dataset_counts": df["dataset"].value_counts().to_dict(),
        "task_type_counts": df["task_type"].value_counts().to_dict(),
        "label_counts": df["label"].value_counts(dropna=False).to_dict(),
        "market_label_counts": (
            df["market_label"].value_counts(dropna=False).to_dict()
        ),
        "text_len_stats": {
            task: {
                k: round(float(v), 1)
                for k, v in (
                    df[df["task_type"] == task]["text"]
                    .str.len().describe().to_dict().items()
                )
            }
            for task in df["task_type"].unique()
        },
        "unique_targets": int(
            df[df["task_type"] == "target_semantic"]["target"].nunique()
        ),
        "null_counts": df.isnull().sum().to_dict(),
    }

    out = os.path.join(OUT_DIR, "eda_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[OK] EDA summary saved to {out}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Stage 2: Exploratory Data Analysis")
    print("=" * 60)

    df_train, df_raw, df_dedup = load_data()

    report_dedup(df_train, df_raw, df_dedup)
    plot_dataset_task_dist(df_train)
    plot_label_dist(df_train)
    plot_market_label_dist(df_train)
    plot_text_length(df_train)
    plot_target_analysis(df_train)
    plot_label_by_dataset(df_train)
    report_null_audit(df_train)
    save_summary(df_train, df_raw, df_dedup)

    print("\n" + "=" * 60)
    print(f"[DONE] All EDA outputs saved to ./{OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
