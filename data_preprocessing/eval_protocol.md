# Evaluation Protocol: FinSentMulti-v1

**Version:** 1.0  
**Date Frozen:** 2026-04-02  
**Applies to:** Model evaluation on the `test` split of FinSentMulti-v1

---

## 1. Scope

This protocol governs **all evaluation** of models trained or fine-tuned on FinSentMulti-v1. Any result claiming to be comparable across systems must follow this protocol exactly.

**In scope:**
- Sentence-level sentiment (`sentence_semantic`)
- Target-entity-level sentiment (`target_semantic`)
- Combined multi-task evaluation

**Out of scope:**
- `market_aux` / FinMarBa rows — these are excluded from all supervised evaluation
- Generation tasks, QA, or any task not directly mapped to `{positive, neutral, negative}`

---

## 2. Primary Metric

> **Macro-F1 on the `test` split, semantic tasks only.**

Macro-F1 equally weights each of the three classes (`positive`, `neutral`, `negative`) regardless of class frequency. This is the standard for imbalanced financial sentiment datasets (cf. Malo et al., 2014; Sehgal et al., 2021) and directly addresses the positive-class dominance in FPB.

```
Macro-F1 = mean(F1_positive, F1_neutral, F1_negative)
```

---

## 3. Secondary Metrics

The following metrics must always be reported alongside Macro-F1:

| Metric | Scope | Rationale |
|---|---|---|
| Per-class F1 (pos/neu/neg) | All semantic | Reveals which classes drive or hurt performance |
| Accuracy | All semantic | For compatibility with prior work |
| Matthews Correlation Coefficient (MCC) | All semantic | More informative than accuracy for imbalanced sets |
| Macro-F1 per dataset | FPB / SEntFiN / FinEntity | Detects dataset-specific failure modes |
| Macro-F1 sentence_semantic only | FPB | Historical comparison baseline |
| Macro-F1 target_semantic only | SEntFiN + FinEntity | Entity-level difficulty |

---

## 4. Hard Subset Evaluation

Hard subsets must be separately reported. These are **not** separate test sets — they are subsets of the test rows tagged via `hard_subset` field.

| Subset | Tag in `hard_subset` | Definition |
|---|---|---|
| Multi-entity | `multi_entity` | Same text has ≥2 distinct target entities |
| Conflict | `conflict` | Same `(text, target)` has ≥2 label values in corpus |
| Ambiguous | `ambiguous` | Same text text has ≥2 labels (sentence-level) |

For each hard subset, report: **Macro-F1, per-class F1, N (count)**

**Why this matters:** A system achieving 80% macro-F1 overall but 55% on conflict cases has a fundamentally different risk profile than one achieving 75% across the board. Financial applications require robust handling of these edge cases.

---

## 5. Prediction Format

Prediction files must be JSONL with one record per test row:

```json
{"uid": "fpb_00001", "pred": "positive"}
{"uid": "sentfin_00023_0012", "pred": "neutral"}
```

Rules:
- `uid` must exactly match UID in `test.jsonl`
- `pred` must be one of: `positive`, `neutral`, `negative`
- Every test UID must have a prediction (no abstentions)
- Extra fields are ignored

---

## 6. Evaluation Procedure

```bash
python run_eval.py \
  --predictions predictions.jsonl \
  --data_file test.jsonl \
  --split_spec split_spec.json \
  --output_dir eval_results/
```

The script:
1. Loads test labels from `test.jsonl` (verified against `split_spec.json` MD5)
2. Loads predictions from `predictions.jsonl`
3. Verifies complete coverage (no missing UIDs)
4. Computes all metrics defined in Sections 2–4
5. Writes `eval_results/metrics.json`, `metrics_by_dataset.csv`, `confusion_matrix.csv`, `hard_subset_metrics.json`, `eval_report.md`

---

## 7. Baseline Reference Points

| System | Macro-F1 (test) | Notes |
|---|---|---|
| Majority class | ~0.18 | Always predicts `neutral` |
| Random | ~0.33 | Uniform random across 3 classes |
| FinBERT (FPB only) | ~0.86 | Single-dataset, sentence-level only |
| ProsusAI/finbert | ~0.88 | FPB sentence_semantic only |

> **Note:** No prior combined benchmark exists for this exact multi-source configuration. Baselines will be updated as results accumulate.

---

## 8. Reporting Requirements

A valid evaluation report **must** include:

- [ ] Test split MD5 hash (from `split_spec.json`) confirming exact data version
- [ ] Model name, architecture, and parameter count
- [ ] Training data used (which splits of which datasets)
- [ ] Overall Macro-F1 on `test` split
- [ ] Per-class F1 (positive / neutral / negative)
- [ ] Macro-F1 by dataset (FPB / SEntFiN / FinEntity)
- [ ] Hard subset Macro-F1 for each of the three subsets
- [ ] Confusion matrix
- [ ] MCC score

Reports missing any of the above are considered **partial** and cannot be directly compared.

---

## 9. Bias Checklist Audit Results

The following checks were performed during data freeze (see `split_spec.json → bias_checklist` for exact values):

| Check | Status | Details |
|---|---|---|
| **Exact cross-split text leakage** | See `split_spec.json` | Measured: val↔train and test↔train |
| **Near-duplicate leakage** | See `split_spec.json` | 8-gram Jaccard ≥ 0.85 check, 500-row sample |
| **UID uniqueness** | ✓ Verified | Each UID in exactly one split |
| **Future information leakage** | N/A | No publication timestamps in FPB/SEntFiN/FinEntity |
| **market_aux contamination** | ✓ Excluded | FinMarBa has `split=analysis`, zero in train/val/test |
| **Label stratification** | ✓ Verified | Stratified by `(dataset, label)` |

### Known Structural Validity Issues

1. **FPB annotator agreement boundary:** FPB uses 50%-agreement threshold. ~15–20% of FPB test rows may be intrinsically ambiguous. This is captured in the `ambiguous` hard subset tag.

2. **Cross-dataset entity inconsistency:** SEntFiN and FinEntity use different entity mention conventions. A model may exploit surface-level stylistic differences between datasets. Report per-dataset F1 to expose this.

3. **Positive class dominance in target_semantic:** Positive labels dominate SEntFiN (~35%). Macro-F1 explicitly controls for this, but analysts should review per-class F1 separately.

4. **FPB temporal/geographic domain:** FPB texts are predominantly Finnish/Nordic, 2004–2011. Results on this subset may not generalize to other markets.

---

## 10. Change Log

| Version | Date | Changes |
|---|---|---|
| 1.0 | 2026-04-02 | Initial freeze. All four sources integrated. Hard subsets defined. |
