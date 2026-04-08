# DeepLearning Project

This repository collects four stages of a financial sentiment analysis project built around `ProsusAI/finbert`, routing-based experts, and retrieval-enhanced analysis.

## Repository layout

- `stage1/`: baseline sentence-level, target-level, and multitask experiments.
- `stage2/`: router-plus-experts training and ablation analysis.
- `stage3/`: training history and visualization assets for the expert-routing model.
- `stage4/`: retrieval-augmented experiments, alignment analysis, and routing diagnostics.

## Result snapshot

### Stage 1: Baseline models

The best Stage 1 model is the multitask marker setup:

- `multitask_ProsusAI_finbert_marker_1_1`: test macro-F1 `0.9030`
- `target_ProsusAI_finbert_concat`: test macro-F1 `0.8944`
- `sentence_ProsusAI_finbert`: test macro-F1 `0.8893`
- `target_ProsusAI_finbert_marker`: test macro-F1 `0.8871`

The detailed leaderboard is stored in `stage1/reports/baseline_leaderboard.{csv,json}`.

### Stage 2: Router + experts

The full router-expert model reaches test macro-F1 `0.8968`, while the uniform-routing ablation (`no_router`) reaches `0.8938`. This shows that the learned router contributes about `+0.0030` macro-F1 over uniform dispatch.

At the same time, the ablation summary shows that removing the conflict or context expert only changes overall macro-F1 slightly:

- `no_conflict_expert`: `0.8972`
- `no_context_expert`: `0.8971`
- `base_expert_only`: `0.8971`

This suggests the router is useful, but the incremental gain from extra experts is still limited in the current setup.

See `stage2/ablation_results/ablation_summary.json` and `stage2/ablation_results/ablation_report.md` for the full analysis.

### Stage 3: Training traces and visualizations

Stage 3 keeps the training history and visual assets for the expert-routing model.

- Best validation macro-F1 in `stage3/train_history.json`: `0.8936` at epoch `5`
- Mean routing weights at the best epoch are approximately `[0.784, 0.117, 0.099]`

The `stage3/expert_viz/` directory contains generated plots for training trends, ablation comparison, routing entropy, and routing radar views.

### Stage 4: Retrieval and alignment analysis

The retrieval-enhanced analysis focuses on disagreement and oracle-routing diagnostics.

- `alignment_summary.json` reports `10,587` total samples and a disagreement rate of `0.4864` between semantic and market signals on the always-on policy.
- `diag3_oracle_routing.json` concludes that oracle routing is **inconclusive**: overall F1 changes by only `-0.0001`, and ambiguous-sample F1 changes by `-0.0002`.

This indicates that the current bottleneck is probably not routing alone, but the difficulty of separating the ambiguous label distribution.

## Reproducing experiments

The repository keeps the training and analysis scripts, but excludes datasets, checkpoints, vector stores, and large retrieval caches from version control.

Useful entry points include:

- `stage1/train_baseline_sentence.py`
- `stage1/train_baseline_target.py`
- `stage1/train_multitask_baseline.py`
- `stage2/train_conflict_router.py`
- `stage2/ablation_router_experts.py`
- `stage4/train_with_retrieval.py`
- `stage4/analyze_alignment.py`
- `stage4/eval_four_policies.py`

Command examples used during the experiments are preserved in `stage2/commamd.txt` and `stage4/command.txt`.

## Notes

- Datasets are intentionally not tracked.
- Checkpoints and model weights are intentionally not tracked.
- Retrieval databases and cached corpora are intentionally not tracked.
