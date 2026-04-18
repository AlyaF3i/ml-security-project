# Other Papers Reviewed

This directory contains four additional papers related to membership inference and dataset inference for language models. I reviewed them to identify inference-time metrics that fit the current `official_repo/metrics.py` interface, which is built around per-sample scalar features computed from a frozen model.

## Summary

### `2025.emnlp-main.370.pdf`
- Title: `Context-Aware Membership Inference Attacks against Pre-trained Large Language Models`
- Main idea:
  The paper argues that memorization in pre-trained LLMs is context-dependent. Instead of relying only on a global sequence score, it studies token-level uncertainty and subsequence behavior to build a stronger membership inference attack.
- Metric or attack style:
  The proposed method, CAMIA, is a learned context-aware attack rather than a single plug-in scalar metric.
- Implemented from this paper:
  None directly.
- Reason not directly implemented:
  CAMIA is not exposed as a simple standalone score in the paper. It is a higher-level attack pipeline built from token-level context signals, and it does not drop cleanly into the current `metrics.py` design without adding a larger training and calibration layer.

### `2025.findings-naacl.234.pdf`
- Title: `Scaling Up Membership Inference: When and How Attacks Succeed on Large Language Models`
- Main idea:
  The paper studies membership inference across larger units of text, from short sequences up to documents and collections of documents, and adapts dataset-inference style aggregation to show that MI becomes more effective at larger scales.
- Metrics discussed:
  `perplexity`, `lowercase perplexity`, `zlib`, and `Min-K`-style features, including stronger normalized variants.
- Implemented from this paper:
  - `lowercase_ppl`
  - `min_k_plus_plus_{0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6}`
- Notes:
  `lowercase_ppl` is a direct scalar metric.
  `min_k_plus_plus_*` is implemented using normalized token negative log-probabilities, following the Min-K++ description referenced by the paper.

### `2026.eacl-long.251.pdf`
- Title: `Detecting Non-Membership in LLM Training Data via Rank Correlations`
- Main idea:
  The paper proposes `PRISM`, a dataset-level non-membership test based on Spearman rank correlation between normalized token log-probability signals produced by a target model and a reference model.
- Metric or attack style:
  This is a dataset-level correlation test rather than a simple per-sample scalar feature.
- Implemented from this paper:
  None directly.
- Reason not directly implemented:
  PRISM requires dataset-level aggregation and token-rank correlations against a reference model. The current `metrics.py` file is designed for per-sample scalar feature extraction, so PRISM fits better as a separate dataset-level evaluation module than as one more feature in `aggregate_metrics`.

### `2601.02751v2.pdf`
- Title: `Window-based Membership Inference Attacks Against Fine-tuned Large Language Models`
- Main idea:
  The paper proposes `WBC`, a sliding-window attack that compares local target-vs-reference loss differences and aggregates binary window votes instead of relying on a single global average.
- Metric or attack style:
  Reference-based, token-level, and window-aggregated.
- Implemented from this paper:
  None directly.
- Reason not directly implemented:
  WBC needs token-level target/reference loss sequences and local window comparisons. The current reference-model path in `metrics.py` stores only scalar per-sample `ppl` values for reference models, so WBC would require a broader redesign of the reference-feature pipeline.

## Metrics Added To `official_repo/metrics.py`

The following new metrics were added:

- `lowercase_ppl`
  - Source: `2025.findings-naacl.234.pdf`
  - Description:
    Computes perplexity after lowercasing the input text before tokenization.

- `min_k_plus_plus_{k}`
  - Source: `2025.findings-naacl.234.pdf` (via the Min-K++ formulation it cites and uses)
  - Description:
    Computes the mean of the top `k%` normalized token losses, where each token loss is normalized by the mean and variance of the model's token-level negative log-probability distribution at that position.

## Practical Scope

Only metrics that satisfy all of the following were added directly to `metrics.py`:

- inference-time only
- scalar per-sample feature
- computable from the current target-model forward pass
- compatible with the current aggregate feature extraction interface

The remaining papers are still useful design references, but their methods are better treated as separate evaluation pipelines rather than simple additions to `aggregate_metrics`.
