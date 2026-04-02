# Replication Milestone 1 Summary

- Canonical run folder: `results/minimal_pythia410m_wikipedia`
- Model: `EleutherAI/pythia-410m`
- Dataset subset: `wikipedia`
- Member split: `train`
- Non-member split: `val`
- Samples per side: `60`
- Seeds: `10`
- Device used: `cuda`

## What Was Reproduced

- Pretrained-model, inference-time feature extraction on the public dataset released with the paper.
- Sample-level scoring with perplexity, min-k, max-k, zlib ratio, and perturbation-based ratios.
- Dataset-level aggregation with a small linear probe over frozen features, followed by one-sided t-tests on held-out splits.

## Main Outcome

- Best single held-out feature: `k_min_probs_0.05`, median AUC `0.6811`
- Dataset-level median p-value across 10 random split seeds: `0.0215`
- Dataset-level mean p-value across 10 random split seeds: `0.0625`
- 9 of 10 seeds produced p-values below `0.1`
- Mean held-out score gap (`non-member - member`): `1.0950`

## Important Differences From The Full Paper

- Only one model and one dataset subset were run here.
- Reference-model features were omitted to keep the milestone cheap and Windows-friendly.
- The official repo has Linux and local-data assumptions, so this project uses a local wrapper instead of the repo entry points for the first run.
- No LM fine-tuning or training was done. The only fitted component is the small post-hoc linear probe on extracted features.

## Next Files To Read

- `results/minimal_pythia410m_wikipedia/summary.md`
- `results/minimal_pythia410m_wikipedia/dataset_level_results.csv`
- `results/minimal_pythia410m_wikipedia/sample_level_feature_metrics.csv`
