# Minimal Dataset Inference Summary

## Run
- Model: `EleutherAI/pythia-410m`
- Dataset subset: `wikipedia`
- Member split: `train`
- Non-member split: `val`
- Samples per side: `8` / `8`
- Seeds: `2`
- Device: `cuda`
- Aggregation method: `linear`
- Reference features enabled: `True`
- Reference models: `tinystories-1M, tinystories-33M, phi-1_5`
- Total runtime: `116.79` seconds
- Peak GPU memory observed: `2.99` GiB

## What Was Reproduced
- Inference-only feature extraction from a pretrained causal LM.
- IID member vs non-member comparison using the paper dataset's train/val splits for one subset.
- Dataset-level aggregation on held-out examples with one-sided t-tests across multiple seeds.
- Combined p-value across dependent tests using the Sidak-style formula from the paper.

## Key Result
- Best single feature on held-out samples: `k_min_probs_0.05` with median AUC `0.7500`.
- Mean dataset-level p-value across seeds: `0.595532`.
- Median dataset-level p-value across seeds: `0.595532`.
- Combined dataset-level p-value: `0.949865`.
- Mean held-out score gap (non-member minus member): `-3.087459`.

## Top Held-Out Features
- `k_min_probs_0.05`: median held-out AUC `0.7500`
- `k_min_probs_0.1`: median held-out AUC `0.5625`
- `k_min_probs_0.2`: median held-out AUC `0.5312`
- `zlib_ratio`: median held-out AUC `0.5000`
- `k_max_probs_0.1`: median held-out AUC `0.4062`
- `ppl_ratio_random_deletion`: median held-out AUC `0.4062`
- `k_max_probs_0.2`: median held-out AUC `0.4062`
- `k_max_probs_0.05`: median held-out AUC `0.3750`

## Seed-Level P-Values
- Seed `11`: p-value `0.932370`, member mean `1.2410`, non-member mean `-7.6682`
- Seed `12`: p-value `0.258695`, member mean `-3.9141`, non-member mean `-1.1798`
