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
- Reference features enabled: `False`
- Total runtime: `5.34` seconds
- Peak GPU memory observed: `1.10` GiB

## What Was Reproduced
- Inference-only feature extraction from a pretrained causal LM.
- IID member vs non-member comparison using the paper dataset's train/val splits for one subset.
- Dataset-level aggregation on held-out examples with one-sided t-tests across multiple seeds.
- Combined p-value across dependent tests using the Sidak-style formula from the paper.

## Key Result
- Best single feature on held-out samples: `k_min_probs_0.05` with median AUC `0.7812`.
- Mean dataset-level p-value across seeds: `0.633609`.
- Median dataset-level p-value across seeds: `0.633609`.
- Combined dataset-level p-value: `0.971099`.
- Mean held-out score gap (non-member minus member): `-5.620100`.

## Top Held-Out Features
- `k_min_probs_0.05`: median held-out AUC `0.7812`
- `k_min_probs_0.1`: median held-out AUC `0.5312`
- `k_min_probs_0.2`: median held-out AUC `0.5312`
- `zlib_ratio`: median held-out AUC `0.5000`
- `k_max_probs_0.2`: median held-out AUC `0.4062`
- `k_max_probs_0.1`: median held-out AUC `0.4062`
- `ppl_ratio_random_deletion`: median held-out AUC `0.4062`
- `k_max_probs_0.05`: median held-out AUC `0.3750`

## Seed-Level P-Values
- Seed `11`: p-value `0.958172`, member mean `1.9127`, non-member mean `-11.2503`
- Seed `12`: p-value `0.309046`, member mean `-1.1378`, non-member mean `0.7850`
