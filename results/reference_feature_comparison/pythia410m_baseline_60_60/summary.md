# Minimal Dataset Inference Summary

## Run
- Model: `EleutherAI/pythia-410m`
- Dataset subset: `wikipedia`
- Member split: `train`
- Non-member split: `val`
- Samples per side: `60` / `60`
- Seeds: `10`
- Device: `cuda`
- Aggregation method: `linear`
- Reference features enabled: `False`
- Total runtime: `16.57` seconds
- Peak GPU memory observed: `2.09` GiB

## What Was Reproduced
- Inference-only feature extraction from a pretrained causal LM.
- IID member vs non-member comparison using the paper dataset's train/val splits for one subset.
- Dataset-level aggregation on held-out examples with one-sided t-tests across multiple seeds.
- Combined p-value across dependent tests using the Sidak-style formula from the paper.

## Key Result
- Best single feature on held-out samples: `k_min_probs_0.05` with median AUC `0.6850`.
- Mean dataset-level p-value across seeds: `0.065157`.
- Median dataset-level p-value across seeds: `0.019851`.
- Combined dataset-level p-value: `0.542065`.
- Mean held-out score gap (non-member minus member): `1.086411`.

## Top Held-Out Features
- `k_min_probs_0.05`: median held-out AUC `0.6850`
- `k_min_probs_0.1`: median held-out AUC `0.6706`
- `k_min_probs_0.2`: median held-out AUC `0.6061`
- `ppl_ratio_butter_fingers`: median held-out AUC `0.5911`
- `ppl_ratio_change_char_case`: median held-out AUC `0.5906`
- `zlib_ratio`: median held-out AUC `0.5856`
- `k_max_probs_0.05`: median held-out AUC `0.5778`
- `ppl_ratio_whitespace_perturbation`: median held-out AUC `0.5650`

## Seed-Level P-Values
- Seed `11`: p-value `0.003898`, member mean `-1.5482`, non-member mean `0.0345`
- Seed `12`: p-value `0.001841`, member mean `-0.3425`, non-member mean `0.8927`
- Seed `13`: p-value `0.417310`, member mean `0.2439`, non-member mean `0.3468`
- Seed `14`: p-value `0.007254`, member mean `-0.7928`, non-member mean `0.4831`
- Seed `15`: p-value `0.026634`, member mean `-0.7717`, non-member mean `0.1633`
- Seed `16`: p-value `0.013068`, member mean `-0.7176`, non-member mean `0.3361`
- Seed `17`: p-value `0.000126`, member mean `-1.6130`, non-member mean `0.6062`
- Seed `18`: p-value `0.037636`, member mean `-0.4743`, non-member mean `0.4695`
- Seed `19`: p-value `0.076710`, member mean `-1.4965`, non-member mean `-0.6690`
- Seed `20`: p-value `0.067094`, member mean `-0.6956`, non-member mean `-0.0074`
