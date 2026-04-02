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
- Reference features enabled: `True`
- Reference models: `tinystories-1M, tinystories-33M, phi-1_5`
- Total runtime: `39.01` seconds
- Peak GPU memory observed: `3.99` GiB

## What Was Reproduced
- Inference-only feature extraction from a pretrained causal LM.
- IID member vs non-member comparison using the paper dataset's train/val splits for one subset.
- Dataset-level aggregation on held-out examples with one-sided t-tests across multiple seeds.
- Combined p-value across dependent tests using the Sidak-style formula from the paper.

## Key Result
- Best single feature on held-out samples: `k_min_probs_0.05` with median AUC `0.6850`.
- Mean dataset-level p-value across seeds: `0.113904`.
- Median dataset-level p-value across seeds: `0.079916`.
- Combined dataset-level p-value: `0.739520`.
- Mean held-out score gap (non-member minus member): `1.080709`.

## Top Held-Out Features
- `k_min_probs_0.05`: median held-out AUC `0.6850`
- `k_min_probs_0.1`: median held-out AUC `0.6706`
- `k_min_probs_0.2`: median held-out AUC `0.6061`
- `ppl_ratio_butter_fingers`: median held-out AUC `0.5911`
- `ppl_ratio_change_char_case`: median held-out AUC `0.5906`
- `ref_ppl_ratio_tinystories-1M`: median held-out AUC `0.5861`
- `zlib_ratio`: median held-out AUC `0.5856`
- `k_max_probs_0.05`: median held-out AUC `0.5778`

## Seed-Level P-Values
- Seed `11`: p-value `0.089100`, member mean `-1.8296`, non-member mean `-0.2036`
- Seed `12`: p-value `0.011109`, member mean `0.0204`, non-member mean `1.6683`
- Seed `13`: p-value `0.467113`, member mean `0.1077`, non-member mean `0.1552`
- Seed `14`: p-value `0.002231`, member mean `-1.0103`, non-member mean `0.8610`
- Seed `15`: p-value `0.070731`, member mean `-0.4665`, non-member mean `0.3606`
- Seed `16`: p-value `0.150172`, member mean `-0.3234`, non-member mean `0.3314`
- Seed `17`: p-value `0.000090`, member mean `-1.2999`, non-member mean `0.7493`
- Seed `18`: p-value `0.153906`, member mean `-0.2524`, non-member mean `0.3149`
- Seed `19`: p-value `0.068144`, member mean `-1.3709`, non-member mean `-0.4117`
- Seed `20`: p-value `0.126445`, member mean `-0.5733`, non-member mean `-0.0166`
