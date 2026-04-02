# Minimal Dataset Inference Summary

## Run
- Model: `EleutherAI/pythia-1.3b`
- Dataset subset: `wikipedia`
- Member split: `train`
- Non-member split: `val`
- Samples per side: `100` / `100`
- Seeds: `10`
- Device: `cuda`
- Aggregation method: `linear`
- Reference features enabled: `True`
- Reference models: `tinystories-1M, tinystories-33M, phi-1_5`
- Total runtime: `98.67` seconds
- Peak GPU memory observed: `3.68` GiB

## What Was Reproduced
- Inference-only feature extraction from a pretrained causal LM.
- IID member vs non-member comparison using the paper dataset's train/val splits for one subset.
- Dataset-level aggregation on held-out examples with one-sided t-tests across multiple seeds.
- Combined p-value across dependent tests using the Sidak-style formula from the paper.

## Key Result
- Best single feature on held-out samples: `k_min_probs_0.05` with median AUC `0.6584`.
- Mean dataset-level p-value across seeds: `0.084066`.
- Median dataset-level p-value across seeds: `0.041055`.
- Combined dataset-level p-value: `0.601279`.
- Mean held-out score gap (non-member minus member): `0.574313`.

## Top Held-Out Features
- `k_min_probs_0.05`: median held-out AUC `0.6584`
- `k_min_probs_0.1`: median held-out AUC `0.6346`
- `k_min_probs_0.2`: median held-out AUC `0.6144`
- `ppl_ratio_butter_fingers`: median held-out AUC `0.5938`
- `ref_ppl_ratio_tinystories-1M`: median held-out AUC `0.5914`
- `ppl_ratio_whitespace_perturbation`: median held-out AUC `0.5854`
- `ppl_ratio_change_char_case`: median held-out AUC `0.5734`
- `zlib_ratio`: median held-out AUC `0.5728`

## Seed-Level P-Values
- Seed `11`: p-value `0.014303`, member mean `-0.8054`, non-member mean `0.1320`
- Seed `12`: p-value `0.050803`, member mean `-0.5422`, non-member mean `0.1227`
- Seed `13`: p-value `0.031306`, member mean `-0.4476`, non-member mean `0.3559`
- Seed `14`: p-value `0.208897`, member mean `-0.1774`, non-member mean `0.2386`
- Seed `15`: p-value `0.118574`, member mean `-0.4422`, non-member mean `-0.1502`
- Seed `16`: p-value `0.012481`, member mean `-0.8561`, non-member mean `-0.1595`
- Seed `17`: p-value `0.003787`, member mean `-0.0059`, non-member mean `0.6754`
- Seed `18`: p-value `0.234481`, member mean `-0.0957`, non-member mean `0.1496`
- Seed `19`: p-value `0.027111`, member mean `-0.2714`, non-member mean `0.3435`
- Seed `20`: p-value `0.138913`, member mean `-0.3050`, non-member mean `0.0863`
