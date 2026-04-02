# Minimal Dataset Inference Summary

## Run
- Model: `EleutherAI/pythia-1.3b`
- Dataset subset: `wikipedia`
- Member split: `train`
- Non-member split: `val`
- Samples per side: `60` / `60`
- Seeds: `10`
- Device: `cuda`
- Aggregation method: `linear`
- Reference features enabled: `True`
- Reference models: `tinystories-1M, tinystories-33M, phi-1_5`
- Total runtime: `129.36` seconds
- Peak GPU memory observed: `3.68` GiB

## What Was Reproduced
- Inference-only feature extraction from a pretrained causal LM.
- IID member vs non-member comparison using the paper dataset's train/val splits for one subset.
- Dataset-level aggregation on held-out examples with one-sided t-tests across multiple seeds.
- Combined p-value across dependent tests using the Sidak-style formula from the paper.

## Key Result
- Best single feature on held-out samples: `k_min_probs_0.05` with median AUC `0.6872`.
- Mean dataset-level p-value across seeds: `0.129522`.
- Median dataset-level p-value across seeds: `0.141032`.
- Combined dataset-level p-value: `0.760798`.
- Mean held-out score gap (non-member minus member): `0.955904`.

## Top Held-Out Features
- `k_min_probs_0.05`: median held-out AUC `0.6872`
- `k_min_probs_0.1`: median held-out AUC `0.6522`
- `k_min_probs_0.2`: median held-out AUC `0.5983`
- `ref_ppl_ratio_tinystories-1M`: median held-out AUC `0.5839`
- `zlib_ratio`: median held-out AUC `0.5839`
- `ppl_ratio_butter_fingers`: median held-out AUC `0.5822`
- `ppl_ratio_change_char_case`: median held-out AUC `0.5617`
- `k_max_probs_0.05`: median held-out AUC `0.5583`

## Seed-Level P-Values
- Seed `11`: p-value `0.210538`, member mean `-1.4808`, non-member mean `-0.5716`
- Seed `12`: p-value `0.188309`, member mean `0.3503`, non-member mean `0.9806`
- Seed `13`: p-value `0.165073`, member mean `-0.3578`, non-member mean `0.2710`
- Seed `14`: p-value `0.002464`, member mean `-1.2029`, non-member mean `0.6046`
- Seed `15`: p-value `0.262537`, member mean `-0.3004`, non-member mean `-0.0227`
- Seed `16`: p-value `0.068634`, member mean `-0.9842`, non-member mean `0.2961`
- Seed `17`: p-value `0.000078`, member mean `-1.0808`, non-member mean `1.0077`
- Seed `18`: p-value `0.115527`, member mean `-0.3539`, non-member mean `0.3455`
- Seed `19`: p-value `0.140952`, member mean `-1.3360`, non-member mean `-0.6854`
- Seed `20`: p-value `0.141111`, member mean `-0.7380`, non-member mean `-0.1510`
