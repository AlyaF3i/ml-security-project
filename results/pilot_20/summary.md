# Minimal Dataset Inference Summary

## Run
- Model: `EleutherAI/pythia-410m`
- Dataset subset: `wikipedia`
- Member split: `train`
- Non-member split: `val`
- Samples per side: `20` / `20`
- Seeds: `5`
- Device: `cuda`
- Aggregation method: `linear`

## What Was Reproduced
- Inference-only feature extraction from a pretrained causal LM.
- IID member vs non-member comparison using the paper dataset's train/val splits for one subset.
- Dataset-level aggregation on held-out examples with one-sided t-tests across multiple seeds.
- Combined p-value across dependent tests using the Sidak-style formula from the paper.

## Key Result
- Best single feature on held-out samples: `k_min_probs_0.1` with median AUC `0.7700`.
- Mean dataset-level p-value across seeds: `0.351743`.
- Median dataset-level p-value across seeds: `0.460407`.
- Combined dataset-level p-value: `0.923507`.
- Mean held-out score gap (non-member minus member): `1.126218`.

## Differences From Full Paper
- This first milestone uses one model (`pythia-410m`) and one dataset subset (`wikipedia`) only.
- It omits reference-model features to keep the run cheap.
- The LM remains frozen; the only fitted component is a tiny linear probe over extracted features, which matches the paper repo more closely than LM fine-tuning would.
- It is meant to validate the core dataset-inference signal before scaling to the repo's broader sweep.

## Top Held-Out Features
- `k_min_probs_0.1`: median held-out AUC `0.7700`
- `k_min_probs_0.05`: median held-out AUC `0.7100`
- `k_max_probs_0.05`: median held-out AUC `0.6900`
- `ppl_ratio_change_char_case`: median held-out AUC `0.6700`
- `ppl_ratio_whitespace_perturbation`: median held-out AUC `0.6500`

## Seed-Level P-Values
- Seed `11`: p-value `0.086593`, member mean `0.2049`, non-member mean `2.9491`
- Seed `12`: p-value `0.003650`, member mean `-3.6823`, non-member mean `0.0271`
- Seed `13`: p-value `0.636001`, member mean `0.1158`, non-member mean `-0.4850`
- Seed `14`: p-value `0.460407`, member mean `-2.2018`, non-member mean `-2.0868`
- Seed `15`: p-value `0.572065`, member mean `0.6229`, non-member mean `0.2863`
