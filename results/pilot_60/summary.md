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

## What Was Reproduced
- Inference-only feature extraction from a pretrained causal LM.
- IID member vs non-member comparison using the paper dataset's train/val splits for one subset.
- Dataset-level aggregation on held-out examples with one-sided t-tests across multiple seeds.
- Combined p-value across dependent tests using the Sidak-style formula from the paper.

## Key Result
- Best single feature on held-out samples: `k_min_probs_0.05` with median AUC `0.6811`.
- Mean dataset-level p-value across seeds: `0.062492`.
- Median dataset-level p-value across seeds: `0.021463`.
- Combined dataset-level p-value: `0.516234`.
- Mean held-out score gap (non-member minus member): `1.095011`.

## Differences From Full Paper
- This first milestone uses one model (`pythia-410m`) and one dataset subset (`wikipedia`) only.
- It omits reference-model features to keep the run cheap.
- The LM remains frozen; the only fitted component is a tiny linear probe over extracted features, which matches the paper repo more closely than LM fine-tuning would.
- It is meant to validate the core dataset-inference signal before scaling to the repo's broader sweep.

## Top Held-Out Features
- `k_min_probs_0.05`: median held-out AUC `0.6811`
- `k_min_probs_0.1`: median held-out AUC `0.6700`
- `k_min_probs_0.2`: median held-out AUC `0.6056`
- `ppl_ratio_butter_fingers`: median held-out AUC `0.5911`
- `ppl_ratio_change_char_case`: median held-out AUC `0.5911`

## Seed-Level P-Values
- Seed `11`: p-value `0.004678`, member mean `-1.4043`, non-member mean `0.0293`
- Seed `12`: p-value `0.003032`, member mean `-0.3594`, non-member mean `0.8776`
- Seed `13`: p-value `0.372979`, member mean `0.1862`, non-member mean `0.3405`
- Seed `14`: p-value `0.007013`, member mean `-0.8201`, non-member mean `0.4965`
- Seed `15`: p-value `0.027775`, member mean `-0.7313`, non-member mean `0.1741`
- Seed `16`: p-value `0.015152`, member mean `-0.6879`, non-member mean `0.3357`
- Seed `17`: p-value `0.000113`, member mean `-1.7839`, non-member mean `0.6741`
- Seed `18`: p-value `0.047444`, member mean `-0.5066`, non-member mean `0.4253`
- Seed `19`: p-value `0.081148`, member mean `-1.4593`, non-member mean `-0.6645`
- Seed `20`: p-value `0.065582`, member mean `-0.6923`, non-member mean `0.0025`
