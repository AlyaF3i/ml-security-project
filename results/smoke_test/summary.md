# Minimal Dataset Inference Summary

## Run
- Model: `EleutherAI/pythia-410m`
- Dataset subset: `wikipedia`
- Member split: `train`
- Non-member split: `val`
- Samples per side: `6` / `6`
- Seeds: `2`
- Device: `cuda`

## What Was Reproduced
- Inference-only feature extraction from a pretrained causal LM.
- IID member vs non-member comparison using the paper dataset's train/val splits for one subset.
- Dataset-level aggregation on held-out examples with one-sided t-tests across multiple seeds.
- Combined p-value across dependent tests using the Sidak-style formula from the paper.

## Key Result
- Best single feature on held-out samples: `k_min_probs_0.05` with median AUC `0.6111`.
- Mean dataset-level p-value across seeds: `0.860461`.
- Median dataset-level p-value across seeds: `0.860461`.
- Combined dataset-level p-value: `0.980535`.
- Mean held-out score gap (non-member minus member): `-0.928796`.

## Differences From Full Paper
- This first milestone uses one model (`pythia-410m`) and one dataset subset (`wikipedia`) only.
- It omits reference-model features to keep the run cheap.
- It uses a no-LM-training, no-fine-tuning feature-weighting scheme instead of the paper repo's linear regressor.
- It is meant to validate the core dataset-inference signal before scaling to the repo's broader sweep.

## Top Held-Out Features
- `k_min_probs_0.05`: median held-out AUC `0.6111`
- `k_min_probs_0.2`: median held-out AUC `0.5556`
- `k_min_probs_0.1`: median held-out AUC `0.3889`
- `ppl_ratio_synonym_substitution`: median held-out AUC `0.3889`
- `ppl_ratio_change_char_case`: median held-out AUC `0.3333`

## Seed-Level P-Values
- Seed `11`: p-value `0.858057`, member mean `0.8808`, non-member mean `-0.0123`
- Seed `12`: p-value `0.862864`, member mean `0.8262`, non-member mean `-0.1383`
