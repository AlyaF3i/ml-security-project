# Minimal Dataset Inference Summary

## Run
- Model: `EleutherAI/pythia-410m`
- Dataset subset: `wikipedia`
- Member split: `train`
- Non-member split: `val`
- Samples per side: `6` / `6`
- Seeds: `2`
- Device: `cuda`
- Aggregation method: `linear`

## What Was Reproduced
- Inference-only feature extraction from a pretrained causal LM.
- IID member vs non-member comparison using the paper dataset's train/val splits for one subset.
- Dataset-level aggregation on held-out examples with one-sided t-tests across multiple seeds.
- Combined p-value across dependent tests using the Sidak-style formula from the paper.

## Key Result
- Best single feature on held-out samples: `k_min_probs_0.05` with median AUC `0.6111`.
- Mean dataset-level p-value across seeds: `0.710723`.
- Median dataset-level p-value across seeds: `0.710723`.
- Combined dataset-level p-value: `0.971662`.
- Mean held-out score gap (non-member minus member): `-5.838281`.

## Differences From Full Paper
- This first milestone uses one model (`pythia-410m`) and one dataset subset (`wikipedia`) only.
- It omits reference-model features to keep the run cheap.
- The LM remains frozen; the only fitted component is a tiny linear probe over extracted features, which matches the paper repo more closely than LM fine-tuning would.
- It is meant to validate the core dataset-inference signal before scaling to the repo's broader sweep.

## Top Held-Out Features
- `k_min_probs_0.05`: median held-out AUC `0.6111`
- `k_min_probs_0.2`: median held-out AUC `0.5556`
- `k_min_probs_0.1`: median held-out AUC `0.3889`
- `ppl_ratio_synonym_substitution`: median held-out AUC `0.3889`
- `ppl_ratio_change_char_case`: median held-out AUC `0.3333`

## Seed-Level P-Values
- Seed `11`: p-value `0.475473`, member mean `2.6095`, non-member mean `2.8517`
- Seed `12`: p-value `0.945974`, member mean `7.7201`, non-member mean `-4.1987`
