# Reference Feature Comparison

## Runs
- Baseline dir: `results\reference_feature_comparison\pythia410m_baseline_60_60`
- Enhanced dir: `results\reference_feature_comparison\pythia410m_reference_60_60`

## Summary
- Baseline best single feature: `k_min_probs_0.05` (0.6850)
- Enhanced best single feature: `k_min_probs_0.05` (0.6850)
- Baseline median dataset-level p-value: `0.019851`
- Enhanced median dataset-level p-value: `0.079916`
- Baseline mean score gap: `1.086411`
- Enhanced mean score gap: `1.080709`
- Baseline runtime: `16.57` seconds
- Enhanced runtime: `39.01` seconds
- Baseline peak GPU memory: `2.09` GiB
- Enhanced peak GPU memory: `3.99` GiB

## Top Enhanced Features
- `k_min_probs_0.05`: baseline `0.685`, enhanced `0.685`
- `k_min_probs_0.1`: baseline `0.6705555555555556`, enhanced `0.6705555555555556`
- `k_min_probs_0.2`: baseline `0.606111111111111`, enhanced `0.606111111111111`
- `ppl_ratio_butter_fingers`: baseline `0.5911111111111111`, enhanced `0.5911111111111111`
- `ppl_ratio_change_char_case`: baseline `0.5905555555555555`, enhanced `0.5905555555555555`
- `ref_ppl_ratio_tinystories-1M`: baseline ``, enhanced `0.5861111111111111`
- `zlib_ratio`: baseline `0.5855555555555556`, enhanced `0.5855555555555556`
- `k_max_probs_0.05`: baseline `0.5777777777777777`, enhanced `0.5777777777777777`
- `ppl_ratio_whitespace_perturbation`: baseline `0.565`, enhanced `0.565`
- `ppl_ratio_underscore_trick`: baseline `0.5644444444444444`, enhanced `0.5644444444444444`

## Reference Feature AUCs
- `ref_ppl_ratio_tinystories-1M`: enhanced median AUC `0.5861111111111111`
