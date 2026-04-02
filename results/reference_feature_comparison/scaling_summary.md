# Scaling Summary

## pythia-1.3b with reference features

- `60/60`, `10` seeds, batch `2`
- Peak GPU memory: `3.68 GiB`
- Median dataset-level p-value: `0.1410`
- Mean dataset-level p-value: `0.1295`
- Mean score gap: `0.9559`
- Reference-model cache hits: yes

## pythia-1.3b with reference features at 100/100

- `100/100`, `10` seeds, batch `2`
- Peak GPU memory: `3.68 GiB`
- Median dataset-level p-value: `0.0411`
- Mean dataset-level p-value: `0.0841`
- Mean score gap: `0.5743`
- Reference-model cache hits: no for the new `100/100` reference perplexities

## Takeaway

- Moving from `60/60` to `100/100` improved the dataset-level p-values more than moving from `pythia-410m` to `pythia-1.3b` alone.
- On this GPU, `pythia-1.3b` is safe with batch `2`.
