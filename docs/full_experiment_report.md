# 150-pair experiment: not run

No paid call was made. The spend gates in `docs/full_experiment_prereg.md` failed.

## Gates

Read from `experiments/results/exp_20261012T163853Z/summary.json` and
`verify_freeze()` in this session:

| Gate | Result |
| --- | --- |
| Scale verdict is SCALE | **Failed.** Verdict on record is **DIAGNOSE**. |
| Frozen-metrics hashes verify | **Failed.** Mismatch for `backend/app/services/eval_metrics/diagnostics.py` and `backend/app/services/eval_metrics/loaders.py`. |
| Step-5 pre-registration committed | **Passed.** `docs/adaptive_selection_prereg.md` is in git. |
| Projected cost within CNY 100 | **Passed** on the stored projection: **87.20587362 CNY** (12.881222 USD at 6.77 CNY per USD) versus budget 100 CNY. |

Because the first two gates failed, extraction for the 90 pairs, B1 generation,
and T pool generation were not started. Actual spend for this step: **0 USD
(0 CNY)**. There is no development analysis and no confirmatory analysis, so
there is no literal Wilcoxon outcome to report. Figures 70–79 were not drawn.

The stored projection, not a new dry-run, is the cost estimate above. It breaks
down as extraction for 90 pairs 0.6402321 USD, B1 × 150 3.01175 USD, and T pool
× 150 9.22924 USD.

## Existing gap records (the 60 pairs that already have them)

Computed from `exp_20260927T215232Z` and `exp_20260927T223303Z` gap files:

| | |
| --- | --- |
| Pairs | 60 |
| Requirements min / median / max | 2 / 17 / 43 |
| Pairs with fewer than 10 requirements | 8 |

The other 90 pairs were not extracted.

## Conflict between documents

The scale-decision rule needs wins against B1, B2, and B0 before SCALE. The
user fixed the confirmatory contrast for this step as **T versus B1**. Neither
rule was changed. The scale verdict remains the reason the paid run did not start.
