# Pre-registration: selector T′

Written before any T′ selection is run. Parameters below are fixed. They are not changed after primary-30 results are seen. This replay is **exploratory**. It is not a confirmatory test. Pools: the 30 stored primary-30 treatment pools in `experiments/results/exp_20261011T130619Z/pool`. No holdout pools. No new generation.

## Freeze decision

Choice **(b)**. Metric definitions are not changed, and `backend/app/services/eval_metrics/` is not edited. `experiments/metrics/metrics_freeze.json` stays at version v1. Every number this replay writes is tagged **post-hoc**.

v1 already fails its own hash check (`diagnostics.py` and `loaders.py` differ from the freeze file). This pre-registration does not re-freeze those files and does not recompute stored v1 tables.

## 1. Selector T′

New module only: `backend/app/services/selection/tprime.py`. `backend/app/services/mmr/select.py`, `backend/app/services/mmr/adaptive.py`, and `backend/app/services/selection/strategies.py` are not modified. T′ calls the same score as `competency_weighted_mmr`:

`score(i) = 0.7 * priority_i * weight_i - 0.3 * max_{j in selected} similarity(i, j)`

with `max` over the empty set defined as 0. The argmax uses strict `>` and walks the remaining set in insertion order of `set(range(n))`, which is `0, 1, ..., n-1`. That is the tie-break. Seed `42` is recorded and is not used: T′ draws no random numbers, matching `competency_weighted_mmr`, which accepts a seed and ignores it.

Fixed numbers:

| Parameter | Value |
| --- | --- |
| k | 10 |
| lambda | 0.7 |
| Dimensions, in this order | Technical, Behavioural, Situational, Cultural/Motivational, Role-Specific |
| Minimum quota per dimension | 1 |
| Reserved non-gap slots | 2 |
| Sum of quotas plus reserved slots | 7, which is ≤ k |

A question is **non-gap eligible** when its stored `target_gap` equals a `job_competency` in the stored gap matrix for that pair and that row's `match_label` is `partial_match` or `strong_match`. `profile_probe` is not eligible for a reserved slot.

Fill order, each step one question:

1. **Quotas.** For each dimension in the order above, if any remaining question has that dimension, take one MMR step restricted to those questions.
2. **Reserved slots.** Take up to 2 MMR steps restricted to remaining non-gap-eligible questions.
3. **Remainder.** Take MMR steps with every remaining question eligible until 10 questions are selected, or the pool is exhausted.

Relaxation: if the restricted candidate set for a quota step or a reserved step is empty, that step is skipped, the slot is released to phase 3, and the log records `{phase, dimension or reserved_index, reason}`. Reasons are `zero_availability` (no question of that dimension in the pool), `exhausted` (the dimension's questions were already taken), and `no_partial_or_strong_remaining`. A pool **meets the quota** only when all five dimensions have selected count ≥ 1. A logged quota relaxation means the pool does not meet the quota. The selector still returns `min(10, pool size)` questions.

## 2. Comparators

Read from stored outputs. Do not regenerate and do not re-select.

| Condition | Source |
| --- | --- |
| T, weighted MMR | `experiments/results/exp_20261011T130619Z/treatment` (`selection_strategy` `competency_weighted_mmr`, selected indices stored) |
| B2, plain MMR | `experiments/results/exp_20261011T131945Z/B2` |
| B1, profile-aware set | `experiments/results/exp_20261011T130310Z/baseline_profile_aware` |

B1 records are question text only. They have no `dimension` and no `target_gap`, so dimension-quota and gap-share are undefined for B1. Coverage and diversity still apply when embeddings exist.

## 3. Metrics

No new formula. Names below are the existing ones.

| Metric | Existing definition | Better |
| --- | --- | --- |
| Dimension quota met | A pool is yes when each of the five dimensions has selected count ≥ 1. This is the adaptive replay's `budget_met` for a min-quota of 1. Selectors without quotas are not scored as meeting a quota. | higher share of pools |
| Gap share | `metadata_distribution` in `backend/app/services/treatment_pilot/diagnostics.py`: fraction of selected questions whose `target_gap` maps to `match_label` `weak_match` or `missing`. Denominator is the number selected. | lower, for this test |
| Coverage | Frozen M3.P, capped requirement coverage, `m2_m3_from_assignment` in `backend/app/services/eval_metrics/metrics_core.py`. | higher |
| Diversity | Frozen M4.P, Vendi score on the cosine kernel, `m4_diversity` in the same module. | higher |

M3.P and M4.P need cached cosine embeddings of the selected questions and, for M3.P, of the requirement strings, plus the frozen `tau_p` for that pair. This replay does not encode text and does not call an embedding API. If those vectors are not already on disk, M3.P and M4.P are not computed.

Gap share of stored T, measured from the treatment files and the primary gap matrices before any T′ run: **0.95** (mean of the 30 per-pool shares; 285 of 300 selected questions).

## 4. Success rule

All three clauses are required.

1. T′ meets the dimension quota on at least 24 of the 30 pools (80%).
2. Mean gap share of T′ is strictly below 0.95.
3. Coverage and diversity are no worse than T by more than a fixed margin: mean M3.P(T′) ≥ mean M3.P(T) − 0.02, and mean M4.P(T′) ≥ mean M4.P(T) − 0.25.

## 5. Fallback

If mean M3.P(B2) − mean M3.P(T′) > 0.05, keep the dimension quotas and the 2 reserved slots, and run one ablation: **T′ with weights removed**. Relevance in that ablation is `priority` alone (every competency weight set to 1). Lambda, k, tie-break, quotas, reserved slots, and relaxation rule stay as above. The gap-weight term is then reported as an ablation, not as part of the successful selector.

## 6. Status

Exploratory. 30 primary-30 pools. Confirmatory use on holdout-30 is out of scope.
