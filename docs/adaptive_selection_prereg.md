# Pre-registration: adaptive lambda and competency budget

Written before any replay of stored question pools. Switches, defaults, the named
grid, and the comparison plan below are fixed here. `lambda_min`, `lambda_max`,
and the quota budgets are **not tuned on primary-30**.

A confirmatory test needs a holdout-30 pool generated under the treatment prompt.
That generation is paid and is out of scope for this pre-registration. Any replay
on the stored primary-30 pilot pools is **exploratory**. Replay outputs must record
the git hash of the commit that added this file.

## Selector

Module: `backend/app/services/mmr/adaptive.py`, entry point `select_adaptive`.
`backend/app/services/mmr/select.py` is not modified. With competency weights on
and budget, adaptive lambda, and relevance normalisation off, `weighted_mmr`
matches `competency_weighted_mmr` (indices and per-step scores), including the
strict `>` argmax and the `remaining`-set iteration order in that function.

No API route is added. Named configs are registered on the existing selection
registry under the stage name `adaptive_selection` (keys
`adaptive_selection/<config>`), not under `selection_ablation`.

## Switches and defaults

| Switch | Default |
| --- | --- |
| `use_competency_weights` | True. Relevance = priority × weight. False: relevance = priority only (plain MMR). |
| `use_budget` | False. Reserved-slot enforcement (below). |
| `budget_mode` | `min_quota`. Also `custom` (dimension → non-negative int) or `none`. |
| `use_adaptive_lambda` | False. |
| `lambda_fixed` | 0.7 |
| `lambda_min` | 0.5 |
| `lambda_max` | 0.9 |
| `lambda_direction` | `coverage_up` (primary). `coverage_down` is an ablation only. |
| `normalize_relevance` | False. |

`min_quota` assigns budget 1 to each canonical dimension: Technical, Behavioural,
Situational, Cultural/Motivational, Role-Specific. These five values are fixed;
they are not fit on primary-30.

Validation: lengths of questions, weights, dimensions, and the similarity matrix
agree; `k <=` pool size; budgets are non-negative ints; `sum(budgets) <= k`;
`lambda_min <= lambda_max`; every lambda is in `[0, 1]`. Adaptive lambda requires
`budget_mode != none`. It does not require `use_budget`.

## Adaptive lambda

At step t, before the pick:

`c_t = mean over budgeted dimensions of min(1, n_d / b_d)`

`n_d` is the number already selected in dimension d. Dimensions with budget `> 0`
and zero questions in the pool are excluded from that mean and recorded as
shortfalls (`zero_availability`), so `c_t` can still reach 1. If no feasible
budgeted dimension remains, `c_t` is defined as 1.

- `coverage_up`: `lambda_t = lambda_min + (lambda_max - lambda_min) * c_t`
- `coverage_down`: `lambda_t = lambda_max - (lambda_max - lambda_min) * c_t`
- switch off: `lambda_t = lambda_fixed` at every step

## Reserved slots

`unmet` is the sum over feasible dimensions (budget `> 0` and at least one pool
question) of `max(0, b_d - n_d)`. `remaining_slots = k - |selected|`. If
`unmet >= remaining_slots`, candidates are restricted to dimensions that still
owe quota. Otherwise every remaining question is eligible. The selector returns
exactly `k` items whenever `k <=` pool size. If a quota cannot be met, the
shortfall is recorded and the step fills by the normal MMR rule. Tie-breaking
stays strict `>` in ascending remaining-set iteration order in every branch.

Relevance normalisation, if turned on, min-max scales the pool relevance vector
to `[0, 1]` when the max exceeds the min; a non-zero constant vector becomes 1;
an all-zero vector stays 0. The default remains off so parity holds.

## Named grid

| Name | Weights | Reserved budget | Adaptive λ | Direction | Budget mode |
| --- | --- | --- | --- | --- | --- |
| `plain_mmr` | off | off | off | — | `none` |
| `weighted_mmr` | on | off | off | — | `none` |
| `weighted_budget` | on | on | off | — | `min_quota` |
| `weighted_adaptive` | on | off | on | `coverage_up` | `min_quota` |
| `full` | on | on | on | `coverage_up` | `min_quota` |
| `full_minus_weights` | off | on | on | `coverage_up` | `min_quota` |
| `full_minus_budget` | on | off | on | `coverage_up` | `min_quota` |
| `full_minus_adaptive` | on | on | off | — | `min_quota` |
| `full_coverage_down` | on | on | on | `coverage_down` | `min_quota` |

`factorial_grid()` is the 2³ over (weights, reserved budget, adaptive lambda).
Cells with adaptive lambda or reserved budget on use `min_quota` targets. The
cell with both off uses `budget_mode=none`.

## Comparisons

- **Primary (exploratory on primary-30):** `full` versus `weighted_mmr`.
- **Secondary (remove-one):** `full_minus_weights`, `full_minus_budget`,
  `full_minus_adaptive`, each versus `full`.
- **Direction ablation:** `full_coverage_down` versus `full`. Not a primary claim.

## Metrics

If `experiments/metrics/metrics_freeze.json` verifies against the frozen
`eval_metrics` module hashes, replay reports frozen metrics from that module
(M1 redundancy and M4 diversity on the selected texts; M2/M3 only when a stored
gap row and the frozen assignment threshold are available). Those numbers stay
exploratory on primary-30.

If the freeze does not verify, replay reports only **non-frozen diagnostics**:
budget compliance, per-dimension counts, lambda trace, shortfalls, and mean
pairwise similarity of the selected set. It does not label those as frozen metrics.

`lambda_min` (0.5), `lambda_max` (0.9), and the min-quota of 1 per canonical
dimension are not adjusted after looking at primary-30 replay output.

## Out of scope

Per-dimension redundancy thresholds, regeneration loops, and new pool generation
(including holdout-30) are not part of this pre-registration.
