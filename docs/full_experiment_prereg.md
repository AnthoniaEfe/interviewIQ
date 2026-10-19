# Pre-registration: 150-pair experiment

Written before any paid call for this step. No parameter below is tuned on
holdout-30 or on the 120 development pairs.

## Pair roles

All 150 pairs in `experiment_pairs_150` are in scope.

- **Development / exploratory:** the 120 pairs in primary-150 that are not in
  holdout-30. This includes primary-30 (seed 42), where the pilot already ran.
- **Confirmation:** holdout-30 only (seed 2026).

Development and confirmatory results are reported separately. They are not
pooled for a confirmatory claim.

## Conditions

Terra (`gpt-5.6-terra`) is the generator in every condition. B0 is the stored
150-pair flat baseline `exp_20260926T223339Z` and is not regenerated.

| Code | What | Generation |
| --- | --- | --- |
| B0 | Flat baseline, prompt `prompts/generation/baseline.md` | Reuse stored run |
| B1 | Profile-aware baseline, `prompts/generation/profile_aware_baseline.md` | One generation per pair |
| T | Treatment pool, `prompts/generation/interviewiq_pool.md` version 1.1.0, then selection | Pool once per pair |
| B2 | Plain MMR (`plain_mmr` / weights off) | Offline replay on the T pool |
| Ablations | Step-5 named grid | Offline replay on the T pool |

The step-5 grid is `plain_mmr`, `weighted_mmr`, `weighted_budget`,
`weighted_adaptive`, `full`, `full_minus_weights`, `full_minus_budget`,
`full_minus_adaptive`, `full_coverage_down`, with the parameters in
`docs/adaptive_selection_prereg.md` (lambda 0.5–0.9, fixed lambda 0.7,
min quota 1 per canonical dimension). Those values are not retuned here.

Pool checks stay: priority in (0, 1], dimension in the five-dimension
vocabulary, `target_gap` an exact competency or `profile_probe`. Pool size 30,
final N = 10, similarity threshold 0.85, embedding `all-MiniLM-L6-v2`,
required weight 1.0, preferred weight 0.6. Pool calls use `max_output_tokens`
12000.

## Prompt hashes (SHA-256 of the files on disk when this pre-registration was written)

| File | SHA-256 |
| --- | --- |
| `prompts/generation/interviewiq_pool.md` | `26e027acc555bdd199a57093cb15b173f15de1b04c2a160718b22727f128eabc` |
| `prompts/generation/profile_aware_baseline.md` | `1428b36886d5e8bea3c658c71e5fe87fd317ad01c4947bb3ee054d8b21c66779` |
| `prompts/generation/baseline.md` | `ce5a6819c8402c9c8099a9421d29ac2903a44c4fe88cc4f764842baeff41c4f8` |
| `prompts/extraction/candidate_profile.md` | `28b1fa2378061ddc353e174fae14056c4168209c3b641090f9b0b8bf1c125dc5` |
| `prompts/extraction/job_description.md` | `6235400538edc2dc00a03fabf0a31b9179e591405718d36caed0d395dcc3f7d7` |
| `prompts/analysis/profile_gap.md` | `1652ef38aa8e7d53643757a8ef6eaf74e3dc11c0f4891ef6130efd20c389ea9b` |

## Primary confirmatory contrast

**T versus B1**, on holdout-30 only, using the frozen metric family M1.P, M2.P,
M3.P, M4.P. The project's primary test is the paired Wilcoxon signed-rank test,
with Holm adjustment across those four metrics. Report rank-biserial effect
sizes and bootstrap confidence intervals.

The scale-decision document uses a different rule: SCALE only if T beats B1,
B2, and B0 under its clause counts. That rule is not edited. It is the spend
gate. The confirmatory scientific contrast for this step, fixed by the user, is
T versus B1. Secondary contrasts, reported separately, are the step-5
remove-one ablations, T versus B0, and T versus B2.

## Spend gate

A paid run is allowed only when the scale verdict is SCALE, the frozen-metrics
hashes verify, this file and the step-5 pre-registration are committed, and the
projected cost is within the CNY 100 budget. If any gate fails, no paid call is
made.

## Order

Development metrics are computed and committed before the confirmatory
holdout-30 analysis is run, and that confirmatory analysis is run once.
Every output records this file's commit and the code commit.
