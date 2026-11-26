# Adoption: selector T′ as forward treatment

**Date:** 2026-11-26  
**Status:** Adopted for all forward work (selection default, human-eval treatment arm, any future pool→select run).  
**Basis:** Exploratory primary-30 replay met the pre-registered success rule in `docs/selector_tprime_prereg.md`. Numbers: `experiments/results/selector_tprime_replay/summary.json`.

## Decision

| Role | Strategy | Artifact |
| --- | --- | --- |
| **Forward treatment** | `tprime` | `experiments/results/selector_tprime_replay/treatment/` |
| Historical pilot T | `competency_weighted_mmr` | `experiments/results/exp_20261011T130619Z/treatment/` (unchanged) |
| Profile-aware baseline | B1 | `experiments/results/exp_20261011T130310Z/baseline_profile_aware/` |

Do **not** describe current T (weighted MMR alone) as competency-balanced. Forward claims and human evaluation use **T′**.

## Success rule (met)

From the stored summary:

1. Dimension quota met on **26 / 30** pools (≥ 24).
2. Mean gap share **0.71** (strictly below 0.95).
3. Mean M3.P / M4.P within margin of historical T; B2−T′ M3 fallback **not** triggered.

Mean Cultural/Motivational selected count: T ≈ 0.23 → T′ ≈ 0.97.

## Code defaults

- `FORWARD_TREATMENT_SELECTION = "tprime"` in `backend/app/services/selection/strategies.py`
- Treatment pipeline / CLI `--selection` default: `tprime`
- Historical strategy `competency_weighted_mmr` remains available for ablations and replay of the pilot

## Human evaluation

Blinded study inputs for the exploratory primary-30 rubric pilot / study:

`experiments/human_eval/inputs/primary30_Tprime_vs_B1.json`

The builder still uses condition labels **T** and **B1**. Here **T means T′** (forward treatment). The frozen holdout main-study protocol in `docs/human_eval_protocol.md` still names T; when holdout pools exist, generate with `--selection tprime` so that T is T′.

## What this does not change

- No new Terra generation
- Frozen pilot DIAGNOSE and historical T metrics stay as recorded
- Gold-set G1–G4 outcomes unchanged
- Gap production weights/thresholds unchanged
