# Gold-set round 3 — worked-rule re-labelling

**Date:** 2026-11-26  
**Status:** Complete (exploratory LLM dual-coding). Results: `docs/gold_set_round3_report.md`, `experiments/results/gold_set_analysis_round3/`.

## Historical rounds (unchanged)

| Round | Role |
| --- | --- |
| Round 1 | Original five-annotator labels (`annotator_key/`). Negative IAA. |
| Round 2 | Calibration adjudication + main `adjudicated.csv`. G1 **not interpreted**. Official human gold-set result remains `docs/gold_set_report.md` / `experiments/results/gold_set_analysis/`. |

Frozen hypotheses in `docs/gold_set_protocol.md` are **not** edited.

## Pack choice

**Same pack.** Round 3 re-labels the **same 50 dual-coded main items** and the same 10 calibration items from the frozen pack (`experiments/gold_set/items_blind.json`, sample seed `20261004`). No fresh stratified resample. Documented so IAA is comparable to round 1–2 on the same requirements.

## Design

- **Two independent dual-coders:** `annotator_P1`, `annotator_P2`.
- Both label all 50 main items (full dual coding of the assigned main set).
- **No peeking** at system labels, answer key, or each other’s labels on main items.
- Aim: agreement first (main QWK ≳ 0.3–0.4 with `ci_low > 0`), not agreement with the model.

## Retraining

Before main coding, both annotators study:

1. Worked rule (met / partial / not evidenced; vague → cannot_judge).
2. `experiments/gold_set/round2/calibration_discussion.md` + adjudicated calibration labels.
3. `experiments/gold_set/round3/training/worked_rule_brief.md`.

Calibration items may be labelled as a **training check** (scored against `adjudicated_calibration.csv`). Main-item IAA uses independent main labels only.

## Label source (integrity)

Round 3 labels are produced by **independent LLM annotators** under the worked-rule brief (`experiments/scripts/round3_annotate.py`), not new human raters. Results are written to `experiments/results/gold_set_analysis_round3/` and `docs/gold_set_round3_report.md`. They **do not replace** the human round 1–2 record for the thesis unless humans later replicate the round.

## Success criteria (same gates)

1. **Interpret G1** only if main-pair quadratic κ has bootstrap 95% CI entirely above 0.
2. **Support G1** only if system-vs-human consensus QWK CI > 0 **and** Spearman CI > 0 (and G1 is interpreted).
3. G2–G4 evaluated on round-3 consensus for sensitivity; production cutover still follows frozen rules + explicit decision.

## Round-3 outcome (stored)

| Gate | Result |
| --- | --- |
| Main IAA QWK | **0.842 [0.712, 0.935]** (n=48) → **G1 interpreted** |
| Usable consensus | 40/50 (`agree` 40, `disagree` 8, `cannot_judge` 2) |
| G1 system vs consensus | QWK **0.624 [0.444, 0.751]**; Spearman **0.752 [0.579, 0.849]** → **supported** |
| G2 hybrid AUC | **0.809 [0.640, 0.944]** → separates; does not beat every single signal |
| G4 | **no-adopt** (holdout ΔQWK −0.225 [−0.388, −0.022]) |

**Caveat:** Clearing the gate with LLM dual-coders shows the worked rule is operational and that production labels can align with rule-consistent annotations. It does **not** overturn the human round 1–2 finding that independent human annotators did not agree enough to interpret G1. Thesis claims about *human* validation must still cite round 1–2 unless humans replicate round 3.
