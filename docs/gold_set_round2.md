# Gold-set round 2

This round is separate from round 1. Round 1 remains in `docs/gold_set_round1_report.md` and `experiments/results/gold_set_analysis/round1_summary.json`. The frozen hypotheses in `docs/gold_set_protocol.md` are unchanged.

## Worked rule for the calibration discussion

A requirement is met (2) only when the CV shows that requirement. Related experience counts as partial (1) at most. If the CV shows neither, the label is 0. The job description defines the requirement. It is not evidence inside the CV.

The discussion sheet for the 10 shared items is `experiments/gold_set/round2/calibration_discussion.md`. Nine of the ten items were split in round 1. One item, `c010`, was unanimous among the annotators who gave a 0/1/2 label. The exact requirement string is absent from the CV on all ten; that check is a prompt for the discussion, not a label.

The adjudicated calibration labels are in `experiments/gold_set/annotations/adjudicated_calibration.csv`:

| Item | Label |
| --- | ---: |
| c001 | 1 |
| c002 | 1 |
| c003 | 2 |
| c004 | 2 |
| c005 | 0 |
| c006 | 0 |
| c007 | 1 |
| c008 | 0 |
| c009 | 2 |
| c010 | 0 |

Independent round-1 labels in `annotator_key/` are unchanged.

## Adjudication of the main set

The 30 main items the two assigned annotators did not agree on are listed in `experiments/gold_set/round2/adjudication_worksheet.csv`. The `adjudicated_label` column is still empty. `experiments/gold_set/annotations/adjudicated.csv` was not created, so these calibration labels are not applied to the main-set consensus.

When the annotators finish the main set, copy a completed file to `experiments/gold_set/annotations/adjudicated.csv` with columns `item_id,label,cannot_judge,snippet,notes`, then re-run the analysis. Filling that file does not change inter-annotator kappa. Kappa is computed from the two original labels.

## Gate

Interpret G1 only when the main-pair quadratic-weighted kappa has a bootstrap 95% interval entirely above 0. Round 1 kappa is −0.033 [−0.317, 0.254], so the interval covers 0. Until a new labelling round moves that interval above 0, G1 stays uninterpreted and the hybrid gap matrix stays an implemented, unvalidated component. Production weights and thresholds stay at 0.25/0.25/0.50 and 0.65/0.40/0.20.
