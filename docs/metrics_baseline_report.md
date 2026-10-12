# Metrics baseline report (Part A)

**Experiment:** `exp_20261012T163747Z`  
**Freeze file:** `/root/interviewIQ/experiments/metrics/metrics_freeze.json`  
**Spec SHA-256:** `31aad6fa2aa3f225f120b075b1da5135fa495a9997931413a67cd9c3f795ed00`  
**Git commit:** `1c62b429108fa3e724660084ea1e2ca3887591af`  

## Limitation

These metrics measure alignment with the gap matrix and surface diversity, **not** question quality. The fit-label control (`exp_20261003T163033Z`) was null (AUC 0.62 primary / 0.50 holdout), so gap labels are noisy.

## Counts

- M1/M4 computed on **150** of 150 B0 sets
- M2 computed on **60** pairs with gap matrices
- M3 computed on **60** pairs
- Gated extraction of missing 90: `{'requested': False, 'ran': False, 'cost_usd': 0.0, 'cost_cny': 0.0, 'n_pairs': 0, 'skipped_reason': None, 'dimension_labels': 'not_requested'}`

## Baseline distributions (mean, 95% CI)

| Metric | Overall | Good Fit | Potential Fit |
|--------|---------|----------|---------------|
| M1.P | 0.3372 [0.3286, 0.3456] (n=150) | 0.3347 [0.3221, 0.3469] (n=75) | 0.3398 [0.3279, 0.3511] (n=75) |
| M4.P | 6.8330 [6.7423, 6.9280] (n=150) | 6.8792 [6.7477, 7.0128] (n=75) | 6.7869 [6.6672, 6.9168] (n=75) |
| M2.P | 0.1850 [0.1467, 0.2233] (n=60) | 0.1967 [0.1433, 0.2533] (n=30) | 0.1733 [0.1233, 0.2233] (n=30) |
| M3.P | 0.3719 [0.3159, 0.4258] (n=60) | 0.3636 [0.2858, 0.4386] (n=30) | 0.3803 [0.3067, 0.4650] (n=30) |

## Validity

- **V1 known-answer passed:** True
- **V2 null control:** true M2.P=0.18499999999999997, null M2.P=0.03133333333333333, gap=0.15366666666666665; true M3.P=0.3719246031746032, null M3.P=0.05378968253968253, gap=0.31813492063492066
- **V3 duplicate injection:** fraction M1.P rises=0.98, fraction M4.P falls=1.0
- **V4 noise floor (primary-30 replicates):**
  - M1.P: mean |Δ_replicate|=0.028733060942354557, mean per-pair SD=0.02259752828534724
  - M2.P: mean |Δ_replicate|=0.08888888888888888, mean per-pair SD=0.0714312374436931
  - M3.P: mean |Δ_replicate|=0.1311111111111111, mean per-pair SD=0.10341032277002211
  - M4.P: mean |Δ_replicate|=0.318404649023268, mean per-pair SD=0.25279768328583796

## Freeze integrity

- Pilot question files were **not** opened before writing the freeze file.
- Pilot stage/status dirs present at freeze: `[{'experiment_id': 'exp_20261011T130141Z', 'stage': 'baseline_profile_aware', 'status': 'completed'}, {'experiment_id': 'exp_20261011T130156Z', 'stage': 'treatment_pipeline', 'status': 'completed'}, {'experiment_id': 'exp_20261011T130253Z', 'stage': 'selection_ablation', 'status': 'completed'}, {'experiment_id': 'exp_20261011T130310Z', 'stage': 'baseline_profile_aware', 'status': 'completed'}, {'experiment_id': 'exp_20261011T130619Z', 'stage': 'treatment_pipeline', 'status': 'completed'}, {'experiment_id': 'exp_20261011T131945Z', 'stage': 'selection_ablation', 'status': 'completed'}]`

If the researcher already inspected pilot outputs in a prior task, the subsequent scale decision is **advisory** (freeze not blind to that history).
