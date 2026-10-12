# Frozen evaluation metrics (v1)

**Spec SHA-256:** `31aad6fa2aa3f225f120b075b1da5135fa495a9997931413a67cd9c3f795ed00`  
**Freeze file:** `experiments/metrics/metrics_freeze.json` (written `2026-10-12T16:37:59Z`)  
**Git commit at freeze:** `1c62b429108fa3e724660084ea1e2ca3887591af`

These definitions are frozen before pilot question files are opened. Metrics measure alignment with the gap matrix and surface diversity, **not** question quality. The fit-label control was null (gap labels are noisy).

## Primary metrics

| ID | Name | Direction |
|----|------|-----------|
| M1.P | Mean pairwise cosine similarity | lower better |
| M2.P | Gap-targeting rate | higher better |
| M3.P | Capped requirement coverage | higher better |
| M4.P | Vendi score (cosine kernel) | higher better |

## Assignment

- `r*(q) = argmax_r cos(q, r)` over requirement `job_competency` strings.
- Linked iff `cos(q, r*) >= tau_p`.
- `tau_p` = 95th percentile of `max_r cos(q', r)` over B0 questions from the other 149 pairs.
- Sensitivity (not primary): threshold-free; fixed 0.35 / 0.45 / 0.55.

## Decision rule (advisory)

- vs_B1: >=2 wins among {M1.P,M2.P,M3.P,M4.P} and 0 losses
- vs_B2: >=1 win among {M2.P,M3.P} and 0 losses among four primaries
- vs_B0: >=3 wins and 0 losses
- projected_150_cost_cny <= scale_budget_cny (else SCALE_BLOCKED_BY_BUDGET advisory)

Otherwise: **DIAGNOSE**.

## Limitations

- Metrics measure alignment with the gap matrix and surface diversity, not question quality.
- Fit-label control was null (AUC 0.62 primary / 0.50 holdout); gap labels are noisy.
- T is conditioned on the gap matrix while B1 is only instructed in words; M2/M3 advantage is partly by construction.
