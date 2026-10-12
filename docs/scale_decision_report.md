# Scale decision report (Part B)

**Primary experiment:** `exp_20261012T163853Z` (freeze hashes matched; primary)  
**POST-HOC diagnostics experiment:** `exp_20261012T164049Z` (loader fix for pool `valid_questions`; tagged `POST-HOC, not primary`)  
**Verdict:** **DIAGNOSE** (advisory)  
**Spec SHA-256:** `31aad6fa2aa3f225f120b075b1da5135fa495a9997931413a67cd9c3f795ed00`  
**Freeze written:** `2026-10-12T16:37:59Z` at commit `1c62b429108fa3e724660084ea1e2ca3887591af`

Pilot outputs already existed (and were inspected in the earlier treatment-pilot task) before this freeze; the decision is **advisory** (freeze not blind to that history). Metric definitions were not tuned to pilot outcomes.

## Frozen decision rule — literal clause outcomes (primary)

- Clause 1: vs B1: ≥2 wins among four primaries and 0 losses → **FAIL** (wins=1, losses=0)
- Clause 2: vs B2: ≥1 win among {M2.P, M3.P} and 0 losses among four → **FAIL** (wins_M2_M3=1, losses=2)
- Clause 3: vs B0: ≥3 wins and 0 losses → **HOLD** (wins=3, losses=0)
- Clause 4: projected 150-pair cost within scale budget (CNY) → **HOLD** (87.21 ≤ 100)

**Rule outcome:** `DIAGNOSE`

## Validity caveat

T is conditioned on the gap matrix while B1 is only told in words to cover gaps; some advantage on M2/M3 is partly by construction. Wins on gap-targeting and coverage are necessary but not sufficient; parity or better on M1/M4 at equal or higher coverage is the more informative result.

These metrics measure alignment with the gap matrix and surface diversity, **not** question quality. Fit-label control was null (AUC 0.62 primary / 0.50 holdout).

## Twelve pre-registered comparisons (T vs baselines)

V4 noise floor (mean |replicate Δ| on primary-30 B0): M1.P≈0.029, M2.P≈0.089, M3.P≈0.131, M4.P≈0.318.

| Baseline | Metric | Favourable Δ | 95% CI | p | p_Holm | r_rb | outcome |
|----------|--------|--------------|--------|---|--------|------|---------|
| B1 | M1.P | 0.0155 | [-0.0085, 0.0388] | 0.164 | 0.821 | 0.295 | tie |
| B1 | M2.P | 0.187 | [0.113, 0.263] | 0.00015 | 0.0015 | 0.929 | win |
| B1 | M3.P | 0.0575 | [-0.010, 0.131] | 0.245 | 0.821 | 0.22 | tie |
| B1 | M4.P | 0.0484 | [-0.214, 0.313] | 0.715 | 0.821 | 0.080 | tie |
| B2 | M1.P | -0.0065 | [-0.0166, 0.0034] | 0.184 | 0.821 | -0.282 | tie |
| B2 | M2.P | 0.030 | [~0, 0.063] | 0.067 | 0.402 | 0.505 | win |
| B2 | M3.P | -0.060 | [-0.100, -0.020] | 0.0098 | 0.069 | -0.628 | lose |
| B2 | M4.P | -0.176 | [-0.302, -0.046] | 0.0081 | 0.065 | -0.544 | lose |
| B0 | M1.P | 0.0552 | [0.031, 0.078] | 3.9e-5 | 0.00043 | 0.798 | win |
| B0 | M2.P | 0.247 | [0.183, 0.313] | 5.8e-6 | 7.0e-5 | 0.980 | win |
| B0 | M3.P | 0.0569 | [-0.016, 0.133] | 0.168 | 0.821 | 0.296 | tie |
| B0 | M4.P | 0.497 | [0.226, 0.745] | 0.00028 | 0.0025 | 0.720 | win |

## Win / tie / loss matrix

| | M1.P | M2.P | M3.P | M4.P |
|--|------|------|------|------|
| T vs B0 | win | win | tie | win |
| T vs B1 | tie | win | tie | tie |
| T vs B2 | tie | win | lose | lose |

## Diagnostics (POST-HOC fill-in; not decision evidence)

Primary `scale_decision` left D1 null because pool records store `valid_questions` (not `questions`). Loader fix + `--allow-post-hoc` re-run (`exp_20261012T164049Z`) supplies:

- **D1:** pool M3.P≈0.52, T selected≈0.40, oracle-M3≈0.82 (oracle−T≈0.42) → **selection** likely bottleneck; oracle-M2≈0.61 vs T M2.P≈0.39.
- **D2:** pool fallback_rate≈0; profile_probe_rate≈0.10; selected T rich metadata often absent on final-10 schema (category counts under-specified for selected).
- **D3 λ sensitivity (exploratory):** T M3.P rises mildly with λ (0.38→0.43); B2 M3.P stays above T at default λ=0.7. Not evidence of improvement.
- **D4:** pool-size subsets computed on stored pools (see `diagnostics` JSON).
- **D5:** validation_failure_rate≈0.024; truncation=0; backfill=0.
- **D6:** two flips — T vs B0 on M2.P becomes tie under threshold-free and fixed-0.45; no flip changes the DIAGNOSE verdict.
- **D7:** Good vs Potential Fit and &lt;10-requirements strata in combined JSON (small n).
- **D8:** selection bottleneck + noisy gap matrix + by-construction M2/M3 vs B1.

## Scale-cost projection (linear extrapolation)

| Component | USD | CNY |
|-----------|-----|-----|
| Extraction for 90 missing pairs | 0.640 | 4.33 |
| B1 × 150 | 3.012 | 20.39 |
| T pool × 150 | 9.229 | 62.48 |
| **Total** | **12.881** | **87.21** |

Per-pair latency (from pilot per-pair `usage.latency_ms`; summary field was null): B1≈6.80 s, T pool≈28.72 s; extraction prior≈31.26 s. Budget default 100 CNY → clause 4 holds; scientific clauses fail → still DIAGNOSE.

## What this means for the thesis

Under the pre-registered rule the outcome is **DIAGNOSE**: do not spend on the 150-pair run until components are investigated. T beats B0 on redundancy/diversity/gap-targeting but only wins M2.P vs B1 (ties elsewhere), and loses M3.P and M4.P to plain-MMR B2. Oracle selection far above T points at the selector. A DIAGNOSE result is a legitimate finding. Metrics are alignment and diversity proxies, not question quality.
