# Treatment pilot report (primary-30)

## Frozen protocol

*Written before any paid B1 / T call. Do not edit this section after freeze.*

**Date frozen:** 2026-10-11  
**Git commit at freeze:** `5308403e16e759a3c833dbf8bf17e189dc0baea3`  
**Context:** fit-label control (`exp_20261003T163033Z`) returned a null result (gap matrix did not separate human fit labels). This pilot must be able to show, plainly, that treatment T is no better than the profile-aware baseline B1 if that is what the numbers say. Proxies are not quality.

### Conditions (primary-30, N = 10, seniority = mid, model = `gpt-5.6-terra`)

| ID | Name | Terra inputs | Selection of final 10 |
|----|------|--------------|------------------------|
| B0 | Flat prompt (stored) | raw CV + JD | none; reuse three stored replicates; do not regenerate |
| B1 | Profile-aware prompt | stored extraction JSON only | none (single call) |
| T | InterviewIQ treatment | stored extraction + stored gap matrix via pool prompt | pool 30 → competency-weighted MMR (λ=0.7) → 10 |
| B2 | Plain MMR | same stored pool as T (no LLM) | MMR, relevance = question priority, weights = 1 |
| S0 | Random | same pool | random 10, seed 42, 5 repeats → mean |
| S1 | Top priority | same pool | top-10 by priority |
| S2 | Query MMR | same pool | MMR, relevance = cosine(question, JD text), λ=0.7 |

### Fairness rules

1. Same model (`gpt-5.6-terra`), same generation parameters (`reasoning_effort=medium`, `max_output_tokens=12000`, temperature omitted for gpt-5*), same 30 pairs, same seniority (`mid`), same N=10, same final schema for comparison (list of question strings).
2. B1 does **not** receive gap matrix, scores, weights, embeddings, or pool.
3. B2 / S* reuse T's stored pool (zero additional LLM cost).
4. B0 is the three stored replicates (`exp_20260920T220958Z`, `exp_20260926T221259Z`, primary-30 subset of `exp_20260926T223339Z`); report each and the per-pair mean.
5. Cost/latency differences between conditions are part of the result.
6. Prompt freeze: new prompts written before looking at generated outputs; only mechanical schema/format fixes allowed, each logged with reason.

### Weight-mapping rule (frozen)

For each validated pool question:

1. If `target_gap == "profile_probe"`: weight = mean of `ProfileGap.weight` over the pair's gap matrix.
2. Else if exact string match of `target_gap` to a `job_competency`: weight = that gap's `weight`.
3. Else: weight = weight of the nearest job competency by embedding cosine similarity; log as fallback.

Never use the `/api/v1/questions/select` default of `q.priority or 1.0`. Call selection functions with explicit weights.

### Diagnostics (descriptive only; not official evaluation metrics)

1. Schema validity and question count (=10).
2. Redundancy: mean pairwise cosine within final 10; mean max-similarity; n pairs with sim ≥ 0.85.
3. Requirement-coverage proxy (embedding assignment at 0.35 / 0.45 / 0.55; exploratory): distinct requirements covered; share gap / partial / strength among covered. Label: "proxy: embedding assignment, not human judgement".
4. Metadata distributions for T/B2/S* (dimension; gap-category of `target_gap`). B0/B1: "not available".
5. Lexical-specificity proxy: share of questions containing ≥1 candidate/job term outside a stoplist.
6. Selection overlap on shared pool: Jaccard(T,B2), Jaccard(T,S1), Jaccard(T,S2).
7. Pool: fraction needing nearest-competency weight fallback; fraction `profile_probe`; validation-failure rate.

### Statistics

- n = 30 pairs; bootstrap 95% CIs over pairs (2000 resamples, seed 42).
- Wilcoxon signed-rank + matched-pairs rank-biserial for T vs B1, T vs B0 (mean of replicates), T vs B2.
- Holm correction across the three comparisons. p-values exploratory (low power).

### Pilot decision rule (advisory)

Recommend scaling to the 150-pair run **only if** T is at least as good as B1 on **both** (a) redundancy (lower mean pairwise similarity is better) and (b) requirement-coverage proxy (higher coverage at threshold 0.45 is better), with paired-difference CIs reported. Otherwise recommend diagnosing components first. The decision to scale remains the researcher's.

### Out of scope (not implemented)

Coverage-dependent λ, competency-budget enforcement, per-dimension redundancy thresholds, regeneration loops, human evaluation, final frozen evaluation metrics.

### Constants

- Extraction source: `exp_20260927T205741Z`
- Gap source: `exp_20260927T215232Z`
- `USD_CNY_RATE = 6.77` (approx. mid-market, July 2026)
- Default `--max-cost` = 4.00 USD
- Seeds: 42 (sampling / bootstrap / random selection)
- Embedding model for diagnostics: `all-MiniLM-L6-v2` (tests: `hashing-test`)

---

## Code inspection

### Pre-change pytest baseline

Command: `EMBEDDING_MODEL=hashing-test python3 -m pytest --tb=no -q`  
Result: **47 passed**, 241 warnings (pre-existing), ~79 s.  
Commit: `5308403` ("classifier reframing").

### Section 0 verification

| Claim | Status |
|-------|--------|
| `question_generation/` contains only `baseline.py` | **Confirmed** |
| No pool-generation / treatment chain stage | **Confirmed**; `run.py` says baseline mode only |
| No profile-aware / plain-MMR / selection ablations | **Confirmed** |
| No disk output cache for LLM generations | **Confirmed** (only OpenAI `prompt_cache_key`) |
| `generate_baseline` → `BaselineGeneratedSet` (`{"question"}` only); prompt 1.0.0; seniority in user template | **Confirmed** |
| Router: QUESTION_GENERATION → Terra; fails closed; `build_router` | **Confirmed** |
| gpt-5*: temperature omitted; reasoning effort default medium; max_output_tokens default 4096 | **Confirmed** |
| Settings defaults (pool 30, final 10, λ 0.7, threshold 0.85, weights 1.0/0.6, Terra pricing) | **Confirmed** |
| Gap matrix formula / severity / weight | **Confirmed** |
| `competency_weighted_mmr` / `redundant_pairs` | **Confirmed** |
| Domain schemas incl. `InterviewQuestion.priority` default 0.0 | **Confirmed** |
| `interviewiq.md` omits priority, controlled question_type, exact target_gap | **Confirmed** — new `interviewiq_pool.md` v1.1.0 required |
| Select API defaults weights to `q.priority or 1.0` | **Confirmed** — not used |
| Fit-label control null (AUC 0.62 / 0.50) | Accepted as given; not re-run |
| README status line stale | **Confirmed** |

### Discrepancies / notes

1. Primary-30 subset mean cost inside the 150-pair B0 run is ~$0.0055/pair (likely more cache hits); the 150-pair overall mean is ~$0.00765/pair. Dry-run estimates use the **150-pair overall** averages (1804 in / 383 out / $0.00765 / 5.46 s) as the B0 prior, matching the thesis table language.
2. Gap-analysis mean latency on stored primary-30 is ~62 ms/pair (spec band 55–62 ms).
3. Stored B0 records do not log `reasoning_effort` / `max_output_tokens` / temperature; equality of parameters with new runs cannot be proven from artifacts (limitation).
4. No OpenAI API key was present in the environment or `.secrets/openai_api_key` at inspection time; paid steps stop after dry-run until a key is provided.

### Pair-id match across artifacts

All 30 manifest request ids (`{profile_id}__{jd_id}__mid`) match exactly across:

- extraction `exp_20260927T205741Z` (30/30)
- gap `exp_20260927T215232Z` (30/30; 633 req: 51 strong / 175 partial / 245 weak / 162 missing)
- B0 `exp_20260920T220958Z` (30/30)
- B0 `exp_20260926T221259Z` (30/30)
- B0 primary-30 subset of `exp_20260926T223339Z` (30/30 of 150)

### B0 validity (regression check 2)

Every stored B0 record in the three runs (primary-30 subset for the 150-pair run) has exactly 10 non-empty question strings. Exceptions: **none**.

### Gap reproduction (regression check 1)

See section "Regression checks" below (filled after the local replay).

---

## Regression checks

- Gap reproduction passed: `True`
  - counts: `{'requirements': 633, 'strong_match': 51, 'partial_match': 175, 'weak_match': 245, 'missing': 162}` (expected `{'requirements': 633, 'strong_match': 51, 'partial_match': 175, 'weak_match': 245, 'missing': 162}`)
  - counts_match_expected: `True`
  - n_label_mismatch: 0, n_score_mismatch: 0, n_weight_mismatch: 0, max_abs_diff: 5.960464477539063e-08
- B0 validity passed: `True` (exceptions: 0)


## Dry-run and spend

- n_pairs: 30
- B1 est total: 0.289457 USD / 1.95962172 CNY
- T pool est total: 0.930489 USD / 6.2994085 CNY
- Combined paid est: 1.219945 USD / 8.25903022 CNY
- Assumptions: `{"b0_mean_input_tokens": 1804.27, "b0_mean_output_tokens": 383.05, "b0_mean_cost_usd": 0.00765, "b1_input_mult": 1.4, "b1_output_mult": 1.0, "pool_input_mult": 3.5, "pool_output_mult": 4.0, "terra_input_usd_per_mtok": 2.0, "terra_output_usd_per_mtok": 12.0}`
- API key present: `False`


## Actual cost and failures

- Actual B1 spend: 0.602350 USD / 4.0779 CNY
- Actual T pool spend: 1.845848 USD / 12.4964 CNY
- Actual paid total: 2.448198 USD / 16.5743 CNY
- USD_CNY_RATE: 6.77
- Git commit: `5308403e16e759a3c833dbf8bf17e189dc0baea3`


## Results

### Stage cost / latency

| stage | model | cost/pair USD | cost/pair CNY | cost 150 CNY | latency s | n |
|---|---|---|---|---|---|---|
| B0_generation | gpt-5.6-terra | 0.007650 | 0.0518 | 7.77 | 5.460 | 30 |
| B1_generation | gpt-5.6-terra | 0.020078 | 0.1359 | 20.39 | 6.797 | 30 |
| T_pool_generation | gpt-5.6-terra | 0.061528 | 0.4165 | 62.48 | 28.717 | 30 |
| selection_local | None | 0.000000 | 0.0000 | 0.00 | 0.002 | 30 |
| extraction_attributed | gpt-5.6-luna | 0.007114 | 0.0482 | 7.22 | 31.260 | 30 |
| gap_analysis_attributed | None | 0.000000 | 0.0000 | 0.00 | 0.062 | 30 |

### Full-pipeline cost per pair

- **B0**: 0.007650 USD / 0.0518 CNY; latency 5.46 s
- **B1**: 0.027192 USD / 0.1841 CNY; latency 38.06 s
- **T**: 0.068642 USD / 0.4647 CNY; latency 60.04 s
- **B2**: 0.068642 USD / 0.4647 CNY; latency 60.04 s

### B0 replicate variability

- Mean pairwise embedding similarity across replicates: 0.9338
- Mean question-text Jaccard across replicates: 0.0006

### Diagnostics (means with bootstrap 95% CIs)

| condition | redundancy mean [CI] | coverage@0.45 mean [CI] | lexical mean [CI] |
|---|---|---|---|
| B0 | 0.3410 [0.3245, 0.3559] | 0.2696 [0.2175, 0.3323] | 0.9744 [0.9589, 0.9867] |
| B1 | 0.3012 [0.2868, 0.3152] | 0.2725 [0.2112, 0.3427] | 0.9933 [0.9833, 1.0000] |
| T | 0.2858 [0.2639, 0.3104] | 0.2671 [0.2239, 0.3132] | 0.9167 [0.8867, 0.9433] |
| B2 | 0.2793 [0.2556, 0.3029] | 0.3164 [0.2632, 0.3789] | 0.9267 [0.9000, 0.9500] |
| S0 | 0.2930 [0.2694, 0.3160] | 0.3027 [0.2408, 0.3780] | 0.9467 [0.9167, 0.9733] |
| S1 | 0.3221 [0.2978, 0.3489] | 0.3041 [0.2553, 0.3596] | 0.9267 [0.9000, 0.9533] |
| S2 | 0.3317 [0.3077, 0.3560] | 0.3214 [0.2716, 0.3811] | 0.9733 [0.9533, 0.9900] |

### Paired comparisons (exploratory)

- **T_vs_B1** redundancy: rank-biserial=-0.2946, p=0.1642, Holm p=0.3284; diff CI=[-0.0388, 0.0077]
  coverage@0.45: rank-biserial=-0.0370, p=0.8688, Holm p=1.0000; diff CI=[-0.0589, 0.0426]
- **T_vs_B0** redundancy: rank-biserial=-0.7978, p=0.0000, Holm p=0.0001; diff CI=[-0.0777, -0.0310]
  coverage@0.45: rank-biserial=0.0741, p=0.7366, Holm p=1.0000; diff CI=[-0.0549, 0.0460]
- **T_vs_B2** redundancy: rank-biserial=0.2817, p=0.1840, Holm p=0.3284; diff CI=[-0.0037, 0.0164]
  coverage@0.45: rank-biserial=-0.8088, p=0.0044, Holm p=0.0133; diff CI=[-0.0872, -0.0193]

### Selection overlaps (Jaccard means)

- T_B2: 0.5809 [0.5145, 0.6463]
- T_S1: 0.6084 [0.5328, 0.6849]
- T_S2: 0.1965 [0.1611, 0.2340]

### Figures

- `docs/figures/fig40_pipeline_cost_latency.png`
- `docs/figures/fig41_redundancy.png`
- `docs/figures/fig42_coverage_proxy.png`
- `docs/figures/fig43_selection_variants.png`

## Prompt freeze record

| Prompt | File | Version | SHA-256 |
|--------|------|---------|---------|
| B1 profile-aware | `prompts/generation/profile_aware_baseline.md` | 1.0.0 | `1428b36886d5e8bea3c658c71e5fe87fd317ad01c4947bb3ee054d8b21c66779` |
| T pool | `prompts/generation/interviewiq_pool.md` | 1.1.0 | `26e027acc555bdd199a57093cb15b173f15de1b04c2a160718b22727f128eabc` |

No post-output prompt iterations. Existing `baseline.md` / `interviewiq.md` / `domain.py` untouched.

---

## Deviations from spec

1. **Dry-run pool cost estimate (~$0.031/pair)** was below the informal $0.05–0.10 expectation; measured T pool was **$0.0615/pair** (within the expected band). Dry-run multipliers remain logged for transparency.
2. Initial session stopped after dry-run for missing API key; the full pilot was completed later once the key was available (`exp_20261011T130310Z` / `exp_20261011T130619Z` / `exp_20261011T131945Z` + `treatment_pilot_combined/`).
3. No other intentional deviations from the frozen protocol.

---

## Limitations

- n = 30; proxy diagnostics; single generation per pair for B1/T unless `--replicates` > 1.
- Single dataset (resume-job-description-fit primary-30).
- B1 and pool prompts are single wording choices (frozen).
- Stored B0 generation parameters unverifiable from artifacts.
- Fit-label control already showed gap scores do not separate human fit labels; gap-targeting is not assumed to improve quality.

---

## Pilot decision-rule outcome

- **Case:** T not at least as good as B1 on both proxies — diagnose components first
- recommend_scale: `False`
- redundancy T≤B1: `True` (T=0.2858, B1=0.3012)
- coverage T≥B1: `False` (T=0.2671, B1=0.2725)
- Note: Advisory only; proxies are not quality; decision to scale remains the researcher's.


## What this means for the thesis

On the frozen proxy diagnostics, T was **not** at least as good as B1 on both redundancy and requirement-coverage. The advisory rule therefore recommends diagnosing components (pool prompt, weight mapping, weighted MMR) before scaling to 150 pairs. This is consistent with not assuming gap-targeting improves quality after the null fit-label control.
