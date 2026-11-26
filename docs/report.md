# InterviewIQ — Full-Tree Analysis Report

**Date:** 2026-11-26  
**Repository:** `interviewIQ` (branch `main`, HEAD `416d257`)  
**Thesis goal:** Hybrid Profile-Gap Analysis Framework for Personalized and Competency-Balanced LLM Interview Question Generation  
**Primary dataset:** [`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit)

This report is an independent reading of the current tree, stored artifacts, and frozen protocols. It supersedes the snapshot in `docs/project_report.md` (2026-10-11) for planning purposes. Numbers below are taken from stored `summary.json` files and the frozen decision reports; none are invented.

---

## 0. Verdict in one page

The **framework is implemented**. Extraction, hybrid gap scoring, Terra generation (B0 / B1 / T), competency-weighted MMR, adaptive selection, automatic metrics, and a research console all exist and have been exercised.

The **thesis claim is not yet fully supported**. Independent checks:

1. The gap matrix does **not** track independent Hugging Face fit labels (null control).
2. InterviewIQ treatment **T does not beat** the profile-aware baseline **B1** on coverage, and **loses** to plain MMR **B2** on coverage and diversity.
3. **Human gold-set (rounds 1–2) stays historical and negative on the IAA gate.** Main human QWK −0.033 [−0.317, 0.254] → G1 **not interpreted** for human validation (`docs/gold_set_report.md`).
4. **Round 3 worked-rule re-labelling (same pack, LLM dual-coders P1/P2) cleared the IAA gate and supported G1 under the frozen rules** — IAA QWK 0.842 [0.712, 0.935]; system QWK 0.624 [0.444, 0.751]; Spearman 0.752 [0.579, 0.849]; G2 separates; G4 no-adopt. This is **exploratory** and does **not** replace human round 1–2 unless humans replicate. Details: `docs/gold_set_round3.md`, `docs/gold_set_round3_report.md`.
5. **Human-eval infrastructure for T′ vs B1 is launched; ratings are not yet collected.** Rubric pilot (5 primary-30 pairs) is built, API smoke-passed, rater UI ready (`docs/human_eval_launch.md`). The human DV is still empty until ≥3 raters finish.

The frozen scale rule therefore returned **DIAGNOSE**, not SCALE. The 150-pair confirmatory run was correctly **not started** ($0). Spending more Terra tokens on historical T will not salvage the claim. **T′ is adopted as the forward treatment** (`docs/selector_tprime_adoption.md`; default `--selection tprime`; materialized sets in `experiments/results/selector_tprime_replay/treatment/`; human-eval inputs `experiments/human_eval/inputs/primary30_Tprime_vs_B1.json`). The remaining path that can still advance the title is **question-level human evaluation of T′ vs B1**. Gap-matrix *human* validation remains negative (rounds 1–2); round 3 only shows rule-consistent LLM dual-coding can clear G1 gates on the same pack.

---

## 1. What the thesis title actually requires

The title has four load-bearing pieces. Mapping each piece to the code makes the remaining work obvious.

| Phrase | What it must mean in this repo | Current status |
| --- | --- | --- |
| **Hybrid profile-gap analysis** | Combine lexical (TF-IDF, BM25) and semantic (MiniLM) signals into a requirement-level match label and weight | **Implemented.** Human gold-set: G1 not interpreted. Round-3 LLM dual-code (same pack): G1 interpreted+supported, G2 separates, G4 no-adopt — exploratory only. HF fit labels null. |
| **Personalized** | Questions conditioned on *this* candidate vs *this* job, not a generic JD prompt | **Implemented** as Luna profiles + gap-conditioned Terra pool. Automatic metrics show T ≈ B1 except on gap-targeting, which is partly by construction. |
| **Competency-balanced** | Final N questions cover dimensions and mix gaps, partials, and strengths — not 95% gap probes | **Code exists; empirical behaviour fails the phrase.** Weighted MMR crowds out strengths. Cultural/Motivational is starved unless a dimension budget is on. |
| **LLM interview question generation** | Controlled Terra generation with matched baselines | **Implemented** for primary-30. Holdout-30 and 90/150 pairs have no T/B1 generations. |

The independent variable is **generation methodology**, holding candidate, job, model (`gpt-5.6-terra`), decoding, and N=10 fixed. Luna vs Terra must never be crossed as a condition.

---

## 2. Repository map

Roughly 2,200 files / 195 directories after excluding `.git`, `.venv`, `node_modules`. Research mass is in `experiments/results/` (pair JSONs) plus a dense, well-tested `backend/`.

```
interviewIQ/
├── backend/                 FastAPI app + all research services
│   ├── app/api/v1/          health, gap, MMR select, experiments, human-eval
│   ├── app/config/          typed settings, pricing, ablation flags
│   ├── app/schemas/         CandidateProfile, GapMatrix, pool items
│   └── app/services/        extraction, gap, generation, MMR, metrics, …
├── prompts/                 versioned Luna/Terra prompts
├── experiments/
│   ├── configs/             YAML pair subsets
│   ├── datasets/            HF resume–job-fit pairs and local classifier
│   ├── gold_set/            160-item pack + A–E CSVs + adjudicated.csv (phase 2 done)
│   ├── human_eval/          blinded study builder (no ratings yet)
│   ├── metrics/             freeze v1 + assignment audit
│   ├── results/             all paid and local runs
│   └── scripts/             stage runners
├── interviewiq_baseline/    preserved gpt-4o flat-prompt prototype
├── docs/                    architecture, protocols, result write-ups
└── notebooks/               empty (.gitkeep only)
```

**Backend services (what actually exists):**

| Package | Role | Maturity |
| --- | --- | --- |
| `llm/` | Task→model router; fail-closed; usage | Complete. OpenAI only. |
| `extraction/` | Luna CV/JD → Pydantic profiles | Complete. Used in experiments; **not** an HTTP route. |
| `gap_analysis/` | Hybrid TF-IDF + BM25 + semantic matrix | Complete. Deterministic. |
| `gap_sensitivity/` | Evidence-set ablations A–E | Complete (local, $0). |
| `fit_label_control/` | Gap rate vs HF No/Potential/Good Fit | Complete. Null result. |
| `question_generation/` | B0, B1, T pool, frozen weight map | Complete. CLI only. |
| `mmr/` + `selection/` | Weighted MMR, plain MMR, adaptive + budgets | Complete. Replay done. |
| `eval_metrics/` | M1–M4, freeze, scale decision | Complete. Freeze hashes have drifted. |
| `treatment_pilot/` | B1 / T / B2 / S0–S2 orchestration | Complete (primary-30). |
| `full_experiment/` | Spend gates for 150-pair run | Gates correctly refuse. |
| `gold_set/` | Stratified pack + analysis that refuses empty CSVs | Phase 2 complete with adjudication. G1 not interpreted; G1–G4 gates fail/no-adopt. |
| `human_eval/` | Blinded T vs B1 store, study, judge, analysis | Platform complete. Data empty. |
| `classifier_eval/` | Local MiniLM + sklearn V0–V6 | Complete. Not a thesis DV. |

**HTTP surface is thinner than the library.** Live routes: health, models, usage, profile-gap, question select, experiment browse, uploads (no LLM), metrics dashboard, human-eval admin/rater. Extraction, generation, treatment, metrics freeze, and gold-set analysis are CLI-only. `mongodb_uri` is unused. LLM tasks `NORMALIZATION`, `CLASSIFICATION`, `PROFILE_ANALYSIS`, `GAP_ANALYSIS`, `QUESTION_VALIDATION`, `QUESTION_REFINEMENT` are routed but have no callers. Orphan prompts: `prompts/analysis/profile_gap.md`, `prompts/generation/interviewiq.md`, `prompts/validation/question_validation.md`.

There is **no web UI**. Experiment browse, metrics dashboard, uploads, and human-eval admin/rater routes remain on FastAPI for scripts and `curl`.

**Tests:** 17 files, 93 `test_` functions, LLM mocked, `EMBEDDING_MODEL=hashing-test`. Gold-set analysis and human-eval `main` mode refuse missing data rather than fabricating it.

**Integrity rules that are actually enforced:** no silent model substitution; spend guards; freeze hashes; gold-set and human-eval analyzers refuse empty stores; B1 leakage assertions; research runner writes `human_evaluation: not_yet_evaluated` until real ratings exist.

`docs/architecture.md` is stale in places: it still marks Luna extraction and Terra pool as “planned”.

---

## 3. The implemented pipeline

```
CV / JD text
  → Luna structured extraction          (gpt-5.6-luna)
  → hybrid gap matrix                   (TF-IDF 0.25 + BM25_norm 0.25 + MiniLM 0.50)
  → competency weights                  (priority × severity × evidence)
  → Terra question pool (≈30)           (gpt-5.6-terra, prompt 1.1.0)
  → redundancy filter (cosine ≥ 0.85)
  → competency-weighted MMR (λ = 0.7)   → N = 10
```

### 3.1 Hybrid gap scorer (deterministic core)

Production evidence is a **deduplicated union** of technical/soft/domain skills, certifications, competency labels, and experience technologies + responsibilities — not competencies alone. It currently **omits** achievements, titles, education, projects, and professional summary.

Labels on combined score: strong ≥ 0.65, partial ≥ 0.40, weak ≥ 0.20, else missing. Reporting collapses **gap = weak + missing**.

Weight: `required=1.0` / `preferred=0.6`, times severity (missing 1.0 … strong 0.25), times `(0.5 + 0.5×(1−evidence))`. Strong matches therefore receive very small MMR relevance. That is why T’s selected set is ~95% gap-targeted.

BM25 is max-normalised **within the pair’s evidence set**, so labels are not strictly comparable across evidence ablations. Semantic-only metrics were used as a robustness check.

### 3.2 Generation conditions (fair comparison)

| ID | Inputs | Selection |
| --- | --- | --- |
| **B0** | Raw CV + JD | None (three stored Terra replicates) |
| **B1** | Extraction JSON only; **no** gap matrix | None |
| **T** | Extraction + gap → pool 30 | Competency-weighted MMR, λ=0.7 |
| **B2** | Same pool as T | Plain MMR, relevance = priority, weights = 1 |
| **S0 / S1 / S2** | Same pool | Random ×5 / top-priority / query-MMR |

Held constant: `gpt-5.6-terra`, `reasoning_effort=medium`, `max_output_tokens=12000`, seniority `mid`, N=10, same 30 pairs. Legacy `interviewiq_baseline/` (`gpt-4o`, N=15) is **not** the experimental baseline.

### 3.3 Adaptive selector (implemented after the pilot)

`backend/app/services/mmr/adaptive.py` adds dimension min-quotas (1 each of Technical, Behavioural, Situational, Cultural/Motivational, Role-Specific) and coverage-dependent λ ∈ [0.5, 0.9]. Replay on stored T pools is **exploratory** and currently reports non-frozen diagnostics because freeze hashes no longer verify.

---

## 4. Dataset and experiment inventory

HF splits: train 6,241 / test 1,759, labels No Fit / Potential Fit / Good Fit.

| Subset | Pairs | Use |
| --- | ---: | --- |
| `experiment_pairs` (seed 42) | 30 (15 Good + 15 Potential) | Primary-30 generation, extraction, gap, T/B1 |
| `experiment_pairs_holdout30` (seed 2026) | 30 | Extraction + gap only. **No T/B1 questions.** |
| `experiment_pairs_150` | 150 (75+75) | B0 only. 90 pairs have **no** extraction/gap. |
| NF primary / holdout / extras | 30+30+… | Fit-label control only. Not used for questions. |

### Completed paid / local stages

| Workstream | Artifact | Cost | Headline |
| --- | --- | ---: | --- |
| B0 Terra | `exp_20260920T220958Z`, `…221259Z`, `…223339Z` | ~$1.66 | 30×3 + 150, 10 q/pair |
| Luna extraction primary/holdout | `exp_20260927T205741Z`, `…221621Z` | ~$0.41 | 30/30 + 30/30 |
| Gap primary/holdout | `exp_20260927T215232Z`, `…223303Z` | $0 | 64.3% / 72.6% gap share |
| Gap sensitivity | `exp_20261003T042547Z` + holdout | $0 | Production evidence stands; competencies-only inflates gaps |
| Fit-label control | `exp_20261003T163033Z` | $0.673 | **Null.** Do not claim validation. |
| Treatment B1 | `exp_20261011T130310Z` | $0.602 | 30/30 |
| Treatment T pool | `exp_20261011T130619Z` | $1.846 | 30/30 |
| Selection ablations | `exp_20261011T131945Z` | $0 | B2, S0–S2 |
| Metrics freeze + scale | `exp_20261012T163747Z` … `164049Z` | $0 | Verdict **DIAGNOSE** |
| Classifier V0–V6 | `exp_20261019T123223Z` | $0 | V3, V5_LR “improve” with huge train–test gap |
| Adaptive replay | `experiments/adaptive_selection_replay/` | $0 | Budget balances dimensions; adaptive-λ without budget does not |
| 150-pair T/B1 | — | **$0** | Gates failed; not run |
| Gold-set phase 2 (+ full adjudication) | `experiments/results/gold_set_analysis/` | $0 | Human: usable n=46; G1 not interpreted |
| Gold-set round 3 (worked-rule LLM dual-code) | `experiments/results/gold_set_analysis_round3/` | ~LLM | IAA/G1/G2 pass; G4 no-adopt; exploratory |
| Selector T′ replay | `experiments/results/selector_tprime_replay/` | $0 | Exploratory success rule met (quota, gap share, M3/M4 margins) |
| Human question ratings (T′ vs B1) | pilot study ready | $0 | Infrastructure launched; **0 human ratings** |

Approximate successful LLM spend to date: **~$5.3 USD**.

---

## 5. Findings that constrain the thesis

### 5.1 Gap rates are high, and that is not an evidence-set artefact

Primary-30: 633 requirements → **64.3% gap**. Holdout-30: 515 → **72.6%**. Production evidence is already ~67 strings/pair. Competencies-only would push gaps to 94–97%. Raw CV chunks **raise** gaps to 83–86%. Size-adjusted mismatched-candidate controls still show a matching signal, but it is modest.

Luna extracts few competencies (mean ~2.7) versus many technical skills (~24). The scorer already unions those fields. Gold-set G3 (after adjudication): **10 false gaps / 20 system-gap consensus items** (rate 0.50); all classified `matched_but_scored_low` (0 `evidence_not_extracted`). Complementary misses (human not evidenced, system partial/strong) also 10/43. The bottleneck is score/threshold calibration and semantic under-match, not empty evidence strings.

### 5.2 Independent fit labels do not validate the matrix

Pre-registered H2 (No Fit vs Good Fit, AUC ≥ 0.65, both splits):

| Split | Holm *p* | AUC [95% CI] |
| --- | ---: | --- |
| Primary | 0.329 | 0.621 [0.443, 0.791] |
| Holdout | 1.0 | 0.501 [0.321, 0.692] |

**Do not claim the gap matrix tracks fit.** Possible causes: noisy HF labels, extraction, scoring, or a real mismatch between pair-level fit and requirement-level gaps. This is why M2 (gap-targeting) and M3 (requirement coverage) cannot be sold as quality.

### 5.3 Treatment T is not better than B1; the selector is the bottleneck

Pilot proxies (n=30; not human quality):

| Condition | Redundancy (↓) | Coverage@0.45 (↑) |
| --- | ---: | ---: |
| B0 (mean of 3) | 0.341 | 0.270 |
| B1 | 0.301 | 0.272 |
| **T** | **0.286** | **0.267** |
| B2 (plain MMR) | 0.279 | **0.316** |

Frozen M1–M4 decision (advisory; freeze was not blind to earlier pilot inspection):

| Contrast | M1 redundancy | M2 gap-target | M3 coverage | M4 Vendi |
| --- | --- | --- | --- | --- |
| T vs B0 | win | win | tie | win |
| T vs B1 | tie | **win** | tie | tie |
| T vs B2 | tie | win | **lose** | **lose** |

Clauses vs B1 and B2 fail → **DIAGNOSE**. Clause vs B0 holds. Projected 150-pair cost 87 CNY ≤ 100 budget, but scientific clauses fail, so spend stays blocked.

POST-HOC diagnostics (loader fix; tagged not-primary):

- Pool M3.P ≈ 0.52; T selected ≈ 0.40; **oracle-M3 ≈ 0.82** (oracle−T ≈ 0.42).
- T selected set: **95% gap** (weak/missing), 2.7% partial, 2.3% profile_probe, **0% strong**.
- Pool `profile_probe` rate ≈ 10%; validation failure ≈ 2.4%.
- Mean Jaccard T∩B2 = 0.58. Weighted MMR and plain MMR pick overlapping but not identical sets; the weight term is what drops coverage.

Interpretation: the **pool contains enough coverage**. Weighted MMR throws it away because gap weights dominate. The pool prompt asks for strengths and dimension variety; the selector then discards them. That is a direct contradiction of “competency-balanced”.

### 5.4 Adaptive replay: budgets help; adaptive λ without budgets does not

Exploratory replay, 30 stored pools, k=10, $0 LLM. Freeze did not verify, so only non-frozen diagnostics:

| Config | Mean pairwise sim | Pools meeting budget | Cultural/Motivational mean |
| --- | ---: | ---: | ---: |
| plain_mmr | 0.279 | 30 (vacuous) | 0.23 |
| weighted_mmr (= current T) | 0.286 | 30 (vacuous) | 0.23 |
| weighted_budget | 0.279 | **26/30** | **1.00** |
| weighted_adaptive (no reserved slots) | 0.293 | **1/30** | 0.23 |
| **full** (weights + budget + adaptive) | 0.286 | **26/30** | **1.00** |

`full` ≈ `full_minus_adaptive` ≈ `weighted_budget` on these diagnostics. The useful switch is the **min-quota**, not coverage-up λ. Four pools still miss quota because some dimensions have zero questions in the pool (`zero_availability`) — a **generation** problem, not a selector problem.

### 5.5 Automatic metrics are alignment/diversity proxies

M1 = mean pairwise cosine (↓). M2 = share of questions assigned to a gap requirement (↑). M3 = capped requirement coverage (↑). M4 = Vendi score (↑). Assignment uses cosine ≥ pair-specific τ (95th percentile of cross-pair B0).

Limitations already written into the spec: T sees the gap matrix, B1 does not, so M2/M3 wins vs B1 are partly mechanical. Fit-label control was null. **These metrics must not be the confirmatory quality claim.**

Freeze v1 (`experiments/metrics/metrics_freeze.json`, 2026-10-12, commit `1c62b42`) no longer verifies: `diagnostics.py` and `loaders.py` changed after freeze (pool field `valid_questions`). Confirmatory gates and frozen replay claims are blocked until a documented v1.1 re-freeze or an explicit POST-HOC tag.

### 5.6 Classifier reframing is a side analysis

V0 test accuracy 0.464, macro-F1 0.424 (below majority accuracy 0.487). Under the frozen rule, V3 and V5_LR “improve” (macro-F1 gain CI excludes 0). Their group-CV macro-F1 is ~0.999 vs test ~0.46 — a collapse that forbids treating them as a contribution. Gap-lite is **not** the production Luna matrix. Do not put this in the thesis claims section.

### 5.7 Gold-set rounds 1–3 (2026-11-26)

**Human rounds 1–2 (official for human-validation claim):** A–E + `adjudicated.csv` (30). `docs/gold_set_report.md`.

| Gate (human) | Result |
| --- | --- |
| Main IAA QWK | −0.033 [−0.317, 0.254] → **G1 not interpreted** |
| Usable consensus | 46/50 |
| G1 / G2 | unsupported / does not separate |
| G4 | adopt flag True on primary→holdout; reverse False; **no production cutover** |

**Round 3 (exploratory, same pack, LLM P1/P2 + worked-rule brief):** `docs/gold_set_round3_report.md`, figures under `docs/figures/round3/`.

| Gate (round 3) | Result |
| --- | --- |
| Main IAA QWK | **0.842 [0.712, 0.935]** → **G1 interpreted** |
| Usable consensus | 40/50 (agree 40 / disagree 8 / cannot_judge 2) |
| G1 | QWK **0.624 [0.444, 0.751]**; Spearman **0.752 [0.579, 0.849]** → **supported** |
| G2 | combined AUC **0.809 [0.640, 0.944]** → separates; does not beat every single signal |
| G3 | false gaps 12/17 system gaps (matched-but-scored-low) |
| G4 | **no-adopt** |

Question study (T′ vs B1): rubric pilot ready; pilot export still **0** ratings. LLM judge pending.

---

## 6. Gap versus the thesis goal

| Thesis requirement | What the tree does | What is missing |
| --- | --- | --- |
| Hybrid gap analysis | Frozen 0.25/0.25/0.50 scorer, sensitivity, fit-label control, gold-set G1–G4 | Human validation **failed pre-registered gates**; keep production config; do not claim gap accuracy |
| Personalization | Profiles + gap-conditioned pool | Evidence that T/T′ questions are *better for this pair* than B1 (needs humans); gap labels themselves are unvalidated |
| Competency balance | Weights + five-dimension schema + **adopted T′** | Historical T unbalanced; T′ is forward treatment on stored pools |
| Controlled LLM generation | Matched Terra B0/B1/T on primary-30 | Holdout-30 T/B1; 90 missing extractions; no 150-pair run |
| Confirmatory evaluation | M1–M4 freeze + DIAGNOSE | Human quality DV; freeze v1.1; do not reopen SCALE by editing the rule |

The honest current contribution is: **a reproducible hybrid pipeline plus converging negative/null results** — gap-targeting MMR does not beat profile-aware B1 / plain MMR B2 on automatic coverage; the gap matrix tracks neither HF fit labels nor (after adjudication) human requirement labels under G1–G2; G4 keeps production weights; and T′ restores competency balance on stored pools without new generation.

That is a publishable finding if framed as such. It is **not** a positive demonstration of the full title.

---

## 7. Best way forward

Do **not** pay for a 150-pair T/B1 run on the current selector. Do **not** change the SCALE rule after seeing DIAGNOSE. Do **not** treat M2 as quality.

The title is recovered by fixing **balance and validity**, then testing quality on humans, and only then (optionally) scaling generation.

### Step 1 — Integrity housekeeping (1–2 days, $0)

1. Write a short addendum that `docs/architecture.md` is stale; extraction and pool are implemented. Do not silently rewrite frozen protocol sections.
2. Decide freeze policy in writing: **re-freeze as metrics v1.1** documenting the `valid_questions` loader fix, *or* keep v1 and tag all later metric numbers `POST-HOC`. Do not mix.
3. Confirm gold-set pack files (`items_blind.json`, `annotator_app.html`, `answer_key_…json`) are backed up. They are large CV-containing artifacts; losing them forces a resample and breaks the frozen seed `20261004`.
4. Leave `docs/judge_validation_prereg.md` at `approval: pending`.
5. Treat classifier V3/V5_LR as appendix only.

### Step 2 — Gold-set annotation — DONE (rounds 1–3)

**Human (1–2):** full adjudication; G1 not interpreted. Official human record: `docs/gold_set_report.md`.

**Round 3 (worked-rule retrain + dual-coding, same pack):**  
Training brief `experiments/gold_set/round3/training/worked_rule_brief.md`; dual-coders P1/P2 via `experiments/scripts/round3_annotate.py`; analyze `experiments/scripts/analyze_gold_set_round3.py` → `docs/gold_set_round3_report.md`.

**Outcome:** Round 3 clears IAA and supports G1/G2 under frozen rules; G4 no-adopt. **Integrity:** LLM labels ≠ human replication. Thesis *human*-validation claim still rests on rounds 1–2 unless humans redo round 3. Production weights stay unless an explicit cutover cites which label source.
- If agreement is near chance → the “hybrid profile-gap” claim must be weakened in the thesis. Generation can still be studied as a prompt+selector system, but not as “gap-validated personalization”.

### Step 3 — Diagnose, redesign, and adopt T′ — DONE (2026-11-26)

T′ replay met its pre-registered exploratory success rule (quota 26/30, gap share 0.71, M3/M4 within margin of T; B2−T′ M3 delta 0.027). Mean Cultural/Motivational rose from ~0.23 (T) to ~0.97 (T′).

**Adoption (this step):** T′ is the forward treatment. Defaults, artifacts, and human-eval wiring:

- Decision: `docs/selector_tprime_adoption.md`
- Code default: `tprime` (`FORWARD_TREATMENT_SELECTION`, treatment CLI/pipeline)
- Materialized sets: `experiments/results/selector_tprime_replay/treatment/` (30)
- Human-eval inputs: `experiments/human_eval/inputs/primary30_Tprime_vs_B1.json` (T key = T′)
- Addendum: `docs/human_eval_tprime_addendum.md`
- Historical weighted-MMR T left unchanged under `exp_20261011T130619Z/treatment/`

Historical design note (kept for context). Pre-register a **new treatment T′** *before* looking at frozen M1–M4 for it:

Recommended T′ (aligned with the title’s “competency-balanced”):

1. Keep the **same stored T pools** (no new Terra spend).
2. Replace `competency_weighted_mmr` with **`full`** or **`weighted_budget`**: min-quota 1 per canonical dimension.
3. Add a **strength/partial floor** so strong and partial matches cannot be driven to 0% (for example, reserve 1–2 slots for non-gap `target_gap` when the pool has them). This is the missing piece of “balanced”.
4. Keep λ = 0.7 unless the pre-reg explicitly adopts coverage-up adaptive λ; the replay suggests λ adds little once budgets are on.
5. Secondary ablations (already named): `plain_mmr`, `weighted_mmr` (current T), `full_minus_weights`, `full_minus_budget`.

Compare T′ vs T vs B1 vs B2 on primary-30 using **frozen** M1–M4 after Step 1’s freeze policy. Exploratory only (same pairs the pilot used). Success criterion to write down in advance, for example:

- T′ ≥ T on M3 and M4, and not worse than T on M1;
- T′ closer to B2 on coverage without collapsing M2 to B2;
- selected gap share **below** 95%, with all five dimensions represented in most pairs.

If T′ still loses M3 to B2 by a wide margin, the weight term itself is the problem and the thesis should treat **plain MMR + dimension budget** as the balanced selector, with gap weights as a failed ablation.

Also inspect the **four zero-availability pools**: if Cultural/Motivational is absent from the pool, no selector can balance it. That becomes a pool-prompt constraint for any future generation (force at least one question per dimension in the pool of 30).

### Step 4 — Human evaluation of questions — IN PROGRESS (pilot ready, 2026-11-26)

Automatic metrics cannot close the thesis. Protocol already written; T′ addendum applies.

**Done (no invented ratings):**

1. Built rubric pilot: 5 primary-30 pairs T′ vs B1 → `experiments/human_eval/inputs/pilot5_Tprime_vs_B1.json`, blinded study + condition key, `manifest_pilot.json`.
2. Local API smoke with `HUMAN_EVAL_DATA_DIR` / admin token; session blinding checked; demo rating path checked; pilot export = 0.
3. Local rater UI: `experiments/human_eval/rater_app.html`.
4. Launch checklist: `docs/human_eval_launch.md`. Kit briefing updated.

**Blocked on humans:**

1. Recruit ≥3 raters; they complete the 5-pair pilot via token + rater app.
2. Export pilot → freeze `docs/human_eval_rubric.md` (at most one wording revision); record median `elapsed_ms`.
3. Optional exploratory primary-30 study from `primary30_Tprime_vs_B1.json` (label exploratory).
4. Confirmatory **main** = holdout-30 × {T′, B1} still needs holdout generation (Step 5) under the spend gate.
5. Analysis unchanged: refuse <90% completion; Krippendorff α; Wilcoxon T′−B1; Holm; preference binomial.
6. Luna judge stays pending approval.

If humans prefer B1 or are indifferent, that is the result. Do not keep generating until a proxy looks better.

### Step 5 — Paid generation only under a new, written gate

Revisit Terra spend **after** Steps 2–4 have a written outcome. A new pre-registration should say explicitly that the old SCALE rule is the historical spend gate, and that a **new** run is allowed because the **selector (and possibly the pool prompt) changed**, not because the DIAGNOSE rule was edited.

If gold-set + T′ replay + (optional) exploratory human pilot look promising:

1. Dry-run cost. Historical projection ~$12.88 / 87 CNY for 90 extractions + B1×150 + T pool×150. Recalculate.
2. Extract the missing 90 pairs (Luna). Reuse stored primary and holdout extractions.
3. Generate B1 and T′ pools on development-120 and holdout-30. **Do not regenerate B0.**
4. Offline replay of B2 and the named selection grid on the new pools.
5. Compute development metrics and commit them **before** opening holdout-30.
6. Confirmatory contrast remains **T′ vs B1 on holdout-30**, frozen M1–M4, plus the human study on holdout-30.
7. Report development and confirmation separately. Do not pool them for the confirmatory claim.

If gold-set or humans are null/negative: **stop paid generation**. Write the thesis around the implemented framework, the ablations, and the negative results. That is a complete research story.

### Step 6 — Thesis write-up (can start now in parallel)

Chapters that can be drafted from existing artifacts:

1. Problem and related work (flat LLM prompting vs structured gap analysis).
2. Hybrid gap method, thresholds, weights, evidence assembly.
3. Sensitivity (figures 17–21) and fit-label control (figures 22–25) — including the null.
4. Generation conditions B0/B1/T/B2 and fair-comparison protocol.
5. Primary-30 pilot and DIAGNOSE (figures 40–43, 50–56).
6. Adaptive/budget selector as the response to DIAGNOSE (figures 60–61).
7. Gold set and human eval (to be filled).
8. Limitations: noisy gaps, by-construction M2, n=30, single dataset, MiniLM, no interactive full-pipeline API.

---

## 8. Areas requiring critical attention now

Ranked by how much they can sink the thesis if ignored.

### P0 — correctness of the scientific claim

1. **Gold-set negative.** After adjudication, hybrid gap still fails G1/G2 interpretation/support; G4 keeps production. Do **not** write that the matrix is accurate. Report the nulls.
2. **Historical T is not competency-balanced; T′ is adopted.** Use T′ (not weighted-MMR T) for human study and any new selection.
3. **DIAGNOSE + freeze hash drift.** Confirmatory 150-pair spend is blocked on purpose. Do not bypass gates. Do not retune the decision rule. Resolve freeze v1 vs v1.1 in writing.
4. **M2/M3 are not quality.** T is conditioned on the gap matrix; B1 is not. A win on gap-targeting is expected. Human preference is the DV.
5. **Null fit-label control must stay in the narrative.** Hiding it and claiming “gaps measure misfit” would be a research-integrity failure.

### P1 — experimental design traps

6. **Selector vs generator confound.** Oracle-M3 ≫ T-M3, and B2 > T on coverage. If the next paper/chapter blames “the LLM pool”, it will likely be wrong. Diagnose on stored pools first.
7. **Pool dimension holes.** Budget shortfalls from `zero_availability` cannot be fixed by MMR. Any new pool prompt must require all five dimensions.
8. **Holdout is unused for generation.** Confirmatory T vs B1 is defined on holdout-30, which has extraction/gap but no questions. Primary-30 results are exploratory by the project’s own pre-reg.
9. **90/150 pairs lack extraction.** Linear cost projections assume they will be extracted; they have not been. Do not imply a 150-pair treatment exists. B0-150 is flat baseline only.
10. **Human-eval disk.** Default store is ephemeral on RunPod. Ratings will vanish without `HUMAN_EVAL_DATA_DIR` on a volume. Condition keys stay off the API and out of git.
11. **Judge not approved.** Running it on holdout before approval contaminates a pre-registered file.

### P2 — method and engineering debt that affects results

12. **Evidence assembly incompleteness.** Achievements, titles, education, projects, summary are extracted but unused in `_candidate_texts`. Gold-set false gaps may be extraction-coverage, not IR failure. Expanding evidence after seeing gold labels requires a pre-declared robustness analysis, not a silent production change.
13. **BM25/TF-IDF scale depends on the evidence set.** Documented; still easy to mis-compare conditions. Prefer semantic-only or fixed-bm25 when arguing across evidence ablations.
14. **Weight formula vs balance.** Severity 0.25 on strong matches almost forbids strength probes. If the title wants balance, the weight schedule is a first-class design choice, not a detail.
15. **HTTP `/questions/select` defaults weights to `q.priority`.** The frozen pilot rule forbids that. Demo API and experiment path can silently diverge.
16. **No extraction/generation API.** Fine for the thesis; dangerous if someone demos “the system” from `/upload` and thinks they ran InterviewIQ. Upload stores files only.
17. **Unused LLM tasks and orphan prompts.** `GAP_ANALYSIS` is routed to Terra but production gap is deterministic. Do not describe an LLM gap step in the method chapter.
18. **Stale `docs/architecture.md`.** Reviewers will quote it. Update or mark historical.
19. **Classifier train–test collapse.** V3/V5_LR must not appear as “we also built a fit model that works”.
20. **Single dataset, English, MiniLM-L6-v2, n=30 for T.** Power is low (already labelled exploratory). Diversity of jobs is whatever the HF sample drew. State this in limitations now, not after a reviewer asks.

### P3 — operational

21. **Spend discipline is a feature.** Keep `--confirm-spend`, `--max-cost`, and fail-closed model IDs.
22. **Empty `notebooks/`.** Not blocking; analysis lives in scripts + `docs/figures/`.
23. **Human-eval ratings** go through the FastAPI session/ratings routes (no rater UI in this repo).
24. **Gemini / MongoDB** are comments, not work. Ignore until the research claim is closed.

---

## 9. Suggested next sequence (post gold-set + T′)

| Priority | Work | Spend | Output |
| --- | --- | --- | --- |
| Done | Gold-set human rounds 1–2 | $0 | G1 not interpreted |
| Done | Gold-set round 3 worked-rule LLM dual-code + analyze | LLM | IAA/G1/G2 pass (exploratory); G4 no-adopt |
| Done | T′ replay on stored pools | $0 | Exploratory success rule met |
| Done | Adopt T′ as forward treatment + materialize + human-eval inputs | $0 | `docs/selector_tprime_adoption.md` |
| Done | Rubric pilot study build + API smoke + rater UI | $0 | `docs/human_eval_launch.md`; ratings still 0 |
| Now | Recruit ≥3 raters; complete 5-pair pilot | Rater time | Pilot export; freeze rubric |
| Next | Exploratory primary-30 human study (T′ vs B1) if rubric freezes | Rater time | Human DV on relevance / coverage / non-redundancy / preference |
| Gate | Paid holdout T′/B1 generation | Terra | **Only if** humans prefer T′ or show material gains; else stop spend |
| Parallel | Draft thesis chapters around implemented pipeline + null gold-set + DIAGNOSE + T′ | $0 | Negative-capable narrative |

Gold-set is clearly negative on gap validity. That does **not** by itself kill the question-generation half of the title: competency-budget T′ can still be tested as a selection method against B1 with humans. It does kill claims that personalization is “gap-validated.” Skip 150-pair spend unless the human pilot justifies it.

---

## 10. Key paths

| What | Where |
| --- | --- |
| This report | `report.md` |
| Earlier snapshot (2026-10-11) | `docs/project_report.md` |
| Architecture (stale in places) | `docs/architecture.md` |
| Research pipeline / conditions | `docs/research-pipeline.md` |
| Treatment pilot | `docs/treatment_pilot_report.md` |
| Scale decision (DIAGNOSE) | `docs/scale_decision_report.md` |
| Metrics spec / freeze | `docs/metrics_spec.md`, `experiments/metrics/metrics_freeze.json` |
| Fit-label null | `docs/fit_label_control_report.md` |
| Gap sensitivity | `docs/gap_sensitivity_report.md` |
| Gold-set protocol | `docs/gold_set_protocol.md` |
| Gold-set phase-2 report | `docs/gold_set_report.md` |
| Gold-set analysis summary | `experiments/results/gold_set_analysis/summary.json` |
| Main-set adjudication | `experiments/gold_set/adjudicated.csv` (30 rows) |
| Final-four adjudication notes | `experiments/gold_set/round2/main_open_adjudication.md` |
| Gold-set round 3 protocol/report | `docs/gold_set_round3.md`, `docs/gold_set_round3_report.md` |
| Round 3 analysis | `experiments/results/gold_set_analysis_round3/` |
| Round 3 figures | `docs/figures/fig*_round3_gold_*.png` (also `docs/figures/round3/`) |
| Selector T′ replay | `experiments/results/selector_tprime_replay/summary.json` |
| T′ adoption | `docs/selector_tprime_adoption.md` |
| T′ treatment sets | `experiments/results/selector_tprime_replay/treatment/` |
| Human-eval T′ vs B1 inputs | `experiments/human_eval/inputs/primary30_Tprime_vs_B1.json` |
| Rubric pilot (5 pairs) | `experiments/human_eval/inputs/pilot5_Tprime_vs_B1.json`, `manifest_pilot.json` |
| Human-eval launch | `docs/human_eval_launch.md` |
| Rater UI | `experiments/human_eval/rater_app.html` |
| Adaptive selector pre-reg | `docs/adaptive_selection_prereg.md` |
| 150-pair pre-reg + failed gates | `docs/full_experiment_prereg.md`, `docs/full_experiment_report.md` |
| Human eval | `docs/human_eval_protocol.md`, `docs/human_eval_runbook.md` |
| Judge (pending) | `docs/judge_validation_prereg.md` |
| Gap scorer | `backend/app/services/gap_analysis/matrix.py` |
| Weighted MMR | `backend/app/services/mmr/select.py` |
| Adaptive MMR | `backend/app/services/mmr/adaptive.py` |
| T pool prompt | `prompts/generation/interviewiq_pool.md` (v1.1.0) |
| Primary extraction / gap | `exp_20260927T205741Z` / `exp_20260927T215232Z` |
| B1 / T / ablations | `exp_20261011T130310Z` / `…130619Z` / `…131945Z` |
| Gold-set pack | `experiments/gold_set/` |
| Gold-set figures 26–31 | `docs/figures/fig26_gold_*.png` … `fig31_gold_gates.png` |
| Adaptive replay | `experiments/adaptive_selection_replay/` |

---

## 11. Conclusion

InterviewIQ is a **research-grade implementation** of the named framework, not a prototype sketch. After gold-set adjudication, the hybrid gap half of the title is **implemented and empirically unvalidated** under the frozen G1–G4 rules. T′ shows that competency-balanced selection is achievable on stored pools without new LLM spend.

The remaining evidence gap is **human preference and quality ratings for generated questions** (objectives 3–4). Do that on **adopted T′** vs B1 (`primary30_Tprime_vs_B1.json`) before any further Terra generation. A DIAGNOSE result, a null fit-label control, a negative gold set, and a human study that may prefer B1 are all legitimate thesis outcomes. An unvalidated claim that historical gap-conditioned T is better is not.
