# Gold-set protocol: independent human labels for the profile-gap matrix

**This hypotheses and decision-rule section was written before the annotation pack
was built and before any gold-set analysis. It must not be edited after that point.**

---

## Pre-registered hypotheses and rules

*(Frozen. Do not edit.)*

The gap matrix’s label ordering has so far been checked only against its own score
(partly circular). This gold set supplies independent human judgement of whether a
**job requirement is evidenced in the whole CV**.

### G1

System three-category labels agree with human consensus labels beyond chance
(quadratic-weighted kappa and Spearman between combined score and human ordinal
label).

### G2

The combined score separates human-judged “met/partial” from “not evidenced”
(ROC AUC with bootstrap CI), and we can say whether the hybrid beats each single
signal (TF-IDF, BM25_norm, semantic).

### G3

Quantify false gaps: requirements the system labels gap (weak/missing) that humans
judge met or partial. Split causes where possible (evidence not extracted vs matched
but scored low).

### G4

Weights/thresholds tuned on one split improve agreement on the other split. Adopt a
recalibrated configuration **ONLY** if the improvement on the untouched split has a
bootstrap 95% CI that excludes 0; otherwise keep the production configuration and
say so.

Landis–Koch bands for kappa are used **only as descriptive conventions** and must
be labelled as such. Null or weak results are reported plainly.

---

## Code inspection

*(Filled after reading stored gap artifacts and mapping code; before sampling.)*

### Per-requirement fields in stored gap JSON

Production gap artifacts (`experiments/results/exp_20260927T215232Z/gap_analysis/*.json`
and holdout `exp_20260927T223303Z`) store `gap_matrix.gaps[]` with:

- `job_competency` — requirement text (extracted normalized label)
- `priority` — `required` / `preferred` / `unspecified`
- `match_label` — `strong_match` / `partial_match` / `weak_match` / `missing`
- `weight` — production competency weight (priority × severity × evidence; **not** a
  stable requirement id)
- `evidence.tfidf_score`, `evidence.bm25_score` (raw), `evidence.semantic_score`
- `evidence.matched_candidate_competency` and `candidate_evidence[]` — best-matching
  production evidence string(s)

Combined score is **not stored**. It is reconstructed as production
`0.25*tfidf + 0.25*(bm25_raw/pair_bm25_max) + 0.50*semantic`. Pair `bm25_max` is the
max raw BM25 among that pair’s gaps.

Three-category: strength = strong; partial = partial; gap = weak + missing.

### Pair → raw CV and JD

`pair_id` = `profile_id` = `jd_id`. Raw CV:
`experiments/datasets/resume_job_fit/experiment_pairs/profiles/{pair_id}.txt` (primary)
or `experiment_pairs_150/profiles/{pair_id}.txt` (holdout). JD: sibling `jobs/{pair_id}.txt`.
Fit label is on the gap JSON (`fit_label`) and the subset manifest.

### Requirement IDs

There is **no stable dataset-level requirement id**. We assign
`requirement_id = "{pair_id}::{index:03d}"` using the **order of `gap_matrix.gaps` in
the stored JSON** (the order produced by `_job_competencies`: required_skills then
preferred_skills). That index is stable for these frozen artifacts.

---

## Sampling design (Phase 1)

- Seed: `20261004`.
- 150 items = 75 Primary-30 + 75 Holdout-30, stratified by production four-way label
  with targets **19 strong / 19 partial / 19 weak / 18 missing** per split (strong
  oversampled because it is rare).
- At most **3 requirements per pair**; at least **25 distinct pairs per split**.
- Balance Good Fit / Potential Fit pair counts where the constraints allow.
- Inclusion probability stored as conventional π = n_stratum_sampled / N_stratum;
  Horvitz–Thompson weight w = 1/π. (The spec’s phrase “stratum size / stratum sample
  size” is the design weight N/n = 1/π; both are written on each item.)
- **10 calibration items** from the same pair pools, disjoint from the 150, annotated
  by both annotators first.

## Rubric summary

Against the **whole CV** (not extracted evidence): 2 = clearly met; 1 = partially met;
0 = not evidenced; plus `cannot_judge`. Full text: `docs/gold_set_guidelines.md`.

## Workflow

1. Both annotators complete the 10 calibration items, then discuss.
2. Independent annotation of the 150 items (separate random orders).
3. Disagreements adjudicated by discussion into `adjudicated.csv` if two annotators.
4. **Fallback if only one annotator:** single annotation of 150 + repeat of a random
   30-item subset after at least 7 days; report **test–retest kappa** instead of
   inter-annotator kappa and state that clearly.

Expected effort: about 1.5 minutes per item ≈ 4 hours per annotator (150 + 10).

## Staffing addendum (2026-11-12)

The frozen sample is unchanged (seed `20261004`, 150 main + 10 calibration).
The two-annotator workload was too large, so main-item dual-coding is split
across **four** annotators without dropping the two-label design required for G1.

- Annotators: A, B, C, D.
- Assignment seed `20261112` (staffing only; does not resample items).
- Calibration (10 items): all four annotators, then a joint discussion.
- Main (150 items): balanced incomplete block. The six pairs AB, AC, AD, BC, BD,
  CD each receive 25 items. Every main item still has exactly two labels.
- Load per annotator: 10 calibration + 75 main ≈ **2 hours**.
- Place `annotator_A.csv` … `annotator_D.csv` under
  `experiments/gold_set/annotations/`. Optional `adjudicated.csv` for
  disagreements on dual-coded main items.
- Analysis pools the 150 assigned pairs for main-set kappa, and reports
  calibration agreement across all four annotators.

G1–G4 and the item sample are not edited.

## Phase 1 stop rule

After the pack is written, **do not run analysis**. Analysis (`gold_set_analyze`)
starts only when real files exist under `experiments/gold_set/annotations/`.
