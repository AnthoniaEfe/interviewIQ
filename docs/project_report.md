# InterviewIQ — Project Report

**Research-grade interview question generation with profile-gap analysis**

| Field | Value |
| --- | --- |
| Repository | `interviewIQ` |
| Report date | 2026-10-04 |
| Primary dataset | [`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit) |
| Generation model (controlled) | `gpt-5.6-terra` |
| Extraction model | `gpt-5.6-luna` |
| Embedding model | `all-MiniLM-L6-v2` |
| Related deep-dives | [Gap sensitivity](gap_sensitivity_report.md) · [Fit-label control](fit_label_control_report.md) · [Gold-set protocol](gold_set_protocol.md) · [Classifier reframing](classifier_reframing_report.md) |

---

## 1. Executive summary

InterviewIQ is a research system for generating interview questions that are conditioned on **structured candidate–job profile gaps**, rather than a single flat prompt over raw CV and JD text. The repository implements:

1. A **legacy flat-prompt baseline** (`interviewiq_baseline/`, historically `gpt-4o`).
2. A **FastAPI application** with Luna/Terra routing, TF-IDF + BM25 + semantic retrieval, gap matrix labelling, competency weighting, and competency-weighted MMR.
3. An **experiment runner** with stages for baseline generation, Luna extraction, gap analysis, evidence-set sensitivity, fit-label control, gold-set pack/analysis, and local classifier evaluation.
4. A **Next.js research console** for browsing runs, costs, and question sets.

**Work completed so far (quantitative):**

| Workstream | Status | Headline result |
| --- | --- | --- |
| Local fit classifier (original V0) | Done (metrics frozen 2026-09-20) | Test accuracy 0.464, macro-F1 0.424 (not a thesis DV) |
| Classifier reframing (V0–V6, group CV, gap-lite) | **Running** | Protocol frozen; MiniLM + gap-lite features being computed on the 8,000-pair splits. Do not cite variant metrics until `classifier_reframing_combined/summary.json` exists |
| Terra baseline generation | Done | 30-pair and 150-pair runs; 10 questions/pair; ~$0.25 / ~$1.15 |
| Luna extraction (Primary-30) | Done | 30/30 pairs; $0.213; mean ~2.8 competencies vs ~24 tech skills |
| Luna extraction (Holdout-30) | Done | 30/30 pairs; $0.196 |
| Luna extraction (fit-label No Fit + extras) | Done | `exp_20261003T163033Z`; 98/98 pairs; **$0.673**; 0 failures |
| Gap analysis (Primary-30) | Done | 633 requirements → **64.3%** gap (weak+missing) |
| Gap analysis (Holdout-30) | Done | 515 requirements → **72.6%** gap |
| Evidence-set sensitivity | Done | Competencies-only would inflate gaps to 94–97%; raw CV does **not** beat production evidence |
| Fit-label control (HF labels vs gap rate) | Done | Frozen rule: **does not separate** No Fit vs Good Fit on both splits; do not claim validation |
| Gold-set (human gap labels) | Phase 1 pack **built**; Phase 2 **not run** | 150 items + 10 calibration; **no human labels yet** (none fabricated) |
| Full InterviewIQ question generation | Not yet run as controlled experiment | Pipeline modules exist; Terra pool + MMR stage pending batch eval |
| Human evaluation of questions | Not yet | Runner marks `human_evaluation: not_yet_evaluated` |

**Thesis-relevant takeaway from gap work:** the stored high gap rate is **not** an artefact of scoring only ~2.7 extracted competencies—the production scorer already uses an expanded evidence union (~67 strings/pair). Substituting raw CV chunks does not reduce gaps. Restricting evidence to competencies alone *would* inflate gaps to ~94–97%, and that inflation exceeds a mismatched-candidate size-bias control. **Independent Hugging Face fit labels do not validate the matrix** under the pre-registered coarse rule (Section 9). Independent **requirement-level** human labels are the next check (gold set, Section 10); they have not been collected yet.

---

## 2. Research goals and design

### 2.1 Independent variable

**Generation methodology**, holding candidate, job, generation model, decoding config, final question count \(N\), and evaluation criteria fixed.

| Mode | Pipeline |
| --- | --- |
| **Baseline** | CV + JD (+ seniority) → Terra → \(N\) questions. No gap matrix, no competency weights, no MMR. |
| **InterviewIQ** | Extract → normalize → TF-IDF/BM25/semantic → gap matrix → weights → Terra question pool → redundancy filter → competency-weighted MMR → \(N\) questions |

Do **not** compare a Luna baseline against a Terra InterviewIQ condition.

### 2.2 Research integrity rules

- No fabricated experiment numbers; unavailable metrics are `null`.
- Model IDs are configuration-driven; business logic selects an `LLMTask`, never a hard-coded model string.
- Fail closed on invalid API key / model ID (no silent substitution).
- Luna/Terra are never fine-tuned; only a local embedding + logistic-regression (and related sklearn) fit classifier is trained.
- Gold-set analysis never invents annotator labels. Classifier “improved over V0” and gold-set G4 recalibration are **pre-registered adoption rules**; they are not applied until the corresponding numbers exist.

---

## 3. System architecture

```
CV/JD text
  → Luna structured extraction (CandidateProfile / JobDescription)
  → TF-IDF + BM25 + Sentence-Transformers
  → profile-gap matrix + competency weights
  → Terra question pool (application path)
  → question embeddings + redundancy filter
  → competency-weighted MMR
  → final question set
```

### 3.1 Repository layout

| Path | Role |
| --- | --- |
| `backend/` | FastAPI app: config, LLM router, retrieval, embeddings, gap analysis, MMR, gap sensitivity, fit-label control, gold-set, classifier eval |
| `prompts/` | Versioned prompt files (extraction, generation, validation) |
| `experiments/` | YAML configs, runner, datasets, gold-set pack, result artifacts |
| `frontend/` | Next.js research console |
| `interviewiq_baseline/` | Preserved flat-prompt prototype (`gpt-4o`) |
| `docs/` | Architecture, routing, experiments, sensitivity / fit-label / gold-set / classifier reports, figures |

### 3.2 Stack

- **Backend:** FastAPI, Pydantic v2, OpenAI SDK  
- **Retrieval:** scikit-learn TF-IDF, `rank-bm25`  
- **Embeddings:** Sentence-Transformers `all-MiniLM-L6-v2` (CPU or CUDA)  
- **Persistence:** file-based `experiments/results/<exp_id>/`  
- **Frontend:** Next.js (experiment list, charts, question inspector, CV/JD upload)

### 3.3 LLM routing

| Task class | Model env | Default |
| --- | --- | --- |
| Extraction, normalization, validation | `LLM_FAST_MODEL` | `gpt-5.6-luna` |
| Profile/gap interpretation, question generation/refinement | `LLM_PRIMARY_MODEL` | `gpt-5.6-terra` |

Default pricing (overridable via env; from OpenAI docs as of 2026-09-20):

- Terra: $2.00 / $0.20 cached / $12.00 output per 1M tokens  
- Luna: $0.20 / $0.02 cached / $1.20 output per 1M tokens  

### 3.4 Gap scoring (deterministic core)

For each job requirement against candidate evidence strings:

\[
\text{combined} = 0.25\cdot\text{tfidf} + 0.25\cdot\text{bm25\_norm} + 0.50\cdot\text{semantic}
\]

| Label | Threshold on combined |
| --- | --- |
| strong | ≥ 0.65 |
| partial | ≥ 0.40 |
| weak | ≥ 0.20 |
| missing | &lt; 0.20 |

**Three-category matrix used in reporting:** strength = strong; partial = partial; **gap = weak + missing**.

`bm25_norm` = raw BM25 / max raw BM25 over requirements in the same pair (evidence-set dependent). Semantic similarity uses max cosine over candidate strings (`all-MiniLM-L6-v2`).

Ablation flags in settings can independently disable TF-IDF, BM25, semantic, gap analysis, weighting, MMR, and redundancy filtering.

---

## 4. Dataset and pair curation

### 4.1 Hugging Face resume–job fit dataset

| Split | Rows | Labels |
| --- | ---: | --- |
| train | 6,241 | No Fit / Potential Fit / Good Fit |
| test | 1,759 | same |

Columns: `resume_text`, `job_description_text`, `label`.

### 4.2 Local fit classifier (calibration aid, not thesis DV)

Trained on train split; evaluated on test. Embedding model: `all-MiniLM-L6-v2`. Artifact: `experiments/datasets/resume_job_fit/models/fit_classifier_metrics.json` (trained 2026-09-20). New reframing models, if any, go under `experiments/datasets/resume_job_fit/models_reframing/` and **must not overwrite** this artifact. A frozen protocol for variants V0–V6, group-aware CV, and LLM-free gap-lite features is in [classifier_reframing_report.md](classifier_reframing_report.md). **Test-set variant tables are not in this report yet** (evaluation still running as of 2026-10-04).

| Metric | Value |
| --- | ---: |
| Accuracy | 0.464 |
| Macro-F1 | 0.424 |
| Weighted-F1 | 0.462 |

**Per-class (test):**

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| No Fit | 0.574 | 0.600 | 0.587 | 857 |
| Potential Fit | 0.319 | 0.351 | 0.334 | 444 |
| Good Fit | 0.389 | 0.319 | 0.351 | 458 |

**Confusion matrix** (rows = true, cols = predicted; order No Fit / Potential Fit / Good Fit):

|  | Pred No Fit | Pred Potential | Pred Good |
| ---: | ---: | ---: | ---: |
| True No Fit | 514 | 205 | 138 |
| True Potential | 197 | 156 | 91 |
| True Good | 184 | 128 | 146 |

### 4.3 Experiment pair subsets

Stratified **Good Fit** and **Potential Fit** pairs for generation / extraction (No Fit excluded from interview-question experiments).

| Subset | Pairs | Composition | Config |
| --- | ---: | --- | --- |
| `experiment_pairs` | 30 | 15 Good + 15 Potential | `primary_hf.yaml` (seed 42) |
| `experiment_pairs_holdout30` | 30 | 15 + 15 | `primary_hf_holdout30.yaml` (seed 2026) |
| `experiment_pairs_150` | 150 | 75 + 75 | `primary_hf_150.yaml` |
| `experiment_pairs_nf_primary30` | 30 | No Fit, seed 42, disjoint from used fit-positive ids | fit-label control |
| `experiment_pairs_nf_holdout30` | 30 | No Fit, seed 2026, disjoint from NF-primary | fit-label control |
| `experiment_pairs_nf_matched` / `_control` | extras | additional No Fit / matched-job pairs in the 98-pair extraction | fit-label control |

No Fit pairs are used only for **gap-vs-label** analysis, not for interview-question generation.

---

## 5. Experiment inventory

All artifacts under `experiments/results/`. Status values and costs are taken from each run’s `summary.json` (not estimated).

### 5.1 Baseline generation (Terra)

| Experiment ID | Status | Pairs | Questions/pair | Cost (USD) | Mean latency (ms) | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `exp_20260920T210557Z` | failed | 3 | — | 0 | — | Early smoke failures |
| `exp_20260920T210619Z` | failed | 3 | — | 0 | — | Early smoke failures |
| `exp_20260920T212014Z` | failed | 3 | — | 0 | — | Early smoke failures |
| `exp_20260920T212053Z` | completed | 3 | 10 | 0.018744 | 6402 | Smoke OK |
| `exp_20260920T220958Z` | completed | 30 | 10 | 0.246540 | 5767 | Primary-30 baseline |
| `exp_20260926T221259Z` | completed | 30 | 10 | 0.245712 | 5736 | Primary-30 baseline (repeat) |
| `exp_20260926T223339Z` | completed | 150 | 10 | 1.147875 | 5461 | 150-pair baseline |

### 5.2 Luna extraction

| Experiment ID | Status | Split | Pairs OK | Cost (USD) | Mean latency (ms) | Prompt versions |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `exp_20260927T205539Z` | failed | smoke | 0/1 | 0 | — | 1.0.0 / 1.0.0 |
| `exp_20260927T205622Z` | completed | smoke | 1/1 | 0.010099 | 42274 | 1.0.0 / 1.0.0 |
| **`exp_20260927T205741Z`** | completed | **Primary-30** | 30/30 | **0.213411** | 31260 | 1.0.0 / 1.0.0 |
| `exp_20260927T220439Z` | completed | (30) | 30/30 | 0.207788 | 33233 | Related holdout-path prep |
| `exp_20260927T221621Z` | completed | **Holdout-30** | 30/30 | **0.196053** | 32751 | 1.0.0 / 1.0.0 |
| **`exp_20261003T163033Z`** | completed | **NF + extras (fit-label)** | **98/98** | **0.673245** | — | 1.0.0 / 1.0.0 |

### 5.3 Gap analysis

| Experiment ID | Source extraction | Split | Requirements | Strength | Partial | Gaps (weak+missing) | Gap share |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `exp_20260927T215127Z` | Primary | smoke (1) | 2 | 1 | 0 | 1 | — |
| **`exp_20260927T215232Z`** | `exp_20260927T205741Z` | **Primary-30** | **633** | **51** | **175** | **407** | **64.3%** |
| **`exp_20260927T223303Z`** | `exp_20260927T221621Z` | **Holdout-30** | **515** | **33** | **108** | **374** | **72.6%** |

Four-label breakdown (same runs):

| Split | Strong | Partial | Weak | Missing |
| --- | ---: | ---: | ---: | ---: |
| Primary-30 | 51 | 175 | 245 | 162 |
| Holdout-30 | 33 | 108 | 206 | 168 |

### 5.4 Gap-sensitivity analysis (local, zero LLM cost)

| Experiment ID | Split | Source extraction | Stored gap (regression target) |
| --- | --- | --- | --- |
| `exp_20261003T042547Z` | Primary-30 | `exp_20260927T205741Z` | `exp_20260927T215232Z` |
| `exp_20261003T042642Z` | Holdout-30 | `exp_20260927T221621Z` | `exp_20260927T223303Z` |
| `exp_20261003T042837Z` | Primary smoke (2 pairs) | — | CLI smoke only |
| `gap_sensitivity_combined/` | both | — | Combined tables, audit, false-gap examples |

Embedding device for sensitivity runs: **CUDA**. Library versions logged in each `summary.json`.

### 5.5 Fit-label control (Luna extraction + local scoring)

| Experiment ID | Status | Notes |
| --- | --- | --- |
| **`exp_20261003T163033Z`** | completed | 98 Luna extractions; Condition A/D scoring; H1–H3 tests; figures 22–25 |
| `fit_label_control_combined/` | completed | Combined tables + frozen decision |

### 5.6 Gold-set (human requirement labels)

| Artifact | Status |
| --- | --- |
| `experiments/gold_set/` pack (seed `20261004`) | **Built** (150 + 10 calibration) |
| `experiments/gold_set/annotations/annotator_*.csv` | **Absent** — Phase 2 analysis refuses until real files exist |

### 5.7 Classifier reframing

| Artifact | Status |
| --- | --- |
| Protocol in `docs/classifier_reframing_report.md` | Frozen before the eval run |
| `experiments/results/classifier_reframing_combined/` | **Not yet written** (gap-lite features still computing) |

---

## 6. Extraction results (Luna)

Mean / median field counts per pair from stored extraction JSONs:

| Field | Primary-30 mean (median) | Holdout-30 mean (median) |
| --- | --- | --- |
| Competencies | 2.77 (0.5) | 2.57 (0.0) |
| Technical skills | 24.17 (21.0) | 24.40 (24.5) |
| Soft skills | 4.77 (4.0) | 5.17 (5.0) |
| Domain experience | 5.60 (5.0) | 5.10 (5.0) |
| Certifications | 0.43 (0.0) | 0.80 (0.0) |
| Experience responsibilities | 22.13 (22.0) | 22.33 (19.5) |
| Experience achievements | 4.90 (5.0) | 3.50 (4.0) |
| JD required skills | 16.67 (16.0) | 13.43 (12.5) |
| JD preferred skills | 4.43 (2.0) | 3.73 (2.0) |

**Observation:** competencies are sparse relative to technical skills and responsibility bullets. This motivated the evidence-set sensitivity analysis (Section 8).

Production gap scoring (`_candidate_texts`) uses a **deduplicated union** of technical/soft/domain skills, certifications, competencies, and experience technologies + responsibilities—not competencies alone.

---

## 7. Profile-gap matrix results

### 7.1 Primary-30 (`exp_20260927T215232Z`)

- Embedding: `all-MiniLM-L6-v2`; signals: TF-IDF + BM25 + semantic enabled  
- Mean gap-analysis latency: **62.2 ms**/pair  
- **633** requirements → strength 8.1% · partial 27.6% · **gap 64.3%**

### 7.2 Holdout-30 (`exp_20260927T223303Z`)

- Mean latency: **54.7 ms**/pair  
- **515** requirements → strength 6.4% · partial 21.0% · **gap 72.6%**

### 7.3 Breakdown under production evidence (from sensitivity recompute)

| Split | Slice | N req. | Gap share |
| --- | --- | ---: | ---: |
| Primary | Good Fit | 341 | 66.3% |
| Primary | Potential Fit | 292 | 62.0% |
| Primary | required | 500 | 63.4% |
| Primary | preferred | 133 | 67.7% |
| Holdout | Good Fit | 277 | 73.3% |
| Holdout | Potential Fit | 238 | 71.8% |
| Holdout | required | 403 | 69.0% |
| Holdout | preferred | 112 | 85.7% |

Gap rates remain high even on fit-positive pairs, motivating checks for evidence-set / extraction artefacts.

---

## 8. Evidence-set sensitivity analysis

Full methodological detail: [gap_sensitivity_report.md](gap_sensitivity_report.md).  
Combined tables: `experiments/results/gap_sensitivity_combined/`.

### 8.1 Design (held fixed vs varied)

| Held fixed | Varied |
| --- | --- |
| Job requirements | Candidate evidence string set |
| Weights 0.25 / 0.25 / 0.50 | — |
| Thresholds 0.65 / 0.40 / 0.20 | — |
| Embedding model | — |

| Condition | Evidence |
| --- | --- |
| **A** | Production `_candidate_texts` (regression baseline) |
| **A0** | Competencies only |
| **B** | Competencies + technical skills |
| **C** | B + experience title / responsibilities / achievements |
| **D** | Raw CV sentence/bullet chunks (no extraction) |
| **E** | C ∪ D |

**Controls / stats:** mismatched-candidate evidence (5 permutations, seed 42); bootstrap 95% CIs (2000 resamples over pairs); Wilcoxon signed-rank vs A; Spearman size↔gap dose-response; semantic-only robustness.

### 8.2 Regression check

Replaying Condition A against stored gap JSONs:

| Split | Passed | Counts match | Max \|Δ combined\| |
| --- | --- | --- | ---: |
| Primary-30 | Yes | Yes (51/175/245/162) | &lt; 1e-7 |
| Holdout-30 | Yes | Yes (33/108/206/168) | &lt; 1e-7 |

### 8.3 Gap shares and per-pair rates

| Split | Condition | N | Strong | Partial | Weak | Missing | Gap share [95% CI] | Mean per-pair gap [95% CI] | Evid. size mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| primary | A | 633 | 51 | 175 | 245 | 162 | 64.3% [58.1, 70.2] | 0.617 [0.552, 0.679] | 67.07 |
| primary | A0 | 633 | 3 | 34 | 35 | 561 | 94.2% [89.2, 97.8] | 0.925 [0.868, 0.973] | 2.77 |
| primary | B | 633 | 55 | 121 | 181 | 276 | 72.2% [66.4, 77.6] | 0.708 [0.640, 0.773] | 26.77 |
| primary | C | 633 | 43 | 157 | 259 | 174 | 68.4% [62.7, 73.4] | 0.653 [0.587, 0.717] | 57.13 |
| primary | D | 633 | 0 | 106 | 275 | 252 | 83.3% [78.2, 87.5] | 0.797 [0.749, 0.842] | 19.27 |
| primary | E | 633 | 32 | 169 | 291 | 141 | 68.2% [61.8, 74.1] | 0.657 [0.582, 0.724] | 76.40 |
| holdout | A | 515 | 33 | 108 | 206 | 168 | 72.6% [66.6, 77.9] | 0.698 [0.639, 0.754] | 66.77 |
| holdout | A0 | 515 | 2 | 14 | 32 | 467 | 96.9% [94.5, 98.9] | 0.975 [0.953, 0.991] | 2.57 |
| holdout | B | 515 | 32 | 74 | 138 | 271 | 79.4% [74.0, 84.1] | 0.772 [0.725, 0.820] | 26.97 |
| holdout | C | 515 | 26 | 102 | 197 | 190 | 75.1% [69.4, 80.3] | 0.720 [0.660, 0.776] | 56.23 |
| holdout | D | 515 | 1 | 70 | 224 | 220 | 86.2% [83.1, 89.0] | 0.828 [0.766, 0.877] | 19.47 |
| holdout | E | 515 | 26 | 110 | 224 | 155 | 73.6% [68.9, 78.1] | 0.709 [0.652, 0.763] | 75.70 |

### 8.4 Size-adjusted effects (true − mismatched gap share)

| Condition | Primary Δ | Holdout Δ |
| --- | ---: | ---: |
| A | −0.138 | −0.071 |
| A0 | −0.021 | −0.001 |
| B | −0.141 | −0.065 |
| C | −0.133 | −0.061 |
| D | −0.063 | −0.026 |
| E | −0.119 | −0.044 |

Negative values mean true candidate evidence yields fewer gaps than a random other candidate’s evidence of the same type (matching signal beyond set size).

### 8.5 Wilcoxon vs production A (per-pair gap rate)

Positive rank-biserial ⇒ condition has **higher** gap rate than A.

| Condition | Primary p / \(r_{rb}\) | Holdout p / \(r_{rb}\) |
| --- | --- | --- |
| A0 | ~0 / 1.00 | ~0 / 0.995 |
| B | 0.0013 / 0.84 | 0.0012 / 0.80 |
| C | 0.021 / 0.61 | 0.062 / 0.64 |
| D | ~0 / 1.00 | 0.0002 / 0.83 |
| E | 0.038 / 0.51 | 0.62 / 0.15 |

### 8.6 Spearman dose-response (evidence size vs per-pair gap rate)

| Condition | Primary ρ (p) | Holdout ρ (p) |
| --- | --- | --- |
| A | −0.057 (0.76) | −0.303 (0.10) |
| A0 | **−0.780 (~0)** | **−0.574 (0.0009)** |
| B | 0.111 (0.56) | 0.160 (0.40) |
| C | −0.111 (0.56) | −0.412 (0.024) |
| D | −0.139 (0.46) | −0.117 (0.54) |
| E | −0.151 (0.42) | −0.402 (0.028) |

Strong size↔gap coupling appears mainly under the sparse A0 regime.

### 8.7 Semantic-only robustness (same thresholds on semantic score)

| Condition | Primary gap share | Holdout gap share |
| --- | ---: | ---: |
| A | 38.2% | 45.2% |
| A0 | 90.7% | 93.0% |
| B | 52.8% | 59.4% |
| C | 43.3% | 51.1% |
| D | 71.7% | 73.2% |
| E | 41.1% | 48.5% |

Ordering is consistent with the combined scorer: A0 and D remain worse than production A.

### 8.8 Mean signal values (combined scorer)

| Split | Condition | TF-IDF | BM25_norm | Semantic | Combined |
| --- | --- | ---: | ---: | ---: | ---: |
| primary | A | 0.060 | 0.327 | 0.512 | 0.353 |
| primary | A0 | 0.012 | 0.055 | 0.143 | 0.088 |
| primary | D | 0.053 | 0.316 | 0.335 | 0.260 |
| holdout | A | 0.042 | 0.289 | 0.470 | 0.318 |
| holdout | A0 | 0.006 | 0.040 | 0.130 | 0.076 |
| holdout | D | 0.035 | 0.308 | 0.330 | 0.251 |

### 8.9 Label transitions A → D

**Primary-30**

| from \ to | strong | partial | weak | missing |
| --- | ---: | ---: | ---: | ---: |
| strong | 0 | 19 | 27 | 5 |
| partial | 0 | 63 | 96 | 16 |
| weak | 0 | 19 | 122 | 104 |
| missing | 0 | 5 | 30 | 127 |

**Holdout-30**

| from \ to | strong | partial | weak | missing |
| --- | ---: | ---: | ---: | ---: |
| strong | 1 | 14 | 18 | 0 |
| partial | 0 | 36 | 64 | 8 |
| weak | 0 | 18 | 114 | 74 |
| missing | 0 | 2 | 28 | 138 |

### 8.10 Figures

All figures at 200 dpi under `docs/figures/`.

#### Figure 17 — Gap-category shares by condition

![Figure 17: Gap-category shares by evidence condition (primary vs holdout)](figures/fig17_gap_category_shares.png)

#### Figure 18 — Per-pair gap-rate distributions

![Figure 18: Per-pair gap-rate distributions by condition with bootstrap CI of the mean](figures/fig18_per_pair_gap_rate.png)

#### Figure 19 — True vs mismatched-control gap rate

![Figure 19: True vs mismatched-control gap rate (size-adjusted effect)](figures/fig19_true_vs_mismatched.png)

#### Figure 20 — Evidence-set size vs per-pair gap rate (A and D)

![Figure 20: Evidence-set size vs per-pair gap rate for conditions A and D](figures/fig20_size_vs_gap_rate.png)

#### Figure 21 — A→D label-transition heatmap

![Figure 21: Label-transition heatmap from Condition A to Condition D](figures/fig21_A_to_D_transition.png)

### 8.11 Audit artifacts

| Artifact | Path |
| --- | --- |
| Annotator audit sheet (CSV) | `experiments/results/gap_sensitivity_combined/audit_sheet.csv` |
| Annotator audit sheet (MD) | `experiments/results/gap_sensitivity_combined/audit_sheet.md` |
| Top-25 A→D score-increase examples | `experiments/results/gap_sensitivity_combined/false_gap_examples.md` |
| Cross-split overview CSV | `experiments/results/gap_sensitivity_combined/overview_by_condition.csv` |

Audit annotator columns (`missed_skills_in_cv`, `extraction_recall_estimate`, `notes`) are intentionally blank.

### 8.12 Interpretation for the thesis

1. **Production gap rates (64–73%) already use expanded extracted evidence** (~67 strings/pair). They are not the competencies-only artefact.
2. **Competencies-only (A0) would inflate gaps to 94–97%.** The A0→A reduction (~0.30 primary / ~0.24 holdout) exceeds the mismatched-control change (~0.18 / ~0.17) ⇒ that inflation is partly extraction-coverage, not only size bias.
3. **Raw CV chunks (D) do not reduce gaps vs production A** (gaps rise to 83–86%; mean evidence size ~19 vs ~67). Against raw-CV substitution, the production gap rate **stands**.
4. BM25/TF-IDF scales depend on the evidence set; semantic-only metrics and fixed-bm25 diagnostics mitigate but do not eliminate that confound.

---

## 9. Fit-label control (gap matrix vs Hugging Face labels)

Full write-up: [fit_label_control_report.md](fit_label_control_report.md).  
Combined tables: `experiments/results/fit_label_control_combined/`.

HF `No Fit` / `Potential Fit` / `Good Fit` labels are **independent of InterviewIQ scoring**. Frozen hypotheses H1–H3 and the coarse decision rule were written **before** sampling and scoring.

### 9.1 Design (computed, not assumed)

- Fit-positive comparison sets: existing Primary-30 and Holdout-30.
- NF-primary-30 / NF-holdout-30: 30 No Fit pairs each (seeds 42 / 2026), disjoint from used fit-positive pair ids.
- Scoring: production Condition A; robustness Condition D and semantic-only; mismatched-candidate control (5 permutations, seed 42).
- Extraction: `gpt-5.6-luna`, prompt 1.0.0 / 1.0.0, spend **$0.673245**, 98/98 pairs, 0 failures (`exp_20261003T163033Z`).
- Primary/holdout Condition A **count regression vs stored gap JSONs: passed**.

### 9.2 Frozen decision (apply literally)

No Fit vs Good Fit was **not** significant with AUC ≥ 0.65 on **both** splits:

| Split | Holm *p* (No Fit vs Good) | AUC (A, No Fit vs Good) [95% CI] | Coarse significant |
| --- | ---: | --- | --- |
| Primary | 0.329 | 0.621 [0.443, 0.791] | no |
| Holdout | 1.0 | 0.501 [0.321, 0.692] | no |

**Outcome (verbatim rule):** the gap matrix does not separate fit labels; possible reasons include label noise, extraction, or scoring; **do not claim validation**. H1 monotonicity is **not** claimed (`h1_monotonic_allowed_both_splits`: false).

Adjacent Potential vs Good is null on both splits (as H3 allowed). Primary No Fit vs Potential Holm *p* = 0.031 is **not** the pre-registered coarse claim.

### 9.3 Figures 22–25

![Figure 22: Gap rate by fit label](figures/fig22_gap_rate_by_fit_label.png)

![Figure 23: ROC for fit labels](figures/fig23_roc_fit_labels.png)

![Figure 24: Stacked shares by fit label](figures/fig24_stacked_shares_by_fit_label.png)

![Figure 25: Confound panel](figures/fig25_confound_panel.png)

---

## 10. Gold set for the gap matrix (Phase 1)

Protocol (frozen before sampling): [gold_set_protocol.md](gold_set_protocol.md).  
Guidelines: [gold_set_guidelines.md](gold_set_guidelines.md).

The four-way system labels have so far been checked mainly against the system’s own combined score. The gold set is **requirement × whole-CV** human judgement (2 / 1 / 0), independent of system scores.

### 10.1 Pack (built; seed `20261004`)

| Item | Count |
| --- | ---: |
| Main items | 150 (75 Primary-30 + 75 Holdout-30) |
| Stratum targets per split | 19 strong / 19 partial / 19 weak / 18 missing (achieved) |
| Distinct pairs | Primary 30; Holdout **28** (≥ 25 required; max 3 requirements/pair) |
| Fit-label mix among selected pairs | Primary 15 Good + 15 Potential; Holdout 14 + 14 |
| Calibration items | 10 (same pairs, disjoint from the 150) |

Inclusion probability π = n/N per four-way stratum is stored on each main item (design weight N/n). Holdout raw CV/JD files are read from `experiment_pairs_150` (`experiment_pairs_holdout30/profiles` is empty on disk).

**Blind pack** (`items_blind.json`, `annotator_app.html`) contains requirement text, full CV, collapsible JD, anonymous ids, and calibration flags only. System labels, scores, split names, and fit labels live only in `answer_key_do_not_open_before_annotating.json`.

### 10.2 Phase 1 stop

Place completed files at:

- `experiments/gold_set/annotations/annotator_A.csv`
- `experiments/gold_set/annotations/annotator_B.csv`
- optional `adjudicated.csv`

Columns: `item_id`, `label`, `cannot_judge`, `snippet`, `notes`. **Do not run** `gold_set_analyze` until those files exist. **No human labels have been created in this repository.** Phase 2 (agreement, system-vs-human, signal AUCs, false gaps, G4 recalibration, figures 26–29) is unimplemented as a delivered result until real CSVs are present.

Single-annotator fallback: `annotator_A.csv` plus a 30-item re-annotation after ≥ 7 days (test–retest kappa).

---

## 11. Classifier reframing (in progress)

Protocol (frozen before the eval run): [classifier_reframing_report.md](classifier_reframing_report.md).

Goal: report the original classifier honestly, test modest variants (V0–V6) with **group-aware CV on train only**, and test whether LLM-free **gap-lite** features (JD chunks vs Condition-D CV chunks; production hybrid weights/thresholds) predict HF fit labels. Gap-lite is **not** the production Luna-extracted matrix and must not be described as validating it.

A variant “improves over V0” only if selected on validation **and** the paired-bootstrap 95% CI of test macro-F1 gain excludes 0.

**As of this report date:** the original V0 metrics in Section 4.2 still stand. The full MiniLM evaluation is computing cached chunk features and then gap-lite on 6,241 train + 1,759 test pairs. Headline tables, leakage-audit counts, figures 30–34, and the literal improved-over-V0 outcomes will be appended to the classifier report when that run writes `experiments/results/classifier_reframing_combined/`. Those numbers are **null here**.

---

## 12. Baseline generation results (Terra)

Controlled baseline: flat prompt, **no** gap matrix / competency budget / MMR. Requested \(N = 10\) questions.

| Run | Pairs | Fit mix | Q/pair | Total cost | Mean latency |
| --- | ---: | --- | ---: | ---: | ---: |
| `exp_20260920T220958Z` | 30 | 15 Good + 15 Potential | 10 | $0.247 | 5.77 s |
| `exp_20260926T221259Z` | 30 | 15 + 15 | 10 | $0.246 | 5.74 s |
| `exp_20260926T223339Z` | 150 | 75 + 75 | 10 | $1.148 | 5.46 s |

Human evaluation of baseline vs InterviewIQ question quality is **not yet completed**.

The legacy `interviewiq_baseline/` package remains as a historical `gpt-4o` prototype; thesis comparisons must use the **same** generation model (Terra) for both conditions.

---

## 13. Application surface

### 10.1 Backend API

- `GET /api/v1/health` — health  
- Pipeline / gap endpoints under `app/api/v1/pipeline.py`  
- Experiment listing / detail under `app/api/v1/experiments.py`  
- System / model routing info under `app/api/v1/system.py`

### 10.2 Frontend research console

- `/` — experiment list + aggregate cost/latency  
- `/experiments/[id]` — latency, cost, token, and fit-label charts; question inspector  
- `/upload` — ad-hoc CV/JD storage for testing  

API base: `NEXT_PUBLIC_API_URL` (default `http://127.0.0.1:8000`).

### 10.3 Experiment runner stages

```bash
# Baseline
PYTHONPATH=backend python -m experiments.run --config experiments/configs/primary_hf.yaml

# Extraction
PYTHONPATH=backend python -m experiments.run --config experiments/configs/primary_hf.yaml --stage extraction

# Gap analysis
PYTHONPATH=backend python -m experiments.run --config experiments/configs/primary_hf.yaml \
  --stage gap_analysis --source-experiment exp_20260927T205741Z

# Gap sensitivity (both splits + figures + report)
PYTHONPATH=backend python experiments/scripts/run_gap_sensitivity.py

# Fit-label control (paid Luna; requires --confirm-spend)
PYTHONPATH=backend python experiments/scripts/run_fit_label_control.py --dry-run
PYTHONPATH=backend python experiments/scripts/run_fit_label_control.py --confirm-spend --max-cost 1.00

# Gold-set pack (no LLM; then STOP for human annotation)
PYTHONPATH=backend python experiments/scripts/build_gold_set.py

# Gold-set analysis (refuses unless real annotation CSVs exist)
PYTHONPATH=backend python experiments/scripts/analyze_gold_set.py

# Classifier reframing (local embeddings only; no LLM)
PYTHONPATH=backend python experiments/scripts/run_classifier_eval.py
```

---

## 14. Testing

Tests mock LLM usage and do not call paid APIs. Prefer `EMBEDDING_MODEL=hashing-test` to avoid downloading Sentence-Transformer weights.

| Suite area | Coverage |
| --- | --- |
| Config / router | Model routing, fail-closed behaviour |
| Retrieval | TF-IDF, BM25 |
| Gap + MMR | Labels, weighting, redundancy, MMR selection |
| Extraction schemas | Pydantic profile parsing |
| Dataset / sources | Resume-fit loading, pair resolution |
| Baseline generation | Prompt path (mocked) |
| API / experiments API | Health and experiment readers |
| **Gap sensitivity** | Chunker, evidence builders, mismatch seeding, bootstrap reproducibility, 2-pair E2E |
| **Fit-label control** | NF sampling disjointness, spend guard, JT toy, synthetic end-to-end |
| **Gold set** | Deterministic stratified sampler, blind-field check, analysis refusal on missing/short/invalid files, kappa/bootstrap toys, weighted mean |
| **Classifier eval** | Connected-component groups + group-fold leakage, gap-lite on 2 synthetic pairs, paired-bootstrap gain, smoke pipeline on tiny hashing data |

As of 2026-10-04: **47 pytest tests passing** (`EMBEDDING_MODEL=hashing-test`).

---

## 15. Cost summary (completed paid LLM stages)

| Stage | Approx. total USD (sum of successful runs cited above) |
| --- | ---: |
| Baseline Terra (30+30+150+smoke) | ≈ 1.66 |
| Luna extraction (primary + holdout + smokes) | ≈ 0.63 |
| Luna extraction (fit-label NF + extras, `exp_20261003T163033Z`) | **0.673245** |
| Gap analysis | 0 (local embeddings only) |
| Gap sensitivity | 0 (local embeddings + IR only) |
| Fit-label scoring / gold-set pack / classifier eval | 0 LLM (local embeddings + sklearn) |

Exact per-run totals are recorded in each experiment’s `summary.json`.

---

## 16. Limitations and open work

### Limitations of completed work

- Gap labels depend on IR + MiniLM similarity, not human adjudication **until the gold set is annotated**.  
- BM25/TF-IDF normalisation is evidence-set-dependent (documented; semantic-only used as robustness check).  
- Hugging Face pair-level fit labels **do not** validate the requirement-level gap matrix under the frozen H2 coarse rule.  
- Local fit classifier accuracy is modest (~0.46); it remains exploratory unless a reframing variant meets the pre-registered rule.  
- Frontend charts visualise run metadata; they are not substitutes for the thesis figures in `docs/figures/`.  
- Full InterviewIQ **question generation** batch (Terra pool + MMR) vs baseline human eval is still outstanding.

### Recommended next steps

1. Annotate the gold set (`annotator_A.csv` / `annotator_B.csv`); then run `analyze_gold_set.py` (G1–G4, figures 26–29).  
2. Finish classifier reframing and report only computed test metrics plus the literal improved-over-V0 outcomes.  
3. Run InterviewIQ generation on the same Primary-30 / Holdout-30 pairs with Terra, logging pool/MMR artifacts.  
4. Blind human evaluation of Baseline vs InterviewIQ question sets.  
5. Optional: fill the gap-sensitivity audit sheet (20 pairs) for extraction-recall estimates.  
6. Optional: ablations of TF-IDF / BM25 / semantic flags under fixed evidence.

---

## 17. Key artifact index

| Artifact | Location |
| --- | --- |
| This report | `docs/project_report.md` |
| Architecture | `docs/architecture.md` |
| LLM routing | `docs/llm-routing.md` |
| Research pipeline | `docs/research-pipeline.md` |
| Gap sensitivity deep-dive | `docs/gap_sensitivity_report.md` |
| Fit-label control | `docs/fit_label_control_report.md` |
| Gold-set protocol / guidelines | `docs/gold_set_protocol.md`, `docs/gold_set_guidelines.md` |
| Classifier reframing protocol | `docs/classifier_reframing_report.md` |
| Figures 17–21 | `docs/figures/fig17_*.png` … `fig21_*.png` |
| Figures 22–25 | `docs/figures/fig22_*.png` … `fig25_*.png` |
| Fit classifier metrics (V0) | `experiments/datasets/resume_job_fit/models/fit_classifier_metrics.json` |
| Primary extraction | `experiments/results/exp_20260927T205741Z/` |
| Primary gap analysis | `experiments/results/exp_20260927T215232Z/` |
| Holdout extraction | `experiments/results/exp_20260927T221621Z/` |
| Holdout gap analysis | `experiments/results/exp_20260927T223303Z/` |
| Gap sensitivity (primary) | `experiments/results/exp_20261003T042547Z/` |
| Gap sensitivity (holdout) | `experiments/results/exp_20261003T042642Z/` |
| Combined sensitivity tables | `experiments/results/gap_sensitivity_combined/` |
| Fit-label control run | `experiments/results/exp_20261003T163033Z/` |
| Fit-label combined tables | `experiments/results/fit_label_control_combined/` |
| Gold-set annotation pack | `experiments/gold_set/` |
| Primary-30 Terra baseline | `experiments/results/exp_20260920T220958Z/` |
| 150-pair Terra baseline | `experiments/results/exp_20260926T223339Z/` |

---

## 18. Conclusion

InterviewIQ’s deterministic gap-analysis stack is implemented, tested, and exercised on two 30-pair fit-positive splits. Production scoring yields **64.3%** (primary) and **72.6%** (holdout) gap shares over job requirements. A local sensitivity study shows that (i) those rates already reflect expanded extracted evidence, not competencies-only sparsity; (ii) competencies-only would severely inflate gaps beyond a size-bias control; and (iii) raw CV chunking does not improve on production evidence for this scorer. Against **independent pair-level fit labels**, the frozen H2 rule is **not** met: do not claim that the gap matrix tracks HF No Fit / Good Fit. A 160-item **human gold set** (requirement vs whole CV) is packed and waiting for annotation; analysis must not proceed without those files. The original local fit classifier remains a weak auxiliary model (macro-F1 0.424); reframing variants are being evaluated under a frozen protocol and are not yet reported. Terra baseline question sets exist at 30- and 150-pair scale. The remaining critical path for the thesis **generation** comparison is **InterviewIQ generation + human evaluation** against those baselines under a matched Terra configuration.
