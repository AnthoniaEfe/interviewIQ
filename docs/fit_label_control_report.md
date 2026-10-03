# Fit-label control: gap matrix vs independent human fit labels

Analysis of whether the profile-gap matrix tracks the Hugging Face dataset
`cnamuangtoun/resume-job-description-fit` labels (No Fit / Potential Fit / Good Fit).
Those labels are assigned independently of InterviewIQ scoring (they are not derived
from TF-IDF, BM25, or semantic similarity in this codebase).

**This hypotheses section was written before any sampling, extraction, scoring, or
statistical run for this analysis. It must not be edited after that point.**

---

## Pre-registered hypotheses and decision rules

*(Frozen. Do not edit.)*

### H1 (gradient)

Per-pair gap rate is ordered **No Fit > Potential Fit > Good Fit**. Equivalently,
strength share and mean combined score are ordered the other way
(Good Fit > Potential Fit > No Fit).

### H2 (negative control)

Per-pair gap rate (and mean combined score) separates **No Fit** from **Good Fit**;
ROC AUC > 0.5. The same holds for No Fit vs fit-positive (Good + Potential) pairs.

### H3 (adjacent classes)

Potential Fit differs from Good Fit. This contrast is expected to be weak; the
dataset’s labels overlap heavily. A null result here is **not** treated as failure
of the method.

### Decision rules (apply literally)

- If No Fit vs Good Fit is significant (Holm-adjusted p < 0.05) with AUC ≥ 0.65
  in **BOTH** splits → “the gap matrix tracks human fit labels at the coarse level”.
- If significant in **one split only** → “partial/inconsistent support”.
- If **neither** → “the gap matrix does not separate fit labels; possible reasons:
  label noise, extraction, scoring; do not claim validation”. Report it as such.
- **Never claim H1 monotonic** unless the Jonckheere-Terpstra trend test **AND**
  both adjacent differences point the right way.

---

## Code inspection

### Fit labels on pair subsets

Subset manifests (`experiment_pairs/manifest.json`, `experiment_pairs_holdout30/manifest.json`, `experiment_pairs_150/manifest.json`) store `pair_id`, `profile_id`, `jd_id`, `label` (`Good Fit` / `Potential Fit`), `label_id` (Good=2, Potential=1, No Fit=0 in `LABEL_TO_ID`), and relative `profile_path` / `job_path`. Extraction and gap JSONs copy `fit_label` from the runner combo (manifest `label`). Sampling is `sample_experiment_pairs` in `backend/app/services/datasets/resume_fit.py`: pandas `DataFrame.sample(n=…, random_state=seed)` per requested label on the **train** split, then a global shuffle with the same seed. Primary-30 is seed 42, 15 Good + 15 Potential (reconstructed exactly). Primary-150 is seed 42, 75 per fit-positive label (reconstructed exactly). Holdout-30 is a 15+15 subset of the 150-pool disjoint from primary-30; config seed is 2026, but that draw is **not** reproduced by `sample_experiment_pairs(..., seed=2026)` on the remaining 150 rows (documented as a deviation for NF-holdout, which uses seed 2026 on remaining No Fit train rows).

### Mapping pairs to extraction / gap JSON

Filenames: `{profile_id}__{jd_id}__{seniority}.json` with seniority `mid`. Extraction lives in `experiments/results/<exp>/extraction/`; production gap in `…/gap_analysis/`. Primary extraction `exp_20260927T205741Z`, gap `exp_20260927T215232Z`. Holdout extraction `exp_20260927T221621Z`, gap `exp_20260927T223303Z`. `pair_id` equals `profile_id` / `jd_id` for this dataset (one resume and one JD file per pair).

### How gap and gap_sensitivity consume extraction

`experiments/run.py` `run_gap_combo` validates `candidate` / `job` with Pydantic and calls `build_gap_matrix`. `gap_sensitivity` loads the same extraction records, builds evidence via `build_evidence`, and scores with `score_requirements` (Condition A = `_candidate_texts`). Raw CV for Condition D is `pairs_dir/profiles/{profile_id}.txt`.

### Per-pair fields

- **JD / CV length:** not stored on extraction JSON; taken from the pair text files (`len` of UTF-8 text).
- **Requirements:** `job.required_skills` then `job.preferred_skills` (`_job_competencies`); count = `len(gap_matrix.gaps)`.
- **Priority:** `required` / `preferred` / `unspecified`; production `ProfileGap.weight` also folds match severity and evidence (not a pure priority weight).
- **Evidence-string count:** `len(_candidate_texts(candidate))` under Condition A (~67 on fit-positive runs, as previously reported).
- Stored gap JSON also has per-requirement TF-IDF / BM25 / semantic / combined label.

### Repeated job descriptions (matched-job feasibility)

Train split cache: 6241 rows, 280 unique JDs, 642 unique resumes. **243 / 280** jobs appear with more than one fit label; **243** have both No Fit and (Good or Potential); **115** have No Fit and Good Fit; **72** have all three labels. All **100** unique JDs in the used fit-positive files also appear as No Fit in train. Primary-30 has 30 unique JDs; holdout-30 has 28. Within those 60-pair samples, no JD appears with two fit labels *inside the sample*. **52** unique JDs from primary+holdout have 831 No Fit rows in train — extra extraction of one No Fit resume per overlapping job is estimated in the dry-run and included only if it fits `--max-cost`.

### Production scoring (unchanged)

Weights 0.25 TF-IDF + 0.25 BM25_norm + 0.50 semantic; strong ≥ 0.65, partial ≥ 0.40, weak ≥ 0.20, else missing; gap = weak + missing. Extraction model `gpt-5.6-luna`, prompt versions 1.0.0 / 1.0.0. Existing `experiments/results/exp_*` artifacts are not modified by this analysis.

<!-- GENERATED_BELOW -->

## Run status

Stopped after dry-run because `OPENAI_API_KEY` is not set. No Luna calls were made. Fit-positive Condition A regression was run locally (no LLM) and **passed** on both splits (primary 633: 51/175/245/162; holdout 515: 33/108/206/168). NF-primary-30 and NF-holdout-30 subset files are on disk. After the key is set, run:

```bash
PYTHONPATH=backend python experiments/scripts/run_fit_label_control.py --confirm-spend --max-cost 1.00
```

Do not paste the key into chat. Estimated cost for 98 uncached pairs (60 No Fit + 38 matched-job extras) is **0.69714162 USD** at the documented primary-30 mean of 0.00711369 USD/pair.

Main H1/H2/H3 tables, figures 22–25, AUCs, and the decision-rule outcome are **null until extraction completes**. They are not estimated.


