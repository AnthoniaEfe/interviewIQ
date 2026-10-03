# Classifier reframing: protocol

The local resume–job fit classifier is an **auxiliary calibration helper**. It carries
no thesis claim unless it earns one under the rules below.

**This protocol was written before any retraining, leakage audit, or variant
evaluation. It must not be edited after that point.**

---

## Pre-registered rules

*(Frozen. Do not edit.)*

- All model/hyperparameter selection uses **group-aware cross-validation on the TRAIN
  split only**. The original TEST split is evaluated **once per final variant** and is
  never used for tuning. Test labels are never altered.
- Primary selection metrics: **macro-F1** (3-class) and **balanced accuracy** (binary).
- A variant counts as **“improved over V0”** only if it was selected on validation
  **AND** the paired-bootstrap 95% CI (2000 resamples, fixed seed) of its macro-F1
  gain over V0 on the test split **excludes 0**. Holm correction is applied across
  variants for any p-values reported.
- If no variant clearly beats V0, say so: the classifier stays exploratory.

### Variants (as specified)

- V0: existing classifier (reference).
- V1: V0 features, `class_weight="balanced"`.
- V2: binary No Fit vs fit-positive, balanced LR; also 3-class collapse of V0/V1.
- V3: chunked embeddings (mean and max pool), same head as V1.
- V4: gap-lite features only (LR balanced and HistGradientBoosting).
- V5: best of V1/V3 + gap-lite (LR and HistGradientBoosting).
- V6: stacked ordinal binaries on the best feature set.

Gap-lite is **not identical to the production matrix**. Features that predict fit
labels give **indirect, independent support for the gap-matrix idea only**. Do **not**
describe gap-lite as validation of the production LLM-extracted matrix.

---

## Code inspection

*(Filled after reading training code; before the evaluation run.)*

### Training and evaluation code

`experiments/scripts/train_fit_classifier.py` trains on Hugging Face
`cnamuangtoun/resume-job-description-fit` **train** (6,241) and evaluates **test**
(1,759). Features: concatenate `[resume_emb, job_emb, |diff|, product]` from
`all-MiniLM-L6-v2` (`_pair_features`). Head: `StandardScaler` + `LogisticRegression`
(`max_iter=2000`, `class_weight="balanced"`, `random_state=seed` default 42).
Labels: No Fit=0, Potential Fit=1, Good Fit=2 (`LABELS` in
`backend/app/services/datasets/resume_fit.py`).

Saved metrics: `experiments/datasets/resume_job_fit/models/fit_classifier_metrics.json`
(accuracy 0.464, macro-F1 0.424, weighted-F1 0.462). The `.joblib` artifact is
listed in `.gitignore` comments and may be absent; V0 is **reproduced by retraining
the same pipeline**, not by mutating the original training script’s defaults. New
models are written under a **new** directory
(`experiments/datasets/resume_job_fit/models_reframing/`).

Embedding encode path: `SentenceTransformerEmbeddingService.embed_batch` uses
`SentenceTransformer.encode` (default truncation at the model max sequence length,
typically 256 word pieces for MiniLM).

Feature cache path used by the trainer: `experiments/datasets/resume_job_fit/cache/features/`.
