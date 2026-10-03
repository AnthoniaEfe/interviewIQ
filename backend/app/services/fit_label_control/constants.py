"""Fixed seeds, paths, and documented prior extraction cost."""

from __future__ import annotations

# Primary-30 Luna extraction `exp_20260927T205741Z` summary.json:
# total_estimated_cost_usd=0.2134107 over 30 pairs.
PRIOR_EXTRACTION_TOTAL_USD = 0.2134107
PRIOR_EXTRACTION_PAIRS = 30
PRIOR_PER_PAIR_USD = PRIOR_EXTRACTION_TOTAL_USD / PRIOR_EXTRACTION_PAIRS

SEED = 42
HOLDOUT_SEED = 2026
BOOTSTRAP_RESAMPLES = 2000
MISMATCH_PERMS = 5
JT_PERMUTATIONS = 10000
SMOKE_PAIRS = 2
N_NO_FIT = 30
DEFAULT_MAX_COST_USD = 1.00
SENIORITY = "mid"

HF_DATASET = "cnamuangtoun/resume-job-description-fit"
SOURCE_SPLIT = "train"
PROMPT_VERSIONS = {
    "candidate_profile": "1.0.0",
    "job_description": "1.0.0",
}
EXTRACTION_MODEL = "gpt-5.6-luna"

LABEL_ORDER = ("Good Fit", "Potential Fit", "No Fit")
ORDINAL = {"Good Fit": 0, "Potential Fit": 1, "No Fit": 2}

USED_MANIFESTS = (
    "experiments/datasets/resume_job_fit/experiment_pairs/manifest.json",
    "experiments/datasets/resume_job_fit/experiment_pairs_holdout30/manifest.json",
    "experiments/datasets/resume_job_fit/experiment_pairs_150/manifest.json",
)

FITPOS_ASSETS = {
    "primary": {
        "split": "primary-30",
        "seed": SEED,
        "extraction": "exp_20260927T205741Z",
        "stored_gap": "exp_20260927T215232Z",
        "pairs_dir": "experiments/datasets/resume_job_fit/experiment_pairs",
        "manifest": "experiments/datasets/resume_job_fit/experiment_pairs/manifest.json",
        "config": "experiments/configs/primary_hf.yaml",
    },
    "holdout": {
        "split": "holdout-30",
        "seed": HOLDOUT_SEED,
        "extraction": "exp_20260927T221621Z",
        "stored_gap": "exp_20260927T223303Z",
        "pairs_dir": "experiments/datasets/resume_job_fit/experiment_pairs_150",
        "manifest": "experiments/datasets/resume_job_fit/experiment_pairs_holdout30/manifest.json",
        "config": "experiments/configs/primary_hf_holdout30.yaml",
    },
}

NF_SUBSET_DIRS = {
    "primary": "experiments/datasets/resume_job_fit/experiment_pairs_nf_primary30",
    "holdout": "experiments/datasets/resume_job_fit/experiment_pairs_nf_holdout30",
}

PRIORITY_WEIGHT = {
    "required": 1.0,
    "preferred": 0.6,
    "unspecified": 0.8,
}
