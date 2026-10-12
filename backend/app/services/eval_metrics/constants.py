"""Pinned constants for frozen evaluation metrics v1."""

from __future__ import annotations

SPEC_VERSION = "v1"
METRIC_SPEC_REL = "experiments/metrics/metrics_spec_v1.json"
METRICS_FREEZE_REL = "experiments/metrics/metrics_freeze.json"
METRICS_DIR_REL = "experiments/metrics"
COMBINED_DIR_NAME = "metrics_combined"
ASSIGNMENT_AUDIT_REL = "experiments/metrics/assignment_audit.csv"

# Stored baseline B0 experiments
B0_150_EXPERIMENT = "exp_20260926T223339Z"
B0_PRIMARY_REPLICATES = (
    "exp_20260920T220958Z",
    "exp_20260926T221259Z",
    "exp_20260926T223339Z",
)

EXTRACTION_PRIMARY = "exp_20260927T205741Z"
EXTRACTION_HOLDOUT = "exp_20260927T221621Z"
GAP_PRIMARY = "exp_20260927T215232Z"
GAP_HOLDOUT = "exp_20260927T223303Z"

MANIFEST_PRIMARY_REL = "experiments/datasets/resume_job_fit/experiment_pairs/manifest.json"
MANIFEST_HOLDOUT_REL = (
    "experiments/datasets/resume_job_fit/experiment_pairs_holdout30/manifest.json"
)
MANIFEST_150_REL = "experiments/datasets/resume_job_fit/experiment_pairs_150/manifest.json"
PAIRS_150_DIR_REL = "experiments/datasets/resume_job_fit/experiment_pairs_150"
CONFIG_150_REL = "experiments/configs/primary_hf_150.yaml"
CONFIG_PRIMARY_REL = "experiments/configs/primary_hf.yaml"

SEED = 42
BOOTSTRAP_RESAMPLES = 2000
NULL_PERMUTATIONS = 5
TAU_PERCENTILE = 95.0
NEAR_DUP_THRESHOLD = 0.85  # settings.question_similarity_threshold
NEAR_DUP_SENSITIVITY = 0.75
FIXED_ASSIGNMENT_THRESHOLDS = (0.35, 0.45, 0.55)
FINAL_N = 10
POOL_N = 30
MMR_LAMBDAS = (0.3, 0.5, 0.7, 0.9, 1.0)
POOL_SIZE_KS = (10, 20, 30)
AUDIT_SAMPLE_N = 40
DUPLICATE_INJECT_N = 3

USD_CNY_RATE = 6.77
DEFAULT_MAX_COST_USD = 2.00
DEFAULT_SCALE_BUDGET_CNY = 100.0
EXTRACTION_PRIOR_PER_PAIR_USD = 0.00711369

GAP_LABELS = frozenset({"weak_match", "missing"})
DIMENSIONS = (
    "Technical",
    "Behavioural",
    "Situational",
    "Cultural/Motivational",
    "Role-Specific",
)

PRIMARY_METRICS = ("M1.P", "M2.P", "M3.P", "M4.P")
# Favourable direction: +1 higher-is-better, -1 lower-is-better
FAVOURABLE_SIGN = {
    "M1.P": -1,
    "M2.P": 1,
    "M3.P": 1,
    "M4.P": 1,
}

PILOT_STAGES = frozenset(
    {"treatment_pipeline", "baseline_profile_aware", "selection_ablation"}
)

DIMENSION_PROMPT_REL = "prompts/analysis/dimension_label_v1.txt"

MODULE_HASH_GLOBS = (
    "backend/app/services/eval_metrics/*.py",
    "prompts/analysis/dimension_label_v1.txt",
)
