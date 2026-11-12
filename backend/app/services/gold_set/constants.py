"""Gold-set constants."""

SEED = 20261004
# Staffing assignment only (who codes which frozen items). Does not resample items.
ASSIGNMENT_SEED = 20261112
ANNOTATORS = ("annotator_A", "annotator_B", "annotator_C", "annotator_D")
PAIR_BLOCKS = (
    ("annotator_A", "annotator_B"),
    ("annotator_A", "annotator_C"),
    ("annotator_A", "annotator_D"),
    ("annotator_B", "annotator_C"),
    ("annotator_B", "annotator_D"),
    ("annotator_C", "annotator_D"),
)
CALIBRATION_N = 10
PER_SPLIT = 75
STRATUM_TARGETS = {
    "strong_match": 19,
    "partial_match": 19,
    "weak_match": 19,
    "missing": 18,
}
MAX_PER_PAIR = 3
MIN_PAIRS_PER_SPLIT = 25
BOOTSTRAP = 2000
STATS_SEED = 42

PRIMARY = {
    "name": "primary-30",
    "gap": "exp_20260927T215232Z",
    "pairs_dir": "experiments/datasets/resume_job_fit/experiment_pairs",
}
HOLDOUT = {
    "name": "holdout-30",
    "gap": "exp_20260927T223303Z",
    # Raw CV/JD for holdout-30 live in the seed-2026 150-pair export; all 30
    # gap artifacts resolve there (experiment_pairs_holdout30/profiles is empty).
    "pairs_dir": "experiments/datasets/resume_job_fit/experiment_pairs_150",
}
