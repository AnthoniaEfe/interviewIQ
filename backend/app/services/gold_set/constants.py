"""Gold-set constants."""

from itertools import combinations

SEED = 20261004
# Staffing assignment only (who codes which frozen items). Does not resample items.
ASSIGNMENT_SEED = 20261112
# Review cap: five people, 30 CV/JD pairs each (10 shared calibration + 20 main).
ANNOTATORS = (
    "annotator_A",
    "annotator_B",
    "annotator_C",
    "annotator_D",
    "annotator_E",
)
PAIR_BLOCKS = tuple(combinations(ANNOTATORS, 2))
ITEMS_PER_ANNOTATOR = 30
CALIBRATION_N = 10
MAIN_PER_ANNOTATOR = ITEMS_PER_ANNOTATOR - CALIBRATION_N
# 5 annotators × 20 main items / 2 labels per item.
ASSIGNED_MAIN = len(ANNOTATORS) * MAIN_PER_ANNOTATOR // 2
# Drawn from the frozen 19/19/19/18 stratum sample. 25 items per split.
ASSIGNED_STRATUM_PER_SPLIT = {
    "strong_match": 7,
    "partial_match": 6,
    "weak_match": 6,
    "missing": 6,
}
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
