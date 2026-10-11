"""Pinned assets and generation parameters for the treatment pilot."""

from __future__ import annotations

EXTRACTION_EXPERIMENT = "exp_20260927T205741Z"
GAP_EXPERIMENT = "exp_20260927T215232Z"

B0_EXPERIMENTS = (
    "exp_20260920T220958Z",
    "exp_20260926T221259Z",
    "exp_20260926T223339Z",
)

MANIFEST_REL = "experiments/datasets/resume_job_fit/experiment_pairs/manifest.json"
PAIRS_DIR_REL = "experiments/datasets/resume_job_fit/experiment_pairs"
CONFIG_REL = "experiments/configs/primary_hf.yaml"

GENERATION_MODEL = "gpt-5.6-terra"
REASONING_EFFORT = "medium"
MAX_OUTPUT_TOKENS = 12000
TEMPERATURE_SETTING = None  # omitted for gpt-5* reasoning models

USD_CNY_RATE = 6.77
DEFAULT_MAX_COST_USD = 4.00

SEED = 42
BOOTSTRAP_RESAMPLES = 2000
RANDOM_SELECTION_REPEATS = 5

# Dry-run priors from 150-pair B0 run (exp_20260926T223339Z).
B0_MEAN_INPUT_TOKENS = 1804.27
B0_MEAN_OUTPUT_TOKENS = 383.05
B0_MEAN_COST_USD = 0.00765
B0_MEAN_LATENCY_S = 5.46

# Multipliers for dry-run estimates (explicit assumptions).
B1_INPUT_MULT = 1.4  # structured profiles vs raw text
B1_OUTPUT_MULT = 1.0  # still 10 questions
POOL_INPUT_MULT = 3.5  # profiles + gap matrix
POOL_OUTPUT_MULT = 4.0  # ~30 structured items vs 10 strings

# Stored extraction / gap attribution (primary-30 measured).
EXTRACTION_COST_PER_PAIR_USD = 0.00711369
EXTRACTION_LATENCY_PER_PAIR_S = 31.259867
GAP_LATENCY_PER_PAIR_S = 0.062179
GAP_COST_PER_PAIR_USD = 0.0

SENIORITY = "mid"
FINAL_N = 10
POOL_N = 30
MMR_LAMBDA = 0.7
SIM_THRESHOLD = 0.85

COVERAGE_THRESHOLDS = (0.35, 0.45, 0.55)
PRIMARY_COVERAGE_THRESHOLD = 0.45

COMBINED_DIR_NAME = "treatment_pilot_combined"
