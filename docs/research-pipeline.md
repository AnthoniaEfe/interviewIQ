# Research pipeline

Independent variable for the main experiment: **generation methodology**.

| Mode | Pipeline |
| --- | --- |
| Baseline (B0) | CV + JD (+ seniority) → Terra → N questions. No gap matrix, no competency weights, no MMR. |
| Profile-aware baseline (B1) | Stored extraction JSON → Terra → N questions (no gap matrix / weights / pool). |
| InterviewIQ (T) | stored extract → stored gap matrix → weights → Terra pool → redundancy → competency-weighted MMR → N questions |
| Plain MMR (B2) | Same stored pool as T → MMR with relevance = priority (weights = 1). |

Held constant: candidate, job, generation model, temperature/reasoning config, final N, evaluation criteria.

Do not compare Luna baseline against Terra InterviewIQ.

Selection strategies are pluggable (`competency_weighted_mmr`, `plain_mmr_priority`, `plain_mmr_query`, `top_priority`, `random`). Coverage-dependent λ and competency budgets are out of scope for the v1 pilot.

## Traceability

Each stage writes inspectable artifacts under `experiments/results/<experiment_id>/`:

- extracted profiles
- similarity matrices
- gap matrix
- weights
- question pool (`pool/`)
- selected questions (`treatment/`, ablation condition dirs)
- question-question similarity / MMR intermediate scores
- usage/cost/latency and generation parameters
- prompt file names, versions, and SHA-256 hashes

Primary-30 treatment pilot protocol and results: `docs/treatment_pilot_report.md`.

## Metrics

Human evaluation uses structured forms. Automatic metrics (diversity, redundancy, coverage, cost, latency) are computed in code. Unavailable metrics are returned as null. Results are never fabricated.
