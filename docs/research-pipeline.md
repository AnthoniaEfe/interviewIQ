# Research pipeline

Independent variable for the main experiment: **generation methodology**.

| Mode | Pipeline |
| --- | --- |
| Baseline | CV + JD (+ seniority) → Terra → N questions. No gap matrix, no competency weights, no MMR. |
| InterviewIQ | extract → normalize → TF-IDF/BM25/semantic → gap matrix → weights → Terra pool → redundancy → competency-weighted MMR → N questions |

Held constant: candidate, job, generation model, temperature/reasoning config, final N, evaluation criteria.

Do not compare Luna baseline against Terra InterviewIQ.

## Traceability

Each stage writes inspectable artifacts (planned under `experiments/results/<experiment_id>/`):

- extracted profiles
- similarity matrices
- gap matrix
- weights
- question pool
- question-question similarity
- MMR intermediate scores
- usage/cost/latency
- prompt versions

## Metrics

Human evaluation uses structured forms. Automatic metrics (diversity, redundancy, coverage, cost, latency) are computed in code. Unavailable metrics are returned as null. Results are never fabricated.
