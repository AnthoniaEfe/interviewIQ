# Experiments

```bash
python -m experiments.run --config experiments/configs/main.yaml
```

The runner currently writes a timestamped `experiments/results/<id>/summary.json` with `status: not_yet_evaluated`. Full batch execution (baseline + InterviewIQ on the dataset) is not implemented yet.

Dataset size (for example 15 profiles × 3 roles × 3 seniorities) belongs in YAML/data files, not in application constants.

## Config keys

See `experiments/configs/main.yaml` for pool size, MMR lambda, similarity threshold, ablation flags, and model-control notes.

## Legacy baseline

`interviewiq_baseline/run_batch_baseline.py` still enumerates seniority in-process and defaults to GPT-4o. Keep it for smoke-testing the original script. The controlled thesis run will go through `experiments/run` once generation is wired.
