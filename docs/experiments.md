# Experiments

Primary dataset: `cnamuangtoun/resume-job-description-fit` (pandas + `hf://` URLs).

## Prepare / train local fit model

```bash
PYTHONPATH=backend python experiments/scripts/train_fit_classifier.py
```

This trains a local embedding + logistic-regression classifier on the HF **train**
split and evaluates on **test**. It does **not** fine-tune Terra/Luna.

## Run baseline generation on curated HF pairs

```bash
PYTHONPATH=backend python -m experiments.run \
  --config experiments/configs/primary_hf.yaml --smoke
```

## Legacy smoke files

```bash
PYTHONPATH=backend python -m experiments.run \
  --config experiments/configs/baseline.yaml --smoke
```
