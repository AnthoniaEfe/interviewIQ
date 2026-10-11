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

## Treatment pilot (primary-30: B1, T, B2, S0–S2)

Pinned assets: extraction `exp_20260927T205741Z`, gap `exp_20260927T215232Z`, stored B0 replicates (do not regenerate). Report: `docs/treatment_pilot_report.md`.

```bash
# Local regression (gap reproduce + B0 validity) — no API spend
PYTHONPATH=backend python experiments/scripts/run_treatment_pilot.py --stage regression

# Dry-run cost estimate (stops if OPENAI_API_KEY missing)
PYTHONPATH=backend python experiments/scripts/run_treatment_pilot.py --dry-run

# 2-pair smoke, then full 30-pair pilot under spend guard
PYTHONPATH=backend python experiments/scripts/run_treatment_pilot.py \
  --smoke --confirm-spend --max-cost 4.00
PYTHONPATH=backend python experiments/scripts/run_treatment_pilot.py \
  --confirm-spend --max-cost 4.00

# Or via the experiment runner stages
PYTHONPATH=backend python -m experiments.run \
  --config experiments/configs/primary_hf.yaml \
  --stage baseline_profile_aware --confirm-spend --max-cost 4.00
PYTHONPATH=backend python -m experiments.run \
  --config experiments/configs/primary_hf.yaml \
  --stage treatment_pipeline --selection competency_weighted_mmr \
  --confirm-spend --max-cost 4.00
PYTHONPATH=backend python -m experiments.run \
  --config experiments/configs/primary_hf.yaml \
  --stage selection_ablation --treatment-experiment <exp_id>
PYTHONPATH=backend python -m experiments.run \
  --config experiments/configs/primary_hf.yaml \
  --stage pilot_compare \
  --b1-experiment <exp_id> --treatment-experiment <exp_id> \
  --ablation-experiment <exp_id>
```

Flags: `--dry-run`, `--confirm-spend`, `--max-cost` (default 4.00 USD), `--resume` / `--no-resume`, `--replicates`, `--source-experiment`, `--gap-experiment`, `--selection`.
