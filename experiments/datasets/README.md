# Research datasets

## Primary dataset

[`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit)

| Split | Rows | Labels |
| --- | ---: | --- |
| train | 6241 | No Fit / Potential Fit / Good Fit |
| test | 1759 | same |

Columns: `resume_text`, `job_description_text`, `label`.

Loaded with pandas:

```python
import pandas as pd

splits = {"train": "train.csv", "test": "test.csv"}
df = pd.read_csv(
    "hf://datasets/cnamuangtoun/resume-job-description-fit/" + splits["train"]
)
```

### What is trained

- **Trained:** local Sentence-Transformer + logistic regression fit classifier on the **train** split; evaluated on **test**.
- **Not trained:** `gpt-5.6-terra` / `gpt-5.6-luna` (API generation models only).

```bash
PYTHONPATH=backend python experiments/scripts/train_fit_classifier.py
```

Artifacts:

- `experiments/datasets/resume_job_fit/cache/` — parquet caches
- `experiments/datasets/resume_job_fit/models/` — classifier + metrics
- `experiments/datasets/resume_job_fit/experiment_pairs/` — stratified Good Fit / Potential Fit pairs for generation

Smoke CV/JD files under `profiles/` and `jobs/` remain for the legacy baseline path only.
