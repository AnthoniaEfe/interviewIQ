# InterviewIQ

Research-grade interview question generation: profile-gap analysis, competency-aware generation, semantic redundancy control, and competency-weighted MMR.

This repository currently contains:

1. `interviewiq_baseline/` — original flat-prompt GPT-4o baseline scripts (preserved).
2. `backend/` — InterviewIQ application (FastAPI): configuration, Luna/Terra routing, retrieval, gap analysis, MMR.
3. `prompts/` — versioned prompt files.
4. `experiments/` — experiment config and runner scaffold.

Implemented: Luna extraction, gap analysis, stored Terra flat-prompt baselines (B0), profile-aware baseline (B1), InterviewIQ pool→MMR treatment (T), plain-MMR / selection ablations (B2, S0–S2), and the primary-30 treatment pilot runner (`experiments/scripts/run_treatment_pilot.py`). See `docs/treatment_pilot_report.md`.

## Architecture

See [docs/architecture.md](docs/architecture.md), [docs/llm-routing.md](docs/llm-routing.md), [docs/research-pipeline.md](docs/research-pipeline.md), and [docs/experiments.md](docs/experiments.md).

## Model routing

| Task | Model env var | Default ID |
| --- | --- | --- |
| CV/JD extraction, normalization, validation | `LLM_FAST_MODEL` | `gpt-5.6-luna` |
| Profile/gap interpretation, question generation/refinement | `LLM_PRIMARY_MODEL` | `gpt-5.6-terra` |

Model IDs are verified against OpenAI documentation (`gpt-5.6-luna`, `gpt-5.6-terra`). If a configured ID is unavailable, the system fails closed and does not substitute another model.

Default pricing (OpenAI model docs, retrieved 2026-09-20), overridable via env:

- Terra: $2.00 input / $0.20 cached input / $12.00 output per 1M tokens
- Luna: $0.20 input / $0.02 cached input / $1.20 output per 1M tokens

## Setup

```bash
cp .env.example .env
# set OPENAI_API_KEY

python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health: `GET /api/v1/health`

There is no web UI. Browse experiments and metrics through the FastAPI routes (`/api/v1/experiments`, `/api/v1/experiments/metrics/dashboard`) or the files under `experiments/results/` and `docs/`. GPU embeddings and Luna/Terra calls stay on the FastAPI host (local or RunPod). See [docs/deployment.md](docs/deployment.md).

## Testing


Tests mock LLM usage and do not call paid APIs.

```bash
pip install -r backend/requirements.txt
pytest
```

Set `EMBEDDING_MODEL=hashing-test` in tests to avoid downloading Sentence-Transformers weights.

## GPU (RunPod RTX 4090)

Used for Sentence-Transformer embeddings and PyTorch experiments. Luna/Terra are API models and are never loaded onto the GPU. Device selection is `EMBEDDING_DEVICE=auto` (`torch.cuda.is_available()`).

## Research integrity

No experiment results are fabricated. The runner writes `status: not_yet_evaluated` until a real run completes.

Primary dataset: [`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit) (pandas + `hf://`). Local fit classifier training uses the train split; Terra/Luna are never fine-tuned.

The legacy baseline in `interviewiq_baseline/` uses `gpt-4o`. The thesis comparison must use the **same generation model** for baseline and InterviewIQ (Terra by default). The legacy script is retained as a historical prototype, not as the controlled experimental condition.
