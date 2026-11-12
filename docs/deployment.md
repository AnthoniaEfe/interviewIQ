# Deployment: FastAPI backend (local or RunPod)

InterviewIQ is a Python research stack. There is no web frontend and nothing is deployed to Vercel.

| Host | Role |
| --- | --- |
| **Local / RunPod** | FastAPI in `backend/` — Luna/Terra, embeddings, experiment APIs, human-eval store |

## Local

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

Docker:

```bash
docker compose up backend
```

## RunPod (GPU)

Used for Sentence-Transformer embeddings and PyTorch experiments. Luna/Terra are API models and are never loaded onto the GPU. Device selection is `EMBEDDING_DEVICE=auto` (`torch.cuda.is_available()`).

```bash
cd /path/to/interviewIQ
PYTHONPATH=backend uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Public proxy must return 200:

```bash
curl -sS -o /dev/null -w "%{http_code}\n" "$API_URL/api/v1/health"
```

### Backend `.env` on RunPod

| Name | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Luna/Terra (server-side only) |
| `EMBEDDING_MODEL` / `EMBEDDING_DEVICE` | MiniLM on GPU (`auto` / `cuda`) |
| `VALIDATE_MODELS_ON_STARTUP` | Prefer `false` unless you want catalog checks at boot |
| `HUMAN_EVAL_DATA_DIR` | Persistent volume for ratings if a human study is run |
| `HUMAN_EVAL_ADMIN_TOKEN` | Admin header for the human-eval API |
| `CORS_ALLOW_ORIGINS` | Optional; only if a browser client is added later |

Embeddings load **lazily** (`app.state.embeddings` starts `None`).

## GPU boundary

- Embeddings and experiment runners stay on the FastAPI host.
- Human-eval ratings are stored on that host’s disk (`HUMAN_EVAL_DATA_DIR`), not in this git tree.
