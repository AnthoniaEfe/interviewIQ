# InterviewIQ frontend

Next.js research console. **No PyTorch / Sentence-Transformers / GPU code** runs here — the UI only calls the FastAPI backend (RunPod).

## Local setup

```bash
# terminal 1 — API on RunPod or local GPU host
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# terminal 2 — UI
cd frontend
cp .env.local.example .env.local
# for local API leave NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
npm install
npm run dev
```

Open http://localhost:3000

## Connect to RunPod (required for Vercel)

### 1. Root Directory (fixes `FUNCTION_INVOCATION_FAILED` / missing `fastapi`)

Vercel → Project → Settings → **Root Directory** = `frontend`

Do **not** use `backend`. That makes Vercel import `app/main.py` as a Python serverless function (`/var/task/app/main.py`) and fail with `ModuleNotFoundError: fastapi`. FastAPI stays on RunPod.

### 2. Public API URL

The browser must call the **public** backend URL, not `localhost` / `127.0.0.1`.

1. Expose FastAPI on RunPod (port 8000) and copy the HTTPS proxy URL, e.g.  
   `https://xxxxx-8000.proxy.runpod.net`
2. Vercel → Environment Variables:

```bash
NEXT_PUBLIC_API_URL=https://xxxxx-8000.proxy.runpod.net
```

3. On the **RunPod** backend `.env`:

```bash
CORS_ALLOW_ORIGINS=http://localhost:3000,https://your-app.vercel.app
# optional previews:
# CORS_ALLOW_ORIGIN_REGEX=https://.*\.vercel\.app
```

4. Redeploy after changing Root Directory or any `NEXT_PUBLIC_*` value.

Optional Docker Compose: `INTERNAL_API_URL=http://backend:8000` for server-side fetches; the browser still uses `NEXT_PUBLIC_API_URL`.

This project uses **Next.js** only — there is no `VITE_API_URL`. Full checklist: [docs/deployment.md](../docs/deployment.md).

## Pages

- `/` — experiment list + scale verdict
- `/metrics` — frozen metrics + DIAGNOSE/SCALE decision
- `/human-eval` — assignment audit + quality-rating scaffold
- `/experiments/[id]` — per-run charts + pair inspector
- `/upload` — store CV/JD pairs for testing

## GPU boundary

| Surface | Allowed |
| --- | --- |
| Vercel / this frontend | Static UI + `fetch` to FastAPI |
| RunPod FastAPI | Luna/Terra calls, MiniLM embeddings, CUDA |

Do not import `torch`, `sentence_transformers`, or embedding services into `frontend/`.
