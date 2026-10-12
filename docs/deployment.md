# Deployment: Vercel frontend + RunPod GPU backend

## Architecture

| Host | Role |
| --- | --- |
| **Vercel** | Next.js UI under `frontend/` only — no FastAPI, no PyTorch, no MiniLM |
| **RunPod** | FastAPI in `backend/` — Luna/Terra, embeddings, experiment APIs |

## Root cause of `FUNCTION_INVOCATION_FAILED` / `No module named 'fastapi'`

Vercel was treating **`backend/app/main.py`** as a Python serverless entrypoint (runtime path `/var/task/app/main.py`). That happens when the project **Root Directory** is `backend` (or the deploy root otherwise resolves to the FastAPI package). Vercel then imports FastAPI without installing `backend/requirements.txt` into that Python function runtime → `ModuleNotFoundError: fastapi`.

**Fix:** deploy Next.js from `frontend/`. Do **not** add FastAPI to Vercel Python deps; the API stays on RunPod.

## Vercel dashboard (required)

**Project → Settings → General → Root Directory**

| Setting | Value |
| --- | --- |
| **Root Directory** | `frontend` |
| Framework Preset | Next.js (auto) |
| Build Command | `npm run build` (default) |
| Install Command | `npm install` (default) |
| Output Directory | *(leave default / empty for Next.js)* |

If Root Directory cannot be changed yet, the repo also has:

- Root `vercel.json` + `package.json` — build via `npm … --prefix frontend`
- Root `.vercelignore` — excludes `backend/**`
- `backend/vercel.json` — **fails the build** with an explicit message if someone points Root Directory at `backend/`

After changing Root Directory, **redeploy**.

## Environment variables

### Vercel (Project → Settings → Environment Variables)

| Name | Where | Purpose |
| --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | Vercel (Production + Preview) | Public **HTTPS** RunPod FastAPI base URL, no trailing slash |

Example shape (replace with your proxy): `https://xxxxx-8000.proxy.runpod.net`

Do **not** set this to `localhost`, `127.0.0.1`, or `http://backend:8000` on Vercel.  
Do **not** put `OPENAI_API_KEY` or other secrets in `NEXT_PUBLIC_*`.

Optional (local Docker only, not Vercel): `INTERNAL_API_URL` for server-side fetch inside Compose.

### RunPod (backend `.env`)

| Name | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Luna/Terra (server-side only) |
| `CORS_ALLOW_ORIGINS` | Comma-separated origins, must include your Vercel URL(s) |
| `EMBEDDING_MODEL` / `EMBEDDING_DEVICE` | MiniLM on GPU (`auto` / `cuda`) |
| `VALIDATE_MODELS_ON_STARTUP` | Prefer `false` unless you want catalog checks at boot |

Example:

```bash
CORS_ALLOW_ORIGINS=http://localhost:3000,https://YOUR-APP.vercel.app,https://YOUR-APP-*.vercel.app
```

(Use exact preview URLs Vercel issues if wildcards are not supported by your CORS parser — this project splits on commas and matches exact origins.)

## RunPod verification

```bash
# On the pod — API listens on all interfaces
cd /path/to/interviewIQ
PYTHONPATH=backend uvicorn app.main:app --host 0.0.0.0 --port 8000

# From your laptop — public proxy must return 200
curl -sS -o /dev/null -w "%{http_code}\n" "$NEXT_PUBLIC_API_URL/api/v1/health"

# CORS preflight (Origin = your Vercel production URL)
curl -sS -D- -o /dev/null -X OPTIONS "$NEXT_PUBLIC_API_URL/api/v1/experiments" \
  -H "Origin: https://YOUR-APP.vercel.app" \
  -H "Access-Control-Request-Method: GET"
```

Expect `access-control-allow-origin` reflecting that Origin when CORS is configured.

## Deployment checklist

1. RunPod: uvicorn up; `/api/v1/health` OK on the **public** HTTPS URL.
2. RunPod: `CORS_ALLOW_ORIGINS` includes the Vercel production (and preview) origin(s); restart uvicorn.
3. Vercel: Root Directory = `frontend`.
4. Vercel: `NEXT_PUBLIC_API_URL` = public RunPod HTTPS base; redeploy after any change.
5. Open the Vercel homepage — experiment list loads (no FastAPI import errors in function logs).
6. Open `/metrics` — dashboard JSON loads from RunPod.
7. Optional: upload a pair / trigger generation only via RunPod-backed API routes (never local GPU on Vercel).

## GPU boundary

- Embeddings load **lazily** on RunPod (`app.state.embeddings` starts `None`).
- Do not add `torch` / `sentence-transformers` to `frontend/package.json`.
- Do not solve Vercel errors by installing FastAPI on Vercel.
