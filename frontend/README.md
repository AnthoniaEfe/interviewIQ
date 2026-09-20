# InterviewIQ frontend

Next.js research console for experiment visualization and CV/JD uploads.

## Setup

```bash
# terminal 1 — API
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# terminal 2 — UI
cd frontend
cp .env.local.example .env.local   # if needed
npm install
npm run dev
```

Open http://localhost:3000

## Pages

- `/` — experiment list + aggregate cost/latency
- `/experiments/[id]` — charts (latency, cost, tokens, fit labels) + question inspector
- `/upload` — store CV/JD pairs for testing

API base: `NEXT_PUBLIC_API_URL` (default `http://127.0.0.1:8000`)
