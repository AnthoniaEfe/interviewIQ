# Human evaluation runbook

The Vercel app is the Next.js site in `frontend/`. It does not store ratings.
The FastAPI process on RunPod does. There is no database in this repo.

## What you must provision

Attach a **RunPod network volume** (or other disk that survives pod restarts)
and set on the RunPod service:

| Variable | Purpose |
| --- | --- |
| `HUMAN_EVAL_DATA_DIR` | Directory on that volume. Ratings, tokens (hashed), and blinded payloads go here. |
| `HUMAN_EVAL_ADMIN_TOKEN` | Secret for loading a study, issuing links, revoking links, and export. Send it only as the `X-Human-Eval-Admin` header. |
| `NEXT_PUBLIC_API_URL` | Already set on Vercel to the public RunPod HTTPS origin. |
| `CORS_ALLOW_ORIGINS` | Must include the Vercel origin. |

Do not put the admin token or the condition key in Vercel env vars. Do not
commit `experiments/human_eval/keys/` or `experiments/human_eval/data/`.

If `HUMAN_EVAL_DATA_DIR` is unset, the API writes under
`experiments/human_eval/data/` on the machine running FastAPI. That disk is
ephemeral on a stock RunPod pod.

## Local

```bash
export HUMAN_EVAL_ADMIN_TOKEN='choose-a-long-secret'
export HUMAN_EVAL_DATA_DIR=/tmp/interviewiq-human-eval
cd backend && PYTHONPATH=. uvicorn app.main:app --reload --port 8000
```

In another shell, from `frontend/`, with Node installed and `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000`:

```bash
npm install
npm run dev
```

Open the printed `/study/<token>?mode=pilot` link.

## Load a study and issue links

Build only from real stored questions. The builder refuses to invent them.

```bash
PYTHONPATH=backend python experiments/human_eval/build_study.py \
  --inputs /path/to/records.json \
  --raters R1 R2 R3 \
  --mode main
```

`records.json` is a list of objects with `job_text`, `candidate_text`, `T`
(list of question strings), and `B1`. The condition key is written to
`experiments/human_eval/keys/condition_key_main.json`. Only
`manifest_main.json` (seed and key hash) is safe to commit.

Then, against the API:

```bash
curl -sS -X POST "$API/api/v1/human-eval/admin/study" \
  -H "X-Human-Eval-Admin: $HUMAN_EVAL_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d @experiments/human_eval/data/main/study.json
```

Wrap that file as `{"mode":"main","blinded": <contents>}` before posting.
Issue one token per rater:

```bash
curl -sS -X POST "$API/api/v1/human-eval/admin/tokens" \
  -H "X-Human-Eval-Admin: $HUMAN_EVAL_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"mode":"main","rater_id":"R1"}'
```

Send the rater `https://<vercel-app>/study/<token>?mode=main`. To revoke, `POST
/api/v1/human-eval/admin/revoke?token=...&mode=main` with the admin header.

Pilot links use `mode=pilot` and a study built from about five primary-30
pairs. Demo uses `mode=demo` and is rejected by export and by the analysis script.

## Pre-launch smoke

1. `GET /api/v1/health` on the public RunPod URL returns 200.
2. A session response has Set X and Set Y and no `condition` field.
3. Changing a score and reloading the link shows the saved score.
4. Export with the admin header returns ratings. Export of `mode=demo` returns 422.
5. The condition-key file is not in the Vercel deployment or `frontend/public/`.

## Export and backup

Export before the window opens (expect an empty list) and again when it closes:

```bash
curl -sS "$API/api/v1/human-eval/admin/export?mode=main" \
  -H "X-Human-Eval-Admin: $HUMAN_EVAL_ADMIN_TOKEN" \
  -o ratings_main.json
```

Copy `HUMAN_EVAL_DATA_DIR` as well. Analysis, only after the window closes:

```bash
PYTHONPATH=backend python experiments/human_eval/analyze.py \
  --ratings ratings_main.json \
  --condition-key experiments/human_eval/keys/condition_key_main.json \
  --mode main \
  --n-items 30
```

The script refuses pilot, demo, and completion under 90% per rater. It does not
read human ratings until you point it at a real export.
