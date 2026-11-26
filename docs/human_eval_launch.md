# Human evaluation launch status (T′ vs B1)

**Date:** 2026-11-26  
**Treatment arm:** T′ (forward treatment). Baseline: B1.  
**Stage:** Rubric pilot ready; **awaiting human raters**. No human ratings have been collected. Do not invent scores.

## Ready artifacts

| Artifact | Path |
| --- | --- |
| Full primary-30 inputs | `experiments/human_eval/inputs/primary30_Tprime_vs_B1.json` |
| Pilot 5-pair sample | `experiments/human_eval/inputs/pilot5_Tprime_vs_B1.json` (seed `20261126`) |
| Blinded pilot study | `experiments/human_eval/data/pilot/study.json` (gitignored) |
| Condition key | `experiments/human_eval/keys/condition_key_pilot.json` (gitignored) |
| Manifest (hash only) | `experiments/human_eval/manifest_pilot.json` |
| Rater UI (local HTML) | `experiments/human_eval/rater_app.html` |
| Rubric (still draft) | `docs/human_eval_rubric.md` |
| Protocol | `docs/human_eval_protocol.md` |
| T′ addendum | `docs/human_eval_tprime_addendum.md` |

Pilot pairs: `rjf_03202_f43978101918`, `rjf_04668_066625dfe7d9`, `rjf_04729_4b4032772de3`, `rjf_05281_56dd66bfd7af`, `rjf_05855_1d0e27a3eacb`.

## Pre-launch smoke (done locally)

1. API health 200 with `HUMAN_EVAL_DATA_DIR` + `HUMAN_EVAL_ADMIN_TOKEN` set.
2. Pilot study loaded; three rater tokens issued for R1–R3.
3. Session payload has Set X / Set Y only (no condition fields).
4. Demo mode accepts a smoke rating; demo export refused (422).
5. Pilot export currently **0** ratings.

## Access from your laptop (SSH remote)

`192.168.0.2` is the **GPU host’s LAN** address. Your browser on a local machine cannot reach it unless you are on that same LAN. Use an SSH tunnel instead:

```bash
# on your LOCAL machine (keep this session open)
ssh -L 8000:127.0.0.1:8000 <user>@<gpu-host>
```

Then open **http://127.0.0.1:8000/rater** in the local browser (not `192.168.0.2`).  
API base: leave blank or `http://127.0.0.1:8000`. Mode: `pilot`. Token: a **rater token** (R1–R3), not an OpenAI key.

## What you do next (humans required)

1. Recruit ≥3 raters (kit: `experiments/human_eval/kit/`). Separate from gold-set annotators; ≥1 with HR/interviewing experience if possible.
2. Send each rater: local-tunnel URL (or same-LAN URL), mode `pilot`, personal token, and `docs/human_eval_rubric.md`. Prefer `http://127.0.0.1:8000/rater` via SSH `-L` when the API runs on a remote GPU host.
3. After all three finish the 5 pairs, export:

```bash
curl -sS "$API/api/v1/human-eval/admin/export?mode=pilot" \
  -H "X-Human-Eval-Admin: $HUMAN_EVAL_ADMIN_TOKEN" \
  -o experiments/human_eval/exports/ratings_pilot.json
```

4. Freeze rubric wording at most once from that pilot (`docs/human_eval_rubric.md` → status frozen). Record median `elapsed_ms`.
5. Exploratory primary-30 study (still not confirmatory holdout) can reuse `primary30_Tprime_vs_B1.json` with `--mode` only after rubric freeze; label it exploratory in the report.
6. Confirmatory **main** study remains holdout-30 × {T′, B1} and still needs holdout T′ generation under the spend gate.

## Integrity

- Pilot ratings are excluded from main analysis by protocol.
- Do not commit `experiments/human_eval/keys/` or raw tokens.
- Do not run the Luna judge on holdout until `docs/judge_validation_prereg.md` says `approval: approved`.
