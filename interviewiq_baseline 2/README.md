# InterviewIQ — Baseline

**Flat-prompt LLM baseline** for the thesis experiment.  
Input: CV text + JD text → single GPT-4o prompt → 15 questions.  
No gap matrix. No competency budget. No redundancy control.

This is the **B condition** in the experiment. The full pipeline (treatment) is in `../interviewiq/`.

---

## Why not dvircohen/LLM-interviewer?

That repo is a **conversational interview simulator** — turn-by-turn, voice-enabled, candidate-facing.  
This thesis needs **batch pre-generation** of question sets for blind human evaluation.  
The two paradigms are incompatible as experimental conditions.

---

## Setup

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

---

## Test one combination first (smoke test)

```bash
python baseline.py \
  --cv data/profiles/p01_junior_swe.txt \
  --jd data/jds/software_engineer.txt \
  --seniority junior \
  --out results/baseline/p01_swe_junior.json

python review.py --file results/baseline/p01_swe_junior.json
```

**Check before running batch:**
- Are there exactly 15 questions?
- Do they look generic (no mention of the candidate's specific skills/gaps)?  
  That's expected — this IS the baseline. It proves the problem.

---

## Run full batch (45 combinations)

```bash
# Smoke: 3 combos only
python run_batch_baseline.py --smoke

# Full: all 45
python run_batch_baseline.py
```

---

## Inspect results

```bash
python review.py            # list all files + question counts
python review.py --all      # print every question set
```

---

## File structure

```
interviewiq_baseline/
├── baseline.py              # Core: flat prompt -> 15 questions
├── run_batch_baseline.py    # Batch runner for all 45 combos
├── inspect.py               # Review generated outputs
├── requirements.txt
├── .env.example
└── data/
    ├── profiles/            # 15 synthetic CV .txt files (add yours here)
    │   └── p01_junior_swe.txt
    └── jds/                 # 3 JD .txt files (add yours here)
        └── software_engineer.txt
```

---

## Output format

```json
{
  "profile_id": "p01_junior_swe",
  "jd_id": "software_engineer",
  "seniority": "junior",
  "model": "gpt-4o",
  "temperature": 0.7,
  "seed": 42,
  "generated_at": "2026-03-14T10:00:00Z",
  "questions": [
    {"id": 1, "question": "..."},
    ...
  ]
}
```

---

## What to verify after smoke test

| Check | Expected |
|-------|----------|
| Question count | Exactly 15 |
| Question content | Generic role-level, not CV-specific |
| Redundancy | Some semantic overlap expected — that's the problem |
| Competency mix | Skewed toward technical — that's the problem |
| JSON validity | No parse errors |

These failures in the baseline are what the treatment pipeline fixes.

---

## Next step

Once baseline passes smoke test and full batch, run the treatment pipeline
in `../interviewiq/` on the same 45 combinations, then compare using
`../experiments/evaluate_results.py`.
