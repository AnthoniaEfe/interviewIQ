"""
baseline.py
===========
Baseline condition: CV + JD -> single flat GPT prompt -> 15 questions.
No gap analysis. No competency budget. No redundancy control.
This is what dvircohen/LLM-interviewer does, but as a batch generator
suitable for the thesis experiment (15 profiles x 3 roles x 3 seniority).

Usage:
    python baseline.py --cv data/profiles/p01_junior.txt \
                       --jd data/jds/software_engineer.txt \
                       --seniority junior \
                       --out results/baseline/p01_swe_junior.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o")
TEMP    = float(os.getenv("OPENAI_TEMP", "0.7"))
SEED    = int(os.getenv("OPENAI_SEED", "42"))
N_QS    = 15  # fixed question count, same as treatment


# ── Prompt ────────────────────────────────────────────────────────────────────

BASELINE_SYSTEM = (
    "You are an expert technical recruiter. "
    "Generate exactly 15 diverse interview questions for the candidate described below."
)

BASELINE_USER = """\
JOB DESCRIPTION:
{jd}

CANDIDATE CV:
{cv}

SENIORITY LEVEL: {seniority}

Generate exactly 15 interview questions appropriate for this candidate and role.
Return a JSON object with this structure:
{{
  "questions": [
    {{"id": 1, "question": "..."}},
    ...
  ]
}}
Return ONLY valid JSON. No explanation, no markdown fences.
"""


# ── Model ─────────────────────────────────────────────────────────────────────

class Question(BaseModel):
    id: int
    question: str


class BaselineOutput(BaseModel):
    profile_id: str
    jd_id: str
    seniority: str
    model: str
    temperature: float
    seed: int
    generated_at: str
    questions: list[Question]


# ── Core ──────────────────────────────────────────────────────────────────────

def generate_baseline(cv_text: str, jd_text: str, seniority: str) -> list[dict]:
    """Single flat prompt -> 15 questions. No pipeline."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": BASELINE_SYSTEM},
            {"role": "user",   "content": BASELINE_USER.format(
                jd=jd_text, cv=cv_text, seniority=seniority
            )},
        ],
        response_format={"type": "json_object"},
        temperature=TEMP,
        seed=SEED,
    )
    raw = response.choices[0].message.content
    data = json.loads(raw)

    questions = data.get("questions", [])
    if len(questions) != N_QS:
        print(f"  [warn] expected {N_QS} questions, got {len(questions)}", file=sys.stderr)

    return questions


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="InterviewIQ — Baseline generator")
    parser.add_argument("--cv",        required=True, help="Path to candidate CV text file")
    parser.add_argument("--jd",        required=True, help="Path to job description text file")
    parser.add_argument("--seniority", required=True, choices=["junior", "mid", "senior"])
    parser.add_argument("--out",       required=True, help="Output JSON file path")
    parser.add_argument("--profile-id", default=None, help="Profile identifier for tracking")
    parser.add_argument("--jd-id",      default=None, help="JD identifier for tracking")
    args = parser.parse_args()

    cv_path = Path(args.cv)
    jd_path = Path(args.jd)

    if not cv_path.exists():
        sys.exit(f"CV file not found: {cv_path}")
    if not jd_path.exists():
        sys.exit(f"JD file not found: {jd_path}")

    cv_text = cv_path.read_text(encoding="utf-8").strip()
    jd_text = jd_path.read_text(encoding="utf-8").strip()

    profile_id = args.profile_id or cv_path.stem
    jd_id      = args.jd_id      or jd_path.stem

    print(f"[baseline] {profile_id} x {jd_id} x {args.seniority} -> {args.out}")

    questions = generate_baseline(cv_text, jd_text, args.seniority)

    out = BaselineOutput(
        profile_id=profile_id,
        jd_id=jd_id,
        seniority=args.seniority,
        model=MODEL,
        temperature=TEMP,
        seed=SEED,
        generated_at=datetime.utcnow().isoformat() + "Z",
        questions=[Question(**q) for q in questions],
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out.model_dump_json(indent=2), encoding="utf-8")
    print(f"[baseline] saved {len(questions)} questions -> {out_path}")


if __name__ == "__main__":
    main()
