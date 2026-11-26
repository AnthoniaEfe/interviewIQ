"""Promote T′ to the forward treatment on stored primary-30 pools.

No LLM calls. Reads selector_tprime_replay indices + stored pools / B1,
writes:
  experiments/results/selector_tprime_replay/treatment/*.json
  experiments/human_eval/inputs/primary30_Tprime_vs_B1.json
  experiments/results/selector_tprime_replay/adoption.json

Historical T (competency_weighted_mmr) artifacts are left unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.gap_sensitivity.io_utils import git_commit_hash, write_json  # noqa: E402

POOL_DIR = ROOT / "experiments/results/exp_20261011T130619Z/pool"
B1_DIR = ROOT / "experiments/results/exp_20261011T130310Z/baseline_profile_aware"
REPLAY = ROOT / "experiments/results/selector_tprime_replay"
PER_POOL = REPLAY / "per_pool.json"
SUMMARY = REPLAY / "summary.json"
TREATMENT_OUT = REPLAY / "treatment"
HUMAN_INPUTS = ROOT / "experiments/human_eval/inputs/primary30_Tprime_vs_B1.json"
PAIRS_DIR = ROOT / "experiments/datasets/resume_job_fit/experiment_pairs"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _question_texts(record: dict) -> list[str]:
    out: list[str] = []
    for item in record.get("questions") or []:
        if isinstance(item, dict):
            text = str(item.get("question") or "").strip()
        else:
            text = str(item).strip()
        if text:
            out.append(text)
    return out


def main() -> int:
    summary = _load(SUMMARY)
    success = (summary.get("success_rule") or {}).get("all_three")
    if not success:
        raise SystemExit("Refusing to promote: T′ success_rule.all_three is not True")

    rows = (_load(PER_POOL).get("rows") or [])
    if len(rows) != 30:
        raise SystemExit(f"Refusing to promote: expected 30 per_pool rows, got {len(rows)}")

    TREATMENT_OUT.mkdir(parents=True, exist_ok=True)
    HUMAN_INPUTS.parent.mkdir(parents=True, exist_ok=True)

    human_pairs: list[dict] = []
    for row in rows:
        stem = row["stem"]
        pair_id = row["pair_id"]
        indices = list(row["Tprime"]["selected_indices"])
        pool = _load(POOL_DIR / f"{stem}.json")
        b1 = _load(B1_DIR / f"{stem}.json")
        valid = pool.get("valid_questions") or []
        selected = [valid[i] for i in indices]
        treatment = {
            "experiment_id": "selector_tprime_replay",
            "stage": "treatment_pipeline",
            "generation_mode": "interviewiq",
            "condition": "T",
            "condition_detail": "Tprime",
            "selection_strategy": "tprime",
            "forward_treatment": True,
            "profile_id": pool.get("profile_id") or pair_id,
            "jd_id": pool.get("jd_id") or pair_id,
            "pair_id": pair_id,
            "fit_label": pool.get("fit_label"),
            "seniority": pool.get("seniority"),
            "replicate": 0,
            "model": pool.get("model"),
            "prompt_hash": pool.get("prompt_hash"),
            "prompt_version": pool.get("prompt_version"),
            "pool_path": f"../exp_20261011T130619Z/pool/{stem}.json",
            "pool_stats": pool.get("pool_stats")
            or {
                "n_raw": pool.get("n_raw"),
                "n_valid": len(valid),
            },
            "selection": {
                "strategy": "tprime",
                "selected_indices": indices,
                "questions": selected,
                "n_selected": len(selected),
                "quota_met": row["Tprime"].get("quota_met"),
                "dimension_counts": row["Tprime"].get("dimension_counts"),
                "gap_share": row["Tprime"].get("gap_share"),
                "relaxations": row["Tprime"].get("relaxations") or [],
                "metrics": row["Tprime"].get("metrics"),
            },
            "questions": [{"question": q["question"]} for q in selected],
            "questions_rich": selected,
            "n_questions": len(selected),
            "source_replay": "experiments/results/selector_tprime_replay",
            "prereg_path": "docs/selector_tprime_prereg.md",
            "adoption_path": "docs/selector_tprime_adoption.md",
        }
        write_json(TREATMENT_OUT / f"{stem}.json", treatment)

        cv_path = PAIRS_DIR / "profiles" / f"{pair_id}.txt"
        jd_path = PAIRS_DIR / "jobs" / f"{pair_id}.txt"
        human_pairs.append(
            {
                "pair_id": pair_id,
                "stem": stem,
                "split": "primary-30",
                "treatment": "Tprime",
                "job_text": jd_path.read_text(encoding="utf-8") if jd_path.is_file() else "",
                "candidate_text": cv_path.read_text(encoding="utf-8") if cv_path.is_file() else "",
                # Blinded study builder expects keys T and B1; T is the forward treatment (T′).
                "T": [q["question"] for q in selected],
                "B1": _question_texts(b1),
            }
        )

    HUMAN_INPUTS.write_text(json.dumps(human_pairs, indent=2) + "\n", encoding="utf-8")
    adoption = {
        "status": "adopted",
        "forward_treatment": "Tprime",
        "selection_strategy": "tprime",
        "historical_T": "competency_weighted_mmr",
        "n_pools": len(rows),
        "success_rule": summary.get("success_rule"),
        "recomputed_means_Tprime": (summary.get("recomputed_means") or {}).get("Tprime"),
        "treatment_dir": str(TREATMENT_OUT.relative_to(ROOT)),
        "human_eval_inputs": str(HUMAN_INPUTS.relative_to(ROOT)),
        "git_commit": git_commit_hash(),
        "note": (
            "T′ replaces competency_weighted_mmr as the forward treatment selector. "
            "Stored historical T artifacts are unchanged. Human-eval inputs label the "
            "T′ set as condition T for blinding compatibility."
        ),
    }
    write_json(REPLAY / "adoption.json", adoption)
    print(
        json.dumps(
            {
                "n_treatment": len(list(TREATMENT_OUT.glob("*.json"))),
                "n_human_pairs": len(human_pairs),
                "human_eval_inputs": str(HUMAN_INPUTS),
                "adoption": str(REPLAY / "adoption.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
