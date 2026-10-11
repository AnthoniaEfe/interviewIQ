"""Selection ablations B2/S0/S1/S2 on the stored treatment pool (local, $0)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.schemas.domain import GapMatrix, GenerationMode
from app.schemas.pool import PoolQuestionItem, pool_item_to_interview_question
from app.services.embeddings.service import EmbeddingService
from app.services.gap_sensitivity.io_utils import write_json
from app.services.mmr.select import redundant_pairs
from app.services.question_generation.weights import map_weights_for_questions
from app.services.selection.strategies import select_questions
from app.services.treatment_pilot.constants import (
    FINAL_N,
    MMR_LAMBDA,
    RANDOM_SELECTION_REPEATS,
    SEED,
    SIM_THRESHOLD,
    USD_CNY_RATE,
)

ABLATION_SPECS = (
    ("B2", "plain_mmr_priority", 1),
    ("S0", "random", RANDOM_SELECTION_REPEATS),
    ("S1", "top_priority", 1),
    ("S2", "plain_mmr_query", 1),
)


def _jd_text(pairs_dir: Path | None, jd_id: str | None) -> str | None:
    if pairs_dir is None or not jd_id:
        return None
    path = pairs_dir / "jobs" / f"{jd_id}.txt"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return None


def run_selection_ablation(
    *,
    pool_dir: Path,
    gap_records: dict[str, dict],
    embeddings: EmbeddingService,
    experiment_id: str,
    out_dir: Path,
    pairs_dir: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_files = sorted(p for p in pool_dir.glob("*.json") if not p.name.startswith("_"))
    summaries: dict[str, Any] = {}
    errors: list[dict] = []

    for condition, strategy, repeats in ABLATION_SPECS:
        cond_dir = out_dir / condition
        cond_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict] = []
        for pool_path in pool_files:
            pool = json.loads(pool_path.read_text(encoding="utf-8"))
            stem = pool_path.stem
            # map stem back to gap key (strip __repN if present)
            gap_stem = stem.split("__rep")[0]
            gap_key = gap_stem + ".json"
            if gap_key not in gap_records:
                errors.append(
                    {
                        "condition": condition,
                        "stem": stem,
                        "error": f"missing gap {gap_key}",
                    }
                )
                continue
            gap_matrix = GapMatrix.model_validate(gap_records[gap_key]["gap_matrix"])
            valid_items = [
                PoolQuestionItem.model_validate(q) for q in pool.get("valid_questions") or []
            ]
            questions = [pool_item_to_interview_question(q) for q in valid_items]
            weights = list(pool.get("weights") or [])
            if len(weights) != len(questions):
                weights, _ = map_weights_for_questions(
                    [q.target_gap for q in questions], gap_matrix, embeddings
                )
            texts = [q.question for q in questions]
            import numpy as np

            sim = (
                embeddings.similarity_matrix(texts, texts)
                if texts
                else np.zeros((0, 0))
            )
            jd_text = _jd_text(pairs_dir, pool.get("jd_id"))
            query_sims = None
            if strategy == "plain_mmr_query":
                if not jd_text:
                    errors.append(
                        {
                            "condition": condition,
                            "stem": stem,
                            "error": "missing JD text for S2",
                        }
                    )
                    continue
                query_sims = embeddings.similarity_matrix(texts, [jd_text])[:, 0]

            repeat_payloads: list[dict] = []
            latencies: list[float] = []
            for r in range(repeats):
                started = time.perf_counter()
                sel = select_questions(
                    strategy,
                    questions,
                    sim,
                    weights,
                    lambda_=MMR_LAMBDA,
                    k=FINAL_N,
                    seed=SEED + r,
                    query_similarities=query_sims,
                )
                latencies.append((time.perf_counter() - started) * 1000)
                selected = [
                    {
                        **questions[i].model_dump(),
                        "weight": weights[i] if i < len(weights) else None,
                    }
                    for i in sel.selected_indices
                ]
                sel_texts = [q["question"] for q in selected]
                sel_sim = (
                    embeddings.similarity_matrix(sel_texts, sel_texts)
                    if len(sel_texts) > 1
                    else None
                )
                red_final = (
                    redundant_pairs(sel_sim, SIM_THRESHOLD) if sel_sim is not None else []
                )
                repeat_payloads.append(
                    {
                        "repeat": r,
                        "selected_indices": sel.selected_indices,
                        "questions": selected,
                        "n_selected": len(selected),
                        "n_redundant_pairs_final": len(red_final),
                        "backfilled": sel.backfilled,
                        "backfill_count": sel.backfill_count,
                        "notes": sel.notes,
                    }
                )

            # For S0, store mean-selected questions as the first-repeat set for
            # question-level diagnostics, plus all repeats for overlap stats.
            primary = repeat_payloads[0]
            rec = {
                "experiment_id": experiment_id,
                "stage": "selection_ablation",
                "condition": condition,
                "selection_strategy": strategy,
                "generation_mode": GenerationMode.INTERVIEWIQ.value
                if condition in {"B2", "T"}
                else None,
                "profile_id": pool.get("profile_id"),
                "jd_id": pool.get("jd_id"),
                "pair_id": pool.get("pair_id"),
                "fit_label": pool.get("fit_label"),
                "seniority": pool.get("seniority"),
                "pool_path": str(pool_path),
                "n_requirements": pool.get("n_requirements"),
                "questions": [{"question": q["question"]} for q in primary["questions"]],
                "questions_rich": primary["questions"],
                "n_questions": primary["n_selected"],
                "selected_indices": primary["selected_indices"],
                "repeats": repeat_payloads if repeats > 1 else None,
                "selection_latency_ms": round(sum(latencies) / len(latencies), 2),
                "cost_usd": 0.0,
                "cost_cny": 0.0,
                "usd_cny_rate": USD_CNY_RATE,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            write_json(cond_dir / f"{stem}.json", rec)
            results.append(rec)

        summaries[condition] = {
            "n_ok": len(results),
            "strategy": strategy,
            "repeats": repeats,
            "mean_selection_latency_ms": (
                round(
                    sum(r["selection_latency_ms"] for r in results) / len(results), 2
                )
                if results
                else None
            ),
        }

    summary = {
        "experiment_id": experiment_id,
        "status": "completed" if not errors else "partial",
        "stage": "selection_ablation",
        "conditions": summaries,
        "errors": errors,
        "usd_cny_rate": USD_CNY_RATE,
        "model": None,
        "generation_mode": GenerationMode.INTERVIEWIQ.value,
    }
    write_json(out_dir / "summary.json", summary)
    return summary
