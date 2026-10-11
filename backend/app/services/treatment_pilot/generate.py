"""Paid generation stages: B1 (profile-aware) and T (pool + selection)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config.settings import Settings
from app.schemas.domain import (
    CandidateProfile,
    GapMatrix,
    GenerationMode,
    InterviewQuestion,
    JobDescription,
    LLMUsage,
)
from app.schemas.pool import PoolQuestionItem
from app.services.embeddings.service import EmbeddingService
from app.services.fit_label_control.extract import api_key_present
from app.services.gap_sensitivity.io_utils import request_id, write_json
from app.services.llm.factory import build_router
from app.services.llm.provider import LLMResult
from app.services.llm.tasks import LLMTask
from app.services.mmr.select import redundant_pairs
from app.services.question_generation.pool import (
    POOL_PROMPT,
    generate_pool,
    prompt_sha256,
    validate_pool_items,
)
from app.services.question_generation.profile_aware import (
    PROFILE_AWARE_PROMPT,
    generate_profile_aware,
)
from app.services.question_generation.weights import map_weights_for_questions
from app.services.selection.strategies import select_questions
from app.services.treatment_pilot.cache import (
    cache_key,
    index_generation_cache,
    read_cache,
    write_cache,
)
from app.services.treatment_pilot.constants import (
    B0_MEAN_COST_USD,
    B0_MEAN_INPUT_TOKENS,
    B0_MEAN_OUTPUT_TOKENS,
    B1_INPUT_MULT,
    B1_OUTPUT_MULT,
    FINAL_N,
    GENERATION_MODEL,
    MAX_OUTPUT_TOKENS,
    MMR_LAMBDA,
    POOL_INPUT_MULT,
    POOL_N,
    POOL_OUTPUT_MULT,
    REASONING_EFFORT,
    SEED,
    SIM_THRESHOLD,
    TEMPERATURE_SETTING,
    USD_CNY_RATE,
)
from app.services.treatment_pilot.currency import usd_to_cny
from app.services.treatment_pilot.spend import SpendAborted, TreatmentSpendGuard


def generation_parameters() -> dict[str, Any]:
    return {
        "temperature": TEMPERATURE_SETTING,
        "reasoning_effort": REASONING_EFFORT,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "model": GENERATION_MODEL,
    }


def _usage_payload(usage: LLMResult, task_type: str) -> dict:
    return LLMUsage(
        provider=usage.provider,
        model=usage.model,
        task_type=task_type,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_tokens=usage.cached_tokens,
        estimated_cost=usage.estimated_cost,
        latency_ms=usage.latency_ms,
        retry_count=usage.retry_count,
        fallback_used=usage.fallback_used,
        fallback_reason=usage.fallback_reason,
    ).model_dump()


def estimate_stage_costs(n_pairs: int) -> dict[str, Any]:
    """Dry-run estimates from B0 averages with explicit multipliers."""
    terra_in = 2.0 / 1_000_000
    terra_out = 12.0 / 1_000_000
    b1_in = B0_MEAN_INPUT_TOKENS * B1_INPUT_MULT
    b1_out = B0_MEAN_OUTPUT_TOKENS * B1_OUTPUT_MULT
    b1_cost = b1_in * terra_in + b1_out * terra_out
    pool_in = B0_MEAN_INPUT_TOKENS * POOL_INPUT_MULT
    pool_out = B0_MEAN_OUTPUT_TOKENS * POOL_OUTPUT_MULT
    pool_cost = pool_in * terra_in + pool_out * terra_out
    return {
        "assumptions": {
            "b0_mean_input_tokens": B0_MEAN_INPUT_TOKENS,
            "b0_mean_output_tokens": B0_MEAN_OUTPUT_TOKENS,
            "b0_mean_cost_usd": B0_MEAN_COST_USD,
            "b1_input_mult": B1_INPUT_MULT,
            "b1_output_mult": B1_OUTPUT_MULT,
            "pool_input_mult": POOL_INPUT_MULT,
            "pool_output_mult": POOL_OUTPUT_MULT,
            "terra_input_usd_per_mtok": 2.0,
            "terra_output_usd_per_mtok": 12.0,
        },
        "b1": {
            "est_input_tokens": round(b1_in, 2),
            "est_output_tokens": round(b1_out, 2),
            "est_cost_per_pair_usd": round(b1_cost, 6),
            "est_total_usd": round(b1_cost * n_pairs, 6),
            "est_total_cny": usd_to_cny(b1_cost * n_pairs),
        },
        "t_pool": {
            "est_input_tokens": round(pool_in, 2),
            "est_output_tokens": round(pool_out, 2),
            "est_cost_per_pair_usd": round(pool_cost, 6),
            "est_total_usd": round(pool_cost * n_pairs, 6),
            "est_total_cny": usd_to_cny(pool_cost * n_pairs),
        },
        "total_paid_usd": round((b1_cost + pool_cost) * n_pairs, 6),
        "total_paid_cny": usd_to_cny((b1_cost + pool_cost) * n_pairs),
        "n_pairs": n_pairs,
        "usd_cny_rate": USD_CNY_RATE,
    }


def _prompt_meta(prompt_obj, system_text: str) -> dict[str, Any]:
    return {
        "file": prompt_obj.relative_path,
        "version": prompt_obj.version,
        "sha256": prompt_sha256(system_text),
    }


async def run_baseline_profile_aware(
    *,
    root: Path,
    settings: Settings,
    extraction_records: list[dict],
    experiment_id: str,
    out_dir: Path,
    guard: TreatmentSpendGuard,
    resume: bool = True,
    replicates: int = 1,
    dry_run: bool = False,
) -> dict[str, Any]:
    stage_dir = out_dir / "baseline_profile_aware"
    stage_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = root / "experiments" / "results" / "_treatment_cache" / "b1"
    system_text = PROFILE_AWARE_PROMPT.load()
    prompt_hash = prompt_sha256(system_text)
    params = generation_parameters()
    cache_index = (
        index_generation_cache(
            root / "experiments" / "results",
            stage_subdir="baseline_profile_aware",
            prompt_hash=prompt_hash,
            model_id=GENERATION_MODEL,
            parameters=params,
        )
        if resume
        else {}
    )

    if dry_run:
        n = len(extraction_records) * replicates
        est = estimate_stage_costs(len(extraction_records))
        print("[baseline_profile_aware] dry-run")
        print(json.dumps({"n_pairs": n, "estimate": est["b1"]}, indent=2))
        return {"dry_run": True, "estimate": est["b1"], "n_pairs": n}

    if not api_key_present(settings):
        return {
            "stopped": "missing_api_key",
            "command": (
                "PYTHONPATH=backend python experiments/scripts/run_treatment_pilot.py "
                f"--confirm-spend --max-cost {guard.max_cost}"
            ),
        }

    guard.require_confirm()
    router = build_router(settings)
    results: list[dict] = []
    errors: list[dict] = []
    n_cache_hits = 0

    for rep in range(replicates):
        for record in extraction_records:
            rid = request_id(record)
            stem = rid if replicates == 1 else f"{rid}__rep{rep}"
            out_path = stage_dir / f"{stem}.json"
            pair_id = str(record.get("pair_id") or record.get("profile_id"))

            cached_path = cache_index.get(stem) or cache_index.get(rid)
            disk_cached = read_cache(
                cache_dir,
                cache_key(
                    pair_id=f"{stem}",
                    prompt_hash=prompt_hash,
                    model_id=GENERATION_MODEL,
                    parameters=params,
                ),
            )
            if resume and out_path.is_file():
                results.append(json.loads(out_path.read_text(encoding="utf-8")))
                n_cache_hits += 1
                continue
            if resume and disk_cached is not None:
                write_json(out_path, disk_cached)
                results.append(disk_cached)
                n_cache_hits += 1
                continue
            if resume and cached_path is not None and cached_path.is_file():
                rec = json.loads(cached_path.read_text(encoding="utf-8"))
                # zero cost on reuse for this experiment's accounting? Spec: re-runs never pay twice.
                # Keep original usage for provenance; mark reused.
                rec = dict(rec)
                rec["experiment_id"] = experiment_id
                rec["reused_from"] = str(cached_path)
                write_json(out_path, rec)
                results.append(rec)
                n_cache_hits += 1
                continue

            try:
                guard.check_next_pair()
                candidate = CandidateProfile.model_validate(record["candidate"])
                job = JobDescription.model_validate(record["job"])
                parsed, usage, system_prompt, user_prompt = await generate_profile_aware(
                    router=router,
                    settings=settings,
                    candidate=candidate,
                    job=job,
                    seniority=record.get("seniority") or "mid",
                    n_questions=FINAL_N,
                    experiment_id=experiment_id,
                    request_id=stem,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    reasoning_effort=REASONING_EFFORT,
                )
                truncated = bool(
                    getattr(usage, "output_tokens", 0) >= MAX_OUTPUT_TOKENS - 1
                )
                payload = {
                    "experiment_id": experiment_id,
                    "stage": "baseline_profile_aware",
                    "generation_mode": GenerationMode.BASELINE.value,
                    "condition": "B1",
                    "profile_id": record.get("profile_id"),
                    "jd_id": record.get("jd_id"),
                    "pair_id": pair_id,
                    "fit_label": record.get("fit_label"),
                    "seniority": record.get("seniority"),
                    "replicate": rep,
                    "provider": usage.provider,
                    "model": usage.model,
                    "task_type": LLMTask.QUESTION_GENERATION.value,
                    "prompt": _prompt_meta(PROFILE_AWARE_PROMPT, system_prompt),
                    "prompt_hash": prompt_hash,
                    "prompt_version": PROFILE_AWARE_PROMPT.version,
                    "generation_parameters": params,
                    "final_question_count_requested": FINAL_N,
                    "questions": [q.model_dump() for q in parsed.questions],
                    "n_questions": len(parsed.questions),
                    "truncated": truncated,
                    "usage": _usage_payload(usage, LLMTask.QUESTION_GENERATION.value),
                    "cost_usd": usage.estimated_cost,
                    "cost_cny": usd_to_cny(usage.estimated_cost),
                    "usd_cny_rate": USD_CNY_RATE,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "user_prompt_sha256": prompt_sha256(user_prompt),
                }
                write_json(out_path, payload)
                write_cache(
                    cache_dir,
                    cache_key(
                        pair_id=stem,
                        prompt_hash=prompt_hash,
                        model_id=GENERATION_MODEL,
                        parameters=params,
                    ),
                    payload,
                )
                guard.add_actual(usage.estimated_cost, pair_id=pair_id, stage="B1")
                results.append(payload)
                print(
                    f"[B1] {stem} q={len(parsed.questions)} "
                    f"cost={usage.estimated_cost:.6f} lat_ms={usage.latency_ms}"
                )
            except SpendAborted:
                raise
            except Exception as exc:  # noqa: BLE001
                err = {
                    "pair_id": pair_id,
                    "stem": stem,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                errors.append(err)
                print(f"[B1] ERROR {stem}: {err['error_type']}: {exc}")

    total_cost = sum(float(r.get("cost_usd") or 0) for r in results)
    summary = {
        "experiment_id": experiment_id,
        "status": "completed" if results and not errors else (
            "failed" if not results else "partial"
        ),
        "stage": "baseline_profile_aware",
        "generation_mode": GenerationMode.BASELINE.value,
        "model": GENERATION_MODEL,
        "prompt_versions": {"profile_aware_baseline": PROFILE_AWARE_PROMPT.version},
        "generation_parameters": params,
        "usd_cny_rate": USD_CNY_RATE,
        "combinations_attempted": len(extraction_records) * replicates,
        "combinations_ok": len(results),
        "combinations_failed": len(errors),
        "cache_hits": n_cache_hits,
        "total_estimated_cost_usd": round(total_cost, 8),
        "total_estimated_cost_cny": usd_to_cny(total_cost),
        "errors": errors,
        "spend_events": guard.events,
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def _select_from_pool(
    *,
    questions: list[InterviewQuestion],
    weights: list[float],
    embeddings: EmbeddingService,
    strategy: str,
    jd_text: str | None,
    lambda_: float = MMR_LAMBDA,
    k: int = FINAL_N,
    seed: int = SEED,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    texts = [q.question for q in questions]
    sim = embeddings.similarity_matrix(texts, texts) if texts else None
    import numpy as np

    if sim is None:
        sim = np.zeros((0, 0))
    query_sims = None
    if strategy == "plain_mmr_query":
        if not jd_text:
            raise ValueError("JD text required for plain_mmr_query")
        q_mat = embeddings.similarity_matrix(texts, [jd_text])
        query_sims = q_mat[:, 0]
    result = select_questions(
        strategy,
        questions,
        sim,
        weights,
        lambda_=lambda_,
        k=k,
        seed=seed,
        query_similarities=query_sims,
    )
    red_pool = redundant_pairs(sim, SIM_THRESHOLD) if texts else []
    selected_texts = [questions[i].question for i in result.selected_indices]
    sel_sim = (
        embeddings.similarity_matrix(selected_texts, selected_texts)
        if selected_texts
        else None
    )
    red_final = (
        redundant_pairs(sel_sim, SIM_THRESHOLD)
        if sel_sim is not None and len(selected_texts) > 1
        else []
    )
    latency_ms = (time.perf_counter() - started) * 1000
    payload = {
        "strategy": result.strategy,
        "selected_indices": result.selected_indices,
        "questions": [
            {
                **questions[i].model_dump(),
                "weight": weights[i] if i < len(weights) else None,
            }
            for i in result.selected_indices
        ],
        "n_selected": len(result.selected_indices),
        "backfilled": result.backfilled,
        "backfill_count": result.backfill_count,
        "notes": result.notes,
        "n_redundant_pairs_pool": len(red_pool),
        "n_redundant_pairs_final": len(red_final),
        "relevance_scores": result.relevance_scores,
        "redundancy_scores": result.redundancy_scores,
        "mmr_scores": result.mmr_scores,
        "competency_weights": result.competency_weights,
    }
    return payload, latency_ms


async def run_treatment_pipeline(
    *,
    root: Path,
    settings: Settings,
    extraction_records: list[dict],
    gap_records: dict[str, dict],
    embeddings: EmbeddingService,
    experiment_id: str,
    out_dir: Path,
    guard: TreatmentSpendGuard,
    selection: str = "competency_weighted_mmr",
    resume: bool = True,
    replicates: int = 1,
    dry_run: bool = False,
    pairs_dir: Path | None = None,
) -> dict[str, Any]:
    stage_dir = out_dir / "treatment"
    pool_dir = out_dir / "pool"
    stage_dir.mkdir(parents=True, exist_ok=True)
    pool_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = root / "experiments" / "results" / "_treatment_cache" / "pool"
    system_text = POOL_PROMPT.load()
    prompt_hash = prompt_sha256(system_text)
    params = generation_parameters()

    if dry_run:
        est = estimate_stage_costs(len(extraction_records))
        print("[treatment_pipeline] dry-run")
        print(json.dumps({"n_pairs": len(extraction_records), "estimate": est["t_pool"]}, indent=2))
        return {"dry_run": True, "estimate": est["t_pool"]}

    if not api_key_present(settings):
        return {
            "stopped": "missing_api_key",
            "command": (
                "PYTHONPATH=backend python experiments/scripts/run_treatment_pilot.py "
                f"--confirm-spend --max-cost {guard.max_cost}"
            ),
        }

    guard.require_confirm()
    router = build_router(settings)
    results: list[dict] = []
    errors: list[dict] = []
    n_cache_hits = 0

    for rep in range(replicates):
        for record in extraction_records:
            rid = request_id(record)
            stem = rid if replicates == 1 else f"{rid}__rep{rep}"
            gap_key = rid + ".json"
            if gap_key not in gap_records:
                errors.append(
                    {
                        "stem": stem,
                        "error_type": "MissingGap",
                        "error": f"No gap record for {gap_key}",
                    }
                )
                continue
            gap_rec = gap_records[gap_key]
            gap_matrix = GapMatrix.model_validate(gap_rec["gap_matrix"])
            pair_id = str(record.get("pair_id") or record.get("profile_id"))
            out_path = stage_dir / f"{stem}.json"
            pool_path = pool_dir / f"{stem}.json"

            if resume and out_path.is_file() and pool_path.is_file():
                results.append(json.loads(out_path.read_text(encoding="utf-8")))
                n_cache_hits += 1
                continue

            disk_cached = read_cache(
                cache_dir,
                cache_key(
                    pair_id=stem,
                    prompt_hash=prompt_hash,
                    model_id=GENERATION_MODEL,
                    parameters=params,
                ),
            )
            pool_payload: dict | None = None
            usage_dict: dict | None = None
            try:
                if resume and disk_cached is not None:
                    pool_payload = disk_cached
                    n_cache_hits += 1
                elif resume and pool_path.is_file():
                    pool_payload = json.loads(pool_path.read_text(encoding="utf-8"))
                    n_cache_hits += 1
                else:
                    guard.check_next_pair()
                    candidate = CandidateProfile.model_validate(record["candidate"])
                    job = JobDescription.model_validate(record["job"])
                    parsed, usage, system_prompt, user_prompt = await generate_pool(
                        router=router,
                        settings=settings,
                        candidate=candidate,
                        job=job,
                        gap_matrix=gap_matrix,
                        seniority=record.get("seniority") or "mid",
                        n_questions=POOL_N,
                        experiment_id=experiment_id,
                        request_id=stem,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                        reasoning_effort=REASONING_EFFORT,
                    )
                    truncated = bool(usage.output_tokens >= MAX_OUTPUT_TOKENS - 1)
                    raw_items = list(parsed.questions)
                    validation = validate_pool_items(raw_items, gap_matrix)
                    weights, weight_outcomes = map_weights_for_questions(
                        [q.target_gap for q in validation.valid_items],
                        gap_matrix,
                        embeddings,
                    )
                    pool_payload = {
                        "experiment_id": experiment_id,
                        "stage": "pool",
                        "condition": "T_pool",
                        "profile_id": record.get("profile_id"),
                        "jd_id": record.get("jd_id"),
                        "pair_id": pair_id,
                        "fit_label": record.get("fit_label"),
                        "seniority": record.get("seniority"),
                        "replicate": rep,
                        "model": usage.model,
                        "provider": usage.provider,
                        "prompt": _prompt_meta(POOL_PROMPT, system_prompt),
                        "prompt_hash": prompt_hash,
                        "prompt_version": POOL_PROMPT.version,
                        "generation_parameters": params,
                        "n_requirements": len(gap_matrix.gaps),
                        "pool_size_requested": POOL_N,
                        "raw_questions": [q.model_dump() for q in raw_items],
                        "valid_questions": [q.model_dump() for q in validation.valid_items],
                        "validation_failures": validation.failures,
                        "n_raw": validation.n_raw,
                        "n_valid": validation.n_valid,
                        "n_failed": validation.n_failed,
                        "weights": weights,
                        "weight_outcomes": [
                            {
                                "rule": o.rule,
                                "weight": o.weight,
                                "matched_competency": o.matched_competency,
                                "cosine": o.cosine,
                            }
                            for o in weight_outcomes
                        ],
                        "fallback_rate": (
                            sum(1 for o in weight_outcomes if o.rule == "embedding_fallback")
                            / len(weight_outcomes)
                            if weight_outcomes
                            else None
                        ),
                        "profile_probe_rate": (
                            sum(1 for o in weight_outcomes if o.rule == "profile_probe")
                            / len(weight_outcomes)
                            if weight_outcomes
                            else None
                        ),
                        "truncated": truncated,
                        "usage": _usage_payload(usage, LLMTask.QUESTION_GENERATION.value),
                        "cost_usd": usage.estimated_cost,
                        "cost_cny": usd_to_cny(usage.estimated_cost),
                        "usd_cny_rate": USD_CNY_RATE,
                        "user_prompt_sha256": prompt_sha256(user_prompt),
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    }
                    write_json(pool_path, pool_payload)
                    write_cache(
                        cache_dir,
                        cache_key(
                            pair_id=stem,
                            prompt_hash=prompt_hash,
                            model_id=GENERATION_MODEL,
                            parameters=params,
                        ),
                        pool_payload,
                    )
                    guard.add_actual(usage.estimated_cost, pair_id=pair_id, stage="T_pool")
                    usage_dict = pool_payload["usage"]
                    print(
                        f"[T-pool] {stem} valid={validation.n_valid}/{validation.n_raw} "
                        f"cost={usage.estimated_cost:.6f}"
                    )

                assert pool_payload is not None
                if "valid_questions" not in pool_payload:
                    raise ValueError("pool payload missing valid_questions")
                valid_items = [
                    PoolQuestionItem.model_validate(q)
                    for q in pool_payload["valid_questions"]
                ]
                from app.schemas.pool import pool_item_to_interview_question

                questions = [pool_item_to_interview_question(q) for q in valid_items]
                weights = list(pool_payload.get("weights") or [])
                if len(weights) != len(questions):
                    weights, weight_outcomes = map_weights_for_questions(
                        [q.target_gap for q in questions],
                        gap_matrix,
                        embeddings,
                    )
                    pool_payload["weights"] = weights
                    pool_payload["weight_outcomes"] = [
                        {
                            "rule": o.rule,
                            "weight": o.weight,
                            "matched_competency": o.matched_competency,
                            "cosine": o.cosine,
                        }
                        for o in weight_outcomes
                    ]

                jd_text = None
                if pairs_dir is not None:
                    job_path = pairs_dir / "jobs" / f"{record.get('jd_id')}.txt"
                    if job_path.is_file():
                        jd_text = job_path.read_text(encoding="utf-8")

                selection_payload, sel_latency = _select_from_pool(
                    questions=questions,
                    weights=weights,
                    embeddings=embeddings,
                    strategy=selection,
                    jd_text=jd_text,
                )
                if selection_payload["n_selected"] < FINAL_N:
                    errors.append(
                        {
                            "stem": stem,
                            "error_type": "InsufficientPool",
                            "error": (
                                f"selected {selection_payload['n_selected']} < {FINAL_N}; "
                                f"notes={selection_payload.get('notes')}"
                            ),
                        }
                    )
                    # still write artifact for diagnostics
                treatment_rec = {
                    "experiment_id": experiment_id,
                    "stage": "treatment_pipeline",
                    "generation_mode": GenerationMode.INTERVIEWIQ.value,
                    "condition": "T",
                    "selection_strategy": selection,
                    "profile_id": record.get("profile_id"),
                    "jd_id": record.get("jd_id"),
                    "pair_id": pair_id,
                    "fit_label": record.get("fit_label"),
                    "seniority": record.get("seniority"),
                    "replicate": rep,
                    "model": pool_payload.get("model", GENERATION_MODEL),
                    "prompt_hash": prompt_hash,
                    "prompt_version": POOL_PROMPT.version,
                    "generation_parameters": params,
                    "n_requirements": len(gap_matrix.gaps),
                    "pool_path": f"pool/{stem}.json",
                    "pool_stats": {
                        "n_raw": pool_payload.get("n_raw"),
                        "n_valid": pool_payload.get("n_valid"),
                        "n_failed": pool_payload.get("n_failed"),
                        "fallback_rate": pool_payload.get("fallback_rate"),
                        "profile_probe_rate": pool_payload.get("profile_probe_rate"),
                        "truncated": pool_payload.get("truncated"),
                    },
                    "selection": selection_payload,
                    "questions": [
                        {"question": q["question"]} for q in selection_payload["questions"]
                    ],
                    "questions_rich": selection_payload["questions"],
                    "n_questions": selection_payload["n_selected"],
                    "usage": pool_payload.get("usage") or usage_dict,
                    "cost_usd": pool_payload.get("cost_usd") or 0.0,
                    "cost_cny": pool_payload.get("cost_cny"),
                    "selection_latency_ms": round(sel_latency, 2),
                    "usd_cny_rate": USD_CNY_RATE,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
                write_json(out_path, treatment_rec)
                if not pool_path.is_file():
                    write_json(pool_path, pool_payload)
                results.append(treatment_rec)
            except SpendAborted:
                raise
            except Exception as exc:  # noqa: BLE001
                err = {
                    "pair_id": pair_id,
                    "stem": stem,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                errors.append(err)
                print(f"[T] ERROR {stem}: {err['error_type']}: {exc}")

    total_cost = sum(float(r.get("cost_usd") or 0) for r in results)
    summary = {
        "experiment_id": experiment_id,
        "status": "completed" if results and not errors else (
            "failed" if not results else "partial"
        ),
        "stage": "treatment_pipeline",
        "generation_mode": GenerationMode.INTERVIEWIQ.value,
        "model": GENERATION_MODEL,
        "selection_strategy": selection,
        "prompt_versions": {"interviewiq_pool": POOL_PROMPT.version},
        "generation_parameters": params,
        "usd_cny_rate": USD_CNY_RATE,
        "combinations_attempted": len(extraction_records) * replicates,
        "combinations_ok": len(results),
        "combinations_failed": len(errors),
        "cache_hits": n_cache_hits,
        "total_estimated_cost_usd": round(total_cost, 8),
        "total_estimated_cost_cny": usd_to_cny(total_cost),
        "errors": errors,
        "spend_events": guard.events,
    }
    write_json(out_dir / "summary.json", summary)
    return summary
